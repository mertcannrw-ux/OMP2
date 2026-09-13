use crate::component::Component;
use crate::semantic::SemanticRegistry;
use crate::terminal::{TerminalBackend, VirtualTerminal};
use crate::transcript::{BlockId, BlockMode, ResizePolicy, TranscriptScheduler};
use omp_types::ElementSnapshot;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DebugSnapshot {
    pub viewport: Vec<String>,
    pub logical_history: Vec<String>,
    pub display_epoch: u64,
    pub commit_frontier: usize,
    pub fail_stopped: bool,
    /// Total visible elements vs how many were actually rendered. When
    /// `rendered_elements < total_elements`, the caller asked for a bounded
    /// page (`max_elements`) and should re-request with an offset.
    pub total_elements: usize,
    pub rendered_elements: usize,
    pub scrollback_evicted: u64,
}

#[derive(Clone)]
pub struct DebugSession {
    pub registry: SemanticRegistry,
    pub scheduler: TranscriptScheduler,
    pub terminal: VirtualTerminal,
    /// Paging counters from the last `apply_session_snapshot*` call.
    pub last_total_elements: usize,
    pub last_rendered_elements: usize,
}

impl DebugSession {
    /// Default per-call element budget for `apply_session_snapshot`.
    pub const DEFAULT_MAX_ELEMENTS: usize = 256;
    /// Hard cap: a single projection call never renders more than this.
    pub const MAX_ELEMENTS_HARD_CAP: usize = 4096;

    pub fn new(width: u16, height: u16) -> Self {
        Self {
            registry: SemanticRegistry::new(),
            scheduler: TranscriptScheduler::new(width, height),
            terminal: VirtualTerminal::new(width, height),
            last_total_elements: 0,
            last_rendered_elements: 0,
        }
    }

    /// Reconstruct an off-screen view from the same materialized snapshot used by peer clients.
    ///
    /// `max_elements` bounds CPU per call (default 256): only the first N
    /// visible elements are admitted/rendered. Callers page with
    /// `skip_elements` for the remainder. Without a bound, a large journal
    /// plus frequent polling is a CPU/alloc DoS on the serving thread.
    pub fn apply_session_snapshot(
        &mut self,
        snapshot: &omp_state::SessionSnapshot,
    ) -> Result<Component, String> {
        self.apply_session_snapshot_paged(snapshot, 0, Self::DEFAULT_MAX_ELEMENTS)
    }

    /// Paged variant: render `snapshot` skipping the first `skip_elements`
    /// visible elements and rendering at most `max_elements` after that.
    pub fn apply_session_snapshot_paged(
        &mut self,
        snapshot: &omp_state::SessionSnapshot,
        skip_elements: usize,
        max_elements: usize,
    ) -> Result<Component, String> {
        let max_elements = max_elements.clamp(1, Self::MAX_ELEMENTS_HARD_CAP);
        let mut next = Self::new(
            self.scheduler.viewport_width,
            self.scheduler.viewport_height,
        );
        let tree = next.registry.render_session(snapshot)?;
        let total_elements = snapshot.get_visible_body().count();
        let mut skipped = 0usize;
        let mut rendered = 0usize;
        for element in snapshot.get_visible_body() {
            if skipped < skip_elements {
                skipped += 1;
                continue;
            }
            if rendered >= max_elements {
                break;
            }
            rendered += 1;
            let component = next
                .registry
                .render_session_element(snapshot, &element.id)?;
            let mut sink = crate::out::StringOutSink::new();
            component
                .render_to_sink(&mut sink, 0)
                .map_err(|error| error.to_string())?;
            let id = BlockId::new(element.id.as_str());
            next.scheduler
                .admit_block(id.clone(), BlockMode::Mutable, element.id.as_str())
                .map_err(|error| error.to_string())?;
            next.scheduler
                .activate_block(&id)
                .map_err(|error| error.to_string())?;
            next.scheduler
                .update_snapshot(&id, sink.as_str().lines().map(str::to_owned).collect())
                .map_err(|error| error.to_string())?;
            let running = matches!(element.attributes.get("status"), Some(omp_types::TypedValue::String(status))
                if matches!(status.as_str(), "queued" | "running" | "active" | "cancel_requested"));
            if !running {
                next.scheduler
                    .finalize_block(&id)
                    .map_err(|error| error.to_string())?;
                next.scheduler.flush().map_err(|error| error.to_string())?;
            }
        }
        next.render_frame()?;
        next.last_total_elements = total_elements;
        next.last_rendered_elements = rendered;
        *self = next;
        Ok(tree)
    }

    pub fn admit_element_block(
        &mut self,
        id: impl Into<String>,
        mode: BlockMode,
        owner: impl Into<String>,
        element: &ElementSnapshot,
    ) -> Result<BlockId, String> {
        let block_id = BlockId::new(id);
        self.scheduler
            .admit_block(block_id.clone(), mode, owner)
            .map_err(|e| e.to_string())?;

        self.scheduler
            .activate_block(&block_id)
            .map_err(|e| e.to_string())?;

        let plain_lines = self.element_to_lines(element)?;
        self.scheduler
            .update_snapshot(&block_id, plain_lines)
            .map_err(|e| e.to_string())?;

        Ok(block_id)
    }

    pub fn update_element_block(
        &mut self,
        id: &BlockId,
        element: &ElementSnapshot,
    ) -> Result<(), String> {
        let plain_lines = self.element_to_lines(element)?;
        self.scheduler
            .update_snapshot(id, plain_lines)
            .map_err(|e| e.to_string())
    }

    fn element_to_lines(&self, element: &ElementSnapshot) -> Result<Vec<String>, String> {
        let rendered = self.registry.render_plain(element)?;
        let lines: Vec<String> = rendered.lines().map(|s| s.to_string()).collect();
        Ok(lines)
    }

    pub fn render_frame(&mut self) -> Result<Vec<String>, String> {
        self.scheduler
            .check_operational()
            .map_err(|error| error.to_string())?;
        self.scheduler.refresh_viewport();
        let mut accepted = 0;
        let result = (|| {
            self.terminal.clear_screen()?;
            for row in &self.scheduler.native_viewport {
                self.terminal.write_row(row)?;
                accepted += 1;
            }
            self.terminal.flush()
        })();
        if let Err(error) = result {
            self.scheduler.handle_write_failure(
                omp_types::StructuredError::new("terminal_write_failure", error.to_string(), false),
                accepted,
            );
            return Err(error.to_string());
        }
        Ok(self.terminal.rendered_text())
    }

    pub fn resize(
        &mut self,
        new_width: u16,
        new_height: u16,
        policy: ResizePolicy,
    ) -> Result<(), String> {
        self.scheduler
            .check_operational()
            .map_err(|error| error.to_string())?;
        self.scheduler
            .resize(new_width, new_height, policy)
            .map_err(|e| e.to_string())?;
        self.terminal.resize(new_width, new_height);
        self.render_frame()?;
        Ok(())
    }

    pub fn snapshot(&self) -> DebugSnapshot {
        let active_head = self.scheduler.blocks.iter().find(|b| !b.is_terminal());
        DebugSnapshot {
            viewport: self.terminal.rendered_text(),
            logical_history: self.scheduler.logical_history.all_lines(active_head),
            display_epoch: self.scheduler.display_epoch,
            commit_frontier: self.scheduler.logical_history.commit_frontier(),
            fail_stopped: self.scheduler.fail_stopped,
            total_elements: self.last_total_elements,
            rendered_elements: self.last_rendered_elements,
            scrollback_evicted: self.scheduler.scrollback_evicted,
        }
    }

    pub fn assert_logical_history_unchanged_on_resize(
        &mut self,
        new_width: u16,
        new_height: u16,
    ) -> Result<(), String> {
        let before_active = self.scheduler.blocks.iter().find(|b| !b.is_terminal());
        let before_history = self.scheduler.logical_history.all_lines(before_active);

        self.resize(new_width, new_height, ResizePolicy::Rebuild)?;

        let after_active = self.scheduler.blocks.iter().find(|b| !b.is_terminal());
        let after_history = self.scheduler.logical_history.all_lines(after_active);

        if before_history != after_history {
            return Err(format!(
                "Logical history changed after resize! Before: {before_history:?}, After: {after_history:?}"
            ));
        }

        Ok(())
    }

    pub fn assert_no_premature_mutable_history(
        &self,
        mutable_block_id: &BlockId,
    ) -> Result<(), String> {
        let block = self
            .scheduler
            .blocks
            .iter()
            .find(|b| &b.id == mutable_block_id)
            .ok_or_else(|| "Block not found".to_string())?;

        if block.mode != BlockMode::Mutable {
            return Err("Block is not mutable".to_string());
        }

        if block.state == crate::transcript::BlockState::Active && block.emitted_prefix != 0 {
            return Err("Active mutable block emitted a logical prefix".into());
        }

        Ok(())
    }
}
