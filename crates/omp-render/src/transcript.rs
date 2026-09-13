use crate::richtext::str_visible_width;
use omp_types::StructuredError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct BlockId(pub String);

impl BlockId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BlockMode {
    Mutable,
    AppendOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BlockState {
    Queued,
    Active,
    Finalized,
    Committed,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TranscriptError {
    #[error("block not found: {0}")]
    BlockNotFound(BlockId),
    #[error("invalid state transition for block {id}: {from:?} -> {to:?}")]
    InvalidStateTransition {
        id: BlockId,
        from: BlockState,
        to: BlockState,
    },
    #[error(
        "prefix monotonicity violated for append-only block {id}: new snapshot must extend prior lines"
    )]
    PrefixMonotonicityViolated { id: BlockId },
    #[error("mutable block {0} cannot stream rows into logical history before finalization")]
    MutableBlockCannotStream(BlockId),
    #[error("block {0} is finalized and immutable")]
    BlockIsFinalized(BlockId),
    #[error("replay gate violation: unexpected operation during replay state {0:?}")]
    ReplayGateViolation(String),
    #[error("transcript scheduler is halted in fail-stop state: {0}")]
    FailStopHalted(String),
    #[error("duplicate transcript block: {0}")]
    DuplicateBlock(BlockId),
    #[error("live transcript block limit reached; flush finalized blocks before admission")]
    LiveBlockLimit,
    #[error("capacity violation: reserved rows {reserved} exceed viewport height {viewport}")]
    CapacityViolation { reserved: usize, viewport: usize },
    #[error(
        "out of order retirement for block {id}: expected block at commit frontier {expected:?}"
    )]
    OutOfOrderRetirement {
        id: BlockId,
        expected: Option<BlockId>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranscriptBlock {
    pub id: BlockId,
    pub mode: BlockMode,
    pub state: BlockState,
    pub owner: String,
    pub snapshot: Vec<String>,
    pub emitted_prefix: usize,
    pub reserved_rows: usize,
    pub last_rendered_height: usize,
}

impl TranscriptBlock {
    pub fn new(id: BlockId, mode: BlockMode, owner: impl Into<String>) -> Self {
        Self {
            id,
            mode,
            state: BlockState::Queued,
            owner: owner.into(),
            snapshot: Vec::new(),
            emitted_prefix: 0,
            reserved_rows: 0,
            last_rendered_height: 0,
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.state == BlockState::Committed
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LogicalHistory {
    committed_blocks: Vec<Vec<String>>,
}

impl LogicalHistory {
    pub fn new() -> Self {
        Self {
            committed_blocks: Vec::new(),
        }
    }

    pub fn commit_frontier(&self) -> usize {
        self.committed_blocks.len()
    }

    pub fn commit_block(&mut self, final_lines: Vec<String>) {
        self.committed_blocks.push(final_lines);
    }

    pub fn all_lines(&self, active_head: Option<&TranscriptBlock>) -> Vec<String> {
        let mut lines = Vec::new();
        for blk in &self.committed_blocks {
            lines.extend(blk.clone());
        }

        if let Some(active) = active_head
            && (active.state == BlockState::Active || active.state == BlockState::Finalized) {
                match active.mode {
                    BlockMode::AppendOnly => {
                        let stream_count = active.emitted_prefix.min(active.snapshot.len());
                        lines.extend(active.snapshot[..stream_count].iter().cloned());
                    }
                    BlockMode::Mutable => {
                        // Mutable active blocks emit ZERO rows to logical history!
                    }
                }
            }

        lines
    }

    pub fn committed_line_count(&self) -> usize {
        self.committed_blocks.iter().map(|b| b.len()).sum()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NativeSource {
    Append,
    Retire,
    Replay,
    Resize,
    FailedWrite,
    Exit,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeRow {
    pub text: String,
    pub source: NativeSource,
    pub source_block: Option<BlockId>,
    pub owner: Option<String>,
    pub width: usize,
    pub epoch: u64,
    pub is_summary: bool,
}

impl NativeRow {
    pub fn new(
        text: impl Into<String>,
        source_block: Option<BlockId>,
        owner: Option<String>,
        epoch: u64,
    ) -> Self {
        let t = text.into();
        let w = str_visible_width(&t);
        Self {
            text: t,
            source: NativeSource::Retire,
            source_block,
            owner,
            width: w,
            epoch,
            is_summary: false,
        }
    }

    pub fn tagged(
        text: impl Into<String>,
        source: NativeSource,
        source_block: Option<BlockId>,
        owner: Option<String>,
        epoch: u64,
    ) -> Self {
        let t = text.into();
        let w = str_visible_width(&t);
        Self {
            text: t,
            source,
            source_block,
            owner,
            width: w,
            epoch,
            is_summary: false,
        }
    }

    pub fn summary(text: impl Into<String>, epoch: u64) -> Self {
        let t = text.into();
        let w = str_visible_width(&t);
        Self {
            text: t,
            source: NativeSource::Resize,
            source_block: None,
            owner: None,
            width: w,
            epoch,
            is_summary: true,
        }
    }

    pub fn with_source(mut self, source: NativeSource) -> Self {
        self.source = source;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ResizePolicy {
    Preserve,
    Append,
    Rebuild,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplayGateState {
    Idle,
    PreparingReplay {
        epoch: u64,
        target_width: u16,
        target_height: u16,
    },
    ReplayWriting {
        epoch: u64,
    },
    FailedStop {
        error: StructuredError,
        forensic_rows_written: usize,
    },
}

#[derive(Clone, Debug)]
pub struct TranscriptScheduler {
    pub blocks: Vec<TranscriptBlock>,
    pub logical_history: LogicalHistory,
    pub native_scrollback: Vec<NativeRow>,
    pub native_viewport: Vec<NativeRow>,
    pub viewport_width: u16,
    pub viewport_height: u16,
    pub display_epoch: u64,
    pub replay_gate: ReplayGateState,
    pub fail_stopped: bool,
    pub forensic_log: Vec<NativeRow>,
    /// Rows evicted from `native_scrollback` by the ring bound. The
    /// authoritative record lives in `logical_history`; this counter makes
    /// the eviction visible to debug/telemetry consumers.
    pub scrollback_evicted: u64,
}

impl TranscriptScheduler {
    /// Maximum retained scrollback rows. Oldest rows are evicted first;
    /// logical history (the commit ledger) is unaffected.
    pub const MAX_SCROLLBACK_ROWS: usize = 10_000;

    pub fn new(viewport_width: u16, viewport_height: u16) -> Self {
        Self {
            blocks: Vec::new(),
            logical_history: LogicalHistory::new(),
            native_scrollback: Vec::new(),
            native_viewport: Vec::new(),
            viewport_width,
            viewport_height,
            display_epoch: 1,
            replay_gate: ReplayGateState::Idle,
            fail_stopped: false,
            forensic_log: Vec::new(),
            scrollback_evicted: 0,
        }
    }

    /// Push one row to scrollback, evicting oldest rows past the ring bound.
    fn push_scrollback(&mut self, row: NativeRow) {
        self.native_scrollback.push(row);
        let excess = self
            .native_scrollback
            .len()
            .saturating_sub(Self::MAX_SCROLLBACK_ROWS);
        if excess > 0 {
            self.native_scrollback.drain(..excess);
            self.scrollback_evicted += excess as u64;
        }
    }

    pub fn commit_frontier(&self) -> usize {
        self.logical_history.commit_frontier()
    }

    fn check_not_fail_stopped(&self) -> Result<(), TranscriptError> {
        if self.fail_stopped {
            return Err(TranscriptError::FailStopHalted(
                "scheduler is in fail-stop state after an unrecoverable write failure".into(),
            ));
        }
        Ok(())
    }

    pub fn check_operational(&self) -> Result<(), TranscriptError> {
        self.check_not_fail_stopped()?;
        if self.replay_gate != ReplayGateState::Idle {
            return Err(TranscriptError::ReplayGateViolation(format!(
                "operation forbidden while replay gate is in state {:?}",
                self.replay_gate
            )));
        }
        Ok(())
    }

    pub fn admit_block(
        &mut self,
        id: BlockId,
        mode: BlockMode,
        owner: impl Into<String>,
    ) -> Result<(), TranscriptError> {
        self.check_operational()?;
        if self.blocks.iter().any(|block| block.id == id) {
            return Err(TranscriptError::DuplicateBlock(id));
        }
        if self.blocks.len().saturating_sub(self.commit_frontier()) >= 128 {
            return Err(TranscriptError::LiveBlockLimit);
        }
        let block = TranscriptBlock::new(id, mode, owner);
        self.blocks.push(block);
        Ok(())
    }

    pub fn activate_block(&mut self, id: &BlockId) -> Result<(), TranscriptError> {
        self.check_operational()?;
        let block = self
            .blocks
            .iter_mut()
            .find(|b| &b.id == id)
            .ok_or_else(|| TranscriptError::BlockNotFound(id.clone()))?;

        if block.state != BlockState::Queued {
            return Err(TranscriptError::InvalidStateTransition {
                id: id.clone(),
                from: block.state,
                to: BlockState::Active,
            });
        }
        block.state = BlockState::Active;
        Ok(())
    }

    pub fn update_snapshot(
        &mut self,
        id: &BlockId,
        new_lines: Vec<String>,
    ) -> Result<(), TranscriptError> {
        self.check_operational()?;
        let block = self
            .blocks
            .iter_mut()
            .find(|b| &b.id == id)
            .ok_or_else(|| TranscriptError::BlockNotFound(id.clone()))?;

        if block.state == BlockState::Finalized || block.state == BlockState::Committed {
            return Err(TranscriptError::BlockIsFinalized(id.clone()));
        }

        match block.mode {
            BlockMode::AppendOnly => {
                // Snapshot discipline: must be prefix-monotone
                if new_lines.len() < block.snapshot.len() {
                    return Err(TranscriptError::PrefixMonotonicityViolated { id: id.clone() });
                }
                for (i, old_line) in block.snapshot.iter().enumerate() {
                    if &new_lines[i] != old_line {
                        return Err(TranscriptError::PrefixMonotonicityViolated { id: id.clone() });
                    }
                }
                block.snapshot = new_lines;
            }
            BlockMode::Mutable => {
                // Live snapshots may replace one another
                block.snapshot = new_lines;
            }
        }

        self.apply_elastic_reservation(id)?;
        Ok(())
    }

    pub fn stream_stable_prefix(
        &mut self,
        id: &BlockId,
        stable_count: usize,
    ) -> Result<usize, TranscriptError> {
        self.check_operational()?;
        let block_idx = self
            .blocks
            .iter()
            .position(|b| &b.id == id)
            .ok_or_else(|| TranscriptError::BlockNotFound(id.clone()))?;

        if block_idx != self.commit_frontier() {
            let expected = self
                .blocks
                .get(self.commit_frontier())
                .map(|b| b.id.clone());
            return Err(TranscriptError::OutOfOrderRetirement {
                id: id.clone(),
                expected,
            });
        }

        let block = &mut self.blocks[block_idx];
        if !matches!(block.state, BlockState::Active | BlockState::Finalized) {
            return Err(TranscriptError::InvalidStateTransition {
                id: id.clone(),
                from: block.state,
                to: BlockState::Active,
            });
        }

        if block.mode == BlockMode::Mutable {
            return Err(TranscriptError::MutableBlockCannotStream(id.clone()));
        }

        let new_prefix = stable_count.min(block.snapshot.len());
        let rows: Vec<NativeRow> = if new_prefix > block.emitted_prefix {
            let previously_emitted = block.emitted_prefix;
            block.emitted_prefix = new_prefix;
            // Natural streaming: append newly streamed rows to native_scrollback with NativeSource::Append.
            // Rows are built as owned values first so the `block` borrow ends
            // before `push_scrollback` re-borrows `self`.
            let epoch = self.display_epoch;
            block.snapshot[previously_emitted..new_prefix]
                .iter()
                .map(|line| {
                    NativeRow::tagged(
                        line.clone(),
                        NativeSource::Append,
                        Some(block.id.clone()),
                        Some(block.owner.clone()),
                        epoch,
                    )
                })
                .collect()
        } else {
            Vec::new()
        };
        for row in rows {
            self.push_scrollback(row);
        }

        let block = &mut self.blocks[block_idx];
        let emitted = block.emitted_prefix;
        block.last_rendered_height = block
            .last_rendered_height
            .min(block.snapshot.len().saturating_sub(emitted));
        self.refresh_viewport();
        Ok(emitted)
    }

    pub fn finalize_block(&mut self, id: &BlockId) -> Result<(), TranscriptError> {
        self.check_operational()?;
        let block = self
            .blocks
            .iter_mut()
            .find(|b| &b.id == id)
            .ok_or_else(|| TranscriptError::BlockNotFound(id.clone()))?;

        if block.state != BlockState::Active {
            return Err(TranscriptError::InvalidStateTransition {
                id: id.clone(),
                from: block.state,
                to: BlockState::Finalized,
            });
        }

        // Finalization itself writes nothing to logical history
        block.state = BlockState::Finalized;
        Ok(())
    }

    pub fn retire_block(&mut self, id: &BlockId) -> Result<(), TranscriptError> {
        self.check_operational()?;
        let block_idx = self
            .blocks
            .iter()
            .position(|b| &b.id == id)
            .ok_or_else(|| TranscriptError::BlockNotFound(id.clone()))?;

        // Monotonic FIFO commit frontier check: blocks retire in strict order
        if block_idx != self.commit_frontier() {
            let expected = self
                .blocks
                .get(self.commit_frontier())
                .map(|b| b.id.clone());
            return Err(TranscriptError::OutOfOrderRetirement {
                id: id.clone(),
                expected,
            });
        }

        let block = &mut self.blocks[block_idx];

        if block.state != BlockState::Finalized {
            return Err(TranscriptError::InvalidStateTransition {
                id: id.clone(),
                from: block.state,
                to: BlockState::Committed,
            });
        }

        // Exactly-once retirement:
        // Suffix after emitted prefix contributes to native_scrollback with NativeSource::Retire.
        // Owned rows + owned snapshot are extracted first so the `block`
        // borrow ends before `push_scrollback` re-borrows `self`.
        let (rows, final_lines): (Vec<NativeRow>, Vec<String>) = {
            let block = &mut self.blocks[block_idx];
            let unstreamed_start = block.emitted_prefix;
            let epoch = self.display_epoch;
            let rows: Vec<NativeRow> = block.snapshot[unstreamed_start..]
                .iter()
                .map(|line| {
                    NativeRow::tagged(
                        line.clone(),
                        NativeSource::Retire,
                        Some(block.id.clone()),
                        Some(block.owner.clone()),
                        epoch,
                    )
                })
                .collect();
            (rows, block.snapshot.clone())
        };
        for row in rows {
            self.push_scrollback(row);
        }

        // Logical history ledger receives the full committed snapshot
        self.logical_history.commit_block(final_lines);

        let block = &mut self.blocks[block_idx];
        block.state = BlockState::Committed;
        block.emitted_prefix = 0;
        self.refresh_viewport();
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), TranscriptError> {
        self.check_operational()?;
        while let Some(block) = self.blocks.get(self.commit_frontier()) {
            if block.state != BlockState::Finalized {
                break;
            }
            let id = block.id.clone();
            self.retire_block(&id)?;
        }
        Ok(())
    }

    fn apply_elastic_reservation(&mut self, id: &BlockId) -> Result<(), TranscriptError> {
        let block = self
            .blocks
            .iter_mut()
            .find(|b| &b.id == id)
            .ok_or_else(|| TranscriptError::BlockNotFound(id.clone()))?;

        let old_height = block.last_rendered_height;
        let new_height = block.snapshot.len().saturating_sub(block.emitted_prefix);

        // Two-row bridge behavior for deep shrink (avoid visual snap)
        let effective_height = if old_height > new_height + 2 {
            old_height.saturating_sub(2).max(new_height)
        } else {
            new_height
        };

        let max_allowed = self.viewport_height as usize;
        if effective_height > max_allowed && max_allowed > 0 {
            block.reserved_rows = max_allowed;
        } else {
            block.reserved_rows = effective_height;
        }

        block.last_rendered_height = effective_height;
        self.refresh_viewport();
        Ok(())
    }

    pub fn refresh_viewport(&mut self) {
        if self.fail_stopped {
            return;
        }
        self.native_viewport.clear();
        let height = self.viewport_height as usize;
        let requested = self
            .blocks
            .iter()
            .filter(|b| matches!(b.state, BlockState::Active | BlockState::Finalized))
            .map(|b| {
                b.last_rendered_height
                    .max(b.snapshot.len().saturating_sub(b.emitted_prefix))
            })
            .sum::<usize>();
        let hidden = requested.saturating_add(self.native_scrollback.len()) > height;
        let summary = usize::from(hidden && height > 1);
        let mut remaining = height.saturating_sub(summary);
        for block in self.blocks.iter_mut().rev() {
            block.reserved_rows =
                if matches!(block.state, BlockState::Active | BlockState::Finalized) {
                    let requested = block
                        .last_rendered_height
                        .max(block.snapshot.len().saturating_sub(block.emitted_prefix));
                    let allocation = requested.min(remaining);
                    remaining -= allocation;
                    allocation
                } else {
                    0
                };
        }
        if height == 0 {
            return;
        }
        if summary != 0 {
            self.native_viewport.push(NativeRow::summary(
                "[earlier output hidden]",
                self.display_epoch,
            ));
        }
        let history_start = self.native_scrollback.len().saturating_sub(remaining);
        self.native_viewport
            .extend(self.native_scrollback[history_start..].iter().cloned());
        for block in &self.blocks {
            let allocation = block.reserved_rows;
            if allocation == 0 {
                continue;
            }
            let stable = if block.mode == BlockMode::AppendOnly {
                block.emitted_prefix
            } else {
                0
            };
            let start = stable.max(block.snapshot.len().saturating_sub(allocation));
            for line in &block.snapshot[start..] {
                self.native_viewport.push(NativeRow::tagged(
                    line.clone(),
                    NativeSource::Append,
                    Some(block.id.clone()),
                    Some(block.owner.clone()),
                    self.display_epoch,
                ));
            }
            for _ in block.snapshot.len().saturating_sub(start)..allocation {
                self.native_viewport.push(NativeRow::tagged(
                    String::new(),
                    NativeSource::Append,
                    Some(block.id.clone()),
                    Some(block.owner.clone()),
                    self.display_epoch,
                ));
            }
        }
        if self.viewport_width > 0 {
            for row in &mut self.native_viewport {
                if row.width > self.viewport_width as usize {
                    row.text = crate::richtext::RichText::from_plain(&row.text)
                        .truncate_width(self.viewport_width as usize, None)
                        .plain_text();
                    row.width = str_visible_width(&row.text);
                }
            }
        }
    }

    pub fn resize(
        &mut self,
        new_width: u16,
        new_height: u16,
        policy: ResizePolicy,
    ) -> Result<(), TranscriptError> {
        self.check_operational()?;
        match policy {
            ResizePolicy::Preserve => {
                self.viewport_width = new_width;
                self.viewport_height = new_height;
                self.refresh_viewport();
            }
            ResizePolicy::Append => {
                let row = NativeRow::summary(
                    format!("--- resize to {new_width}x{new_height} ---"),
                    self.display_epoch,
                )
                .with_source(NativeSource::Resize);
                self.push_scrollback(row);
                self.viewport_width = new_width;
                self.viewport_height = new_height;
                self.refresh_viewport();
            }
            ResizePolicy::Rebuild => {
                self.prepare_rebuild_replay(new_width, new_height)?;
                self.execute_rebuild_replay()?;
            }
        }

        Ok(())
    }

    /// Step 1 of Replay: prepare replay with new dimensions and next epoch.
    pub fn prepare_rebuild_replay(
        &mut self,
        target_width: u16,
        target_height: u16,
    ) -> Result<(), TranscriptError> {
        self.check_operational()?;
        self.replay_gate = ReplayGateState::PreparingReplay {
            epoch: self.display_epoch + 1,
            target_width,
            target_height,
        };
        Ok(())
    }

    /// Step 2 of Replay: execute synchronous replay writing and restore Idle.
    pub fn execute_rebuild_replay(&mut self) -> Result<(), TranscriptError> {
        self.check_not_fail_stopped()?;
        match self.replay_gate {
            ReplayGateState::PreparingReplay {
                epoch,
                target_width,
                target_height,
            } => {
                self.display_epoch = epoch;
                self.viewport_width = target_width;
                self.viewport_height = target_height;
                self.native_viewport.clear();
                self.native_scrollback.clear();
                // Rebuild replays from the authoritative block snapshots, so
                // the bound is applied to the *retained* result, not bypassed:
                // collect first, then keep only the newest MAX rows.
                let mut replayed: Vec<NativeRow> = Vec::new();
                for (index, block) in self.blocks.iter().enumerate() {
                    let replay = if index < self.commit_frontier() {
                        block.snapshot.as_slice()
                    } else if index == self.commit_frontier()
                        && matches!(block.state, BlockState::Active | BlockState::Finalized)
                        && block.mode == BlockMode::AppendOnly
                    {
                        &block.snapshot[..block.emitted_prefix.min(block.snapshot.len())]
                    } else {
                        &[]
                    };
                    for line in replay {
                        replayed.push(NativeRow::tagged(
                            line.clone(),
                            NativeSource::Replay,
                            Some(block.id.clone()),
                            Some(block.owner.clone()),
                            epoch,
                        ));
                    }
                }
                let evicted = replayed.len().saturating_sub(Self::MAX_SCROLLBACK_ROWS);
                if evicted > 0 {
                    replayed.drain(..evicted);
                    self.scrollback_evicted += evicted as u64;
                }
                self.native_scrollback = replayed;
                self.replay_gate = ReplayGateState::ReplayWriting { epoch };
                self.refresh_viewport();
                self.replay_gate = ReplayGateState::Idle;
                Ok(())
            }
            _ => Err(TranscriptError::ReplayGateViolation(format!(
                "cannot execute replay while replay gate is in state {:?}",
                self.replay_gate
            ))),
        }
    }

    pub fn handle_write_failure(&mut self, error: StructuredError, accepted_prefix: usize) {
        if self.fail_stopped {
            return;
        }
        let accepted_prefix = accepted_prefix.min(self.native_viewport.len());
        let mut accepted_rows: Vec<NativeRow> = self
            .native_viewport
            .drain(..accepted_prefix.min(self.native_viewport.len()))
            .collect();

        for row in &mut accepted_rows {
            row.source = NativeSource::FailedWrite;
        }

        self.forensic_log.extend(accepted_rows);
        self.native_viewport.clear();
        self.fail_stopped = true;
        self.replay_gate = ReplayGateState::FailedStop {
            error,
            forensic_rows_written: accepted_prefix,
        };
    }
}
