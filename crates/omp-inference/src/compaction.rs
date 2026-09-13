use crate::request::MessageFold;
use omp_types::{BranchId, JournalOffset, StructuredError};
use serde::{Deserialize, Serialize};

/// Snapshot of the session state at the moment speculative compaction begins.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionSnapshot {
    pub branch_id: BranchId,
    pub journal_offset: JournalOffset,
    pub token_count_at_snapshot: usize,
    pub threshold_tokens: usize,
    pub created_at_unix_ms: u64,
}

/// Evaluates whether context bounds warrant triggering speculative compaction.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompactionTrigger;

impl CompactionTrigger {
    /// Returns true if the token count exceeds the configured compaction threshold.
    pub fn should_trigger(current_tokens: usize, threshold_tokens: usize) -> bool {
        current_tokens >= threshold_tokens
    }

    /// Returns true if token count enters the predictive speculative window (e.g. 85% of threshold).
    /// `margin_ratio` is clamped to `0.1..=1.0`: values below 0.1 would
    /// trigger speculation almost immediately, values above 1.0 past the
    /// threshold itself.
    pub fn should_speculate(
        current_tokens: usize,
        threshold_tokens: usize,
        margin_ratio: f32,
    ) -> bool {
        let margin_tokens = (threshold_tokens as f32 * margin_ratio.clamp(0.1, 1.0)) as usize;
        current_tokens >= margin_tokens
    }
}

/// Guard managing the lifecycle and safe splicing of concurrent speculative compaction results.
///
/// Invariants:
/// 1. Speculative compaction runs concurrently against a frozen snapshot offset and branch.
/// 2. If the active branch diverged while compaction was running, the speculative fold is discarded.
/// 3. Any change to the source snapshot invalidates the speculative result.
/// 4. Live session authority is NEVER mutated on branch mismatch.
#[derive(Clone, Debug, Default)]
pub struct SpeculativeCompactionGuard;

impl SpeculativeCompactionGuard {
    pub fn new() -> Self {
        Self
    }

    /// Creates a snapshot marking the beginning of a speculative compaction attempt.
    pub fn create_snapshot(
        &self,
        branch_id: BranchId,
        journal_offset: JournalOffset,
        current_tokens: usize,
        threshold_tokens: usize,
    ) -> CompactionSnapshot {
        let created_at_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        CompactionSnapshot {
            branch_id,
            journal_offset,
            token_count_at_snapshot: current_tokens,
            threshold_tokens,
            created_at_unix_ms,
        }
    }

    /// Validates a finished speculative compaction result against the current session branch and offset.
    ///
    /// Slices the resulting MessageFold into the prompt projection ONLY if the branch and offset
    /// remain consistent with the snapshot. Otherwise discards the result without mutating state.
    pub fn validate_and_splice(
        &self,
        snapshot: &CompactionSnapshot,
        current_branch: &BranchId,
        current_offset: &JournalOffset,
        fold_candidate: MessageFold,
    ) -> Result<MessageFold, StructuredError> {
        // Invariant 2: Active branch check
        if current_branch != &snapshot.branch_id {
            let mut err = StructuredError::new(
                "compaction_branch_diverged",
                format!(
                    "Speculative compaction for branch '{}' discarded because active branch switched to '{}'",
                    snapshot.branch_id, current_branch
                ),
                false,
            );
            err.diagnostics = Some(serde_json::json!({
                "snapshot_branch": snapshot.branch_id.as_str(),
                "current_branch": current_branch.as_str(),
                "snapshot_offset": snapshot.journal_offset.0,
                "current_offset": current_offset.0,
            }));
            return Err(err);
        }

        // A fold was computed from exactly this snapshot, not future mutations.
        if current_offset != &snapshot.journal_offset {
            let mut err = StructuredError::new(
                "compaction_snapshot_changed",
                format!(
                    "Speculative compaction base offset {} differs from current offset {}",
                    snapshot.journal_offset.0, current_offset.0
                ),
                false,
            );
            err.diagnostics = Some(serde_json::json!({
                "snapshot_offset": snapshot.journal_offset.0,
                "current_offset": current_offset.0,
                "branch_id": current_branch.as_str(),
            }));
            return Err(err);
        }

        // The branch and complete source snapshot are still current.
        Ok(fold_candidate)
    }

    /// Checks whether a speculative compaction snapshot is stale relative to current session branch and offset.
    /// Stale if either the selected branch or source offset has changed.
    pub fn is_stale(
        &self,
        snapshot: &CompactionSnapshot,
        current_branch: &BranchId,
        current_offset: &JournalOffset,
    ) -> bool {
        current_branch != &snapshot.branch_id || current_offset != &snapshot.journal_offset
    }

    /// Discards stale speculative results without mutating live session authority.
    /// Returns Some(fold) only when branch and offset invariants hold.
    pub fn discard_if_stale(
        &self,
        snapshot: &CompactionSnapshot,
        current_branch: &BranchId,
        current_offset: &JournalOffset,
        fold_candidate: MessageFold,
    ) -> Option<MessageFold> {
        if self.is_stale(snapshot, current_branch, current_offset) {
            None
        } else {
            Some(fold_candidate)
        }
    }
}

/// High-level compaction integration manager for session host turns.
#[derive(Clone, Debug, Default)]
pub struct CompactionIntegration {
    guard: SpeculativeCompactionGuard,
}

impl CompactionIntegration {
    pub fn new() -> Self {
        Self {
            guard: SpeculativeCompactionGuard::new(),
        }
    }

    /// Starts speculative compaction for the given branch and offset.
    pub fn begin_speculative(
        &self,
        branch_id: BranchId,
        journal_offset: JournalOffset,
        current_tokens: usize,
        threshold_tokens: usize,
    ) -> CompactionSnapshot {
        self.guard
            .create_snapshot(branch_id, journal_offset, current_tokens, threshold_tokens)
    }

    /// Validates speculative fold candidate; discards stale result cleanly without mutating state.
    pub fn complete_or_discard(
        &self,
        snapshot: &CompactionSnapshot,
        current_branch: &BranchId,
        current_offset: &JournalOffset,
        candidate: MessageFold,
    ) -> Result<MessageFold, StructuredError> {
        self.guard
            .validate_and_splice(snapshot, current_branch, current_offset, candidate)
    }

    /// Safely discards if stale without error, returning Option.
    pub fn try_splice(
        &self,
        snapshot: &CompactionSnapshot,
        current_branch: &BranchId,
        current_offset: &JournalOffset,
        candidate: MessageFold,
    ) -> Option<MessageFold> {
        self.guard
            .discard_if_stale(snapshot, current_branch, current_offset, candidate)
    }
}
