use crate::request::MessageFold;
use omp_types::{BranchId, JournalOffset, StructuredError};
use serde::{Deserialize, Serialize};

/// Deterministic token estimate for one text body.
///
/// OMP2 has no tokenizer for arbitrary provider dialects, so the compaction
/// planner needs an estimate that is (a) identical on every replica replaying
/// the same journal and (b) never optimistic — an under-estimate is what turns
/// a "fits the window" decision into a provider error.
///
/// ASCII is charged four bytes per token, which matches English/code BPE
/// vocabularies within roughly ±20%. Every non-ASCII character is charged two
/// tokens regardless of script: production BPE rates are around 1–1.5 tokens
/// per CJK character and below 1 per accented Latin character, so two is an
/// upper bound rather than a measurement. The `CompactionBudget` reserve then
/// only has to cover the ASCII term.
pub fn estimate_tokens(text: &str) -> usize {
    let mut ascii_bytes = 0usize;
    let mut wide_chars = 0usize;
    for character in text.chars() {
        if character.is_ascii() {
            ascii_bytes += 1;
        } else {
            wide_chars += 1;
        }
    }
    ascii_bytes.div_ceil(4) + wide_chars * 2
}

/// Soft/hard limits for the provider projection of one session.
///
/// `soft_tokens` is where elision starts; `hard_tokens` is what the projection
/// must be under once it is done. Both are derived from the advertised context
/// window so a provider that never advertises one simply never compacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactionBudget {
    pub soft_tokens: usize,
    pub hard_tokens: usize,
}

impl CompactionBudget {
    /// Derives the budget from the advertised window, the session's
    /// `ai_compaction_threshold` ratio, and the tokens reserved for tool
    /// schemas, pinned instructions and the completion.
    ///
    /// Returns `None` when the inputs cannot support a meaningful budget: an
    /// unknown/zero window, or a reserve that leaves less than a quarter of the
    /// window for history.
    pub fn derive(window_tokens: usize, threshold_ratio: f64, reserve_tokens: usize) -> Option<Self> {
        if window_tokens == 0 || window_tokens <= reserve_tokens {
            return None;
        }
        if window_tokens.saturating_sub(reserve_tokens) < window_tokens / 4 {
            return None;
        }
        let ratio = if threshold_ratio.is_finite() {
            threshold_ratio.clamp(0.1, 1.0)
        } else {
            0.8
        };
        let hard_tokens = window_tokens - reserve_tokens;
        // A ratio that lands above the hard bound is clamped down to it: the
        // trigger may never sit above the limit it is supposed to protect.
        let soft_tokens = ((window_tokens as f64) * ratio).floor().max(1.0) as usize;
        Some(Self {
            soft_tokens: soft_tokens.min(hard_tokens).max(1),
            hard_tokens,
        })
    }

    /// Tokens the projection should be reduced to when elision runs: half the
    /// soft bound, so a single pass buys headroom instead of eliding again on
    /// the next turn.
    pub fn target_tokens(&self) -> usize {
        (self.soft_tokens / 2).max(1)
    }

    pub fn should_trigger(&self, projected_tokens: usize) -> bool {
        CompactionTrigger::should_trigger(projected_tokens, self.soft_tokens)
    }

    pub fn exceeds_hard_bound(&self, projected_tokens: usize) -> bool {
        projected_tokens > self.hard_tokens
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_is_deterministic_and_monotone() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert!(estimate_tokens("a longer body of text") > estimate_tokens("short"));
    }

    #[test]
    fn estimate_tokens_never_under_charges_non_ascii_text() {
        // Three-byte CJK characters: 100 of them are ~100-150 real tokens on
        // production vocabularies, so the estimate must not come in below one
        // token per character.
        let cjk = "漢字かな交じり文".repeat(12);
        assert!(estimate_tokens(&cjk) >= cjk.chars().count());
        // Two-byte accented Latin is charged at least as much as its character
        // count, which is above the real rate but never below it.
        let accents = "é".repeat(200);
        assert!(estimate_tokens(&accents) >= 400);
        // Mixed text still tracks the ASCII rule.
        assert_eq!(estimate_tokens("abcd漢"), 3);
    }

    #[test]
    fn budget_keeps_the_trigger_below_the_limit_it_protects() {
        // 64k window, default ratio, 18k reserved for tools and completion.
        let budget = CompactionBudget::derive(64_000, 0.8, 18_000).unwrap();
        assert_eq!(budget.hard_tokens, 46_000);
        assert!(budget.soft_tokens <= budget.hard_tokens);
        assert_eq!(budget.soft_tokens, 46_000, "ratio above the hard bound clamps down");
        assert_eq!(budget.target_tokens(), 23_000);

        // A window with room to spare keeps the ratio.
        let budget = CompactionBudget::derive(200_000, 0.8, 20_000).unwrap();
        assert_eq!(budget.soft_tokens, 160_000);
        assert_eq!(budget.hard_tokens, 180_000);
        assert!(budget.should_trigger(160_000));
        assert!(!budget.should_trigger(159_999));
        assert!(budget.exceeds_hard_bound(180_001));
    }

    #[test]
    fn budget_refuses_inputs_it_cannot_honour() {
        assert!(CompactionBudget::derive(0, 0.8, 0).is_none());
        // Reserve eats the window.
        assert!(CompactionBudget::derive(8_000, 0.8, 8_000).is_none());
        // Reserve leaves less than a quarter of the window for history.
        assert!(CompactionBudget::derive(8_000, 0.8, 6_500).is_none());
    }

    #[test]
    fn budget_clamps_hostile_ratios() {
        for ratio in [f64::NAN, f64::INFINITY, -3.0, 0.0, 99.0] {
            let budget = CompactionBudget::derive(100_000, ratio, 10_000).unwrap();
            assert!(budget.soft_tokens >= 1);
            assert!(budget.soft_tokens <= budget.hard_tokens);
        }
    }
}
