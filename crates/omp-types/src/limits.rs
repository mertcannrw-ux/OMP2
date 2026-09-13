use crate::ArtifactId;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Centralized resource limits enforced by the trusted host across crossing streams,
/// processes, artifacts, and execution time.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LimitPolicy {
    /// Maximum bytes allowed across stream output before truncation occurs.
    pub max_bytes: usize,
    /// Maximum lines allowed across stream output.
    pub max_lines: usize,
    /// Maximum crossing event count before throttling or terminating.
    pub max_events: usize,
    /// Maximum resident bytes enforced by the OS process boundary.
    pub max_memory_bytes: usize,
    /// Maximum wall-clock execution time allowed for the job.
    pub max_wall_time: Duration,
    /// Optional CPU execution time budget.
    pub max_cpu_time: Option<Duration>,
    /// Maximum number of child processes allowed to be spawned.
    pub max_child_processes: u32,
    /// Maximum artifact size in bytes.
    pub max_artifact_bytes: usize,
    /// Maximum concurrent jobs allowed for the host or view.
    pub max_concurrent_jobs: u32,
    /// Grace period between cancel_requested and forced_kill termination.
    pub cancel_grace_period: Duration,
}

impl Default for LimitPolicy {
    fn default() -> Self {
        Self {
            max_bytes: 1024 * 1024, // 1 MiB default
            max_lines: 50_000,
            max_events: 10_000,
            max_memory_bytes: 256 * 1024 * 1024,
            max_wall_time: Duration::from_secs(120),
            max_cpu_time: None,
            max_child_processes: 16,
            max_artifact_bytes: 50 * 1024 * 1024, // 50 MiB
            max_concurrent_jobs: 8,
            cancel_grace_period: Duration::from_secs(5),
        }
    }
}

impl LimitPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reject degenerate policies: zero byte/line/event/memory/concurrency
    /// budgets silently truncate everything, and zero wall time kills jobs
    /// instantly. Builders can still construct them; enforcement points call
    /// this before use.
    pub fn validate(&self) -> Result<(), crate::StructuredError> {
        let invalid = self.max_bytes == 0
            || self.max_lines == 0
            || self.max_events == 0
            || self.max_memory_bytes == 0
            || self.max_wall_time.is_zero()
            || self.max_child_processes == 0
            || self.max_artifact_bytes == 0
            || self.max_concurrent_jobs == 0;
        if invalid {
            return Err(crate::StructuredError::new(
                "invalid_limits",
                "LimitPolicy budgets must all be nonzero",
                false,
            ));
        }
        Ok(())
    }

    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    pub fn with_max_lines(mut self, max_lines: usize) -> Self {
        self.max_lines = max_lines;
        self
    }

    pub fn with_max_wall_time(mut self, duration: Duration) -> Self {
        self.max_wall_time = duration;
        self
    }

    pub fn with_grace_period(mut self, duration: Duration) -> Self {
        self.cancel_grace_period = duration;
        self
    }
}

/// Structured diagnostic emitted on stream or output truncation.
/// Records the limit breached, total observed amount, preserved artifact reference (if saved),
/// and whether the model context may fetch the full artifact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TruncationDiag {
    pub limit_bytes: usize,
    pub observed_bytes: usize,
    pub limit_lines: usize,
    pub observed_lines: usize,
    pub artifact_id: Option<ArtifactId>,
    pub fetchable: bool,
    pub reason: String,
}

impl TruncationDiag {
    pub fn new(
        limit_bytes: usize,
        observed_bytes: usize,
        limit_lines: usize,
        observed_lines: usize,
        artifact_id: Option<ArtifactId>,
        fetchable: bool,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            limit_bytes,
            observed_bytes,
            limit_lines,
            observed_lines,
            artifact_id,
            fetchable,
            reason: reason.into(),
        }
    }
}
