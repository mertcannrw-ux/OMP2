use serde::{Deserialize, Serialize};
/// Lifecycle status shared across jobs, sessions, and streams.
///
/// NOTE: this enum mixes lifecycles today (see `is_terminal_job` /
/// `is_terminal_session`). New code should match the subset for its own
/// lifecycle; the DOM stores these as `TypedValue::String` via
/// `serde(rename_all = "snake_case")`, so the strings `"succeeded"`,
/// `"failed"`, `"cancelled"`, `"committed"`, `"completed"`, and `"detached"`
/// are the canonical terminal spellings (with `"completed"` kept as a legacy
/// alias of `Finalized`).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Queued,
    Active,
    Running,
    Finalized,
    Committed,
    CancelRequested,
    Cancelled,
    Failed,
    Succeeded,
    Detached,
    Truncated,
    Unknown,
    WriteFailure,
}

impl Status {
    /// Terminal states for background/sandbox jobs: no further transitions.
    pub fn is_terminal_job(&self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Cancelled | Self::Finalized | Self::Detached
        )
    }
    /// Terminal states for session/turn lifecycles (superset: commits end a
    /// session, detachment ends it from this host's view).
    pub fn is_terminal_session(&self) -> bool {
        matches!(
            self,
            Self::Succeeded
                | Self::Failed
                | Self::Cancelled
                | Self::Finalized
                | Self::Committed
                | Self::Detached
        )
    }
    /// Parse the DOM `status` attribute spelling, accepting the legacy
    /// `"completed"` alias for `Finalized`.
    pub fn from_dom_str(value: &str) -> Option<Self> {
        match value {
            "completed" => Some(Self::Finalized),
            other => serde_json::from_value(serde_json::Value::String(other.into())).ok(),
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
    pub diagnostics: Option<serde_json::Value>,
}
impl StructuredError {
    pub fn new(code: impl Into<String>, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            retryable,
            diagnostics: None,
        }
    }
}
impl std::fmt::Display for StructuredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}
impl std::error::Error for StructuredError {}
