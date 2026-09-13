use crate::role::ActorRole;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("actor '{actor}' with role {role:?} is unauthorized for {attempted_action}")]
    Unauthorized {
        actor: String,
        role: ActorRole,
        attempted_action: String,
    },

    #[error("stale base offset: provided {actual}, current authoritative offset {expected}")]
    StaleBaseOffset { actual: u64, expected: u64 },

    #[error("offset {requested} unavailable (oldest retained {oldest_available})")]
    OffsetUnavailable {
        requested: u64,
        oldest_available: u64,
    },

    #[error("actor not found: {0}")]
    ActorNotFound(String),

    #[error("actor already attached: {0}")]
    ActorAlreadyAttached(String),

    #[error("underlying state error: {0}")]
    State(#[from] omp_state::StateError),
    #[error("host execution failed: {0}")]
    Execution(omp_types::StructuredError),

    #[error("subscriber queue full for actor: {0}")]
    SubscriberQueueFull(String),

    #[error("subscriber lagged: dropped {dropped_frames} presentation frames")]
    SubscriberLagged { dropped_frames: usize },

    #[error("artifact not found: {0}")]
    ArtifactNotFound(String),

    #[error("artifact scope violation: actor '{actor}' cannot access artifact '{artifact}'")]
    ArtifactScopeViolation { artifact: String, actor: String },

    #[error("command replayed or duplicated: {0}")]
    CommandReplayed(String),

    #[error("invalid command: {0}")]
    InvalidCommand(String),

    #[error("invalid patch: {0}")]
    InvalidPatch(String),

    #[error("job not found: {0}")]
    JobNotFound(String),

    #[error("handshake failed: {0}")]
    HandshakeFailed(String),

    #[error("session terminated: {0}")]
    SessionTerminated(String),
    #[error(
        "unsupported transport '{0}': network transport requires an external framing provider; use in-memory transport or local channel"
    )]
    UnsupportedTransport(String),

    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error("wire limit exceeded: {0}")]
    WireLimit(String),
}

impl ServerError {
    pub fn structured(&self) -> omp_types::StructuredError {
        let (code, retryable) = match self {
            Self::Unauthorized { .. } => ("unauthorized", false),
            Self::StaleBaseOffset { .. } => ("stale_base_offset", true),
            Self::OffsetUnavailable { .. } => ("offset_unavailable", false),
            Self::ActorNotFound(_) => ("actor_not_found", false),
            Self::ActorAlreadyAttached(_) => ("actor_already_attached", false),
            Self::State(e) => return e.structured(),
            Self::Execution(error) => return error.clone(),
            Self::SubscriberQueueFull(_) => ("subscriber_queue_full", true),
            Self::SubscriberLagged { .. } => ("subscriber_lagged", false),
            Self::ArtifactNotFound(_) => ("artifact_not_found", false),
            Self::ArtifactScopeViolation { .. } => ("artifact_scope_violation", false),
            Self::CommandReplayed(_) => ("command_replayed", false),
            Self::InvalidCommand(_) => ("invalid_command", false),
            Self::InvalidPatch(_) => ("invalid_patch", false),
            Self::JobNotFound(_) => ("job_not_found", false),
            Self::HandshakeFailed(_) => ("handshake_failed", false),
            Self::SessionTerminated(_) => ("session_terminated", false),
            Self::UnsupportedTransport(_) => ("unsupported_transport", false),
            Self::Io(_) => ("io_error", false),
            Self::WireLimit(_) => ("wire_size_limit", false),
        };
        omp_types::StructuredError::new(code, self.to_string(), retryable)
    }
}

impl From<ServerError> for omp_types::StructuredError {
    fn from(err: ServerError) -> Self {
        err.structured()
    }
}
