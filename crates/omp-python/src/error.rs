use thiserror::Error;

#[derive(Debug, Error)]
pub enum PythonExtensionError {
    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("protocol violation: {0}")]
    ProtocolViolation(String),

    #[error("remote validation failed: {0}")]
    ValidationFailed(String),

    #[error("oversized payload: size {size} bytes exceeds limit of {limit} bytes")]
    OversizedPayload { size: usize, limit: usize },

    #[error("undeclared capability: '{capability}' required but not declared")]
    UndeclaredCapability { capability: String },

    #[error("prohibited import '{module}' in remote function")]
    ProhibitedImport { module: String },

    #[error("forbidden dynamic code construct: {0}")]
    DynamicCodeForbidden(String),

    #[error("extension '{extension_id}' terminated or unavailable: {reason}")]
    ExtensionTerminated {
        extension_id: String,
        reason: String,
    },

    #[error("operation timed out after {timeout_ms} ms: {context}")]
    Timeout { timeout_ms: u64, context: String },
}
