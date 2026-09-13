use crate::limits::{LimitPolicy, TruncationDiag};
use crate::{ArtifactId, JobId, ProtocolVersion, WorkspaceViewId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Explicit, scoped capability granted to the sandbox for execution.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum SandboxCapability {
    Read { root: PathBuf },
    Write { root: PathBuf },
    Execute { command: String },
    Network { hosts: Vec<String> },
    EnvAccess { keys: Vec<String> },
    SpawnSubprocess,
    Custom(String),
}

/// Request sent from the trusted host across the boundary into the sandbox environment.
/// The sandbox cannot select tools, mutate session state, or escalate capabilities.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SandboxRequest {
    pub request_id: String,
    pub job_id: JobId,
    pub operation_name: String,
    pub capabilities: Vec<SandboxCapability>,
    pub workspace_view_id: Option<WorkspaceViewId>,
    pub input_artifacts: Vec<ArtifactId>,
    pub limits: LimitPolicy,
    pub protocol_version: ProtocolVersion,
    pub env: BTreeMap<String, String>,
    pub args: Vec<String>,
    pub stdin: Option<String>,
}

impl SandboxRequest {
    pub fn new(
        request_id: impl Into<String>,
        job_id: JobId,
        operation_name: impl Into<String>,
        limits: LimitPolicy,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            job_id,
            operation_name: operation_name.into(),
            capabilities: Vec::new(),
            workspace_view_id: None,
            input_artifacts: Vec::new(),
            limits,
            protocol_version: ProtocolVersion::CURRENT,
            env: BTreeMap::new(),
            args: Vec::new(),
            stdin: None,
        }
    }

    pub fn with_capability(mut self, cap: SandboxCapability) -> Self {
        self.capabilities.push(cap);
        self
    }

    pub fn with_workspace_view(mut self, view_id: WorkspaceViewId) -> Self {
        self.workspace_view_id = Some(view_id);
        self
    }

    pub fn with_input_artifact(mut self, id: ArtifactId) -> Self {
        self.input_artifacts.push(id);
        self
    }

    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    pub fn with_env(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.env.insert(key.into(), val.into());
        self
    }

    pub fn with_stdin(mut self, stdin: impl Into<String>) -> Self {
        self.stdin = Some(stdin.into());
        self
    }
    pub fn validate(&self) -> Result<(), crate::StructuredError> {
        // `LimitPolicy::validate` covers zero budgets and is the single place
        // that rule lives; per-field string bounds below give early, precise
        // errors instead of a late 1 MiB wire-size failure.
        self.limits.validate().map_err(|error| {
            crate::StructuredError::new("invalid_sandbox_request", error.message, false)
        })?;
        let invalid = self.protocol_version != ProtocolVersion::CURRENT
            || self.request_id.is_empty()
            || self.request_id.len() > 128
            || self.operation_name.is_empty()
            || self.operation_name.len() > 256
            || self.workspace_view_id.is_none()
            || self.capabilities.len() > 128
            || self.args.len() > 1024
            || self.args.iter().any(|arg| arg.len() > 65_536)
            || self.env.len() > 256
            || self
                .env
                .iter()
                .any(|(key, value)| key.is_empty() || key.len() > 256 || value.len() > 65_536)
            || self.stdin.as_ref().is_some_and(|stdin| stdin.len() > crate::MAX_WIRE_BYTES)
            || self.limits.max_bytes > crate::MAX_WIRE_BYTES;
        if invalid {
            return Err(crate::StructuredError::new(
                "invalid_sandbox_request",
                "Missing identity, view, finite limits, oversized field, or supported protocol",
                false,
            ));
        }
        let bytes = serde_json::to_vec(self).map_err(|e| {
            crate::StructuredError::new("invalid_sandbox_request", e.to_string(), false)
        })?;
        if bytes.len() > crate::MAX_WIRE_BYTES {
            return Err(crate::StructuredError::new(
                "wire_size_limit",
                "Sandbox request exceeds one MiB",
                false,
            ));
        }
        Ok(())
    }
}

/// Resource usage metrics reported by the sandbox environment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SandboxUsage {
    pub wall_time_ms: u64,
    pub cpu_time_ms: Option<u64>,
    pub memory_bytes: Option<u64>,
    pub bytes_written: u64,
    pub child_processes_spawned: u32,
}

/// Final exit status returned by a sandbox job execution.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SandboxExitStatus {
    Success { code: i32 },
    Error { code: i32, message: String },
    Signaled { signal: i32 },
    Timeout,
    ForcedKill,
    Cancelled,
}

impl SandboxExitStatus {
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success { code: 0 })
    }
}

/// Events streamed back from the sandbox execution environment across the boundary.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SandboxEvent {
    StdoutChunk(Vec<u8>),
    StderrChunk(Vec<u8>),
    StructuredResult(serde_json::Value),
    Diagnostic(TruncationDiag),
    Usage(SandboxUsage),
    ArtifactReference(ArtifactId),
    CancellationAck,
    Exit(SandboxExitStatus),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SandboxEventEnvelope {
    pub protocol_version: ProtocolVersion,
    pub request_id: String,
    pub job_id: JobId,
    pub sequence: u64,
    pub event: SandboxEvent,
}
