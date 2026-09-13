use omp_types::{
    ArtifactId, Status, StructuredError, ToolCallId,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// Diagnostic severity for tool execution events.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

/// Output format emitted by a tool execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    Text,
    Json,
    Markdown,
    Binary,
}

/// Diagnostic message produced during tool evaluation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolDiagnostic {
    pub severity: DiagnosticSeverity,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl ToolDiagnostic {
    pub fn info(message: impl Into<String>) -> Self {
        Self {
            severity: DiagnosticSeverity::Info,
            message: message.into(),
            code: None,
            details: None,
        }
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: DiagnosticSeverity::Warning,
            message: message.into(),
            code: None,
            details: None,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            severity: DiagnosticSeverity::Error,
            message: message.into(),
            code: None,
            details: None,
        }
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }
}

/// Structured usage statistics for a tool call.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolUsage {
    pub duration_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_time_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_read: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_written: Option<u64>,
}

/// Output produced by a tool execution.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
    pub truncated: bool,
    pub format: OutputFormat,
}

impl ToolOutput {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            payload: None,
            truncated: false,
            format: OutputFormat::Text,
        }
    }

    pub fn json(payload: serde_json::Value) -> Self {
        let content = serde_json::to_string_pretty(&payload).unwrap_or_default();
        Self {
            content,
            payload: Some(payload),
            truncated: false,
            format: OutputFormat::Json,
        }
    }

    pub fn truncated_text(content: impl Into<String>, payload: Option<serde_json::Value>) -> Self {
        Self {
            content: content.into(),
            payload,
            truncated: true,
            format: OutputFormat::Text,
        }
    }
}

/// Bounded limits governing tool execution.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolLimits {
    pub max_output_bytes: usize,
    pub max_runtime_ms: u64,
    pub max_artifacts: usize,
    pub max_concurrent_jobs: usize,
}

impl Default for ToolLimits {
    fn default() -> Self {
        Self {
            max_output_bytes: 100_000,
            max_runtime_ms: 30_000,
            max_artifacts: 10,
            max_concurrent_jobs: 4,
        }
    }
}

/// Parse a user-supplied `timeout` (seconds, float) into clamped milliseconds.
///
/// NaN/negative values fall back to `default_ms`; finite values are clamped to
/// `max_ms` so `1e20` cannot overflow the `as u64` cast or stall the host.
pub fn parse_timeout_ms(input: &serde_json::Value, default_ms: u64, max_ms: u64) -> u64 {
    input
        .get("timeout")
        .and_then(|v| v.as_f64())
        .filter(|s| s.is_finite() && *s > 0.0)
        .map(|s| (s * 1000.0).clamp(1.0, max_ms as f64) as u64)
        .unwrap_or(default_ms)
}

/// Required capability markers for a tool.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolCapabilities {
    pub required: Vec<String>,
}

impl ToolCapabilities {
    pub fn new(caps: &[&str]) -> Self {
        Self {
            required: caps.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn contains(&self, cap: &str) -> bool {
        self.required.iter().any(|c| c == cap)
    }
}

/// Error type produced during tool execution or input validation.
#[derive(thiserror::Error, Debug, Clone, Serialize, Deserialize)]
pub enum ToolError {
    #[error("validation error: {message}")]
    Validation {
        message: String,
        details: Option<serde_json::Value>,
    },
    #[error("capability denied: {capability}")]
    CapabilityDenied { capability: String },
    #[error("resource limit exceeded: {limit} (observed: {observed}, max: {max})")]
    LimitExceeded {
        limit: String,
        observed: u64,
        max: u64,
    },
    #[error("host execution failed: {message}")]
    Execution {
        message: String,
        details: Option<serde_json::Value>,
    },
    #[error("resource not found: {0}")]
    NotFound(String),
    #[error("host execution failed: {0}")]
    HostExecution(String),
    #[error("tool cancelled: {0}")]
    Cancelled(String),
    #[error("structured failure: {0:?}")]
    Structured(StructuredError),
}

impl From<StructuredError> for ToolError {
    fn from(err: StructuredError) -> Self {
        Self::Structured(err)
    }
}

impl From<ToolError> for StructuredError {
    /// Errors that already carry a structured code keep it; the remaining
    /// variants are reported under `tool_error` with their rendered message.
    fn from(error: ToolError) -> Self {
        match error {
            ToolError::Structured(inner) => inner,
            other => Self::new("tool_error", other.to_string(), false),
        }
    }
}

/// Host requests generated by tools during execution.
/// Tools only describe semantic state and request host capabilities; they never touch raw OS shell directly.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum HostRequest {
    ReadResource {
        path: String,
        selector: Option<String>,
        raw: bool,
    },
    ExecuteProcess {
        command: String,
        cwd: Option<String>,
        env: BTreeMap<String, String>,
        timeout_ms: Option<u64>,
        pty: bool,
        is_async: bool,
        capability_demands: Vec<String>,
    },
    WriteFile {
        path: String,
        content: String,
        atomic: bool,
    },
    EditFile {
        path: String,
        expected_tag: Option<String>,
        patch_text: String,
    },
    EvalCode {
        code: String,
        language: String,
        reset: bool,
        timeout_ms: Option<u64>,
    },
    SpawnAgent {
        context: String,
        tasks: Vec<serde_json::Value>,
        isolated_workspace: bool,
        convar_overrides: BTreeMap<String, String>,
    },
    ReportQa {
        report: serde_json::Value,
    },
    DynLookup {
        query: Option<String>,
        action: Option<String>,
        help: bool,
        args: Option<serde_json::Value>,
    },
}

/// Response returned by the host gateway in response to a HostRequest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostResponse {
    pub output: ToolOutput,
    pub diagnostics: Vec<ToolDiagnostic>,
    pub usage: Option<ToolUsage>,
    pub artifacts: Vec<ArtifactId>,
}

impl HostResponse {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            output: ToolOutput::text(content),
            diagnostics: Vec::new(),
            usage: None,
            artifacts: Vec::new(),
        }
    }

    pub fn with_diagnostics(mut self, diags: Vec<ToolDiagnostic>) -> Self {
        self.diagnostics = diags;
        self
    }

    pub fn with_usage(mut self, usage: ToolUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn with_artifacts(mut self, artifacts: Vec<ArtifactId>) -> Self {
        self.artifacts = artifacts;
        self
    }
}

/// Host capability gateway trait implemented by the runtime/host control plane.
pub trait HostGateway: Send + Sync {
    fn request(&self, req: HostRequest) -> Result<HostResponse, ToolError>;
}

/// Simulated in-memory gateway restricted to unit tests and local evaluation.
/// Callers must push explicit responses; deceptive success fallbacks have been removed.
#[derive(Default)]
pub struct DirectHostGateway {
    pub responses: Mutex<Vec<HostResponse>>,
}

impl DirectHostGateway {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_response(&self, resp: HostResponse) {
        self.responses.lock().push(resp);
    }
}

impl HostGateway for DirectHostGateway {
    fn request(&self, _req: HostRequest) -> Result<HostResponse, ToolError> {
        let mut queue = self.responses.lock();
        if !queue.is_empty() {
            return Ok(queue.remove(0));
        }
        Err(ToolError::Execution {
            message: "DirectHostGateway has no queued responses; real execution requires ToolHost"
                .into(),
            details: None,
        })
    }
}

/// Execution result returned by a tool executor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolExecutionResult {
    pub output: ToolOutput,
    pub diagnostics: Vec<ToolDiagnostic>,
    pub usage: Option<ToolUsage>,
    pub artifacts: Vec<ArtifactId>,
}

impl ToolExecutionResult {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            output: ToolOutput::text(content),
            diagnostics: Vec::new(),
            usage: None,
            artifacts: Vec::new(),
        }
    }

    pub fn from_host(resp: HostResponse) -> Self {
        Self {
            output: resp.output,
            diagnostics: resp.diagnostics,
            usage: resp.usage,
            artifacts: resp.artifacts,
        }
    }
}

/// Execution handler trait for tools.
pub trait ToolExecutor: Send + Sync {
    fn execute(
        &self,
        call: &ToolCall,
        gateway: &dyn HostGateway,
    ) -> Result<ToolExecutionResult, ToolError>;
}

/// A live or journaled tool call instance.
/// Every call carries `i` intent, tool name/version, typed input, output, diagnostics, usage, and artifact references.
/// The intent streams early and is journaled in the call element.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: ToolCallId,
    pub name: String,
    pub version: String,
    /// Intent `i` parameter describing user/model intent, streamed early.
    pub intent: String,
    pub input: serde_json::Value,
    pub output: Option<ToolOutput>,
    pub diagnostics: Vec<ToolDiagnostic>,
    pub usage: Option<ToolUsage>,
    pub artifacts: Vec<ArtifactId>,
    pub status: Status,
}

impl ToolCall {
    pub fn new(
        id: ToolCallId,
        name: impl Into<String>,
        version: impl Into<String>,
        input: serde_json::Value,
    ) -> Self {
        let name = name.into();
        let version = version.into();
        let intent = input
            .get("i")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Self {
            id,
            name,
            version,
            intent,
            input,
            output: None,
            diagnostics: Vec::new(),
            usage: None,
            artifacts: Vec::new(),
            status: Status::Active,
        }
    }

    pub fn tool(&self) -> &str {
        &self.name
    }

    pub fn set_result(&mut self, result: ToolExecutionResult) {
        self.output = Some(result.output);
        self.diagnostics.extend(result.diagnostics);
        self.usage = result.usage;
        self.artifacts.extend(result.artifacts);
        self.status = Status::Succeeded;
    }

    pub fn set_failed(&mut self, err: ToolError) {
        self.diagnostics
            .push(ToolDiagnostic::error(err.to_string()));
        self.status = Status::Failed;
    }
}

/// Exact ToolDefinition as specified in the playbook:
/// `name`, `version`, `intent_schema`, `parameters`, `capabilities`, `limits`, `execute`, and `component_projection`.
#[derive(Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub version: String,
    pub intent_schema: serde_json::Value,
    pub parameters: serde_json::Value,
    pub capabilities: Vec<String>,
    pub limits: ToolLimits,
    pub execute: Arc<dyn ToolExecutor>,
    pub component_projection: String,
}

impl fmt::Debug for ToolDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolDefinition")
            .field("name", &self.name)
            .field("version", &self.version)
            .field("capabilities", &self.capabilities)
            .field("limits", &self.limits)
            .field("component_projection", &self.component_projection)
            .finish()
    }
}

impl ToolDefinition {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        intent_schema: serde_json::Value,
        parameters: serde_json::Value,
        capabilities: Vec<String>,
        limits: ToolLimits,
        execute: Arc<dyn ToolExecutor>,
        component_projection: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            intent_schema,
            parameters,
            capabilities,
            limits,
            execute,
            component_projection: component_projection.into(),
        }
    }

    pub fn validate_input(&self, input: &serde_json::Value) -> Result<(), ToolError> {
        if !input.is_object() {
            return Err(ToolError::Validation {
                message: "tool input must be a JSON object".into(),
                details: Some(input.clone()),
            });
        }
        Ok(())
    }
}
