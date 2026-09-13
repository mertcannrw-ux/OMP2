use serde_json::json;
use std::sync::Arc;

use crate::definition::{
    HostGateway, HostRequest, ToolCall, ToolDefinition,
    ToolDiagnostic, ToolError, ToolExecutionResult, ToolExecutor, ToolLimits,
};

/// Executor for the permanent `Write` tool.
#[derive(Default)]
pub struct WriteExecutor;

impl WriteExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl ToolExecutor for WriteExecutor {
    fn execute(
        &self,
        call: &ToolCall,
        gateway: &dyn HostGateway,
    ) -> Result<ToolExecutionResult, ToolError> {
        let path = call
            .input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::Validation {
                message: "missing required 'path' parameter".into(),
                details: Some(call.input.clone()),
            })?;

        let content = call
            .input
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::Validation {
                message: "missing required 'content' parameter".into(),
                details: Some(call.input.clone()),
            })?;

        if path.trim().is_empty() {
            return Err(ToolError::Validation {
                message: "file path cannot be empty".into(),
                details: Some(call.input.clone()),
            });
        }
        if path.contains('\0') {
            return Err(ToolError::Validation {
                message: "file path cannot contain null bytes".into(),
                details: Some(call.input.clone()),
            });
        }

        let line_count = content.lines().count();
        let byte_count = content.len();

        let req = HostRequest::WriteFile {
            path: path.to_string(),
            content: content.to_string(),
            atomic: true,
        };

        let resp = gateway.request(req)?;
        let mut result = ToolExecutionResult::from_host(resp);

        let has_conflict = result.diagnostics.iter().any(|d| {
            d.severity == crate::definition::DiagnosticSeverity::Error
                || d.code.as_deref() == Some("conflict")
                || d.code.as_deref() == Some("write_failure")
        });

        if !has_conflict {
            result.diagnostics.push(ToolDiagnostic::info(format!(
                "Wrote {} bytes ({} lines) to {}",
                byte_count, line_count, path
            )));
        }

        // Strict bounded output enforcement (50,000 bytes)
        const MAX_OUTPUT_BYTES: usize = 50_000;
        if result.output.content.len() > MAX_OUTPUT_BYTES {
            let mut cut_idx = MAX_OUTPUT_BYTES;
            while cut_idx > 0 && !result.output.content.is_char_boundary(cut_idx) {
                cut_idx -= 1;
            }
            result.output.content.truncate(cut_idx);
            result.output.truncated = true;
            result.diagnostics.push(ToolDiagnostic::warning(format!(
                "Write output exceeded limit of {} bytes and was truncated.",
                MAX_OUTPUT_BYTES
            )));
        }

        Ok(result)
    }
}

/// Creates the permanent `Write` tool definition.
pub fn write_tool_definition(executor: Arc<WriteExecutor>) -> ToolDefinition {
    ToolDefinition::new(
        "Write",
        "1.0.0",
        json!({
            "type": "string",
            "description": "Present-participle intent describing what file is being written"
        }),
        json!({
            "type": "object",
            "required": ["path", "content", "i"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to create or overwrite"
                },
                "content": {
                    "type": "string",
                    "description": "Complete file content"
                },
                "i": {
                    "type": "string",
                    "description": "Present-participle intent"
                }
            }
        }),
        vec!["write_fs".into()],
        ToolLimits {
            max_output_bytes: 50_000,
            max_runtime_ms: 15_000,
            max_artifacts: 5,
            max_concurrent_jobs: 4,
        },
        executor,
        "write",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::DirectHostGateway;

    #[test]
    fn test_write_reject_empty_path() {
        let executor = WriteExecutor;
        let gateway = DirectHostGateway::new();
        let call = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "Write",
            "1.0.0",
            json!({
                "path": "   ",
                "content": "test",
                "i": "Writing empty"
            }),
        );
        let err = executor.execute(&call, &gateway).unwrap_err();
        assert!(
            matches!(err, ToolError::Validation { message, .. } if message.contains("cannot be empty"))
        );
    }

    #[test]
    fn test_write_reject_null_byte_path() {
        let executor = WriteExecutor;
        let gateway = DirectHostGateway::new();
        let call = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "Write",
            "1.0.0",
            json!({
                "path": "src/\0bad.rs",
                "content": "test",
                "i": "Writing null byte"
            }),
        );
        let err = executor.execute(&call, &gateway).unwrap_err();
        assert!(
            matches!(err, ToolError::Validation { message, .. } if message.contains("null bytes"))
        );
    }
}
