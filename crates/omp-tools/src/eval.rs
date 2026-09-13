use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

use crate::definition::{
    HostGateway, HostRequest, ToolCall, ToolDefinition,
    ToolDiagnostic, ToolError, ToolExecutionResult, ToolExecutor, ToolLimits,
};

/// Supported execution language in Eval.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalLanguage {
    Py,
    Js,
}

impl EvalLanguage {
    pub fn parse(s: &str) -> Result<Self, ToolError> {
        match s.to_ascii_lowercase().as_str() {
            "py" | "python" => Ok(Self::Py),
            "js" | "javascript" => Ok(Self::Js),
            other => Err(ToolError::Validation {
                message: format!(
                    "unsupported eval language '{}': only 'py' and 'js' supported",
                    other
                ),
                details: None,
            }),
        }
    }
}

/// Structured error classification for Eval results.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EvalFailure {
    pub error_type: String,
    pub message: String,
    pub retryable: bool,
}

impl EvalFailure {
    pub fn classify(err_str: &str) -> Self {
        let is_syntax = err_str.contains("SyntaxError") || err_str.contains("ParseError");
        let is_timeout = err_str.contains("Timeout") || err_str.contains("timed out");
        let is_name_err = err_str.contains("NameError") || err_str.contains("ReferenceError");

        let retryable = is_timeout;
        let error_type = if is_syntax {
            "SyntaxError".to_string()
        } else if is_timeout {
            "Timeout".to_string()
        } else if is_name_err {
            "ReferenceError".to_string()
        } else {
            "RuntimeError".to_string()
        };

        Self {
            error_type,
            message: err_str.to_string(),
            retryable,
        }
    }
}

/// Executor for the permanent `Eval` tool.
#[derive(Default)]
pub struct EvalExecutor;

impl EvalExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl ToolExecutor for EvalExecutor {
    fn execute(
        &self,
        call: &ToolCall,
        gateway: &dyn HostGateway,
    ) -> Result<ToolExecutionResult, ToolError> {
        let code = call
            .input
            .get("code")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::Validation {
                message: "missing required 'code' parameter".into(),
                details: Some(call.input.clone()),
            })?;

        let lang_str = call
            .input
            .get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("py");
        let lang = EvalLanguage::parse(lang_str)?;

        let reset = call
            .input
            .get("reset")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let timeout_ms = Some(crate::definition::parse_timeout_ms(
            &call.input,
            30_000,
            120_000,
        ));

        let req = HostRequest::EvalCode {
            code: code.to_string(),
            language: match lang {
                EvalLanguage::Py => "py".to_string(),
                EvalLanguage::Js => "js".to_string(),
            },
            reset,
            timeout_ms,
        };

        let resp = gateway.request(req)?;
        let mut result = ToolExecutionResult::from_host(resp);

        // Check if output content indicates error and classify retryability
        if result
            .output
            .content
            .contains("Traceback (most recent call last):")
            || result.output.content.contains("Error:")
        {
            let failure = EvalFailure::classify(&result.output.content);
            result.diagnostics.push(ToolDiagnostic::warning(format!(
                "Eval execution failed (type: {}, retryable: {}): {}",
                failure.error_type, failure.retryable, failure.message
            )));
        }
        // Strict bounded output enforcement (100,000 bytes)
        const MAX_OUTPUT_BYTES: usize = 100_000;
        if result.output.content.len() > MAX_OUTPUT_BYTES {
            let mut cut_idx = MAX_OUTPUT_BYTES;
            while cut_idx > 0 && !result.output.content.is_char_boundary(cut_idx) {
                cut_idx -= 1;
            }
            result.output.content.truncate(cut_idx);
            result.output.truncated = true;
            result.diagnostics.push(ToolDiagnostic::warning(format!(
                "Eval output exceeded limit of {} bytes and was truncated.",
                MAX_OUTPUT_BYTES
            )));
        }

        Ok(result)
    }
}

/// Creates the permanent `Eval` tool definition.
pub fn eval_tool_definition(executor: Arc<EvalExecutor>) -> ToolDefinition {
    ToolDefinition::new(
        "Eval",
        "1.0.0",
        json!({
            "type": "string",
            "description": "Present-participle intent describing code execution in persistent kernel"
        }),
        json!({
            "type": "object",
            "required": ["code", "language"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Code to run in persistent kernel"
                },
                "language": {
                    "type": "string",
                    "enum": ["py", "js"],
                    "description": "Kernel runtime language ('py' for Python, 'js' for JavaScript)"
                },
                "title": {
                    "type": "string",
                    "description": "Short label for transcript"
                },
                "reset": {
                    "type": "boolean",
                    "description": "Reset persistent kernel before running"
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (0 = disabled)"
                },
                "i": {
                    "type": "string",
                    "description": "Present-participle intent"
                }
            }
        }),
        vec!["embedded_runtime".into(), "job_control".into()],
        ToolLimits {
            max_output_bytes: 100_000,
            max_runtime_ms: 60_000,
            max_artifacts: 10,
            max_concurrent_jobs: 2,
        },
        executor,
        "eval",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::HostResponse;
    use crate::definition::DirectHostGateway;

    #[test]
    fn test_eval_language_parse() {
        assert_eq!(EvalLanguage::parse("py").unwrap(), EvalLanguage::Py);
        assert_eq!(EvalLanguage::parse("python").unwrap(), EvalLanguage::Py);
        assert_eq!(EvalLanguage::parse("js").unwrap(), EvalLanguage::Js);
        assert_eq!(EvalLanguage::parse("javascript").unwrap(), EvalLanguage::Js);
        assert!(EvalLanguage::parse("ruby").is_err());
    }

    #[test]
    fn test_eval_failure_classification() {
        let timeout = EvalFailure::classify("Execution timed out after 30 seconds");
        assert_eq!(timeout.error_type, "Timeout");
        assert!(timeout.retryable);

        let syntax = EvalFailure::classify("SyntaxError: invalid syntax (<string>, line 1)");
        assert_eq!(syntax.error_type, "SyntaxError");
        assert!(!syntax.retryable);
    }

    #[test]
    fn test_eval_output_bounding() {
        let executor = EvalExecutor;
        let gateway = DirectHostGateway::new();

        let huge = "print(1)\n".repeat(15_000);
        gateway.push_response(HostResponse::success(huge));

        let call = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "Eval",
            "1.0.0",
            json!({
                "code": "print(1)",
                "language": "py",
                "i": "Evaluating large output"
            }),
        );
        let res = executor.execute(&call, &gateway).unwrap();
        assert!(res.output.content.len() <= 100_000);
        assert!(res.output.truncated);
    }
}
