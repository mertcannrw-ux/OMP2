use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

use crate::definition::{
    HostGateway, HostRequest, ToolCall, ToolDefinition,
    ToolDiagnostic, ToolError, ToolExecutionResult, ToolExecutor, ToolLimits,
};

/// Severity rating for an automated QA defect report.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QaSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl QaSeverity {
    pub fn parse(s: &str) -> Result<Self, ToolError> {
        match s.to_ascii_lowercase().as_str() {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "critical" => Ok(Self::Critical),
            other => Err(ToolError::Validation {
                message: format!(
                    "invalid severity '{}': must be low, medium, high, or critical",
                    other
                ),
                details: None,
            }),
        }
    }
}

/// Structured AutoQA defect report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoQaReport {
    pub tool: String,
    pub tool_version: Option<String>,
    pub input_summary: String,
    pub expected_behavior: String,
    pub observed_behavior: String,
    pub severity: QaSeverity,
    pub reproduction_data: Option<serde_json::Value>,
    pub session_id: Option<String>,
    pub journal_offset: Option<u64>,
    pub artifacts: Vec<String>,
}

/// Quality filter that validates defect reports before ingestion.
pub struct ReportQualityFilter;

impl ReportQualityFilter {
    pub fn filter(report: &AutoQaReport) -> Result<Option<ToolDiagnostic>, ToolError> {
        if report.expected_behavior.trim() == report.observed_behavior.trim() {
            return Err(ToolError::Validation {
                message: "expected_behavior and observed_behavior cannot be identical".into(),
                details: None,
            });
        }

        if report.input_summary.trim().len() < 5 {
            return Err(ToolError::Validation {
                message: "input_summary is too brief to be reproducible".into(),
                details: None,
            });
        }

        // Flag potential ungrounded blame
        if report
            .observed_behavior
            .to_lowercase()
            .contains("did not do what i wanted")
            && report.reproduction_data.is_none()
        {
            return Ok(Some(ToolDiagnostic::warning(
                "Report lacks concrete reproduction data; marked for review.",
            )));
        }

        Ok(None)
    }
}

/// Executor for the permanent `AutoQA` tool.
#[derive(Default)]
pub struct AutoQaExecutor;

impl AutoQaExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl ToolExecutor for AutoQaExecutor {
    fn execute(
        &self,
        call: &ToolCall,
        gateway: &dyn HostGateway,
    ) -> Result<ToolExecutionResult, ToolError> {
        let tool = call
            .input
            .get("tool")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::Validation {
                message: "missing required 'tool' parameter".into(),
                details: Some(call.input.clone()),
            })?;

        let input_summary = call
            .input
            .get("input_summary")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::Validation {
                message: "missing required 'input_summary' parameter".into(),
                details: Some(call.input.clone()),
            })?;

        let expected = call
            .input
            .get("expected_behavior")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::Validation {
                message: "missing required 'expected_behavior' parameter".into(),
                details: Some(call.input.clone()),
            })?;

        let observed = call
            .input
            .get("observed_behavior")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::Validation {
                message: "missing required 'observed_behavior' parameter".into(),
                details: Some(call.input.clone()),
            })?;

        let sev_str = call
            .input
            .get("severity")
            .and_then(|v| v.as_str())
            .unwrap_or("medium");
        let severity = QaSeverity::parse(sev_str)?;

        let report = AutoQaReport {
            tool: tool.to_string(),
            tool_version: call
                .input
                .get("tool_version")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            input_summary: input_summary.to_string(),
            expected_behavior: expected.to_string(),
            observed_behavior: observed.to_string(),
            severity,
            reproduction_data: call.input.get("reproduction_data").cloned(),
            session_id: call
                .input
                .get("session_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            journal_offset: call.input.get("journal_offset").and_then(|v| v.as_u64()),
            artifacts: call
                .input
                .get("artifacts")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
        };

        let filter_diag = ReportQualityFilter::filter(&report)?;

        let req = HostRequest::ReportQa {
            report: json!(report),
        };

        let resp = gateway.request(req)?;
        let mut result = ToolExecutionResult::from_host(resp);

        if let Some(diag) = filter_diag {
            result.diagnostics.push(diag);
        }

        result.diagnostics.push(ToolDiagnostic::info(format!(
            "Recorded AutoQA defect for tool '{}' (severity: {:?})",
            tool, severity
        )));
        // Strict bounded output enforcement (20,000 bytes)
        const MAX_OUTPUT_BYTES: usize = 20_000;
        if result.output.content.len() > MAX_OUTPUT_BYTES {
            let mut cut_idx = MAX_OUTPUT_BYTES;
            while cut_idx > 0 && !result.output.content.is_char_boundary(cut_idx) {
                cut_idx -= 1;
            }
            result.output.content.truncate(cut_idx);
            result.output.truncated = true;
            result.diagnostics.push(ToolDiagnostic::warning(format!(
                "AutoQA output exceeded limit of {} bytes and was truncated.",
                MAX_OUTPUT_BYTES
            )));
        }

        Ok(result)
    }
}

/// Creates the permanent `AutoQA` tool definition.
pub fn autoqa_tool_definition(executor: Arc<AutoQaExecutor>) -> ToolDefinition {
    ToolDefinition::new(
        "AutoQA",
        "1.0.0",
        json!({
            "type": "string",
            "description": "Present-participle intent describing automated defect reporting"
        }),
        json!({
            "type": "object",
            "required": ["tool", "input_summary", "expected_behavior", "observed_behavior", "severity", "i"],
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "Name of the tool that exhibited unexpected behavior"
                },
                "tool_version": {
                    "type": "string",
                    "description": "Optional version string"
                },
                "input_summary": {
                    "type": "string",
                    "description": "Concise summary of tool input that triggered defect"
                },
                "expected_behavior": {
                    "type": "string",
                    "description": "Specification of intended behavior"
                },
                "observed_behavior": {
                    "type": "string",
                    "description": "Actual failure observed"
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "Severity assessment"
                },
                "reproduction_data": {
                    "type": "object",
                    "description": "Structured reproduction payload"
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID where defect occurred"
                },
                "journal_offset": {
                    "type": "number",
                    "description": "Journal offset when defect occurred"
                },
                "artifacts": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Referenced artifact IDs"
                },
                "i": {
                    "type": "string",
                    "description": "Present-participle intent"
                }
            }
        }),
        vec!["qa_report".into()],
        ToolLimits {
            max_output_bytes: 20_000,
            max_runtime_ms: 10_000,
            max_artifacts: 5,
            max_concurrent_jobs: 2,
        },
        executor,
        "autoqa",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qa_severity_parse() {
        assert_eq!(QaSeverity::parse("low").unwrap(), QaSeverity::Low);
        assert_eq!(QaSeverity::parse("medium").unwrap(), QaSeverity::Medium);
        assert_eq!(QaSeverity::parse("high").unwrap(), QaSeverity::High);
        assert_eq!(QaSeverity::parse("critical").unwrap(), QaSeverity::Critical);
        assert!(QaSeverity::parse("catastrophic").is_err());
    }

    #[test]
    fn test_report_quality_filter_identical_behaviors() {
        let report = AutoQaReport {
            tool: "Read".into(),
            tool_version: None,
            input_summary: "Reading valid path".into(),
            expected_behavior: "File read successfully".into(),
            observed_behavior: "File read successfully".into(),
            severity: QaSeverity::Low,
            reproduction_data: None,
            session_id: None,
            journal_offset: None,
            artifacts: Vec::new(),
        };
        let err = ReportQualityFilter::filter(&report).unwrap_err();
        assert!(
            matches!(err, ToolError::Validation { message, .. } if message.contains("cannot be identical"))
        );
    }

    #[test]
    fn test_report_quality_filter_brief_summary() {
        let report = AutoQaReport {
            tool: "Read".into(),
            tool_version: None,
            input_summary: "abc".into(),
            expected_behavior: "Return 0".into(),
            observed_behavior: "Return 1".into(),
            severity: QaSeverity::Low,
            reproduction_data: None,
            session_id: None,
            journal_offset: None,
            artifacts: Vec::new(),
        };
        let err = ReportQualityFilter::filter(&report).unwrap_err();
        assert!(
            matches!(err, ToolError::Validation { message, .. } if message.contains("too brief"))
        );
    }
}
