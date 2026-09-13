use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

use crate::definition::{
    HostGateway, HostRequest, ToolCall, ToolDefinition,
    ToolDiagnostic, ToolError, ToolExecutionResult, ToolExecutor, ToolLimits, ToolOutput,
};

/// An operation in the hashline patch language.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EditOp {
    /// Replace inclusive line range `PUT N.=M:`
    PutRange {
        start: usize,
        end: usize,
        body: Vec<String>,
    },
    /// Replace syntactic block `PUT N*:`
    PutBlock { start: usize, body: Vec<String> },
    /// Insert before line `PUT <N:`
    InsertBefore { line: usize, body: Vec<String> },
    /// Insert after line `PUT >N:`
    InsertAfter { line: usize, body: Vec<String> },
    /// Paste named or anonymous register `PUT <N @reg`
    PasteBefore {
        line: usize,
        register: Option<String>,
    },
    /// Paste named or anonymous register `PUT >N @reg`
    PasteAfter {
        line: usize,
        register: Option<String>,
    },
    /// Cut inclusive lines `CUT N.=M`
    CutRange {
        start: usize,
        end: usize,
        register: Option<String>,
    },
    /// Cut syntactic block `CUT N*`
    CutBlock {
        start: usize,
        register: Option<String>,
    },
    /// Delete target file `REM`
    RemoveFile,
    /// Move/rename target file `MV DEST`
    MoveFile { destination: String },
}

/// A parsed section targeting one file.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParsedEditSection {
    pub path: String,
    pub tag: String,
    pub ops: Vec<EditOp>,
}

/// Structured diff summary produced from parsed edit operations.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct EditDiffSummary {
    pub files_modified: Vec<String>,
    pub lines_added: usize,
    pub lines_removed: usize,
}

/// Parser for the hashline patch language.
pub struct HashlineParser;

impl HashlineParser {
    pub fn parse(input: &str) -> Result<Vec<ParsedEditSection>, ToolError> {
        let mut sections = Vec::new();
        let mut current_section: Option<ParsedEditSection> = None;
        let mut current_op: Option<EditOp> = None;

        for line in input.lines() {
            let trimmed = line.trim();

            // Check for section header: [PATH#TAG]
            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                // Flush previous op
                if let Some(op) = current_op.take()
                    && let Some(sec) = current_section.as_mut() {
                        sec.ops.push(op);
                    }
                // Flush previous section
                if let Some(sec) = current_section.take() {
                    sections.push(sec);
                }

                let inner = &trimmed[1..trimmed.len() - 1];
                let (path, tag) = inner.split_once('#').ok_or_else(|| ToolError::Validation {
                    message: format!("invalid section header '{}': missing '#TAG'", trimmed),
                    details: None,
                })?;
                let clean_tag = tag.trim();
                if clean_tag.is_empty() {
                    return Err(ToolError::Validation {
                        message: format!("invalid section header '{}': empty tag", trimmed),
                        details: None,
                    });
                }

                current_section = Some(ParsedEditSection {
                    path: path.trim().to_string(),
                    tag: clean_tag.to_string(),
                    ops: Vec::new(),
                });
                continue;
            }

            // Body line starting with '+'
            if let Some(body_line) = line.strip_prefix('+') {
                match current_op.as_mut() {
                    Some(EditOp::PutRange { body, .. })
                    | Some(EditOp::PutBlock { body, .. })
                    | Some(EditOp::InsertBefore { body, .. })
                    | Some(EditOp::InsertAfter { body, .. }) => {
                        body.push(body_line.to_string());
                        continue;
                    }
                    _ => {
                        return Err(ToolError::Validation {
                            message: format!(
                                "unexpected body line without preceding PUT header: {}",
                                line
                            ),
                            details: None,
                        });
                    }
                }
            }

            // Flush previous op before parsing new operation header
            if let Some(op) = current_op.take()
                && let Some(sec) = current_section.as_mut() {
                    sec.ops.push(op);
                }

            if trimmed.is_empty() {
                continue;
            }

            if current_section.is_none() {
                return Err(ToolError::Validation {
                    message: format!("operation header without preceding [PATH#TAG]: {}", line),
                    details: None,
                });
            }

            // Parse operations
            if trimmed == "REM" {
                current_op = Some(EditOp::RemoveFile);
            } else if let Some(dest) = trimmed.strip_prefix("MV ") {
                current_op = Some(EditOp::MoveFile {
                    destination: dest.trim().to_string(),
                });
            } else if let Some(rest) = trimmed.strip_prefix("CUT ") {
                if let Some((s, e)) = rest.split_once(".=") {
                    let start = s.parse::<usize>().map_err(|_| ToolError::Validation {
                        message: format!("invalid CUT start line: {}", s),
                        details: None,
                    })?;
                    let end = e
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .parse::<usize>()
                        .map_err(|_| ToolError::Validation {
                            message: format!("invalid CUT end line: {}", e),
                            details: None,
                        })?;
                    if start == 0 || end == 0 {
                        return Err(ToolError::Validation {
                            message: "line numbers in CUT are 1-indexed, got 0".into(),
                            details: None,
                        });
                    }
                    if start > end {
                        return Err(ToolError::Validation {
                            message: format!(
                                "invalid CUT range: start line {start} is greater than end line {end}"
                            ),
                            details: None,
                        });
                    }
                    let reg = e
                        .split_whitespace()
                        .nth(1)
                        .map(|r| r.trim_start_matches('@').to_string());
                    current_op = Some(EditOp::CutRange {
                        start,
                        end,
                        register: reg,
                    });
                } else if let Some(s) = rest.strip_suffix('*') {
                    let start = s
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| ToolError::Validation {
                            message: format!("invalid CUT block line: {}", s),
                            details: None,
                        })?;
                    if start == 0 {
                        return Err(ToolError::Validation {
                            message: "line numbers in CUT block are 1-indexed, got 0".into(),
                            details: None,
                        });
                    }
                    current_op = Some(EditOp::CutBlock {
                        start,
                        register: None,
                    });
                }
            } else if let Some(rest) = trimmed.strip_prefix("PUT ") {
                let clean_rest = rest.trim_end_matches(':').trim();
                if let Some(line_str) = clean_rest.strip_prefix('<') {
                    let line = line_str
                        .parse::<usize>()
                        .map_err(|_| ToolError::Validation {
                            message: format!("invalid PUT insert-before line: {}", line_str),
                            details: None,
                        })?;
                    if line == 0 {
                        return Err(ToolError::Validation {
                            message:
                                "line numbers in PUT < are 1-indexed, got 0 (use <1 for file head)"
                                    .into(),
                            details: None,
                        });
                    }
                    current_op = Some(EditOp::InsertBefore {
                        line,
                        body: Vec::new(),
                    });
                } else if let Some(line_str) = clean_rest.strip_prefix('>') {
                    let line = line_str
                        .parse::<usize>()
                        .map_err(|_| ToolError::Validation {
                            message: format!("invalid PUT insert-after line: {}", line_str),
                            details: None,
                        })?;
                    if line == 0 {
                        return Err(ToolError::Validation {
                            message: "line numbers in PUT > are 1-indexed, got 0".into(),
                            details: None,
                        });
                    }
                    current_op = Some(EditOp::InsertAfter {
                        line,
                        body: Vec::new(),
                    });
                } else if let Some((s, e)) = clean_rest.split_once(".=") {
                    let start = s.parse::<usize>().map_err(|_| ToolError::Validation {
                        message: format!("invalid PUT start line: {}", s),
                        details: None,
                    })?;
                    let end = e.parse::<usize>().map_err(|_| ToolError::Validation {
                        message: format!("invalid PUT end line: {}", e),
                        details: None,
                    })?;
                    if start == 0 || end == 0 {
                        return Err(ToolError::Validation {
                            message: "line numbers in PUT are 1-indexed, got 0".into(),
                            details: None,
                        });
                    }
                    if start > end {
                        return Err(ToolError::Validation {
                            message: format!(
                                "invalid PUT range: start line {start} is greater than end line {end}"
                            ),
                            details: None,
                        });
                    }
                    current_op = Some(EditOp::PutRange {
                        start,
                        end,
                        body: Vec::new(),
                    });
                } else if let Some(s) = clean_rest.strip_suffix('*') {
                    let start = s
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| ToolError::Validation {
                            message: format!("invalid PUT block line: {}", s),
                            details: None,
                        })?;
                    if start == 0 {
                        return Err(ToolError::Validation {
                            message: "line numbers in PUT block are 1-indexed, got 0".into(),
                            details: None,
                        });
                    }
                    current_op = Some(EditOp::PutBlock {
                        start,
                        body: Vec::new(),
                    });
                }
            }
        }

        // Flush remaining op and section
        if let Some(op) = current_op.take()
            && let Some(sec) = current_section.as_mut() {
                sec.ops.push(op);
            }
        if let Some(sec) = current_section.take() {
            sections.push(sec);
        }

        if sections.is_empty() {
            return Err(ToolError::Validation {
                message: "no valid edit sections found in input".into(),
                details: None,
            });
        }

        Ok(sections)
    }

    pub fn summarize_diff(sections: &[ParsedEditSection]) -> EditDiffSummary {
        let mut summary = EditDiffSummary::default();
        for sec in sections {
            summary.files_modified.push(sec.path.clone());
            for op in &sec.ops {
                match op {
                    EditOp::PutRange { start, end, body } => {
                        summary.lines_removed += (end - start) + 1;
                        summary.lines_added += body.len();
                    }
                    EditOp::PutBlock { body, .. } => {
                        summary.lines_added += body.len();
                    }
                    EditOp::InsertBefore { body, .. } | EditOp::InsertAfter { body, .. } => {
                        summary.lines_added += body.len();
                    }
                    EditOp::CutRange { start, end, .. } => {
                        summary.lines_removed += (end - start) + 1;
                    }
                    EditOp::CutBlock { .. }
                    | EditOp::RemoveFile
                    | EditOp::MoveFile { .. }
                    | EditOp::PasteBefore { .. }
                    | EditOp::PasteAfter { .. } => {}
                }
            }
        }
        summary
    }
}

/// Executor for the permanent `Edit` tool.
#[derive(Default)]
pub struct EditExecutor;

impl EditExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl ToolExecutor for EditExecutor {
    fn execute(
        &self,
        call: &ToolCall,
        gateway: &dyn HostGateway,
    ) -> Result<ToolExecutionResult, ToolError> {
        let input_text = call
            .input
            .get("input")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::Validation {
                message: "missing required 'input' parameter".into(),
                details: Some(call.input.clone()),
            })?;

        // Parse hashline operations
        let sections = HashlineParser::parse(input_text)?;
        let summary = HashlineParser::summarize_diff(&sections);

        let mut all_diags = Vec::new();

        // Submit edit request to host gateway for each section
        for sec in &sections {
            let req = HostRequest::EditFile {
                path: sec.path.clone(),
                expected_tag: Some(sec.tag.clone()),
                patch_text: input_text.to_string(),
            };

            let resp = gateway.request(req)?;
            all_diags.extend(resp.diagnostics);
        }

        let has_conflict = all_diags.iter().any(|d| {
            d.severity == crate::definition::DiagnosticSeverity::Error
                || d.code.as_deref() == Some("conflict")
                || d.code.as_deref() == Some("stale_tag")
        });

        if has_conflict {
            all_diags.push(ToolDiagnostic::error(
                "Edit conflict detected: file has been modified or tag is stale. Re-read file to obtain latest [PATH#TAG].",
            ).with_code("conflict"));
        } else {
            all_diags.push(ToolDiagnostic::info(format!(
                "Edit applied: {} file(s) modified (+{} lines, -{} lines)",
                summary.files_modified.len(),
                summary.lines_added,
                summary.lines_removed,
            )));
        }

        let mut output = ToolOutput::json(json!({
            "summary": summary,
            "sections": sections,
            "conflict": has_conflict,
        }));

        // Strict bounded output enforcement (50,000 bytes)
        const MAX_OUTPUT_BYTES: usize = 50_000;
        if output.content.len() > MAX_OUTPUT_BYTES {
            let mut cut_idx = MAX_OUTPUT_BYTES;
            while cut_idx > 0 && !output.content.is_char_boundary(cut_idx) {
                cut_idx -= 1;
            }
            output.content.truncate(cut_idx);
            output.truncated = true;
            all_diags.push(ToolDiagnostic::warning(format!(
                "Edit output exceeded limit of {} bytes and was truncated.",
                MAX_OUTPUT_BYTES
            )));
        }

        Ok(ToolExecutionResult {
            output,
            diagnostics: all_diags,
            usage: None,
            artifacts: Vec::new(),
        })
    }
}

/// Creates the permanent `Edit` tool definition.
pub fn edit_tool_definition(executor: Arc<EditExecutor>) -> ToolDefinition {
    ToolDefinition::new(
        "Edit",
        "1.0.0",
        json!({
            "type": "string",
            "description": "Present-participle intent describing what surgical edit is being performed"
        }),
        json!({
            "type": "object",
            "required": ["input", "i"],
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Hashline patch language commands targeting [PATH#TAG]"
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
            max_runtime_ms: 20_000,
            max_artifacts: 5,
            max_concurrent_jobs: 4,
        },
        executor,
        "edit",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::HostResponse;
    use crate::definition::DirectHostGateway;

    #[test]
    fn test_reject_inverted_range_in_put() {
        let patch = "[file.rs#1A2B]\nPUT 10.=5:\n+hello\n";
        let err = HashlineParser::parse(patch).unwrap_err();
        assert!(
            matches!(err, ToolError::Validation { message, .. } if message.contains("greater than end line"))
        );
    }

    #[test]
    fn test_reject_zero_line_number() {
        let patch = "[file.rs#1A2B]\nPUT 0.=5:\n+hello\n";
        let err = HashlineParser::parse(patch).unwrap_err();
        assert!(
            matches!(err, ToolError::Validation { message, .. } if message.contains("1-indexed"))
        );
    }

    #[test]
    fn test_reject_empty_tag() {
        let patch = "[file.rs#]\nPUT 1.=2:\n+hello\n";
        let err = HashlineParser::parse(patch).unwrap_err();
        assert!(
            matches!(err, ToolError::Validation { message, .. } if message.contains("empty tag"))
        );
    }

    #[test]
    fn test_parse_and_summarize_diff() {
        let patch = r#"[src/main.rs#ABCD]
PUT 1.=3:
+fn main() {
+    println!("updated");
+}
PUT >5:
+// comment
"#;
        let sections = HashlineParser::parse(patch).unwrap();
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].path, "src/main.rs");
        assert_eq!(sections[0].tag, "ABCD");
        assert_eq!(sections[0].ops.len(), 2);

        let summary = HashlineParser::summarize_diff(&sections);
        assert_eq!(summary.files_modified, vec!["src/main.rs"]);
        assert_eq!(summary.lines_removed, 3);
        assert_eq!(summary.lines_added, 4); // 3 from PUT 1.=3 + 1 from PUT >5
    }

    #[test]
    fn test_edit_executor_conflict_diagnostic() {
        let executor = EditExecutor;
        let gateway = DirectHostGateway::new();

        // Simulate host response returning a conflict diagnostic
        let resp = HostResponse::success("edit failed").with_diagnostics(vec![
            ToolDiagnostic::error("snapshot tag mismatch: expected ABCD, observed EF01")
                .with_code("conflict"),
        ]);
        gateway.push_response(resp);

        let call = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "Edit",
            "1.0.0",
            json!({
                "input": "[src/main.rs#ABCD]\nPUT 1.=1:\n+new line\n",
                "i": "Updating main"
            }),
        );

        let res = executor.execute(&call, &gateway).unwrap();
        assert!(
            res.diagnostics
                .iter()
                .any(|d| d.code.as_deref() == Some("conflict"))
        );
        assert_eq!(res.output.payload.as_ref().unwrap()["conflict"], true);
    }
}
