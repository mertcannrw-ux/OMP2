use crate::request::ToolCallSpec;
use omp_types::{StructuredError, ToolCallId};
use serde::{Deserialize, Serialize};

/// Record of an automated repair applied to tool arguments.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairDiagnostic {
    pub parameter: String,
    pub original_value: String,
    pub repaired_value: String,
    pub reason: String,
}

/// Outcome of attempting to repair tool arguments against schema.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairOutcome {
    pub arguments: serde_json::Value,
    pub diagnostics: Vec<RepairDiagnostic>,
    pub modified: bool,
}

/// Validates and repairs model tool arguments against the tool parameters schema.
/// Unambiguous dialect errors are normalized automatically. Ambiguous errors return a retryable StructuredError.
pub fn repair_arguments(
    tool_name: &str,
    raw_args: &serde_json::Value,
    schema: &serde_json::Value,
) -> Result<RepairOutcome, StructuredError> {
    let mut diagnostics = Vec::new();

    // If raw_args is a string that might be stringified JSON, parse it first
    let mut current_args = if let serde_json::Value::String(s) = raw_args {
        match serde_json::from_str::<serde_json::Value>(s.trim()) {
            Ok(parsed) => {
                diagnostics.push(RepairDiagnostic {
                    parameter: "root".into(),
                    original_value: s.clone(),
                    repaired_value: parsed.to_string(),
                    reason: "parsed stringified JSON arguments into structured object".into(),
                });
                parsed
            }
            Err(_) => raw_args.clone(),
        }
    } else {
        raw_args.clone()
    };
    // Validate required parameters if specified in schema
    if let Some(required) = schema.get("required").and_then(|r| r.as_array())
        && let serde_json::Value::Object(args_map) = &current_args {
            for req_field in required {
                if let Some(field_name) = req_field.as_str()
                    && !args_map.contains_key(field_name) {
                        return Err(StructuredError::new(
                            "tool_argument_repair_failed",
                            format!(
                                "Missing required parameter '{field_name}' for tool '{tool_name}'"
                            ),
                            true,
                        ));
                    }
            }
        }

    // If the top-level schema is an object with properties
    if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
        if let serde_json::Value::Object(args_map) = &mut current_args {
            for (param_name, param_schema) in properties {
                if let Some(val) = args_map.get_mut(param_name)
                    && let Some(repaired) =
                        repair_single_value(param_name, val, param_schema, &mut diagnostics)?
                    {
                        *val = repaired;
                    }
            }
        } else {
            return Err(StructuredError::new(
                "tool_argument_repair_failed",
                format!(
                    "Tool '{tool_name}' expects an object map of arguments, got: {current_args}"
                ),
                true,
            ));
        }
    }

    let modified = !diagnostics.is_empty();
    Ok(RepairOutcome {
        arguments: current_args,
        diagnostics,
        modified,
    })
}

fn repair_single_value(
    param_name: &str,
    val: &serde_json::Value,
    param_schema: &serde_json::Value,
    diagnostics: &mut Vec<RepairDiagnostic>,
) -> Result<Option<serde_json::Value>, StructuredError> {
    let expected_type = param_schema
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("");

    match expected_type {
        "array" => match val {
            serde_json::Value::Array(_) => Ok(None),
            serde_json::Value::String(s) => {
                let trimmed = s.trim();
                // 1. Check if string is a stringified JSON array "[...]"
                if trimmed.starts_with('[') && trimmed.ends_with(']')
                    && let Ok(parsed) = serde_json::from_str::<Vec<serde_json::Value>>(trimmed) {
                        diagnostics.push(RepairDiagnostic {
                            parameter: param_name.into(),
                            original_value: s.clone(),
                            repaired_value: serde_json::to_string(&parsed).unwrap_or_default(),
                            reason: "parsed stringified JSON array into native array".into(),
                        });
                        return Ok(Some(serde_json::Value::Array(parsed)));
                    }
                // 2. Delimited string: comma-separated or newline-separated
                let items: Vec<serde_json::Value> = if trimmed.contains('\n') {
                    trimmed
                        .lines()
                        .map(|l| l.trim().trim_start_matches("- ").trim_start_matches("* "))
                        .filter(|l| !l.is_empty())
                        .map(|l| serde_json::Value::String(l.to_string()))
                        .collect()
                } else if trimmed.contains(',') {
                    trimmed
                        .split(',')
                        .map(|item| item.trim())
                        .filter(|item| !item.is_empty())
                        .map(|item| serde_json::Value::String(item.to_string()))
                        .collect()
                } else {
                    vec![serde_json::Value::String(trimmed.to_string())]
                };

                diagnostics.push(RepairDiagnostic {
                    parameter: param_name.into(),
                    original_value: s.clone(),
                    repaired_value: serde_json::to_string(&items).unwrap_or_default(),
                    reason: "parsed delimited string into array".into(),
                });
                Ok(Some(serde_json::Value::Array(items)))
            }
            serde_json::Value::Number(_)
            | serde_json::Value::Bool(_)
            | serde_json::Value::Object(_) => {
                // Scalar where array is expected: wrap in 1-element array
                let items = vec![val.clone()];
                diagnostics.push(RepairDiagnostic {
                    parameter: param_name.into(),
                    original_value: val.to_string(),
                    repaired_value: serde_json::to_string(&items).unwrap_or_default(),
                    reason: "wrapped single scalar in array".into(),
                });
                Ok(Some(serde_json::Value::Array(items)))
            }
            serde_json::Value::Null => Ok(None),
        },
        "object" => match val {
            serde_json::Value::Object(_) => Ok(None),
            serde_json::Value::String(s) => {
                let trimmed = s.trim();
                if trimmed.starts_with('{') && trimmed.ends_with('}')
                    && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) {
                        diagnostics.push(RepairDiagnostic {
                            parameter: param_name.into(),
                            original_value: s.clone(),
                            repaired_value: parsed.to_string(),
                            reason: "parsed stringified JSON object into native object".into(),
                        });
                        return Ok(Some(parsed));
                    }
                Err(StructuredError::new(
                    "tool_argument_repair_failed",
                    format!(
                        "Parameter '{param_name}' expects an object, got unparseable string: '{s}'"
                    ),
                    true,
                ))
            }
            _ => Err(StructuredError::new(
                "tool_argument_repair_failed",
                format!("Parameter '{param_name}' expects an object, got: {val}"),
                true,
            )),
        },
        "integer" => match val {
            serde_json::Value::Number(n) if n.is_i64() || n.is_u64() => Ok(None),
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    if f.fract() == 0.0 {
                        let int_val = f as i64;
                        diagnostics.push(RepairDiagnostic {
                            parameter: param_name.into(),
                            original_value: val.to_string(),
                            repaired_value: int_val.to_string(),
                            reason: "converted float with zero fraction to integer".into(),
                        });
                        Ok(Some(serde_json::json!(int_val)))
                    } else {
                        Err(StructuredError::new(
                            "tool_argument_repair_failed",
                            format!(
                                "Parameter '{param_name}' expects an integer, got ambiguous non-integer float: {f}"
                            ),
                            true,
                        ))
                    }
                } else {
                    Ok(None)
                }
            }
            serde_json::Value::String(s) => match s.trim().parse::<i64>() {
                Ok(int_val) => {
                    diagnostics.push(RepairDiagnostic {
                        parameter: param_name.into(),
                        original_value: s.clone(),
                        repaired_value: int_val.to_string(),
                        reason: "parsed string to integer".into(),
                    });
                    Ok(Some(serde_json::json!(int_val)))
                }
                Err(_) => Err(StructuredError::new(
                    "tool_argument_repair_failed",
                    format!("Parameter '{param_name}' expects an integer, got string: '{s}'"),
                    true,
                )),
            },
            _ => Err(StructuredError::new(
                "tool_argument_repair_failed",
                format!("Parameter '{param_name}' expects an integer, got: {val}"),
                true,
            )),
        },
        "number" => match val {
            serde_json::Value::Number(_) => Ok(None),
            serde_json::Value::String(s) => match s.trim().parse::<f64>() {
                Ok(num_val) => {
                    diagnostics.push(RepairDiagnostic {
                        parameter: param_name.into(),
                        original_value: s.clone(),
                        repaired_value: num_val.to_string(),
                        reason: "parsed string to float".into(),
                    });
                    Ok(Some(serde_json::json!(num_val)))
                }
                Err(_) => Err(StructuredError::new(
                    "tool_argument_repair_failed",
                    format!("Parameter '{param_name}' expects a number, got string: '{s}'"),
                    true,
                )),
            },
            _ => Err(StructuredError::new(
                "tool_argument_repair_failed",
                format!("Parameter '{param_name}' expects a number, got: {val}"),
                true,
            )),
        },
        "boolean" => match val {
            serde_json::Value::Bool(_) => Ok(None),
            serde_json::Value::String(s) => match s.trim().to_lowercase().as_str() {
                "true" | "1" | "yes" => {
                    diagnostics.push(RepairDiagnostic {
                        parameter: param_name.into(),
                        original_value: s.clone(),
                        repaired_value: "true".into(),
                        reason: "parsed string to boolean true".into(),
                    });
                    Ok(Some(serde_json::Value::Bool(true)))
                }
                "false" | "0" | "no" => {
                    diagnostics.push(RepairDiagnostic {
                        parameter: param_name.into(),
                        original_value: s.clone(),
                        repaired_value: "false".into(),
                        reason: "parsed string to boolean false".into(),
                    });
                    Ok(Some(serde_json::Value::Bool(false)))
                }
                _ => Err(StructuredError::new(
                    "tool_argument_repair_failed",
                    format!("Parameter '{param_name}' expects a boolean, got string: '{s}'"),
                    true,
                )),
            },
            serde_json::Value::Number(n) => {
                let b = n.as_i64().map(|i| i != 0).unwrap_or(false);
                diagnostics.push(RepairDiagnostic {
                    parameter: param_name.into(),
                    original_value: n.to_string(),
                    repaired_value: b.to_string(),
                    reason: "converted numeric flag to boolean".into(),
                });
                Ok(Some(serde_json::Value::Bool(b)))
            }
            _ => Err(StructuredError::new(
                "tool_argument_repair_failed",
                format!("Parameter '{param_name}' expects a boolean, got: {val}"),
                true,
            )),
        },
        "string" => match val {
            serde_json::Value::String(_) => Ok(None),
            serde_json::Value::Number(n) => {
                let s = n.to_string();
                diagnostics.push(RepairDiagnostic {
                    parameter: param_name.into(),
                    original_value: s.clone(),
                    repaired_value: s.clone(),
                    reason: "converted number to string".into(),
                });
                Ok(Some(serde_json::Value::String(s)))
            }
            serde_json::Value::Bool(b) => {
                let s = b.to_string();
                diagnostics.push(RepairDiagnostic {
                    parameter: param_name.into(),
                    original_value: s.clone(),
                    repaired_value: s.clone(),
                    reason: "converted boolean to string".into(),
                });
                Ok(Some(serde_json::Value::String(s)))
            }
            serde_json::Value::Array(arr) if arr.len() == 1 => {
                if let Some(first_str) = arr[0].as_str() {
                    diagnostics.push(RepairDiagnostic {
                        parameter: param_name.into(),
                        original_value: serde_json::to_string(arr).unwrap_or_default(),
                        repaired_value: first_str.to_string(),
                        reason: "unwrapped single string element from array".into(),
                    });
                    Ok(Some(serde_json::Value::String(first_str.to_string())))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        },
        _ => Ok(None),
    }
}

/// Synthesizes canonical tool call specifications from leaked text dialects.
///
/// Supported formats:
/// 1. `<tool_call>{"name": "...", "arguments": ...}</tool_call>`
/// 2. ````json\n{"name": "...", "parameters": ...}\n````
/// 3. `Action: <name>\nAction Input: <json>`
pub fn extract_leaked_tool_calls(text: &str) -> Vec<ToolCallSpec> {
    let mut calls = Vec::new();
    let mut call_counter = 0;

    // Pattern 1: <tool_call>...</tool_call>
    let mut cursor = 0;
    while let Some(start_idx) = text[cursor..].find("<tool_call>") {
        let abs_start = cursor + start_idx + "<tool_call>".len();
        if let Some(end_idx) = text[abs_start..].find("</tool_call>") {
            let json_str = text[abs_start..abs_start + end_idx].trim();
            let parsed_val = serde_json::from_str::<serde_json::Value>(json_str)
                .ok()
                .or_else(|| crate::corrective::repair_malformed_json(json_str, 5).ok());

            if let Some(val) = parsed_val
                && let Some(spec) = parse_json_tool_call(&val, &mut call_counter) {
                    calls.push(spec);
                }
            cursor = abs_start + end_idx + "</tool_call>".len();
        } else {
            break;
        }
    }

    // Pattern 2: Action: ... Action Input: ...
    cursor = 0;
    while let Some(action_idx) = text[cursor..].find("Action:") {
        let abs_action = cursor + action_idx + "Action:".len();
        let rest = &text[abs_action..];
        if let Some(input_idx) = rest.find("Action Input:") {
            let tool_name = rest[..input_idx].trim().to_string();
            let input_start = input_idx + "Action Input:".len();
            let input_str = rest[input_start..].trim();
            // Find end of json or next section
            let json_candidate = extract_first_json_block(input_str);
            let parsed_val = serde_json::from_str::<serde_json::Value>(&json_candidate)
                .ok()
                .or_else(|| crate::corrective::repair_malformed_json(&json_candidate, 5).ok());

            if let Some(val) = parsed_val {
                call_counter += 1;
                let id = ToolCallId::new(format!("call_leaked_{call_counter}"))
                    .unwrap_or_else(|_| ToolCallId::mint());
                calls.push(ToolCallSpec {
                    id,
                    name: tool_name,
                    arguments: val,
                });
            }
            cursor = abs_action + input_start + json_candidate.len();
        } else {
            cursor = abs_action;
        }
    }

    // Pattern 3: ```json codeblocks that look like tool calls
    cursor = 0;
    while let Some(block_start) = text[cursor..].find("```json") {
        let abs_start = cursor + block_start + "```json".len();
        if let Some(block_end) = text[abs_start..].find("```") {
            let json_str = text[abs_start..abs_start + block_end].trim();
            let parsed_val = serde_json::from_str::<serde_json::Value>(json_str)
                .ok()
                .or_else(|| crate::corrective::repair_malformed_json(json_str, 5).ok());

            if let Some(val) = parsed_val
                && let Some(spec) = parse_json_tool_call(&val, &mut call_counter) {
                    calls.push(spec);
                }
            cursor = abs_start + block_end + "```".len();
        } else {
            break;
        }
    }

    // Pattern 4: XML-like tags <tool_use name="...">...</tool_use> or <invoke name="...">...</invoke>
    for tag_name in &["tool_use", "invoke", "function_call"] {
        let open_pattern = format!("<{tag_name}");
        let close_pattern = format!("</{tag_name}>");
        let mut tag_cursor = 0;
        while let Some(start_idx) = text[tag_cursor..].find(&open_pattern) {
            let abs_start = tag_cursor + start_idx;
            if let Some(tag_end) = text[abs_start..].find('>') {
                let tag_header = &text[abs_start..abs_start + tag_end];
                let content_start = abs_start + tag_end + 1;
                if let Some(close_idx) = text[content_start..].find(&close_pattern) {
                    let body_str = text[content_start..content_start + close_idx].trim();
                    let tool_name = if let Some(name_idx) = tag_header.find("name=\"") {
                        let name_part = &tag_header[name_idx + 6..];
                        name_part.split('"').next().unwrap_or("").to_string()
                    } else {
                        String::new()
                    };

                    let parsed_val = serde_json::from_str::<serde_json::Value>(body_str)
                        .ok()
                        .or_else(|| crate::corrective::repair_malformed_json(body_str, 5).ok());

                    if let Some(val) = parsed_val {
                        if !tool_name.is_empty() {
                            call_counter += 1;
                            let id = ToolCallId::new(format!("call_leaked_{call_counter}"))
                                .unwrap_or_else(|_| ToolCallId::mint());
                            calls.push(ToolCallSpec {
                                id,
                                name: tool_name,
                                arguments: val,
                            });
                        } else if let Some(spec) = parse_json_tool_call(&val, &mut call_counter) {
                            calls.push(spec);
                        }
                    }
                    tag_cursor = content_start + close_idx + close_pattern.len();
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    calls
}

fn parse_json_tool_call(val: &serde_json::Value, counter: &mut usize) -> Option<ToolCallSpec> {
    let name = val
        .get("name")
        .or_else(|| val.get("tool"))
        .or_else(|| val.get("function"))
        .and_then(|v| v.as_str())?
        .to_string();

    let arguments = val
        .get("arguments")
        .or_else(|| val.get("parameters"))
        .or_else(|| val.get("input"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    *counter += 1;
    let id = ToolCallId::new(format!("call_synthesized_{counter}"))
        .unwrap_or_else(|_| ToolCallId::mint());

    Some(ToolCallSpec {
        id,
        name,
        arguments,
    })
}

fn extract_first_json_block(s: &str) -> String {
    let mut depth = 0;
    let mut in_string = false;
    let mut escaped = false;
    let mut started = false;
    let mut end = 0;

    for (idx, ch) in s.char_indices() {
        if !started {
            if ch == '{' || ch == '[' {
                started = true;
                depth = 1;
            }
            continue;
        }

        if escaped {
            escaped = false;
            continue;
        }

        if ch == '\\' {
            escaped = true;
            continue;
        }

        if ch == '"' {
            in_string = !in_string;
            continue;
        }

        if !in_string {
            if ch == '{' || ch == '[' {
                depth += 1;
            } else if ch == '}' || ch == ']' {
                depth -= 1;
                if depth == 0 {
                    end = idx + ch.len_utf8();
                    break;
                }
            }
        }
    }

    if started && end > 0 {
        s[..end].to_string()
    } else {
        s.lines().next().unwrap_or(s).trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repair_missing_required_field() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": { "type": "string" }
            }
        });
        let raw_args = serde_json::json!({});
        let err = repair_arguments("Read", &raw_args, &schema).unwrap_err();
        assert_eq!(err.code, "tool_argument_repair_failed");
        assert!(err.retryable);
    }

    #[test]
    fn test_repair_ambiguous_float_for_integer() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "line": { "type": "integer" }
            }
        });
        // Non-integer float 3.7 must be rejected as ambiguous
        let raw_args = serde_json::json!({ "line": 3.7 });
        let err = repair_arguments("Read", &raw_args, &schema).unwrap_err();
        assert_eq!(err.code, "tool_argument_repair_failed");

        // Integer float 3.0 must be accepted and converted to 3
        let raw_args_clean = serde_json::json!({ "line": 3.0 });
        let outcome = repair_arguments("Read", &raw_args_clean, &schema).unwrap();
        assert_eq!(outcome.arguments["line"], 3);
    }

    #[test]
    fn test_repair_delimited_string_to_array() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "tags": { "type": "array" }
            }
        });
        let raw_args = serde_json::json!({ "tags": "alpha, beta, gamma" });
        let outcome = repair_arguments("TagTool", &raw_args, &schema).unwrap();
        assert_eq!(
            outcome.arguments["tags"],
            serde_json::json!(["alpha", "beta", "gamma"])
        );
    }

    #[test]
    fn test_extract_leaked_calls_with_malformed_json() {
        // Leaked tool call with trailing comma in arguments
        let text = r#"Here is the call:
<tool_call>
{"name": "Read", "arguments": {"path": "src/main.rs",}}
</tool_call>"#;
        let calls = extract_leaked_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["path"], "src/main.rs");
    }

    #[test]
    fn test_extract_leaked_tool_use_xml() {
        let text = r#"<tool_use name="Write">{"path": "out.txt", "content": "hello"}</tool_use>"#;
        let calls = extract_leaked_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Write");
        assert_eq!(calls[0].arguments["path"], "out.txt");
    }
}
