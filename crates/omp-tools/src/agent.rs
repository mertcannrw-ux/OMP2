use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::definition::{
    HostGateway, HostRequest, ToolCall, ToolDefinition,
    ToolDiagnostic, ToolError, ToolExecutionResult, ToolExecutor, ToolLimits,
};

/// Individual task specification for a delegated subagent.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SubagentTaskSpec {
    pub task: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_mode: Option<String>,
}

/// Executor for the permanent `Agent` tool.
#[derive(Default)]
pub struct AgentExecutor;

impl AgentExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl ToolExecutor for AgentExecutor {
    fn execute(
        &self,
        call: &ToolCall,
        gateway: &dyn HostGateway,
    ) -> Result<ToolExecutionResult, ToolError> {
        let context = call
            .input
            .get("context")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::Validation {
                message: "missing required 'context' parameter".into(),
                details: Some(call.input.clone()),
            })?;

        let tasks_val = call
            .input
            .get("tasks")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ToolError::Validation {
                message: "missing required 'tasks' array".into(),
                details: Some(call.input.clone()),
            })?;

        if tasks_val.is_empty() {
            return Err(ToolError::Validation {
                message: "tasks array cannot be empty".into(),
                details: Some(call.input.clone()),
            });
        }

        if tasks_val.len() > 32 {
            return Err(ToolError::LimitExceeded {
                limit: "concurrent subagents".into(),
                observed: tasks_val.len() as u64,
                max: 32,
            });
        }

        let mut tasks = Vec::new();
        for item in tasks_val {
            let task_text =
                item.get("task")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ToolError::Validation {
                        message: "each task item requires a 'task' string".into(),
                        details: Some(item.clone()),
                    })?;

            tasks.push(json!({
                "task": task_text,
                "name": item.get("name"),
                "agent": item.get("agent"),
                "tools": item.get("tools"),
                "outputSchema": item.get("outputSchema"),
                "schemaMode": item.get("schemaMode"),
            }));
        }

        let req = HostRequest::SpawnAgent {
            context: context.to_string(),
            tasks,
            isolated_workspace: true,
            convar_overrides: BTreeMap::new(),
        };

        let resp = gateway.request(req)?;
        let mut result = ToolExecutionResult::from_host(resp);

        result.diagnostics.push(ToolDiagnostic::info(format!(
            "Spawned {} subagent task(s) with isolated workspace",
            tasks_val.len()
        )));
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
                "Agent output exceeded limit of {} bytes and was truncated.",
                MAX_OUTPUT_BYTES
            )));
        }

        Ok(result)
    }
}

/// Creates the permanent `Agent` tool definition.
pub fn agent_tool_definition(executor: Arc<AgentExecutor>) -> ToolDefinition {
    ToolDefinition::new(
        "Agent",
        "1.0.0",
        json!({
            "type": "string",
            "description": "Present-participle intent describing batch subagent delegation"
        }),
        json!({
            "type": "object",
            "required": ["context", "tasks", "i"],
            "properties": {
                "context": {
                    "type": "string",
                    "description": "Shared project context, goals, and constraints"
                },
                "tasks": {
                    "type": "array",
                    "description": "Batch of subagents to spawn concurrently (max 32)",
                    "items": {
                        "type": "object",
                        "required": ["task"],
                        "properties": {
                            "task": { "type": "string" },
                            "name": { "type": "string" },
                            "agent": { "type": "string" },
                            "tools": { "type": "array", "items": { "type": "string" } },
                            "outputSchema": { "type": "object" },
                            "schemaMode": { "type": "string", "enum": ["permissive", "strict"] }
                        }
                    }
                },
                "i": {
                    "type": "string",
                    "description": "Present-participle intent"
                }
            }
        }),
        vec!["actor_spawn".into(), "workspace_isolate".into()],
        ToolLimits {
            max_output_bytes: 50_000,
            max_runtime_ms: 120_000,
            max_artifacts: 10,
            max_concurrent_jobs: 32,
        },
        executor,
        "agent",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::DirectHostGateway;

    #[test]
    fn test_agent_reject_empty_tasks() {
        let executor = AgentExecutor;
        let gateway = DirectHostGateway::new();
        let call = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "Agent",
            "1.0.0",
            json!({
                "context": "Context",
                "tasks": [],
                "i": "Delegating empty"
            }),
        );
        let err = executor.execute(&call, &gateway).unwrap_err();
        assert!(
            matches!(err, ToolError::Validation { message, .. } if message.contains("cannot be empty"))
        );
    }

    #[test]
    fn test_agent_reject_over_32_tasks() {
        let executor = AgentExecutor;
        let gateway = DirectHostGateway::new();
        let mut tasks = Vec::new();
        for i in 0..33 {
            tasks.push(json!({ "task": format!("task {i}") }));
        }
        let call = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "Agent",
            "1.0.0",
            json!({
                "context": "Context",
                "tasks": tasks,
                "i": "Delegating too many"
            }),
        );
        let err = executor.execute(&call, &gateway).unwrap_err();
        assert!(
            matches!(err, ToolError::LimitExceeded { limit, .. } if limit.contains("concurrent subagents"))
        );
    }
}
