use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::definition::{
    HostGateway, HostRequest, ToolCall, ToolDefinition, ToolError, ToolExecutionResult, ToolExecutor, ToolLimits,
};

/// A dynamic tool registered outside the permanent core roster.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicTool {
    pub namespace: String,
    pub action: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub returns: serde_json::Value,
}

impl DynamicTool {
    pub fn full_name(&self) -> String {
        format!("{}/{}", self.namespace, self.action)
    }

    /// Synthesizes human-readable manual / CLI help from parameter JSON Schema.
    pub fn synthesize_help(&self) -> String {
        let mut help = format!(
            "NAME\n    {} - {}\n\nSYNOPSIS\n    dyn {} [OPTIONS]\n\n",
            self.full_name(),
            self.description,
            self.full_name()
        );

        help.push_str("PARAMETERS\n");
        if let Some(props) = self
            .parameters
            .get("properties")
            .and_then(|v| v.as_object())
        {
            let required: Vec<&str> = self
                .parameters
                .get("required")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_str()).collect())
                .unwrap_or_default();

            for (name, spec) in props {
                let ptype = spec.get("type").and_then(|v| v.as_str()).unwrap_or("any");
                let desc = spec
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let req_marker = if required.contains(&name.as_str()) {
                    " (REQUIRED)"
                } else {
                    " (OPTIONAL)"
                };

                help.push_str(&format!(
                    "    --{} <{}>{}\n        {}\n\n",
                    name, ptype, req_marker, desc
                ));
            }
        } else {
            help.push_str("    No parameter schema declared.\n\n");
        }

        help
    }
}

/// Dynamic tool catalog that maintains dynamic tools separately from the permanent roster.
/// Dynamic discovery must not mutate the permanent roster or invalidate the model prefix cache.
#[derive(Default)]
pub struct DynCatalog {
    tools: RwLock<BTreeMap<String, DynamicTool>>,
}

impl DynCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, tool: DynamicTool) {
        self.tools.write().insert(tool.full_name(), tool);
    }

    pub fn get(&self, full_name: &str) -> Option<DynamicTool> {
        self.tools.read().get(full_name).cloned()
    }

    pub fn list(&self) -> Vec<DynamicTool> {
        self.tools.read().values().cloned().collect()
    }

    pub fn search(&self, query: &str) -> Vec<DynamicTool> {
        let q = query.to_lowercase();
        self.tools
            .read()
            .values()
            .filter(|t| {
                t.namespace.to_lowercase().contains(&q)
                    || t.action.to_lowercase().contains(&q)
                    || t.description.to_lowercase().contains(&q)
            })
            .cloned()
            .collect()
    }
}

/// Parsed dynamic CLI command.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DynCommand {
    ListAll,
    Search(String),
    Help(String),
    Execute {
        action: String,
        args: serde_json::Value,
    },
}

impl DynCommand {
    pub fn parse(args: &[String]) -> Self {
        if args.is_empty() {
            return Self::ListAll;
        }

        if args[0] == "--q" || args[0] == "-q" {
            return Self::Search(args.get(1).cloned().unwrap_or_default());
        }

        let action = &args[0];
        if args.iter().any(|a| a == "--help" || a == "-h") {
            return Self::Help(action.clone());
        }

        let mut map = serde_json::Map::new();
        let mut i = 1;
        while i < args.len() {
            if let Some(key) = args[i].strip_prefix("--") {
                let val = if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 1;
                    serde_json::Value::String(args[i].clone())
                } else {
                    serde_json::Value::Bool(true)
                };
                map.insert(key.to_string(), val);
            }
            i += 1;
        }

        Self::Execute {
            action: action.clone(),
            args: serde_json::Value::Object(map),
        }
    }
}

/// Executor for the `dyn` tool.
pub struct DynExecutor {
    pub catalog: Arc<DynCatalog>,
}

impl Default for DynExecutor {
    fn default() -> Self {
        Self {
            catalog: Arc::new(DynCatalog::new()),
        }
    }
}

impl DynExecutor {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ToolExecutor for DynExecutor {
    fn execute(
        &self,
        call: &ToolCall,
        gateway: &dyn HostGateway,
    ) -> Result<ToolExecutionResult, ToolError> {
        let response = gateway.request(HostRequest::DynLookup {
            query: call
                .input
                .get("query")
                .and_then(|value| value.as_str())
                .map(str::to_owned),
            action: call
                .input
                .get("action")
                .and_then(|value| value.as_str())
                .map(str::to_owned),
            help: call
                .input
                .get("help")
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
            args: call.input.get("args").cloned(),
        })?;
        Ok(ToolExecutionResult::from_host(response))
    }
}

/// Creates the `dyn` discovery tool definition.
pub fn dyn_tool_definition(executor: Arc<DynExecutor>) -> ToolDefinition {
    ToolDefinition::new(
        "dyn",
        "1.0.0",
        json!({
            "type": "string",
            "description": "Present-participle intent describing dynamic tool discovery"
        }),
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword search query across dynamic tools"
                },
                "action": {
                    "type": "string",
                    "description": "Dynamic action to inspect or execute (namespace/action)"
                },
                "help": {
                    "type": "boolean",
                    "description": "Display synthesized manual/help for action"
                },
                "args": {
                    "type": "object",
                    "description": "Arguments payload for dynamic tool execution"
                },
                "i": {
                    "type": "string",
                    "description": "Present-participle intent"
                }
            }
        }),
        vec!["dyn_discovery".into()],
        ToolLimits {
            max_output_bytes: 50_000,
            max_runtime_ms: 30_000,
            max_artifacts: 5,
            max_concurrent_jobs: 2,
        },
        executor,
        "dyn",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dyn_command_parse() {
        assert_eq!(DynCommand::parse(&[]), DynCommand::ListAll);
        assert_eq!(
            DynCommand::parse(&["--q".into(), "database".into()]),
            DynCommand::Search("database".into())
        );
        assert_eq!(
            DynCommand::parse(&["git/status".into(), "--help".into()]),
            DynCommand::Help("git/status".into())
        );
        assert_eq!(
            DynCommand::parse(&[
                "custom/action".into(),
                "--flag".into(),
                "--opt".into(),
                "val".into()
            ]),
            DynCommand::Execute {
                action: "custom/action".into(),
                args: serde_json::json!({
                    "flag": true,
                    "opt": "val",
                }),
            }
        );
    }
}
