use std::collections::BTreeMap;
use std::sync::Arc;

use crate::agent::{AgentExecutor, agent_tool_definition};
use crate::autoqa::{AutoQaExecutor, autoqa_tool_definition};
use crate::bash::{BashExecutor, bash_tool_definition};
use crate::definition::ToolDefinition;
use crate::dyn_discovery::{DynCatalog, DynExecutor, DynamicTool, dyn_tool_definition};
use crate::edit::{EditExecutor, edit_tool_definition};
use crate::eval::{EvalExecutor, eval_tool_definition};
use crate::read::{ReadExecutor, read_tool_definition};
use crate::write::{WriteExecutor, write_tool_definition};

/// Exact permanent roster names defined by the playbook.
pub const PERMANENT_ROSTER: &[&str] = &["Read", "Bash", "Write", "Edit", "Eval", "Agent", "AutoQA"];

/// Constructs the permanent roster containing the 7 core tools.
pub fn create_permanent_roster() -> Vec<ToolDefinition> {
    vec![
        read_tool_definition(Arc::new(ReadExecutor::default())),
        bash_tool_definition(Arc::new(BashExecutor)),
        write_tool_definition(Arc::new(WriteExecutor)),
        edit_tool_definition(Arc::new(EditExecutor)),
        eval_tool_definition(Arc::new(EvalExecutor)),
        agent_tool_definition(Arc::new(AgentExecutor)),
        autoqa_tool_definition(Arc::new(AutoQaExecutor)),
    ]
}

/// Central registry managing permanent core tools and dynamic discovery.
pub struct ToolRegistry {
    permanent: BTreeMap<String, ToolDefinition>,
    dyn_catalog: Arc<DynCatalog>,
    dyn_tool: ToolDefinition,
    dynamic_handlers: parking_lot::RwLock<BTreeMap<String, ToolDefinition>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        let mut permanent = BTreeMap::new();
        for tool in create_permanent_roster() {
            permanent.insert(tool.name.clone(), tool);
        }

        let dyn_catalog = Arc::new(DynCatalog::new());
        let dyn_executor = Arc::new(DynExecutor {
            catalog: dyn_catalog.clone(),
        });
        let dyn_tool = dyn_tool_definition(dyn_executor);

        Self {
            permanent,
            dyn_catalog,
            dyn_tool,
            dynamic_handlers: Default::default(),
        }
    }

    /// Checks whether a tool name belongs to the permanent core roster.
    pub fn is_permanent(&self, name: &str) -> bool {
        self.permanent.contains_key(name)
    }

    /// Retrieves a permanent tool definition by name.
    pub fn get_permanent(&self, name: &str) -> Option<&ToolDefinition> {
        self.permanent.get(name)
    }

    /// Returns an iterator over all permanent core tools.
    pub fn permanent_tools(&self) -> impl Iterator<Item = &ToolDefinition> {
        self.permanent.values()
    }

    /// Returns the dynamic discovery tool definition (`dyn`).
    pub fn dyn_tool(&self) -> &ToolDefinition {
        &self.dyn_tool
    }

    /// Dynamic catalog accessor.
    pub fn dynamic_catalog(&self) -> Arc<DynCatalog> {
        self.dyn_catalog.clone()
    }

    /// Total count of permanent core tools.
    pub fn permanent_roster_count(&self) -> usize {
        self.permanent.len()
    }

    /// Registers a dynamic tool without modifying the permanent roster.
    pub fn register_dynamic(&self, definition: ToolDefinition) {
        let (namespace, action) = definition
            .name
            .split_once('/')
            .unwrap_or(("user", &definition.name));
        self.dyn_catalog.register(DynamicTool {
            namespace: namespace.into(),
            action: action.into(),
            description: definition.component_projection.clone(),
            parameters: definition.parameters.clone(),
            returns: serde_json::json!({}),
        });
        self.dynamic_handlers
            .write()
            .insert(definition.name.clone(), definition);
    }

    pub fn dynamic_definition(&self, name: &str) -> Option<ToolDefinition> {
        self.dynamic_handlers.read().get(name).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permanent_roster_exact_seven() {
        assert_eq!(PERMANENT_ROSTER.len(), 7);
        let expected = ["Read", "Bash", "Write", "Edit", "Eval", "Agent", "AutoQA"];
        for name in &expected {
            assert!(PERMANENT_ROSTER.contains(name));
        }

        let registry = ToolRegistry::new();
        assert_eq!(registry.permanent_roster_count(), 7);
        for name in &expected {
            assert!(registry.is_permanent(name));
            assert!(registry.get_permanent(name).is_some());
        }
    }
}
