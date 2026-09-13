use omp_inference::compaction::SpeculativeCompactionGuard;
use omp_inference::request::{
    InferenceRequest, SamplingParams, SemanticMessage, ThinkingMode,
    ToolCallSpec, ToolChoiceRequirement, ToolSchema,
};
use omp_state::{Journal, SessionSnapshot};
use omp_tools::definition::{
    ToolCall, ToolDiagnostic,
    ToolExecutionResult, ToolOutput,
};
use omp_types::{
    ActorId, ElementId, ElementSnapshot, JournalOffset, Patch, PatchOp, StructuredError, ToolCallId, TypedValue,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::agent_loop::DirectorStack;
use crate::command::{CommandEffect, CommandEngine};
use crate::convar::{ConVarStore, register_builtin_convars};
use crate::director::{
    AgentView, DirectorSpec, ForceTool, ForceToolFailureBehavior, ToolCallInfo, TurnView,
    YieldDecision, create_director_from_spec,
};

use crate::director::Director;
use omp_inference::provider::{CanonicalTurn, InferenceDelta, ProviderClient};
use omp_tools::host::ToolHost;

/// The authoritative host session managing turns, convars, commands, tools, and Directors.
pub struct SessionHost {
    pub workspace: PathBuf,
    pub owner: ActorId,
    pub convars: ConVarStore,
    pub command_engine: CommandEngine,
    pub director_stack: DirectorStack,
    pub provider: ProviderClient,
    pub tool_host: ToolHost,
    pub compaction_guard: SpeculativeCompactionGuard,
    pub max_turn_steps: usize,
    pub(crate) children: BTreeMap<ElementId, crate::children::ChildHandle>,
    pub(crate) cancellation: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub(crate) provider_injected: bool,
}

impl SessionHost {
    /// Initialize a new SessionHost for the specified workspace and owner actor.
    pub fn new(workspace: PathBuf, owner: ActorId) -> Result<Self, StructuredError> {
        let mut convars = ConVarStore::new();
        register_builtin_convars(&mut convars);

        let command_engine = CommandEngine::new();
        let director_stack = DirectorStack::new();

        let provider = ProviderClient::unconfigured();
        let tool_host = ToolHost::new(workspace.clone(), owner.clone()).map_err(|e| {
            StructuredError::new(
                "tool_host_init_failed",
                format!("Failed to init ToolHost: {}", e),
                false,
            )
        })?;

        Ok(Self {
            workspace,
            owner,
            convars,
            command_engine,
            director_stack,
            provider,
            tool_host,
            compaction_guard: SpeculativeCompactionGuard::new(),
            max_turn_steps: 20,
            children: BTreeMap::new(),
            cancellation: None,
            provider_injected: false,
        })
    }

    pub fn with_provider_endpoint(
        mut self,
        endpoint: impl Into<String>,
    ) -> Result<Self, StructuredError> {
        self.provider = self.provider.with_endpoint(endpoint)?;
        self.provider_injected = true;
        Ok(self)
    }

    pub fn with_provider_client(mut self, client: ProviderClient) -> Self {
        self.provider = client;
        self.provider_injected = true;
        self
    }

    pub fn with_max_turn_steps(mut self, steps: usize) -> Self {
        self.max_turn_steps = steps;
        self
    }

    /// Process a user turn through the complete host loop:
    /// 1. Hydrates ConVars and Directors from authoritative session DOM.
    /// 2. Appends user message to body via journal patch.
    /// 3. Executes bounded turn loop deriving InferenceRequests from the selected branch.
    /// 4. Executes returned tool calls and journals transitions before acknowledging.
    /// 5. Evaluates Director constraints only after tool calls settle.
    /// 6. Persists retry counters and Director state changes to DOM.
    pub fn run_turn(
        &mut self,
        journal: &mut Journal,
        message: &str,
    ) -> Result<(), StructuredError> {
        self.convars.hydrate_from_dom(journal.snapshot());
        self.refresh_provider()?;
        self.restore_provider_metadata(journal.snapshot())?;
        if !self.provider_injected
            && journal
                .snapshot()
                .element(journal.snapshot().container("capabilities"))
                .and_then(|node| node.attributes.get("provider_refresh_error"))
                .is_some_and(|value| matches!(value, TypedValue::Json(_)))
        {
            return Err(StructuredError::new(
                "provider_catalog_unavailable",
                "Provider refresh failed; use /provider refresh before inference",
                true,
            ));
        }
        if !self.provider.is_configured() {
            return Err(StructuredError::new(
                "provider_not_configured",
                "No inference model is available. Use /provider <endpoint> --key-env OMP_API_KEY to discover models.",
                false,
            ));
        }
        self.director_stack = DirectorStack::from_session_snapshot(journal.snapshot())?;

        // Step 2: Append user message element to <body>
        let body_container = journal.snapshot().container("body").clone();
        let body_count = journal.snapshot().children(&body_container).count() as u32;
        let user_elem_id = ElementId::mint();

        let mut user_attrs = BTreeMap::new();
        user_attrs.insert("role".into(), TypedValue::String("user".into()));

        let user_elem = ElementSnapshot {
            id: user_elem_id,
            schema_version: 1,
            kind: "user".into(),
            attributes: user_attrs,
            text: message.to_string(),
            payload: None,
        };

        let user_patch = Patch {
            base_offset: JournalOffset(journal.snapshot().offset),
            result_offset: journal.next_offset(),
            by: self.owner.clone().into(),
            reason: "user turn".into(),
            ops: vec![PatchOp::Create {
                parent: body_container.clone(),
                index: body_count,
                element: user_elem,
            }],
        };
        journal
            .append_patch(user_patch)
            .map_err(|e| e.structured())?;

        // Step 3: Bounded Host Loop
        let mut step = 0;
        let mut executed_tools_in_turn = Vec::new();

        while step < self.max_turn_steps {
            self.poll_children(journal)?;
            if self
                .cancellation
                .as_ref()
                .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
            {
                self.tool_host.shutdown_jobs(journal).map_err(|error| {
                    StructuredError::new("child_cancel", error.to_string(), false)
                })?;
                return Err(StructuredError::new(
                    "cancelled",
                    "Child host cancelled by parent",
                    false,
                ));
            }
            step += 1;

            // Derive base InferenceRequest from the selected branch
            let mut request = self.derive_inference_request(journal.snapshot())?;

            // Walk Directors outer-to-inner to prepare request
            request = self.director_stack.prepare_inference(request)?;

            if let ToolChoiceRequirement::Forced { tool } = &request.tool_choice {
                let specs = self.director_stack.to_specs();
                let state = specs.iter().rev().find(|spec| {
                    spec.kind == "ForceTool"
                        && spec.state.get("tool").and_then(|value| value.as_str()) == Some(tool)
                });
                let mut policy = omp_inference::tool_force::ToolForcePolicy::new(tool.clone());
                if let Some(spec) = state {
                    policy.current_attempt = spec
                        .state
                        .get("attempt_count")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0) as u32;
                    policy.max_attempts = spec
                        .state
                        .get("max_attempts")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(1) as u32;
                }
                policy.prepare_request(&mut request, &self.provider.caps)?;
            }

            let turn = self.infer_assistant_turn(journal, &request)?;

            // If model emitted tool calls, execute them and continue loop
            if !turn.tool_calls.is_empty() {
                for tc in &turn.tool_calls {
                    let call_info = ToolCallInfo {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    };

                    let call_id = self.execute_tool_inner(
                        journal,
                        &tc.name,
                        tc.arguments.clone(),
                        Some(tc.id.clone()),
                    )?;
                    let succeeded = journal.snapshot().get_visible_body().any(|element| {
                        element.attributes.get("call_id")
                            == Some(&TypedValue::String(call_id.to_string()))
                            && element.attributes.get("status")
                                == Some(&TypedValue::String("succeeded".into()))
                    });
                    if succeeded {
                        executed_tools_in_turn.push(call_info);
                    }
                }
                continue;
            }

            // No pending tool calls: all calls have settled. Candidate yield evaluation.
            let turn_view = TurnView {
                turn_index: journal.snapshot().turn_count() as u32,
                assistant_text: Some(turn.text.clone()),
                tool_calls: Vec::new(),
                executed_tools: executed_tools_in_turn.clone(),
                finish_reason: Some(turn.finish_reason.clone()),
            };

            let pending_todos: Vec<String> = journal
                .snapshot()
                .active_todos()
                .map(|e| {
                    if !e.text.is_empty() {
                        e.text.clone()
                    } else if let Some(TypedValue::String(t)) = e.attributes.get("title") {
                        t.clone()
                    } else {
                        e.id.to_string()
                    }
                })
                .collect();

            let agent_view = AgentView {
                actor_id: self.owner.clone(),
                session_id: journal.snapshot().session_id.clone(),
                model: if !self.provider.model.is_empty() {
                    self.provider.model.clone()
                } else {
                    self.convars
                        .get_typed::<String>("ai_model")
                        .unwrap_or_default()
                },
                active_tools: journal
                    .snapshot()
                    .active_tool_roster()
                    .map(|e| e.id.to_string())
                    .collect(),
                workspace_view: Some(self.workspace.to_string_lossy().into_owned()),
                pending_todos,
            };

            let decision = self.director_stack.handle_yield(&agent_view, &turn_view)?;

            // Persist Director state changes (attempt counts, stack mutations) to DOM
            self.sync_directors_to_journal(journal)?;

            match decision {
                YieldDecision::Yield | YieldDecision::Pass | YieldDecision::Done => {
                    return Ok(());
                }
                YieldDecision::Continue { prompt } => {
                    if let Some(prompt_text) = prompt {
                        self.append_steering_message(journal, &prompt_text)?;
                    }
                    continue;
                }
                YieldDecision::Push(spec) => {
                    self.push_director(journal, spec)?;
                    continue;
                }
                YieldDecision::Fail(err) => {
                    return Err(err);
                }
            }
        }

        Err(StructuredError::new(
            "turn_step_limit_exceeded",
            format!(
                "Agent turn exceeded maximum step budget ({})",
                self.max_turn_steps
            ),
            false,
        ))
    }

    /// Execute a tool call, journal input, invoke ToolHost, and journal result/diag/usage/status.
    pub fn execute_tool(
        &mut self,
        journal: &mut Journal,
        name: &str,
        input: serde_json::Value,
    ) -> Result<ToolCallId, StructuredError> {
        self.execute_tool_inner(journal, name, input, None)
    }

    fn execute_tool_inner(
        &mut self,
        journal: &mut Journal,
        name: &str,
        input: serde_json::Value,
        call_id_opt: Option<ToolCallId>,
    ) -> Result<ToolCallId, StructuredError> {
        let call_id = call_id_opt.unwrap_or_else(ToolCallId::mint);
        let call = ToolCall::new(call_id.clone(), name, "1.0.0", input);
        let element_id = ElementId::mint();
        let mut element = ElementSnapshot::new(element_id.clone(), "tool_call");
        element
            .attributes
            .insert("call_id".into(), TypedValue::String(call_id.to_string()));
        element
            .attributes
            .insert("tool".into(), TypedValue::String(name.into()));
        element
            .attributes
            .insert("version".into(), TypedValue::String(call.version.clone()));
        element
            .attributes
            .insert("status".into(), TypedValue::String("running".into()));
        element
            .attributes
            .insert("intent".into(), TypedValue::String(call.intent.clone()));
        let mut input_node = ElementSnapshot::new(ElementId::mint(), "input");
        input_node.payload = Some(call.input.clone());
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason: format!("invoke tool {name}"),
                ops: vec![
                    PatchOp::Create {
                        parent: journal.snapshot().container("body").clone(),
                        index: journal.snapshot().get_visible_body().count() as u32,
                        element,
                    },
                    PatchOp::Create {
                        parent: element_id.clone(),
                        index: 0,
                        element: input_node,
                    },
                ],
            })
            .map_err(|error| error.structured())?;

        let execution = if name == "Agent" {
            self.spawn_children(journal, &call.input)
        } else {
            self.tool_host
                .execute_tool(journal, &call)
                .map_err(|error| {
                    StructuredError::new("tool_execution_failed", error.to_string(), false)
                })
        };
        let result = match execution {
            Ok(result) => result,
            Err(error) => ToolExecutionResult {
                output: ToolOutput::text(error.to_string()),
                diagnostics: vec![ToolDiagnostic::error(error.to_string()).with_code(error.code)],
                usage: None,
                artifacts: Vec::new(),
            },
        };
        let failed = result.diagnostics.iter().any(|diagnostic| {
            diagnostic.severity == omp_tools::definition::DiagnosticSeverity::Error
        });
        let mut children = Vec::new();
        let mut output = ElementSnapshot::new(ElementId::mint(), "result");
        output.text = result.output.content;
        output.payload = result.output.payload;
        output.attributes.insert(
            "truncated".into(),
            TypedValue::Bool(result.output.truncated),
        );
        output.attributes.insert(
            "format".into(),
            TypedValue::Json(serde_json::to_value(result.output.format).map_err(|error| {
                StructuredError::new("tool_encoding", error.to_string(), false)
            })?),
        );
        children.push(output);
        for diagnostic in result.diagnostics {
            let mut node = ElementSnapshot::new(ElementId::mint(), "diag");
            node.text = diagnostic.message.clone();
            node.payload = Some(serde_json::to_value(diagnostic).map_err(|error| {
                StructuredError::new("tool_encoding", error.to_string(), false)
            })?);
            children.push(node);
        }
        if let Some(usage) = result.usage {
            let mut node = ElementSnapshot::new(ElementId::mint(), "usage");
            node.payload = Some(serde_json::to_value(usage).map_err(|error| {
                StructuredError::new("tool_encoding", error.to_string(), false)
            })?);
            children.push(node);
        }
        for artifact in result.artifacts {
            let mut node = ElementSnapshot::new(ElementId::mint(), "artifact_ref");
            node.text = format!("artifact://{artifact}");
            children.push(node);
        }
        let mut ops = vec![PatchOp::SetAttribute {
            element: element_id.clone(),
            name: "status".into(),
            value: TypedValue::String(if failed { "failed" } else { "succeeded" }.into()),
        }];
        ops.extend(
            children
                .into_iter()
                .enumerate()
                .map(|(index, element)| PatchOp::Create {
                    parent: element_id.clone(),
                    index: index as u32 + 1,
                    element,
                }),
        );
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason: format!("settle tool {name}"),
                ops,
            })
            .map_err(|error| error.structured())?;
        Ok(call_id)
    }

    /// Execute a command atomically using the journal-derived command stream.
    /// Supports convars, force, tool JSON, dyn discovery, and job cancellation.
    pub fn execute_command(
        &mut self,
        journal: &mut Journal,
        command_str: &str,
    ) -> Result<(), StructuredError> {
        // Hydrate current ConVars from DOM authority
        self.convars.hydrate_from_dom(journal.snapshot());
        self.command_engine.hydrate_from_dom(journal.snapshot());
        self.director_stack = DirectorStack::from_session_snapshot(journal.snapshot())?;

        let mut staged_convars = self.convars.clone();
        let mut staged_engine = self.command_engine.clone();
        staged_convars.begin_transaction();
        let effects = staged_engine
            .execute_loaded(command_str, &mut staged_convars, &mut |path| {
                let root = self
                    .workspace
                    .canonicalize()
                    .map_err(|e| crate::command::CommandError::Parse(e.to_string()))?;
                let path = root
                    .join(path)
                    .canonicalize()
                    .map_err(|e| crate::command::CommandError::Parse(e.to_string()))?;
                if !path.starts_with(&root) {
                    return Err(crate::command::CommandError::Parse(
                        "cfg is outside the workspace".into(),
                    ));
                }
                let file = std::fs::File::open(path)
                    .map_err(|e| crate::command::CommandError::Parse(e.to_string()))?;
                let mut source = String::new();
                std::io::Read::read_to_string(&mut std::io::Read::take(file, 65_537), &mut source)
                    .map_err(|e| crate::command::CommandError::Parse(e.to_string()))?;
                if source.len() > 65_536 {
                    return Err(crate::command::CommandError::ExpansionLimit {
                        bytes: source.len(),
                        limit: 65_536,
                    });
                }
                Ok(source)
            })
            .map_err(|e| e.to_structured_error())?;

        let external = effects
            .iter()
            .filter(|effect| {
                matches!(
                    effect,
                    CommandEffect::ToolExecuted { .. }
                        | CommandEffect::DynDiscovered { .. }
                        | CommandEffect::JobCancelled { .. }
                        | CommandEffect::Provider { .. }
                )
            })
            .count();
        if external > 0 && effects.len() != 1 {
            return Err(StructuredError::new(
                "non_atomic_command_stream",
                "Execute a tool or job control operation separately from configuration commands",
                false,
            ));
        }
        if let Some(CommandEffect::Provider { action }) = effects.first() {
            return self.execute_provider_command(journal, action.clone());
        }
        let provider_changed = effects.iter().any(|effect| matches!(effect,
            CommandEffect::ConVarUpdated { name, .. } if matches!(name.as_str(), "ai_provider" | "ai_endpoint" | "ai_api_key_env" | "ai_model")));
        if provider_changed {
            self.provider_from_config(&staged_convars)?;
        }
        let mut ops = Vec::new();
        let convars_container = journal.snapshot().container("convars").clone();
        let directors_container = journal.snapshot().container("directors").clone();
        let mut director_index = journal.snapshot().children(&directors_container).count() as u32;
        let mut body_index = journal
            .snapshot()
            .children(journal.snapshot().container("body"))
            .count() as u32;
        if let Some(patch) = crate::command::command_effects_to_patch(
            &effects,
            JournalOffset(journal.snapshot().offset),
            journal.next_offset(),
            self.owner.clone(),
            &convars_container,
        ) {
            ops.extend(patch.ops);
        }

        for effect in effects {
            match effect {
                CommandEffect::ConVarUpdated { .. } => {}
                CommandEffect::ForcePushed {
                    tool,
                    reminder,
                    max_attempts,
                } => {
                    let dir_id = omp_types::DirectorId::mint();
                    let force_dir =
                        ForceTool::new(dir_id.clone(), tool.clone(), reminder, max_attempts)
                            .with_failure_behavior(ForceToolFailureBehavior::Fail);

                    let spec = force_dir.to_spec();
                    let dir_elem_id = ElementId::new(dir_id.as_str()).expect("valid director id");
                    let mut attrs = BTreeMap::new();
                    attrs.insert("kind".into(), TypedValue::String("ForceTool".into()));

                    let elem = ElementSnapshot {
                        id: dir_elem_id,
                        schema_version: 1,
                        kind: "director".into(),
                        attributes: attrs,
                        text: String::new(),
                        payload: Some(spec.state),
                    };

                    let idx = director_index;
                    director_index += 1;
                    ops.push(PatchOp::Create {
                        parent: directors_container.clone(),
                        index: idx,
                        element: elem,
                    });
                }
                CommandEffect::ToolExecuted { name, input } => {
                    self.execute_tool(journal, &name, input)?;
                }
                CommandEffect::DynDiscovered { query } => {
                    let query_str = query.unwrap_or_default();
                    self.execute_tool(
                        journal,
                        "dyn",
                        serde_json::json!({ "query": query_str, "i": "Discovering dynamic tools" }),
                    )?;
                }
                CommandEffect::JobCancelled { job_id } => {
                    let child_id = ElementId::new(job_id.clone())?;
                    if let Some(child) = self.children.get(&child_id) {
                        child
                            .cancel
                            .store(true, std::sync::atomic::Ordering::Release);
                    } else {
                        self.tool_host
                            .cancel_job(journal, &job_id)
                            .map_err(|error| {
                                StructuredError::new("job_cancel_failed", error.to_string(), false)
                            })?;
                    }
                }
                CommandEffect::ConVarQueried { name, value } => {
                    let mut node = ElementSnapshot::new(ElementId::mint(), "system");
                    node.text = format!(
                        "{name} = {}",
                        serde_json::to_string(&value).unwrap_or_else(|_| "\"<unserializable>\"".into())
                    );
                    ops.push(PatchOp::Create {
                        parent: journal.snapshot().container("body").clone(),
                        index: body_index,
                        element: node,
                    });
                    body_index += 1;
                }
                CommandEffect::Output { message } => {
                    let mut node = ElementSnapshot::new(ElementId::mint(), "system");
                    node.text = message;
                    ops.push(PatchOp::Create {
                        parent: journal.snapshot().container("body").clone(),
                        index: body_index,
                        element: node,
                    });
                    body_index += 1;
                }
                CommandEffect::ExecRequested { path } => {
                    ops.push(PatchOp::SetAttribute {
                        element: convars_container.clone(),
                        name: "source".into(),
                        value: TypedValue::String(path),
                    });
                }
                _ => {}
            }
        }

        if !ops.is_empty() {
            // `Patch::validate` caps `reason` at 4096 bytes; a long script
            // must not execute its tools and then fail the trailing patch.
            let reason = truncate_to_boundary(format!("exec command: {command_str}"), 128);
            let patch = Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason,
                ops,
            };
            journal.append_patch(patch).map_err(|e| e.structured())?;
        }

        staged_convars.commit_transaction();
        self.convars = staged_convars;
        self.refresh_provider()?;
        self.command_engine = staged_engine;
        self.director_stack = DirectorStack::from_session_snapshot(journal.snapshot())?;
        if provider_changed {
            self.initialize_provider(journal)?;
        }
        Ok(())
    }

    /// Seed a child session host with complete inherited effective configuration.
    pub fn seed_child_host(
        &self,
        parent_snapshot: &SessionSnapshot,
        child_journal: &mut Journal,
        child_workspace: PathBuf,
        child_owner: ActorId,
        agent_name: Option<&str>,
    ) -> Result<SessionHost, StructuredError> {
        if child_journal.snapshot().offset != 0 {
            return Err(StructuredError::new(
                "child_already_initialized",
                "Child configuration can only be seeded into a fresh journal",
                false,
            ));
        }
        let mut inherited = self.convars.clone();
        inherited.hydrate_from_dom(parent_snapshot);
        let mut child = SessionHost::new(child_workspace, child_owner)?;
        child.convars = inherited.seed_child();
        let mut ops = child
            .convars
            .iter()
            .map(|(name, _, value)| PatchOp::SetAttribute {
                element: child_journal.snapshot().container("convars").clone(),
                name: name.clone(),
                value: value.clone(),
            })
            .collect::<Vec<_>>();
        for (name, value) in parent_snapshot.session_globals() {
            if name.starts_with("alias:") || name.starts_with("bind:") {
                ops.push(PatchOp::SetAttribute {
                    element: child_journal.snapshot().container("convars").clone(),
                    name: name.clone(),
                    value: value.clone(),
                });
            }
        }
        ops.push(PatchOp::SetAttribute {
            element: child_journal.snapshot().container("convars").clone(),
            name: "source".into(),
            value: TypedValue::String(format!(
                "parent:{}@{}",
                parent_snapshot.session_id, parent_snapshot.offset
            )),
        });
        child_journal
            .append_patch(Patch {
                base_offset: JournalOffset(0),
                result_offset: child_journal.next_offset(),
                by: self.owner.clone().into(),
                reason: "inherit parent effective configuration".into(),
                ops,
            })
            .map_err(|error| error.structured())?;
        let mut files = vec![PathBuf::from("configs/subagent.cfg")];
        if let Some(agent) = agent_name {
            if agent.is_empty()
                || !agent
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
            {
                return Err(StructuredError::new(
                    "invalid_agent_class",
                    "Agent class must be an identifier, not a path",
                    false,
                ));
            }
            files.push(PathBuf::from(format!("configs/{agent}.cfg")));
        }
        let script = files
            .into_iter()
            .filter(|path| child.workspace.join(path).is_file())
            .map(|path| {
                format!(
                    "exec {}",
                    serde_json::to_string(&path.to_string_lossy()).unwrap()
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        if !script.is_empty() {
            child.execute_command(child_journal, &script)?;
        }
        child.convars.hydrate_from_dom(child_journal.snapshot());
        child.refresh_provider()?;
        if !child.provider.is_configured() && self.provider.is_configured() {
            child.provider = self.provider.clone();
        }
        Ok(child)
    }

    fn derive_inference_request(
        &self,
        snapshot: &SessionSnapshot,
    ) -> Result<InferenceRequest, StructuredError> {
        let mut messages = Vec::new();

        for elem in snapshot.get_visible_body() {
            match elem.kind.as_str() {
                "user" => {
                    messages.push(SemanticMessage::user(&elem.text));
                }
                "assistant" => {
                    // Interrupted output remains inspectable, but is not a completed model turn.
                    if matches!(elem.attributes.get("status"), Some(TypedValue::String(status))
                        if matches!(status.as_str(), "running" | "failed" | "cancelled"))
                        || elem.attributes.get("streaming") == Some(&TypedValue::Bool(true))
                    {
                        continue;
                    }
                    let mut msg = SemanticMessage::assistant(&elem.text);
                    if let Some(TypedValue::String(call_id_str)) =
                        elem.attributes.get("tool_call_id")
                        && let Ok(cid) = ToolCallId::new(call_id_str) {
                            msg.tool_call_id = Some(cid);
                        }
                    messages.push(msg);
                }
                "tool_call" => {
                    if let Some(TypedValue::String(cid_str)) = elem.attributes.get("call_id")
                        && let Ok(cid) = ToolCallId::new(cid_str) {
                            let tool_name = elem
                                .attributes
                                .get("tool")
                                .and_then(|v| match v {
                                    TypedValue::String(s) => Some(s.clone()),
                                    _ => None,
                                })
                                .unwrap_or_else(|| "unknown".into());

                            let args = snapshot
                                .children(&elem.id)
                                .find(|child| child.kind == "input")
                                .and_then(|child| child.payload.clone())
                                .unwrap_or_default();
                            let mut msg = SemanticMessage::assistant("");
                            msg.tool_calls.push(ToolCallSpec {
                                id: cid,
                                name: tool_name,
                                arguments: args,
                            });
                            messages.push(msg);
                            if let Some(result) = snapshot
                                .children(&elem.id)
                                .find(|child| child.kind == "result")
                            {
                                messages.push(SemanticMessage::tool_result(
                                    ToolCallId::new(cid_str).map_err(|error| {
                                        StructuredError::new(
                                            "invalid_call_id",
                                            error.to_string(),
                                            false,
                                        )
                                    })?,
                                    &result.text,
                                ));
                            }
                        }
                }
                "steering" | "system" => messages.push(SemanticMessage::system(&elem.text)),
                _ => {}
            }
        }

        // The permanent grammar is host-defined; discovery stays behind Bash/Eval.
        let active_tools = self
            .tool_host
            .registry
            .permanent_tools()
            .map(|tool| ToolSchema {
                name: tool.name.clone(),
                description: format!("{}: {}", tool.name, tool.component_projection),
                parameters: tool.parameters.clone(),
            })
            .collect();

        let temp = self
            .convars
            .get_typed::<f64>("ai_temperature")
            .ok()
            .map(|v| v as f32);
        let max_tokens = self
            .convars
            .get_typed::<i64>("ai_max_tokens")
            .ok()
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value > 0);
        let thinking = self
            .convars
            .get_typed::<String>("ai_thinking")
            .unwrap_or_default();
        let desired_thinking = match thinking.as_str() {
            "" | "auto" | "unknown" => ThinkingMode::None,
            "off" | "none" | "0" => ThinkingMode::None,
            level => {
                if self
                    .provider
                    .active_model_metadata()
                    .and_then(|m| m.thinking_levels.as_ref())
                    .is_some_and(|levels| !levels.iter().any(|known| known == level))
                {
                    return Err(StructuredError::new(
                        "unsupported_thinking_level",
                        "Thinking level is not advertised for the selected model",
                        false,
                    ));
                }
                ThinkingMode::Effort {
                    level: level.to_string(),
                }
            }
        };

        Ok(InferenceRequest {
            messages,
            folds: Vec::new(),
            active_tools,
            desired_thinking,
            tool_choice: ToolChoiceRequirement::Auto,
            strict_schema: false,
            grammar: None,
            sampling: SamplingParams {
                temperature: temp,
                top_p: None,
                max_tokens,
                stop_sequences: Vec::new(),
            },
            usage_request: true,
            compaction_policy: Default::default(),
        })
    }

    fn infer_assistant_turn(
        &mut self,
        journal: &mut Journal,
        request: &InferenceRequest,
    ) -> Result<CanonicalTurn, StructuredError> {
        let id = ElementId::mint();
        let mut element = ElementSnapshot::new(id.clone(), "assistant");
        element
            .attributes
            .insert("role".into(), TypedValue::String("assistant".into()));
        element
            .attributes
            .insert("status".into(), TypedValue::String("running".into()));
        element
            .attributes
            .insert("streaming".into(), TypedValue::Bool(true));
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason: "start assistant stream".into(),
                ops: vec![PatchOp::Create {
                    parent: journal.snapshot().container("body").clone(),
                    index: journal.snapshot().get_visible_body().count() as u32,
                    element,
                }],
            })
            .map_err(|error| error.structured())?;

        let owner = self.owner.clone();
        let cancellation = self.cancellation.clone();
        let mut pending_text = String::new();
        let mut pending_thinking = String::new();
        let mut last_flush = Instant::now();
        let result = self.provider.infer(request, &mut |delta| {
            if cancellation
                .as_ref()
                .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
            {
                return Err(StructuredError::new(
                    "cancelled",
                    "Inference cancelled by parent",
                    false,
                ));
            }
            match delta {
                InferenceDelta::Text(text) => pending_text.push_str(text),
                InferenceDelta::Thinking(text) => pending_thinking.push_str(text),
            }
            // Journal deltas in bounded flushes: one fsync per token makes
            // streaming disk-bound, so buffers spill on a byte or time budget
            // (same shape as job output journaling).
            if pending_text.len() + pending_thinking.len() >= STREAM_FLUSH_BYTES
                || last_flush.elapsed() >= Duration::from_millis(STREAM_FLUSH_INTERVAL_MS)
            {
                flush_stream_buffers(
                    journal,
                    &owner,
                    &id,
                    &mut pending_text,
                    &mut pending_thinking,
                )?;
                last_flush = Instant::now();
            }
            Ok(())
        });
        flush_stream_buffers(journal, &owner, &id, &mut pending_text, &mut pending_thinking)?;

        let mut ops = vec![PatchOp::SetAttribute {
            element: id.clone(),
            name: "streaming".into(),
            value: TypedValue::Bool(false),
        }];
        match &result {
            Ok(turn) => {
                if journal
                    .snapshot()
                    .element(&id)
                    .is_some_and(|element| element.text != turn.text)
                {
                    // Normalization may revise provisional text; settle the same message, never duplicate it.
                    ops.push(PatchOp::ReplaceText {
                        element: id.clone(),
                        text: turn.text.clone(),
                    });
                }
                for (name, value) in [
                    ("status", TypedValue::String("succeeded".into())),
                    (
                        "finish_reason",
                        TypedValue::String(turn.finish_reason.clone()),
                    ),
                    ("usage", TypedValue::Json(turn.usage.clone())),
                ] {
                    ops.push(PatchOp::SetAttribute {
                        element: id.clone(),
                        name: name.into(),
                        value,
                    });
                }
                if let Some(thinking) = &turn.thinking {
                    ops.push(PatchOp::SetAttribute {
                        element: id.clone(),
                        name: "thinking".into(),
                        value: TypedValue::String(thinking.clone()),
                    });
                }
                ops.extend(
                    journal
                        .snapshot()
                        .children(&id)
                        .filter(|child| child.kind == "think")
                        .map(|child| PatchOp::Delete {
                            element: child.id.clone(),
                        }),
                );
            }
            Err(error) => {
                ops.push(PatchOp::SetAttribute {
                    element: id.clone(),
                    name: "status".into(),
                    value: TypedValue::String(
                        if error.code == "cancelled" {
                            "cancelled"
                        } else {
                            "failed"
                        }
                        .into(),
                    ),
                });
                let mut diagnostic = ElementSnapshot::new(ElementId::mint(), "diag");
                diagnostic.text = error.message.clone();
                diagnostic
                    .attributes
                    .insert("code".into(), TypedValue::String(error.code.clone()));
                diagnostic.payload = Some(serde_json::to_value(error).map_err(|error| {
                    StructuredError::new("error_encoding", error.to_string(), false)
                })?);
                ops.push(PatchOp::Create {
                    parent: id.clone(),
                    index: journal.snapshot().children(&id).count() as u32,
                    element: diagnostic,
                });
            }
        }
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason: "settle assistant stream".into(),
                ops,
            })
            .map_err(|error| error.structured())?;
        result
    }

    fn append_steering_message(
        &self,
        journal: &mut Journal,
        prompt: &str,
    ) -> Result<(), StructuredError> {
        let body_container = journal.snapshot().container("body").clone();
        let body_count = journal.snapshot().children(&body_container).count() as u32;
        let elem_id = ElementId::mint();

        let mut attrs = BTreeMap::new();
        attrs.insert("role".into(), TypedValue::String("system".into()));

        let elem = ElementSnapshot {
            id: elem_id,
            schema_version: 1,
            kind: "steering".into(),
            attributes: attrs,
            text: prompt.to_string(),
            payload: None,
        };

        let patch = Patch {
            base_offset: JournalOffset(journal.latest_offset()),
            result_offset: journal.next_offset(),
            by: self.owner.clone().into(),
            reason: "steering prompt".into(),
            ops: vec![PatchOp::Create {
                parent: body_container,
                index: body_count,
                element: elem,
            }],
        };
        journal.append_patch(patch).map_err(|e| e.structured())?;
        Ok(())
    }

    fn push_director(
        &mut self,
        journal: &mut Journal,
        spec: DirectorSpec,
    ) -> Result<(), StructuredError> {
        // Validate before journaling: a rejected spec must not leave a
        // director node that later breaks stack hydration on resume.
        let dir_instance = create_director_from_spec(&spec)?;
        let directors_container = journal.snapshot().container("directors").clone();
        let count = journal.snapshot().children(&directors_container).count() as u32;

        let dir_elem_id = match spec.element_id.clone() {
            Some(id) => id,
            None => ElementId::new(spec.id.as_str()).map_err(|e| {
                StructuredError::new(
                    "invalid_director_id",
                    format!("Journal-derived director id is not a valid element id: {e}"),
                    false,
                )
            })?,
        };

        let mut attrs = BTreeMap::new();
        attrs.insert("kind".into(), TypedValue::String(spec.kind.clone()));

        let elem = ElementSnapshot {
            id: dir_elem_id,
            schema_version: 1,
            kind: "director".into(),
            attributes: attrs,
            text: String::new(),
            payload: Some(spec.state.clone()),
        };

        let patch = Patch {
            base_offset: JournalOffset(journal.latest_offset()),
            result_offset: journal.next_offset(),
            by: self.owner.clone().into(),
            reason: format!("push director {}", spec.kind),
            ops: vec![PatchOp::Create {
                parent: directors_container,
                index: count,
                element: elem,
            }],
        };
        journal.append_patch(patch).map_err(|e| e.structured())?;

        self.director_stack.push(dir_instance);
        Ok(())
    }

    fn sync_directors_to_journal(&mut self, journal: &mut Journal) -> Result<(), StructuredError> {
        let current_specs = self.director_stack.to_specs();
        let _directors_container = journal.snapshot().container("directors").clone();

        // Check if any existing director in DOM needs an updated payload (e.g. attempt_count)
        let mut ops = Vec::new();
        for spec in &current_specs {
            let elem_id = match spec.element_id.clone() {
                Some(id) => id,
                None => match ElementId::new(spec.id.as_str()) {
                    Ok(id) => id,
                    Err(_) => continue,
                },
            };

            if let Some(existing) = journal.snapshot().element(&elem_id)
                && existing.payload.as_ref() != Some(&spec.state) {
                    ops.push(PatchOp::ReplacePayload {
                        element: elem_id,
                        payload: spec.state.clone(),
                    });
                }
        }

        for element in journal.snapshot().active_directors() {
            if !current_specs.iter().any(|spec| {
                spec.element_id.as_ref() == Some(&element.id)
                    || spec.id.as_str() == element.id.as_str()
            }) {
                ops.push(PatchOp::Delete {
                    element: element.id.clone(),
                });
            }
        }

        if !ops.is_empty() {
            let patch = Patch {
                base_offset: JournalOffset(journal.latest_offset()),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason: "sync director states".into(),
                ops,
            };
            journal.append_patch(patch).map_err(|e| e.structured())?;
        }

        Ok(())
    }
}

/// Streamed deltas spill to the journal once either budget is reached:
/// journaling one patch per token makes streaming disk-bound on Windows
/// (an fsync per append), so text and thinking accumulate instead.
const STREAM_FLUSH_INTERVAL_MS: u64 = 100;
const STREAM_FLUSH_BYTES: usize = 64 * 1024;

/// Appends buffered stream deltas as a single journal patch: text onto the
/// assistant element, thinking onto the existing `think` child (created on
/// first flush). Empty buffers are a no-op.
fn flush_stream_buffers(
    journal: &mut Journal,
    owner: &ActorId,
    element: &ElementId,
    pending_text: &mut String,
    pending_thinking: &mut String,
) -> Result<(), StructuredError> {
    if pending_text.is_empty() && pending_thinking.is_empty() {
        return Ok(());
    }
    let mut ops = Vec::new();
    if !pending_text.is_empty() {
        ops.push(PatchOp::AppendText {
            element: element.clone(),
            text: std::mem::take(pending_text),
        });
    }
    if !pending_thinking.is_empty() {
        match journal
            .snapshot()
            .children(element)
            .find(|child| child.kind == "think")
            .map(|child| child.id.clone())
        {
            Some(think_id) => ops.push(PatchOp::AppendText {
                element: think_id,
                text: std::mem::take(pending_thinking),
            }),
            None => {
                let mut child = ElementSnapshot::new(ElementId::mint(), "think");
                child.text = std::mem::take(pending_thinking);
                ops.push(PatchOp::Create {
                    parent: element.clone(),
                    index: 0,
                    element: child,
                });
            }
        }
    }
    journal
        .append_patch(Patch {
            base_offset: JournalOffset(journal.snapshot().offset),
            result_offset: journal.next_offset(),
            by: owner.clone().into(),
            reason: "assistant stream delta".into(),
            ops,
        })
        .map_err(|error| error.structured())
        .map(|_| ())
}

/// Truncate `text` to at most `max_bytes` on a UTF-8 boundary so patch
/// `reason` fields can never exceed the journal validation cap.
fn truncate_to_boundary(mut text: String, max_bytes: usize) -> String {
    if text.len() > max_bytes {
        let mut cut = max_bytes;
        while cut > 0 && !text.is_char_boundary(cut) {
            cut -= 1;
        }
        text.truncate(cut);
    }
    text
}
