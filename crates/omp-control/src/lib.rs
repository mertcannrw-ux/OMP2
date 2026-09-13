pub mod agent_loop;
mod children;
pub mod command;
pub mod convar;
pub mod director;
pub mod host;
mod provider_config;

pub use agent_loop::*;
pub use command::*;
pub use convar::*;
pub use director::*;
pub use host::*;
#[cfg(test)]
mod tests {
    use super::*;
    use omp_inference::compaction::SpeculativeCompactionGuard;
    use omp_types::{
        ActorId, BranchId, DirectorId, ElementId, JournalOffset, SessionId, ToolCallId, TypedValue,
    };

    #[test]
    fn test_convar_flags_and_tristate() {
        let flags = ConVarFlags::SESSION | ConVarFlags::REPLICATED | ConVarFlags::INHERIT;
        assert!(flags.contains(ConVarFlags::SESSION));
        assert!(flags.contains(ConVarFlags::REPLICATED));
        assert!(flags.contains(ConVarFlags::INHERIT));
        assert!(!flags.contains(ConVarFlags::READONLY));

        assert_eq!(TriState::Unknown.to_string(), "unknown");
        assert_eq!("true".parse::<TriState>().unwrap(), TriState::True);
        assert_eq!("0".parse::<TriState>().unwrap(), TriState::False);
        assert_eq!("unknown".parse::<TriState>().unwrap(), TriState::Unknown);
    }

    #[test]
    fn test_builtin_convars_and_store() {
        let mut store = ConVarStore::new();
        register_builtin_convars(&mut store);

        // Mutate convar and verify dirty session tracking
        store.set_from_str("ai_temperature", "0.2").unwrap();
        assert_eq!(store.get_typed::<f64>("ai_temperature").unwrap(), 0.2);

        let patch = store.drain_session_patch(
            omp_types::JournalOffset(0),
            omp_types::JournalOffset(1),
            ActorId::new("test-actor").unwrap(),
            &ElementId::new("meta-convars").unwrap(),
        );
        assert!(patch.is_some());
        assert_eq!(patch.unwrap().ops.len(), 1);

        // Second drain should be empty
        assert!(
            store
                .drain_session_patch(
                    omp_types::JournalOffset(1),
                    omp_types::JournalOffset(2),
                    ActorId::new("test-actor").unwrap(),
                    &ElementId::new("meta-convars").unwrap(),
                )
                .is_none()
        );
    }

    #[test]
    fn test_command_parser_deterministic() {
        let script = r#"
            // Set model and temperature
            ai_model "gpt-4o";
            ai_temperature 0.5;
            bind ctrl+t "toggle cl_showthinking";
            alias +think "cl_showthinking 1";
            # Shell style comment
            toggle sandbox_network;
        "#;

        let commands = CommandParser::parse_script(script).unwrap();
        assert_eq!(commands.len(), 5);
        assert_eq!(
            commands[0],
            Command::SetConVar {
                name: "ai_model".into(),
                value: "gpt-4o".into()
            }
        );
        assert_eq!(
            commands[1],
            Command::SetConVar {
                name: "ai_temperature".into(),
                value: "0.5".into()
            }
        );
        assert_eq!(
            commands[2],
            Command::Bind {
                key: "ctrl+t".into(),
                command: "toggle cl_showthinking".into()
            }
        );
        assert_eq!(
            commands[3],
            Command::Alias {
                name: "+think".into(),
                command: "cl_showthinking 1".into()
            }
        );
        assert_eq!(
            commands[4],
            Command::Toggle {
                name: "sandbox_network".into(),
                values: vec![]
            }
        );
    }

    #[test]
    fn test_command_engine_atomic_rollback() {
        let mut store = ConVarStore::new();
        register_builtin_convars(&mut store);
        let mut engine = CommandEngine::new();

        // One valid, one invalid command in the same script
        let script = r#"
            ai_temperature 0.1;
            non_existent_convar_xyz 123;
        "#;

        let result = engine.execute(script, &mut store);
        assert!(result.is_err());

        // State must NOT have partially mutated
        assert_eq!(store.get_typed::<f64>("ai_temperature").unwrap(), 0.7);
    }

    #[test]
    fn test_command_engine_alias_expansion_and_recursion_limit() {
        let mut store = ConVarStore::new();
        register_builtin_convars(&mut store);
        let mut engine = CommandEngine::new();

        // Valid alias
        engine
            .execute(r#"alias setlow "ai_temperature 0.1""#, &mut store)
            .unwrap();
        engine.execute("setlow", &mut store).unwrap();
        assert_eq!(store.get_typed::<f64>("ai_temperature").unwrap(), 0.1);

        // Recursive alias: a calls b, b calls a
        engine.execute(r#"alias a "b""#, &mut store).unwrap();
        engine.execute(r#"alias b "a""#, &mut store).unwrap();
        let rec_result = engine.execute("a", &mut store);
        assert!(rec_result.is_err());
    }

    #[test]
    fn test_director_stack_ordering_outer_inner() {
        struct TestDirector {
            id: DirectorId,
            name: String,
            tx: std::sync::mpsc::Sender<String>,
        }

        impl Director for TestDirector {
            fn id(&self) -> &DirectorId {
                &self.id
            }
            fn kind(&self) -> &str {
                &self.name
            }
            fn to_spec(&self) -> DirectorSpec {
                DirectorSpec {
                    id: self.id.clone(),
                    kind: self.name.clone(),
                    state: serde_json::Value::Null,
                    element_id: None,
                }
            }
            fn prepare_inference(
                &mut self,
                request: InferenceRequest,
            ) -> ControlResult<InferenceRequest> {
                let _ = self.tx.send(format!("prepare:{}", self.name));
                Ok(request)
            }
            fn on_yield(
                &mut self,
                _agent: &AgentView,
                _turn: &TurnView,
            ) -> ControlResult<YieldDecision> {
                let _ = self.tx.send(format!("yield:{}", self.name));
                Ok(YieldDecision::Pass)
            }
        }

        let (tx, rx) = std::sync::mpsc::channel();
        let mut stack = DirectorStack::new();

        stack.push(Box::new(TestDirector {
            id: DirectorId::new("d1-outer").unwrap(),
            name: "outer".into(),
            tx: tx.clone(),
        }));
        stack.push(Box::new(TestDirector {
            id: DirectorId::new("d2-inner").unwrap(),
            name: "inner".into(),
            tx,
        }));

        // prepare_inference must walk outer-to-inner: outer then inner
        let req = stack
            .prepare_inference(InferenceRequest::default())
            .unwrap();
        assert_eq!(req.messages.len(), 0);
        assert_eq!(rx.try_recv().unwrap(), "prepare:outer");
        assert_eq!(rx.try_recv().unwrap(), "prepare:inner");
        assert!(rx.try_recv().is_err());

        // on_yield must walk inner-to-outer: inner then outer
        let agent = AgentView {
            actor_id: ActorId::new("actor1").unwrap(),
            session_id: SessionId::new("sess1").unwrap(),
            model: "test".into(),
            active_tools: vec![],
            workspace_view: None,
            pending_todos: vec![],
        };
        let turn = TurnView::default();
        let decision = stack.handle_yield(&agent, &turn).unwrap();
        assert!(matches!(decision, YieldDecision::Yield));
        assert_eq!(rx.try_recv().unwrap(), "yield:inner");
        assert_eq!(rx.try_recv().unwrap(), "yield:outer");
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_forcetool_retries_and_bounded_exhaustion() {
        let mut force = ForceTool::new(
            DirectorId::new("ft1").unwrap(),
            "write",
            "Please call write",
            2,
        );

        let agent = AgentView {
            actor_id: ActorId::new("actor1").unwrap(),
            session_id: SessionId::new("sess1").unwrap(),
            model: "test".into(),
            active_tools: vec!["write".into()],
            workspace_view: None,
            pending_todos: vec![],
        };

        // Turn 1: Model did not call tool
        let turn_empty = TurnView {
            turn_index: 1,
            assistant_text: Some("I cannot write".into()),
            tool_calls: vec![],
            executed_tools: vec![],
            finish_reason: None,
        };

        let dec1 = force.on_yield(&agent, &turn_empty).unwrap();
        match dec1 {
            YieldDecision::Continue { .. } => {}
            other => panic!("expected Continue, got {:?}", other),
        }

        // Turn 2: Model still did not call tool -> exhaustion
        let dec2 = force.on_yield(&agent, &turn_empty).unwrap();
        match dec2 {
            YieldDecision::Fail(err) => {
                assert_eq!(err.code, "force_tool_exhausted");
                assert!(!err.retryable);
            }
            other => panic!("expected Fail, got {:?}", other),
        }

        // Test satisfaction: if tool was invoked, director returns Done
        let mut force_success = ForceTool::new(
            DirectorId::new("ft2").unwrap(),
            "write",
            "Please call write",
            3,
        );
        let turn_with_tool = TurnView {
            turn_index: 1,
            assistant_text: None,
            tool_calls: vec![ToolCallInfo {
                id: ToolCallId::new("call_1").unwrap(),
                name: "write".into(),
                arguments: serde_json::json!({ "path": "file.txt" }),
            }],
            executed_tools: vec![],
            finish_reason: None,
        };
        let dec_success = force_success.on_yield(&agent, &turn_with_tool).unwrap();
        assert!(matches!(dec_success, YieldDecision::Done));
    }

    #[test]
    fn test_agent_turn_evaluation_prioritizes_tool_calls() {
        let mut stack = DirectorStack::new();
        let agent = AgentView {
            actor_id: ActorId::new("actor1").unwrap(),
            session_id: SessionId::new("sess1").unwrap(),
            model: "test".into(),
            active_tools: vec!["read".into()],
            workspace_view: None,
            pending_todos: vec![],
        };

        // If tool calls exist, evaluate_agent_turn returns ExecuteTools without calling on_yield
        let turn_with_tools = TurnView {
            turn_index: 1,
            assistant_text: None,
            tool_calls: vec![ToolCallInfo {
                id: ToolCallId::new("call_r").unwrap(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": "test.txt" }),
            }],
            executed_tools: vec![],
            finish_reason: None,
        };

        let action = evaluate_agent_turn(&mut stack, &agent, &turn_with_tools).unwrap();
        match action {
            AgentTurnAction::ExecuteTools(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "read");
            }
            other => panic!("expected ExecuteTools, got {:?}", other),
        }

        // When no tool calls exist, evaluate_agent_turn yields to user
        let turn_no_tools = TurnView {
            turn_index: 2,
            assistant_text: Some("Done".into()),
            tool_calls: vec![],
            executed_tools: vec![],
            finish_reason: None,
        };
        let action2 = evaluate_agent_turn(&mut stack, &agent, &turn_no_tools).unwrap();
        assert!(matches!(action2, AgentTurnAction::YieldToUser));
    }

    #[test]
    fn test_transactional_rejection_and_change_hook_atomicity() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut store = ConVarStore::new();
        register_builtin_convars(&mut store);

        let hook_fired = Arc::new(AtomicUsize::new(0));
        let hook_fired_clone = hook_fired.clone();

        store.register(
            ConVar::new(
                "hooked_convar",
                "initial".to_string(),
                "ConVar with change hook",
                ConVarFlags::SESSION,
            )
            .with_change_hook(move |_old, _new| {
                hook_fired_clone.fetch_add(1, Ordering::SeqCst);
            }),
        );

        let mut engine = CommandEngine::new();

        // Script with valid set on hooked_convar, followed by invalid validator on ai_temperature (99.0 > 2.0)
        let failing_script = r#"
            hooked_convar "new_value";
            ai_temperature 99.0;
        "#;

        let res = engine.execute(failing_script, &mut store);
        assert!(res.is_err());

        // State must NOT have mutated
        assert_eq!(
            store.get_typed::<String>("hooked_convar").unwrap(),
            "initial"
        );
        assert_eq!(store.get_typed::<f64>("ai_temperature").unwrap(), 0.7);

        // Change hook must NOT have fired due to transactional rollback
        assert_eq!(hook_fired.load(Ordering::SeqCst), 0);

        // Successful script execution: change hook MUST fire
        let success_script = r#"
            hooked_convar "committed_value";
            ai_temperature 1.2;
        "#;
        let res_ok = engine.execute(success_script, &mut store);
        assert!(res_ok.is_ok());
        assert_eq!(
            store.get_typed::<String>("hooked_convar").unwrap(),
            "committed_value"
        );
        assert_eq!(store.get_typed::<f64>("ai_temperature").unwrap(), 1.2);
        assert_eq!(hook_fired.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_readonly_and_cheat_rejection() {
        let mut store = ConVarStore::new();
        register_builtin_convars(&mut store);

        // Register a READONLY convar
        store.register(ConVar::new(
            "sys_architecture",
            "x86_64".to_string(),
            "Hardware architecture",
            ConVarFlags::READONLY,
        ));

        // Register a CHEAT convar
        store.register(ConVar::new(
            "god_mode",
            false,
            "Invincibility cheat",
            ConVarFlags::CHEAT | ConVarFlags::SESSION,
        ));

        // Modifying READONLY convar must be rejected
        let err_ro = store.set_from_str("sys_architecture", "arm64").unwrap_err();
        assert!(matches!(err_ro, ConVarError::ReadOnly(_)));
        let err_reset_ro = store.reset("sys_architecture").unwrap_err();
        assert!(matches!(err_reset_ro, ConVarError::ReadOnly(_)));

        // Modifying CHEAT convar when cheats are disabled must be rejected
        assert!(!store.cheats_enabled());
        let err_cheat = store.set_from_str("god_mode", "1").unwrap_err();
        assert!(matches!(err_cheat, ConVarError::CheatProtected(_)));

        // Enable cheats via sv_cheats convar or set_cheats_enabled
        store.set_from_str("sv_cheats", "1").unwrap();
        assert!(store.cheats_enabled());

        // Now modifying CHEAT convar succeeds
        store.set_from_str("god_mode", "1").unwrap();
        assert!(store.get_typed::<bool>("god_mode").unwrap());
    }

    #[test]
    fn test_nested_director_precedence_and_short_circuit() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        struct SimpleDirector {
            id: DirectorId,
            kind: String,
            decision: YieldDecision,
            was_called: Arc<AtomicBool>,
        }

        impl Director for SimpleDirector {
            fn id(&self) -> &DirectorId {
                &self.id
            }
            fn kind(&self) -> &str {
                &self.kind
            }
            fn to_spec(&self) -> DirectorSpec {
                DirectorSpec {
                    id: self.id.clone(),
                    kind: self.kind.clone(),
                    state: serde_json::Value::Null,
                    element_id: None,
                }
            }
            fn prepare_inference(
                &mut self,
                request: InferenceRequest,
            ) -> ControlResult<InferenceRequest> {
                Ok(request)
            }
            fn on_yield(
                &mut self,
                _agent: &AgentView,
                _turn: &TurnView,
            ) -> ControlResult<YieldDecision> {
                self.was_called.store(true, Ordering::SeqCst);
                Ok(self.decision.clone())
            }
        }

        let outer_called = Arc::new(AtomicBool::new(false));
        let inner_called = Arc::new(AtomicBool::new(false));

        let mut stack = DirectorStack::new();
        stack.push(Box::new(SimpleDirector {
            id: DirectorId::new("outer").unwrap(),
            kind: "Outer".into(),
            decision: YieldDecision::Yield,
            was_called: outer_called.clone(),
        }));
        stack.push(Box::new(SimpleDirector {
            id: DirectorId::new("inner").unwrap(),
            kind: "Inner".into(),
            decision: YieldDecision::Continue {
                prompt: Some("inner needs another turn".into()),
            },
            was_called: inner_called.clone(),
        }));

        let agent = AgentView {
            actor_id: ActorId::new("actor1").unwrap(),
            session_id: SessionId::new("sess1").unwrap(),
            model: "test".into(),
            active_tools: vec![],
            workspace_view: None,
            pending_todos: vec![],
        };
        let turn = TurnView::default();

        let decision = stack.handle_yield(&agent, &turn).unwrap();
        // Inner decision must prevail
        match decision {
            YieldDecision::Continue { prompt } => {
                assert_eq!(prompt.unwrap(), "inner needs another turn");
            }
            other => panic!("expected Continue, got {:?}", other),
        }

        // Inner must have been evaluated
        assert!(inner_called.load(Ordering::SeqCst));
        // Outer must NOT have been called because inner made an active Continue decision!
        assert!(!outer_called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_director_stack_journal_spec_hydration() {
        let mut stack = DirectorStack::new();
        stack.push(Box::new(ForceTool::new(
            DirectorId::new("ft_test").unwrap(),
            "bash",
            "run command",
            3,
        )));
        stack.push(Box::new(TodoReminderDirector::new(
            DirectorId::new("todo_test").unwrap(),
            vec!["task1".into(), "task2".into()],
        )));

        let specs = stack.to_specs();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].kind, "ForceTool");
        assert_eq!(specs[1].kind, "TodoReminder");

        // Reconstruct from specs
        let hydrated = DirectorStack::from_specs(&specs).unwrap();
        assert_eq!(hydrated.len(), 2);
        let re_specs = hydrated.to_specs();
        assert_eq!(re_specs[0].id, specs[0].id);
        assert_eq!(re_specs[0].kind, specs[0].kind);
        assert_eq!(re_specs[1].id, specs[1].id);
        assert_eq!(re_specs[1].kind, specs[1].kind);
    }

    #[test]
    fn test_child_store_effective_preservation_and_isolation() {
        let mut parent = ConVarStore::new();
        register_builtin_convars(&mut parent);

        // Mutate parent
        parent.set_from_str("ai_temperature", "0.3").unwrap();
        parent
            .set_from_str("ai_model", "claude-3-5-sonnet")
            .unwrap();

        // Seed child
        let mut child = parent.seed_child();

        // Child must have inherited effective values
        assert_eq!(child.get_typed::<f64>("ai_temperature").unwrap(), 0.3);
        assert_eq!(
            child.get_typed::<String>("ai_model").unwrap(),
            "claude-3-5-sonnet"
        );

        // Apply subagent cfg override stream to child
        let mut engine = CommandEngine::new();
        let child_cfg = r#"
            ai_temperature 0.1;
            ai_model "gpt-4o";
        "#;
        child.apply_cfg_stream(&mut engine, child_cfg).unwrap();

        // Child reflects overrides
        assert_eq!(child.get_typed::<f64>("ai_temperature").unwrap(), 0.1);
        assert_eq!(child.get_typed::<String>("ai_model").unwrap(), "gpt-4o");

        // Parent remains untouched (isolation!)
        assert_eq!(parent.get_typed::<f64>("ai_temperature").unwrap(), 0.3);
        assert_eq!(
            parent.get_typed::<String>("ai_model").unwrap(),
            "claude-3-5-sonnet"
        );
    }

    #[test]
    fn test_tristate_no_coercion_and_toggle_cycle() {
        let mut store = ConVarStore::new();
        register_builtin_convars(&mut store);

        // Default is TriState::Unknown
        assert_eq!(
            store.get_typed::<TriState>("ai_thinking").unwrap(),
            TriState::Unknown
        );

        // Attempting to read Unknown as boolean must fail, NOT coerce to false
        assert!(store.get_typed::<bool>("ai_thinking").is_err());

        // Toggle ai_thinking: from Unknown it toggles to "1" (True)
        let mut engine = CommandEngine::new();
        engine.execute("toggle ai_thinking", &mut store).unwrap();
        assert_eq!(
            store.get_typed::<TriState>("ai_thinking").unwrap(),
            TriState::True
        );
        assert!(store.get_typed::<bool>("ai_thinking").unwrap());

        // Toggle again: from True it toggles to "0" (False)
        engine.execute("toggle ai_thinking", &mut store).unwrap();
        assert_eq!(
            store.get_typed::<TriState>("ai_thinking").unwrap(),
            TriState::False
        );
        assert!(!store.get_typed::<bool>("ai_thinking").unwrap());

        // Cycle toggle with explicit values: 0 -> 1 -> unknown
        engine
            .execute("toggle ai_thinking 0 1 unknown", &mut store)
            .unwrap();
        assert_eq!(
            store.get_typed::<TriState>("ai_thinking").unwrap(),
            TriState::True
        );

        engine
            .execute("toggle ai_thinking 0 1 unknown", &mut store)
            .unwrap();
        assert_eq!(
            store.get_typed::<TriState>("ai_thinking").unwrap(),
            TriState::Unknown
        );
    }

    #[test]
    fn test_session_host_creation_and_command_execution() {
        let temp_dir = std::env::temp_dir().join(format!("omp_test_host_{}", SessionId::mint()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let journal_path = temp_dir.join("test.journal");

        let session_id = SessionId::new("sess_host_test").unwrap();
        let mut journal = omp_state::Journal::create(&journal_path, session_id).unwrap();

        let actor_id = ActorId::new("test_user").unwrap();
        let mut host = SessionHost::new(temp_dir.clone(), actor_id).unwrap();

        // Execute ConVar command
        host.execute_command(&mut journal, "ai_temperature 0.42")
            .unwrap();
        assert_eq!(
            journal
                .snapshot()
                .session_globals()
                .get("ai_temperature")
                .unwrap(),
            &TypedValue::Number(0.42)
        );

        // Execute Force command
        host.execute_command(&mut journal, "force write \"Write report\" 3")
            .unwrap();
        assert_eq!(journal.snapshot().active_directors().count(), 1);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_forcetool_satisfaction_observes_executed_tools() {
        let mut force = ForceTool::new(
            DirectorId::new("ft_exec").unwrap(),
            "write",
            "Must call write",
            2,
        );

        let agent = AgentView {
            actor_id: ActorId::new("actor1").unwrap(),
            session_id: SessionId::new("sess1").unwrap(),
            model: "test".into(),
            active_tools: vec!["write".into()],
            workspace_view: None,
            pending_todos: vec![],
        };

        // When candidate turn has empty pending tool_calls, but executed_tools has "write":
        let turn = TurnView {
            turn_index: 1,
            assistant_text: Some("File written".into()),
            tool_calls: vec![],
            executed_tools: vec![ToolCallInfo {
                id: ToolCallId::new("call_w1").unwrap(),
                name: "write".into(),
                arguments: serde_json::json!({ "path": "output.txt" }),
            }],
            finish_reason: Some("stop".into()),
        };

        // ForceTool must observe executed_tools and return Done!
        let decision = force.on_yield(&agent, &turn).unwrap();
        assert!(matches!(decision, YieldDecision::Done));
    }

    #[test]
    fn test_plan_mode_exact_plan_path_inspection() {
        let mut plan_mode =
            PlanModeDirector::new(DirectorId::new("pm1").unwrap(), "local://plan.md");

        let agent = AgentView {
            actor_id: ActorId::new("actor1").unwrap(),
            session_id: SessionId::new("sess1").unwrap(),
            model: "test".into(),
            active_tools: vec!["read".into()],
            workspace_view: None,
            pending_todos: vec![],
        };

        // Reading a DIFFERENT file must NOT satisfy plan_file_inspected
        let turn_wrong_file = TurnView {
            turn_index: 1,
            assistant_text: Some("Here is the plan: step 1".into()),
            tool_calls: vec![],
            executed_tools: vec![ToolCallInfo {
                id: ToolCallId::new("call_r1").unwrap(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": "other_file.txt" }),
            }],
            finish_reason: Some("stop".into()),
        };

        let decision = plan_mode.on_yield(&agent, &turn_wrong_file).unwrap();
        assert!(matches!(decision, YieldDecision::Continue { .. }));

        // Reading the EXACT plan file DOES satisfy plan_file_inspected
        let turn_exact_file = TurnView {
            turn_index: 2,
            assistant_text: Some("Here is the proposed plan for approval".into()),
            tool_calls: vec![],
            executed_tools: vec![ToolCallInfo {
                id: ToolCallId::new("call_r2").unwrap(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": "local://plan.md" }),
            }],
            finish_reason: Some("stop".into()),
        };

        let decision2 = plan_mode.on_yield(&agent, &turn_exact_file).unwrap();
        assert!(matches!(decision2, YieldDecision::Pass));
    }

    #[test]
    fn test_child_host_seed_and_speculative_compaction_guard() {
        let temp_dir = std::env::temp_dir().join(format!("omp_test_child_{}", SessionId::mint()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let child_dir = temp_dir.join("child");
        let _ = std::fs::create_dir_all(&child_dir);

        let actor_id = ActorId::new("parent_actor").unwrap();
        let mut parent_host = SessionHost::new(temp_dir.clone(), actor_id).unwrap();
        let mut parent_journal =
            omp_state::Journal::create(temp_dir.join("parent.journal"), SessionId::mint()).unwrap();
        parent_host
            .execute_command(&mut parent_journal, "ai_temperature 0.2")
            .unwrap();

        let child_actor = ActorId::new("child_actor").unwrap();
        let mut child_journal =
            omp_state::Journal::create(temp_dir.join("child.journal"), SessionId::mint()).unwrap();
        let child_host = parent_host
            .seed_child_host(
                parent_journal.snapshot(),
                &mut child_journal,
                child_dir.clone(),
                child_actor,
                None,
            )
            .unwrap();

        assert_eq!(
            child_host
                .convars
                .get_typed::<f64>("ai_temperature")
                .unwrap(),
            0.2
        );

        // Test SpeculativeCompactionGuard
        let guard = SpeculativeCompactionGuard::new();
        let branch = BranchId::new("main").unwrap();
        let snap = guard.create_snapshot(branch.clone(), JournalOffset(10), 8000, 10000);

        // Only the unchanged source snapshot may accept a speculative fold.
        let fold = omp_inference::request::MessageFold::Shake {
            kept_prefix_count: 2,
            dropped_count: 5,
        };
        let res = guard.validate_and_splice(&snap, &branch, &JournalOffset(10), fold.clone());
        assert!(res.is_ok());

        // Diverged branch: rejected
        let other_branch = BranchId::new("fork_1").unwrap();
        let res_diverged =
            guard.validate_and_splice(&snap, &other_branch, &JournalOffset(15), fold.clone());
        assert!(res_diverged.is_err());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
