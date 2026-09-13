use omp_control::{SessionHost, TriState};
use omp_inference::{
    capability::{NativeToolCost, ProviderCapability, TriState as Capability},
    compaction::SpeculativeCompactionGuard,
    provider::ProviderClient,
    request::MessageFold,
};
use omp_state::Journal;
use omp_types::{ActorId, BranchId, JournalOffset, SessionId, TypedValue};
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::PathBuf,
    thread,
    time::Duration,
};

struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("omp-control-{}", SessionId::mint()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn host(&self) -> SessionHost {
        SessionHost::new(self.0.clone(), ActorId::new("owner").unwrap()).unwrap()
    }
    fn journal(&self) -> Journal {
        Journal::create(self.0.join("session.journal"), SessionId::mint()).unwrap()
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn provider(responses: Vec<Value>) -> (ProviderClient, thread::JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!(
        "http://{}/v1/chat/completions",
        listener.local_addr().unwrap()
    );
    let handle = thread::spawn(move || {
        let mut requests = Vec::new();
        for response in responses {
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            let mut socket = loop {
                match listener.accept() {
                    Ok((socket, _)) => break socket,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            std::time::Instant::now() < deadline,
                            "host did not request expected turn"
                        );
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("{error}"),
                }
            };
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(socket.try_clone().unwrap());
            let mut length = 0;
            loop {
                let mut line = String::new();
                assert!(reader.read_line(&mut line).unwrap() > 0);
                if line == "\r\n" {
                    break;
                }
                if let Some((name, value)) = line.split_once(':')
                    && name.eq_ignore_ascii_case("content-length") {
                        length = value.trim().parse().unwrap();
                    }
            }
            assert!(length < 1_000_000);
            let mut bytes = vec![0; length];
            reader.read_exact(&mut bytes).unwrap();
            requests.push(serde_json::from_slice(&bytes).unwrap());
            let body = serde_json::to_vec(&response).unwrap();
            write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
            socket.write_all(&body).unwrap();
        }
        requests
    });
    let mut client = ProviderClient::from_env("openai", "gpt-4o")
        .unwrap()
        .with_endpoint(endpoint)
        .unwrap();
    client.api_key = None;
    client
        .caps
        .set(ProviderCapability::Streaming, Capability::Unsupported);
    (client, handle)
}
fn answer(text: &str) -> Value {
    json!({"choices":[{"message":{"role":"assistant","content":text},"finish_reason":"stop"}],"usage":{"total_tokens":8}})
}

#[test]
fn alias_bind_cfg_and_convars_follow_rewind() {
    let workspace = Workspace::new();
    let mut journal = workspace.journal();
    let mut host = workspace.host();
    std::fs::write(
        workspace.0.join("child.cfg"),
        "alias cold \"ai_temperature 0.1\"; cold; bind ctrl+t \"toggle ai_thinking 0 1 unknown\"",
    )
    .unwrap();
    host.execute_command(&mut journal, "exec child.cfg")
        .unwrap();
    assert_eq!(
        journal.snapshot().session_globals()["ai_temperature"],
        TypedValue::Number(0.1)
    );
    let before = journal.snapshot().offset;
    host.execute_command(
        &mut journal,
        "toggle ai_thinking 0 1 unknown; toggle ai_thinking 0 1 unknown",
    )
    .unwrap();
    assert_eq!(
        host.convars.get_typed::<TriState>("ai_thinking").unwrap(),
        TriState::True
    );
    let offset = journal.snapshot().offset;
    assert!(
        host.execute_command(&mut journal, "ai_temperature 0.5; exec absent.cfg")
            .is_err()
    );
    assert_eq!(journal.snapshot().offset, offset);
    journal.rewind_to(0).unwrap();
    assert!(host.execute_command(&mut journal, "cold").is_err());
    assert_eq!(
        host.convars.get_typed::<f64>("ai_temperature").unwrap(),
        0.7
    );
    assert!(host.command_engine.get_bind("ctrl+t").is_none());
    journal.rewind_to(before).unwrap();
    host.execute_command(&mut journal, "cold").unwrap();
    assert_eq!(
        journal.snapshot().session_globals()["ai_temperature"],
        TypedValue::Number(0.1)
    );
    assert!(host.command_engine.get_bind("ctrl+t").is_some());
}

#[test]
fn completed_force_is_removed_durably_after_actual_write() {
    let workspace = Workspace::new();
    let mut journal = workspace.journal();
    let write = json!({"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_write","type":"function","function":{"name":"Write","arguments":"{\"path\":\"proof.txt\",\"content\":\"persisted\",\"i\":\"Writing proof\"}"}}]},"finish_reason":"tool_calls"}]});
    let (mut client, server) = provider(vec![write, answer("Written")]);
    client.caps.set(
        ProviderCapability::NativeToolChoice,
        Capability::Unsupported,
    );
    let mut host = workspace.host().with_provider_client(client);
    host.execute_command(&mut journal, "force Write \"Write proof\" 3")
        .unwrap();
    host.run_turn(&mut journal, "Write proof").unwrap();
    assert_eq!(
        std::fs::read_to_string(workspace.0.join("proof.txt")).unwrap(),
        "persisted"
    );
    assert_eq!(journal.snapshot().active_directors().count(), 0);
    let call = journal
        .snapshot()
        .get_visible_body()
        .find(|e| e.kind == "tool_call")
        .unwrap();
    assert_eq!(
        call.attributes["status"],
        TypedValue::String("succeeded".into())
    );
    let input = journal
        .snapshot()
        .children(&call.id)
        .find(|e| e.kind == "input")
        .unwrap();
    assert_eq!(input.payload.as_ref().unwrap()["path"], "proof.txt");
    let requests = server.join().unwrap();
    assert_eq!(requests[0]["tools"].as_array().unwrap().len(), 7);
    assert!(
        requests[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|message| message["role"] == "tool"
                && message["content"].as_str().unwrap().contains("proof.txt"))
    );
    let path = workspace.0.join("session.journal");
    drop(journal);
    let restored = Journal::open(path).unwrap();
    assert_eq!(restored.snapshot().active_directors().count(), 0);
}

#[test]
fn costly_force_escalates_and_exhaustion_survives_resume() {
    let workspace = Workspace::new();
    let mut journal = workspace.journal();
    let (mut client, server) = provider(vec![answer("No call"), answer("Still no call")]);
    client
        .caps
        .set(ProviderCapability::NativeToolChoice, Capability::Supported);
    client.caps.native_tool_cost = NativeToolCost::Costly;
    let mut host = workspace.host().with_provider_client(client);
    host.execute_command(&mut journal, "force Write \"Write proof\" 2")
        .unwrap();
    let error = host.run_turn(&mut journal, "Write proof").unwrap_err();
    assert_eq!(error.code, "force_tool_exhausted");
    let requests = server.join().unwrap();
    assert_ne!(requests[0]["tool_choice"]["function"]["name"], "Write");
    assert_eq!(requests[1]["tool_choice"]["function"]["name"], "Write");
    for request in requests {
        assert!(
            request["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|message| message["content"]
                    .as_str()
                    .is_some_and(|text| text.contains("Write") && text.contains("MUST")))
        );
    }
    drop(journal);
    let restored = Journal::open(workspace.0.join("session.journal")).unwrap();
    let director = restored.snapshot().active_directors().next().unwrap();
    assert_eq!(director.payload.as_ref().unwrap()["attempt_count"], 2);
}

#[test]
fn compaction_rejects_both_forward_mutation_and_branch_change() {
    let guard = SpeculativeCompactionGuard::new();
    let branch = BranchId::new("main").unwrap();
    let source = guard.create_snapshot(branch.clone(), JournalOffset(10), 100, 110);
    let fold = MessageFold::Shake {
        kept_prefix_count: 1,
        dropped_count: 2,
    };
    assert!(
        guard
            .validate_and_splice(&source, &branch, &JournalOffset(10), fold.clone())
            .is_ok()
    );
    assert!(
        guard
            .validate_and_splice(&source, &branch, &JournalOffset(11), fold.clone())
            .is_err()
    );
    assert!(
        guard
            .validate_and_splice(&source, &BranchId::mint(), &JournalOffset(10), fold)
            .is_err()
    );
}

#[test]
fn rejected_mixed_stream_has_no_filesystem_or_configuration_effects() {
    let workspace = Workspace::new();
    let mut journal = workspace.journal();
    let mut host = workspace.host();
    let error = host.execute_command(&mut journal,
        r#"ai_temperature 0.2; tool Write "{\"path\":\"unexpected.txt\",\"content\":\"bad\",\"i\":\"Writing mixed stream\"}""#).unwrap_err();
    assert_eq!(error.code, "non_atomic_command_stream");
    assert!(!workspace.0.join("unexpected.txt").exists());
    assert_eq!(journal.snapshot().offset, 0);
    assert_eq!(
        host.convars.get_typed::<f64>("ai_temperature").unwrap(),
        0.7
    );
}

#[test]
fn rewind_revokes_cheat_gate_and_restores_client_settings() {
    use omp_control::{ConVar, ConVarFlags};
    let workspace = Workspace::new();
    let mut journal = workspace.journal();
    let mut host = workspace.host();
    host.convars.register(ConVar::new(
        "guarded",
        false,
        "Protected value",
        ConVarFlags::SESSION | ConVarFlags::CHEAT,
    ));
    host.execute_command(&mut journal, "sv_cheats 1; guarded 1; cl_showthinking 0")
        .unwrap();
    let saved = journal.snapshot().offset;
    journal.rewind_to(0).unwrap();
    assert!(host.execute_command(&mut journal, "guarded 1").is_err());
    assert!(!host.convars.cheats_enabled());
    assert!(host.convars.get_typed::<bool>("cl_showthinking").unwrap());
    journal.rewind_to(saved).unwrap();
    host.execute_command(&mut journal, "guarded 0").unwrap();
    assert!(!host.convars.get_typed::<bool>("cl_showthinking").unwrap());
}

#[test]
fn inherited_child_settings_survive_host_recreation() {
    let workspace = Workspace::new();
    let mut parent = workspace.journal();
    let mut host = workspace.host();
    host.execute_command(&mut parent, "ai_temperature 0.2; cl_showthinking 0")
        .unwrap();
    let child_workspace = Workspace::new();
    std::fs::create_dir(child_workspace.0.join("configs")).unwrap();
    std::fs::write(
        child_workspace.0.join("configs/subagent.cfg"),
        "ai_max_tokens 1234",
    )
    .unwrap();
    let mut child_journal = child_workspace.journal();
    let child = host
        .seed_child_host(
            parent.snapshot(),
            &mut child_journal,
            child_workspace.0.clone(),
            ActorId::mint(),
            None,
        )
        .unwrap();
    drop(child);
    let mut restored = child_workspace.host();
    restored
        .execute_command(&mut child_journal, "ai_temperature")
        .unwrap();
    assert_eq!(
        restored.convars.get_typed::<f64>("ai_temperature").unwrap(),
        0.2
    );
    assert_eq!(
        restored.convars.get_typed::<i64>("ai_max_tokens").unwrap(),
        1234
    );
    assert!(
        !restored
            .convars
            .get_typed::<bool>("cl_showthinking")
            .unwrap()
    );
    assert_eq!(
        parent.snapshot().session_globals()["ai_temperature"],
        TypedValue::Number(0.2)
    );
}
