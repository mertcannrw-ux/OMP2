//! End-to-end compaction against a real turn loop.
//!
//! A long session must (a) stop sending the oldest turns, (b) carry an expand
//! marker for them, and (c) keep the originals retrievable through `Read
//! summary://<id>` — compaction that loses the history would be a regression,
//! not a feature.

use omp_control::SessionHost;
use omp_inference::{
    capability::{ProviderCapability, TriState as Capability},
    provider::ProviderClient,
};
use omp_state::Journal;
use omp_types::{
    ActorId, ElementId, JournalOffset, Patch, PatchOp, SessionId, SummaryNode, ToolCallId,
    TypedValue,
};
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
        let path = std::env::temp_dir().join(format!("omp-compaction-e2e-{}", SessionId::mint()));
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

fn assistant_reply(text: &str) -> Value {
    json!({"role": "assistant", "content": text})
}

fn tool_call_reply(tool: &str, arguments: Value) -> Value {
    json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": "scripted-call",
            "type": "function",
            "function": {"name": tool, "arguments": arguments.to_string()}
        }]
    })
}

/// Serves one completion per scripted message and records every request body.
fn scripted_provider(replies: Vec<Value>) -> (ProviderClient, thread::JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!(
        "http://{}/v1/chat/completions",
        listener.local_addr().unwrap()
    );
    let handle = thread::spawn(move || {
        let mut requests = Vec::new();
        for reply in replies {
            let deadline = std::time::Instant::now() + Duration::from_secs(60);
            let mut socket = loop {
                match listener.accept() {
                    Ok((socket, _)) => break socket,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            std::time::Instant::now() < deadline,
                            "host did not request the expected turn"
                        );
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("{error}"),
                }
            };
            socket
                .set_read_timeout(Some(Duration::from_secs(30)))
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
                    && name.eq_ignore_ascii_case("content-length")
                {
                    length = value.trim().parse().unwrap();
                }
            }
            let mut bytes = vec![0; length];
            reader.read_exact(&mut bytes).unwrap();
            requests.push(serde_json::from_slice(&bytes).unwrap());
            let finish = if reply["tool_calls"].is_array() {
                "tool_calls"
            } else {
                "stop"
            };
            let body = serde_json::to_vec(&json!({
                "choices": [{"index": 0, "message": reply, "finish_reason": finish}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10}
            }))
            .unwrap();
            write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
            socket.write_all(&body).unwrap();
        }
        requests
    });
    let mut client = ProviderClient::from_env("openai", "small-model")
        .unwrap()
        .with_endpoint(endpoint)
        .unwrap();
    client.api_key = None;
    client
        .caps
        .set(ProviderCapability::Streaming, Capability::Unsupported);
    (client, handle)
}

/// Writes a convar the way an accepted provider refresh does — through the
/// journal, not through the READONLY-guarded setter.
fn set_convar(journal: &mut Journal, name: &str, value: TypedValue) {
    let container = journal.snapshot().container("convars").clone();
    journal
        .append_patch(Patch {
            base_offset: JournalOffset(journal.snapshot().offset),
            result_offset: journal.next_offset(),
            by: ActorId::new("owner").unwrap().into(),
            reason: "test provider metadata".into(),
            ops: vec![PatchOp::SetAttribute {
                element: container,
                name: name.into(),
                value,
            }],
        })
        .unwrap();
}

/// A long reply with no repeated phrase, so the harness' repetition-loop
/// detector stays out of the way.
fn long_reply() -> String {
    format!(
        "reply {}",
        (0..200)
            .map(|index| format!("token{index}"))
            .collect::<Vec<_>>()
            .join(" ")
    )
}

/// The marker sits past the inventory preview budget, so its absence from a
/// request proves the element itself was elided rather than merely summarised.
fn turn_message(index: usize) -> String {
    format!(
        "turn {index} {} UNIQUE-TURN-{index}-MARKER",
        "filler ".repeat(500)
    )
}

#[test]
fn a_long_session_compacts_and_keeps_the_originals_retrievable() {
    let workspace = Workspace::new();
    let mut journal = workspace.journal();
    // Small advertised window: the compaction budget is derived from it.
    set_convar(&mut journal, "ai_context_length", TypedValue::Integer(2_500));
    set_convar(&mut journal, "ai_max_tokens", TypedValue::Integer(512));

    let (client, server) = scripted_provider(vec![assistant_reply(&long_reply()); 5]);
    let mut host = workspace.host().with_provider_client(client);

    for index in 0..5 {
        host.run_turn(&mut journal, &turn_message(index)).unwrap();
    }
    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 5);

    let first_request = requests.first().unwrap().to_string();
    let last_request = requests.last().unwrap().to_string();
    assert!(
        !first_request.contains("summary://"),
        "a short session must not compact"
    );
    assert!(
        last_request.contains("summary://"),
        "the long session folds earlier turns and says how to expand them"
    );
    assert!(
        !last_request.contains("UNIQUE-TURN-0-MARKER"),
        "the first turn is no longer sent verbatim"
    );
    assert!(
        last_request.contains("UNIQUE-TURN-4-MARKER"),
        "the turn in flight is still sent verbatim"
    );

    let snapshot = journal.snapshot();
    let node = snapshot
        .children(snapshot.container("summaries"))
        .next()
        .expect("compaction journaled a summary node")
        .clone();
    let parsed = SummaryNode::from_element(&node).expect("the node payload decodes");
    assert!(!parsed.covered.is_empty());
    for covered in &parsed.covered {
        assert!(
            snapshot.element(covered).is_some(),
            "a covered element is still in the document"
        );
    }

    // Losslessness, through the same tool the model would use.
    let response = host
        .tool_host
        .execute_tool(
            &mut journal,
            &omp_tools::definition::ToolCall::new(
                ToolCallId::mint(),
                "Read",
                "1.0.0",
                json!({"path": format!("summary://{}", node.id), "i": "Expanding compacted history"}),
            ),
        )
        .expect("summary:// expansion succeeds");
    assert!(
        response.output.content.contains("UNIQUE-TURN-0-MARKER"),
        "the elided turn comes back verbatim: {}",
        response.output.content
    );
}

#[test]
fn a_session_without_an_advertised_window_never_compacts() {
    let workspace = Workspace::new();
    let mut journal = workspace.journal();
    let (client, server) = scripted_provider(vec![assistant_reply(&long_reply()); 3]);
    let mut host = workspace.host().with_provider_client(client);

    for index in 0..3 {
        host.run_turn(&mut journal, &turn_message(index)).unwrap();
    }
    let requests = server.join().unwrap();
    assert!(
        requests
            .iter()
            .all(|request| !request.to_string().contains("summary://")),
        "no advertised context window means no compaction"
    );
    assert_eq!(
        journal
            .snapshot()
            .children(journal.snapshot().container("summaries"))
            .count(),
        0
    );
}

#[test]
fn compaction_survives_a_forked_branch() {
    let workspace = Workspace::new();
    let mut journal = workspace.journal();
    set_convar(&mut journal, "ai_context_length", TypedValue::Integer(2_500));
    set_convar(&mut journal, "ai_max_tokens", TypedValue::Integer(512));
    let (client, server) = scripted_provider(vec![assistant_reply(&long_reply()); 4]);
    let mut host = workspace.host().with_provider_client(client);
    for index in 0..4 {
        host.run_turn(&mut journal, &turn_message(index)).unwrap();
    }
    let offset_before = journal.snapshot().offset;
    let node_count = journal
        .snapshot()
        .children(journal.snapshot().container("summaries"))
        .count();
    assert!(node_count > 0);

    // Fork from the last offset: the DAG is durable state, so replaying the
    // branch reproduces it exactly.
    journal.fork_at(offset_before).unwrap();
    let replayed = journal.snapshot();
    assert_eq!(
        replayed.children(replayed.container("summaries")).count(),
        node_count,
        "the summary DAG is part of the durable branch state"
    );
    let id: Vec<ElementId> = replayed
        .children(replayed.container("summaries"))
        .map(|element| element.id.clone())
        .collect();
    assert!(!id.is_empty());
    server.join().unwrap();
}

#[test]
fn oversized_tool_arguments_never_reach_the_next_request() {
    let workspace = Workspace::new();
    let mut journal = workspace.journal();
    set_convar(&mut journal, "ai_context_length", TypedValue::Integer(2_500));
    set_convar(&mut journal, "ai_max_tokens", TypedValue::Integer(512));

    // A 40 KB tool payload the model already acted on: re-sending it would
    // consume the whole window, so the projection replaces it with a marker.
    let payload = "z".repeat(40_000);
    let (client, server) = scripted_provider(vec![
        tool_call_reply(
            "Write",
            json!({"path": "big.txt", "content": payload, "i": "Writing a large file"}),
        ),
        assistant_reply("written"),
    ]);
    let mut host = workspace.host().with_provider_client(client);
    host.run_turn(&mut journal, "write a big file").unwrap();
    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 2, "the tool call is answered by a second completion");

    let follow_up = requests[1].to_string();
    assert!(
        !follow_up.contains(&payload),
        "the payload must not be re-sent verbatim"
    );
    assert!(
        follow_up.contains("omitted"),
        "the projection marks the arguments as elided"
    );
    assert_eq!(
        std::fs::read_to_string(workspace.0.join("big.txt")).unwrap().len(),
        40_000,
        "the tool still wrote the real file"
    );
    // The journal keeps the original arguments, so nothing is lost.
    let call = journal
        .snapshot()
        .get_visible_body()
        .find(|element| element.kind == "tool_call")
        .expect("the tool call is journaled");
    let arguments = journal
        .snapshot()
        .children(&call.id)
        .find(|child| child.kind == "input")
        .and_then(|child| child.payload.clone())
        .unwrap();
    assert!(arguments.to_string().contains(&payload));
}
