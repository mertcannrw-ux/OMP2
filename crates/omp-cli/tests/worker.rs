use omp_state::{Journal, SessionSnapshot};
use omp_types::{
    ActorId, ElementId, ElementSnapshot, JournalOffset, Patch, PatchOp, SessionId, TypedValue,
};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::mpsc,
    time::Duration,
};

struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
struct Workspace(PathBuf);
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn worker_projects_durable_commands_and_fork_without_resurrecting_state() {
    let workspace =
        Workspace(std::env::temp_dir().join(format!("omp2-ui-worker-{}", SessionId::mint())));
    fs::create_dir_all(&workspace.0).unwrap();
    let path = workspace.0.join("session.journal");
    let journal = Journal::create(&path, SessionId::mint()).unwrap();
    drop(journal);
    let mut process = Process(
        Command::new(env!("CARGO_BIN_EXE_omp2"))
            .args([
                "__ui-worker",
                workspace.0.to_str().unwrap(),
                path.to_str().unwrap(),
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let mut stdin = process.0.stdin.take().unwrap();
    let stdout = process.0.stdout.take().unwrap();
    let (send, receive) = mpsc::sync_channel(32);
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if send.send(line).is_err() {
                break;
            }
        }
    });
    writeln!(stdin, "__READY__").unwrap();
    let initial: serde_json::Value = serde_json::from_str(
        &receive
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    let mut snapshot: SessionSnapshot =
        serde_json::from_value(initial["Snapshot"]["snapshot"].clone()).unwrap();
    for line in ["/ai_temperature 0.23", "/fork 0"] {
        writeln!(stdin, "{}", serde_json::json!({"Submit":{"line":line}})).unwrap();
        loop {
            let frame: serde_json::Value = serde_json::from_str(
                &receive
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap()
                    .unwrap(),
            )
            .unwrap();
            if let Some(value) = frame.get("Snapshot") {
                snapshot = serde_json::from_value(value["snapshot"].clone()).unwrap();
            }
            if let Some(value) = frame.get("Patch") {
                let patch: Patch = serde_json::from_value(value["patch"].clone()).unwrap();
                omp_state::apply_patch(&mut snapshot, &patch).unwrap();
                snapshot.selected_branch = serde_json::from_value(value["branch"].clone()).unwrap();
            }
            if let Some(value) = frame.get("Settled") {
                assert!(value["error"].is_null(), "{value}");
                break;
            }
        }
        if line.starts_with("/ai_temperature") {
            assert_eq!(
                snapshot.session_globals().get("ai_temperature"),
                Some(&TypedValue::Number(0.23))
            );
        } else {
            assert!(!snapshot.session_globals().contains_key("ai_temperature"));
            assert_ne!(snapshot.current_branch().as_str(), "main");
        }
    }
    writeln!(
        stdin,
        "{}",
        serde_json::json!({"Shutdown":{"detach":false}})
    )
    .unwrap();
    assert!(process.0.wait().unwrap().success());
    let reopened = Journal::open(&path).unwrap();
    assert_eq!(
        reopened.snapshot().current_branch(),
        snapshot.current_branch()
    );
    assert!(
        !reopened
            .snapshot()
            .session_globals()
            .contains_key("ai_temperature")
    );
}

#[test]
fn new_session_preserves_old_journal_and_routes_subsequent_commands_to_clean_state() {
    let workspace = Workspace(std::env::temp_dir().join(format!("omp2-new-{}", SessionId::mint())));
    fs::create_dir_all(&workspace.0).unwrap();
    let path = workspace.0.join("old.journal");
    let old_id = SessionId::mint();
    let mut journal = Journal::create(&path, old_id.clone()).unwrap();
    let mut user = ElementSnapshot::new(ElementId::mint(), "user");
    user.text = "Old private conversation".into();
    let mut discovery = ElementSnapshot::new(ElementId::mint(), "tool");
    discovery
        .attributes
        .insert("name".into(), TypedValue::String("old_dynamic_tool".into()));
    journal
        .append_patch(Patch {
            base_offset: JournalOffset(0),
            result_offset: journal.next_offset(),
            by: ActorId::new("owner").unwrap().into(),
            reason: "previous session state".into(),
            ops: vec![
                PatchOp::Create {
                    parent: journal.snapshot().container("body").clone(),
                    index: 0,
                    element: user,
                },
                PatchOp::Create {
                    parent: journal.snapshot().container("tools").clone(),
                    index: 0,
                    element: discovery,
                },
                PatchOp::SetAttribute {
                    element: journal.snapshot().container("convars").clone(),
                    name: "ai_temperature".into(),
                    value: TypedValue::Number(0.23),
                },
            ],
        })
        .unwrap();
    drop(journal);
    let old_bytes = fs::read(&path).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_omp2"));
    for name in [
        "OMP_ENDPOINT",
        "AI_ENDPOINT",
        "OPENAI_BASE_URL",
        "OMP_PROVIDER",
        "AI_PROVIDER",
        "OMP_MODEL",
        "AI_MODEL",
    ] {
        command.env_remove(name);
    }
    let mut process = Process(
        command
            .args([
                "__ui-worker",
                workspace.0.to_str().unwrap(),
                path.to_str().unwrap(),
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let mut stdin = process.0.stdin.take().unwrap();
    let stdout = process.0.stdout.take().unwrap();
    let (send, receive) = mpsc::sync_channel(32);
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if send.send(line).is_err() {
                break;
            }
        }
    });
    let frame = || -> serde_json::Value {
        serde_json::from_str(
            &receive
                .recv_timeout(Duration::from_secs(15))
                .unwrap()
                .unwrap(),
        )
        .unwrap()
    };
    writeln!(stdin, "__READY__").unwrap();
    assert_eq!(
        frame()["Snapshot"]["snapshot"]["session_id"],
        old_id.as_str()
    );
    let mut new_path = None;
    let mut snapshot = None;
    for line in ["/new", "/ai_temperature 0.71"] {
        writeln!(stdin, "{}", serde_json::json!({"Submit":{"line":line}})).unwrap();
        loop {
            let value = frame();
            if let Some(changed) = value.get("SessionChanged") {
                let next: SessionSnapshot =
                    serde_json::from_value(changed["snapshot"].clone()).unwrap();
                assert_ne!(next.session_id, old_id);
                assert_eq!(next.turn_count(), 0);
                assert!(next.get_visible_body().next().is_none());
                assert!(next.active_tool_roster().next().is_none());
                assert_eq!(
                    next.session_globals()["ai_temperature"],
                    TypedValue::Number(0.23)
                );
                new_path = Some(PathBuf::from(changed["journal"].as_str().unwrap()));
                snapshot = Some(next);
            }
            if let Some(changed) = value.get("Patch") {
                let patch: Patch = serde_json::from_value(changed["patch"].clone()).unwrap();
                omp_state::apply_patch(snapshot.as_mut().unwrap(), &patch).unwrap();
            }
            if let Some(settled) = value.get("Settled") {
                assert!(settled["error"].is_null(), "{settled}");
                break;
            }
        }
    }
    writeln!(
        stdin,
        "{}",
        serde_json::json!({"Shutdown":{"detach":false}})
    )
    .unwrap();
    assert!(process.0.wait().unwrap().success());
    assert_eq!(fs::read(&path).unwrap(), old_bytes);
    let new_journal = Journal::open(new_path.unwrap()).unwrap();
    assert_eq!(
        new_journal.snapshot().session_id,
        snapshot.unwrap().session_id
    );
    assert_eq!(
        new_journal.snapshot().session_globals()["ai_temperature"],
        TypedValue::Number(0.71)
    );
}

#[test]
fn worker_restart_settles_interrupted_tool_without_cancelling_controller() {
    let workspace =
        Workspace(std::env::temp_dir().join(format!("omp2-ui-recovery-{}", SessionId::mint())));
    fs::create_dir_all(&workspace.0).unwrap();
    let path = workspace.0.join("session.journal");
    let mut journal = Journal::create(&path, SessionId::mint()).unwrap();
    let mut controller = ElementSnapshot::new(ElementId::mint(), "actor");
    let controller_id = controller.id.clone();
    controller
        .attributes
        .insert("status".into(), TypedValue::String("active".into()));
    let mut tool = ElementSnapshot::new(ElementId::mint(), "tool_call");
    let tool_id = tool.id.clone();
    tool.attributes
        .insert("status".into(), TypedValue::String("running".into()));
    journal
        .append_patch(Patch {
            base_offset: JournalOffset(0),
            result_offset: journal.next_offset(),
            by: ActorId::new("owner").unwrap().into(),
            reason: "interrupted operation".into(),
            ops: vec![
                PatchOp::Create {
                    parent: journal.snapshot().container("actors").clone(),
                    index: 0,
                    element: controller,
                },
                PatchOp::Create {
                    parent: journal.snapshot().container("body").clone(),
                    index: 0,
                    element: tool,
                },
            ],
        })
        .unwrap();
    drop(journal);
    let output = Command::new(env!("CARGO_BIN_EXE_omp2"))
        .args([
            "__ui-worker",
            workspace.0.to_str().unwrap(),
            path.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut process = Process(output);
    {
        let input = process.0.stdin.as_mut().unwrap();
        writeln!(input, "__READY__").unwrap();
        writeln!(
            input,
            "{}",
            serde_json::json!({"Shutdown":{"detach":false}})
        )
        .unwrap();
    }
    let mut reader = BufReader::new(process.0.stdout.take().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let value: serde_json::Value = serde_json::from_str(&line).unwrap();
    let snapshot: SessionSnapshot =
        serde_json::from_value(value["Snapshot"]["snapshot"].clone()).unwrap();
    assert_eq!(
        snapshot.element(&controller_id).unwrap().attributes["status"],
        TypedValue::String("active".into())
    );
    assert_eq!(
        snapshot.element(&tool_id).unwrap().attributes["status"],
        TypedValue::String("cancelled".into())
    );
    assert!(
        snapshot
            .children(&tool_id)
            .any(|child| child.kind == "result")
    );
    std::io::copy(&mut reader, &mut std::io::sink()).unwrap();
    assert!(process.0.wait().unwrap().success());
}

#[test]
fn worker_restart_settles_interrupted_assistant_streaming_and_preserves_partial() {
    let workspace =
        Workspace(std::env::temp_dir().join(format!("omp2-ui-asst-rec-{}", SessionId::mint())));
    fs::create_dir_all(&workspace.0).unwrap();
    let path = workspace.0.join("session.journal");
    let mut journal = Journal::create(&path, SessionId::mint()).unwrap();

    let mut asst = ElementSnapshot::new(ElementId::mint(), "assistant");
    let asst_id = asst.id.clone();
    asst.attributes
        .insert("status".into(), TypedValue::String("running".into()));
    asst.attributes
        .insert("streaming".into(), TypedValue::Bool(true));
    asst.text = "Partial response that was cut off by restart".into();

    let mut think = ElementSnapshot::new(ElementId::mint(), "think");
    let think_id = think.id.clone();
    think.text = "Partial reasoning before kill".into();

    journal
        .append_patch(Patch {
            base_offset: JournalOffset(0),
            result_offset: journal.next_offset(),
            by: ActorId::new("owner").unwrap().into(),
            reason: "interrupted assistant streaming".into(),
            ops: vec![
                PatchOp::Create {
                    parent: journal.snapshot().container("body").clone(),
                    index: 0,
                    element: asst,
                },
                PatchOp::Create {
                    parent: asst_id.clone(),
                    index: 0,
                    element: think,
                },
            ],
        })
        .unwrap();
    drop(journal);

    let output = Command::new(env!("CARGO_BIN_EXE_omp2"))
        .args([
            "__ui-worker",
            workspace.0.to_str().unwrap(),
            path.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut process = Process(output);
    {
        let input = process.0.stdin.as_mut().unwrap();
        writeln!(input, "__READY__").unwrap();
        writeln!(
            input,
            "{}",
            serde_json::json!({"Shutdown":{"detach":false}})
        )
        .unwrap();
    }
    let mut reader = BufReader::new(process.0.stdout.take().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let value: serde_json::Value = serde_json::from_str(&line).unwrap();
    let snapshot: SessionSnapshot =
        serde_json::from_value(value["Snapshot"]["snapshot"].clone()).unwrap();

    let asst_elem = snapshot.element(&asst_id).unwrap();
    assert_eq!(
        asst_elem.attributes.get("status"),
        Some(&TypedValue::String("cancelled".into()))
    );
    assert_eq!(
        asst_elem.attributes.get("streaming"),
        Some(&TypedValue::Bool(false))
    );
    assert_eq!(
        asst_elem.text,
        "Partial response that was cut off by restart"
    );
    let think_child = snapshot.element(&think_id).unwrap();
    assert_eq!(think_child.text, "Partial reasoning before kill");

    assert!(snapshot.children(&asst_id).any(|child| child.kind == "diag"
        && child.attributes.get("code")
            == Some(&TypedValue::String("worker_restart_cancelled".into()))));
    std::io::copy(&mut reader, &mut std::io::sink()).unwrap();
    assert!(process.0.wait().unwrap().success());
}
