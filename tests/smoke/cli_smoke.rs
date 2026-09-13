use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

fn scripted_provider() -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!(
        "http://{}/v1/chat/completions",
        listener.local_addr().unwrap()
    );
    let worker = std::thread::spawn(move || {
        for step in 0..5 {
            let deadline = Instant::now() + Duration::from_secs(30);
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(connection) => break connection,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && Instant::now() < deadline =>
                    {
                        std::thread::sleep(Duration::from_millis(10))
                    }
                    Err(error) => panic!("provider accept: {error}"),
                }
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut size = 0;
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            loop {
                let mut line = String::new();
                assert_ne!(reader.read_line(&mut line).unwrap(), 0);
                if line == "\r\n" {
                    break;
                }
                if let Some((key, value)) = line.split_once(':')
                    && key.eq_ignore_ascii_case("content-length") {
                        size = value.trim().parse::<usize>().unwrap();
                    }
            }
            if request_line.starts_with("GET /v1/models ") {
                let body = serde_json::json!({"data":[{"id":"smoke-model","context_length":if step == 4 {64000} else {32000},"max_output_tokens":if step == 4 {16000} else {8000}}]}).to_string();
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
                continue;
            }
            assert!(size <= 1_048_576);
            let mut body = vec![0; size];
            reader.read_exact(&mut body).unwrap();
            let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(request["model"], "smoke-model");
            assert_eq!(request["max_tokens"], 8000);
            let message = match step - 1 {
                0 => {
                    serde_json::json!({"role":"assistant", "content":null, "tool_calls":[{"id":"smoke-write", "type":"function", "function":{"name":"Write","arguments":serde_json::json!({"i":"Writing smoke evidence", "path":"proof.txt", "content":"durable smoke evidence\n"}).to_string()}}]})
                }
                1 => {
                    serde_json::json!({"role":"assistant", "content":null, "tool_calls":[{"id":"smoke-read", "type":"function", "function":{"name":"Read","arguments":serde_json::json!({"i":"Reading smoke evidence", "path":"proof.txt:raw"}).to_string()}}]})
                }
                _ => {
                    assert!(
                        request["messages"].as_array().unwrap().iter().any(
                            |message| message["role"] == "tool"
                                && message["content"]
                                    .as_str()
                                    .is_some_and(|text| text.contains("durable smoke evidence"))
                        )
                    );
                    serde_json::json!({"role":"assistant", "content":"Verified durable smoke evidence."})
                }
            };
            let body = serde_json::json!({"choices":[{"index":0,"message":message,"finish_reason":if step < 3 {"tool_calls"} else {"stop"}}],"usage":{"prompt_tokens":20,"completion_tokens":10}}).to_string();
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
        }
    });
    (endpoint, worker)
}

fn omp_binary() -> PathBuf {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_omp2") {
        return PathBuf::from(path);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path.push("target");
    path.push("debug");
    path.push(if cfg!(windows) { "omp2.exe" } else { "omp2" });
    path
}

fn create_temp_workspace(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    dir.push(format!("omp2_smoke_{}_{}", name, nonce));
    fs::create_dir_all(&dir).expect("failed to create temp workspace dir");
    dir
}

#[test]
fn bare_command_creates_session_while_help_does_not() {
    let workspace = create_temp_workspace("bare_command");
    let mut command = Command::new(omp_binary());
    command
        .current_dir(&workspace)
        .stdin(std::process::Stdio::null());
    for name in [
        "OMP_ENDPOINT",
        "AI_ENDPOINT",
        "OPENAI_BASE_URL",
        "OMP_PROVIDER",
        "AI_PROVIDER",
        "OMP_MODEL",
        "AI_MODEL",
        "OMP_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ] {
        command.env_remove(name);
    }
    let help = command.arg("--help").output().unwrap();
    assert!(help.status.success());
    assert!(!workspace.join(".omp").exists());
    // Prevent ancestor workspace markers in the shared temporary directory from
    // redirecting this isolated CLI scenario's journal into another workspace.
    fs::create_dir(workspace.join(".omp")).unwrap();

    let mut command = Command::new(omp_binary());
    command
        .current_dir(&workspace)
        .stdin(std::process::Stdio::null());
    for name in [
        "OMP_ENDPOINT",
        "AI_ENDPOINT",
        "OPENAI_BASE_URL",
        "OMP_PROVIDER",
        "AI_PROVIDER",
        "OMP_MODEL",
        "AI_MODEL",
        "OMP_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ] {
        command.env_remove(name);
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let journals: Vec<_> = fs::read_dir(workspace.join(".omp/state"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "journal")
        })
        .collect();
    assert_eq!(journals.len(), 1);
    drop(omp_state::Journal::open(&journals[0]).unwrap());
    fs::remove_dir_all(workspace).unwrap();
}

#[test]
fn test_all_seven_subcommands_lifecycle() {
    let bin = omp_binary();
    let ws = create_temp_workspace("all_subcommands");
    let journal_path = ws.join(".omp").join("state").join("test_session.journal");
    let (endpoint, provider) = scripted_provider();

    // 1. run subcommand
    let run_output = Command::new(&bin)
        .env("OMP_ENDPOINT", &endpoint)
        .env("AI_PROVIDER", "openai_compatible")
        .env("AI_MODEL", "")
        .env("OMP_MODEL", "")
        .env("OMP_API_KEY", "local-smoke-only")
        .args([
            "run",
            "--workspace",
            ws.to_str().unwrap(),
            "--journal",
            journal_path.to_str().unwrap(),
            "--session-id",
            "smoke-sess-1",
            "--message",
            "Hello, omp2 smoke test!",
            "--json",
        ])
        .output()
        .expect("failed to spawn omp2 run");

    assert!(
        run_output.status.success(),
        "omp2 run failed: {}",
        String::from_utf8_lossy(&run_output.stderr)
    );
    let run_val: serde_json::Value =
        serde_json::from_slice(&run_output.stdout).expect("omp2 run output should be valid JSON");
    assert_eq!(run_val["status"], "created");
    assert_eq!(run_val["session_id"], "smoke-sess-1");
    assert_eq!(run_val["turn_count"], 1);
    assert!(
        journal_path.exists(),
        "journal file must be created on disk"
    );
    assert_eq!(
        fs::read_to_string(ws.join("proof.txt")).unwrap(),
        "durable smoke evidence\n"
    );

    // 2. resume subcommand
    let resume_output = Command::new(&bin)
        .env("OMP_API_KEY", "local-smoke-only")
        .args([
            "resume",
            "--workspace",
            ws.to_str().unwrap(),
            "--journal",
            journal_path.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("failed to spawn omp2 resume");

    assert!(
        resume_output.status.success(),
        "omp2 resume failed: {}",
        String::from_utf8_lossy(&resume_output.stderr)
    );
    let resume_val: serde_json::Value = serde_json::from_slice(&resume_output.stdout)
        .expect("omp2 resume output should be valid JSON");
    assert_eq!(resume_val["status"], "resumed");
    assert_eq!(resume_val["session_id"], "smoke-sess-1");
    assert_eq!(resume_val["turn_count"], 1);
    provider.join().unwrap();
    let inspected = Command::new(&bin)
        .args([
            "inspect",
            "--workspace",
            ws.to_str().unwrap(),
            "--journal",
            journal_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    let state: serde_json::Value = serde_json::from_slice(&inspected.stdout).unwrap();
    let settings = &state["snapshot"]["nodes"]["convars"]["element"]["attributes"];
    assert_eq!(settings["ai_context_length"]["Integer"], 64000);
    assert_eq!(settings["ai_max_tokens"]["Integer"], 16000);

    // 3. fork subcommand (requires --offset)
    let fork_fail = Command::new(&bin)
        .args([
            "fork",
            "--workspace",
            ws.to_str().unwrap(),
            "--journal",
            journal_path.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("failed to spawn omp2 fork without offset");
    assert!(
        !fork_fail.status.success(),
        "fork without --offset must fail"
    );
    let fork_fail_err = String::from_utf8_lossy(&fork_fail.stderr);
    assert!(
        fork_fail_err.contains("missing_offset") || fork_fail_err.contains("requires --offset"),
        "expected missing_offset error, got: {fork_fail_err}"
    );

    let fork_ok = Command::new(&bin)
        .args([
            "fork",
            "--workspace",
            ws.to_str().unwrap(),
            "--journal",
            journal_path.to_str().unwrap(),
            "--offset",
            "1",
            "--json",
        ])
        .output()
        .expect("failed to spawn omp2 fork with offset");
    assert!(
        fork_ok.status.success(),
        "omp2 fork failed: {}",
        String::from_utf8_lossy(&fork_ok.stderr)
    );
    let fork_val: serde_json::Value =
        serde_json::from_slice(&fork_ok.stdout).expect("omp2 fork output should be valid JSON");
    assert_eq!(fork_val["status"], "forked");
    assert_eq!(fork_val["parent_offset"], 1);
    assert!(fork_val["branch"].is_string());

    // 4. inspect subcommand (JSON format)
    let inspect_json = Command::new(&bin)
        .args([
            "inspect",
            "--workspace",
            ws.to_str().unwrap(),
            "--journal",
            journal_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("failed to spawn omp2 inspect json");
    assert!(inspect_json.status.success());
    let inspect_val: serde_json::Value = serde_json::from_slice(&inspect_json.stdout)
        .expect("omp2 inspect json output must be valid JSON");
    assert!(inspect_val.get("tree").is_some());
    assert!(inspect_val.get("snapshot").is_some());

    // 4b. inspect subcommand (XML format)
    let inspect_xml = Command::new(&bin)
        .args([
            "inspect",
            "--workspace",
            ws.to_str().unwrap(),
            "--journal",
            journal_path.to_str().unwrap(),
            "--format",
            "xml",
        ])
        .output()
        .expect("failed to spawn omp2 inspect xml");
    assert!(inspect_xml.status.success());
    let xml_str = String::from_utf8_lossy(&inspect_xml.stdout);
    assert!(xml_str.contains("<root") && xml_str.contains("</root>"));

    // 5. doctor subcommand
    let doctor_out = Command::new(&bin)
        .args([
            "doctor",
            "--workspace",
            ws.to_str().unwrap(),
            "--journal",
            journal_path.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("failed to spawn omp2 doctor");
    assert!(doctor_out.status.success());
    let doc_val: serde_json::Value =
        serde_json::from_slice(&doctor_out.stdout).expect("doctor output should be valid JSON");
    assert_eq!(doc_val["status"], "healthy");
    assert_eq!(doc_val["protocol_ok"], true);

    // 6. replay subcommand
    let replay_out = Command::new(&bin)
        .args([
            "replay",
            "--workspace",
            ws.to_str().unwrap(),
            "--journal",
            journal_path.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("failed to spawn omp2 replay");
    assert!(replay_out.status.success());
    let replay_val: serde_json::Value =
        serde_json::from_slice(&replay_out.stdout).expect("replay output should be valid JSON");
    assert_eq!(replay_val["status"], "ok");
    assert!(replay_val["ancestry_offsets"].is_array());

    // 7. serve subcommand (unsupported transport check)
    let serve_unsupported = Command::new(&bin)
        .args([
            "serve",
            "--workspace",
            ws.to_str().unwrap(),
            "--journal",
            journal_path.to_str().unwrap(),
            "--transport",
            "unix",
            "--json",
        ])
        .output()
        .expect("failed to spawn omp2 serve with unsupported transport");
    assert!(!serve_unsupported.status.success());
    let stdout_str = String::from_utf8_lossy(&serve_unsupported.stdout);
    let stderr_str = String::from_utf8_lossy(&serve_unsupported.stderr);
    assert!(
        stdout_str.contains("unsupported_transport")
            || stderr_str.contains("unsupported_transport")
    );

    // 8. path policy rejection of ambiguous paths
    let ambiguous_run = Command::new(&bin)
        .args(["run", "--journal", "ambiguous*glob.journal"])
        .output()
        .expect("failed to spawn omp2 with ambiguous path");
    assert!(!ambiguous_run.status.success());
    let ambig_err = String::from_utf8_lossy(&ambiguous_run.stderr);
    assert!(ambig_err.contains("ambiguous_path"));

    let _ = fs::remove_dir_all(&ws);
}
