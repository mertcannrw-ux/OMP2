//! Provider registry: many providers recorded per session, one active, and a
//! model list that names where each model comes from.

use omp_control::SessionHost;
use omp_state::Journal;
use omp_types::{ActorId, SessionId, TypedValue};
use serde_json::Value;
use std::{
    io::{BufRead, BufReader, Write},
    net::TcpListener,
    path::PathBuf,
    thread,
    time::Duration,
};

struct Workspace(PathBuf);

impl Workspace {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("omp-providers-{}", SessionId::mint()));
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

/// Serves one catalog: `GET …/models` answering with the given models.
fn catalog_provider(models: Vec<(&'static str, u64)>) -> (String, thread::JoinHandle<usize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    let endpoint = format!("http://127.0.0.1:{port}/v1");
    let handle = thread::spawn(move || {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut served = 0;
        while std::time::Instant::now() < deadline {
            let mut socket = match listener.accept() {
                Ok((socket, _)) => socket,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                    continue;
                }
                Err(error) => panic!("{error}"),
            };
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(socket.try_clone().unwrap());
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                continue;
            }
            loop {
                let mut header = String::new();
                if reader.read_line(&mut header).unwrap_or(0) == 0 || header == "\r\n" {
                    break;
                }
            }
            served += 1;
            let body = serde_json::json!({
                "data": models
                    .iter()
                    .map(|(id, context)| serde_json::json!({"id": id, "context_length": context}))
                    .collect::<Vec<Value>>()
            })
            .to_string();
            let _ = write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            if served >= 4 {
                break;
            }
        }
        served
    });
    (endpoint, handle)
}

/// Last diagnostic the console would have shown.
fn last_diagnostic(journal: &Journal) -> String {
    journal
        .snapshot()
        .get_visible_body()
        .filter(|element| element.kind == "diagnostic")
        .last()
        .map(|element| element.text.clone())
        .unwrap_or_default()
}

#[test]
fn providers_are_recorded_listed_switched_and_forgotten() {
    let workspace = Workspace::new();
    let mut journal = workspace.journal();
    let mut host = workspace.host();
    let (endpoint, server) = catalog_provider(vec![("stub-large", 200_000), ("stub-small", 32_000)]);
    let key_env = format!("OMP_TEST_PROVIDER_KEY_{}", std::process::id());

    // 1. Register: the record lands even when the credential is missing, so the
    //    user can fix the environment afterwards instead of losing the entry.
    host.execute_command(
        &mut journal,
        &format!("provider add go-stub {endpoint} --key-env {key_env}"),
    )
    .unwrap();
    let records = host.providers(journal.snapshot());
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].name, "go-stub");
    assert_eq!(records[0].adapter, "openai_compatible");
    assert_eq!(records[0].key_env, key_env);
    assert!(
        last_diagnostic(&journal).contains("fetched at startup"),
        "registration is pure configuration, so a profile may declare providers: {}",
        last_diagnostic(&journal)
    );
    assert!(
        host.provider_models(journal.snapshot(), &records[0]).is_empty(),
        "no network work happens during registration"
    );

    // 2. List names it and marks which provider is active.
    host.execute_command(&mut journal, "provider list").unwrap();
    let listing = last_diagnostic(&journal);
    assert!(listing.contains("go-stub"), "{listing}");
    assert!(listing.contains(&endpoint), "{listing}");
    assert!(listing.contains("Switch: /provider use <name>"), "{listing}");

    // 3. Switching to an unknown name explains what is available.
    let error = host
        .execute_command(&mut journal, "provider use nope")
        .unwrap_err();
    assert_eq!(error.code, "provider_not_registered");
    assert!(error.message.contains("go-stub"), "{}", error.message);

    // 4. With the credential present, a refresh fills in the catalog that
    //    registration deliberately skipped.
    unsafe {
        std::env::set_var(&key_env, "test-key");
    }
    host.execute_command(&mut journal, "provider refresh").unwrap();
    let record = host.providers(journal.snapshot()).remove(0);
    assert_eq!(
        host.provider_models(journal.snapshot(), &record).len(),
        2,
        "refresh fetched the registered provider's catalog"
    );

    // 5. Switching adopts the provider and keeps the catalog on its record.
    host.execute_command(&mut journal, "provider use go-stub")
        .unwrap();
    let globals = journal.snapshot().session_globals();
    assert_eq!(
        globals["ai_endpoint"],
        TypedValue::String(endpoint.clone())
    );
    assert_eq!(
        globals["ai_provider"],
        TypedValue::String("openai_compatible".into())
    );
    assert_eq!(globals["ai_api_key_env"], TypedValue::String(key_env.clone()));
    assert_eq!(
        globals["ai_model"],
        TypedValue::String("stub-large".into()),
        "the catalog's first model is selected"
    );
    assert!(last_diagnostic(&journal).contains("Active provider is now 'go-stub'"));

    // 6. Forgetting removes the record and says the active settings are untouched.
    host.execute_command(&mut journal, "provider remove go-stub")
        .unwrap();
    assert!(
        host.providers(journal.snapshot())
            .iter()
            .all(|record| record.name != "go-stub"),
        "the forgotten provider is gone"
    );
    assert!(last_diagnostic(&journal).contains("still point at"));

    // A provider the environment configures is recorded as well (the test
    // binary inherits OPENAI_BASE_URL): the registry mirrors reality rather than
    // only what the user typed.
    let environment_provider = host
        .providers(journal.snapshot())
        .into_iter()
        .find(|record| record.name != "go-stub");
    if let Some(record) = environment_provider {
        assert!(
            !record.endpoint.is_empty() && !record.adapter.is_empty(),
            "an auto-recorded provider still carries its endpoint and adapter"
        );
    }

    unsafe {
        std::env::remove_var(&key_env);
    }
    server.join().unwrap();
}

#[test]
fn two_providers_coexist_and_only_one_is_active() {
    let workspace = Workspace::new();
    let mut journal = workspace.journal();
    let mut host = workspace.host();
    let (first, first_server) = catalog_provider(vec![("alpha-1", 100_000)]);
    let (second, second_server) = catalog_provider(vec![("beta-1", 200_000)]);
    let key_env = format!("OMP_TEST_MULTI_KEY_{}", std::process::id());
    unsafe {
        std::env::set_var(&key_env, "test-key");
    }

    host.execute_command(
        &mut journal,
        &format!("provider add alpha {first} --key-env {key_env}"),
    )
    .unwrap();
    host.execute_command(
        &mut journal,
        &format!("provider add beta {second} --key-env {key_env} --model beta-1"),
    )
    .unwrap();
    assert_eq!(host.providers(journal.snapshot()).len(), 2);

    host.execute_command(&mut journal, "provider use alpha").unwrap();
    assert_eq!(
        journal.snapshot().session_globals()["ai_endpoint"],
        TypedValue::String(first.clone())
    );
    host.execute_command(&mut journal, "provider use beta").unwrap();
    assert_eq!(
        journal.snapshot().session_globals()["ai_endpoint"],
        TypedValue::String(second.clone())
    );
    assert_eq!(
        journal.snapshot().session_globals()["ai_model"],
        TypedValue::String("beta-1".into()),
        "a provider's pinned model is honoured on switch"
    );

    // Both catalogs stay available: the inactive one is cached on its record.
    let records = host.providers(journal.snapshot());
    let alpha = records.iter().find(|record| record.name == "alpha").unwrap();
    assert_eq!(host.provider_models(journal.snapshot(), alpha).len(), 1);

    // The report marks exactly the active provider.
    let report = host.provider_report(journal.snapshot());
    assert!(report.contains("* beta"), "{report}");
    assert!(report.contains("- alpha"), "{report}");

    unsafe {
        std::env::remove_var(&key_env);
    }
    first_server.join().unwrap();
    second_server.join().unwrap();
}
