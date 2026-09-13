use omp_runtime::artifact::{ArtifactStore, ScopeCredential};
use omp_runtime::{Job, JobKind, JobSignal, WorkspaceView};
use omp_state::Journal;
use omp_types::*;
use std::{fs, time::Duration};

#[test]
#[cfg(windows)]
fn overflow_forced_kill_and_reopen_preserve_authority() {
    let root = std::env::temp_dir().join(format!("omp-job-{}", SessionId::mint()));
    fs::create_dir_all(root.join("source")).unwrap();
    fs::write(
        root.join("source/overflow.cmd"),
        "@echo off\r\n:repeat\r\necho overflowing isolated output\r\ngoto repeat\r\n",
    )
    .unwrap();
    let sid = SessionId::mint();
    let view =
        WorkspaceView::allocate(root.join("source"), root.join("views"), sid.clone()).unwrap();
    let mut journal = Journal::create(root.join("session.journal"), sid.clone()).unwrap();
    let store = ArtifactStore::open(root.join("artifacts")).unwrap();
    let credential = ScopeCredential::session(sid);
    let program =
        std::path::PathBuf::from(std::env::var("SystemRoot").unwrap()).join("System32/cmd.exe");
    let limits = LimitPolicy {
        max_bytes: 128,
        max_artifact_bytes: 65536,
        max_child_processes: 1,
        max_wall_time: Duration::from_secs(10),
        cancel_grace_period: Duration::from_millis(20),
        ..LimitPolicy::default()
    };
    let request = SandboxRequest::new("overflow", JobId::mint(), "Bash", limits)
        .with_workspace_view(view.view_id.clone())
        .with_capability(SandboxCapability::Execute {
            command: program.to_string_lossy().into_owned(),
        })
        .with_capability(SandboxCapability::Write {
            root: view.isolated_path.clone(),
        })
        .with_arg("/d")
        .with_arg("/c")
        .with_arg("overflow.cmd");
    let mut job = Job::spawn(
        &mut journal,
        request,
        JobKind::BackgroundShell {
            command: "bounded stream".into(),
        },
        ActorId::mint(),
        &program,
        view,
    )
    .unwrap();
    let started = std::time::Instant::now();
    while journal
        .snapshot()
        .element(&ElementId::new(format!("job-{}", job.id())).unwrap())
        .unwrap()
        .text
        .len()
        < 128
    {
        assert!(!job.poll(&mut journal, &store, &credential).unwrap());
        assert!(started.elapsed() < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(3));
    }
    job.signal(&mut journal, JobSignal::Terminate).unwrap();
    for _ in 0..20 {
        job.poll(&mut journal, &store, &credential).unwrap();
        std::thread::sleep(Duration::from_millis(3));
    }
    assert!(
        job.wait(&mut journal, &store, &credential, Duration::from_secs(5))
            .unwrap()
    );
    assert_eq!(job.status(journal.snapshot()).unwrap(), Status::Cancelled);
    assert!(matches!(
        job.state(journal.snapshot()).unwrap().termination_reason,
        Some(omp_runtime::JobTerminationReason::ForcedKill { .. })
    ));
    let id = ElementId::new(format!("job-{}", job.id())).unwrap();
    assert!(journal.snapshot().element(&id).unwrap().text.len() <= 128);
    assert!(journal.snapshot().children(&id).any(|e| e.kind == "diag"));
    let artifact = journal
        .snapshot()
        .children(journal.snapshot().container("artifacts"))
        .next()
        .unwrap();
    assert!(
        artifact.payload.as_ref().unwrap()["byte_length"]
            .as_u64()
            .unwrap()
            > 128
    );
    let offset = journal.snapshot().offset;
    assert!(
        job.wait(&mut journal, &store, &credential, Duration::ZERO)
            .unwrap()
    );
    assert_eq!(journal.snapshot().offset, offset);
    assert!(job.send(journal.snapshot(), b"ignored").is_err());
    assert_eq!(journal.snapshot().offset, offset);
    drop(job);
    let expected = journal.resume_latest();
    drop(journal);
    let reopened = Journal::open(root.join("session.journal")).unwrap();
    assert_eq!(reopened.snapshot(), &expected);
    drop(reopened);
    drop(store);
    fs::remove_dir_all(root).unwrap();
}
