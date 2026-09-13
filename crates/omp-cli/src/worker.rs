use parking_lot::Mutex;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};
use std::time::{Duration, Instant};

use omp_control::SessionHost;
use omp_state::journal::JournalRecord;
use omp_state::{Journal, SessionSnapshot};
use omp_types::{
    ActorId, BranchId, ElementId, ElementSnapshot, JournalOffset, MAX_WIRE_BYTES, Patch, PatchOp,
    SessionId, Status, StructuredError, TypedValue,
};
use serde::{Deserialize, Serialize};

use crate::CliError;

// ---------------------------------------------------------------------------
// Wire Protocol Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerEvent {
    Snapshot {
        snapshot: SessionSnapshot,
    },
    SessionChanged {
        snapshot: SessionSnapshot,
        journal: PathBuf,
    },
    Patch {
        patch: Patch,
        branch: BranchId,
    },
    Settled {
        error: Option<StructuredError>,
    },
    /// Out-of-band failure that belongs to no submitted turn — a background
    /// job/child poll that could not be committed, for instance. Presented as
    /// a notice: unlike `Settled` it never claims an in-flight turn finished.
    Diagnostic {
        error: StructuredError,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerRequest {
    Submit { line: String },
    Shutdown { detach: bool },
}

// ---------------------------------------------------------------------------
// Platform Process Boundary (Kill-On-Close Job Object / Process Group)
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod platform {
    use std::io::{Error as IoError, Result as IoResult};
    use std::os::windows::io::AsRawHandle;
    use std::ptr::null;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject, TerminateJobObject,
    };

    pub struct ProcessBoundary {
        job: HANDLE,
    }

    unsafe impl Send for ProcessBoundary {}
    unsafe impl Sync for ProcessBoundary {}

    impl ProcessBoundary {
        pub fn new() -> IoResult<Self> {
            let job = unsafe { CreateJobObjectW(null(), null()) };
            if job.is_null() || job == INVALID_HANDLE_VALUE {
                return Err(IoError::last_os_error());
            }

            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

            let ok = unsafe {
                SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const _,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if ok == 0 {
                let err = IoError::last_os_error();
                unsafe { CloseHandle(job) };
                return Err(err);
            }

            Ok(Self { job })
        }

        pub fn assign(&mut self, child: &std::process::Child) -> IoResult<()> {
            if self.job.is_null() || self.job == INVALID_HANDLE_VALUE {
                return Err(IoError::other(
                    "invalid job object handle",
                ));
            }
            let handle = child.as_raw_handle() as HANDLE;
            let ok = unsafe { AssignProcessToJobObject(self.job, handle) };
            if ok == 0 {
                return Err(IoError::last_os_error());
            }
            Ok(())
        }

        pub fn terminate(&mut self) -> IoResult<()> {
            if !self.job.is_null() && self.job != INVALID_HANDLE_VALUE {
                let ok = unsafe { TerminateJobObject(self.job, 1) };
                if ok == 0 {
                    return Err(IoError::last_os_error());
                }
            }
            Ok(())
        }
    }

    impl Drop for ProcessBoundary {
        fn drop(&mut self) {
            if !self.job.is_null() && self.job != INVALID_HANDLE_VALUE {
                unsafe {
                    CloseHandle(self.job);
                }
            }
        }
    }
}

#[cfg(not(windows))]
mod platform {
    use std::io::Result as IoResult;

    pub struct ProcessBoundary {
        pgid: Option<u32>,
    }

    impl ProcessBoundary {
        pub fn new() -> IoResult<Self> {
            Ok(Self { pgid: None })
        }

        pub fn assign(&mut self, child: &std::process::Child) -> IoResult<()> {
            self.pgid = Some(child.id());
            Ok(())
        }

        pub fn terminate(&mut self) -> IoResult<()> {
            if let Some(pid) = self.pgid.take() {
                unsafe extern "C" {
                    fn kill(pid: i32, sig: i32) -> i32;
                }
                unsafe {
                    kill(-(pid as i32), 9);
                }
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Worker Helpers (Bounded Output Emitter & Diagnostic Persistence)
// ---------------------------------------------------------------------------

struct BoundedVecWriter {
    buf: Vec<u8>,
    limit: usize,
}

impl Write for BoundedVecWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        if self.buf.len() + data.len() > self.limit {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("WorkerEvent frame exceeded MAX_WIRE_BYTES ({})", self.limit),
            ));
        }
        self.buf.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn emit_event(event: &WorkerEvent) -> std::io::Result<()> {
    let mut writer = BoundedVecWriter {
        buf: Vec::with_capacity(1024),
        limit: MAX_WIRE_BYTES,
    };
    serde_json::to_writer(&mut writer, event)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    writer.buf.push(b'\n');
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&writer.buf)?;
    stdout.flush()
}

fn append_diagnostic_if_possible(journal: &mut Journal, error: &StructuredError, owner: &ActorId) {
    let parent = journal.snapshot().container("body").clone();
    let index = journal.snapshot().children(&parent).count() as u32;
    let mut diag = ElementSnapshot::new(ElementId::mint(), "diag");
    diag.text = error.message.clone();
    diag.attributes
        .insert("code".into(), TypedValue::String(error.code.clone()));
    diag.attributes
        .insert("retryable".into(), TypedValue::Bool(error.retryable));
    if let Some(d) = &error.diagnostics {
        diag.payload = Some(d.clone());
    }
    let patch = Patch {
        base_offset: JournalOffset(journal.snapshot().offset),
        result_offset: journal.next_offset(),
        by: owner.clone().into(),
        reason: "command error diagnostic".into(),
        ops: vec![PatchOp::Create {
            parent,
            index,
            element: diag,
        }],
    };
    let _ = journal.append_patch(patch);
}

/// Report a failure from the idle-time job/child polling loop.
///
/// The poll happens outside any submitted turn, so the failure is recorded in
/// the durable transcript when the journal still accepts writes and is always
/// surfaced to the parent as a `Diagnostic` notice. Swallowing it would leave
/// a job or subagent node stuck at `status=running` with no trace of why.
fn report_poll_failure(journal: &mut Journal, owner: &ActorId, error: StructuredError) {
    append_diagnostic_if_possible(journal, &error, owner);
    let _ = emit_event(&WorkerEvent::Diagnostic { error });
}

fn reconcile_orphans(journal: &mut Journal, owner: &ActorId) -> Result<(), omp_state::StateError> {
    let snapshot = journal.snapshot();
    let mut ops = Vec::new();

    // 1. Reconcile active jobs in <jobs> container
    for job in snapshot.active_jobs() {
        ops.push(PatchOp::SetAttribute {
            element: job.id.clone(),
            name: "status".into(),
            value: TypedValue::String("cancelled".into()),
        });
    }

    // 2. Reconcile only subagents in <actors> container (not peer actors/inspectors)
    for actor in snapshot.children(snapshot.container("actors")) {
        if actor.kind != "subagent" {
            continue;
        }
        let is_terminal = matches!(
            actor.attributes.get("status"),
            Some(TypedValue::String(s))
                if Status::from_dom_str(s).is_some_and(|status| status.is_terminal_session())
        );
        if !is_terminal {
            ops.push(PatchOp::SetAttribute {
                element: actor.id.clone(),
                name: "status".into(),
                value: TypedValue::String("cancelled".into()),
            });
        }
    }

    // 3. Reconcile live running tools in visible body (including async tools with partial results)
    for elem in snapshot.get_visible_body() {
        if elem.kind == "tool_call" {
            let is_running = matches!(
                elem.attributes.get("status"),
                Some(TypedValue::String(s))
                    if matches!(s.as_str(), "queued" | "running" | "cancel_requested")
            );
            let has_result = snapshot.children(&elem.id).any(|c| c.kind == "result");

            if is_running || !has_result {
                ops.push(PatchOp::SetAttribute {
                    element: elem.id.clone(),
                    name: "status".into(),
                    value: TypedValue::String("cancelled".into()),
                });

                if !has_result {
                    let mut result_node = ElementSnapshot::new(ElementId::mint(), "result");
                    result_node.text = "Tool call cancelled due to process termination".into();
                    let child_idx = snapshot.children(&elem.id).count() as u32;
                    ops.push(PatchOp::Create {
                        parent: elem.id.clone(),
                        index: child_idx,
                        element: result_node,
                    });
                }

                let mut diag_node = ElementSnapshot::new(ElementId::mint(), "diag");
                diag_node.text = "Tool execution cancelled by process restart".into();
                diag_node.attributes.insert(
                    "code".into(),
                    TypedValue::String("worker_restart_cancelled".into()),
                );
                let child_idx = snapshot.children(&elem.id).count() as u32;
                ops.push(PatchOp::Create {
                    parent: elem.id.clone(),
                    index: child_idx,
                    element: diag_node,
                });
            }
        } else if elem.kind == "assistant"
            || (elem.kind == "message"
                && matches!(
                    elem.attributes.get("role"),
                    Some(TypedValue::String(r)) if r == "assistant"
                ))
        {
            let is_running = matches!(
                elem.attributes.get("status"),
                Some(TypedValue::String(s))
                    if matches!(s.as_str(), "queued" | "running" | "cancel_requested")
            );
            let is_streaming = elem.attributes.get("streaming") == Some(&TypedValue::Bool(true));

            if is_running || is_streaming {
                ops.push(PatchOp::SetAttribute {
                    element: elem.id.clone(),
                    name: "status".into(),
                    value: TypedValue::String("cancelled".into()),
                });
                ops.push(PatchOp::SetAttribute {
                    element: elem.id.clone(),
                    name: "streaming".into(),
                    value: TypedValue::Bool(false),
                });

                let mut diag_node = ElementSnapshot::new(ElementId::mint(), "diag");
                diag_node.text = "Assistant inference interrupted by process restart".into();
                diag_node.attributes.insert(
                    "code".into(),
                    TypedValue::String("worker_restart_cancelled".into()),
                );
                let child_idx = snapshot.children(&elem.id).count() as u32;
                ops.push(PatchOp::Create {
                    parent: elem.id.clone(),
                    index: child_idx,
                    element: diag_node,
                });
            }
        }
    }

    if !ops.is_empty() {
        let patch = Patch {
            base_offset: JournalOffset(journal.snapshot().offset),
            result_offset: journal.next_offset(),
            by: owner.clone().into(),
            reason: "reconcile orphan jobs, tool calls, and assistant turns after restart".into(),
            ops,
        };
        journal.append_patch(patch)?;
    }
    Ok(())
}

fn observe_journal(journal: &mut Journal) {
    // 6. Wire up Journal observer: contiguous base emits Patch; differing base (e.g. fork) emits Snapshot.
    let current_base = Arc::new(AtomicU64::new(journal.snapshot().offset));
    let base_ref = current_base.clone();
    journal.set_observer(Some(Box::new(
        move |record: &JournalRecord, snapshot: &SessionSnapshot| {
            let expected = base_ref.load(Ordering::SeqCst);
            let event = if record.patch.base_offset.0 == expected {
                base_ref.store(record.offset, Ordering::SeqCst);
                WorkerEvent::Patch {
                    patch: record.patch.clone(),
                    branch: record.branch_id.clone(),
                }
            } else {
                base_ref.store(snapshot.offset, Ordering::SeqCst);
                WorkerEvent::Snapshot {
                    snapshot: snapshot.clone(),
                }
            };
            // Durable commit already completed. If stdout pipe is broken, fail-stop exit immediately.
            if let Err(err) = emit_event(&event) {
                eprintln!("Observer failed to emit event: {err}");
                std::process::exit(1);
            }
        },
    )));
}

// ---------------------------------------------------------------------------
// Child Subprocess Runner (`__ui-worker <workspace> <journal>`)
// ---------------------------------------------------------------------------

pub fn run_worker(workspace: &Path, journal_path: &Path) -> Result<(), CliError> {
    // 0. Await bounded startup handshake from parent ensuring process boundary assignment succeeded
    let stdin = std::io::stdin();
    let mut handshake_reader = stdin.lock();
    let mut handshake = String::new();
    let mut bounded_handshake = Read::take(&mut handshake_reader, 256);
    if bounded_handshake.read_line(&mut handshake)? == 0 || handshake.trim() != "__READY__" {
        return Err(CliError::InvalidArgument(
            "failed startup handshake: process boundary not assigned".into(),
        ));
    }
    drop(handshake_reader);

    // 1. Open or create Journal
    let mut journal = if journal_path.exists() {
        Journal::open(journal_path)?
    } else {
        if let Some(parent) = journal_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        Journal::create(journal_path, SessionId::mint())?
    };

    let mut host = SessionHost::new(workspace.to_path_buf(), ActorId::new("owner").unwrap())?;

    // 2. Recover damaged journal suffix if previous worker was forcibly killed mid-write
    if journal.recovery().is_some() {
        let diag = journal.recovery().cloned().unwrap();
        let preserved = journal.repair_suffix()?;
        let err = StructuredError::new(
            "journal_recovered",
            format!(
                "Repaired damaged journal suffix (last valid offset: {}, damaged bytes: {}). Preserved at: {:?}",
                diag.last_valid_offset, diag.damaged_bytes, preserved
            ),
            false,
        );
        append_diagnostic_if_possible(&mut journal, &err, &host.owner);
    }

    // 3. Reconcile orphan active jobs, subagents, and interrupted tool calls from previous worker
    reconcile_orphans(&mut journal, &host.owner)?;

    // 4. Restore convars and directors from authoritative DOM
    host.convars.hydrate_from_dom(journal.snapshot());
    host.command_engine.hydrate_from_dom(journal.snapshot());
    host.director_stack =
        omp_control::agent_loop::DirectorStack::from_session_snapshot(journal.snapshot())?;
    // Startup does NOT fetch catalog or refresh provider over network;
    // initial Snapshot is emitted and provider metadata is restored through run_turn or /provider refresh.

    // 5. Emit initial Snapshot
    emit_event(&WorkerEvent::Snapshot {
        snapshot: journal.snapshot().clone(),
    })?;

    observe_journal(&mut journal);

    // 7. Bounded channel for input requests from stdin thread (queue size 8)
    let (req_tx, req_rx) = sync_channel::<Result<WorkerRequest, StructuredError>>(8);

    // Stdin reading thread with bounded per-frame memory and fail-closed error propagation
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut reader = BufReader::new(stdin.lock());
        let mut line = String::new();
        loop {
            line.clear();
            let mut bounded = Read::take(&mut reader, (MAX_WIRE_BYTES + 1) as u64);
            match bounded.read_line(&mut line) {
                Ok(0) => break,
                Ok(n) => {
                    if n > MAX_WIRE_BYTES {
                        let _ = req_tx.send(Err(StructuredError::new(
                            "wire_frame_limit",
                            format!("Input frame exceeded MAX_WIRE_BYTES ({})", MAX_WIRE_BYTES),
                            false,
                        )));
                        break;
                    }
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<WorkerRequest>(trimmed) {
                        Ok(req) => {
                            if req_tx.send(Ok(req)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = req_tx.send(Err(StructuredError::new(
                                "wire_decode_error",
                                format!("Malformed worker request JSON: {}", e),
                                false,
                            )));
                            break;
                        }
                    }
                }
                Err(e) => {
                    let _ = req_tx.send(Err(StructuredError::new(
                        "wire_io_error",
                        format!("Stdin read error: {}", e),
                        false,
                    )));
                    break;
                }
            }
        }
    });

    // 8. Main worker loop: poll background jobs & children while idle, handle requests promptly
    loop {
        if let Err(error) = host.tool_host.poll_jobs(&mut journal) {
            report_poll_failure(&mut journal, &host.owner, error.into());
        }
        if let Err(error) = host.poll_children(&mut journal) {
            report_poll_failure(&mut journal, &host.owner, error);
        }

        match req_rx.recv_timeout(Duration::from_millis(25)) {
            Ok(Ok(WorkerRequest::Submit { line })) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    if emit_event(&WorkerEvent::Settled { error: None }).is_err() {
                        break;
                    }
                    continue;
                }

                // Explicitly check /detach submit
                if trimmed == "/detach" {
                    let active_jobs_count = journal.snapshot().active_jobs().count();
                    if active_jobs_count > 0 {
                        let err = StructuredError::new(
                            "detach_unsupported_with_active_jobs",
                            format!(
                                "Cannot detach session with {} active job(s); job survival without host is unsupported",
                                active_jobs_count
                            ),
                            false,
                        );
                        append_diagnostic_if_possible(&mut journal, &err, &host.owner);
                        let _ = emit_event(&WorkerEvent::Settled { error: Some(err) });
                        continue;
                    }
                    let _ = emit_event(&WorkerEvent::Settled { error: None });
                    break;
                }

                let result = if trimmed == "/new" {
                    match crate::create_clean_session(workspace, journal.snapshot()) {
                        Ok((next, next_host, path)) => {
                            journal = next;
                            host = next_host;
                            emit_event(&WorkerEvent::SessionChanged {
                                snapshot: journal.snapshot().clone(),
                                journal: path,
                            })?;
                            observe_journal(&mut journal);
                            host.initialize_provider(&mut journal)
                        }
                        Err(error) => Err(error.to_structured()),
                    }
                } else if trimmed == "/fork" {
                    Err(StructuredError::new(
                        "missing_offset",
                        "fork requires an integer offset: /fork <offset>",
                        false,
                    ))
                } else if let Some(offset_str) = trimmed.strip_prefix("/fork ") {
                    let offset_str = offset_str.trim();
                    match offset_str.parse::<u64>() {
                        Ok(offset) => match journal.fork_at(offset) {
                            Ok(_) => Ok(()),
                            Err(e) => Err(e.structured()),
                        },
                        Err(_) => Err(StructuredError::new(
                            "invalid_argument",
                            "fork requires valid integer offset",
                            false,
                        )),
                    }
                } else if let Some(cmd) = trimmed.strip_prefix('/') {
                    host.execute_command(&mut journal, cmd)
                } else {
                    host.run_turn(&mut journal, trimmed)
                };

                let settled_err = match result {
                    Ok(()) => None,
                    Err(err) => {
                        append_diagnostic_if_possible(&mut journal, &err, &host.owner);
                        Some(err)
                    }
                };

                if emit_event(&WorkerEvent::Settled { error: settled_err }).is_err() {
                    break;
                }
            }
            Ok(Ok(WorkerRequest::Shutdown { detach })) => {
                let active_jobs_count = journal.snapshot().active_jobs().count();
                if detach && active_jobs_count > 0 {
                    let err = StructuredError::new(
                        "detach_unsupported_with_active_jobs",
                        format!(
                            "Cannot detach session with {} active job(s); job survival without host is unsupported",
                            active_jobs_count
                        ),
                        false,
                    );
                    append_diagnostic_if_possible(&mut journal, &err, &host.owner);
                    let _ = emit_event(&WorkerEvent::Settled { error: Some(err) });
                    continue;
                }

                if !detach {
                    if let Err(err) = host.shutdown(&mut journal) {
                        append_diagnostic_if_possible(&mut journal, &err, &host.owner);
                        let _ = emit_event(&WorkerEvent::Settled { error: Some(err) });
                    } else {
                        let _ = emit_event(&WorkerEvent::Settled { error: None });
                    }
                } else {
                    let _ = emit_event(&WorkerEvent::Settled { error: None });
                }
                break;
            }
            Ok(Err(structured_err)) => {
                append_diagnostic_if_possible(&mut journal, &structured_err, &host.owner);
                let _ = emit_event(&WorkerEvent::Settled {
                    error: Some(structured_err),
                });
                break;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Continue idle polling
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // UI parent died or closed stdin pipe
                break;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Parent Worker Handle (Process Boundary & Channel Bridge)
// ---------------------------------------------------------------------------

const MAX_WORKER_STDERR_BYTES: usize = 16 * 1024;

#[derive(Default)]
struct WorkerStderr {
    bytes: Vec<u8>,
    truncated: bool,
}

pub struct Worker {
    workspace: PathBuf,
    journal: PathBuf,
    child: Child,
    stdin: ChildStdin,
    event_rx: Receiver<Result<WorkerEvent, StructuredError>>,
    stderr: Arc<Mutex<WorkerStderr>>,
    boundary: platform::ProcessBoundary,
    pending_error: Option<StructuredError>,
    running: bool,
}

type SpawnedChildBundle = (
    Child,
    ChildStdin,
    Receiver<Result<WorkerEvent, StructuredError>>,
    platform::ProcessBoundary,
    Arc<Mutex<WorkerStderr>>,
);

impl Worker {
    fn spawn_child(
        workspace: &Path,
        journal: &Path,
    ) -> Result<SpawnedChildBundle, CliError> {
        let current_exe = std::env::current_exe()?;
        let mut cmd = Command::new(current_exe);
        cmd.arg("__ui-worker")
            .arg(workspace)
            .arg(journal)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }

        let mut boundary = platform::ProcessBoundary::new()?;
        let mut child = cmd.spawn()?;
        if let Err(err) = boundary.assign(&child) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(CliError::Io(err));
        }

        let mut stderr = child.stderr.take().expect("worker stderr pipe");
        let captured = Arc::new(Mutex::new(WorkerStderr::default()));
        let capture = Arc::clone(&captured);
        std::thread::spawn(move || {
            let mut chunk = [0; 4096];
            while let Ok(count) = stderr.read(&mut chunk) {
                if count == 0 {
                    break;
                }
                let mut output = capture.lock();
                let keep = count.min(MAX_WORKER_STDERR_BYTES - output.bytes.len());
                output.bytes.extend_from_slice(&chunk[..keep]);
                output.truncated |= keep < count;
            }
        });

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| CliError::InvalidArgument("failed to capture worker stdin".into()))?;

        // Send startup handshake confirming process boundary assignment succeeded
        stdin.write_all(b"__READY__\n")?;
        stdin.flush()?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| CliError::InvalidArgument("failed to capture worker stdout".into()))?;

        // Bounded queue (8 frames) to avoid excessive memory usage
        let (tx, rx) = sync_channel::<Result<WorkerEvent, StructuredError>>(8);

        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                let mut bounded = Read::take(&mut reader, (MAX_WIRE_BYTES + 1) as u64);
                match bounded.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(n) => {
                        if n > MAX_WIRE_BYTES {
                            let _ = tx.send(Err(StructuredError::new(
                                "wire_frame_limit",
                                format!(
                                    "Worker stdout frame exceeded MAX_WIRE_BYTES ({})",
                                    MAX_WIRE_BYTES
                                ),
                                false,
                            )));
                            break;
                        }
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<WorkerEvent>(trimmed) {
                            Ok(event) => {
                                if tx.send(Ok(event)).is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(Err(StructuredError::new(
                                    "wire_decode_error",
                                    format!("Malformed worker stdout frame: {}", e),
                                    false,
                                )));
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(StructuredError::new(
                            "wire_io_error",
                            format!("Worker stdout read error: {}", e),
                            false,
                        )));
                        break;
                    }
                }
            }
        });

        Ok((child, stdin, rx, boundary, captured))
    }

    pub fn start(workspace: &Path, journal: &Path) -> Result<Self, CliError> {
        let (child, stdin, event_rx, boundary, stderr) = Self::spawn_child(workspace, journal)?;
        Ok(Self {
            workspace: workspace.to_path_buf(),
            journal: journal.to_path_buf(),
            child,
            stdin,
            event_rx,
            boundary,
            stderr,
            pending_error: None,
            running: true,
        })
    }

    pub fn submit(&mut self, line: &str) -> Result<(), CliError> {
        if !self.running {
            return Err(CliError::InvalidArgument("worker is not running".into()));
        }
        let req = WorkerRequest::Submit {
            line: line.to_string(),
        };
        let mut bytes = serde_json::to_vec(&req)?;
        bytes.push(b'\n');
        self.stdin.write_all(&bytes)?;
        self.stdin.flush()?;
        Ok(())
    }

    fn disconnected_error(&mut self) -> StructuredError {
        let status = self.child.try_wait().ok().flatten();
        let stderr = self.stderr.lock();
        let detail = omp_render::richtext::sanitize_text(&String::from_utf8_lossy(&stderr.bytes));
        let mut error = StructuredError::new(
            "worker_disconnected",
            format!(
                "Worker process disconnected{}{}; committed work is preserved",
                status
                    .map(|status| format!(" ({status})"))
                    .unwrap_or_default(),
                if detail.trim().is_empty() {
                    String::new()
                } else {
                    format!(": {}", detail.trim())
                },
            ),
            false,
        );
        error.diagnostics = Some(serde_json::json!({
            "exit_code": status.and_then(|status| status.code()),
            "stderr": detail,
            "stderr_truncated": stderr.truncated,
        }));
        error
    }

    pub fn poll(&mut self) -> Result<Vec<WorkerEvent>, CliError> {
        if let Some(err) = self.pending_error.take() {
            self.running = false;
            return Err(CliError::Host(err));
        }

        let mut events = Vec::new();
        while events.len() < 128 {
            match self.event_rx.try_recv() {
                Ok(Ok(event)) => {
                    if let WorkerEvent::SessionChanged { journal, .. } = &event {
                        self.journal = journal.clone();
                    }
                    events.push(event);
                }
                Ok(Err(structured_err)) => {
                    if events.is_empty() {
                        self.running = false;
                        return Err(CliError::Host(structured_err));
                    } else {
                        self.pending_error = Some(structured_err);
                        break;
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    if self.running {
                        self.running = false;
                        let error = self.disconnected_error();
                        if events.is_empty() {
                            return Err(CliError::Host(error));
                        }
                        self.pending_error = Some(error);
                    }
                    break;
                }
            }
        }
        Ok(events)
    }

    pub fn stop(&mut self, detach: bool) -> Result<(), CliError> {
        if !self.running {
            return Ok(());
        }
        let req = WorkerRequest::Shutdown { detach };
        let mut bytes = serde_json::to_vec(&req)?;
        bytes.push(b'\n');
        self.stdin.write_all(&bytes)?;
        self.stdin.flush()?;

        let deadline = Instant::now() + Duration::from_secs(3);
        let mut settled_error = None;
        while Instant::now() < deadline {
            while let Ok(event) = self.event_rx.try_recv() {
                match event {
                    Ok(WorkerEvent::Settled { error: Some(error) }) | Err(error) => {
                        settled_error = Some(error)
                    }
                    _ => {}
                }
            }
            if let Some(status) = self.child.try_wait()? {
                self.running = false;
                if let Some(error) = settled_error {
                    return Err(CliError::Host(error));
                }
                return if status.success() {
                    Ok(())
                } else {
                    Err(CliError::Host(StructuredError::new(
                        "worker_exit",
                        "Session worker exited unsuccessfully; the journal is preserved",
                        false,
                    )))
                };
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        self.boundary.terminate()?;
        self.child.wait()?;
        self.running = false;
        Err(CliError::Host(settled_error.unwrap_or_else(|| StructuredError::new("shutdown_timeout", "Session worker exceeded shutdown grace and was terminated; resume the journal to recover", true))))
    }

    pub fn interrupt(&mut self) -> Result<(), CliError> {
        // Kill process boundary immediately, propagate error if terminate fails, and reap child
        self.boundary.terminate().map_err(CliError::Io)?;
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.running = false;

        // Restart worker on the same journal and workspace. The old reader
        // threads are not drained explicitly: they observe EOF on the killed
        // child's pipes and exit on their own.
        let (child, stdin, event_rx, boundary, stderr) =
            Self::spawn_child(&self.workspace, &self.journal)?;
        self.child = child;
        self.stdin = stdin;
        self.event_rx = event_rx;
        self.boundary = boundary;
        self.stderr = stderr;
        self.pending_error = None;
        self.running = true;

        Ok(())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.boundary.terminate();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
