use crate::{
    RestrictedProcess, WorkspaceView,
    artifact::{ArtifactOrigin, ArtifactScope, ArtifactStore, ScopeCredential},
    stream::BoundedStreamAccumulator,
};
use omp_state::{Journal, SessionSnapshot};
use omp_types::*;
use serde::{Deserialize, Serialize};
use std::{
    io::{Read, Write},
    path::Path,
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JobKind {
    ToolCall {
        name: String,
        tool_call_id: ToolCallId,
    },
    BackgroundShell {
        command: String,
    },
    Subagent {
        actor_id: ActorId,
    },
    DevServer {
        name: String,
        port: Option<u16>,
    },
    RemoteFunction {
        function_name: String,
    },
    RemoteExecution {
        target: String,
    },
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum JobSignal {
    Interrupt,
    Terminate,
    Kill,
    Hangup,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JobTerminationReason {
    Normal {
        exit_code: i32,
    },
    Cancelled {
        cooperative: bool,
    },
    ForcedKill {
        reason: String,
        elapsed_after_cancel_ms: u64,
    },
    Timeout {
        max_duration_ms: u64,
    },
    ResourceLimitExceeded {
        limit_name: String,
    },
    HostShutdown,
    ExecutionError(StructuredError),
}
#[derive(Debug, thiserror::Error)]
pub enum JobError {
    #[error(transparent)]
    State(#[from] omp_state::StateError),
    #[error(transparent)]
    Boundary(#[from] StructuredError),
    #[error("job is terminal")]
    AlreadyTerminal,
    #[error("job handle is missing; reconcile from the journal")]
    MissingHandle,
    #[error("artifact persistence failed: {0}")]
    Artifact(String),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobState {
    pub request: SandboxRequest,
    pub kind: JobKind,
    pub owner: ActorId,
    pub started_ms: u64,
    pub cancel_requested_ms: Option<u64>,
    /// Monotonic elapsed millis at cancel time, paired with
    /// `cancel_requested_ms` (wall). Grace expiry uses this, not the wall
    /// clock, so NTP steps cannot stretch or skip the kill deadline.
    /// `#[serde(default)]` keeps journals written by older binaries decodable.
    #[serde(default)]
    pub cancel_requested_elapsed_ms: Option<u64>,
    pub termination_reason: Option<JobTerminationReason>,
    pub workspace: WorkspaceView,
}
struct Chunk {
    stderr: bool,
    data: Vec<u8>,
}
/// Only OS handles/buffers live here. Lifecycle, owner, policy and request live in the DOM.
pub struct Job {
    id: JobId,
    element: ElementId,
    process: Option<RestrictedProcess>,
    output: Receiver<Result<Chunk, String>>,
    streams: BoundedStreamAccumulator,
    bytes_journaled: usize,
    eof: bool,
    lease: Option<crate::workspace::WorkspaceLease>,
    input: Option<mpsc::SyncSender<Vec<u8>>>,
    monotonic_start: Instant,
    last_journal_ms: u64,
}
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
fn boundary_err(code: &str, message: impl Into<String>) -> JobError {
    JobError::Boundary(StructuredError::new(code, message, false))
}
fn element_id(id: &JobId) -> Result<ElementId, JobError> {
    ElementId::new(format!("job-{id}")).map_err(JobError::from)
}
fn static_element(name: &str) -> Result<ElementId, JobError> {
    ElementId::new(name).map_err(JobError::from)
}
fn host_actor() -> Result<ActorId, JobError> {
    ActorId::new("host").map_err(JobError::from)
}
fn status_string(status: &Status) -> Result<String, JobError> {
    let value = serde_json::to_value(status)
        .map_err(|e| StructuredError::new("job_encoding", e.to_string(), false))
        .map_err(JobError::from)?;
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| boundary_err("job_encoding", "status value is not a string"))
}
fn encode_state(state: &JobState) -> Result<serde_json::Value, JobError> {
    encode_payload(state)
}
fn encode_payload(value: &impl serde::Serialize) -> Result<serde_json::Value, JobError> {
    serde_json::to_value(value).map_err(|e| boundary_err("job_encoding", e.to_string()))
}
fn transition(
    journal: &mut Journal,
    element: &ElementId,
    status: Status,
    state: JobState,
    extra: Vec<PatchOp>,
    reason: &str,
) -> Result<(), JobError> {
    let status_value = status_string(&status)?;
    let payload = encode_state(&state)?;
    let mut ops = vec![
        PatchOp::SetAttribute {
            element: element.clone(),
            name: "status".into(),
            value: TypedValue::String(status_value),
        },
        PatchOp::ReplacePayload {
            element: element.clone(),
            payload,
        },
    ];
    ops.extend(extra);
    journal.append_patch(Patch {
        base_offset: JournalOffset(journal.snapshot().offset),
        result_offset: journal.next_offset(),
        by: host_actor()?.into(),
        reason: reason.into(),
        ops,
    })?;
    Ok(())
}
/// Truncate a string to at most `max_bytes` bytes, flooring to a UTF-8 char boundary.
fn truncate_to_boundary(mut text: String, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text
}
impl Job {
    pub fn spawn(
        journal: &mut Journal,
        request: SandboxRequest,
        kind: JobKind,
        owner: ActorId,
        program: &Path,
        view: WorkspaceView,
    ) -> Result<Self, JobError> {
        request.validate()?;
        let fail = |message: &str| {
            JobError::Boundary(StructuredError::new("capability_denied", message, false))
        };
        if request.workspace_view_id.as_ref() != Some(&view.view_id)
            || view.session_id != journal.snapshot().session_id
        {
            return Err(fail("Workspace view does not belong to request/session"));
        }
        if !request
            .capabilities
            .iter()
            .any(|c| matches!(c,SandboxCapability::Execute{command} if Path::new(command)==program))
        {
            return Err(fail("Program is not declared in execution capability"));
        }
        for capability in &request.capabilities {
            match capability {
                SandboxCapability::Read { root } | SandboxCapability::Write { root }
                    if root == &view.isolated_path => {}
                SandboxCapability::Execute { command } if Path::new(command) == program => {}
                SandboxCapability::SpawnSubprocess => {}
                _ => {
                    return Err(fail(
                        "Capability is unavailable in restricted local backend",
                    ));
                }
            }
        }
        if journal.snapshot().active_jobs().count() >= request.limits.max_concurrent_jobs as usize {
            return Err(fail("Concurrent job budget exhausted"));
        }
        // This backend never inherits host env; `SandboxRequest::with_env`
        // is intentionally unsupported here (rejected, not silently dropped).
        if !request.env.is_empty() {
            return Err(fail(
                "Arbitrary host environment is not inherited; remove env or use a backend with EnvAccess grants",
            ));
        }
        if !request
            .capabilities
            .iter()
            .any(|cap| matches!(cap,SandboxCapability::Write{root} if root==&view.isolated_path))
        {
            return Err(fail(
                "This backend requires an explicit isolated-workspace write grant",
            ));
        }
        if !request
            .capabilities
            .iter()
            .any(|cap| matches!(cap, SandboxCapability::SpawnSubprocess))
            && request.limits.max_child_processes > 1
        {
            return Err(fail("Child process grant required for multiple processes"));
        }
        let lease = view.acquire()?;
        let id = request.job_id.clone();
        let element = element_id(&id)?;
        let started_ms = now_ms();
        let mut node = ElementSnapshot::new(element.clone(), "job");
        node.attributes
            .insert("status".into(), TypedValue::String("queued".into()));
        let state = JobState {
            request: request.clone(),
            kind,
            owner,
            started_ms,
            cancel_requested_ms: None,
            cancel_requested_elapsed_ms: None,
            termination_reason: None,
            workspace: view.clone(),
        };
        node.payload = Some(encode_state(&state)?);
        journal.append_patch(Patch {
            base_offset: JournalOffset(journal.snapshot().offset),
            result_offset: journal.next_offset(),
            by: host_actor()?.into(),
            reason: "queue bounded job".into(),
            ops: vec![PatchOp::Create {
                parent: static_element("jobs")?,
                index: journal.snapshot().active_jobs().count() as u32,
                element: node,
            }],
        })?;
        let mut process = match RestrictedProcess::spawn(
            program,
            &request.args,
            &view.isolated_path,
            &request.limits,
        ) {
            Ok(p) => p,
            Err(error) => {
                let mut failed = state;
                failed.termination_reason =
                    Some(JobTerminationReason::ExecutionError(error.clone()));
                transition(
                    journal,
                    &element,
                    Status::Failed,
                    failed,
                    vec![],
                    "sandbox spawn failed",
                )?;
                return Err(error.into());
            }
        };
        let (send, output) = mpsc::sync_channel(8);
        fn reader<R: Read + Send + 'static>(
            mut pipe: R,
            send: mpsc::SyncSender<Result<Chunk, String>>,
            stderr: bool,
        ) {
            thread::spawn(move || {
                let mut bytes = [0; 8192];
                loop {
                    match pipe.read(&mut bytes) {
                        Ok(0) => break,
                        Ok(n) => {
                            if send
                                .send(Ok(Chunk {
                                    stderr,
                                    data: bytes[..n].to_vec(),
                                }))
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = send.send(Err(e.to_string()));
                            break;
                        }
                    }
                }
            });
        }
        if let Some(pipe) = process.take_stdout() {
            reader(pipe, send.clone(), false);
        }
        if let Some(pipe) = process.take_stderr() {
            reader(pipe, send.clone(), true);
        }
        let (input, pending_input) = mpsc::sync_channel::<Vec<u8>>(4);
        if let Some(mut pipe) = process.stdin().take() {
            thread::spawn(move || {
                while let Ok(bytes) = pending_input.recv() {
                    if pipe.write_all(&bytes).is_err() {
                        break;
                    }
                }
            });
        }
        drop(send);
        transition(
            journal,
            &element,
            Status::Running,
            state,
            vec![],
            "sandbox running",
        )?;
        let monotonic_start = Instant::now();
        let mut job = Self {
            id,
            element,
            process: Some(process),
            output,
            streams: BoundedStreamAccumulator::new(&request.limits, true),
            bytes_journaled: 0,
            eof: false,
            lease: Some(lease),
            input: Some(input),
            monotonic_start,
            last_journal_ms: started_ms,
        };
        if let Some(stdin) = request.stdin {
            job.send(journal.snapshot(), stdin.as_bytes())?;
        }
        Ok(job)
    }
    pub fn id(&self) -> &JobId {
        &self.id
    }
    pub fn state(&self, snapshot: &SessionSnapshot) -> Result<JobState, JobError> {
        let payload = snapshot
            .element(&self.element)
            .and_then(|e| e.payload.clone())
            .ok_or(JobError::MissingHandle)?;
        serde_json::from_value(payload)
            .map_err(|e| StructuredError::new("invalid_job_state", e.to_string(), false).into())
    }
    pub fn status(&self, snapshot: &SessionSnapshot) -> Result<Status, JobError> {
        let value = snapshot
            .element(&self.element)
            .and_then(|e| e.attributes.get("status"));
        match value {
            // Accept the legacy "completed" alias via the canonical parser.
            Some(TypedValue::String(s)) => Status::from_dom_str(s).ok_or_else(|| {
                StructuredError::new(
                    "invalid_job_status",
                    format!("unknown job status '{s}'"),
                    false,
                )
                .into()
            }),
            _ => Err(JobError::MissingHandle),
        }
    }
    fn ensure_active(&self, snapshot: &SessionSnapshot) -> Result<(), JobError> {
        if self.status(snapshot)?.is_terminal_job() {
            Err(JobError::AlreadyTerminal)
        } else {
            Ok(())
        }
    }
    pub fn send(&mut self, snapshot: &SessionSnapshot, bytes: &[u8]) -> Result<(), JobError> {
        self.ensure_active(snapshot)?;
        if bytes.len() > 8192 {
            return Err(StructuredError::new(
                "stdin_limit",
                "Send at most 8192 bytes per request",
                true,
            )
            .into());
        }
        self.input
            .as_ref()
            .ok_or(JobError::MissingHandle)?
            .try_send(bytes.to_vec())
            .map_err(|_| {
                StructuredError::new("stdin_backpressure", "Input queue is full or closed", true)
            })?;
        Ok(())
    }
    pub fn signal(&mut self, journal: &mut Journal, signal: JobSignal) -> Result<(), JobError> {
        self.ensure_active(journal.snapshot())?;
        let mut state = self.state(journal.snapshot())?;
        if state.cancel_requested_ms.is_none() {
            state.cancel_requested_ms = Some(now_ms());
            state.cancel_requested_elapsed_ms = Some(self.monotonic_start.elapsed().as_millis() as u64);
            transition(
                journal,
                &self.element,
                Status::CancelRequested,
                state.clone(),
                vec![],
                "job cancellation requested",
            )?;
            self.input.take();
        }
        if matches!(signal, JobSignal::Kill) {
            self.process
                .as_mut()
                .ok_or(JobError::MissingHandle)?
                .kill()?;
            state.termination_reason = Some(JobTerminationReason::ForcedKill {
                reason: "explicit forced kill".into(),
                elapsed_after_cancel_ms: 0,
            });
            transition(
                journal,
                &self.element,
                Status::CancelRequested,
                state,
                vec![],
                "job boundary forcibly terminated",
            )?;
        }
        Ok(())
    }
    /// Drains a bounded number of chunks; slow clients cannot monopolize the host loop.
    pub fn poll(
        &mut self,
        journal: &mut Journal,
        store: &ArtifactStore,
        credential: &ScopeCredential,
    ) -> Result<bool, JobError> {
        if matches!(
            self.status(journal.snapshot())?,
            Status::Succeeded | Status::Failed | Status::Cancelled | Status::Finalized
        ) {
            return Ok(true);
        }
        let mut state = self.state(journal.snapshot())?;
        for _ in 0..8 {
            match self.output.try_recv() {
                Ok(Ok(chunk)) => {
                    if chunk.stderr {
                        self.streams.push_stderr(&chunk.data);
                    } else {
                        self.streams.push_stdout(&chunk.data);
                    }
                    if !self.streams.record_event() && state.cancel_requested_ms.is_none() {
                        self.signal(journal, JobSignal::Terminate)?;
                        state = self.state(journal.snapshot())?;
                        state.termination_reason =
                            Some(JobTerminationReason::ResourceLimitExceeded {
                                limit_name: "events".into(),
                            });
                        transition(
                            journal,
                            &self.element,
                            Status::CancelRequested,
                            state.clone(),
                            vec![],
                            "job event limit",
                        )?;
                    }
                }
                Ok(Err(error)) => {
                    state.termination_reason = Some(JobTerminationReason::ExecutionError(
                        StructuredError::new("pipe_read", error, false),
                    ));
                    self.process
                        .as_mut()
                        .ok_or(JobError::MissingHandle)?
                        .kill()?;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.eof = true;
                    break;
                }
            }
        }
        let stdout = self.streams.stdout_lossy();
        let stderr = self.streams.stderr_lossy();
        let total_bytes = stdout.len() + stderr.len();
        if total_bytes != self.bytes_journaled {
            const JOURNAL_MIN_INTERVAL_MS: u64 = 100;
            const JOURNAL_MIN_BYTE_DELTA: usize = 64 * 1024;
            let now_wall = now_ms();
            let byte_delta = total_bytes.saturating_sub(self.bytes_journaled);
            let interval_elapsed = now_wall.saturating_sub(self.last_journal_ms);
            let should_journal = self.bytes_journaled == 0
                || byte_delta >= JOURNAL_MIN_BYTE_DELTA
                || interval_elapsed >= JOURNAL_MIN_INTERVAL_MS;
            if should_journal {
                self.bytes_journaled = total_bytes;
                self.last_journal_ms = now_wall;
                // Streams stay separated in `BoundedStreamAccumulator` (and in
                // persisted artifacts); the journaled text node keeps a merged
                // view with an explicit stderr marker so failures stay visible.
                let mut combined = stdout;
                if !stderr.is_empty() {
                    combined.push_str("\n[stderr]\n");
                    combined.push_str(&stderr);
                }
                combined = truncate_to_boundary(combined, state.request.limits.max_bytes);
                transition(
                    journal,
                    &self.element,
                    self.status(journal.snapshot())?,
                    state.clone(),
                    vec![PatchOp::ReplaceText {
                        element: self.element.clone(),
                        text: combined,
                    }],
                    "job bounded output",
                )?;
            }
        }
        let monotonic_elapsed = self.monotonic_start.elapsed();
        if monotonic_elapsed >= state.request.limits.max_wall_time
            && state.cancel_requested_ms.is_none()
        {
            self.signal(journal, JobSignal::Terminate)?;
            state = self.state(journal.snapshot())?;
            state.termination_reason = Some(JobTerminationReason::Timeout {
                max_duration_ms: state.request.limits.max_wall_time.as_millis() as u64,
            });
            transition(
                journal,
                &self.element,
                Status::CancelRequested,
                state.clone(),
                vec![],
                "job wall timeout",
            )?;
        }
        // Grace expiry uses the monotonic clock: wall `now_ms()` is only for
        // display/diagnostics, never for the kill deadline. Jobs cancelled by
        // an older binary (no monotonic stamp) fall back to the wall timestamp.
        let elapsed_ms = self.monotonic_start.elapsed().as_millis() as u64;
        let grace_period_ms = state.request.limits.cancel_grace_period.as_millis() as u64;
        let (cancelled, grace_exceeded, elapsed_after_cancel_ms) = match (
            state.cancel_requested_elapsed_ms,
            state.cancel_requested_ms,
        ) {
            (Some(requested_elapsed), _) => {
                let after = elapsed_ms.saturating_sub(requested_elapsed);
                (true, after >= grace_period_ms, after)
            }
            (None, Some(requested_wall)) => {
                let after = now_ms().saturating_sub(requested_wall);
                (true, after >= grace_period_ms, after)
            }
            // Not cancelled: no grace deadline applies; fall through to the
            // exit-status handling below.
            (None, None) => (false, false, 0),
        };
        if cancelled
            && grace_exceeded
            && !matches!(
                state.termination_reason,
                Some(JobTerminationReason::ForcedKill { .. })
            )
            {
                self.process
                    .as_mut()
                    .ok_or(JobError::MissingHandle)?
                    .kill()?;
                state.termination_reason = Some(JobTerminationReason::ForcedKill {
                    reason: "cancellation grace expired".into(),
                    elapsed_after_cancel_ms,
                });
                transition(
                    journal,
                    &self.element,
                    Status::CancelRequested,
                    state.clone(),
                    vec![],
                    "forced_kill",
                )?;
            }
        let exit = self
            .process
            .as_mut()
            .ok_or(JobError::MissingHandle)?
            .try_wait()?;
        if let Some(code) = exit
            && self.eof {
                let artifacts = self
                    .streams
                    .persist_artifacts(
                        store,
                        ArtifactOrigin::Job(self.id.clone()),
                        ArtifactScope::Session,
                        credential,
                    )
                    .map_err(|e| JobError::Artifact(e.to_string()))?;
                let mut ops = vec![];
                let first = artifacts.first().map(|a| a.id.clone());
                let index_base = journal
                    .snapshot()
                    .children(journal.snapshot().container("artifacts"))
                    .count() as u32;
                let artifacts_parent = static_element("artifacts")?;
                for (offset, metadata) in artifacts.into_iter().enumerate() {
                    let mut node = ElementSnapshot::new(
                        ElementId::new(format!("artifact-{}", metadata.id))
                            .map_err(JobError::from)?,
                        "artifact",
                    );
                    node.payload = Some(encode_payload(&metadata)?);
                    ops.push(PatchOp::Create {
                        parent: artifacts_parent.clone(),
                        index: index_base + offset as u32,
                        element: node,
                    });
                }
                if self.streams.is_truncated() {
                    let mut diag = ElementSnapshot::new(ElementId::mint(), "diag");
                    diag.payload = Some(
                        serde_json::json!({"code":"truncated","artifact":first.map(|id|format!("artifact://{id}")),"fetchable":true,"limit":state.request.limits.max_bytes}),
                    );
                    ops.push(PatchOp::Create {
                        parent: self.element.clone(),
                        index: 0,
                        element: diag,
                    });
                }
                let status = if state.cancel_requested_ms.is_some() {
                    Status::Cancelled
                } else if code == 0 && state.termination_reason.is_none() {
                    Status::Succeeded
                } else {
                    Status::Failed
                };
                if state.termination_reason.is_none() {
                    state.termination_reason =
                        Some(JobTerminationReason::Normal { exit_code: code });
                }
                transition(journal, &self.element, status, state, ops, "job terminal")?;
                self.process.take();
                self.input.take();
                self.lease.take();
                return Ok(true);
            }
        Ok(false)
    }
    pub fn wait(
        &mut self,
        journal: &mut Journal,
        store: &ArtifactStore,
        credential: &ScopeCredential,
        timeout: Duration,
    ) -> Result<bool, JobError> {
        let start = Instant::now();
        loop {
            if self.poll(journal, store, credential)? {
                return Ok(true);
            }
            if start.elapsed() >= timeout {
                return Ok(false);
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        if let Some(process) = &mut self.process {
            let _ = process.kill();
        }
    }
}
