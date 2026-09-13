use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::json;
use sha2::{Digest, Sha256};

use omp_python::runtime::{PythonRuntime, PythonSession};
use omp_runtime::artifact::{
    ArtifactMetadata, ArtifactOrigin, ArtifactRetention, ArtifactScope, ArtifactStore,
    ScopeCredential,
};
use omp_runtime::job::{Job, JobKind};
use omp_runtime::workspace::{IsolationMode, WorkspaceView, WorkspaceViewId};
use omp_state::{Journal, SessionSnapshot};
use omp_types::{
    ActorId, ArtifactId, ElementId, ElementSnapshot, JournalOffset, LimitPolicy, Patch, PatchOp,
    ProtocolVersion, SUMMARIES_CONTAINER, SandboxCapability, SandboxRequest, SessionId,
    StructuredError, SummaryKind, SummaryNode, TypedValue, expand_covered,
};

use crate::autoqa::{AutoQaReport, ReportQualityFilter};
use crate::definition::{
    HostGateway, HostRequest, HostResponse, ToolCall,
    ToolDefinition, ToolDiagnostic, ToolError, ToolExecutionResult, ToolLimits,
};
use crate::dyn_discovery::DynamicTool;
use crate::edit::{EditOp, HashlineParser};
use crate::read::{
    ProjectionRegistry, ReadSelector, ResourceScheme,
};
use crate::roster::ToolRegistry;

/// Maximum bytes for Read output before truncation.
const MAX_READ_BYTES: usize = 100_000;

#[cfg(windows)]
mod bundled_shell {
    //! The busybox shell is embedded and checksum-pinned so the spawn path
    //! never depends on the build-machine layout (a compile-time
    //! `CARGO_MANIFEST_DIR` path breaks for any installed or relocated
    //! binary). Mirrors `omp_python::runtime::PythonRuntime::bundled()`.

    use crate::definition::ToolError;
    use omp_types::ElementId;
    use sha2::{Digest, Sha256};
    use std::path::PathBuf;

    /// SHA-256 of the embedded `assets/busybox.exe`.
    pub const BUSYBOX_SHA256: &str = "6e263d154d8548d1eb936f65d1d8312c80df31c45974e48d6335e4dcc0f4f34c";
    const BUSYBOX_BYTES: &[u8] = include_bytes!("../assets/busybox.exe");

    /// Materializes the embedded busybox into a versioned host cache
    /// directory, verifying the pinned checksum before first use.
    pub fn busybox_path() -> Result<PathBuf, ToolError> {
        let digest = hex::encode(Sha256::digest(BUSYBOX_BYTES));
        if digest != BUSYBOX_SHA256 {
            return Err(ToolError::Execution {
                message: "Bundled busybox checksum mismatch".into(),
                details: None,
            });
        }
        let root = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("omp2")
            .join(format!("busybox-{}", &digest[..16]));
        let target = root.join("busybox.exe");
        if target.is_file() {
            return Ok(target);
        }
        std::fs::create_dir_all(&root).map_err(|error| ToolError::Execution {
            message: format!("Failed to create busybox cache dir: {}", error),
            details: None,
        })?;
        // Install via temp file + rename so concurrent hosts never observe a
        // partially written shell.
        let staging = root.join(format!(".busybox-{}", ElementId::mint()));
        std::fs::write(&staging, BUSYBOX_BYTES).map_err(|error| ToolError::Execution {
            message: format!("Failed to write bundled busybox: {}", error),
            details: None,
        })?;
        if std::fs::rename(&staging, &target).is_err() {
            let _ = std::fs::remove_file(&staging);
            // Another host installed it concurrently; accept that copy.
            if !target.is_file() {
                return Err(ToolError::Execution {
                    message: "Failed to install bundled busybox".into(),
                    details: None,
                });
            }
        }
        Ok(target)
    }
}

/// Authoritative tool host providing execution boundary for all tools.
///
/// Implements the tool integration contract:
/// - `ToolHost::new(workspace: PathBuf, owner: ActorId) -> Result<Self, ToolError>`
/// - `execute_tool(&mut self, journal: &mut Journal, call: &ToolCall) -> Result<ToolExecutionResult, ToolError>`
///
/// Tools only describe semantic state and request host capabilities via a temporary
/// `HostGateway` holding `Mutex<&mut Journal>` and `Mutex<&mut ToolHost>`, so existing
/// `ToolExecutor` APIs stay intact and caches never become authority.
pub struct ToolHost {
    pub workspace: PathBuf,
    pub owner: ActorId,
    pub registry: Arc<ToolRegistry>,
    pub artifact_store: Arc<ArtifactStore>,
    pub workspace_view: WorkspaceView,
    pub python_runtime: Option<PythonRuntime>,
    pub python_session: Option<PythonSession>,
    pub projections: Arc<ProjectionRegistry>,
    jobs: BTreeMap<omp_types::JobId, Job>,
}

impl ToolHost {
    /// Creates a new ToolHost bound to a workspace directory and owning actor.
    pub fn new(workspace: PathBuf, owner: ActorId) -> Result<Self, ToolError> {
        let _ = fs::create_dir_all(&workspace);

        let artifacts_dir = workspace.join(".omp").join("artifacts");
        let _ = fs::create_dir_all(&artifacts_dir);
        let artifact_store =
            ArtifactStore::open(&artifacts_dir).map_err(|e| ToolError::Execution {
                message: format!("Failed to open artifact store: {}", e),
                details: None,
            })?;

        let session_id = SessionId::mint();
        let view_id = WorkspaceViewId::mint();
        let workspace_view = WorkspaceView::new(
            view_id,
            session_id,
            &workspace,
            &workspace,
            IsolationMode::CopyFallback,
        );

        let registry = Arc::new(ToolRegistry::new());
        let projections = Arc::new(ProjectionRegistry::new());

        Ok(Self {
            workspace,
            owner,
            registry,
            artifact_store: Arc::new(artifact_store),
            workspace_view,
            python_runtime: None,
            python_session: None,
            projections,
            jobs: BTreeMap::new(),
        })
    }

    pub fn poll_jobs(&mut self, journal: &mut Journal) -> Result<(), ToolError> {
        let credential = ScopeCredential::session(journal.snapshot().session_id.clone());
        let mut completed = Vec::new();
        for (id, job) in &mut self.jobs {
            if job.state(journal.snapshot()).is_err() {
                completed.push(id.clone());
                continue;
            }
            let element = ElementId::new(format!("job-{id}")).unwrap();
            let signal = journal
                .snapshot()
                .element(&element)
                .and_then(|node| node.attributes.get("requested_signal"))
                .cloned();
            if let Some(TypedValue::String(signal)) = signal {
                let signal = match signal.as_str() {
                    "kill" | "SIGKILL" => omp_runtime::JobSignal::Kill,
                    "interrupt" | "SIGINT" => omp_runtime::JobSignal::Interrupt,
                    _ => omp_runtime::JobSignal::Terminate,
                };
                job.signal(journal, signal).map_err(|error| {
                    ToolError::Structured(StructuredError::new(
                        "job_signal",
                        error.to_string(),
                        false,
                    ))
                })?;
                journal
                    .append_patch(Patch {
                        base_offset: JournalOffset(journal.snapshot().offset),
                        result_offset: journal.next_offset(),
                        by: self.owner.clone().into(),
                        reason: "consume job signal".into(),
                        ops: vec![PatchOp::RemoveAttribute {
                            element,
                            name: "requested_signal".into(),
                        }],
                    })
                    .map_err(|error| ToolError::Structured(error.structured()))?;
            }
            if job
                .poll(journal, &self.artifact_store, &credential)
                .map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?
            {
                completed.push(id.clone());
            }
        }
        for id in completed {
            if let Some(job) = self.jobs.remove(&id)
                && let Ok(state) = job.state(journal.snapshot()) {
                    self.settle_workspace(journal, &id, &state)?;
                }
        }
        Ok(())
    }

    fn settle_workspace(
        &self,
        journal: &mut Journal,
        id: &omp_types::JobId,
        state: &omp_runtime::job::JobState,
    ) -> Result<(), ToolError> {
        if matches!(state.kind, JobKind::BackgroundShell { .. }) {
            let state_file = state.workspace.isolated_path.join(".omp-shell-state");
            let cwd_file = state.workspace.isolated_path.join(".omp-shell-cwd");
            if state_file.is_file()
                && matches!(
                    state.termination_reason,
                    Some(omp_runtime::JobTerminationReason::Normal { .. })
                )
            {
                let mut bytes = Vec::new();
                File::open(&state_file)
                    .and_then(|file| file.take(60_001).read_to_end(&mut bytes))
                    .map_err(|error| ToolError::Execution {
                        message: error.to_string(),
                        details: None,
                    })?;
                if bytes.len() > 60_000 {
                    return Err(ToolError::Validation {
                        message: "Persisted shell state exceeds 60000 bytes".into(),
                        details: None,
                    });
                }
                let shell = String::from_utf8(bytes).map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?;
                let cwd = fs::read_to_string(&cwd_file).unwrap_or_default();
                let isolated = state
                    .workspace
                    .isolated_path
                    .to_string_lossy()
                    .replace('\\', "/");
                let relative = cwd
                    .trim()
                    .replace('\\', "/")
                    .strip_prefix(&isolated)
                    .unwrap_or("")
                    .trim_start_matches('/')
                    .to_owned();
                let exit_code = match state.termination_reason {
                    Some(omp_runtime::JobTerminationReason::Normal { exit_code }) => exit_code,
                    _ => 1,
                };
                journal
                    .append_patch(Patch {
                        base_offset: JournalOffset(journal.snapshot().offset),
                        result_offset: journal.next_offset(),
                        by: self.owner.clone().into(),
                        reason: "Persist interpreter state".into(),
                        ops: vec![PatchOp::SetAttribute {
                            element: journal.snapshot().container("meta").clone(),
                            name: "shell_state".into(),
                            value: TypedValue::Json(
                                json!({"source":shell,"cwd":relative,"exit_code":exit_code}),
                            ),
                        }],
                    })
                    .map_err(|error| ToolError::Execution {
                        message: error.to_string(),
                        details: None,
                    })?;
            }
            for file in [&state_file, &cwd_file] {
                if file.exists() {
                    fs::remove_file(file).map_err(|error| ToolError::Execution {
                        message: error.to_string(),
                        details: None,
                    })?;
                }
            }
        }
        let diff = state
            .workspace
            .compute_diff()
            .map_err(|error| ToolError::Execution {
                message: error.to_string(),
                details: None,
            })?;
        let node_id = ElementId::new(format!("job-{id}")).unwrap();
        let mut ops = vec![PatchOp::SetAttribute {
            element: node_id.clone(),
            name: "workspace_diff".into(),
            value: TypedValue::Json(
                serde_json::to_value(&diff).map_err(|error| ToolError::Execution {
                    message: format!("Failed to serialize workspace diff: {error}"),
                    details: None,
                })?,
            ),
        }];
        if matches!(
            state.termination_reason,
            Some(omp_runtime::JobTerminationReason::Normal { .. })
        ) {
            let error = state.workspace.apply_to_base().err();
            if let Some(error) = error {
                ops.push(PatchOp::SetAttribute {
                    element: node_id.clone(),
                    name: "merge_error".into(),
                    value: TypedValue::Json(serde_json::to_value(error).unwrap()),
                });
            }
        }
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason: "settle isolated workspace result".into(),
                ops,
            })
            .map_err(|error| ToolError::Execution {
                message: error.to_string(),
                details: None,
            })?;
        Ok(())
    }

    pub fn cancel_job(&mut self, journal: &mut Journal, id: &str) -> Result<(), ToolError> {
        let id = omp_types::JobId::new(id.strip_prefix("job-").unwrap_or(id)).map_err(|error| {
            ToolError::Validation {
                message: error.to_string(),
                details: None,
            }
        })?;
        let job = self
            .jobs
            .get_mut(&id)
            .ok_or_else(|| ToolError::NotFound(format!("Live job {id}")))?;
        job.signal(journal, omp_runtime::JobSignal::Terminate)
            .map_err(|error| ToolError::Execution {
                message: error.to_string(),
                details: None,
            })
    }

    pub fn shutdown_jobs(&mut self, journal: &mut Journal) -> Result<(), ToolError> {
        if let Some(mut worker) = self.python_session.take() {
            worker
                .close(journal)
                .map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?;
        }
        for job in self.jobs.values_mut() {
            job.signal(journal, omp_runtime::JobSignal::Kill)
                .map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?;
        }
        while !self.jobs.is_empty() {
            self.poll_jobs(journal)?;
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(())
    }

    /// Executes a tool call against the authoritative journal and host state.
    pub fn execute_tool(
        &mut self,
        journal: &mut Journal,
        call: &ToolCall,
    ) -> Result<ToolExecutionResult, ToolError> {
        self.poll_jobs(journal)?;
        // Resolve executor for tool
        let definition = if self.registry.is_permanent(&call.name) {
            self.registry
                .get_permanent(&call.name)
                .cloned()
                .ok_or_else(|| ToolError::NotFound(call.name.clone()))?
        } else if call.name == "dyn" {
            self.registry.dyn_tool().clone()
        } else {
            let active = journal
                .snapshot()
                .active_tool_roster()
                .find(|element| {
                    element.attributes.get("name") == Some(&TypedValue::String(call.name.clone()))
                })
                .ok_or_else(|| ToolError::NotFound(call.name.clone()))?;
            if let Some(TypedValue::String(worker)) = active.attributes.get("worker") {
                let live = self
                    .python_session
                    .as_ref()
                    .is_some_and(|session| session.element.as_str() == worker)
                    && journal
                        .snapshot()
                        .element(&ElementId::new(worker.clone()).map_err(ToolError::Structured)?)
                        .is_some_and(|node| {
                            node.attributes.get("status")
                                == Some(&TypedValue::String("running".into()))
                        });
                if !live {
                    return Err(ToolError::Structured(StructuredError::new(
                        "extension_unavailable",
                        "Reload the extension in a live worker before invoking this tool",
                        false,
                    )));
                }
            }
            self.registry
                .dynamic_definition(&call.name)
                .ok_or_else(|| {
                    ToolError::NotFound(format!("Execution handler unavailable for {}", call.name))
                })?
        };

        // Temporary HostGateway holding Mutex<&mut Journal> and Mutex<&mut ToolHost>
        let gateway = LiveHostGateway {
            journal: Mutex::new(journal),
            host: Mutex::new(self),
        };

        definition.execute.execute(call, &gateway)
    }

    /// Persist availability before publishing an executable handle. The DOM gates every call.
    pub fn register_dynamic(
        &mut self,
        journal: &mut Journal,
        definition: ToolDefinition,
    ) -> Result<(), ToolError> {
        if self.registry.is_permanent(&definition.name)
            || definition.name == "dyn"
            || !definition.name.contains('/')
        {
            return Err(ToolError::Validation {
                message: "Dynamic tools require a non-reserved namespace/action name".into(),
                details: None,
            });
        }
        if journal.snapshot().active_tool_roster().any(|element| {
            element.attributes.get("name") == Some(&TypedValue::String(definition.name.clone()))
        }) {
            return Err(ToolError::Validation {
                message: "Dynamic tool already active".into(),
                details: None,
            });
        }
        let mut node = ElementSnapshot::new(ElementId::mint(), "tool");
        node.attributes
            .insert("name".into(), TypedValue::String(definition.name.clone()));
        node.attributes.insert(
            "version".into(),
            TypedValue::String(definition.version.clone()),
        );
        node.attributes.insert(
            "description".into(),
            TypedValue::String(definition.component_projection.clone()),
        );
        node.payload = Some(definition.parameters.clone());
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason: "register dynamic tool".into(),
                ops: vec![PatchOp::Create {
                    parent: journal.snapshot().container("tools").clone(),
                    index: journal.snapshot().active_tool_roster().count() as u32,
                    element: node,
                }],
            })
            .map_err(|error| ToolError::Structured(error.structured()))?;
        self.registry.register_dynamic(definition);
        Ok(())
    }

    /// Handles an individual HostRequest dispatch.
    fn handle_request(
        &mut self,
        journal: &mut Journal,
        req: HostRequest,
    ) -> Result<HostResponse, ToolError> {
        match req {
            HostRequest::ReadResource {
                path,
                selector,
                raw,
            } => self.handle_read(journal, &path, selector.as_deref(), raw),
            HostRequest::WriteFile {
                path,
                content,
                atomic,
            } => self.handle_write(journal, &path, &content, atomic),
            HostRequest::EditFile {
                path,
                expected_tag,
                patch_text,
            } => self.handle_edit(journal, &path, expected_tag.as_deref(), &patch_text),
            HostRequest::ExecuteProcess {
                command,
                cwd,
                env,
                timeout_ms,
                pty,
                is_async,
                capability_demands,
            } => self.handle_execute_process(
                journal,
                &command,
                cwd.as_deref(),
                env,
                timeout_ms,
                pty,
                is_async,
                capability_demands,
            ),
            HostRequest::EvalCode {
                code,
                language,
                reset,
                timeout_ms,
            } => self.handle_eval(journal, &code, &language, reset, timeout_ms),
            HostRequest::SpawnAgent {
                context,
                tasks,
                isolated_workspace,
                convar_overrides,
            } => self.handle_spawn_agent(
                journal,
                &context,
                tasks,
                isolated_workspace,
                convar_overrides,
            ),
            HostRequest::ReportQa { report } => self.handle_report_qa(journal, report),
            HostRequest::DynLookup {
                query,
                action,
                help,
                args,
            } => self.handle_dyn_lookup(journal, query.as_deref(), action.as_deref(), help, args),
        }
    }

    // =========================================================================
    // 1. READ RESOURCE
    // =========================================================================
    fn handle_read(
        &mut self,
        journal: &mut Journal,
        raw_path: &str,
        selector_str: Option<&str>,
        raw: bool,
    ) -> Result<HostResponse, ToolError> {
        let (scheme, clean_target) = ResourceScheme::from_uri(raw_path);
        let selector = selector_str
            .map(ReadSelector::parse)
            .unwrap_or_else(|| ReadSelector {
                raw,
                ..Default::default()
            });

        match scheme {
            ResourceScheme::Summary => self.handle_summary_read(journal, clean_target, &selector),
            ResourceScheme::Artifact => {
                let artifact_id =
                    ArtifactId::new(clean_target).map_err(|e| ToolError::Validation {
                        message: format!("Invalid artifact ID: {}", e),
                        details: None,
                    })?;

                // Find artifact metadata in session DOM
                let artifact_meta = journal
                    .snapshot()
                    .children(journal.snapshot().container("artifacts"))
                    .find_map(|e| {
                        if let Some(payload) = &e.payload
                            && let Ok(meta) =
                                serde_json::from_value::<ArtifactMetadata>(payload.clone())
                                && meta.id == artifact_id {
                                    return Some(meta);
                                }
                        None
                    });

                let metadata = artifact_meta.ok_or_else(|| {
                    ToolError::NotFound(format!(
                        "Artifact is not visible on this branch: {artifact_id}"
                    ))
                })?;
                let bytes = self
                    .artifact_store
                    .read_bounded(
                        &metadata,
                        MAX_READ_BYTES,
                        &ScopeCredential::actor(
                            Some(journal.snapshot().session_id.clone()),
                            self.owner.clone(),
                        ),
                    )
                    .map_err(|error| ToolError::Execution {
                        message: error.to_string(),
                        details: None,
                    })?;
                let text = String::from_utf8_lossy(&bytes);

                let formatted = if let Some(lines_spec) = &selector.lines {
                    let lines: Vec<&str> = text.lines().collect();
                    let (selected, _) =
                        lines_spec.select_lines_bounded(&lines, usize::MAX, MAX_READ_BYTES);
                    if raw {
                        selected
                            .into_iter()
                            .map(|(_, s)| s)
                            .collect::<Vec<_>>()
                            .join("\n")
                    } else {
                        selected
                            .into_iter()
                            .map(|(n, s)| format!("{}:{}", n, s))
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                } else if raw {
                    text.to_string()
                } else {
                    format!("[artifact://{}#RAW]\n{}", artifact_id, text)
                };

                let mut resp = HostResponse::success(formatted);
                resp.output.truncated = metadata.byte_length > bytes.len();
                resp.artifacts.push(artifact_id);
                Ok(resp)
            }
            ResourceScheme::Agent | ResourceScheme::History => {
                let agent_id = clean_target;
                let snapshot = journal.snapshot();

                // Search in actors or jobs
                let actor_element = snapshot
                    .children(snapshot.container("actors"))
                    .chain(snapshot.children(snapshot.container("jobs")))
                    .find(|e| {
                        e.id.as_str() == agent_id
                            || e.attributes
                                .get("actor_id")
                                .map(|v| match v {
                                    TypedValue::String(s) => s.as_str() == agent_id,
                                    _ => false,
                                })
                                .unwrap_or(false)
                    });

                if let Some(elem) = actor_element {
                    let status = elem
                        .attributes
                        .get("status")
                        .map(|v| match v {
                            TypedValue::String(s) => s.clone(),
                            _ => "unknown".into(),
                        })
                        .unwrap_or_else(|| "unknown".into());

                    let mut out = format!("# Agent/Job {}\nStatus: {}\n", elem.id, status);
                    if let Some(payload) = &elem.payload {
                        out.push_str(&format!(
                            "\nPayload:\n{}\n",
                            serde_json::to_string_pretty(payload).unwrap_or_default()
                        ));
                    }
                    if !elem.text.is_empty() {
                        out.push_str(&format!("\nContent:\n{}\n", elem.text));
                    }
                    Ok(HostResponse::success(out))
                } else {
                    Err(ToolError::NotFound(format!(
                        "Agent/History not found: {}",
                        agent_id
                    )))
                }
            }
            ResourceScheme::Skill => {
                let skill_path =
                    self.resolve_file_path(&format!(".omp/skills/{}.md", clean_target))?;
                if skill_path.exists() {
                    let content =
                        fs::read_to_string(&skill_path).map_err(|e| ToolError::Execution {
                            message: format!("Failed to read skill {}: {}", clean_target, e),
                            details: None,
                        })?;
                    Ok(HostResponse::success(content))
                } else {
                    Err(ToolError::NotFound(format!("Skill {clean_target}")))
                }
            }
            ResourceScheme::Rule => {
                let rule_path =
                    self.resolve_file_path(&format!(".omp/rules/{}.md", clean_target))?;
                if rule_path.exists() {
                    let content =
                        fs::read_to_string(&rule_path).map_err(|e| ToolError::Execution {
                            message: format!("Failed to read rule {}: {}", clean_target, e),
                            details: None,
                        })?;
                    Ok(HostResponse::success(content))
                } else {
                    Err(ToolError::NotFound(format!("Rule {clean_target}")))
                }
            }
            ResourceScheme::Issue | ResourceScheme::Pr => {
                let parts: Vec<_> = clean_target.split('/').collect();
                if parts.len() != 3 || !parts[2].bytes().all(|byte| byte.is_ascii_digit()) {
                    return Err(ToolError::Validation {
                        message: "Use owner/repository/number for GitHub resources".into(),
                        details: None,
                    });
                }
                let kind = if matches!(scheme, ResourceScheme::Issue) {
                    "issues"
                } else {
                    "pulls"
                };
                crate::resource::fetch(
                    &format!(
                        "https://api.github.com/repos/{}/{}/{}/{}",
                        parts[0], parts[1], kind, parts[2]
                    ),
                    true,
                    true,
                )
            }
            ResourceScheme::Ssh => {
                crate::ssh::validate(clean_target)?;
                self.authorize_host_read(journal, raw_path)?;
                crate::ssh::read(clean_target, &selector)
            }
            ResourceScheme::Http | ResourceScheme::Https => {
                // Model-controlled network fetch is gated exactly like SSH
                // remote reads: link-local/metadata endpoints can probe the
                // cloud control plane or intranet, so require an approval.
                crate::resource::validate_public_url(raw_path)?;
                self.authorize_host_read(journal, raw_path)?;
                crate::resource::fetch(raw_path, false, raw)
            }
            ResourceScheme::Sqlite | ResourceScheme::Archive | ResourceScheme::Document => {
                let resolved = self.resolve_file_path(clean_target)?;
                if resolved
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("pdf"))
                {
                    let content = pdf_extract::extract_text(&resolved).map_err(|error| {
                        ToolError::Execution {
                            message: error.to_string(),
                            details: None,
                        }
                    })?;
                    return Ok(HostResponse::success(content));
                }
                let runtime = PythonRuntime::bundled().map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?;
                let root = self
                    .workspace
                    .canonicalize()
                    .map_err(|error| ToolError::Execution {
                        message: error.to_string(),
                        details: None,
                    })?;
                let relative =
                    resolved
                        .strip_prefix(&root)
                        .map_err(|error| ToolError::Validation {
                            message: error.to_string(),
                            details: None,
                        })?;
                let view = WorkspaceView::allocate(
                    &self.workspace,
                    WorkspaceView::default_views_root(&self.workspace),
                    journal.snapshot().session_id.clone(),
                )
                .map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?;
                let mut worker = PythonSession::spawn(
                    &runtime,
                    journal,
                    &view,
                    self.owner.clone(),
                    LimitPolicy::default().with_max_wall_time(Duration::from_secs(15)),
                )
                .map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?;
                let resource = serde_json::to_string(
                    &json!({"path":relative.to_string_lossy(),"selector":selector}),
                )
                .unwrap();
                let code = format!(
                    "import json\nRESOURCE = json.loads({})\n{}",
                    serde_json::to_string(&resource).unwrap(),
                    include_str!("resource_worker.py")
                );
                let result = worker.eval(journal, &code, false);
                worker
                    .close(journal)
                    .map_err(|error| ToolError::Execution {
                        message: error.to_string(),
                        details: None,
                    })?;
                let result = result.map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?;
                if result["status"] != "ok" {
                    return Err(ToolError::Execution {
                        message: result["error"].to_string(),
                        details: Some(result),
                    });
                }
                let value: serde_json::Value = serde_json::from_str(
                    result["output"].as_str().unwrap_or(""),
                )
                .map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?;
                let mut response = HostResponse::success(
                    value
                        .get("content")
                        .and_then(|value| value.as_str())
                        .map(str::to_owned)
                        .unwrap_or_else(|| serde_json::to_string_pretty(&value).unwrap()),
                );
                response.output.truncated = value["truncated"].as_bool().unwrap_or(false);
                response.output.payload = Some(value);
                Ok(response)
            }
            ResourceScheme::Custom(name) => {
                if name == "local" {
                    self.read_filesystem_resource(
                        journal,
                        &format!(".omp/local/{clean_target}"),
                        &selector,
                        raw,
                    )
                } else {
                    Err(ToolError::NotFound(format!(
                        "No projection registered for {name}://"
                    )))
                }
            }
            ResourceScheme::File => {
                self.read_filesystem_resource(journal, clean_target, &selector, raw)
            }
        }
    }

    /// Reads the compaction DAG.
    ///
    /// * `summary://` lists every node.
    /// * `summary://<id>` re-renders the transcript elements that node elides —
    ///   the originals, not the inventory, which is what makes compaction
    ///   lossless in practice.
    /// * `summary://<id>?q=<pattern>` (or `summary://?q=<pattern>` across the
    ///   whole DAG) searches the elided originals, so a model can decide what to
    ///   expand without paying for the expansion first.
    ///
    /// Line selectors (`summary://<id>:50-200`) apply to the rendered expansion.
    fn handle_summary_read(
        &self,
        journal: &Journal,
        target: &str,
        selector: &ReadSelector,
    ) -> Result<HostResponse, ToolError> {
        let snapshot = journal.snapshot();
        let mut nodes: BTreeMap<ElementId, SummaryNode> = BTreeMap::new();
        let mut order: Vec<ElementId> = Vec::new();
        let mut texts: BTreeMap<ElementId, String> = BTreeMap::new();
        let mut kinds: BTreeMap<ElementId, SummaryKind> = BTreeMap::new();
        for element in snapshot.children(snapshot.container(SUMMARIES_CONTAINER)) {
            let Ok(node) = SummaryNode::from_element(element) else {
                continue;
            };
            order.push(element.id.clone());
            texts.insert(element.id.clone(), element.text.clone());
            kinds.insert(
                element.id.clone(),
                SummaryNode::kind_of(element).unwrap_or(SummaryKind::Leaf),
            );
            nodes.insert(element.id.clone(), node);
        }
        if order.is_empty() {
            let present = snapshot
                .children(snapshot.container(SUMMARIES_CONTAINER))
                .count();
            return Ok(HostResponse::success(if present == 0 {
                "Nothing has been compacted in this session; the transcript is complete."
            } else {
                "This session has compaction nodes this build cannot decode; the transcript itself is unaffected."
            }));
        }

        let scope: Vec<ElementId> = if target.is_empty() {
            order.clone()
        } else {
            let id = ElementId::new(target).map_err(|_| {
                ToolError::Validation {
                    message: format!(
                        "invalid summary id {target:?}: expected summary://<id> as listed by summary://"
                    ),
                    details: None,
                }
            })?;
            if !nodes.contains_key(&id) {
                return Err(ToolError::NotFound(format!(
                    "No summary node {id} on this branch; run Read summary:// to list the DAG"
                )));
            }
            vec![id]
        };

        if let Some(pattern) = selector.query.as_deref().filter(|q| !q.is_empty()) {
            return Ok(HostResponse::success(render_summary_search(
                snapshot, &nodes, &scope, pattern,
            )));
        }

        if target.is_empty() {
            return Ok(HostResponse::success(render_summary_list(
                &nodes, &order, &texts, &kinds,
            )));
        }

        let rendered = render_summary_expansion(snapshot, &nodes, &scope[0]);
        let Some(lines) = &selector.lines else {
            return Ok(HostResponse::success(rendered));
        };
        let collected: Vec<&str> = rendered.lines().collect();
        let (selected, truncated) =
            lines.select_lines_bounded(&collected, 4_000, MAX_READ_BYTES);
        let mut out = String::new();
        for (number, line) in selected {
            out.push_str(&format!("{number}:{line}\n"));
        }
        if truncated {
            out.push_str("[...selector output truncated]\n");
        }
        Ok(HostResponse::success(out))
    }

    /// Reads a local filesystem file or directory.
    /// Remote credential use is authorized for one exact resource, on this branch.
    fn authorize_host_read(&self, journal: &mut Journal, resource: &str) -> Result<(), ToolError> {
        let description = format!("Read remote resource {resource}");
        let prior = journal
            .snapshot()
            .children(journal.snapshot().container("approvals"))
            .find(|node| {
                node.payload.as_ref().is_some_and(|payload| {
                    payload["action_description"] == description
                        && payload["requested_by"] == self.owner.as_str()
                }) && node.attributes.get("status") != Some(&TypedValue::String("consumed".into()))
            })
            .cloned();
        if let Some(node) = prior {
            if node.attributes.get("approved") == Some(&TypedValue::Bool(true)) {
                journal
                    .append_patch(Patch {
                        base_offset: JournalOffset(journal.snapshot().offset),
                        result_offset: journal.next_offset(),
                        by: self.owner.clone().into(),
                        reason: "consume one-shot remote read approval".into(),
                        ops: vec![PatchOp::SetAttribute {
                            element: node.id,
                            name: "status".into(),
                            value: TypedValue::String("consumed".into()),
                        }],
                    })
                    .map_err(|error| ToolError::Structured(error.structured()))?;
                return Ok(());
            }
            return Err(StructuredError::new(
                "approval_required",
                format!("Remote read awaits approval {}", node.id),
                true,
            )
            .into());
        }
        let mut node = ElementSnapshot::new(ElementId::mint(), "approval");
        let id = node.id.clone();
        node.attributes
            .insert("status".into(), TypedValue::String("pending".into()));
        node.payload = Some(
            json!({"request_id":id,"requested_by":self.owner,"action_description":description,"approved":null}),
        );
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason: "request remote credential read approval".into(),
                ops: vec![PatchOp::Create {
                    parent: journal.snapshot().container("approvals").clone(),
                    index: journal
                        .snapshot()
                        .children(journal.snapshot().container("approvals"))
                        .count() as u32,
                    element: node,
                }],
            })
            .map_err(|error| ToolError::Structured(error.structured()))?;
        Err(StructuredError::new(
            "approval_required",
            format!("Remote read awaits approval {id}"),
            true,
        )
        .into())
    }

    fn read_filesystem_resource(
        &self,
        journal: &mut Journal,
        clean_target: &str,
        selector: &ReadSelector,
        raw: bool,
    ) -> Result<HostResponse, ToolError> {
        let resolved = self.resolve_file_path(clean_target)?;

        if !resolved.exists() {
            return Err(ToolError::NotFound(format!(
                "Path not found: {}",
                clean_target
            )));
        }

        let meta = fs::metadata(&resolved).map_err(|e| ToolError::Execution {
            message: format!("Failed to get metadata for {}: {}", clean_target, e),
            details: None,
        })?;

        // 1. Directory listing
        if meta.is_dir() {
            let mut entries_text = format!("# Directory: {}\n", clean_target);
            let mut count = 0;
            if let Ok(entries) = fs::read_dir(&resolved) {
                let mut dir_entries = Vec::new();
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                    if dir_entries.len() >= 2000 {
                        break;
                    }
                    let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    dir_entries.push((name, is_dir, size));
                }
                dir_entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                count = dir_entries.len();
                for (name, is_dir, size) in dir_entries {
                    if is_dir {
                        entries_text.push_str(&format!("{}/\n", name));
                    } else {
                        entries_text.push_str(&format!("{} ({} B)\n", name, size));
                    }
                }
            }
            entries_text.push_str(&format!("# Total: {} entries\n", count));
            return Ok(HostResponse::success(entries_text));
        }
        let extension = resolved
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !raw
            && (selector.is_img
                || matches!(
                    extension.as_str(),
                    "svg" | "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp"
                ))
        {
            let (png, width, height) =
                crate::resource::preview_image(&resolved, extension == "svg")?;
            let metadata = self
                .artifact_store
                .store_bytes(
                    ArtifactId::mint(),
                    &png,
                    LimitPolicy::default().max_artifact_bytes,
                    "image/png",
                    ArtifactOrigin::System,
                    ArtifactRetention::Session,
                    ArtifactScope::Session,
                    &ScopeCredential::session(journal.snapshot().session_id.clone()),
                )
                .map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?;
            self.register_artifact(journal, &metadata)?;
            let mut response =
                HostResponse::success(format!("Image {width}×{height}: {}", metadata.uri()));
            response.output.payload = Some(
                json!({"image":{"src":metadata.uri(),"media_type":"image/png","width":width,"height":height},"resolved_path":clean_target}),
            );
            response.artifacts.push(metadata.id);
            return Ok(response);
        }
        if meta.len() > 1_000_000 {
            let mut response = crate::resource::text_file(
                &resolved,
                clean_target,
                selector.lines.as_ref(),
                raw,
                selector.conflicts_only,
            )?;
            if response.output.truncated
                && meta.len() <= LimitPolicy::default().max_artifact_bytes as u64
            {
                let mut source = File::open(&resolved).map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?;
                let metadata = self
                    .artifact_store
                    .store_stream(
                        ArtifactId::mint(),
                        &mut source,
                        LimitPolicy::default().max_artifact_bytes,
                        "text/plain",
                        ArtifactOrigin::System,
                        ArtifactRetention::Session,
                        ArtifactScope::Session,
                        &ScopeCredential::session(journal.snapshot().session_id.clone()),
                    )
                    .map_err(|error| ToolError::Execution {
                        message: error.to_string(),
                        details: None,
                    })?;
                self.register_artifact(journal, &metadata)?;
                response
                    .output
                    .content
                    .push_str(&format!("\nFull source: {}", metadata.uri()));
                response.artifacts.push(metadata.id);
            }
            return Ok(response);
        }

        // 2. Regular file reading
        let bytes = fs::read(&resolved).map_err(|e| ToolError::Execution {
            message: format!("Failed to read file {}: {}", clean_target, e),
            details: None,
        })?;

        // Compute 4-hex tag: uppercase hex of first 2 bytes of SHA256 of file content
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = hasher.finalize();
        let tag = format!("{:02X}{:02X}", hash[0], hash[1]);

        let text = String::from_utf8_lossy(&bytes);

        // Raw output mode
        if raw {
            if let Some(lines_spec) = &selector.lines {
                let lines: Vec<&str> = text.lines().collect();
                let (selected, truncated) =
                    lines_spec.select_lines_bounded(&lines, usize::MAX, MAX_READ_BYTES);
                let content = selected
                    .into_iter()
                    .map(|(_, s)| s)
                    .collect::<Vec<_>>()
                    .join("\n");
                let mut resp = HostResponse::success(content);
                if truncated {
                    resp.output.truncated = true;
                }
                return Ok(resp);
            }
            return Ok(HostResponse::success(text.to_string()));
        }

        // Selected range mode
        if let Some(lines_spec) = &selector.lines {
            let lines: Vec<&str> = text.lines().collect();
            let (selected, truncated) =
                lines_spec.select_lines_bounded(&lines, usize::MAX, MAX_READ_BYTES);
            let mut out = format!("[{}#{}]\n", clean_target, tag);
            for (line_num, line_str) in selected {
                out.push_str(&format!("{}:{}\n", line_num, line_str));
            }
            let mut resp = HostResponse::success(out);
            if truncated {
                resp.output.truncated = true;
            }
            return Ok(resp);
        }

        // Conflicts-only mode
        if selector.conflicts_only {
            let mut out = format!("[{}#{} conflicts]\n", clean_target, tag);
            let mut in_conflict = false;
            for (idx, line) in text.lines().enumerate() {
                let num = idx + 1;
                if line.starts_with("<<<<<<<")
                    || line.starts_with("=======")
                    || line.starts_with(">>>>>>>")
                {
                    in_conflict = true;
                    out.push_str(&format!("{}:{}\n", num, line));
                } else if in_conflict {
                    out.push_str(&format!("{}:{}\n", num, line));
                }
                if line.starts_with(">>>>>>>") {
                    in_conflict = false;
                }
            }
            return Ok(HostResponse::success(out));
        }

        // Parseable code structural summary mode if no selector and file has > 40 lines
        let lines: Vec<&str> = text.lines().collect();
        if is_code_file(clean_target) && lines.len() > 40 {
            let summary = build_structural_summary(clean_target, &tag, &lines);
            return Ok(HostResponse::success(summary));
        }

        // Standard line-numbered output with [PATH#TAG] header
        let mut out = format!("[{}#{}]\n", clean_target, tag);
        for (idx, line) in lines.iter().enumerate() {
            out.push_str(&format!("{}:{}\n", idx + 1, line));
        }
        Ok(HostResponse::success(out))
    }

    fn register_artifact(
        &self,
        journal: &mut Journal,
        metadata: &ArtifactMetadata,
    ) -> Result<(), ToolError> {
        let mut element = ElementSnapshot::new(ElementId::mint(), "artifact");
        element.payload =
            Some(
                serde_json::to_value(metadata).map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?,
            );
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason: "Preserve resource artifact".into(),
                ops: vec![PatchOp::Create {
                    parent: journal.snapshot().container("artifacts").clone(),
                    index: journal
                        .snapshot()
                        .children(journal.snapshot().container("artifacts"))
                        .count() as u32,
                    element,
                }],
            })
            .map_err(|error| ToolError::Execution {
                message: error.to_string(),
                details: None,
            })?;
        Ok(())
    }

    // =========================================================================
    // 2. WRITE FILE
    // =========================================================================
    fn handle_write(
        &self,
        journal: &mut Journal,
        path: &str,
        content: &str,
        atomic: bool,
    ) -> Result<HostResponse, ToolError> {
        let resolved = self.write_path(journal, path)?;

        if let Some(parent) = resolved.parent() {
            fs::create_dir_all(parent).map_err(|e| ToolError::Execution {
                message: format!("Failed to create parent directory: {}", e),
                details: None,
            })?;
        }

        let line_count = content.lines().count();
        let byte_count = content.len();

        if atomic {
            let tmp_filename = format!(
                ".tmp_write_{}_{}.tmp",
                resolved
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("file"),
                ElementId::mint()
            );
            let tmp_path = resolved
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(tmp_filename);

            let mut f = File::create(&tmp_path).map_err(|e| ToolError::Execution {
                message: format!("Failed to create temporary file for atomic write: {}", e),
                details: None,
            })?;
            f.write_all(content.as_bytes())
                .map_err(|e| ToolError::Execution {
                    message: format!("Failed to write content to temporary file: {}", e),
                    details: None,
                })?;
            f.sync_all().map_err(|e| ToolError::Execution {
                message: format!("Failed to sync temporary file: {}", e),
                details: None,
            })?;
            drop(f);

            fs::rename(&tmp_path, &resolved).map_err(|e| ToolError::Execution {
                message: format!("Failed to atomically rename file to {}: {}", path, e),
                details: None,
            })?;
        } else {
            fs::write(&resolved, content.as_bytes()).map_err(|e| ToolError::Execution {
                message: format!("Failed to write file {}: {}", path, e),
                details: None,
            })?;
        }

        // Journal the write event into session DOM
        let offset = journal.snapshot().offset;
        let patch = Patch {
            base_offset: JournalOffset(offset),
            result_offset: journal.next_offset(),
            by: self.owner.clone().into(),
            reason: format!("Write file: {}", path),
            ops: vec![],
        };
        journal
            .append_patch(patch)
            .map_err(|error| ToolError::Execution {
                message: error.to_string(),
                details: None,
            })?;

        Ok(HostResponse::success(format!(
            "Wrote {} bytes ({} lines) to {}",
            byte_count, line_count, path
        )))
    }

    // =========================================================================
    // 3. EDIT FILE
    // =========================================================================
    fn handle_edit(
        &self,
        journal: &mut Journal,
        path: &str,
        expected_tag: Option<&str>,
        patch_text: &str,
    ) -> Result<HostResponse, ToolError> {
        let resolved = self.write_path(journal, path)?;

        if !resolved.exists() {
            return Err(ToolError::Execution {
                message: format!("File to edit not found: {}", path),
                details: None,
            });
        }

        let bytes = fs::read(&resolved).map_err(|e| ToolError::Execution {
            message: format!("Failed to read file for edit {}: {}", path, e),
            details: None,
        })?;

        // Current 4-hex tag
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = hasher.finalize();
        let current_tag = format!("{:02X}{:02X}", hash[0], hash[1]);

        // Tag check: if expected_tag is provided, it must match
        if let Some(exp) = expected_tag
            && !exp.eq_ignore_ascii_case(&current_tag) {
                let diag = ToolDiagnostic::error(format!(
                    "Tag mismatch on {}: expected #{}, current is #{}. Re-read file before editing.",
                    path, exp, current_tag
                )).with_code("stale_tag");
                // CRITICAL: File is left unchanged!
                return Ok(HostResponse::success(format!(
                    "Edit aborted due to tag mismatch on {}",
                    path
                ))
                .with_diagnostics(vec![diag]));
            }

        // Parse edit sections
        let sections = HashlineParser::parse(patch_text)?;
        // The section must target the file being edited. A bare `ends_with`
        // would let a patch for a different file sharing a path suffix
        // (e.g. `a/foo.rs` vs `b/foo.rs`) pass the guard and then rewrite
        // this file below.
        let section = match sections
            .iter()
            .find(|s| section_targets_path(&s.path, path))
        {
            Some(section) => section,
            None => {
                return Ok(HostResponse::success(format!(
                    "Edit aborted: no section targets {}",
                    path
                ))
                .with_diagnostics(vec![ToolDiagnostic::error(format!(
                    "Patch contains no section targeting {}; file left unchanged",
                    path
                ))
                .with_code("conflict")]));
            }
        };

        let current_text = String::from_utf8_lossy(&bytes).to_string();
        let mut lines: Vec<String> = current_text.lines().map(|s| s.to_string()).collect();

        for op in &section.ops {
                match op {
                    EditOp::PutRange { start, end, body } => {
                        let s = *start;
                        let e = *end;
                        if s < 1 || e < s || s > lines.len() + 1 {
                            let diag = ToolDiagnostic::error(format!(
                                "PUT range {}.={} is invalid for file with {} lines",
                                s,
                                e,
                                lines.len()
                            ))
                            .with_code("conflict");
                            // Failed edit leaves file unchanged
                            return Ok(HostResponse::success("Edit failed due to invalid range")
                                .with_diagnostics(vec![diag]));
                        }
                        let end_idx = e.min(lines.len());
                        let start_idx = s - 1;
                        lines.splice(start_idx..end_idx, body.clone());
                    }
                    EditOp::PutBlock { start, body } => {
                        let s = *start;
                        if s < 1 || s > lines.len() {
                            let diag = ToolDiagnostic::error(format!(
                                "PUT block at line {} is out of bounds for file with {} lines",
                                s,
                                lines.len()
                            ))
                            .with_code("conflict");
                            return Ok(HostResponse::success(
                                "Edit failed due to invalid block start",
                            )
                            .with_diagnostics(vec![diag]));
                        }
                        let start_idx = s - 1;
                        let mut block_end = start_idx + 1;
                        while block_end < lines.len() && !lines[block_end].trim().is_empty() {
                            block_end += 1;
                        }
                        lines.splice(start_idx..block_end, body.clone());
                    }
                    EditOp::InsertBefore { line, body } => {
                        let l = *line;
                        if l < 1 || l > lines.len() + 1 {
                            let diag = ToolDiagnostic::error(format!(
                                "InsertBefore <{} is out of bounds for file with {} lines",
                                l,
                                lines.len()
                            ))
                            .with_code("conflict");
                            return Ok(HostResponse::success(
                                "Edit failed due to invalid insertion point",
                            )
                            .with_diagnostics(vec![diag]));
                        }
                        let idx = l - 1;
                        lines.splice(idx..idx, body.clone());
                    }
                    EditOp::InsertAfter { line, body } => {
                        let l = *line;
                        if l > lines.len() {
                            let diag = ToolDiagnostic::error(format!(
                                "InsertAfter >{} is out of bounds for file with {} lines",
                                l,
                                lines.len()
                            ))
                            .with_code("conflict");
                            return Ok(HostResponse::success(
                                "Edit failed due to invalid insertion point",
                            )
                            .with_diagnostics(vec![diag]));
                        }
                        lines.splice(l..l, body.clone());
                    }
                    EditOp::CutRange { start, end, .. } => {
                        let s = *start;
                        let e = *end;
                        if s < 1 || e < s || s > lines.len() {
                            let diag = ToolDiagnostic::error(format!(
                                "CUT range {}.={} is invalid for file with {} lines",
                                s,
                                e,
                                lines.len()
                            ))
                            .with_code("conflict");
                            return Ok(HostResponse::success("Edit failed due to invalid range")
                                .with_diagnostics(vec![diag]));
                        }
                        let end_idx = e.min(lines.len());
                        let start_idx = s - 1;
                        lines.drain(start_idx..end_idx);
                    }
                    EditOp::CutBlock { start, .. } => {
                        let s = *start;
                        if s < 1 || s > lines.len() {
                            let diag = ToolDiagnostic::error(format!(
                                "CUT block at line {} is out of bounds",
                                s
                            ))
                            .with_code("conflict");
                            return Ok(HostResponse::success(
                                "Edit failed due to invalid block start",
                            )
                            .with_diagnostics(vec![diag]));
                        }
                        let start_idx = s - 1;
                        let mut block_end = start_idx + 1;
                        while block_end < lines.len() && !lines[block_end].trim().is_empty() {
                            block_end += 1;
                        }
                        lines.drain(start_idx..block_end);
                    }
                    EditOp::RemoveFile => {
                        fs::remove_file(&resolved).map_err(|e| ToolError::Execution {
                            message: format!("Failed to delete file {}: {}", path, e),
                            details: None,
                        })?;
                        return Ok(HostResponse::success(format!("Deleted file {}", path)));
                    }
                    EditOp::MoveFile { destination } => {
                        let dest_resolved = self.write_path(journal, destination)?;
                        if let Some(parent) = dest_resolved.parent() {
                            let _ = fs::create_dir_all(parent);
                        }
                        fs::rename(&resolved, &dest_resolved).map_err(|e| {
                            ToolError::Execution {
                                message: format!("Failed to move file to {}: {}", destination, e),
                                details: None,
                            }
                        })?;
                        return Ok(HostResponse::success(format!(
                            "Moved {} to {}",
                            path, destination
                        )));
                    }
                    _ => {}
                }
        }

        // Atomic write of edited file
        let new_content = lines.join("\n")
            + if current_text.ends_with('\n') {
                "\n"
            } else {
                ""
            };
        let tmp_path = resolved
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!(
                ".tmp_edit_{}_{}.tmp",
                resolved
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("file"),
                ElementId::mint()
            ));
        let mut f = File::create(&tmp_path).map_err(|e| ToolError::Execution {
            message: format!("Failed to create temporary file for edit: {}", e),
            details: None,
        })?;
        f.write_all(new_content.as_bytes())
            .map_err(|e| ToolError::Execution {
                message: format!("Failed to write edited content: {}", e),
                details: None,
            })?;
        f.sync_all().map_err(|e| ToolError::Execution {
            message: format!("Failed to sync edited file: {}", e),
            details: None,
        })?;
        drop(f);
        fs::rename(&tmp_path, &resolved).map_err(|e| ToolError::Execution {
            message: format!("Failed to rename edited file: {}", e),
            details: None,
        })?;

        // Compute new tag
        let mut new_hasher = Sha256::new();
        new_hasher.update(new_content.as_bytes());
        let new_hash = new_hasher.finalize();
        let new_tag = format!("{:02X}{:02X}", new_hash[0], new_hash[1]);

        // Journal the edit
        let offset = journal.snapshot().offset;
        let patch = Patch {
            base_offset: JournalOffset(offset),
            result_offset: journal.next_offset(),
            by: self.owner.clone().into(),
            reason: format!("Edit file: {} (new tag: #{})", path, new_tag),
            ops: vec![],
        };
        journal
            .append_patch(patch)
            .map_err(|error| ToolError::Execution {
                message: error.to_string(),
                details: None,
            })?;
        Ok(HostResponse::success(format!(
            "Applied edit to {} (new tag: #{})",
            path, new_tag
        )))
    }

    // =========================================================================
    // 4. EXECUTE PROCESS (BASH via Runtime Job)
    // =========================================================================
    #[allow(clippy::too_many_arguments)]
    fn handle_execute_process(
        &mut self,
        journal: &mut Journal,
        command: &str,
        cwd: Option<&str>,
        env: BTreeMap<String, String>,
        timeout_ms: Option<u64>,
        pty: bool,
        is_async: bool,
        capability_demands: Vec<String>,
    ) -> Result<HostResponse, ToolError> {
        if pty {
            return Err(ToolError::CapabilityDenied {
                capability: "PTY is unavailable in the restricted process backend".into(),
            });
        }
        if !capability_demands.is_empty() {
            let mut approval = ElementSnapshot::new(ElementId::mint(), "approval");
            approval
                .attributes
                .insert("status".into(), TypedValue::String("pending".into()));
            approval.payload = Some(
                json!({"command":command,"capabilities":capability_demands,"owner":self.owner}),
            );
            journal
                .append_patch(Patch {
                    base_offset: JournalOffset(journal.snapshot().offset),
                    result_offset: journal.next_offset(),
                    by: self.owner.clone().into(),
                    reason: "shell capability approval required".into(),
                    ops: vec![PatchOp::Create {
                        parent: journal.snapshot().container("approvals").clone(),
                        index: journal
                            .snapshot()
                            .children(journal.snapshot().container("approvals"))
                            .count() as u32,
                        element: approval,
                    }],
                })
                .map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?;
            return Err(ToolError::CapabilityDenied {
                capability: "Sensitive shell operation awaits host approval".into(),
            });
        }
        self.write_path(journal, ".")?;
        #[cfg(windows)]
        let shell_prog = bundled_shell::busybox_path()?;
        #[cfg(not(windows))]
        let shell_prog = PathBuf::from("/bin/bash");
        if !shell_prog.is_file() {
            return Err(ToolError::NotFound("Bundled shell runtime".into()));
        }
        let quote = |value: &str| format!("'{}'", value.replace('\'', "'\\''"));
        let mut script = String::new();
        let persisted = journal
            .snapshot()
            .element(journal.snapshot().container("meta"))
            .and_then(|node| node.attributes.get("shell_state"))
            .and_then(|value| {
                if let TypedValue::Json(value) = value {
                    Some(value)
                } else {
                    None
                }
            });
        if let Some(source) = persisted.and_then(|value| value["source"].as_str()) {
            script.push_str(source);
            script.push('\n');
        }
        for (key, value) in &env {
            if key == "?" || key == "!" {
                continue;
            }
            if key.is_empty()
                || !key.bytes().enumerate().all(|(index, byte)| {
                    byte == b'_'
                        || byte.is_ascii_alphabetic()
                        || (index > 0 && byte.is_ascii_digit())
                })
            {
                return Err(ToolError::Validation {
                    message: "Invalid shell variable identifier".into(),
                    details: None,
                });
            }
            script.push_str(&format!("export {key}={}\n", quote(value)));
        }

        let working_dir = cwd
            .or_else(|| {
                persisted
                    .and_then(|value| value["cwd"].as_str())
                    .filter(|value| !value.is_empty())
            })
            .map(|d| self.resolve_file_path(d))
            .transpose()?
            .unwrap_or(self.resolve_file_path(".")?);
        if !working_dir.is_dir() {
            return Err(ToolError::NotFound("Shell working directory".into()));
        }
        let view = WorkspaceView::direct(&self.workspace, journal.snapshot().session_id.clone())
            .map_err(|error| ToolError::Execution {
                message: error.to_string(),
                details: None,
            })?;
        let root = self
            .workspace
            .canonicalize()
            .map_err(|error| ToolError::Execution {
                message: error.to_string(),
                details: None,
            })?;
        let relative = working_dir
            .strip_prefix(&root)
            .map_err(|error| ToolError::Validation {
                message: error.to_string(),
                details: None,
            })?;
        let relative = if relative.as_os_str().is_empty() {
            ".".into()
        } else {
            relative.to_string_lossy().replace('\\', "/")
        };
        let state_path = view
            .isolated_path
            .join(".omp-shell-state")
            .to_string_lossy()
            .replace('\\', "/");
        let cwd_path = view
            .isolated_path
            .join(".omp-shell-cwd")
            .to_string_lossy()
            .replace('\\', "/");
        let root_str = root.to_string_lossy();
        let ws_clean = root_str
            .strip_prefix(r"\\?\")
            .unwrap_or(&root_str)
            .to_string();
        let ws_posix = ws_clean.replace('\\', "/");
        let ws_win = ws_clean.replace('/', "\\");

        let view_str = view.isolated_path.to_string_lossy().to_string();
        let view_clean = view_str
            .strip_prefix(r"\\?\")
            .unwrap_or(&view_str)
            .to_string();
        let view_posix = view_clean.replace('\\', "/");
        let view_win = view_clean.replace('/', "\\");

        let cleanup = format!("set > {}; pwd > {}", quote(&state_path), quote(&cwd_path));
        script.push_str(&format!("trap {} EXIT\n", quote(&cleanup)));
        script.push_str(&format!("_OMP_VIEW_ROOT={}\n", quote(&view_posix)));
        script.push_str(&format!("_OMP_WORKSPACE={}\n", quote(&ws_posix)));
        script.push_str(&format!("_OMP_WORKSPACE_WIN={}\n", quote(&ws_win)));
        script.push_str(&format!("export PWD={}\n", quote(&ws_posix)));
        script.push_str("pwd() {\n");
        script.push_str("  local p=\"$(command pwd)\"\n");
        script.push_str("  case \"$p\" in\n");
        script.push_str(
            "    \"$_OMP_VIEW_ROOT\"*) echo \"$_OMP_WORKSPACE${p#$_OMP_VIEW_ROOT}\" ;;\n",
        );
        script.push_str("    *) echo \"$p\" ;;\n");
        script.push_str("  esac\n");
        script.push_str("}\n");
        script.push_str("cd() {\n");
        script.push_str("  local target=\"$1\"\n");
        script.push_str("  if [ -z \"$target\" ]; then\n");
        script.push_str("    command cd \"$_OMP_VIEW_ROOT\"\n");
        script.push_str("    return\n");
        script.push_str("  fi\n");
        script.push_str("  case \"$target\" in\n");
        script.push_str(
            "    \"$_OMP_WORKSPACE\"*) target=\"$_OMP_VIEW_ROOT${target#$_OMP_WORKSPACE}\" ;;\n",
        );
        script.push_str("    \"$_OMP_WORKSPACE_WIN\"*) target=\"$_OMP_VIEW_ROOT${target#$_OMP_WORKSPACE_WIN}\" ;;\n");
        script.push_str("  esac\n");
        script.push_str("  command cd \"$target\"\n");
        script.push_str("}\n");
        script.push_str(&format!("cd -- {} || exit\n", quote(&relative)));
        let previous_exit = persisted
            .and_then(|value| value["exit_code"].as_i64())
            .unwrap_or(0)
            .clamp(0, 255);
        script.push_str(&format!("(exit {previous_exit})\n"));
        script.push_str(command);
        #[cfg(windows)]
        let shell_args = vec!["sh".into(), "-c".into(), script];
        #[cfg(not(windows))]
        let shell_args = vec!["--noprofile".into(), "--norc".into(), "-c".into(), script];

        let globals = journal.snapshot().session_globals();
        let integer = |name: &str, fallback: u64| match globals.get(name) {
            Some(TypedValue::Integer(value)) if *value > 0 => *value as u64,
            _ => fallback,
        };
        let configured_ms = integer("tool_max_runtime_ms", 300_000);
        let max_wall_time = Duration::from_millis(
            timeout_ms
                .filter(|value| *value > 0)
                .unwrap_or(configured_ms)
                .min(configured_ms),
        );

        let job_id = omp_types::JobId::mint();
        let request = SandboxRequest {
            protocol_version: ProtocolVersion::CURRENT,
            request_id: format!("req-{}", ElementId::mint()),
            job_id: job_id.clone(),
            operation_name: "bash.execute".into(),
            workspace_view_id: Some(view.view_id.clone()),
            limits: LimitPolicy {
                max_wall_time,
                max_bytes: integer("tool_max_output_bytes", 100_000)
                    .min(omp_types::MAX_WIRE_BYTES as u64) as usize,
                max_concurrent_jobs: integer("job_max_concurrency", 4).min(32) as u32,
                ..LimitPolicy::default()
            },
            capabilities: vec![
                SandboxCapability::Execute {
                    command: shell_prog.to_string_lossy().to_string(),
                },
                SandboxCapability::Read {
                    root: view.isolated_path.clone(),
                },
                SandboxCapability::Write {
                    root: view.isolated_path.clone(),
                },
                SandboxCapability::SpawnSubprocess,
            ],
            args: shell_args,
            env: BTreeMap::new(),
            stdin: None,
            input_artifacts: vec![],
        };

        // Spawn through runtime Job boundary
        let mut job = match Job::spawn(
            journal,
            request,
            JobKind::BackgroundShell {
                command: command.to_string(),
            },
            self.owner.clone(),
            &shell_prog,
            view,
        ) {
            Ok(j) => j,
            Err(e) => {
                return Err(ToolError::Execution {
                    message: format!("Failed to spawn sandbox job: {}", e),
                    details: None,
                });
            }
        };

        if is_async {
            let id = job.id().clone();
            self.jobs.insert(id.clone(), job);
            return Ok(HostResponse::success(format!(
                "Background shell job started: {id}"
            )));
        }

        // Wait for execution completion
        let credential = ScopeCredential::session(journal.snapshot().session_id.clone());
        let wait_time =
            max_wall_time + LimitPolicy::default().cancel_grace_period + Duration::from_secs(5);
        let finished = job
            .wait(journal, &self.artifact_store, &credential, wait_time)
            .map_err(|error| ToolError::Execution {
                message: error.to_string(),
                details: None,
            })?;
        if !finished {
            job.signal(journal, omp_runtime::JobSignal::Kill)
                .map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?;
            self.jobs.insert(job.id().clone(), job);
            return Err(ToolError::Execution {
                message: "Job boundary has not drained after forced termination".into(),
                details: None,
            });
        }

        // Fetch state and output
        let state = job
            .state(journal.snapshot())
            .map_err(|e| ToolError::Execution {
                message: format!("Failed to read job state: {}", e),
                details: None,
            })?;
        self.settle_workspace(journal, job.id(), &state)?;

        let mut output = journal
            .snapshot()
            .element(&ElementId::new(format!("job-{}", job.id())).unwrap())
            .map(|e| e.text.clone())
            .unwrap_or_default();

        output = output
            .replace(&view_posix, &ws_posix)
            .replace(&view_win, &ws_win)
            .replace(&view_str, &ws_clean);

        if output.is_empty() {
            output = format!("Process exited (status: {:?})", state.termination_reason);
        }
        let mut resp = HostResponse::success(output);
        let job_element = ElementId::new(format!("job-{}", job.id())).unwrap();
        for element in journal
            .snapshot()
            .children(journal.snapshot().container("artifacts"))
        {
            if let Some(metadata) = element.payload.as_ref().and_then(|payload| {
                serde_json::from_value::<ArtifactMetadata>(payload.clone()).ok()
            })
                && matches!(&metadata.origin, ArtifactOrigin::Job(id) if id == job.id()) {
                    resp.output
                        .content
                        .push_str(&format!("\nartifact://{}", metadata.id));
                    resp.artifacts.push(metadata.id);
                }
        }
        resp.output.truncated = journal.snapshot().children(&job_element).any(|element| {
            element.kind == "diag"
                && element
                    .payload
                    .as_ref()
                    .is_some_and(|payload| payload["code"] == "truncated")
        });
        if let Some(error) = journal
            .snapshot()
            .element(&job_element)
            .and_then(|element| element.attributes.get("merge_error"))
        {
            resp.diagnostics.push(
                ToolDiagnostic::error(format!("Isolated changes were not applied: {error:?}"))
                    .with_code("workspace_conflict"),
            );
        }
        if let Some(reason) = &state.termination_reason {
            match reason {
                omp_runtime::JobTerminationReason::Normal { exit_code } => {
                    if *exit_code != 0 {
                        resp.diagnostics.push(
                            ToolDiagnostic::error(format!(
                                "Command exited with non-zero status: {}",
                                exit_code
                            ))
                            .with_code("process_exit"),
                        );
                    }
                }
                omp_runtime::JobTerminationReason::Timeout { .. } => {
                    resp.diagnostics
                        .push(ToolDiagnostic::error("Process timed out").with_code("timeout"));
                }
                omp_runtime::JobTerminationReason::ForcedKill { reason, .. } => {
                    resp.diagnostics.push(
                        ToolDiagnostic::error(format!("Process forcibly killed: {}", reason))
                            .with_code("forced_kill"),
                    );
                }
                omp_runtime::JobTerminationReason::Cancelled { .. } => {
                    resp.diagnostics.push(
                        ToolDiagnostic::error("Process was cancelled").with_code("cancelled"),
                    );
                }
                _ => {}
            }
        }

        Ok(resp)
    }

    // =========================================================================
    // 5. EVAL CODE (Python Runtime Worker)
    // =========================================================================
    fn handle_eval(
        &mut self,
        journal: &mut Journal,
        code: &str,
        language: &str,
        reset: bool,
        timeout_ms: Option<u64>,
    ) -> Result<HostResponse, ToolError> {
        if language != "py" && language != "python" {
            return Err(ToolError::Validation {
                message: format!(
                    "Only Python ('py') evaluation is supported currently, got '{}'",
                    language
                ),
                details: None,
            });
        }

        // Initialize bundled Python runtime if not already loaded
        if self.python_runtime.is_none() {
            let runtime = PythonRuntime::bundled().map_err(|e| ToolError::Execution {
                message: format!("Failed to initialize bundled Python runtime: {}", e),
                details: None,
            })?;
            self.python_runtime = Some(runtime);
        }

        let runtime = self.python_runtime.as_ref().unwrap();

        // Spawn PythonSession worker if not active or reset requested
        if self
            .python_session
            .as_ref()
            .is_some_and(|session| journal.snapshot().element(&session.element).is_none())
        {
            self.python_session.take();
        }
        if self.python_session.is_none() || reset {
            if let Some(mut old_sess) = self.python_session.take() {
                old_sess
                    .close(journal)
                    .map_err(|error| ToolError::Execution {
                        message: error.to_string(),
                        details: None,
                    })?;
                let obsolete: Vec<_> = journal
                    .snapshot()
                    .active_tool_roster()
                    .filter(|node| {
                        node.attributes.get("worker")
                            == Some(&TypedValue::String(old_sess.element.to_string()))
                    })
                    .map(|node| PatchOp::Delete {
                        element: node.id.clone(),
                    })
                    .collect();
                if !obsolete.is_empty() {
                    journal
                        .append_patch(Patch {
                            base_offset: JournalOffset(journal.snapshot().offset),
                            result_offset: journal.next_offset(),
                            by: self.owner.clone().into(),
                            reason: "invalidate reloaded worker tools".into(),
                            ops: obsolete,
                        })
                        .map_err(|error| ToolError::Structured(error.structured()))?;
                }
            }
            let view = WorkspaceView::allocate(
                &self.workspace,
                WorkspaceView::default_views_root(&self.workspace),
                journal.snapshot().session_id.clone(),
            )
            .map_err(|error| ToolError::Execution {
                message: error.to_string(),
                details: None,
            })?;
            let mut limits = LimitPolicy::default();
            if let Some(timeout) = timeout_ms.filter(|value| *value > 0) {
                limits.max_wall_time = Duration::from_millis(timeout).min(limits.max_wall_time);
            }
            let session = PythonSession::spawn(runtime, journal, &view, self.owner.clone(), limits)
                .map_err(|e| ToolError::Execution {
                    message: format!("Failed to spawn Python session: {}", e),
                    details: None,
                })?;
            self.python_session = Some(session);
        }

        let mut session = self.python_session.take().unwrap();
        let worker_id = session.element.clone();
        let result = session.eval_with_host(journal, code, reset, |journal, request| {
            self.handle_python_request(journal, &worker_id, request)
        });
        self.python_session = Some(session);
        let eval_result = result.map_err(ToolError::Structured)?;

        // Format result from JSON response
        let status = eval_result
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("ok");
        if status == "ok" {
            let output = eval_result
                .get("output")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let res_val = eval_result.get("result");
            let mut out = String::new();
            if !output.is_empty() {
                out.push_str(output);
            }
            if let Some(res) = res_val
                && !res.is_null() {
                    if !out.is_empty() && !out.ends_with('\n') {
                        out.push('\n');
                    }
                    out.push_str(&res.to_string());
                }
            if out.is_empty() {
                out = "None".to_string();
            }
            let mut response = HostResponse::success(out);
            response.output.truncated = eval_result
                .get("truncated")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            if response.output.truncated {
                response.diagnostics.push(
                    ToolDiagnostic::warning("Python output truncated at boundary")
                        .with_code("truncated"),
                );
            }
            Ok(response)
        } else {
            let err_obj = eval_result.get("error");
            let err_msg = err_obj
                .and_then(|o| o.get("message"))
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown Python error");
            let err_type = err_obj
                .and_then(|o| o.get("type"))
                .and_then(|v| v.as_str())
                .unwrap_or("RuntimeError");

            let diag = ToolDiagnostic::error(format!("{}: {}", err_type, err_msg));
            Ok(
                HostResponse::success(format!("Error: {}: {}", err_type, err_msg))
                    .with_diagnostics(vec![diag]),
            )
        }
    }

    fn handle_python_request(
        &mut self,
        journal: &mut Journal,
        worker_id: &ElementId,
        request: serde_json::Value,
    ) -> Result<serde_json::Value, StructuredError> {
        let invalid = |message: &str| StructuredError::new("sdk_request", message, false);
        match request["type"].as_str() {
            Some("query") => match request["query"].as_str() {
                Some("get_snapshot") => {
                    let snapshot = match request["params"]["offset"].as_u64() {
                        Some(offset) => journal
                            .materialize(offset)
                            .map_err(|error| error.structured())?,
                        None => journal.snapshot().clone(),
                    };
                    serde_json::to_value(snapshot).map_err(|error| invalid(&error.to_string()))
                }
                Some("get_convar") => {
                    let name = request["params"]["name"]
                        .as_str()
                        .ok_or_else(|| invalid("Convar name required"))?;
                    let value =
                        journal
                            .snapshot()
                            .session_globals()
                            .get(name)
                            .ok_or_else(|| {
                                invalid("Convar is not in the effective session snapshot")
                            })?;
                    Ok(match value {
                        TypedValue::Null => serde_json::Value::Null,
                        TypedValue::Bool(v) => json!(v),
                        TypedValue::Integer(v) => json!(v),
                        TypedValue::Number(v) => json!(v),
                        TypedValue::String(v) => json!(v),
                        TypedValue::Json(v) => v.clone(),
                    })
                }
                Some("get_artifact") => {
                    let id = request["params"]["id"]
                        .as_str()
                        .ok_or_else(|| invalid("Artifact id required"))?;
                    let response = self
                        .handle_read(journal, &format!("artifact://{id}"), None, true)
                        .map_err(|error| invalid(&error.to_string()))?;
                    Ok(json!({"content":response.output.content,"payload":response.output.payload}))
                }
                Some("dyn") => {
                    let params = &request["params"];
                    let response = self
                        .handle_dyn_lookup(
                            journal,
                            params["query"].as_str(),
                            params["action"].as_str(),
                            params["help"].as_bool().unwrap_or(false),
                            params.get("args").cloned(),
                        )
                        .map_err(|error| invalid(&error.to_string()))?;
                    Ok(json!({"content":response.output.content,"payload":response.output.payload}))
                }
                _ => Err(StructuredError::new(
                    "capability_denied",
                    "This worker query requires a host control capability",
                    false,
                )),
            },
            Some("patch") => {
                let ops: Vec<PatchOp> = serde_json::from_value(request["ops"].clone())
                    .map_err(|error| invalid(&error.to_string()))?;
                let mut staged = journal.snapshot().clone();
                for op in &ops {
                    let mut targets = Vec::new();
                    match op {
                        PatchOp::Create { parent, .. } => targets.push(parent),
                        PatchOp::Move {
                            element, parent, ..
                        } => {
                            targets.push(element);
                            targets.push(parent);
                        }
                        PatchOp::Delete { element }
                        | PatchOp::SetAttribute { element, .. }
                        | PatchOp::RemoveAttribute { element, .. }
                        | PatchOp::ReplaceText { element, .. }
                        | PatchOp::AppendText { element, .. }
                        | PatchOp::ReplacePayload { element, .. } => targets.push(element),
                    }
                    for target in targets {
                        let mut cursor = Some(target);
                        let mut allowed = false;
                        while let Some(id) = cursor {
                            if matches!(id.as_str(), "todo" | "views") {
                                allowed = true;
                                break;
                            }
                            cursor = staged.node(id).and_then(|node| node.parent.as_ref());
                        }
                        if !allowed
                            || (matches!(op, PatchOp::Delete { .. } | PatchOp::Move { .. })
                                && matches!(target.as_str(), "todo" | "views"))
                        {
                            return Err(StructuredError::new(
                                "capability_denied",
                                "Worker patches are limited to todo and presentation subtrees",
                                false,
                            ));
                        }
                    }
                    let patch = Patch {
                        base_offset: JournalOffset(staged.offset),
                        result_offset: JournalOffset(staged.offset + 1),
                        by: self.owner.clone().into(),
                        reason: "validate SDK patch".into(),
                        ops: vec![op.clone()],
                    };
                    omp_state::apply_patch(&mut staged, &patch)
                        .map_err(|error| error.structured())?;
                }
                journal
                    .append_patch(Patch {
                        base_offset: JournalOffset(journal.snapshot().offset),
                        result_offset: journal.next_offset(),
                        by: self.owner.clone().into(),
                        reason: request["reason"].as_str().unwrap_or("SDK patch").into(),
                        ops,
                    })
                    .map_err(|error| error.structured())?;
                Ok(json!({"offset":journal.snapshot().offset}))
            }
            Some("register") => {
                if request["directors"]
                    .as_array()
                    .is_some_and(|items| !items.is_empty())
                    || request["components"]
                        .as_array()
                        .is_some_and(|items| !items.is_empty())
                {
                    return Err(StructuredError::new(
                        "capability_denied",
                        "Custom Director and component execution requires a host registration capability",
                        false,
                    ));
                }
                let extension = request["extension_id"]
                    .as_str()
                    .ok_or_else(|| invalid("Extension identifier required"))?;
                let valid_name = |name: &str| {
                    !name.is_empty()
                        && name.len() <= 64
                        && name
                            .bytes()
                            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
                };
                if !valid_name(extension) {
                    return Err(invalid("Invalid extension namespace"));
                }
                let declarations: Vec<omp_python::protocol::ToolDeclaration> =
                    serde_json::from_value(request["tools"].clone())
                        .map_err(|error| invalid(&error.to_string()))?;
                if declarations.len() > 32 {
                    return Err(invalid("Extension declaration budget exceeded"));
                }
                let mut definitions = BTreeMap::new();
                let mut ops = Vec::new();
                let mut index = journal.snapshot().active_tool_roster().count();
                for node in journal.snapshot().active_tool_roster() {
                    if node.attributes.get("extension")
                        == Some(&TypedValue::String(extension.into()))
                    {
                        ops.push(PatchOp::Delete {
                            element: node.id.clone(),
                        });
                        index -= 1;
                    }
                }
                for declaration in declarations {
                    if !valid_name(&declaration.name)
                        || declaration.version.is_empty()
                        || declaration.parameter_schema["type"] != "object"
                    {
                        return Err(invalid(
                            "Tools require a valid action, version, and object parameter schema",
                        ));
                    }
                    let name = format!("{extension}/{}", declaration.name);
                    if definitions.contains_key(&name)
                        || journal.snapshot().active_tool_roster().any(|node| {
                            node.attributes.get("name") == Some(&TypedValue::String(name.clone()))
                                && node.attributes.get("extension")
                                    != Some(&TypedValue::String(extension.into()))
                        })
                    {
                        return Err(invalid("Conflicting dynamic tool declaration"));
                    }
                    let mut node = ElementSnapshot::new(ElementId::mint(), "tool");
                    node.attributes
                        .insert("name".into(), TypedValue::String(name.clone()));
                    node.attributes.insert(
                        "version".into(),
                        TypedValue::String(declaration.version.clone()),
                    );
                    node.attributes.insert(
                        "description".into(),
                        TypedValue::String(declaration.description.clone()),
                    );
                    node.attributes
                        .insert("worker".into(), TypedValue::String(worker_id.to_string()));
                    node.attributes
                        .insert("extension".into(), TypedValue::String(extension.into()));
                    node.payload = Some(declaration.parameter_schema.clone());
                    ops.push(PatchOp::Create {
                        parent: journal.snapshot().container("tools").clone(),
                        index: index as u32,
                        element: node,
                    });
                    index += 1;
                    let executor = Arc::new(PythonToolExecutor { name: name.clone() });
                    definitions.insert(
                        name.clone(),
                        ToolDefinition::new(
                            name,
                            declaration.version,
                            json!({"type":"string"}),
                            declaration.parameter_schema,
                            vec![],
                            ToolLimits::default(),
                            executor,
                            declaration.description,
                        ),
                    );
                }
                if !ops.is_empty() {
                    journal
                        .append_patch(Patch {
                            base_offset: JournalOffset(journal.snapshot().offset),
                            result_offset: journal.next_offset(),
                            by: self.owner.clone().into(),
                            reason: "publish extension declarations".into(),
                            ops,
                        })
                        .map_err(|error| error.structured())?;
                }
                for definition in definitions.into_values() {
                    self.registry.register_dynamic(definition);
                }
                Ok(json!({"registered":extension}))
            }
            Some("job") if request["operation"] == "execute_remote" => {
                let payload = &request["payload"];
                let source = payload["source_code"]
                    .as_str()
                    .ok_or_else(|| invalid("Remote source required"))?;
                let name = payload["function_name"]
                    .as_str()
                    .ok_or_else(|| invalid("Remote function name required"))?;
                let capabilities: Vec<String> =
                    serde_json::from_value(request["capabilities"].clone())
                        .map_err(|error| invalid(&error.to_string()))?;
                if capabilities
                    .iter()
                    .any(|value| !matches!(value.as_str(), "fs_read" | "fs_write"))
                {
                    return Err(StructuredError::new(
                        "capability_denied",
                        "Remote worker grants are limited to isolated filesystem reads/writes",
                        false,
                    ));
                }
                if hex::encode(Sha256::digest(source.as_bytes()))
                    != payload["source_hash"].as_str().unwrap_or("")
                {
                    return Err(invalid("Remote source hash mismatch"));
                }
                let runtime = PythonRuntime::bundled()?;
                let view = WorkspaceView::allocate(
                    &self.workspace,
                    WorkspaceView::default_views_root(&self.workspace),
                    journal.snapshot().session_id.clone(),
                )?;
                let limits = LimitPolicy {
                    max_wall_time: Duration::from_millis(
                        payload["timeout_ms"]
                            .as_u64()
                            .unwrap_or(30_000)
                            .clamp(1, 30_000),
                    ),
                    ..LimitPolicy::default()
                };
                let mut worker =
                    PythonSession::spawn(&runtime, journal, &view, self.owner.clone(), limits)?;
                let args = payload["arguments"].clone();
                let script = format!(
                    "import ast, json, sys\n_source = {}\n_name = {}\n_args = json.loads({})\n_caps = json.loads({})\n_tree = ast.parse(_source)\nif len(_tree.body) != 1 or not isinstance(_tree.body[0], ast.FunctionDef) or _tree.body[0].name != _name: raise ValueError('Remote source must contain exactly one named function')\n_tree.body[0].decorator_list = []\nfrom omp_sdk.remote import ASTValidator\n_validator = ASTValidator(set(_caps))\n_validator.visit(_tree)\nif _validator.errors: raise ValueError('; '.join(_validator.errors))\nfor _module in _validator.imported_modules:\n if _module not in ('math','json','re','statistics','decimal','fractions','itertools','functools','collections','datetime','string','hashlib'): raise ValueError('Unavailable remote import: ' + _module)\n __import__(_module)\ndef _audit(event, args):\n if event == 'open':\n  mode = args[1] if len(args) > 1 else 'r'\n  write = isinstance(mode, str) and any(flag in mode for flag in 'wax+')\n  if ('fs_write' if write else 'fs_read') not in _caps: raise PermissionError('Undeclared filesystem capability')\n if event.startswith(('socket.','subprocess.','os.system','ctypes.')): raise PermissionError('Undeclared system capability')\nsys.addaudithook(_audit)\n_scope = {{}}\nexec(compile(_tree, '<remote>', 'exec'), _scope)\n_remote_result = _scope[_name](*_args['args'], **_args['kwargs'])\n",
                    serde_json::to_string(source).unwrap(),
                    serde_json::to_string(name).unwrap(),
                    serde_json::to_string(&args.to_string()).unwrap(),
                    serde_json::to_string(&json!(capabilities).to_string()).unwrap()
                );
                let result = worker.eval(journal, &script, false).and_then(|value| {
                    if value["status"] != "ok" {
                        return Err(invalid(&value["error"].to_string()));
                    }
                    worker.eval(journal, "_remote_result", false)
                });
                let close = worker.close(journal);
                let result = result?;
                close?;
                if result["status"] != "ok" {
                    return Err(invalid(&result["error"].to_string()));
                }
                Ok(
                    json!({"result":result["result"],"output":result["output"],"diff":view.compute_diff()?,"workspace_view_id":view.view_id}),
                )
            }
            _ => Err(StructuredError::new(
                "capability_denied",
                "Worker operation is not granted",
                false,
            )),
        }
    }

    // =========================================================================
    // 6. SPAWN AGENT (Isolated Child Jobs)
    // =========================================================================
    fn handle_spawn_agent(
        &self,
        _journal: &mut Journal,
        _context: &str,
        _tasks: Vec<serde_json::Value>,
        _isolated_workspace: bool,
        _convar_overrides: BTreeMap<String, String>,
    ) -> Result<HostResponse, ToolError> {
        Err(ToolError::CapabilityDenied {
            capability: "Child inference must be dispatched by SessionHost".into(),
        })
    }

    // =========================================================================
    // 7. REPORT QA
    // =========================================================================
    fn handle_report_qa(
        &self,
        journal: &mut Journal,
        report_val: serde_json::Value,
    ) -> Result<HostResponse, ToolError> {
        let report: AutoQaReport =
            serde_json::from_value(report_val).map_err(|e| ToolError::Validation {
                message: format!("Malformed AutoQA report: {}", e),
                details: None,
            })?;

        // Filter and quality check
        let warning_diag = ReportQualityFilter::filter(&report)?;

        // Journal report to session DOM
        let offset = journal.snapshot().offset;
        let mut node = ElementSnapshot::new(ElementId::mint(), "qa-report");
        node.payload =
            Some(
                serde_json::to_value(&report).map_err(|error| ToolError::Execution {
                    message: error.to_string(),
                    details: None,
                })?,
            );
        node.attributes.insert(
            "verification".into(),
            TypedValue::String("unverified".into()),
        );
        node.attributes
            .insert("tool".into(), TypedValue::String(report.tool));
        node.attributes.insert(
            "severity".into(),
            TypedValue::String(format!("{:?}", report.severity)),
        );
        node.attributes
            .insert("summary".into(), TypedValue::String(report.input_summary));

        let patch = Patch {
            base_offset: JournalOffset(offset),
            result_offset: journal.next_offset(),
            by: self.owner.clone().into(),
            reason: "Recorded AutoQA defect report".into(),
            ops: vec![PatchOp::Create {
                parent: ElementId::new("meta").unwrap(),
                index: 0,
                element: node,
            }],
        };
        journal
            .append_patch(patch)
            .map_err(|error| ToolError::Execution {
                message: error.to_string(),
                details: None,
            })?;

        let mut resp = HostResponse::success("AutoQA report recorded as an unverified observation");
        if let Some(diag) = warning_diag {
            resp.diagnostics.push(diag);
        }
        Ok(resp)
    }

    // =========================================================================
    // 8. DYN LOOKUP
    // =========================================================================
    fn handle_dyn_lookup(
        &mut self,
        journal: &mut Journal,
        query: Option<&str>,
        action: Option<&str>,
        help: bool,
        args: Option<serde_json::Value>,
    ) -> Result<HostResponse, ToolError> {
        // Collect dynamic tools from selected active branch in Journal DOM
        // This ensures tools discovered on abandoned branches disappear on rewind
        let mut active_dyn_tools = Vec::new();

        for elem in journal.snapshot().active_tool_roster() {
            if let Some(TypedValue::String(name)) = elem.attributes.get("name")
                && !self.registry.is_permanent(name) && name != "dyn" {
                    let desc = elem
                        .attributes
                        .get("description")
                        .and_then(|v| match v {
                            TypedValue::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let (ns, act) = name.split_once('/').unwrap_or(("dyn", name.as_str()));
                    active_dyn_tools.push(DynamicTool {
                        namespace: ns.to_string(),
                        action: act.to_string(),
                        description: desc,
                        parameters: elem
                            .payload
                            .clone()
                            .unwrap_or_else(|| json!({ "type": "object" })),
                        returns: json!({ "type": "any" }),
                    });
                }
        }

        // Action help synthesis
        if let Some(act) = action {
            if let Some(target) = active_dyn_tools
                .iter()
                .find(|t| t.full_name() == act || t.action == act)
            {
                if help {
                    return Ok(HostResponse::success(target.synthesize_help()));
                }
                let definition = self
                    .registry
                    .dynamic_definition(&target.full_name())
                    .ok_or_else(|| {
                        ToolError::NotFound(format!(
                            "No execution handler registered for {}",
                            target.full_name()
                        ))
                    })?;
                let call = ToolCall::new(
                    omp_types::ToolCallId::mint(),
                    target.full_name(),
                    definition.version.clone(),
                    args.unwrap_or_else(|| json!({})),
                );
                let result = self.execute_tool(journal, &call)?;
                return Ok(HostResponse {
                    output: result.output,
                    diagnostics: result.diagnostics,
                    usage: result.usage,
                    artifacts: result.artifacts,
                });
            }
            return Err(ToolError::NotFound(format!(
                "Dynamic tool not found: {}",
                act
            )));
        }

        // Query search
        if let Some(q) = query {
            let q_lower = q.to_lowercase();
            let matches: Vec<_> = active_dyn_tools
                .into_iter()
                .filter(|t| {
                    t.namespace.to_lowercase().contains(&q_lower)
                        || t.action.to_lowercase().contains(&q_lower)
                        || t.description.to_lowercase().contains(&q_lower)
                })
                .collect();
            let mut out = format!(
                "Found {} dynamic tool(s) on active branch matching '{}':\n",
                matches.len(),
                q
            );
            for m in matches.into_iter().take(50) {
                out.push_str(&format!("  - {}: {}\n", m.full_name(), m.description));
            }
            return Ok(HostResponse::success(out));
        }

        // Default listing
        let mut out = format!(
            "Dynamic tools available on active branch ({}):\n",
            active_dyn_tools.len()
        );
        for t in active_dyn_tools.into_iter().take(100) {
            out.push_str(&format!("  - {}: {}\n", t.full_name(), t.description));
        }
        Ok(HostResponse::success(out))
    }

    // =========================================================================
    // Helper: Path resolution with ~ expansion and unique workspace suffix recovery
    // =========================================================================
    pub fn resolve_file_path(&self, raw: &str) -> Result<PathBuf, ToolError> {
        let path = self.direct_path(raw)?;
        if path.exists() {
            return Ok(path);
        }
        let relative = Path::new(raw);
        if !relative.is_absolute() && relative.components().count() > 1 {
            let mut matches = Vec::new();
            find_files_with_suffix(&self.workspace, &raw.replace('\\', "/"), &mut matches, 2);
            match matches.len() {
                1 => return self.checked_path(matches.remove(0)),
                2.. => {
                    return Err(ToolError::Validation {
                        message: format!("Ambiguous workspace suffix: {raw}"),
                        details: None,
                    });
                }
                _ => {}
            }
        }
        Ok(path)
    }

    fn direct_path(&self, raw: &str) -> Result<PathBuf, ToolError> {
        let mut raw = raw.trim();
        if raw.is_empty() || raw.contains('\0') {
            return Err(ToolError::Validation {
                message: "Invalid filesystem path".into(),
                details: None,
            });
        }
        if let Some(stripped) = raw.strip_prefix(r"\\?\") {
            raw = stripped;
        }
        if (raw.starts_with('/') || raw.starts_with('\\'))
            && raw.len() >= 3
            && raw.as_bytes()[1].is_ascii_alphabetic()
            && raw.as_bytes()[2] == b':'
        {
            raw = &raw[1..];
        }

        let views_prefix_posix = "omp2-workspaces/";
        let views_prefix_win = "omp2-workspaces\\";
        if let Some(idx) = raw
            .find(views_prefix_posix)
            .or_else(|| raw.find(views_prefix_win))
        {
            let after = &raw[idx + views_prefix_posix.len()..];
            if let Some(slash) = after.find('/').or_else(|| after.find('\\')) {
                let subpath = &after[slash + 1..];
                return self.checked_path(self.workspace.join(subpath));
            } else {
                return self.checked_path(self.workspace.clone());
            }
        }

        let path = if raw == "~" {
            home_dir()
        } else if let Some(suffix) = raw.strip_prefix("~/").or_else(|| raw.strip_prefix("~\\")) {
            home_dir().join(suffix)
        } else {
            let direct_joined = self.workspace.join(raw);
            if direct_joined.exists() {
                direct_joined
            } else if raw.starts_with('/') || raw.starts_with('\\') {
                let trimmed = raw.trim_start_matches(['/', '\\']);
                let trimmed_joined = self.workspace.join(trimmed);
                if trimmed_joined.exists() {
                    trimmed_joined
                } else {
                    direct_joined
                }
            } else {
                direct_joined
            }
        };
        self.checked_path(path)
    }

    fn checked_path(&self, path: PathBuf) -> Result<PathBuf, ToolError> {
        let root = self
            .workspace
            .canonicalize()
            .map_err(|error| ToolError::Execution {
                message: error.to_string(),
                details: None,
            })?;
        let mut ancestor = path.as_path();
        let mut missing = Vec::new();
        while !ancestor.exists() {
            let name = ancestor.file_name().ok_or_else(|| ToolError::Validation {
                message: "Invalid path ancestor".into(),
                details: None,
            })?;
            missing.push(name.to_owned());
            ancestor = ancestor.parent().ok_or_else(|| ToolError::Validation {
                message: "Invalid path root".into(),
                details: None,
            })?;
        }
        let mut resolved = ancestor
            .canonicalize()
            .map_err(|error| ToolError::Execution {
                message: error.to_string(),
                details: None,
            })?;
        for name in missing.into_iter().rev() {
            resolved.push(name);
        }
        let starts_with_root = resolved.starts_with(&root);
        #[cfg(windows)]
        let starts_with_root = starts_with_root || {
            let res_str = resolved.to_string_lossy();
            let root_str = root.to_string_lossy();
            let res_clean = res_str.strip_prefix(r"\\?\").unwrap_or(&res_str);
            let root_clean = root_str.strip_prefix(r"\\?\").unwrap_or(&root_str);
            let res_lower = res_clean.to_lowercase();
            let root_lower = root_clean.trim_end_matches(['\\', '/']).to_lowercase();
            res_lower == root_lower
                || res_lower
                    .strip_prefix(&root_lower)
                    .is_some_and(|suffix| suffix.starts_with('\\') || suffix.starts_with('/'))
        };
        if !starts_with_root {
            return Err(ToolError::Validation {
                message: "Filesystem capability excludes paths outside workspace".into(),
                details: None,
            });
        }
        Ok(resolved)
    }

    fn write_path(&self, journal: &Journal, raw: &str) -> Result<PathBuf, ToolError> {
        let path = self.direct_path(raw)?;
        let root = self
            .workspace
            .canonicalize()
            .map_err(|error| ToolError::Execution {
                message: error.to_string(),
                details: None,
            })?;
        // `direct_path` constrains to the workspace, but canonicalize races
        // or case-fold edges must be a validation error, never a panic.
        let relative = path.strip_prefix(&root).map_err(|_| ToolError::Validation {
            message: "Filesystem capability excludes paths outside workspace".into(),
            details: None,
        })?;
        if relative.components().any(|component| {
            component
                .as_os_str()
                .to_string_lossy()
                .eq_ignore_ascii_case(".omp")
        }) {
            return Err(ToolError::Validation {
                message: "Host state is not tool-writable".into(),
                details: None,
            });
        }
        if matches!(journal.snapshot().session_globals().get("sandbox_write_scope"),
            Some(TypedValue::String(scope)) if scope != "workspace")
        {
            return Err(ToolError::Validation {
                message: "Workspace writes are not approved".into(),
                details: None,
            });
        }
        Ok(path)
    }
}

/// Helper: returns user home directory.
fn home_dir() -> PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

/// Helper: recursively searches directory for files ending with suffix.
fn find_files_with_suffix(dir: &Path, suffix: &str, matches: &mut Vec<PathBuf>, max: usize) {
    fn visit(
        dir: &Path,
        suffix: &str,
        matches: &mut Vec<PathBuf>,
        max: usize,
        remaining: &mut usize,
        depth: usize,
    ) {
        if depth > 32 || *remaining == 0 || matches.len() >= max {
            return;
        }
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                if *remaining == 0 || matches.len() >= max {
                    return;
                }
                *remaining -= 1;
                let Ok(kind) = entry.file_type() else {
                    continue;
                };
                if kind.is_symlink() {
                    continue;
                }
                let path = entry.path();
                if kind.is_dir() {
                    if matches!(
                        entry.file_name().to_str(),
                        Some(".omp" | ".git" | "target" | "node_modules")
                    ) {
                        continue;
                    }
                    visit(&path, suffix, matches, max, remaining, depth + 1);
                } else if path.to_string_lossy().replace('\\', "/").ends_with(suffix) {
                    matches.push(path);
                }
            }
        }
    }
    visit(dir, suffix, matches, max, &mut 10_000, 0);
}

/// Temporary HostGateway holding Mutex<&mut Journal> and Mutex<&mut ToolHost>.
pub struct LiveHostGateway<'a> {
    pub journal: Mutex<&'a mut Journal>,
    pub host: Mutex<&'a mut ToolHost>,
}

impl<'a> HostGateway for LiveHostGateway<'a> {
    fn request(&self, req: HostRequest) -> Result<HostResponse, ToolError> {
        let mut host = self.host.lock();
        let mut journal = self.journal.lock();
        host.handle_request(&mut journal, req)
    }
}

struct PythonToolExecutor {
    name: String,
}
impl crate::definition::ToolExecutor for PythonToolExecutor {
    fn execute(
        &self,
        call: &ToolCall,
        gateway: &dyn HostGateway,
    ) -> Result<ToolExecutionResult, ToolError> {
        let code = format!(
            "__import__('omp_sdk.protocol', fromlist=['invoke_tool']).invoke_tool({}, __import__('json').loads({}))",
            serde_json::to_string(&self.name).unwrap(),
            serde_json::to_string(&call.input.to_string()).unwrap()
        );
        let response = gateway.request(HostRequest::EvalCode {
            code,
            language: "py".into(),
            reset: false,
            timeout_ms: None,
        })?;
        Ok(ToolExecutionResult::from_host(response))
    }
}
/// True when a patch section's path targets the file being edited: exact
/// match, or a suffix match only at a path-separator boundary so
/// `a/foo.rs` can never satisfy an edit intended for `b/foo.rs`.
fn section_targets_path(section_path: &str, requested: &str) -> bool {
    if section_path == requested {
        return true;
    }
    let (long, short) = if section_path.len() > requested.len() {
        (section_path, requested)
    } else {
        (requested, section_path)
    };
    long.len() > short.len()
        && long.ends_with(short)
        && matches!(long.as_bytes()[long.len() - short.len() - 1], b'/' | b'\\')
}

/// Helper: checks if a filename corresponds to parseable code.
fn is_code_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    lower.ends_with(".rs")
        || lower.ends_with(".py")
        || lower.ends_with(".js")
        || lower.ends_with(".ts")
        || lower.ends_with(".tsx")
        || lower.ends_with(".jsx")
        || lower.ends_with(".go")
        || lower.ends_with(".c")
        || lower.ends_with(".cpp")
        || lower.ends_with(".h")
        || lower.ends_with(".hpp")
        || lower.ends_with(".java")
        || lower.ends_with(".kt")
        || lower.ends_with(".scala")
        || lower.ends_with(".cs")
        || lower.ends_with(".swift")
}

/// Helper: builds structural summary for parseable code files (> 40 lines).
fn build_structural_summary(path: &str, tag: &str, lines: &[&str]) -> String {
    let mut out = format!("[{}#{}]\n", path, tag);
    let mut declarations = 0;
    let mut elided_lines = 0;
    let mut in_elision = false;
    let mut elision_start = 0;

    for (idx, line) in lines.iter().enumerate() {
        let line_num = idx + 1;
        let trimmed = line.trim();

        let is_decl = trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("pub struct ")
            || trimmed.starts_with("enum ")
            || trimmed.starts_with("pub enum ")
            || trimmed.starts_with("impl ")
            || trimmed.starts_with("pub trait ")
            || trimmed.starts_with("trait ")
            || trimmed.starts_with("def ")
            || trimmed.starts_with("async def ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("function ")
            || trimmed.starts_with("async function ")
            || trimmed.starts_with("interface ")
            || trimmed.starts_with("type ")
            || trimmed.starts_with("export ");

        if is_decl {
            if in_elision {
                let elided_count = (line_num - 1).saturating_sub(elision_start) + 1;
                elided_lines += elided_count;
                out.push_str(&format!(
                    "{}-{}: … ({} lines elided)\n",
                    elision_start,
                    line_num - 1,
                    elided_count
                ));
                in_elision = false;
            }
            declarations += 1;
            out.push_str(&format!("{}:{}\n", line_num, line));
        } else if !in_elision {
            in_elision = true;
            elision_start = line_num;
        }
    }

    if in_elision {
        let elided_count = lines.len().saturating_sub(elision_start) + 1;
        elided_lines += elided_count;
        out.push_str(&format!(
            "{}-{}: … ({} lines elided)\n",
            elision_start,
            lines.len(),
            elided_count
        ));
    }

    out.push_str(&format!(
        "\n[Structural summary: {} declarations, {} lines elided. Use :N-M selector to read specific ranges]",
        declarations, elided_lines
    ));
    out
}

/// Maximum bytes of expansion or search output returned for one summary read.
const MAX_SUMMARY_BYTES: usize = 64 * 1024;
/// Maximum matches reported by one summary search.
const MAX_SUMMARY_MATCHES: usize = 200;

/// Lists the DAG: every node with its reach and the first line of its text.
fn render_summary_list(
    nodes: &BTreeMap<ElementId, SummaryNode>,
    order: &[ElementId],
    texts: &BTreeMap<ElementId, String>,
    kinds: &BTreeMap<ElementId, SummaryKind>,
) -> String {
    let children: BTreeSet<ElementId> = nodes
        .values()
        .flat_map(|node| node.children.iter().cloned())
        .collect();
    let mut out = format!(
        "# Compacted context: {} node(s). Expand originals with Read summary://<id>, search them with Read summary://?q=<pattern>\n",
        order.len()
    );
    for id in order {
        let node = nodes.get(id).cloned().unwrap_or_default();
        let kind = kinds.get(id).map(|kind| kind.as_str()).unwrap_or("leaf");
        let role = if children.contains(id) { "child" } else { "root" };
        let first = texts
            .get(id)
            .and_then(|text| text.lines().nth(1))
            .unwrap_or("")
            .trim();
        out.push_str(&format!(
            "- summary://{id} — {kind}/{role}, {} elements, ~{} tokens\n    {first}\n",
            expand_covered(nodes, id).len().max(node.covered.len()),
            node.covered_tokens,
        ));
    }
    out
}

/// Renders the original transcript elements a node elides, in body order.
fn render_summary_expansion(
    snapshot: &SessionSnapshot,
    nodes: &BTreeMap<ElementId, SummaryNode>,
    id: &ElementId,
) -> String {
    let covered: BTreeSet<ElementId> = expand_covered(nodes, id).into_iter().collect();
    let node = nodes.get(id).cloned().unwrap_or_default();
    let mut out = format!(
        "# summary://{id} — {} elements, ~{} tokens\n",
        covered.len(), node.covered_tokens
    );
    if covered.is_empty() {
        out.push_str("(this node condenses other nodes; expand its children instead)\n");
        return out;
    }
    let mut rendered = 0usize;
    for element in snapshot.get_visible_body() {
        if !covered.contains(&element.id) {
            continue;
        }
        let mut block = render_transcript_element(snapshot, element);
        if out.len() + block.len() > MAX_SUMMARY_BYTES {
            out.push_str(&format!(
                "[...expansion truncated after {rendered} of {} elements; read a line range with summary://{id}:<from>-<to>]\n",
                covered.len()
            ));
            return out;
        }
        block.push('\n');
        out.push_str(&block);
        rendered += 1;
    }
    if rendered < covered.len() {
        out.push_str(&format!(
            "[{} covered elements are no longer in the transcript]\n",
            covered.len() - rendered
        ));
    }
    out
}

/// One transcript element as the model would have seen it.
fn render_transcript_element(
    snapshot: &SessionSnapshot,
    element: &ElementSnapshot,
) -> String {
    match element.kind.as_str() {
        "user" => format!("--- user ---\n{}\n", element.text),
        "assistant" => format!("--- assistant ---\n{}\n", element.text),
        "tool_call" => {
            let tool = match element.attributes.get("tool") {
                Some(TypedValue::String(name)) => name.clone(),
                _ => "tool".to_string(),
            };
            let args = snapshot
                .children(&element.id)
                .find(|child| child.kind == "input")
                .and_then(|child| child.payload.as_ref())
                .map(|payload| payload.to_string())
                .unwrap_or_default();
            let result = snapshot
                .children(&element.id)
                .find(|child| child.kind == "result")
                .map(|child| child.text.clone())
                .unwrap_or_default();
            format!("--- tool {tool} ---\narguments: {args}\n{result}\n")
        }
        other => format!("--- {other} ---\n{}\n", element.text),
    }
}

/// Searches the originals behind `scope`, so the model can decide what to
/// expand without expanding it first.
fn render_summary_search(
    snapshot: &SessionSnapshot,
    nodes: &BTreeMap<ElementId, SummaryNode>,
    scope: &[ElementId],
    pattern: &str,
) -> String {
    let needle = pattern.to_lowercase();
    let mut out = format!("# summary search for {pattern:?}\n");
    let mut seen: BTreeSet<ElementId> = BTreeSet::new();
    let mut matches = 0usize;
    for id in scope {
        for element in expand_covered(nodes, id) {
            if !seen.insert(element.clone()) {
                continue;
            }
            let Some(node) = snapshot.element(&element) else {
                continue;
            };
            let rendered = render_transcript_element(snapshot, node);
            for (index, line) in rendered.lines().enumerate() {
                if matches >= MAX_SUMMARY_MATCHES {
                    out.push_str(&format!(
                        "[...more than {MAX_SUMMARY_MATCHES} matches; narrow the pattern]\n"
                    ));
                    return out;
                }
                if line.to_lowercase().contains(&needle) {
                    matches += 1;
                    out.push_str(&format!(
                        "summary://{id}:{}: {}\n",
                        index + 1,
                        truncate_summary_line(line)
                    ));
                }
            }
        }
    }
    if matches == 0 {
        out.push_str("no matches in the compacted history\n");
    }
    out
}

fn truncate_summary_line(line: &str) -> String {
    let collapsed: String = line.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.len() <= 200 {
        collapsed
    } else {
        let mut end = 200;
        while end > 0 && !collapsed.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &collapsed[..end])
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use omp_types::{ActorId, SessionId};
    use serde_json::json;

    fn create_test_journal(dir: &Path) -> Journal {
        let jpath = dir.join("test.omp2j");
        Journal::create(jpath, SessionId::mint()).unwrap()
    }

    #[test]
    #[cfg(windows)]
    fn bundled_busybox_matches_pinned_checksum() {
        use sha2::{Digest, Sha256};
        let path = bundled_shell::busybox_path().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            hex::encode(Sha256::digest(&bytes)),
            bundled_shell::BUSYBOX_SHA256
        );
        assert!(path.is_file());
    }

    fn push_body_element(journal: &mut Journal, kind: &str, text: &str) -> ElementId {
        let id = ElementId::mint();
        let mut element = ElementSnapshot::new(id.clone(), kind);
        element.text = text.to_string();
        let container = journal.snapshot().container("body").clone();
        let index = journal.snapshot().children(&container).count() as u32;
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: ActorId::new("test_user").unwrap().into(),
                reason: "test body element".into(),
                ops: vec![PatchOp::Create {
                    parent: container,
                    index,
                    element,
                }],
            })
            .unwrap();
        id
    }

    fn push_summary_node(journal: &mut Journal, covered: Vec<ElementId>, text: &str) -> ElementId {
        let node = omp_types::SummaryNode {
            covered,
            children: Vec::new(),
            covered_tokens: 128,
            created_offset: journal.snapshot().offset,
        };
        let id = ElementId::new(format!("sum-{}", ElementId::mint())).unwrap();
        let element = node.to_element(id.clone(), omp_types::SummaryKind::Leaf, text.to_string());
        let container = journal.snapshot().container("summaries").clone();
        let index = journal.snapshot().children(&container).count() as u32;
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: ActorId::new("test_user").unwrap().into(),
                reason: "test summary node".into(),
                ops: vec![PatchOp::Create {
                    parent: container,
                    index,
                    element,
                }],
            })
            .unwrap();
        id
    }

    #[test]
    fn summary_read_lists_expands_and_searches_compacted_context() {
        let temp_dir = std::env::temp_dir().join(format!("omp_test_summary_{}", ElementId::mint()));
        let _ = fs::create_dir_all(&temp_dir);
        let mut journal = create_test_journal(&temp_dir);
        let mut host = ToolHost::new(temp_dir.clone(), ActorId::new("test_user").unwrap()).unwrap();

        let user = push_body_element(&mut journal, "user", "Explain the parser bug and its fix");
        let assistant = push_body_element(&mut journal, "assistant", "The prefix guard was wrong");
        let node = push_summary_node(
            &mut journal,
            vec![user.clone(), assistant.clone()],
            "Elided transcript elements (originals retrievable by id):\n- user: Explain the parser bug\n- assistant: The prefix guard was wrong\n",
        );

        let listed = host.handle_read(&mut journal, "summary://", None, false).unwrap();
        assert!(listed.output.content.contains(&format!("summary://{node}")));
        assert!(listed.output.content.contains("2 elements"));

        let expanded = host
            .handle_read(&mut journal, &format!("summary://{node}"), None, false)
            .unwrap();
        assert!(expanded.output.content.contains("Explain the parser bug and its fix"));
        assert!(expanded.output.content.contains("The prefix guard was wrong"));
        assert!(expanded.output.content.contains("--- user ---"));

        let selected = host
            .handle_read(&mut journal, &format!("summary://{node}"), Some("3-4"), false)
            .unwrap();
        assert!(selected.output.content.contains("3:"));
        assert!(!selected.output.content.contains("1:--- user ---"));

        let searched = host
            .handle_read(&mut journal, "summary://", Some("?q=prefix guard"), false)
            .unwrap();
        assert!(searched.output.content.contains("prefix guard was wrong"));

        let missing = host.handle_read(&mut journal, "summary://sum-nope", None, false);
        assert!(matches!(missing, Err(ToolError::NotFound(_))));

        let no_matches = host
            .handle_read(&mut journal, "summary://", Some("?q=absent-token"), false)
            .unwrap();
        assert!(no_matches.output.content.contains("no matches"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn summary_read_says_so_when_nothing_was_compacted() {
        let temp_dir = std::env::temp_dir().join(format!("omp_test_summary_empty_{}", ElementId::mint()));
        let _ = fs::create_dir_all(&temp_dir);
        let mut journal = create_test_journal(&temp_dir);
        let mut host = ToolHost::new(temp_dir.clone(), ActorId::new("test_user").unwrap()).unwrap();
        let response = host.handle_read(&mut journal, "summary://", None, false).unwrap();
        assert!(response.output.content.contains("Nothing has been compacted"));
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_tool_host_write_then_read_real_file() {
        let temp_dir = std::env::temp_dir().join(format!("omp_test_host_{}", ElementId::mint()));
        let _ = fs::create_dir_all(&temp_dir);

        let mut journal = create_test_journal(&temp_dir);
        let mut host = ToolHost::new(temp_dir.clone(), ActorId::new("test_user").unwrap()).unwrap();

        // 1. Write real file
        let write_call = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "Write",
            "1.0.0",
            json!({
                "path": "hello.txt",
                "content": "Line 1: Hello\nLine 2: World\nLine 3: Foo\n",
                "i": "Writing test file"
            }),
        );

        host.execute_tool(&mut journal, &write_call).unwrap();

        // Verify file on disk
        let disk_content = fs::read_to_string(temp_dir.join("hello.txt")).unwrap();
        assert_eq!(disk_content, "Line 1: Hello\nLine 2: World\nLine 3: Foo\n");

        // 2. Read real file
        let read_call = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "Read",
            "1.0.0",
            json!({
                "path": "hello.txt",
                "i": "Reading test file"
            }),
        );

        let read_res = host.execute_tool(&mut journal, &read_call).unwrap();
        assert!(read_res.output.content.contains("1:Line 1: Hello"));
        assert!(read_res.output.content.contains("2:Line 2: World"));
        assert!(read_res.output.content.contains("3:Line 3: Foo"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    #[cfg(windows)]
    fn remove_file_reports_failure_when_the_os_refuses_deletion() {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_SHARE_READ: u32 = 0x0000_0001;

        let temp_dir = std::env::temp_dir().join(format!("omp_test_host_rem_{}", ElementId::mint()));
        let _ = fs::create_dir_all(&temp_dir);

        let mut journal = create_test_journal(&temp_dir);
        let host = ToolHost::new(temp_dir.clone(), ActorId::new("test_user").unwrap()).unwrap();

        // A handle that grants reads but withholds delete sharing makes
        // `DeleteFile` fail with a sharing violation, so the host has to report
        // the failure instead of claiming the file is gone.
        let target = temp_dir.join("locked.txt");
        fs::write(&target, "keep me\n").unwrap();
        let lock = fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .open(&target)
            .unwrap();

        let result = host.handle_edit(&mut journal, "locked.txt", None, "[locked.txt#0000]\nREM\n");

        assert!(
            result.is_err(),
            "deleting a file the OS refuses to delete must not report success"
        );
        assert!(target.exists(), "the file must still be on disk");

        drop(lock);
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_tool_host_failed_edit_leaves_file_unchanged() {
        let temp_dir =
            std::env::temp_dir().join(format!("omp_test_host_edit_{}", ElementId::mint()));
        let _ = fs::create_dir_all(&temp_dir);

        let mut journal = create_test_journal(&temp_dir);
        let mut host = ToolHost::new(temp_dir.clone(), ActorId::new("test_user").unwrap()).unwrap();

        // Initial write
        let original_content = "original line 1\noriginal line 2\n";
        fs::write(temp_dir.join("target.txt"), original_content).unwrap();

        // Failed Edit: wrong tag
        let edit_call_bad_tag = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "Edit",
            "1.0.0",
            json!({
                "input": "[target.txt#DEAD]\nPUT 1.=1:\n+modified\n",
                "i": "Editing with wrong tag"
            }),
        );

        let edit_res = host.execute_tool(&mut journal, &edit_call_bad_tag).unwrap();
        assert!(
            edit_res
                .diagnostics
                .iter()
                .any(|d| d.code.as_deref() == Some("stale_tag"))
        );

        // Verify file is UNCHANGED
        let current_disk = fs::read_to_string(temp_dir.join("target.txt")).unwrap();
        assert_eq!(current_disk, original_content);

        // Failed Edit: out of bounds range
        let read_call = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "Read",
            "1.0.0",
            json!({
                "path": "target.txt",
                "i": "Reading tag"
            }),
        );
        let read_res = host.execute_tool(&mut journal, &read_call).unwrap();
        let header = read_res.output.content.lines().next().unwrap();
        let tag = header.split('#').nth(1).unwrap().trim_end_matches(']');

        let edit_call_out_of_bounds = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "Edit",
            "1.0.0",
            json!({
                "input": format!("[target.txt#{}]\nPUT 100.=200:\n+modified\n", tag),
                "i": "Editing out of bounds"
            }),
        );

        let oob_res = host
            .execute_tool(&mut journal, &edit_call_out_of_bounds)
            .unwrap();
        assert!(
            oob_res
                .diagnostics
                .iter()
                .any(|d| d.code.as_deref() == Some("conflict"))
        );

        // Verify file is still UNCHANGED
        let current_disk_after = fs::read_to_string(temp_dir.join("target.txt")).unwrap();
        assert_eq!(current_disk_after, original_content);

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn edit_with_no_targeting_section_leaves_file_unchanged() {
        let temp_dir =
            std::env::temp_dir().join(format!("omp_test_host_edit_miss_{}", ElementId::mint()));
        let _ = fs::create_dir_all(&temp_dir);

        let mut journal = create_test_journal(&temp_dir);
        let mut host = ToolHost::new(temp_dir.clone(), ActorId::new("test_user").unwrap()).unwrap();

        // CRLF content so an accidental rewrite would be impossible to miss.
        let original = b"line one\r\nline two\r\n".to_vec();
        fs::write(temp_dir.join("target.txt"), &original).unwrap();

        // Patch whose only section targets a DIFFERENT path ("other.txt"):
        // handle_edit on target.txt must abort with a conflict diagnostic.
        let res = host
            .handle_edit(
                &mut journal,
                "target.txt",
                None,
                "[other.txt#TAG]\nPUT 1.=1:\n+changed\n",
            )
            .unwrap();
        assert!(
            res.diagnostics
                .iter()
                .any(|d| d.code.as_deref() == Some("conflict"))
        );
        // Byte-identical: no rewrite, no CRLF normalization, no new tag.
        assert_eq!(fs::read(temp_dir.join("target.txt")).unwrap(), original);

        // Non-boundary suffix: a section for "rget.txt" must NOT satisfy an
        // edit for "target.txt" (old bare ends_with matched this).
        let res2 = host
            .handle_edit(
                &mut journal,
                "target.txt",
                None,
                "[rget.txt#TAG]\nPUT 1.=1:\n+changed\n",
            )
            .unwrap();
        assert!(
            res2.diagnostics
                .iter()
                .any(|d| d.code.as_deref() == Some("conflict"))
        );
        assert_eq!(fs::read(temp_dir.join("target.txt")).unwrap(), original);

        // Sanity: a correctly-targeted patch still applies.
        let read_res = host
            .execute_tool(
                &mut journal,
                &ToolCall::new(
                    omp_types::ToolCallId::mint(),
                    "Read",
                    "1.0.0",
                    json!({ "path": "target.txt", "i": "Reading tag" }),
                ),
            )
            .unwrap();
        let header = read_res.output.content.lines().next().unwrap().to_string();
        let tag = header.split('#').nth(1).unwrap().trim_end_matches(']');
        let edit_ok = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "Edit",
            "1.0.0",
            json!({
                "input": format!("[target.txt#{}]\nPUT 1.=1:\n+changed\n", tag),
                "i": "Editing target"
            }),
        );
        let ok_res = host.execute_tool(&mut journal, &edit_ok).unwrap();
        assert!(
            !ok_res
                .diagnostics
                .iter()
                .any(|d| d.code.as_deref() == Some("conflict"))
        );
        assert!(fs::read_to_string(temp_dir.join("target.txt"))
            .unwrap()
            .contains("changed"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_dyn_disappears_on_rewind() {
        let temp_dir =
            std::env::temp_dir().join(format!("omp_test_host_dyn_{}", ElementId::mint()));
        let _ = fs::create_dir_all(&temp_dir);

        let mut journal = create_test_journal(&temp_dir);
        let mut host = ToolHost::new(temp_dir.clone(), ActorId::new("test_user").unwrap()).unwrap();

        // Branch 1: Discover dynamic tool
        let offset = journal.snapshot().offset;
        let mut dyn_node = ElementSnapshot::new(ElementId::mint(), "tool");
        dyn_node
            .attributes
            .insert("name".into(), TypedValue::String("pkg/dyn_action".into()));
        dyn_node.attributes.insert(
            "description".into(),
            TypedValue::String("Dynamic test action".into()),
        );
        dyn_node
            .attributes
            .insert("dynamic".into(), TypedValue::Bool(true));

        let patch = Patch {
            base_offset: JournalOffset(offset),
            result_offset: journal.next_offset(),
            by: host.owner.clone().into(),
            reason: "Discover dynamic tool".into(),
            ops: vec![PatchOp::Create {
                parent: ElementId::new("tools").unwrap(),
                index: 0,
                element: dyn_node,
            }],
        };
        journal.append_patch(patch).unwrap();

        // dyn list should find it
        let dyn_call = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "dyn",
            "1.0.0",
            json!({
                "query": "dyn_action",
                "i": "Searching dynamic tool"
            }),
        );
        let res = host.execute_tool(&mut journal, &dyn_call).unwrap();
        assert!(res.output.content.contains("pkg/dyn_action"));

        // Rewind to offset before discovery
        journal.rewind_to(offset).unwrap();

        // dyn list after rewind should NOT find pkg/dyn_action
        let dyn_call_rewound = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "dyn",
            "1.0.0",
            json!({
                "query": "dyn_action",
                "i": "Searching dynamic tool after rewind"
            }),
        );
        let res_rewound = host.execute_tool(&mut journal, &dyn_call_rewound).unwrap();
        assert!(!res_rewound.output.content.contains("pkg/dyn_action"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_autoqa_report_filtering_and_recording() {
        let temp_dir = std::env::temp_dir().join(format!("omp_test_host_qa_{}", ElementId::mint()));
        let _ = fs::create_dir_all(&temp_dir);

        let mut journal = create_test_journal(&temp_dir);
        let mut host = ToolHost::new(temp_dir.clone(), ActorId::new("test_user").unwrap()).unwrap();

        // Identical expected and observed behavior -> reject
        let invalid_qa = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "AutoQA",
            "1.0.0",
            json!({
                "tool": "Bash",
                "input_summary": "Running grep on directory",
                "expected_behavior": "Should succeed",
                "observed_behavior": "Should succeed",
                "severity": "medium",
                "i": "Reporting QA issue"
            }),
        );
        let err = host.execute_tool(&mut journal, &invalid_qa).unwrap_err();
        assert!(
            matches!(err, ToolError::Validation { message, .. } if message.contains("identical"))
        );

        // Valid QA -> recorded
        let valid_qa = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "AutoQA",
            "1.0.0",
            json!({
                "tool": "Bash",
                "input_summary": "Running grep on directory",
                "expected_behavior": "Should execute command",
                "observed_behavior": "Returned exit code 1",
                "severity": "medium",
                "i": "Reporting QA issue"
            }),
        );
        host.execute_tool(&mut journal, &valid_qa).unwrap();
        let report = journal
            .snapshot()
            .all_elements()
            .find(|element| element.kind == "qa-report")
            .unwrap();
        assert_eq!(
            report.payload.as_ref().unwrap()["observed_behavior"],
            "Returned exit code 1"
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    #[cfg(windows)]
    fn extension_tools_follow_rewind_reload_and_worker_reset() {
        let root = std::env::temp_dir().join(format!("omp-extension-{}", ElementId::mint()));
        fs::create_dir_all(root.join("source")).unwrap();
        let mut journal = create_test_journal(&root);
        let mut host = ToolHost::new(root.join("source"), ActorId::mint()).unwrap();
        let load = "from omp_sdk.examples import StatefulTodoExtension\nfrom omp_sdk.extension import ExtensionContext\nfrom omp_sdk.protocol import StreamTransport\nt = StreamTransport()\next = StatefulTodoExtension(t)\next.on_load(ExtensionContext(ext.extension_id, t))";
        let before = journal.snapshot().offset;
        let response = host
            .handle_eval(&mut journal, load, "py", false, Some(5000))
            .unwrap();
        assert!(
            response.diagnostics.is_empty(),
            "{:?}",
            response.diagnostics
        );
        let name = "stateful_todo_example/todo_counter";
        let invoke = ToolCall::new(
            omp_types::ToolCallId::mint(),
            name,
            "1.0.0",
            json!({"title":"durable tool item","i":"Adding item"}),
        );
        let result = host.execute_tool(&mut journal, &invoke).unwrap();
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert!(
            journal
                .snapshot()
                .active_todos()
                .any(|node| node.text == "durable tool item")
        );
        let response = host
            .handle_eval(&mut journal, "ext.on_reload()", "py", false, Some(5000))
            .unwrap();
        assert!(
            response.diagnostics.is_empty(),
            "{:?}",
            response.diagnostics
        );
        assert_eq!(journal.snapshot().active_tool_roster().filter(|node| node.attributes.get("name") == Some(&TypedValue::String(name.into()))).count(), 1);
        journal.rewind_to(before).unwrap();
        assert!(host.execute_tool(&mut journal, &invoke).is_err());
        assert!(
            !host
                .handle_dyn_lookup(&mut journal, None, None, false, None)
                .unwrap()
                .output
                .content
                .contains(name)
        );
        host.handle_eval(&mut journal, load, "py", false, Some(5000))
            .unwrap();
        host.handle_eval(&mut journal, "42", "py", true, Some(5000))
            .unwrap();
        assert!(host.execute_tool(&mut journal, &invoke).is_err());
        host.shutdown_jobs(&mut journal).unwrap();
        drop(host);
        drop(journal);
        fs::remove_dir_all(root).unwrap();
    }
}
