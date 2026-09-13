use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use omp_runtime::RestrictedProcess;
use omp_runtime::workspace::WorkspaceView;
use omp_state::Journal;
use omp_types::{
    ActorId, ElementId, ElementSnapshot, JournalOffset, LimitPolicy, Patch, PatchOp,
    StructuredError, TypedValue,
};
use serde::{Deserialize, Serialize};

pub const OFFICIAL_PYTHON_VERSION: &str = "3.11.9";
pub const OFFICIAL_PYTHON_SHA256: &str =
    "009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PythonRuntime {
    pub python_bin: PathBuf,
    pub runtime_dir: PathBuf,
    pub version: String,
    pub checksum: String,
}

impl PythonRuntime {
    pub fn bundled() -> Result<Self, StructuredError> {
        use sha2::{Digest, Sha256};
        let archive_bytes = include_bytes!("../assets/python-3.11.9-embed-amd64.zip");
        if hex::encode(Sha256::digest(archive_bytes)) != OFFICIAL_PYTHON_SHA256 {
            return Err(StructuredError::new(
                "runtime_checksum",
                "Bundled Python archive checksum mismatch",
                false,
            ));
        }
        let root = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("omp2")
            .join("runtimes");
        std::fs::create_dir_all(&root)
            .map_err(|error| StructuredError::new("runtime_install", error.to_string(), false))?;
        let runtime_dir = root.join(format!(
            "python-{OFFICIAL_PYTHON_VERSION}-{OFFICIAL_PYTHON_SHA256}"
        ));
        let installing = !runtime_dir.exists();
        let target = if installing {
            root.join(format!("install-{}", ElementId::mint()))
        } else {
            runtime_dir.clone()
        };
        std::fs::create_dir_all(&target)
            .map_err(|error| StructuredError::new("runtime_install", error.to_string(), false))?;
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(archive_bytes))
            .map_err(|error| StructuredError::new("runtime_archive", error.to_string(), false))?;
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index).map_err(|error| {
                StructuredError::new("runtime_archive", error.to_string(), false)
            })?;
            let relative = entry.enclosed_name().ok_or_else(|| {
                StructuredError::new("runtime_archive", "Unsafe archive member", false)
            })?;
            let path = target.join(relative);
            if entry.is_dir() {
                std::fs::create_dir_all(&path).map_err(|error| {
                    StructuredError::new("runtime_install", error.to_string(), false)
                })?;
                continue;
            }
            let mut expected = Vec::new();
            std::io::Read::read_to_end(&mut entry, &mut expected).map_err(|error| {
                StructuredError::new("runtime_archive", error.to_string(), false)
            })?;
            if installing {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).map_err(|error| {
                        StructuredError::new("runtime_install", error.to_string(), false)
                    })?;
                }
                std::fs::write(&path, &expected).map_err(|error| {
                    StructuredError::new("runtime_install", error.to_string(), false)
                })?;
            } else {
                let actual = std::fs::read(&path).map_err(|error| {
                    StructuredError::new("runtime_checksum", error.to_string(), false)
                })?;
                if actual != expected {
                    return Err(StructuredError::new(
                        "runtime_checksum",
                        format!("Installed runtime changed: {}", path.display()),
                        false,
                    ));
                }
            }
        }
        if installing
            && let Err(error) = std::fs::rename(&target, &runtime_dir) {
                let _ = std::fs::remove_dir_all(&target);
                if runtime_dir.exists() {
                    return Self::bundled();
                }
                return Err(StructuredError::new(
                    "runtime_install",
                    error.to_string(),
                    false,
                ));
            }
        Ok(Self {
            python_bin: runtime_dir.join("python.exe"),
            runtime_dir,
            version: OFFICIAL_PYTHON_VERSION.into(),
            checksum: OFFICIAL_PYTHON_SHA256.into(),
        })
    }
}

pub struct PythonSession {
    pub session_id: String,
    pub runtime: PythonRuntime,
    pub workspace_view: WorkspaceView,
    pub owner: ActorId,
    pub limits: LimitPolicy,
    pub element: ElementId,
    process: Option<RestrictedProcess>,
    stdout_rx: Option<mpsc::Receiver<String>>,
    lease: Option<omp_runtime::workspace::WorkspaceLease>,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl PythonSession {
    pub fn spawn(
        runtime: &PythonRuntime,
        journal: &mut Journal,
        workspace: &WorkspaceView,
        owner: ActorId,
        limits: LimitPolicy,
    ) -> Result<Self, StructuredError> {
        if workspace.session_id != journal.snapshot().session_id
            || workspace.base_path == workspace.isolated_path
        {
            return Err(StructuredError::new(
                "invalid_python_workspace",
                "Python requires a session-owned isolated workspace",
                false,
            ));
        }
        let lease = workspace.acquire()?;
        let session_id = format!("py-session-{}", ElementId::mint());
        let element = ElementId::new(session_id.clone())?;

        let mut node = ElementSnapshot::new(element.clone(), "python-session");
        node.attributes
            .insert("status".into(), TypedValue::String("running".into()));
        node.attributes
            .insert("owner".into(), TypedValue::String(owner.to_string()));
        node.attributes.insert(
            "version".into(),
            TypedValue::String(runtime.version.clone()),
        );
        node.payload = Some(serde_json::json!({"workspace":workspace,"limits":limits}));

        let offset = journal.snapshot().offset;
        let patch = Patch {
            base_offset: JournalOffset(offset),
            result_offset: journal.next_offset(),
            by: ActorId::new("host").unwrap().into(),
            reason: "spawn persistent Python worker session".into(),
            ops: vec![PatchOp::Create {
                parent: ElementId::new("jobs").unwrap(),
                index: journal.snapshot().active_jobs().count() as u32,
                element: node,
            }],
        };
        journal.append_patch(patch).map_err(|e| e.structured())?;

        let mut session = Self {
            session_id,
            runtime: runtime.clone(),
            workspace_view: workspace.clone(),
            owner,
            limits,
            element,
            process: None,
            stdout_rx: None,
            lease: Some(lease),
        };

        if let Err(error) = session.ensure_process() {
            session.record_status(journal, "failed", &error.message)?;
            return Err(error);
        }
        Ok(session)
    }

    fn ensure_process(&mut self) -> Result<(), StructuredError> {
        if let Some(proc) = &mut self.process
            && let Ok(None) = proc.try_wait() {
                return Ok(());
            }

        if self.lease.is_none() {
            self.lease = Some(self.workspace_view.acquire()?);
        }
        let worker_script = r#"
import sys, json, io, traceback
MAX_OUTPUT = 100000
class BoundedCapture(io.StringIO):
    def __init__(self):
        super().__init__()
        self.remaining = MAX_OUTPUT
        self.truncated = False
    def write(self, value):
        data = str(value).encode('utf-8')
        part = data[:self.remaining].decode('utf-8', 'ignore')
        self.remaining -= len(part.encode('utf-8'))
        self.truncated |= len(data) > len(part.encode('utf-8'))
        super().write(part)
        return len(value)

def _eval_loop():
    scope = {}
    while True:
        try:
            line = sys.stdin.readline(65537)
            if len(line) > 65536:
                raise ValueError('request size exceeded')
        except Exception:
            break
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            sys.stdout.write(json.dumps({"status": "error", "error": {"type": "JsonDecodeError", "message": str(e), "retryable": False}}) + "\n")
            sys.stdout.flush()
            continue

        action = req.get("action", "eval")
        if action == "reset":
            scope.clear()
            sys.stdout.write(json.dumps({"status": "ok", "result": None, "output": "", "text": "Session reset."}) + "\n")
            sys.stdout.flush()
            continue

        code = req.get("code", "")
        if req.get("reset", False):
            scope.clear()

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured = BoundedCapture()
        sys.stdout = captured
        sys.stderr = captured

        result = None
        error = None
        try:
            try:
                compiled = compile(code, "<eval>", "eval")
                result = eval(compiled, scope)
            except SyntaxError:
                compiled = compile(code, "<eval>", "exec")
                exec(compiled, scope)
        except Exception as e:
            error = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "retryable": False
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        out_str = captured.getvalue()
        if error is not None:
            text = out_str + "\n" + error["traceback"] if out_str else error["traceback"]
            resp = {"status": "error", "error": error, "output": out_str, "text": text.strip()}
        else:
            try:
                json.dumps(result)
                val = result
            except Exception:
                val = repr(result) if result is not None else None
            text = out_str
            if result is not None:
                r_repr = repr(result)
                text = f"{out_str}\n{r_repr}".strip() if out_str else r_repr
            resp = {"status": "ok", "result": val, "output": out_str, "text": text}

        resp['truncated'] = captured.truncated
        encoded = json.dumps(resp)
        if len(encoded.encode('utf-8')) > 800000:
            encoded = json.dumps({'status':'error','error':{'type':'OutputLimit','message':'Python result exceeds response budget'},'truncated':True})
        sys.stdout.write(encoded + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    _eval_loop()
"#;

        let modules = serde_json::json!({
            "protocol": include_str!("../../../python/omp_sdk/protocol.py"),
            "wrappers": include_str!("../../../python/omp_sdk/wrappers.py"),
            "remote": include_str!("../../../python/omp_sdk/remote.py"),
            "extension": include_str!("../../../python/omp_sdk/extension.py"),
            "examples": include_str!("../../../python/omp_sdk/examples.py"),
            "__init__": include_str!("../../../python/omp_sdk/__init__.py"),
        });
        let bootstrap = format!(
            "import sys, types, json\n_sources = json.loads({})\n_pkg = types.ModuleType('omp_sdk')\n_pkg.__path__ = []\nsys.modules['omp_sdk'] = _pkg\nfor _name in ['protocol','wrappers','remote','extension','examples']:\n _module = types.ModuleType('omp_sdk.' + _name)\n _module.__package__ = 'omp_sdk'\n sys.modules[_module.__name__] = _module\n exec(compile(_sources[_name], '<omp_sdk/' + _name + '>', 'exec'), _module.__dict__)\nexec(compile(_sources['__init__'], '<omp_sdk>', 'exec'), _pkg.__dict__)\n",
            serde_json::to_string(&modules.to_string()).unwrap()
        );
        use sha2::{Digest, Sha256};
        let source = bootstrap + worker_script;
        let script_path = self.runtime.runtime_dir.parent().unwrap().join(format!(
            "worker-{}.py",
            hex::encode(Sha256::digest(source.as_bytes()))
        ));
        if script_path.exists() {
            if std::fs::read(&script_path)
                .map_err(|error| StructuredError::new("worker_script", error.to_string(), false))?
                != source.as_bytes()
            {
                return Err(StructuredError::new(
                    "worker_checksum",
                    "Trusted worker script changed",
                    false,
                ));
            }
        } else {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&script_path)
                .map_err(|error| StructuredError::new("worker_script", error.to_string(), false))?;
            file.write_all(source.as_bytes())
                .map_err(|error| StructuredError::new("worker_script", error.to_string(), false))?;
        }
        let args = vec![
            "-I".into(),
            "-S".into(),
            "-u".into(),
            script_path.to_string_lossy().into_owned(),
        ];
        let cwd = &self.workspace_view.isolated_path;
        let mut runtime_files = std::fs::read_dir(&self.runtime.runtime_dir)
            .map_err(|error| StructuredError::new("runtime_files", error.to_string(), false))?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| StructuredError::new("runtime_files", error.to_string(), false))?;
        runtime_files.push(self.runtime.runtime_dir.clone());
        runtime_files.push(script_path);
        let mut proc = RestrictedProcess::spawn_with_runtime(
            &self.runtime.python_bin,
            &args,
            cwd,
            &self.limits,
            &runtime_files,
        )?;
        if let Some(mut stderr) = proc.take_stderr() {
            thread::spawn(move || {
                let _ = std::io::copy(&mut stderr, &mut std::io::sink());
            });
        }
        let (tx, rx) = mpsc::sync_channel(2);
        if let Some(stdout) = proc.take_stdout() {
            thread::spawn(move || {
                let mut reader = BufReader::new(stdout);
                let mut line = String::new();
                while let Ok(n) =
                    BufRead::read_line(&mut std::io::Read::take(&mut reader, 1_048_577), &mut line)
                {
                    if line.len() > 1_048_576 {
                        break;
                    }
                    if n == 0 {
                        break;
                    }
                    if tx.send(line.clone()).is_err() {
                        break;
                    }
                    line.clear();
                }
            });
        }

        self.process = Some(proc);
        self.stdout_rx = Some(rx);
        Ok(())
    }

    pub fn eval(
        &mut self,
        journal: &mut Journal,
        code: &str,
        reset: bool,
    ) -> Result<serde_json::Value, StructuredError> {
        self.eval_with_host(journal, code, reset, |_, _| {
            Err(StructuredError::new(
                "capability_denied",
                "Host callback is unavailable in this worker context",
                false,
            ))
        })
    }

    pub fn eval_with_host<F>(
        &mut self,
        journal: &mut Journal,
        code: &str,
        reset: bool,
        mut host: F,
    ) -> Result<serde_json::Value, StructuredError>
    where
        F: FnMut(&mut Journal, serde_json::Value) -> Result<serde_json::Value, StructuredError>,
    {
        if code.len() > 60_000 {
            return Err(StructuredError::new(
                "eval_input_limit",
                "Python code exceeds 60000 bytes",
                false,
            ));
        }
        if journal.snapshot().element(&self.element).is_none() {
            self.terminate()?;
            return Err(StructuredError::new(
                "python_branch_changed",
                "Python worker is not on the selected branch",
                false,
            ));
        }
        if reset {
            let _ = self.terminate();
        }

        self.ensure_process()?;

        let req = serde_json::json!({
            "action": "eval",
            "code": code,
            "reset": reset,
        });

        let req_str = format!(
            "{}\n",
            serde_json::to_string(&req).map_err(|e| StructuredError::new(
                "json_encode_error",
                e.to_string(),
                false
            ))?
        );

        let proc = self.process.as_mut().ok_or_else(|| {
            StructuredError::new(
                "worker_missing",
                "Python worker process is not active",
                false,
            )
        })?;

        proc.write_stdin(req_str.as_bytes())?;

        if self.stdout_rx.is_none() {
            return Err(StructuredError::new(
                "pipe_missing",
                "stdout pipe channel is missing",
                false,
            ));
        }
        let timeout = if self.limits.max_wall_time.is_zero() {
            Duration::from_secs(30)
        } else {
            self.limits.max_wall_time
        };

        let deadline = std::time::Instant::now() + timeout;
        let mut requests = 0;
        loop {
            match self
                .stdout_rx
                .as_ref()
                .unwrap()
                .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
            {
                Ok(line) => {
                    let value: serde_json::Value = serde_json::from_str(&line).map_err(|e| {
                        StructuredError::new(
                            "protocol_error",
                            format!("malformed worker response: {} (raw: {})", e, line.trim()),
                            false,
                        )
                    })?;
                    if let Some(request) = value.get("host_request") {
                        requests += 1;
                        if requests > 128 {
                            self.terminate()?;
                            return Err(StructuredError::new(
                                "host_request_limit",
                                "Worker exceeded 128 host requests per evaluation",
                                false,
                            ));
                        }
                        let result = if line.len() > 65536 {
                            Err(StructuredError::new(
                                "host_request_limit",
                                "Host request exceeds 65536 bytes",
                                false,
                            ))
                        } else {
                            host(journal, request.clone())
                        };
                        let response = match result {
                            Ok(value) => serde_json::json!({"result":value}),
                            Err(error) => serde_json::json!({"error":error}),
                        };
                        let mut response = serde_json::to_vec(&response).map_err(|error| {
                            StructuredError::new("host_encoding", error.to_string(), false)
                        })?;
                        if response.len() > 1_000_000 {
                            response = br#"{"error":{"code":"host_response_limit","message":"Host response exceeds boundary"}}"#.to_vec();
                        }
                        response.push(b'\n');
                        self.process.as_mut().unwrap().write_stdin(&response)?;
                        continue;
                    }

                    let offset = journal.snapshot().offset;
                    let patch = Patch {
                        base_offset: JournalOffset(offset),
                        result_offset: journal.next_offset(),
                        by: self.owner.clone().into(),
                        reason: "python eval completed".into(),
                        ops: vec![PatchOp::SetAttribute {
                            element: self.element.clone(),
                            name: "last_eval".into(),
                            value: TypedValue::String(now_ms().to_string()),
                        }],
                    };
                    journal
                        .append_patch(patch)
                        .map_err(|error| error.structured())?;

                    return Ok(value);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    let _ = self.terminate();
                    self.record_status(
                        journal,
                        "failed",
                        "forced_kill: Python evaluation timeout",
                    )?;
                    return Err(StructuredError::new(
                        "timeout",
                        format!(
                            "Python eval exceeded timeout of {:?}; worker terminated",
                            timeout
                        ),
                        false,
                    ));
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    let _ = self.terminate();
                    self.record_status(journal, "failed", "Python worker disconnected")?;
                    return Err(StructuredError::new(
                        "worker_disconnected",
                        "Python worker process disconnected unexpectedly",
                        false,
                    ));
                }
            }
        }
    }

    fn record_status(
        &self,
        journal: &mut Journal,
        status: &str,
        reason: &str,
    ) -> Result<(), StructuredError> {
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason: reason.into(),
                ops: vec![PatchOp::SetAttribute {
                    element: self.element.clone(),
                    name: "status".into(),
                    value: TypedValue::String(status.into()),
                }],
            })
            .map_err(|error| error.structured())?;
        Ok(())
    }

    pub fn close(&mut self, journal: &mut Journal) -> Result<(), StructuredError> {
        self.terminate()?;
        if let Some(node) = journal.snapshot().element(&self.element)
            && matches!(node.attributes.get("status"), Some(TypedValue::String(status)) if status == "running")
            {
                self.record_status(journal, "cancelled", "Python worker closed by host")?;
            }
        Ok(())
    }

    pub fn terminate(&mut self) -> Result<(), StructuredError> {
        if let Some(mut proc) = self.process.take() {
            let _ = proc.kill();
        }
        self.stdout_rx = None;
        self.lease.take();
        Ok(())
    }
}

impl Drop for PythonSession {
    fn drop(&mut self) {
        let _ = self.terminate();
    }
}
