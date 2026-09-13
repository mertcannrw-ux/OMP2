mod editor;
mod input;
mod tui;
mod view;
mod worker;

use omp_state::Journal;
use omp_types::{ActorId, JOURNAL_FORMAT, ProtocolVersion, SessionId, StructuredError};
use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

// ---------------------------------------------------------------------------
// Path Policy
// ---------------------------------------------------------------------------

pub mod path_policy {
    use super::*;

    /// Expand leading `~` into the user's home directory across Windows and POSIX.
    pub fn expand_tilde(path: &Path) -> PathBuf {
        let path_str = path.to_string_lossy();
        if path_str == "~" || path_str.starts_with("~/") || path_str.starts_with("~\\") {
            let home = if cfg!(windows) {
                std::env::var_os("USERPROFILE")
                    .or_else(|| std::env::var_os("HOME"))
                    .or_else(|| {
                        let drive = std::env::var_os("HOMEDRIVE");
                        let path = std::env::var_os("HOMEPATH");
                        match (drive, path) {
                            (Some(d), Some(p)) => {
                                let mut combined = d;
                                combined.push(p);
                                Some(combined)
                            }
                            _ => None,
                        }
                    })
            } else {
                std::env::var_os("HOME")
            };

            if let Some(home_dir) = home {
                let home_path = PathBuf::from(home_dir);
                if path_str == "~" {
                    return home_path;
                }
                let remainder = &path_str[2..];
                return home_path.join(remainder);
            }
        }
        path.to_path_buf()
    }

    /// Reject ambiguous paths that contain wildcards, null bytes, or conflicting traversal.
    pub fn validate_unambiguous_path(path: &Path) -> Result<PathBuf, CliError> {
        let path_str = path.to_string_lossy();
        let stripped = path_str.strip_prefix(r"\\?\").unwrap_or(&path_str);
        if stripped.contains('*') || stripped.contains('?') || stripped.contains('\0') {
            return Err(CliError::AmbiguousPath {
                query: path_str.to_string(),
                candidates: vec!["wildcards are not permitted in authoritative paths".into()],
            });
        }
        Ok(path.to_path_buf())
    }

    /// Resolve a unique suffix within a directory. If multiple entries match, reject as ambiguous.
    pub fn resolve_unique_suffix(dir: &Path, suffix: &str) -> Result<PathBuf, CliError> {
        if !dir.exists() || !dir.is_dir() {
            return Err(CliError::NotFound(format!(
                "directory '{}' does not exist",
                dir.display()
            )));
        }

        let mut matches = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().to_string();
            // Suffix-boundary match: `foo` matches `a-foo` or `a/foo` but not
            // `afoobar` (same rule as the tool path resolver).
            let boundary_match = file_name == suffix
                || file_name.ends_with(suffix)
                    && file_name[..file_name.len() - suffix.len()]
                        .chars()
                        .next_back()
                        .is_some_and(|c| !c.is_ascii_alphanumeric());
            if boundary_match {
                matches.push(path);
            }
        }

        match matches.len() {
            0 => Err(CliError::NotFound(format!(
                "no entry matching suffix '{suffix}' in '{}'",
                dir.display()
            ))),
            1 => Ok(matches.remove(0)),
            _ => Err(CliError::AmbiguousPath {
                query: suffix.to_string(),
                candidates: matches
                    .into_iter()
                    .map(|p| p.display().to_string())
                    .collect(),
            }),
        }
    }

    /// Discover workspace root by ascending parent hierarchy looking for markers:
    /// Cargo.toml, .omp, .git, or configs/.
    /// Discover workspace root by ascending parent hierarchy looking for project markers:
    /// .git, Cargo.toml, package.json, pyproject.toml, go.mod.
    /// Never climbs into or past the user's home directory.
    /// Never treats parent .omp as a project marker (as ~/.omp is the user-wide state dir).
    pub fn discover_workspace_root(start: &Path) -> Result<PathBuf, CliError> {
        let canonical_start = if start.exists() {
            start.canonicalize().unwrap_or_else(|_| start.to_path_buf())
        } else {
            start.to_path_buf()
        };

        if canonical_start.join(".omp").join("state").exists() {
            return Ok(canonical_start);
        }

        let home = expand_tilde(Path::new("~"));
        let canonical_home = home.canonicalize().unwrap_or(home);

        let mut current = canonical_start.clone();
        loop {
            if current == canonical_home {
                break;
            }
            if current.join(".git").exists()
                || current.join("Cargo.toml").exists()
                || current.join("package.json").exists()
                || current.join("pyproject.toml").exists()
                || current.join("go.mod").exists()
            {
                return Ok(current);
            }
            if let Some(parent) = current.parent() {
                current = parent.to_path_buf();
            } else {
                break;
            }
        }

        Ok(canonical_start)
    }

    /// Default workspace state directory: `<workspace_root>/.omp/state`
    pub fn default_state_dir(workspace_root: &Path) -> PathBuf {
        workspace_root.join(".omp").join("state")
    }

    /// Default journal path for a session: `<workspace_root>/.omp/state/<session_id>.journal`
    pub fn default_journal_path(workspace_root: &Path, session_id: &SessionId) -> PathBuf {
        default_state_dir(workspace_root).join(format!("{}.journal", session_id.as_str()))
    }

    /// Locate the latest journal in the state directory by modification time.
    pub fn find_latest_journal(state_dir: &Path) -> Result<PathBuf, CliError> {
        if !state_dir.exists() {
            return Err(CliError::NotFound(format!(
                "state directory '{}' does not exist",
                state_dir.display()
            )));
        }

        let mut latest: Option<(PathBuf, std::time::SystemTime)> = None;
        for entry in fs::read_dir(state_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("journal") {
                let mtime = entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

                match &latest {
                    None => latest = Some((path, mtime)),
                    Some((_, prev_mtime)) if mtime > *prev_mtime => {
                        latest = Some((path, mtime));
                    }
                    _ => {}
                }
            }
        }

        latest
            .map(|(p, _)| p)
            .ok_or_else(|| CliError::NotFound("no .journal files found in state directory".into()))
    }
}

// ---------------------------------------------------------------------------
// Error Handling
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum CliError {
    #[error("ambiguous path query '{query}': multiple candidates: {candidates:?}")]
    AmbiguousPath {
        query: String,
        candidates: Vec<String>,
    },

    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("fork requires --offset <u64>")]
    MissingOffset,

    #[error("state error: {0}")]
    State(#[from] omp_state::StateError),

    #[error("server error: {0}")]
    Server(#[from] omp_server::ServerError),

    #[error("command error: {0}")]
    Command(#[from] omp_control::command::CommandError),

    #[error("{0}")]
    Host(#[from] StructuredError),

    #[error("convar error: {0}")]
    ConVar(#[from] omp_control::convar::ConVarError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("unsupported transport '{transport}': {reason}")]
    UnsupportedTransport { transport: String, reason: String },
}

impl CliError {
    pub fn to_structured(&self) -> StructuredError {
        match self {
            Self::AmbiguousPath { query, candidates } => {
                let mut err = StructuredError::new(
                    "ambiguous_path",
                    format!("Path query '{query}' is ambiguous ({candidates:?})"),
                    false,
                );
                err.diagnostics = Some(serde_json::json!({
                    "query": query,
                    "candidates": candidates
                }));
                err
            }
            Self::MissingOffset => {
                StructuredError::new("missing_offset", "fork requires --offset <u64>", false)
            }
            Self::NotFound(msg) => StructuredError::new("not_found", msg, false),
            Self::UnsupportedTransport { transport, reason } => {
                let mut err = StructuredError::new("unsupported_transport", reason, false);
                err.diagnostics = Some(serde_json::json!({
                    "transport": transport,
                    "supported": ["tcp"]
                }));
                err
            }
            Self::State(e) => e.structured(),
            Self::Host(error) => error.clone(),
            Self::Server(e) => StructuredError::new("server_error", e.to_string(), false),
            other => StructuredError::new("cli_error", other.to_string(), false),
        }
    }
}

// ---------------------------------------------------------------------------
// CLI Options and Subcommands
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct CommonOptions {
    pub journal: Option<PathBuf>,
    pub json: bool,
    pub workspace: Option<PathBuf>,
    pub session_id: Option<SessionId>,
}

#[derive(Clone, Debug)]
pub enum Subcommand {
    Run {
        common: CommonOptions,
        session_id: Option<SessionId>,
        message: Option<String>,
        profile: Option<String>,
        cfg: Option<PathBuf>,
    },
    Resume {
        common: CommonOptions,
    },
    Fork {
        common: CommonOptions,
        offset: Option<u64>,
    },
    Inspect {
        common: CommonOptions,
        offset: Option<u64>,
        format: String,
    },
    Doctor {
        common: CommonOptions,
        repair: bool,
    },
    Replay {
        common: CommonOptions,
        offset: Option<u64>,
    },
    Serve {
        common: CommonOptions,
        bind: String,
        transport: String,
        once: bool,
    },
}

// ---------------------------------------------------------------------------
// Argument Parsing
// ---------------------------------------------------------------------------

pub fn parse_args<I: Iterator<Item = String>>(mut args: I) -> Result<Option<Subcommand>, CliError> {
    let _prog = args.next(); // skip program name

    let Some(subcmd) = args.next() else {
        return Ok(None);
    };

    let mut common = CommonOptions::default();
    let mut message = None;
    let mut session_id = None;
    let mut profile = None;
    let mut cfg = None;
    let mut offset = None;
    let mut format = "json".to_string();
    let mut repair = false;
    let mut bind = "127.0.0.1:0".to_string();
    let mut transport = "tcp".to_string();
    let mut once = false;

    let mut peekable = args.peekable();
    while let Some(arg) = peekable.next() {
        match arg.as_str() {
            "--journal" => {
                let val = peekable.next().ok_or_else(|| {
                    CliError::InvalidArgument("missing value for --journal".into())
                })?;
                let path = path_policy::expand_tilde(Path::new(&val));
                common.journal = Some(path_policy::validate_unambiguous_path(&path)?);
            }
            "--workspace" => {
                let val = peekable.next().ok_or_else(|| {
                    CliError::InvalidArgument("missing value for --workspace".into())
                })?;
                let path = path_policy::expand_tilde(Path::new(&val));
                common.workspace = Some(path_policy::validate_unambiguous_path(&path)?);
            }
            "--json" => {
                common.json = true;
            }
            "--message" => {
                let val = peekable.next().ok_or_else(|| {
                    CliError::InvalidArgument("missing value for --message".into())
                })?;
                message = Some(val);
            }
            "--session-id" => {
                let val = peekable.next().ok_or_else(|| {
                    CliError::InvalidArgument("missing value for --session-id".into())
                })?;
                session_id = Some(
                    SessionId::new(val).map_err(|e| CliError::InvalidArgument(e.to_string()))?,
                );
            }
            "--profile" => {
                let val = peekable.next().ok_or_else(|| {
                    CliError::InvalidArgument("missing value for --profile".into())
                })?;
                profile = Some(val);
            }
            "--cfg" => {
                let val = peekable
                    .next()
                    .ok_or_else(|| CliError::InvalidArgument("missing value for --cfg".into()))?;
                let path = path_policy::expand_tilde(Path::new(&val));
                cfg = Some(path_policy::validate_unambiguous_path(&path)?);
            }
            "--offset" => {
                let val = peekable.next().ok_or_else(|| {
                    CliError::InvalidArgument("missing value for --offset".into())
                })?;
                let num = val.parse::<u64>().map_err(|_| {
                    CliError::InvalidArgument(format!("invalid integer for --offset: '{val}'"))
                })?;
                offset = Some(num);
            }
            "--format" => {
                let val = peekable.next().ok_or_else(|| {
                    CliError::InvalidArgument("missing value for --format".into())
                })?;
                format = val.to_ascii_lowercase();
            }
            "--repair" => {
                repair = true;
            }
            "--bind" => {
                let val = peekable
                    .next()
                    .ok_or_else(|| CliError::InvalidArgument("missing value for --bind".into()))?;
                bind = val;
            }
            "--transport" => {
                let val = peekable.next().ok_or_else(|| {
                    CliError::InvalidArgument("missing value for --transport".into())
                })?;
                transport = val.to_ascii_lowercase();
            }
            "--once" => {
                once = true;
            }
            "--help" | "-h" => return Ok(None),
            unknown => {
                return Err(CliError::InvalidArgument(format!(
                    "unknown option: '{unknown}'"
                )));
            }
        }
    }

    common.session_id = session_id.clone();
    match subcmd.as_str() {
        "run" => Ok(Some(Subcommand::Run {
            common,
            session_id,
            message,
            profile,
            cfg,
        })),
        "resume" => Ok(Some(Subcommand::Resume { common })),
        "fork" => Ok(Some(Subcommand::Fork { common, offset })),
        "inspect" => Ok(Some(Subcommand::Inspect {
            common,
            offset,
            format,
        })),
        "doctor" => Ok(Some(Subcommand::Doctor { common, repair })),
        "replay" => Ok(Some(Subcommand::Replay { common, offset })),
        "serve" => Ok(Some(Subcommand::Serve {
            common,
            bind,
            transport,
            once,
        })),
        _ => Err(CliError::InvalidArgument(format!(
            "unknown subcommand: '{subcmd}'"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Execution Logic for Subcommands
// ---------------------------------------------------------------------------

fn resolve_workspace_root(common: &CommonOptions) -> Result<PathBuf, CliError> {
    if let Some(ws) = &common.workspace {
        Ok(ws.canonicalize()?)
    } else {
        let cwd = std::env::current_dir()?;
        path_policy::discover_workspace_root(&cwd)
    }
}

fn resolve_target_journal(common: &CommonOptions, workspace: &Path) -> Result<PathBuf, CliError> {
    if let Some(j) = &common.journal {
        path_policy::validate_unambiguous_path(j)
    } else if let Some(session_id) = &common.session_id {
        Ok(path_policy::default_journal_path(workspace, session_id))
    } else {
        let state_dir = path_policy::default_state_dir(workspace);
        path_policy::find_latest_journal(&state_dir)
    }
}

/// Execute a .cfg script against a ConVarStore and record patches.
fn apply_cfg_file(
    path: &Path,
    host: &mut omp_control::SessionHost,
    journal: &mut Journal,
) -> Result<(), CliError> {
    if !path.exists() {
        return Ok(());
    }
    let path = path.canonicalize()?;
    host.execute_command(
        journal,
        &format!("exec {}", serde_json::to_string(&path.to_string_lossy())?),
    )?;
    Ok(())
}

/// Start a separate journal with the current settings, never conversation or task state.
fn create_clean_session(
    workspace: &Path,
    previous: &omp_state::SessionSnapshot,
) -> Result<(Journal, omp_control::SessionHost, PathBuf), CliError> {
    if previous.active_jobs().next().is_some()
        || previous.children(previous.container("actors")).any(|actor| {
            actor.kind == "subagent"
                && !matches!(actor.attributes.get("status"),
                    Some(omp_types::TypedValue::String(status))
                        if matches!(status.as_str(), "succeeded" | "failed" | "cancelled" | "detached"))
        })
    {
        return Err(CliError::Host(StructuredError::new(
            "session_busy",
            "Stop active work before starting a new session",
            true,
        )));
    }
    let mut host =
        omp_control::SessionHost::new(workspace.to_path_buf(), ActorId::new("owner").expect("static actor id"))?;
    host.convars.hydrate_from_dom(previous);
    host.convars = host.convars.seed_child();
    let sid = SessionId::mint();
    let path = path_policy::default_journal_path(workspace, &sid);
    fs::create_dir_all(path_policy::default_state_dir(workspace))?;
    let mut journal = Journal::create(&path, sid)?;
    if let Some(patch) = host.convars.drain_session_patch(
        omp_types::JournalOffset(journal.snapshot().offset),
        journal.next_offset(),
        host.owner.clone(),
        journal.snapshot().container("convars"),
    ) {
        journal.append_patch(patch)?;
    }
    Ok((journal, host, path))
}

fn write_paced_terminal(text: &str) -> Result<(), CliError> {
    let mut pacer = omp_render::StreamPacer::new(omp_render::StreamPacingConfig {
        min_delay_ms: 15,
        chunk_char_target: 128,
        burst_limit: 4096,
    });
    let mut pending = text;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    while !pending.is_empty() || pacer.has_buffered() {
        pending = &pending[pacer.push_chunk(pending)..];
        if let Some(chunk) = pacer.next_drawable_chunk() {
            out.write_all(chunk.as_bytes())?;
            out.flush()?;
        } else {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    Ok(())
}
enum ConsoleOutcome {
    GracefulExit,
    Detached,
}

fn run_console(
    host: &mut omp_control::SessionHost,
    journal: &mut Journal,
) -> Result<ConsoleOutcome, CliError> {
    use std::io::IsTerminal;
    let terminal = std::io::stdout().is_terminal();
    let registry = omp_render::SemanticRegistry::new();
    let mut displayed = std::collections::BTreeSet::new();
    let (send, input) = std::sync::mpsc::sync_channel::<Result<String, std::io::Error>>(8);
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut reader = stdin.lock();
        loop {
            let mut line = String::new();
            match BufRead::read_line(&mut std::io::Read::take(&mut reader, 65_537), &mut line) {
                Ok(0) => break,
                Ok(_) => {
                    if send.send(Ok(line)).is_err() {
                        break;
                    }
                }
                Err(error) => {
                    let _ = send.send(Err(error));
                    break;
                }
            }
        }
    });
    let mut prompted = false;
    let outcome = loop {
        host.tool_host
            .poll_jobs(journal)
            .map_err(|error| CliError::InvalidArgument(error.to_string()))?;
        host.poll_children(journal)?;
        for element in journal.snapshot().get_visible_body() {
            let running = element.attributes.get("status").is_some_and(|status| {
                matches!(
                    status,
                    omp_types::TypedValue::String(value)
                        if matches!(value.as_str(), "queued" | "running" | "cancel_requested")
                )
            });
            if running || !displayed.insert(element.id.clone()) {
                continue;
            }
            let component = registry
                .render_session_element(journal.snapshot(), &element.id)
                .map_err(CliError::InvalidArgument)?;
            if terminal {
                let mut sink = omp_render::AnsiOutSink::new();
                component
                    .render_to_sink(&mut sink, 0)
                    .map_err(|error| CliError::InvalidArgument(error.to_string()))?;
                write_paced_terminal(&sink.into_string())?;
            } else {
                let mut sink = omp_render::StringOutSink::new();
                component
                    .render_to_sink(&mut sink, 0)
                    .map_err(|error| CliError::InvalidArgument(error.to_string()))?;
                println!("{}", sink.into_string());
            }
        }
        if terminal && !prompted {
            print!("omp2> ");
            std::io::stdout().flush()?;
            prompted = true;
        }
        let line = match input.recv_timeout(std::time::Duration::from_millis(25)) {
            Ok(line) => {
                prompted = false;
                line?
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                break ConsoleOutcome::GracefulExit;
            }
        };
        if line.len() > 65_536 {
            return Err(CliError::InvalidArgument(
                "console input exceeds 65536 bytes".into(),
            ));
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/exit" {
            break ConsoleOutcome::GracefulExit;
        }
        if line == "/detach" {
            break ConsoleOutcome::Detached;
        }
        let result = if line == "/new" {
            match create_clean_session(&host.workspace, journal.snapshot()) {
                Ok((next, next_host, path)) => {
                    *journal = next;
                    *host = next_host;
                    displayed.clear();
                    println!(
                        "New session '{}' at '{}'",
                        journal.snapshot().session_id,
                        path.display()
                    );
                    host.initialize_provider(journal)
                }
                Err(error) => Err(error.to_structured()),
            }
        } else if line == "/inspect" {
            println!(
                "{}",
                serde_json::to_string_pretty(&journal.inspect(journal.snapshot().offset)?)?
            );
            Ok(())
        } else if line == "/actors" {
            for actor in journal
                .snapshot()
                .children(journal.snapshot().container("actors"))
            {
                println!("{}", serde_json::to_string(actor)?);
            }
            Ok(())
        } else if let Some(offset) = line.strip_prefix("/fork ") {
            let offset = offset
                .parse()
                .map_err(|_| CliError::InvalidArgument("fork offset must be an integer".into()))?;
            journal.fork_at(offset)?;
            displayed.clear();
            Ok(())
        } else if let Some(command) = line.strip_prefix('/') {
            host.execute_command(journal, command)
        } else {
            host.run_turn(journal, line)
        };
        if let Err(error) = result {
            eprintln!("{error}");
        }
    };
    if matches!(outcome, ConsoleOutcome::GracefulExit) {
        host.shutdown(journal)?;
    }
    Ok(outcome)
}

/// 1. `run`: Create durable session, load configs, record initial message deterministically.
pub fn cmd_run(
    common: CommonOptions,
    session_id: Option<SessionId>,
    message: Option<String>,
    profile: Option<String>,
    cfg: Option<PathBuf>,
) -> Result<(), CliError> {
    let workspace = resolve_workspace_root(&common)?;
    let sid = session_id.unwrap_or_else(SessionId::mint);

    let journal_path = if let Some(j) = common.journal {
        path_policy::validate_unambiguous_path(&j)?
    } else {
        let state_dir = path_policy::default_state_dir(&workspace);
        fs::create_dir_all(&state_dir)?;
        path_policy::default_journal_path(&workspace, &sid)
    };

    if let Some(parent) = journal_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut journal = Journal::create(&journal_path, sid.clone())?;
    let mut host =
        omp_control::SessionHost::new(workspace.clone(), ActorId::new("owner").expect("static actor id"))?;

    // 1. Load default config: configs/config.cfg
    let default_cfg = workspace.join("configs").join("config.cfg");
    apply_cfg_file(&default_cfg, &mut host, &mut journal)?;

    // 2. Load profile cfg if specified: profiles/<profile>.cfg
    if let Some(p) = &profile {
        let profile_cfg = workspace.join("profiles").join(format!("{p}.cfg"));
        if profile_cfg.exists() {
            apply_cfg_file(&profile_cfg, &mut host, &mut journal)?;
        }
    }

    // 3. Load explicit cfg if specified
    if let Some(c) = &cfg {
        apply_cfg_file(c, &mut host, &mut journal)?;
    }
    if !common.json && tui::interactive() {
        let snapshot = journal.snapshot().clone();
        let refresh = provider_configured(&snapshot);
        drop(host);
        drop(journal);
        return tui::run(
            &workspace,
            &journal_path,
            snapshot,
            refresh,
            message.as_deref(),
        );
    }
    if let Err(error) = host.initialize_provider(&mut journal) {
        eprintln!("{}", serde_json::to_string(&error)?);
    }

    if let Some(message) = &message {
        host.run_turn(&mut journal, message)?;
    }

    let snapshot = journal.resume_latest();
    let result = serde_json::json!({
        "status": "created",
        "session_id": sid.as_str(),
        "journal_path": journal_path.display().to_string(),
        "offset": snapshot.offset,
        "branch": snapshot.current_branch().as_str(),
        "turn_count": snapshot.turn_count(),
        "profile": profile,
    });

    if common.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!(
            "Session '{}' initialized at '{}' (offset {}, branch '{}', turns: {})",
            sid,
            journal_path.display(),
            snapshot.offset,
            snapshot.current_branch(),
            snapshot.turn_count()
        );
    }
    if !common.json {
        let _ = run_console(&mut host, &mut journal)?;
    } else {
        host.shutdown(&mut journal)?;
    }
    Ok(())
}

/// 2. `resume`: Open latest session journal.
pub fn cmd_resume(common: CommonOptions) -> Result<(), CliError> {
    let workspace = resolve_workspace_root(&common)?;
    let journal_path = resolve_target_journal(&common, &workspace)?;
    let mut journal = Journal::open(&journal_path)?;
    let mut host =
        omp_control::SessionHost::new(workspace.clone(), ActorId::new("owner").expect("static actor id"))?;
    if !common.json && tui::interactive() {
        let snapshot = journal.snapshot().clone();
        let refresh = provider_configured(&snapshot);
        drop(host);
        drop(journal);
        return tui::run(&workspace, &journal_path, snapshot, refresh, None);
    }
    if let Err(error) = host.initialize_provider(&mut journal) {
        eprintln!("{}", serde_json::to_string(&error)?);
    }
    let snapshot = journal.resume_latest();

    let result = serde_json::json!({
        "status": "resumed",
        "session_id": snapshot.session_id.as_str(),
        "journal_path": journal_path.display().to_string(),
        "offset": snapshot.offset,
        "branch": snapshot.current_branch().as_str(),
        "turn_count": snapshot.turn_count(),
        "active_jobs": snapshot.active_jobs().count(),
        "active_tools": snapshot.active_tool_roster().count(),
    });

    if common.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!(
            "Resumed session '{}' from '{}' (offset {}, branch '{}', turns: {})",
            snapshot.session_id,
            journal_path.display(),
            snapshot.offset,
            snapshot.current_branch(),
            snapshot.turn_count()
        );
    }
    if !common.json {
        let _ = run_console(&mut host, &mut journal)?;
    }
    Ok(())
}

/// 3. `fork`: Requires --offset, forks journal into a new branch.
pub fn cmd_fork(common: CommonOptions, offset: Option<u64>) -> Result<(), CliError> {
    let Some(fork_offset) = offset else {
        return Err(CliError::MissingOffset);
    };

    let workspace = resolve_workspace_root(&common)?;
    let journal_path = resolve_target_journal(&common, &workspace)?;
    let mut journal = Journal::open(&journal_path)?;

    let new_branch = journal.fork_at(fork_offset)?;
    let current_offset = journal.latest_offset();

    let result = serde_json::json!({
        "status": "forked",
        "session_id": journal.snapshot().session_id.as_str(),
        "journal_path": journal_path.display().to_string(),
        "parent_offset": fork_offset,
        "new_offset": current_offset,
        "branch": new_branch.as_str(),
    });

    if common.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!(
            "Forked session at offset {} into new branch '{}' (new offset: {})",
            fork_offset, new_branch, current_offset
        );
    }
    Ok(())
}

/// 4. `inspect`: Print materialized inspect JSON or XML.
pub fn cmd_inspect(
    common: CommonOptions,
    offset: Option<u64>,
    format: String,
) -> Result<(), CliError> {
    let workspace = resolve_workspace_root(&common)?;
    let journal_path = resolve_target_journal(&common, &workspace)?;
    let journal = Journal::open(&journal_path)?;
    let target_offset = offset.unwrap_or_else(|| journal.latest_offset());

    if format == "xml" {
        let snapshot = journal.materialize(target_offset)?;
        println!("{}", snapshot.inspect_xml());
    } else {
        let inspect_val = journal.inspect(target_offset)?;
        println!("{}", serde_json::to_string_pretty(&inspect_val)?);
    }
    Ok(())
}

/// 5. `doctor`: Validate journal recovery, checksums, and protocol compatibility.
pub fn cmd_doctor(common: CommonOptions, repair: bool) -> Result<(), CliError> {
    let workspace = resolve_workspace_root(&common)?;
    let journal_path = resolve_target_journal(&common, &workspace)?;
    let mut journal = Journal::open(&journal_path)?;

    let mut diagnostics = Vec::new();
    let mut is_healthy = true;

    // Check protocol version and format
    let protocol_match = true; // successfully checked during Journal::open header validation
    diagnostics.push(serde_json::json!({
        "check": "protocol_version",
        "status": "ok",
        "version": ProtocolVersion::CURRENT,
        "format": JOURNAL_FORMAT,
    }));

    // Check recovery status
    let recovery_diag = journal.recovery().cloned();
    let mut preserved_suffix_path = None;

    if let Some(diag) = recovery_diag {
        is_healthy = false;
        diagnostics.push(serde_json::json!({
            "check": "journal_integrity",
            "status": "warning",
            "code": diag.code,
            "message": diag.message,
            "last_valid_offset": diag.last_valid_offset,
            "damaged_bytes": diag.damaged_bytes,
        }));

        if repair {
            preserved_suffix_path = journal.repair_suffix()?;
            diagnostics.push(serde_json::json!({
                "check": "repair_status",
                "status": "repaired",
                "preserved_suffix": preserved_suffix_path.as_ref().map(|p| p.display().to_string()),
            }));
            is_healthy = true;
        }
    } else {
        diagnostics.push(serde_json::json!({
            "check": "journal_integrity",
            "status": "ok",
            "valid_records": journal.records().count(),
            "latest_offset": journal.latest_offset(),
        }));
    }

    let report = serde_json::json!({
        "status": if is_healthy { "healthy" } else { "degraded" },
        "session_id": journal.snapshot().session_id.as_str(),
        "journal_path": journal_path.display().to_string(),
        "protocol_ok": protocol_match,
        "diagnostics": diagnostics,
        "repaired": preserved_suffix_path.is_some(),
    });

    if common.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "Journal Doctor: {} (Session: {}, Latest Offset: {})",
            if is_healthy { "HEALTHY" } else { "DEGRADED" },
            journal.snapshot().session_id,
            journal.latest_offset()
        );
        for diag in diagnostics {
            println!(" - [{}] {}: {}", diag["status"], diag["check"], diag);
        }
    }
    Ok(())
}

/// 6. `replay`: Print ancestry chain and selected branch.
pub fn cmd_replay(common: CommonOptions, offset: Option<u64>) -> Result<(), CliError> {
    let workspace = resolve_workspace_root(&common)?;
    let journal_path = resolve_target_journal(&common, &workspace)?;
    let journal = Journal::open(&journal_path)?;
    let target_offset = offset.unwrap_or_else(|| journal.latest_offset());

    let ancestry = journal.branch_ancestry(target_offset)?;
    let branch = journal.snapshot().selected_branch.clone();

    let records_info: Vec<_> = ancestry
        .iter()
        .filter_map(|off| {
            journal.records().find(|r| r.offset == *off).map(|r| {
                serde_json::json!({
                    "offset": r.offset,
                    "parent_offset": r.parent_offset,
                    "branch_id": r.branch_id.as_str(),
                    "reason": r.patch.reason,
                    "timestamp_ms": r.timestamp_ms,
                })
            })
        })
        .collect();

    let result = serde_json::json!({
        "status": "ok",
        "session_id": journal.snapshot().session_id.as_str(),
        "target_offset": target_offset,
        "selected_branch": branch.as_str(),
        "ancestry_offsets": ancestry,
        "chain": records_info,
    });

    if common.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!(
            "Session '{}' Replay (Offset {}, Branch '{}'):",
            journal.snapshot().session_id,
            target_offset,
            branch
        );
        println!("Ancestry: {:?}", ancestry);
        for rec in &records_info {
            println!(
                "  [{}] branch: {}, reason: {}",
                rec["offset"], rec["branch_id"], rec["reason"]
            );
        }
    }
    Ok(())
}

/// 7. `serve`: Host-owned local service lifecycle abstraction.
pub fn cmd_serve(
    common: CommonOptions,
    bind: String,
    transport: String,
    once: bool,
) -> Result<(), CliError> {
    if transport != "tcp" {
        return Err(CliError::UnsupportedTransport {
            transport,
            reason: "only tcp is supported".into(),
        });
    }
    let workspace = resolve_workspace_root(&common)?;
    let path = resolve_target_journal(&common, &workspace).or_else(|_| {
        let sid = SessionId::mint();
        Ok::<_, CliError>(path_policy::default_journal_path(&workspace, &sid))
    })?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let journal = if path.exists() {
        Journal::open(path)?
    } else {
        Journal::create(path, SessionId::mint())?
    };
    omp_server::transport::serve(journal, workspace, &bind, once, common.json)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Main Entrypoint
// ---------------------------------------------------------------------------

fn provider_configured(snapshot: &omp_state::SessionSnapshot) -> bool {
    ["ai_endpoint", "ai_provider"]
        .iter()
        .any(|name| !view::setting(snapshot, name).is_empty())
        || [
            "OMP_ENDPOINT",
            "AI_ENDPOINT",
            "OPENAI_BASE_URL",
            "AI_PROVIDER",
            "OMP_PROVIDER",
        ]
        .iter()
        .any(|name| std::env::var(name).is_ok_and(|value| !value.trim().is_empty()))
}

fn print_usage() {
    println!(
        r#"omp2 - Systems-oriented agent harness

LAUNCH:
    omp2                  Start the interactive terminal in the current workspace
    omp2 run              Equivalent explicit form

SUBCOMMANDS:
    run       Create a durable session with optional deterministic --message
    resume    Resume latest session from workspace state directory
    fork      Fork a session branch at --offset <u64>
    inspect   Inspect materialized session state as JSON or XML
    doctor    Validate journal protocol compatibility and recovery diagnostics
    replay    Display branch ancestry chain and selected branch
    serve     Run host-owned local service abstraction over TCP

OPTIONS:
    --journal <path>     Explicit journal file path
    --workspace <path>   Explicit workspace root
    --offset <u64>       Journal offset (required for fork)
    --message <text>     Deterministic initial user message (for run)
    --profile <name>     Profile cfg to load (e.g. default, factory, remote)
    --cfg <path>         Custom .cfg script to execute
    --format <json|xml>  Inspect output format (default: json)
    --repair             Repair damaged journal suffix in doctor
    --bind <addr>        Service bind address (default: 127.0.0.1:0)
    --transport <tcp>    Transport protocol (default: tcp)
    --once               Handle one connection then exit (serve)
    --json               Output structured JSON
    -h, --help           Print help information

INTERACTIVE TERMINAL:
    run / resume open the styled interface when stdin and stdout are terminals.
    / opens the command menu; type to filter, arrows select, Tab fills, Enter accepts.
    /model selects an advertised model; /help shows keyboard shortcuts.
    Responses and provider-exposed thinking stream live; /thinking toggles reasoning visibility.
    Interrupted output stays visible as partial, not a completed answer.
    Enter sends; Ctrl+J inserts a newline. Bracketed paste remains an editable draft.
    Wheel, PgUp, and PgDn scroll history; Ctrl+Home/End jump to ends.
    /new starts a clean session, keeping settings and preserving the previous journal.
    Esc stops running work. /exit closes the session and restores the console.
    Redirected streams and TERM=dumb retain the plain line-oriented interface.

PROVIDERS:
    Set OMP_ENDPOINT and OMP_API_KEY before run; no model preset is selected.
    Startup fetches available models and advertised settings from the endpoint.
    /provider                 Show catalog and effective settings
    /provider <endpoint>      Configure endpoint using OMP_API_KEY
    /provider refresh         Refetch catalog and settings
    /provider select <id>     Choose a discovered model
    /provider --help          Show adapter and key-environment options
    API keys stay in host environment variables, never in console commands.
    Missing provider metadata remains unknown; no model limits are guessed.
"#
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args
        .get(1)
        .is_some_and(|argument| argument == "__ui-worker")
    {
        if args.len() != 4 {
            return ExitCode::FAILURE;
        }
        return match worker::run_worker(Path::new(&args[2]), Path::new(&args[3])) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("{}", error.to_structured());
                ExitCode::FAILURE
            }
        };
    }
    if args.len() == 2 && matches!(args[1].as_str(), "--help" | "-h") {
        print_usage();
        return ExitCode::SUCCESS;
    }
    let default_run = args.len() == 1;
    let parsed = match parse_args(args.into_iter()) {
        Ok(cmd) => cmd,
        Err(err) => {
            eprintln!("Error: {err}");
            let structured = err.to_structured();
            if let Ok(json_err) = serde_json::to_string(&structured) {
                eprintln!("{json_err}");
            }
            return ExitCode::from(1);
        }
    };

    let cmd = match parsed {
        Some(cmd) => cmd,
        None if default_run => Subcommand::Run {
            common: CommonOptions::default(),
            session_id: None,
            message: None,
            profile: None,
            cfg: None,
        },
        None => {
            print_usage();
            return ExitCode::SUCCESS;
        }
    };

    let result = match cmd {
        Subcommand::Run {
            common,
            session_id,
            message,
            profile,
            cfg,
        } => cmd_run(common, session_id, message, profile, cfg),
        Subcommand::Resume { common } => cmd_resume(common),
        Subcommand::Fork { common, offset } => cmd_fork(common, offset),
        Subcommand::Inspect {
            common,
            offset,
            format,
        } => cmd_inspect(common, offset, format),
        Subcommand::Doctor { common, repair } => cmd_doctor(common, repair),
        Subcommand::Replay { common, offset } => cmd_replay(common, offset),
        Subcommand::Serve {
            common,
            bind,
            transport,
            once,
        } => cmd_serve(common, bind, transport, once),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("Error: {err}");
            let structured = err.to_structured();
            if let Ok(json_err) = serde_json::to_string(&structured) {
                eprintln!("{json_err}");
            }
            ExitCode::from(1)
        }
    }
}
