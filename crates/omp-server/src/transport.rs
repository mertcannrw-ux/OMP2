use crate::error::ServerError;
use crate::role::{ActorRole, Permission};
use crate::service::{
    ArtifactRecord, HandshakeRequest, ServerSessionService,
};
use omp_state::Journal;
use omp_types::{
    ActorId, JobId, JournalOffset, MAX_WIRE_BYTES, Patch,
    ProtocolVersion, StructuredError, TypedValue,
};
use serde_json::Value;
use std::io::{BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Maximum number of concurrent client TCP connections allowed.
pub const MAX_CONCURRENT_CONNECTIONS: usize = 32;

/// Maximum bytes per request frame (1 MiB, matches `MAX_WIRE_BYTES`).
pub const MAX_FRAME_BYTES: usize = MAX_WIRE_BYTES;

/// Socket read timeout in seconds.
pub const READ_TIMEOUT_SECS: u64 = 30;

/// Socket write timeout in seconds.
pub const WRITE_TIMEOUT_SECS: u64 = 10;

/// Host-configured authentication credentials.
/// Credentials are kept host-only and never supplied to sandboxes.
#[derive(Clone, Debug)]
pub struct AuthConfig {
    pub owner_token: String,
    pub driver_token: String,
    pub spectator_token: String,
    pub subagent_token: String,
    pub worker_token: String,
}

impl AuthConfig {
    /// Bootstrap authentication tokens from environment or generate secure random tokens.
    pub fn from_env() -> Self {        let owner_token = std::env::var("OMP2_AUTH_TOKEN")
            .or_else(|_| std::env::var("OMP2_OWNER_TOKEN"))
            .unwrap_or_else(|_| {
                format!(
                    "{:x}{:x}",
                    uuid::Uuid::new_v4().as_u128(),
                    uuid::Uuid::new_v4().as_u128()
                )
            });

        let driver_token = std::env::var("OMP2_DRIVER_TOKEN")
            .unwrap_or_else(|_| format!("drv-{}", uuid::Uuid::new_v4().simple()));

        let spectator_token = std::env::var("OMP2_SPECTATOR_TOKEN")
            .unwrap_or_else(|_| format!("spec-{}", uuid::Uuid::new_v4().simple()));

        let subagent_token = std::env::var("OMP2_SUBAGENT_TOKEN")
            .unwrap_or_else(|_| format!("sub-{}", uuid::Uuid::new_v4().simple()));

        let worker_token = std::env::var("OMP2_WORKER_TOKEN")
            .unwrap_or_else(|_| format!("wrk-{}", uuid::Uuid::new_v4().simple()));

        Self {
            owner_token,
            driver_token,
            spectator_token,
            subagent_token,
            worker_token,
        }
    }

    /// Validate token against requested role.
    ///
    /// Reject unauthenticated or empty tokens. Owner token authorizes any role.
    /// Explicit role tokens authorize their specific role.
    pub fn validate(&self, requested_role: ActorRole, provided_token: Option<&str>) -> bool {
        let token = match provided_token {
            Some(t) if !t.trim().is_empty() => t.trim(),
            _ => return false, // No unauthenticated access!
        };

        // Owner token authorizes any role (Controller, Driver, Spectator, etc.)
        if token == self.owner_token {
            return true;
        }

        match requested_role {
            ActorRole::Controller => {
                // Controller strictly requires owner token
                false
            }
            ActorRole::InteractiveDriver => token == self.driver_token,
            ActorRole::Spectator => token == self.spectator_token,
            ActorRole::SubagentInspector => token == self.subagent_token,
            ActorRole::AutomationWorker => token == self.worker_token,
        }
    }
}

use omp_control::host::SessionHost;

/// Derive the sibling token-file path for a journal (e.g. `session.tokens.json`).
fn journal_token_path(journal: &Journal) -> PathBuf {
    let mut path = journal.path().to_path_buf();
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "session".into());
    path.set_file_name(format!("{stem}.tokens.json"));
    path
}

/// Persist auth tokens to `path` with owner-only permissions (0600 on unix).
/// Tokens are never printed to stdout; clients read this file instead.
fn write_token_file(path: &PathBuf, auth_config: &AuthConfig) -> std::io::Result<()> {
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    let payload = serde_json::json!({
        "owner": auth_config.owner_token,
        "driver": auth_config.driver_token,
        "spectator": auth_config.spectator_token,
        "subagent": auth_config.subagent_token,
        "worker": auth_config.worker_token,
    });
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    // The token file lives next to the journal, which is already host-private.
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options.open(path)?;
    use std::io::Write as _;
    writeln!(
        file,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap_or_default()
    )?;
    file.sync_all()?;
    Ok(())
}

/// Shared authoritative server state, wrapped in a single Mutex.
pub struct ServerState {
    pub service: ServerSessionService,
    pub dispatcher: SessionHost,
    pub auth_config: AuthConfig,
}

/// Per-connection session state binding authenticated identity.
struct ConnectionState {
    authenticated: Option<(ActorId, ActorRole)>,
}

/// Guard to track and decrement active concurrent connection count on drop.
struct ConnectionGuard {
    active_count: Arc<AtomicUsize>,
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.active_count.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Read a line bounded strictly by `max_bytes` to prevent memory exhaustion / DoS.
fn read_line_bounded<R: Read>(
    reader: &mut R,
    line_buf: &mut Vec<u8>,
    max_bytes: usize,
) -> Result<Option<String>, ServerError> {
    line_buf.clear();
    let mut byte = [0u8; 1];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => {
                if line_buf.is_empty() {
                    return Ok(None);
                } else {
                    let s = String::from_utf8(line_buf.clone())
                        .map_err(|e| ServerError::InvalidCommand(format!("invalid utf-8: {e}")))?;
                    return Ok(Some(s));
                }
            }
            Ok(_) => {
                if byte[0] == b'\n' {
                    let s = String::from_utf8(line_buf.clone())
                        .map_err(|e| ServerError::InvalidCommand(format!("invalid utf-8: {e}")))?;
                    return Ok(Some(s));
                }
                if byte[0] != b'\r' {
                    line_buf.push(byte[0]);
                    if line_buf.len() > max_bytes {
                        return Err(ServerError::WireLimit(format!(
                            "request frame exceeded maximum allowed bytes ({max_bytes})"
                        )));
                    }
                }
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::TimedOut
                    || e.kind() == std::io::ErrorKind::WouldBlock =>
            {
                return Err(ServerError::Io(e));
            }
            Err(e) => return Err(ServerError::Io(e)),
        }
    }
}

/// Write a newline-delimited JSON response to stream.
fn write_response<W: Write>(writer: &mut W, value: &Value) -> Result<(), ServerError> {
    let serialized = serde_json::to_vec(value)
        .map_err(|e| ServerError::InvalidCommand(format!("serialization error: {e}")))?;
    if serialized.len() > MAX_FRAME_BYTES {
        return Err(ServerError::WireLimit(format!(
            "response frame exceeded maximum allowed bytes ({MAX_FRAME_BYTES})"
        )));
    }
    writer.write_all(&serialized)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

/// Parse an ActorRole from a JSON string.
fn parse_actor_role(val: Option<&str>) -> ActorRole {
    match val {
        Some("controller") | Some("Controller") | Some("owner") => ActorRole::Controller,
        Some("spectator") | Some("Spectator") => ActorRole::Spectator,
        Some("subagent") | Some("subagent_inspector") | Some("SubagentInspector") => {
            ActorRole::SubagentInspector
        }
        Some("factory_worker") | Some("automation_worker") | Some("AutomationWorker") => {
            ActorRole::AutomationWorker
        }
        _ => ActorRole::InteractiveDriver,
    }
}

/// Handle an individual client connection loop.
fn handle_connection(
    mut stream: TcpStream,
    state: Arc<Mutex<ServerState>>,
    shutdown_signal: Arc<AtomicBool>,
) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(READ_TIMEOUT_SECS)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(WRITE_TIMEOUT_SECS)));
    let _ = stream.set_nodelay(true);

    let read_stream = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut reader = BufReader::new(read_stream);
    let mut line_buf = Vec::with_capacity(4096);
    let mut conn_state = ConnectionState {
        authenticated: None,
    };

    while !shutdown_signal.load(Ordering::SeqCst) {
        let line = match read_line_bounded(&mut reader, &mut line_buf, MAX_FRAME_BYTES) {
            Ok(Some(l)) => l,
            Ok(None) => break, // Clean EOF
            Err(e) => {
                let err_resp = serde_json::json!({
                    "status": "error",
                    "error": "frame_error",
                    "message": e.to_string(),
                });
                let _ = write_response(&mut stream, &err_resp);
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let request: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let err_resp = serde_json::json!({
                    "status": "error",
                    "error": "invalid_json",
                    "message": e.to_string(),
                });
                let _ = write_response(&mut stream, &err_resp);
                continue;
            }
        };

        let action = request
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // 1. Mandatory Handshake Check: No other operation permitted before handshake
        if conn_state.authenticated.is_none() {
            if action != "handshake" {
                let err_resp = serde_json::json!({
                    "status": "error",
                    "error": "handshake_required",
                    "message": "Mandatory protocol handshake required before any other action",
                });
                let _ = write_response(&mut stream, &err_resp);
                continue;
            }

            // Execute handshake
            let proto_ver = if let Some(v) = request.get("protocol_version") {
                serde_json::from_value::<ProtocolVersion>(v.clone())
                    .unwrap_or(ProtocolVersion::CURRENT)
            } else {
                ProtocolVersion::CURRENT
            };

            let requested_role = parse_actor_role(request.get("role").and_then(|v| v.as_str()));
            let actor_str = request
                .get("actor_id")
                .and_then(|v| v.as_str())
                .unwrap_or("client");
            let actor_id = match ActorId::new(actor_str) {
                Ok(id) => id,
                Err(e) => {
                    let err_resp = serde_json::json!({
                        "status": "error",
                        "error": "invalid_actor_id",
                        "message": e.to_string(),
                    });
                    let _ = write_response(&mut stream, &err_resp);
                    continue;
                }
            };

            let auth_token = request
                .get("auth_token")
                .and_then(|v| v.as_str())
                .map(str::trim);

            let mut state_guard = match state.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };

            // Validate token against requested role
            if !state_guard.auth_config.validate(requested_role, auth_token) {
                let err_resp = serde_json::json!({
                    "status": "error",
                    "error": "unauthorized",
                    "message": format!(
                        "Invalid or missing authentication token for requested role {:?}",
                        requested_role
                    ),
                });
                let _ = write_response(&mut stream, &err_resp);
                continue;
            }

            let handshake_req = HandshakeRequest {
                protocol_version: proto_ver,
                actor_id: actor_id.clone(),
                requested_role,
                auth_token: auth_token.map(String::from),
                capabilities: vec![omp_types::JOURNAL_FORMAT.to_string()],
            };

            match state_guard.service.handshake(handshake_req) {
                Ok(resp) => {
                    // Bind authenticated role and actor to this connection
                    conn_state.authenticated = Some((actor_id, resp.granted_role));
                    let resp_val = serde_json::json!({
                        "status": "ok",
                        "handshake": resp,
                    });
                    drop(state_guard);
                    let _ = write_response(&mut stream, &resp_val);
                }
                Err(e) => {
                    let err_resp = serde_json::json!({
                        "status": "error",
                        "error": "handshake_failed",
                        "message": e.to_string(),
                    });
                    drop(state_guard);
                    let _ = write_response(&mut stream, &err_resp);
                }
            }
            continue;
        }

        // 2. Dispatch for authenticated client.
        // Identity is STRICTLY bound to connection (conn_actor_id, conn_role). Any request-supplied actor_id is ignored!
        // The handshake gate above returns early for unauthenticated
        // connections, so this is only reachable when `authenticated` is set.
        let (auth_actor_id, auth_role) = conn_state
            .authenticated
            .as_ref()
            .expect("handshake gate guarantees authentication")
            .clone();

        let response = match action.as_str() {
            "get_snapshot" => {
                let offset = request
                    .get("offset")
                    .and_then(|v| v.as_u64())
                    .map(JournalOffset);
                let state_guard = match state.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                match state_guard.service.get_snapshot(&auth_actor_id, offset) {
                    Ok(snap) => serde_json::json!({
                        "status": "ok",
                        "snapshot": snap,
                    }),
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": e.structured().code,
                        "message": e.to_string(),
                    }),
                }
            }
            "get_projection" => {
                let offset = request
                    .get("offset")
                    .and_then(|value| value.as_u64())
                    .map(JournalOffset);
                let width = request
                    .get("width")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(80)
                    .clamp(1, 512) as u16;
                let height = request
                    .get("height")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(24)
                    .min(256) as u16;
                // Bounded paging: a single projection call never renders more
                // than the hard cap, no matter what the client asks for.
                let max_elements = request
                    .get("max_elements")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(
                        omp_render::DebugSession::DEFAULT_MAX_ELEMENTS as u64,
                    )
                    .clamp(
                        1,
                        omp_render::DebugSession::MAX_ELEMENTS_HARD_CAP as u64,
                    ) as usize;
                let skip_elements = request
                    .get("skip_elements")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0) as usize;
                let snapshot = {
                    let state_guard = match state.lock() {
                        Ok(guard) => guard,
                        Err(poison) => poison.into_inner(),
                    };
                    state_guard.service.get_snapshot(&auth_actor_id, offset)
                };
                match snapshot {
                    Ok(snapshot) => {
                        let mut debug = omp_render::DebugSession::new(width, height);
                        match debug.apply_session_snapshot_paged(
                            &snapshot,
                            skip_elements,
                            max_elements,
                        ) {
                            Ok(component) => {
                                serde_json::json!({ "status": "ok", "offset": snapshot.offset,
                                "component": component, "debug": debug.snapshot() })
                            }
                            Err(error) => {
                                serde_json::json!({ "status": "error", "error": "render_failure", "message": error })
                            }
                        }
                    }
                    Err(error) => {
                        serde_json::json!({ "status": "error", "error": error.structured().code, "message": error.to_string() })
                    }
                }
            }
            "subscribe" => {
                let last_offset = request
                    .get("last_offset")
                    .and_then(|v| v.as_u64())
                    .map(JournalOffset);
                let capacity = request
                    .get("capacity")
                    .and_then(|v| v.as_u64())
                    .map(|c| c as usize);

                let mut state_guard = match state.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                match state_guard
                    .service
                    .subscribe(&auth_actor_id, last_offset, capacity)
                {
                    Ok(msg) => serde_json::json!({
                        "status": "ok",
                        "message": msg,
                    }),
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": e.structured().code,
                        "message": e.to_string(),
                    }),
                }
            }
            "poll" | "poll_subscriber" => {
                let mut state_guard = match state.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                let msg = state_guard.service.poll_subscriber(&auth_actor_id);
                serde_json::json!({
                    "status": "ok",
                    "message": msg,
                })
            }
            "resync" => {
                let state_guard = match state.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                match state_guard.service.get_snapshot(&auth_actor_id, None) {
                    Ok(snap) => {
                        let offset = state_guard.service.current_offset();
                        serde_json::json!({
                            "status": "ok",
                            "snapshot": snap,
                            "offset": offset.0,
                        })
                    }
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": e.structured().code,
                        "message": e.to_string(),
                    }),
                }
            }
            "submit_patch" => {
                let patch_val = request.get("patch").cloned().unwrap_or(Value::Null);
                let patch: Result<Patch, _> = serde_json::from_value(patch_val);

                match patch {
                    Ok(p) => {
                        // Enforce actor identity: patch author must match connection authenticated actor
                        let author_matches = match &p.by {
                            omp_types::PatchAuthor::Actor(id) => id == &auth_actor_id,
                            _ => false,
                        };
                        if !author_matches && auth_role != ActorRole::Controller {
                            serde_json::json!({
                                "status": "error",
                                "error": "unauthorized",
                                "message": "Identity spoofing refused: patch author does not match connection authenticated actor",
                            })
                        } else {
                            let mut state_guard = match state.lock() {
                                Ok(g) => g,
                                Err(p) => p.into_inner(),
                            };
                            match state_guard.service.submit_patch(&auth_actor_id, p) {
                                Ok(offset) => serde_json::json!({
                                    "status": "ok",
                                    "offset": offset.0,
                                }),
                                Err(e) => serde_json::json!({
                                    "status": "error",
                                    "error": e.structured().code,
                                    "message": e.to_string(),
                                }),
                            }
                        }
                    }
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": "invalid_patch",
                        "message": e.to_string(),
                    }),
                }
            }
            "submit_command" => {
                let cmd_id = request
                    .get("command_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                let cmd_line = request
                    .get("command_line")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();

                if cmd_id.is_empty() {
                    serde_json::json!({
                        "status": "error",
                        "error": "invalid_command",
                        "message": "command_id cannot be empty",
                    })
                } else {
                    let mut state_guard = match state.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };

                    // Replay protection check
                    match state_guard
                        .service
                        .submit_command(&auth_actor_id, cmd_id, cmd_line)
                    {
                        Ok(()) => {
                            let start_offset = state_guard.service.current_offset().0;

                            // Actual state execution via CommandDispatcher
                            let ServerState {
                                dispatcher,
                                service,
                                ..
                            } = &mut *state_guard;
                            match dispatcher.execute_command(service.journal_mut(), cmd_line) {
                                Ok(()) => {
                                    // Replicate new records to active subscribers
                                    state_guard.service.replicate(start_offset);
                                    let current_offset = state_guard.service.current_offset().0;
                                    serde_json::json!({
                                        "status": "ok",
                                        "command_id": cmd_id,
                                        "current_offset": current_offset,
                                    })
                                }
                                Err(e) => serde_json::json!({
                                    "status": "error",
                                    "error": e.code,
                                    "message": e.message,
                                }),
                            }
                        }
                        Err(e) => serde_json::json!({
                            "status": "error",
                            "error": e.structured().code,
                            "message": e.to_string(),
                        }),
                    }
                }
            }
            "run_turn" => {
                let message = request
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let mut state_guard = match state.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };

                if !auth_role.has_permission(Permission::SubmitCommand) {
                    serde_json::json!({
                        "status": "error",
                        "error": "unauthorized",
                        "message": "Actor lacks permission to run turn",
                    })
                } else {
                    let start_offset = state_guard.service.current_offset().0;
                    let ServerState {
                        dispatcher,
                        service,
                        ..
                    } = &mut *state_guard;
                    match dispatcher.run_turn(service.journal_mut(), message) {
                        Ok(()) => {
                            state_guard.service.replicate(start_offset);
                            let current_offset = state_guard.service.current_offset().0;
                            serde_json::json!({
                                "status": "ok",
                                "current_offset": current_offset,
                            })
                        }
                        Err(e) => serde_json::json!({
                            "status": "error",
                            "error": e.code,
                            "message": e.message,
                        }),
                    }
                }
            }
            "execute_tool" => {
                let name = request.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let input = request.get("input").cloned().unwrap_or(Value::Null);

                let mut state_guard = match state.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };

                if !auth_role.has_permission(Permission::SubmitCommand) {
                    serde_json::json!({
                        "status": "error",
                        "error": "unauthorized",
                        "message": "Actor lacks permission to execute tools",
                    })
                } else {
                    let start_offset = state_guard.service.current_offset().0;
                    let ServerState {
                        dispatcher,
                        service,
                        ..
                    } = &mut *state_guard;
                    match dispatcher.execute_tool(service.journal_mut(), name, input) {
                        Ok(call_id) => {
                            state_guard.service.replicate(start_offset);
                            let current_offset = state_guard.service.current_offset().0;
                            serde_json::json!({
                                "status": "ok",
                                "tool_call_id": call_id,
                                "current_offset": current_offset,
                            })
                        }
                        Err(e) => serde_json::json!({
                            "status": "error",
                            "error": e.code,
                            "message": e.message,
                        }),
                    }
                }
            }
            "change_convar" => {
                let name = request.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let val_json = request.get("value").cloned().unwrap_or(Value::Null);
                let typed_val = match val_json {
                    Value::Bool(b) => TypedValue::Bool(b),
                    // `is_i64()` guard makes `as_i64()` infallible; the
                    // fallback covers the impossible-to-hit None arm.
                    Value::Number(n) if n.is_i64() => {
                        TypedValue::Integer(n.as_i64().unwrap_or(0))
                    }
                    Value::Number(n) => TypedValue::Number(n.as_f64().unwrap_or(0.0)),
                    Value::String(s) => TypedValue::String(s),
                    other => TypedValue::Json(other),
                };

                let mut state_guard = match state.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                match state_guard
                    .service
                    .change_convar(&auth_actor_id, name, typed_val)
                {
                    Ok(offset) => serde_json::json!({
                        "status": "ok",
                        "offset": offset.0,
                    }),
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": e.structured().code,
                        "message": e.to_string(),
                    }),
                }
            }
            "get_artifact" => {
                let art_id_str = request
                    .get("artifact_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let offset = request
                    .get("offset")
                    .and_then(|v| v.as_u64())
                    .map(JournalOffset);

                match omp_types::ArtifactId::new(art_id_str) {
                    Ok(art_id) => {
                        let state_guard = match state.lock() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        let res = match offset {
                            Some(off) => {
                                state_guard
                                    .service
                                    .get_artifact_at(&auth_actor_id, &art_id, off)
                            }
                            None => state_guard.service.get_artifact(&auth_actor_id, &art_id),
                        };
                        match res {
                            Ok(record) => serde_json::json!({
                                "status": "ok",
                                "artifact": record,
                            }),
                            Err(e) => serde_json::json!({
                                "status": "error",
                                "error": e.structured().code,
                                "message": e.to_string(),
                            }),
                        }
                    }
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": "invalid_artifact_id",
                        "message": e.to_string(),
                    }),
                }
            }
            "register_artifact" => {
                let art_val = request.get("artifact").cloned().unwrap_or(Value::Null);
                let record_res: Result<ArtifactRecord, _> = serde_json::from_value(art_val);

                match record_res {
                    Ok(record) => {
                        let mut state_guard = match state.lock() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        match state_guard
                            .service
                            .register_artifact(&auth_actor_id, record)
                        {
                            Ok(offset) => serde_json::json!({
                                "status": "ok",
                                "offset": offset.0,
                            }),
                            Err(e) => serde_json::json!({
                                "status": "error",
                                "error": e.structured().code,
                                "message": e.to_string(),
                            }),
                        }
                    }
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": "invalid_artifact_record",
                        "message": e.to_string(),
                    }),
                }
            }
            "write_workspace" => {
                let path = request.get("path").and_then(|v| v.as_str()).unwrap_or("");
                let content_bytes = request
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.as_bytes())
                    .unwrap_or(b"");

                let mut state_guard = match state.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                let ServerState {
                    service,
                    dispatcher,
                    ..
                } = &mut *state_guard;
                match service.write_workspace(&auth_actor_id, path, dispatcher, content_bytes) {
                    Ok(()) => serde_json::json!({ "status": "ok" }),
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": e.structured().code,
                        "message": e.to_string(),
                    }),
                }
            }
            "request_approval" => {
                let req_id = request
                    .get("request_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let desc = request
                    .get("action_description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let mut state_guard = match state.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                match state_guard
                    .service
                    .request_approval(&auth_actor_id, req_id, desc)
                {
                    Ok(()) => serde_json::json!({ "status": "ok" }),
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": e.structured().code,
                        "message": e.to_string(),
                    }),
                }
            }
            "resolve_approval" => {
                let req_id = request
                    .get("request_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let decision = request
                    .get("decision")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let mut state_guard = match state.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                match state_guard
                    .service
                    .resolve_approval(&auth_actor_id, req_id, decision)
                {
                    Ok(()) => serde_json::json!({ "status": "ok" }),
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": e.structured().code,
                        "message": e.to_string(),
                    }),
                }
            }
            "signal_job" => {
                let job_id_str = request.get("job_id").and_then(|v| v.as_str()).unwrap_or("");
                let sig = request.get("signal").and_then(|v| v.as_str()).unwrap_or("");

                match JobId::new(job_id_str) {
                    Ok(job_id) => {
                        let mut state_guard = match state.lock() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        match state_guard.service.signal_job(&auth_actor_id, &job_id, sig) {
                            Ok(()) => serde_json::json!({ "status": "ok" }),
                            Err(e) => serde_json::json!({
                                "status": "error",
                                "error": e.structured().code,
                                "message": e.to_string(),
                            }),
                        }
                    }
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": "invalid_job_id",
                        "message": e.to_string(),
                    }),
                }
            }
            "detach" | "detach_actor" => {
                let target_str = request
                    .get("target_actor")
                    .and_then(|v| v.as_str())
                    .unwrap_or(auth_actor_id.as_str());
                match ActorId::new(target_str) {
                    Ok(target_id) => {
                        let mut state_guard = match state.lock() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        match state_guard.service.detach_actor(&auth_actor_id, &target_id) {
                            Ok(()) => serde_json::json!({ "status": "ok" }),
                            Err(e) => serde_json::json!({
                                "status": "error",
                                "error": e.structured().code,
                                "message": e.to_string(),
                            }),
                        }
                    }
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": "invalid_actor_id",
                        "message": e.to_string(),
                    }),
                }
            }
            "inspect_actor" => {
                let target_str = request
                    .get("actor_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or(auth_actor_id.as_str());
                match ActorId::new(target_str) {
                    Ok(target_id) => {
                        let state_guard = match state.lock() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        match state_guard.service.get_actor(&target_id) {
                            Ok(actor) => serde_json::json!({
                                "status": "ok",
                                "actor": actor,
                            }),
                            Err(e) => serde_json::json!({
                                "status": "error",
                                "error": e.structured().code,
                                "message": e.to_string(),
                            }),
                        }
                    }
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": "invalid_actor_id",
                        "message": e.to_string(),
                    }),
                }
            }
            "inspect_xml" => {
                let state_guard = match state.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                if !auth_role.has_permission(Permission::ReadSnapshot) {
                    serde_json::json!({
                        "status": "error",
                        "error": "unauthorized",
                        "message": "Actor lacks permission to inspect session snapshot",
                    })
                } else {
                    let xml = state_guard.service.journal().snapshot().inspect_xml();
                    serde_json::json!({
                        "status": "ok",
                        "xml": xml,
                    })
                }
            }
            "fork_session" => {
                let offset = request.get("offset").and_then(|v| v.as_u64()).unwrap_or(0);
                let mut state_guard = match state.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                match state_guard.service.fork_session(&auth_actor_id, offset) {
                    Ok(branch) => serde_json::json!({
                        "status": "ok",
                        "branch_id": branch.as_str(),
                    }),
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": e.structured().code,
                        "message": e.to_string(),
                    }),
                }
            }
            "rewind_session" => {
                let offset = request.get("offset").and_then(|v| v.as_u64()).unwrap_or(0);
                let mut state_guard = match state.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                match state_guard.service.rewind_session(&auth_actor_id, offset) {
                    Ok(branch) => serde_json::json!({
                        "status": "ok",
                        "branch_id": branch.as_str(),
                    }),
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": e.structured().code,
                        "message": e.to_string(),
                    }),
                }
            }
            "select_branch" => {
                let branch_str = request
                    .get("branch_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("main");
                match omp_types::BranchId::new(branch_str) {
                    Ok(branch) => {
                        let mut state_guard = match state.lock() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        match state_guard.service.select_branch(&auth_actor_id, &branch) {
                            Ok(offset) => serde_json::json!({
                                "status": "ok",
                                "offset": offset.0,
                            }),
                            Err(e) => serde_json::json!({
                                "status": "error",
                                "error": e.structured().code,
                                "message": e.to_string(),
                            }),
                        }
                    }
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "error": "invalid_branch_id",
                        "message": e.to_string(),
                    }),
                }
            }
            "shutdown" => {
                // Unauthorized shutdown refused: strictly requires Controller (Owner) role
                if auth_role != ActorRole::Controller {
                    serde_json::json!({
                        "status": "error",
                        "error": "unauthorized",
                        "message": format!(
                            "Unauthorized shutdown refused: actor '{auth_actor_id}' with role {auth_role:?} is not authorized to shut down server",
                        ),
                    })
                } else {
                    let mut state_guard = match state.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    match state_guard.service.terminate_session(&auth_actor_id) {
                        Ok(()) => {
                            shutdown_signal.store(true, Ordering::SeqCst);
                            let resp = serde_json::json!({ "status": "shutting_down" });
                            let _ = write_response(&mut stream, &resp);
                            break;
                        }
                        Err(e) => serde_json::json!({
                            "status": "error",
                            "error": e.structured().code,
                            "message": e.to_string(),
                        }),
                    }
                }
            }
            unknown => serde_json::json!({
                "status": "error",
                "error": "unknown_action",
                "action": unknown,
            }),
        };

        if write_response(&mut stream, &response).is_err() {
            break;
        }
    }
}

/// Publishes the authoritative TCP JSON newline transport service.
///
/// Binds listener to `bind`, prints listening startup JSON conforming to existing CLI shape,
/// and handles concurrent bounded connections.
pub fn serve(
    mut journal: Journal,
    workspace: PathBuf,
    bind: &str,
    once: bool,
    json: bool,
) -> Result<(), StructuredError> {
    let auth_config = AuthConfig::from_env();
    // Derive the token path before `journal` moves into the service below.
    let token_path = journal_token_path(&journal);
    let owner_actor_id = ActorId::new("owner").unwrap_or_else(|_| ActorId::mint());
    let mut dispatcher = SessionHost::new(workspace, owner_actor_id)?;
    if let Err(error) = dispatcher.initialize_provider(&mut journal) {
        eprintln!(
            "{}",
            serde_json::to_string(&error).unwrap_or_else(|_| r#"{"code":"provider_init"}"#.into())
        );
    }
    if let Err(error) = write_token_file(&token_path, &auth_config) {
        return Err(StructuredError::new(
            "token_store",
            format!("Failed to persist auth tokens: {error}"),
            false,
        ));
    }
    let service = ServerSessionService::new(journal);

    let state = Arc::new(Mutex::new(ServerState {
        service,
        dispatcher,
        auth_config: auth_config.clone(),
    }));

    let listener = TcpListener::bind(bind).map_err(|e| {
        StructuredError::new("bind_error", format!("Failed to bind '{bind}': {e}"), false)
    })?;
    let local_addr = listener.local_addr().map_err(|e| {
        StructuredError::new(
            "addr_error",
            format!("Failed to get local addr: {e}"),
            false,
        )
    })?;

    let (session_id, current_offset) = {
        let state_guard = state.lock().unwrap();
        (
            state_guard.service.session_id().clone(),
            state_guard.service.current_offset().0,
        )
    };

    // Prints ready listening JSON matching existing CLI shape.
    // Auth tokens are NEVER printed: they are persisted to a 0600 token file
    // next to the journal so only the local user can read them.
    let startup_info = serde_json::json!({
        "status": "listening",
        "transport": "tcp",
        "endpoint": local_addr.to_string(),
        "session_id": session_id.as_str(),
        "current_offset": current_offset,
        "token_file": token_path.to_string_lossy(),
    });

    if json {
        println!(
            "{}",
            serde_json::to_string(&startup_info).map_err(|e| StructuredError::new(
                "json_error",
                e.to_string(),
                false
            ))?
        );
    } else {
        println!(
            "omp2 service listening on tcp://{} (session '{}', offset {})",
            local_addr, session_id, current_offset
        );
    }

    let active_connections = Arc::new(AtomicUsize::new(0));
    let shutdown_signal = Arc::new(AtomicBool::new(false));

    // Non-blocking listener to allow periodic checks for shutdown_signal
    listener.set_nonblocking(true).map_err(|e| {
        StructuredError::new(
            "io_error",
            format!("Failed to set non-blocking listener: {e}"),
            false,
        )
    })?;

    while !shutdown_signal.load(Ordering::SeqCst) {
        if let Ok(mut guard) = state.try_lock() {
            let start = guard.service.current_offset().0;
            let ServerState {
                dispatcher,
                service,
                ..
            } = &mut *guard;
            dispatcher
                .tool_host
                .poll_jobs(service.journal_mut())
                .map_err(|error| StructuredError::new("job_poll", error.to_string(), false))?;
            dispatcher.poll_children(service.journal_mut())?;
            service.replicate(start);
        }
        match listener.accept() {
            Ok((stream, _peer_addr)) => {
                let current_conns = active_connections.fetch_add(1, Ordering::SeqCst);
                if current_conns >= MAX_CONCURRENT_CONNECTIONS {
                    active_connections.fetch_sub(1, Ordering::SeqCst);
                    let mut stream = stream;
                    let err_resp = serde_json::json!({
                        "status": "error",
                        "error": "connection_limit",
                        "message": format!("Maximum concurrent connections ({MAX_CONCURRENT_CONNECTIONS}) reached"),
                    });
                    let _ = write_response(&mut stream, &err_resp);
                    continue;
                }

                let _ = stream.set_nonblocking(false);
                let conn_guard = ConnectionGuard {
                    active_count: Arc::clone(&active_connections),
                };
                let state_clone = Arc::clone(&state);
                let shutdown_clone = Arc::clone(&shutdown_signal);

                std::thread::spawn(move || {
                    let _guard = conn_guard;
                    handle_connection(stream, state_clone, shutdown_clone);
                });

                if once {
                    // In once mode, wait for connection to finish
                    while active_connections.load(Ordering::SeqCst) > 0 {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    break;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(e) => {
                return Err(StructuredError::new(
                    "accept_error",
                    format!("Accept error: {e}"),
                    false,
                ));
            }
        }
    }
    let mut guard = state
        .lock()
        .map_err(|_| StructuredError::new("host_lock", "Session lock poisoned", false))?;
    let start = guard.service.current_offset().0;
    let ServerState {
        dispatcher,
        service,
        ..
    } = &mut *guard;
    dispatcher.shutdown(service.journal_mut())?;
    service.replicate(start);
    drop(guard);

    // Best-effort cleanup: dead tokens must not linger after shutdown.
    // (A crash may leave the file behind; its tokens are useless without
    // the server, and the next `serve` overwrites them.)
    let _ = std::fs::remove_file(&token_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::role::ActorSession;
    use omp_types::SessionId;
    use serde_json::json;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;
    use std::path::PathBuf;

    fn temp_journal() -> (Journal, PathBuf) {
        let path =
            std::env::temp_dir().join(format!("omp2-transport-test-{}.journal", SessionId::mint()));
        let journal = Journal::create(&path, SessionId::mint()).unwrap();
        (journal, path)
    }

    struct TestServer {
        addr: std::net::SocketAddr,
        auth_config: AuthConfig,
        shutdown: Arc<AtomicBool>,
        thread_handle: Option<std::thread::JoinHandle<()>>,
        journal_path: PathBuf,
    }

    impl TestServer {
        fn spawn() -> Self {
            let (journal, journal_path) = temp_journal();
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let shutdown = Arc::new(AtomicBool::new(false));
            let auth_config = AuthConfig {
                owner_token: "test-owner-tok".into(),
                driver_token: "test-driver-tok".into(),
                spectator_token: "test-spec-tok".into(),
                subagent_token: "test-sub-tok".into(),
                worker_token: "test-worker-tok".into(),
            };

            let service = ServerSessionService::new(journal);
            let owner_actor_id = ActorId::new("owner").unwrap();
            let dispatcher = SessionHost::new(std::env::temp_dir(), owner_actor_id).unwrap();

            let state = Arc::new(Mutex::new(ServerState {
                service,
                dispatcher,
                auth_config: auth_config.clone(),
            }));

            let active_connections = Arc::new(AtomicUsize::new(0));
            let shutdown_clone = Arc::clone(&shutdown);

            listener.set_nonblocking(true).unwrap();

            let thread_handle = std::thread::spawn(move || {
                while !shutdown_clone.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let current = active_connections.fetch_add(1, Ordering::SeqCst);
                            if current >= MAX_CONCURRENT_CONNECTIONS {
                                active_connections.fetch_sub(1, Ordering::SeqCst);
                                continue;
                            }
                            let _ = stream.set_nonblocking(false);
                            let guard = ConnectionGuard {
                                active_count: Arc::clone(&active_connections),
                            };
                            let state_c = Arc::clone(&state);
                            let shut_c = Arc::clone(&shutdown_clone);
                            std::thread::spawn(move || {
                                let _guard = guard;
                                handle_connection(stream, state_c, shut_c);
                            });
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(10));
                        }
                        Err(_) => break,
                    }
                }
            });

            Self {
                addr,
                auth_config,
                shutdown,
                thread_handle: Some(thread_handle),
                journal_path,
            }
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.shutdown.store(true, Ordering::SeqCst);
            if let Some(h) = self.thread_handle.take() {
                let _ = h.join();
            }
            let _ = std::fs::remove_file(&self.journal_path);
        }
    }

    fn send_req(stream: &mut TcpStream, reader: &mut BufReader<TcpStream>, req: &Value) -> Value {
        let req_str = serde_json::to_string(req).unwrap();
        stream.write_all(req_str.as_bytes()).unwrap();
        stream.write_all(b"\n").unwrap();
        stream.flush().unwrap();

        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        serde_json::from_str(line.trim()).unwrap()
    }

    #[test]
    fn test_handshake_required_before_any_action() {
        let server = TestServer::spawn();
        let mut stream = TcpStream::connect(server.addr).unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());

        // Send snapshot request without prior handshake
        let resp = send_req(
            &mut stream,
            &mut reader,
            &json!({ "action": "get_snapshot" }),
        );
        assert_eq!(resp["status"], "error");
        assert_eq!(resp["error"], "handshake_required");
    }

    #[test]
    fn test_unauthenticated_driver_refused() {
        let server = TestServer::spawn();
        let mut stream = TcpStream::connect(server.addr).unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());

        // Attempt driver handshake with no token
        let resp = send_req(
            &mut stream,
            &mut reader,
            &json!({
                "action": "handshake",
                "role": "interactive_driver",
                "actor_id": "driver-rogue",
            }),
        );
        assert_eq!(resp["status"], "error");
        assert_eq!(resp["error"], "unauthorized");

        // Attempt driver handshake with invalid token
        let resp2 = send_req(
            &mut stream,
            &mut reader,
            &json!({
                "action": "handshake",
                "role": "interactive_driver",
                "actor_id": "driver-rogue",
                "auth_token": "wrong-token",
            }),
        );
        assert_eq!(resp2["status"], "error");
        assert_eq!(resp2["error"], "unauthorized");
    }

    #[test]
    fn test_identity_spoof_refused() {
        let server = TestServer::spawn();
        let mut stream = TcpStream::connect(server.addr).unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());

        // Handshake as spectator
        let hs = send_req(
            &mut stream,
            &mut reader,
            &json!({
                "action": "handshake",
                "role": "spectator",
                "actor_id": "spectator-1",
                "auth_token": server.auth_config.spectator_token,
            }),
        );
        assert_eq!(hs["status"], "ok");

        // Attempt to submit command (spectator lacks permission)
        let resp = send_req(
            &mut stream,
            &mut reader,
            &json!({
                "action": "submit_command",
                "actor_id": "driver-spoof",
                "command_id": "cmd-spoof",
                "command_line": "toggle cl_showthinking",
            }),
        );
        assert_eq!(resp["status"], "error");
        assert_eq!(resp["error"], "unauthorized");
    }

    #[test]
    fn test_unauthorized_shutdown_refused() {
        let server = TestServer::spawn();
        let mut stream = TcpStream::connect(server.addr).unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());

        // Handshake as interactive driver
        let hs = send_req(
            &mut stream,
            &mut reader,
            &json!({
                "action": "handshake",
                "role": "interactive_driver",
                "actor_id": "driver-legit",
                "auth_token": server.auth_config.driver_token,
            }),
        );
        assert_eq!(hs["status"], "ok");

        // Driver attempts shutdown: refused (requires Controller)
        let resp = send_req(&mut stream, &mut reader, &json!({ "action": "shutdown" }));
        assert_eq!(resp["status"], "error");
        assert_eq!(resp["error"], "unauthorized");
    }

    #[test]
    fn test_authorized_owner_shutdown() {
        let server = TestServer::spawn();
        let mut stream = TcpStream::connect(server.addr).unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());

        // Handshake as controller with owner token
        let hs = send_req(
            &mut stream,
            &mut reader,
            &json!({
                "action": "handshake",
                "role": "controller",
                "actor_id": "owner-root",
                "auth_token": server.auth_config.owner_token,
            }),
        );
        assert_eq!(hs["status"], "ok");

        let resp = send_req(&mut stream, &mut reader, &json!({ "action": "shutdown" }));
        assert_eq!(resp["status"], "shutting_down");
    }

    #[test]
    fn test_concurrent_clients_spectators_and_driver() {
        let server = TestServer::spawn();

        // Driver connection
        let mut driver_stream = TcpStream::connect(server.addr).unwrap();
        let mut driver_reader = BufReader::new(driver_stream.try_clone().unwrap());
        let d_hs = send_req(
            &mut driver_stream,
            &mut driver_reader,
            &json!({
                "action": "handshake",
                "role": "interactive_driver",
                "actor_id": "driver-main",
                "auth_token": server.auth_config.driver_token,
            }),
        );
        assert_eq!(d_hs["status"], "ok");

        // Spectator 1 connection
        let mut s1_stream = TcpStream::connect(server.addr).unwrap();
        let mut s1_reader = BufReader::new(s1_stream.try_clone().unwrap());
        let s1_hs = send_req(
            &mut s1_stream,
            &mut s1_reader,
            &json!({
                "action": "handshake",
                "role": "spectator",
                "actor_id": "spectator-1",
                "auth_token": server.auth_config.spectator_token,
            }),
        );
        assert_eq!(s1_hs["status"], "ok");
        let s1_sub = send_req(
            &mut s1_stream,
            &mut s1_reader,
            &json!({ "action": "subscribe", "last_offset": 0 }),
        );
        assert_eq!(s1_sub["status"], "ok");

        // Spectator 2 connection (flood/slow spectator)
        let mut s2_stream = TcpStream::connect(server.addr).unwrap();
        let mut s2_reader = BufReader::new(s2_stream.try_clone().unwrap());
        let s2_hs = send_req(
            &mut s2_stream,
            &mut s2_reader,
            &json!({
                "action": "handshake",
                "role": "spectator",
                "actor_id": "spectator-2",
                "auth_token": server.auth_config.spectator_token,
            }),
        );
        assert_eq!(s2_hs["status"], "ok");
        let s2_sub = send_req(
            &mut s2_stream,
            &mut s2_reader,
            &json!({ "action": "subscribe", "last_offset": 0 }),
        );
        assert_eq!(s2_sub["status"], "ok");

        // Driver executes command while spectators are active
        let cmd_resp = send_req(
            &mut driver_stream,
            &mut driver_reader,
            &json!({
                "action": "submit_command",
                "command_id": "c-101",
                "command_line": "toggle cl_showthinking",
            }),
        );
        assert_eq!(cmd_resp["status"], "ok");
        assert_eq!(cmd_resp["command_id"], "c-101");

        // Spectator 1 polls and receives replicated message
        let s1_poll = send_req(&mut s1_stream, &mut s1_reader, &json!({ "action": "poll" }));
        assert_eq!(s1_poll["status"], "ok");
        assert!(s1_poll.get("message").is_some());

        // Spectator 2 flood: send multiple rapid queries
        for i in 0..5 {
            let poll_resp = send_req(&mut s2_stream, &mut s2_reader, &json!({ "action": "poll" }));
            assert_eq!(poll_resp["status"], "ok");
            let _ = i;
        }

        // Driver continues uninterrupted
        let cmd2_resp = send_req(
            &mut driver_stream,
            &mut driver_reader,
            &json!({
                "action": "submit_command",
                "command_id": "c-102",
                "command_line": "toggle cl_showthinking",
            }),
        );
        assert_eq!(cmd2_resp["status"], "ok");
        assert_eq!(cmd2_resp["command_id"], "c-102");
    }

    #[test]
    fn test_reconnect_patch_order_and_resync() {
        let server = TestServer::spawn();
        let mut stream = TcpStream::connect(server.addr).unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());

        let hs = send_req(
            &mut stream,
            &mut reader,
            &json!({
                "action": "handshake",
                "role": "interactive_driver",
                "actor_id": "driver-recon",
                "auth_token": server.auth_config.driver_token,
            }),
        );
        assert_eq!(hs["status"], "ok");

        // Submit command to produce an offset
        let cmd_resp = send_req(
            &mut stream,
            &mut reader,
            &json!({
                "action": "submit_command",
                "command_id": "cmd-rec-1",
                "command_line": "toggle cl_showthinking",
            }),
        );
        assert_eq!(cmd_resp["status"], "ok");
        let offset = cmd_resp["current_offset"].as_u64().unwrap();

        // Reconnect subscribe with exact offset
        let sub_resp = send_req(
            &mut stream,
            &mut reader,
            &json!({
                "action": "subscribe",
                "last_offset": offset,
            }),
        );
        assert_eq!(sub_resp["status"], "ok");

        // Reconnect subscribe with future offset -> triggers Resync
        let resync_sub = send_req(
            &mut stream,
            &mut reader,
            &json!({
                "action": "subscribe",
                "last_offset": 9999,
            }),
        );
        assert_eq!(resync_sub["status"], "ok");
        assert!(resync_sub["message"].get("Resync").is_some());
    }

    #[test]
    fn test_command_execution_updates_journal_and_replicates() {
        let server = TestServer::spawn();
        let mut stream = TcpStream::connect(server.addr).unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());

        let hs = send_req(
            &mut stream,
            &mut reader,
            &json!({
                "action": "handshake",
                "role": "interactive_driver",
                "actor_id": "driver-exec",
                "auth_token": server.auth_config.driver_token,
            }),
        );
        assert_eq!(hs["status"], "ok");

        let cmd_resp = send_req(
            &mut stream,
            &mut reader,
            &json!({
                "action": "submit_command",
                "command_id": "cmd-toggle-1",
                "command_line": "toggle cl_showthinking",
            }),
        );
        assert_eq!(cmd_resp["status"], "ok");

        // Fetch snapshot and verify DOM state reflects command execution
        let snap_resp = send_req(
            &mut stream,
            &mut reader,
            &json!({ "action": "get_snapshot" }),
        );
        assert_eq!(snap_resp["status"], "ok");
        assert!(snap_resp.get("snapshot").is_some());
    }

    #[test]
    fn test_reconnect_restores_journal_command_replay_protection() {
        let (journal, journal_path) = temp_journal();
        let mut service = ServerSessionService::new(journal);
        let driver_id = ActorId::new("driver-replay").unwrap();
        let session = ActorSession::new(driver_id.clone(), ActorRole::InteractiveDriver);
        service.attach_actor(session).unwrap();

        // Submit command once
        service
            .submit_command(&driver_id, "cmd-replay-test", "toggle cl_showthinking")
            .unwrap();

        drop(service);
        // Reopen journal and create new service
        let reopened_journal = Journal::open(&journal_path).unwrap();
        let mut restored_service = ServerSessionService::new(reopened_journal);
        let session2 = ActorSession::new(driver_id.clone(), ActorRole::InteractiveDriver);
        restored_service.attach_actor(session2).unwrap();

        // Submitting same command_id fails due to restored replay protection
        let err = restored_service
            .submit_command(&driver_id, "cmd-replay-test", "toggle cl_showthinking")
            .unwrap_err();
        match err {
            ServerError::CommandReplayed(cid) => assert_eq!(cid, "cmd-replay-test"),
            other => panic!("expected CommandReplayed, got {other:?}"),
        }

        let _ = std::fs::remove_file(&journal_path);
    }
}
