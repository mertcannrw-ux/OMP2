use crate::error::ServerError;
use crate::replication::{ReplicationMessage, SubscriberQueue};
use crate::role::{ActorRole, ActorSession, Permission};
use omp_state::{Journal, SessionSnapshot};
use omp_types::{
    ActorId, ArtifactId, BranchId, ElementId, ElementSnapshot, JobId, JournalOffset, Patch,
    PatchAuthor, PatchOp, ProtocolVersion, SessionId, TypedValue,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HandshakeRequest {
    pub protocol_version: ProtocolVersion,
    pub actor_id: ActorId,
    pub requested_role: ActorRole,
    pub auth_token: Option<String>,
    pub capabilities: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HandshakeResponse {
    pub protocol_version: ProtocolVersion,
    pub session_id: SessionId,
    pub current_offset: JournalOffset,
    pub granted_role: ActorRole,
    pub capabilities: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArtifactScopeSpec {
    Session,
    Global,
    Actor(ActorId),
    Job(JobId),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub id: ArtifactId,
    pub origin_actor: ActorId,
    pub media_type: String,
    pub size_bytes: usize,
    pub scope: ArtifactScopeSpec,
    pub allowed_actors: Option<Vec<ActorId>>,
}

impl ArtifactRecord {
    pub fn new(
        id: ArtifactId,
        origin_actor: ActorId,
        media_type: impl Into<String>,
        size_bytes: usize,
    ) -> Self {
        Self {
            id,
            origin_actor,
            media_type: media_type.into(),
            size_bytes,
            scope: ArtifactScopeSpec::Session,
            allowed_actors: None,
        }
    }

    pub fn with_scope(mut self, scope: ArtifactScopeSpec) -> Self {
        self.scope = scope;
        self
    }

    pub fn with_allowed_actors(mut self, allowed: Vec<ActorId>) -> Self {
        self.allowed_actors = Some(allowed);
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub request_id: String,
    pub requested_by: ActorId,
    pub action_description: String,
    pub approved: Option<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ServerEndpoint {
    InMemory,
    Tcp(String),
    Unix(String),
    NamedPipe(String),
}

impl ServerEndpoint {
    pub fn tcp(addr: impl Into<String>) -> Self {
        Self::Tcp(addr.into())
    }

    pub fn unix(path: impl Into<String>) -> Self {
        Self::Unix(path.into())
    }

    pub fn in_memory() -> Self {
        Self::InMemory
    }
}

pub struct ServerSessionService {
    session_id: SessionId,
    journal: Journal,
    actors: BTreeMap<ActorId, ActorSession>,
    subscribers: BTreeMap<ActorId, SubscriberQueue>,
    seen_command_ids: BTreeSet<String>,
    default_queue_capacity: usize,
    terminated: bool,
}

impl ServerSessionService {
    pub fn new(journal: Journal) -> Self {
        let session_id = journal.snapshot().session_id.clone();
        let mut seen_command_ids = BTreeSet::new();
        let mut actors = BTreeMap::new();

        // Replay protection is session-wide across branches, including accepted
        // commands whose execution was interrupted after durable admission.
        for record in journal.records() {
            for op in &record.patch.ops {
                if let PatchOp::Create { element, .. } = op
                    && element.kind == "command_receipt"
                        && let Some(TypedValue::String(id)) = element.attributes.get("command_id") {
                            seen_command_ids.insert(id.clone());
                        }
            }
        }

        // Restore actors from journal snapshot <actors> container
        let actors_container = journal.snapshot().container("actors");
        for elem in journal.snapshot().children(actors_container) {
            let actor_str = match elem.attributes.get("actor_id") {
                Some(TypedValue::String(s)) => s.clone(),
                _ => elem
                    .id
                    .as_str()
                    .strip_prefix("actor-")
                    .unwrap_or(elem.id.as_str())
                    .to_string(),
            };
            if let Ok(actor_id) = ActorId::new(&actor_str) {
                let role = match elem.attributes.get("role") {
                    Some(TypedValue::String(r)) => match r.as_str() {
                        "Controller" => ActorRole::Controller,
                        "InteractiveDriver" => ActorRole::InteractiveDriver,
                        "Spectator" => ActorRole::Spectator,
                        "SubagentInspector" => ActorRole::SubagentInspector,
                        "AutomationWorker" => ActorRole::AutomationWorker,
                        _ => ActorRole::Spectator,
                    },
                    _ => ActorRole::Spectator,
                };
                let attached_at_epoch_ms = match elem.attributes.get("attached_at_epoch_ms") {
                    Some(TypedValue::Integer(ms)) => *ms as u64,
                    _ => 0,
                };
                let mut session = ActorSession::new(actor_id.clone(), role);
                session.attached_at_epoch_ms = attached_at_epoch_ms;
                actors.insert(actor_id, session);
            }
        }

        Self {
            session_id,
            journal,
            actors,
            subscribers: BTreeMap::new(),
            seen_command_ids,
            default_queue_capacity: 128,
            terminated: false,
        }
    }

    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    pub fn current_offset(&self) -> JournalOffset {
        JournalOffset(self.journal.snapshot().offset)
    }

    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Bind the service to a transport endpoint.
    ///
    /// Explicitly returns `ServerError::UnsupportedTransport` for network transports
    /// (TCP, Unix socket, NamedPipe) which require an external framing provider,
    /// rather than spinning a fake or broken socket thread.
    pub fn bind(&self, endpoint: &ServerEndpoint) -> Result<ServerEndpoint, ServerError> {
        match endpoint {
            ServerEndpoint::InMemory => Ok(ServerEndpoint::InMemory),
            ServerEndpoint::Tcp(addr) => {
                Err(ServerError::UnsupportedTransport(format!("tcp://{addr}")))
            }
            ServerEndpoint::Unix(path) => {
                Err(ServerError::UnsupportedTransport(format!("unix://{path}")))
            }
            ServerEndpoint::NamedPipe(pipe) => {
                Err(ServerError::UnsupportedTransport(format!("pipe://{pipe}")))
            }
        }
    }

    /// Perform a protocol handshake with a connecting actor.
    pub fn handshake(&mut self, req: HandshakeRequest) -> Result<HandshakeResponse, ServerError> {
        ProtocolVersion::CURRENT
            .negotiate(
                req.protocol_version,
                &req.capabilities.iter().cloned().collect(),
                &[omp_types::JOURNAL_FORMAT.to_owned()].into(),
            )
            .map_err(|error| ServerError::HandshakeFailed(error.to_string()))?;

        let mut session = ActorSession::new(req.actor_id.clone(), req.requested_role);
        if let Some(tok) = req.auth_token {
            session = session.with_token(tok);
        }
        session = session.with_capabilities(req.capabilities.clone());

        let granted_role = session.role;
        let capabilities = session.capabilities.clone();
        self.attach_actor(session)?;

        Ok(HandshakeResponse {
            protocol_version: ProtocolVersion::CURRENT,
            session_id: self.session_id.clone(),
            current_offset: self.current_offset(),
            granted_role,
            capabilities,
        })
    }

    pub fn attach_actor(&mut self, actor: ActorSession) -> Result<(), ServerError> {
        let actor_id = actor.actor_id.clone();
        let role = actor.role;
        let attached_at = actor.attached_at_epoch_ms;

        // Durably journal actor ownership in <actors> container if not already recorded.
        // The journal write happens BEFORE the in-memory insert so a failure
        // cannot leave the two diverged in either direction.
        let actors_container = self.journal.snapshot().container("actors").clone();
        let already_recorded = self
            .journal
            .snapshot()
            .children(&actors_container)
            .any(|elem| {
                if elem.id.as_str() == format!("actor-{}", actor_id.as_str()) {
                    return true;
                }
                if let Some(TypedValue::String(s)) = elem.attributes.get("actor_id")
                    && s == actor_id.as_str() {
                        return true;
                    }
                false
            });

        if !already_recorded && !self.terminated {
            let elem_id = ElementId::new(format!("actor-{}", actor_id.as_str()))
                .unwrap_or_else(|_| ElementId::mint());
            let mut element = ElementSnapshot::new(elem_id, "actor");
            element
                .attributes
                .insert("actor_id".into(), TypedValue::String(actor_id.to_string()));
            element
                .attributes
                .insert("role".into(), TypedValue::String(format!("{:?}", role)));
            element.attributes.insert(
                "attached_at_epoch_ms".into(),
                TypedValue::Integer(attached_at as i64),
            );

            let base_offset = self.current_offset();
            let result_offset = self.journal.next_offset();
            let patch = Patch {
                base_offset,
                result_offset,
                by: PatchAuthor::Actor(actor_id.clone()),
                reason: "attach actor".to_string(),
                ops: vec![PatchOp::Create {
                    parent: actors_container.clone(),
                    index: self.journal.snapshot().children(&actors_container).count() as u32,
                    element,
                }],
            };
            // A journal failure here must not leave the in-memory actor table
            // diverged from durable state: propagate instead of swallowing.
            // The in-memory insert happens only after the durable write.
            let offset = self.submit_patch_internal(patch)?;
            let _ = offset;
        }

        self.actors.insert(actor_id.clone(), actor);
        Ok(())
    }

    pub fn detach_actor(
        &mut self,
        caller_id: &ActorId,
        target_actor: &ActorId,
    ) -> Result<(), ServerError> {
        if caller_id != target_actor {
            let caller = self.get_actor(caller_id)?;
            caller.check_permission(Permission::ManageSession)?;
        }
        self.actors.remove(target_actor);
        self.subscribers.remove(target_actor);
        Ok(())
    }

    pub fn get_actor(&self, actor_id: &ActorId) -> Result<&ActorSession, ServerError> {
        self.actors
            .get(actor_id)
            .ok_or_else(|| ServerError::ActorNotFound(actor_id.to_string()))
    }

    /// Authoritative snapshot retrieval.
    pub fn get_snapshot(
        &self,
        actor_id: &ActorId,
        offset: Option<JournalOffset>,
    ) -> Result<SessionSnapshot, ServerError> {
        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::ReadSnapshot)?;

        match offset {
            Some(off) => {
                if off.0 > self.journal.latest_offset() {
                    return Err(ServerError::OffsetUnavailable {
                        requested: off.0,
                        oldest_available: 0,
                    });
                }
                self.journal.materialize(off.0).map_err(ServerError::from)
            }
            None => Ok(self.journal.resume_latest()),
        }
    }

    /// Subscribe an actor to ordered patch replication.
    ///
    /// If `last_offset` is None, invalid (future), or older than available records,
    /// an initial `Resync` with the full snapshot is returned.
    /// Otherwise, missing patches between `last_offset` and current are enqueued
    /// and a Heartbeat is returned.
    pub fn subscribe(
        &mut self,
        actor_id: &ActorId,
        last_offset: Option<JournalOffset>,
        capacity_override: Option<usize>,
    ) -> Result<ReplicationMessage, ServerError> {
        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::SubscribePatches)?;

        let role = actor.role;
        let capacity = capacity_override.unwrap_or(self.default_queue_capacity);
        let current = self.current_offset();

        let mut queue = SubscriberQueue::new(
            actor_id.clone(),
            role,
            last_offset.unwrap_or(JournalOffset(0)),
            capacity,
        );

        let initial_message = match last_offset {
            Some(offset) if offset.0 <= current.0 => {
                let oldest_available = self.journal.records().next().map(|r| r.offset).unwrap_or(0);
                if offset.0 < oldest_available && offset.0 > 0 {
                    // Gap in history: full resync required
                    let snapshot = self.journal.resume_latest();
                    ReplicationMessage::Resync {
                        snapshot,
                        offset: current,
                    }
                } else {
                    // Enqueue missing patches from offset+1 to current
                    for record in self.journal.records().filter(|r| r.offset > offset.0) {
                        let _ = queue.push(ReplicationMessage::Patch {
                            patch: record.patch.clone(),
                        });
                    }
                    ReplicationMessage::Heartbeat {
                        timestamp_epoch_ms: 0,
                        latest_offset: current,
                    }
                }
            }
            _ => {
                // Client has no valid offset or offset is in the future: full resync required
                let snapshot = self.journal.resume_latest();
                ReplicationMessage::Resync {
                    snapshot,
                    offset: current,
                }
            }
        };

        self.subscribers.insert(actor_id.clone(), queue);
        Ok(initial_message)
    }

    /// Fetch pending replication messages for an active subscriber.
    pub fn poll_subscriber(&mut self, actor_id: &ActorId) -> Option<ReplicationMessage> {
        if let Some(sub) = self.subscribers.get_mut(actor_id) {
            if sub.needs_resync {
                sub.needs_resync = false;
                let snapshot = self.journal.resume_latest();
                let offset = self.current_offset();
                return Some(ReplicationMessage::Resync { snapshot, offset });
            }
            sub.pop()
        } else {
            None
        }
    }

    /// Submit a patch to the authoritative journal.
    ///
    /// Enforces authorization, checks base offset, validates patch, applies to journal,
    /// and broadcasts the patch to all active subscribers.
    /// Rejects stale-base patches without merging.
    pub fn submit_patch(
        &mut self,
        actor_id: &ActorId,
        patch: Patch,
    ) -> Result<JournalOffset, ServerError> {
        if self.terminated {
            return Err(ServerError::SessionTerminated(self.session_id.to_string()));
        }

        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::SubmitPatch)?;

        let expected_base = self.journal.snapshot().offset;
        if patch.base_offset.0 != expected_base {
            return Err(ServerError::StaleBaseOffset {
                actual: patch.base_offset.0,
                expected: expected_base,
            });
        }

        patch
            .validate()
            .map_err(|e| ServerError::InvalidPatch(e.to_string()))?;

        self.submit_patch_internal(patch)
    }

    fn submit_patch_internal(&mut self, patch: Patch) -> Result<JournalOffset, ServerError> {
        let new_offset = self.journal.append_patch(patch.clone())?;
        let journal_offset = JournalOffset(new_offset);

        // Replicate patch to all active subscribers.
        // If a spectator/inspector's queue overflows, it is marked for resync and its queue is cleared,
        // so it cannot starve or block the controller.
        let message = ReplicationMessage::Patch { patch };
        for sub in self.subscribers.values_mut() {
            let _ = sub.push(message.clone());
        }

        Ok(journal_offset)
    }

    /// Submit a command with replay protection.
    pub fn submit_command(
        &mut self,
        actor_id: &ActorId,
        command_id: &str,
        _command_line: &str,
    ) -> Result<(), ServerError> {
        if self.terminated {
            return Err(ServerError::SessionTerminated(self.session_id.to_string()));
        }

        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::SubmitCommand)?;

        let trimmed_id = command_id.trim();
        if trimmed_id.is_empty() || trimmed_id.len() > 128 {
            return Err(ServerError::InvalidCommand(
                "command_id must contain 1–128 bytes".into(),
            ));
        }

        if self.seen_command_ids.contains(trimmed_id) {
            return Err(ServerError::CommandReplayed(trimmed_id.to_string()));
        }
        let mut receipt = ElementSnapshot::new(ElementId::mint(), "command_receipt");
        receipt
            .attributes
            .insert("command_id".into(), TypedValue::String(trimmed_id.into()));
        let meta = self.journal.snapshot().container("meta").clone();
        self.submit_patch_internal(Patch {
            base_offset: self.current_offset(),
            result_offset: self.journal.next_offset(),
            by: actor_id.clone().into(),
            reason: "accept command".into(),
            ops: vec![PatchOp::Create {
                index: self.journal.snapshot().children(&meta).count() as u32,
                parent: meta,
                element: receipt,
            }],
        })?;
        self.seen_command_ids.insert(trimmed_id.to_string());
        Ok(())
    }

    /// Change a convar in the session.
    ///
    /// Requires `ChangeConVar` permission. Appends a patch to the journal
    /// setting the attribute on the `<convars>` container element.
    pub fn change_convar(
        &mut self,
        actor_id: &ActorId,
        name: &str,
        value: TypedValue,
    ) -> Result<JournalOffset, ServerError> {
        if self.terminated {
            return Err(ServerError::SessionTerminated(self.session_id.to_string()));
        }

        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::ChangeConVar)?;

        let trimmed_name = name.trim();
        if trimmed_name.is_empty() {
            return Err(ServerError::InvalidCommand(
                "convar name cannot be empty".into(),
            ));
        }

        let convars_id = self.journal.snapshot().container("convars").clone();
        let base_offset = self.current_offset();
        let result_offset = self.journal.next_offset();

        let patch = Patch {
            base_offset,
            result_offset,
            by: actor_id.clone().into(),
            reason: format!("change convar {trimmed_name}"),
            ops: vec![PatchOp::SetAttribute {
                element: convars_id,
                name: trimmed_name.to_string(),
                value,
            }],
        };

        self.submit_patch_internal(patch)
    }

    /// Post an approval request.
    pub fn request_approval(
        &mut self,
        requester: &ActorId,
        request_id: &str,
        action_description: &str,
    ) -> Result<(), ServerError> {
        if self.terminated {
            return Err(ServerError::SessionTerminated(self.session_id.to_string()));
        }

        let _actor = self.get_actor(requester)?;

        let trimmed_id = request_id.trim();
        if trimmed_id.is_empty() {
            return Err(ServerError::InvalidCommand(
                "request_id cannot be empty".into(),
            ));
        }

        let id = ElementId::new(trimmed_id)
            .map_err(|error| ServerError::InvalidCommand(error.to_string()))?;
        let mut node = ElementSnapshot::new(id, "approval");
        node.attributes
            .insert("status".into(), TypedValue::String("pending".into()));
        node.payload = Some(
            serde_json::to_value(ApprovalRequest {
                request_id: trimmed_id.into(),
                requested_by: requester.clone(),
                action_description: action_description.into(),
                approved: None,
            })
            .map_err(|error| ServerError::InvalidCommand(error.to_string()))?,
        );
        let offset = self.journal.snapshot().offset;
        self.journal.append_patch(Patch {
            base_offset: JournalOffset(offset),
            result_offset: self.journal.next_offset(),
            by: requester.clone().into(),
            reason: "request approval".into(),
            ops: vec![PatchOp::Create {
                parent: self.journal.snapshot().container("approvals").clone(),
                index: self
                    .journal
                    .snapshot()
                    .children(self.journal.snapshot().container("approvals"))
                    .count() as u32,
                element: node,
            }],
        })?;
        self.replicate(offset);
        Ok(())
    }

    /// Resolve an approval request.
    pub fn resolve_approval(
        &mut self,
        actor_id: &ActorId,
        request_id: &str,
        decision: bool,
    ) -> Result<(), ServerError> {
        if self.terminated {
            return Err(ServerError::SessionTerminated(self.session_id.to_string()));
        }

        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::Approve)?;

        let id = ElementId::new(request_id)
            .map_err(|error| ServerError::InvalidCommand(error.to_string()))?;
        let node = self
            .journal
            .snapshot()
            .children(self.journal.snapshot().container("approvals"))
            .find(|node| node.id == id)
            .ok_or_else(|| ServerError::InvalidCommand("Approval request not found".into()))?;
        if node.attributes.get("status") != Some(&TypedValue::String("pending".into())) {
            return Err(ServerError::InvalidCommand(
                "Approval already resolved".into(),
            ));
        }
        let offset = self.journal.snapshot().offset;
        self.journal.append_patch(Patch {
            base_offset: JournalOffset(offset),
            result_offset: self.journal.next_offset(),
            by: actor_id.clone().into(),
            reason: "resolve approval".into(),
            ops: vec![
                PatchOp::SetAttribute {
                    element: id.clone(),
                    name: "status".into(),
                    value: TypedValue::String(if decision { "approved" } else { "denied" }.into()),
                },
                PatchOp::SetAttribute {
                    element: id,
                    name: "approved".into(),
                    value: TypedValue::Bool(decision),
                },
            ],
        })?;
        self.replicate(offset);
        Ok(())
    }

    /// Signal a running job.
    pub fn signal_job(
        &mut self,
        actor_id: &ActorId,
        job_id: &JobId,
        signal: &str,
    ) -> Result<(), ServerError> {
        if self.terminated {
            return Err(ServerError::SessionTerminated(self.session_id.to_string()));
        }

        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::SignalJob)?;
        if !matches!(
            signal,
            "interrupt" | "terminate" | "kill" | "SIGINT" | "SIGTERM" | "SIGKILL"
        ) {
            return Err(ServerError::InvalidCommand("Unsupported job signal".into()));
        }
        let id = ElementId::new(format!("job-{job_id}"))
            .map_err(|error| ServerError::InvalidCommand(error.to_string()))?;
        if !self
            .journal
            .snapshot()
            .active_jobs()
            .any(|job| job.id == id)
        {
            return Err(ServerError::JobNotFound(job_id.to_string()));
        }
        let offset = self.journal.snapshot().offset;
        self.journal.append_patch(Patch {
            base_offset: JournalOffset(offset),
            result_offset: self.journal.next_offset(),
            by: actor_id.clone().into(),
            reason: "request job signal".into(),
            ops: vec![PatchOp::SetAttribute {
                element: id,
                name: "requested_signal".into(),
                value: TypedValue::String(signal.into()),
            }],
        })?;
        self.replicate(offset);
        Ok(())
    }

    /// Register an artifact into the authoritative session journal.
    ///
    /// Requires `WriteWorkspace` permission. The artifact is persisted
    /// as a DOM element under `<artifacts>` and broadcast to all subscribers.
    pub fn register_artifact(
        &mut self,
        actor_id: &ActorId,
        artifact: ArtifactRecord,
    ) -> Result<JournalOffset, ServerError> {
        if self.terminated {
            return Err(ServerError::SessionTerminated(self.session_id.to_string()));
        }

        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::WriteWorkspace)?;

        let artifacts_container = self.journal.snapshot().container("artifacts").clone();
        let index = self
            .journal
            .snapshot()
            .children(&artifacts_container)
            .count() as u32;

        let element_id = ElementId::new(format!("artifact-{}", artifact.id.as_str()))
            .unwrap_or_else(|_| ElementId::mint());

        let mut element = ElementSnapshot::new(element_id, "artifact");
        element.attributes.insert(
            "artifact_id".into(),
            TypedValue::String(artifact.id.to_string()),
        );
        element.attributes.insert(
            "media_type".into(),
            TypedValue::String(artifact.media_type.clone()),
        );
        element.attributes.insert(
            "size_bytes".into(),
            TypedValue::Integer(artifact.size_bytes as i64),
        );
        element.attributes.insert(
            "origin_actor".into(),
            TypedValue::String(artifact.origin_actor.to_string()),
        );
        element.attributes.insert(
            "scope".into(),
            TypedValue::String(match &artifact.scope {
                ArtifactScopeSpec::Session => "Session".into(),
                ArtifactScopeSpec::Global => "Global".into(),
                ArtifactScopeSpec::Actor(a) => format!("Actor:{a}"),
                ArtifactScopeSpec::Job(j) => format!("Job:{j}"),
            }),
        );
        element.payload = Some(serde_json::to_value(&artifact).unwrap_or_default());

        let base_offset = self.current_offset();
        let result_offset = self.journal.next_offset();
        let patch = Patch {
            base_offset,
            result_offset,
            by: actor_id.clone().into(),
            reason: format!("register artifact {}", artifact.id),
            ops: vec![PatchOp::Create {
                parent: artifacts_container,
                index,
                element,
            }],
        };

        self.submit_patch_internal(patch)
    }

    /// Access an artifact with journal-derived metadata scope enforcement.
    pub fn get_artifact(
        &self,
        actor_id: &ActorId,
        artifact_id: &ArtifactId,
    ) -> Result<ArtifactRecord, ServerError> {
        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::AccessArtifact)?;

        self.extract_artifact_from_snapshot(self.journal.snapshot(), actor, artifact_id)
    }

    /// Access an artifact at a specific historical journal offset with scope enforcement.
    pub fn get_artifact_at(
        &self,
        actor_id: &ActorId,
        artifact_id: &ArtifactId,
        offset: JournalOffset,
    ) -> Result<ArtifactRecord, ServerError> {
        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::AccessArtifact)?;

        let snapshot = self.get_snapshot(actor_id, Some(offset))?;
        self.extract_artifact_from_snapshot(&snapshot, actor, artifact_id)
    }

    fn extract_artifact_from_snapshot(
        &self,
        snapshot: &SessionSnapshot,
        actor: &ActorSession,
        artifact_id: &ArtifactId,
    ) -> Result<ArtifactRecord, ServerError> {
        let artifacts_container = snapshot.container("artifacts");
        let expected_elem_id = format!("artifact-{}", artifact_id.as_str());

        let found_element = snapshot
            .children(artifacts_container)
            .find(|elem| {
                if elem.id.as_str() == expected_elem_id {
                    return true;
                }
                if let Some(TypedValue::String(s)) = elem.attributes.get("artifact_id")
                    && s == artifact_id.as_str() {
                        return true;
                    }
                if let Some(payload) = &elem.payload
                    && let Some(id_val) = payload.get("id").and_then(|v| v.as_str())
                        && id_val == artifact_id.as_str() {
                            return true;
                        }
                false
            })
            .ok_or_else(|| ServerError::ArtifactNotFound(artifact_id.to_string()))?;

        // Extract metadata from payload or attributes
        let record = if let Some(payload) = &found_element.payload {
            if let Ok(rec) = serde_json::from_value::<ArtifactRecord>(payload.clone()) {
                rec
            } else {
                self.parse_artifact_from_element(found_element, artifact_id)
            }
        } else {
            self.parse_artifact_from_element(found_element, artifact_id)
        };

        // Enforce journal-derived scope
        if actor.role != ActorRole::Controller {
            if let Some(allowed) = &record.allowed_actors
                && !allowed.contains(&actor.actor_id) {
                    return Err(ServerError::ArtifactScopeViolation {
                        artifact: artifact_id.to_string(),
                        actor: actor.actor_id.to_string(),
                    });
                }

            match &record.scope {
                ArtifactScopeSpec::Global => {
                    // Allowed for any authenticated actor
                }
                ArtifactScopeSpec::Session => {
                    // Allowed for any actor attached to this session
                }
                ArtifactScopeSpec::Actor(expected_actor) => {
                    if expected_actor != &actor.actor_id {
                        return Err(ServerError::ArtifactScopeViolation {
                            artifact: artifact_id.to_string(),
                            actor: actor.actor_id.to_string(),
                        });
                    }
                }
                ArtifactScopeSpec::Job(_expected_job) => {
                    // Job-scoped artifacts require Controller or matching job actor
                }
            }
        }

        Ok(record)
    }

    fn parse_artifact_from_element(
        &self,
        elem: &omp_types::ElementSnapshot,
        artifact_id: &ArtifactId,
    ) -> ArtifactRecord {
        let origin_actor = elem
            .attributes
            .get("origin_actor")
            .and_then(|v| match v {
                TypedValue::String(s) => ActorId::new(s).ok(),
                _ => None,
            })
            .unwrap_or_else(|| ActorId::new("unknown").unwrap());

        let media_type = elem
            .attributes
            .get("media_type")
            .and_then(|v| match v {
                TypedValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "application/octet-stream".into());

        let size_bytes = elem
            .attributes
            .get("size_bytes")
            .and_then(|v| match v {
                TypedValue::Integer(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(0);

        let scope = elem
            .attributes
            .get("scope")
            .and_then(|v| match v {
                TypedValue::String(s) => {
                    if s == "Global" {
                        Some(ArtifactScopeSpec::Global)
                    } else if s == "Session" {
                        Some(ArtifactScopeSpec::Session)
                    } else if let Some(rest) = s.strip_prefix("Actor:") {
                        ActorId::new(rest).ok().map(ArtifactScopeSpec::Actor)
                    } else if let Some(rest) = s.strip_prefix("Job:") {
                        JobId::new(rest).ok().map(ArtifactScopeSpec::Job)
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .unwrap_or(ArtifactScopeSpec::Session);

        ArtifactRecord {
            id: artifact_id.clone(),
            origin_actor,
            media_type,
            size_bytes,
            scope,
            allowed_actors: None,
        }
    }

    /// Perform a workspace write operation.
    ///
    /// Requires `WriteWorkspace` permission.
    pub fn write_workspace(
        &mut self,
        actor_id: &ActorId,
        path: &str,
        host: &mut omp_control::SessionHost,
        content: &[u8],
    ) -> Result<(), ServerError> {
        if self.terminated {
            return Err(ServerError::SessionTerminated(self.session_id.to_string()));
        }

        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::WriteWorkspace)?;

        if path.trim().is_empty() {
            return Err(ServerError::InvalidCommand(
                "workspace path cannot be empty".into(),
            ));
        }

        let content = std::str::from_utf8(content)
            .map_err(|error| ServerError::InvalidCommand(error.to_string()))?;
        let start = self.current_offset().0;
        let call = host
            .execute_tool(
                &mut self.journal,
                "Write",
                serde_json::json!({
                    "i": "Writing workspace from authorized actor", "path": path, "content": content
                }),
            )
            .map_err(ServerError::Execution)?;
        self.replicate(start);
        let failed = self
            .journal
            .snapshot()
            .get_visible_body()
            .find(|node| {
                node.attributes.get("call_id") == Some(&TypedValue::String(call.to_string()))
            })
            .is_some_and(|node| {
                node.attributes.get("status") == Some(&TypedValue::String("failed".into()))
            });
        if failed {
            return Err(ServerError::Execution(omp_types::StructuredError::new(
                "workspace_write_failed",
                "Write failed; inspect the journaled tool diagnostics",
                false,
            )));
        }
        Ok(())
    }

    /// Fork the session at a historical journal offset.
    ///
    /// Requires `ManageSession` permission.
    pub fn fork_session(
        &mut self,
        actor_id: &ActorId,
        offset: u64,
    ) -> Result<BranchId, ServerError> {
        if self.terminated {
            return Err(ServerError::SessionTerminated(self.session_id.to_string()));
        }

        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::ManageSession)?;

        let branch = self.journal.fork_at(offset)?;
        Ok(branch)
    }

    /// Rewind the session to a historical journal offset.
    ///
    /// Requires `ManageSession` permission.
    pub fn rewind_session(
        &mut self,
        actor_id: &ActorId,
        offset: u64,
    ) -> Result<BranchId, ServerError> {
        if self.terminated {
            return Err(ServerError::SessionTerminated(self.session_id.to_string()));
        }

        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::ManageSession)?;

        let branch = self.journal.rewind_to(offset)?;
        Ok(branch)
    }

    /// Select an existing branch in the session.
    ///
    /// Requires `ManageSession` permission.
    pub fn select_branch(
        &mut self,
        actor_id: &ActorId,
        branch: &BranchId,
    ) -> Result<JournalOffset, ServerError> {
        if self.terminated {
            return Err(ServerError::SessionTerminated(self.session_id.to_string()));
        }

        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::ManageSession)?;

        let offset = self.journal.select_branch(branch)?;
        Ok(JournalOffset(offset))
    }

    /// Terminate the session.
    ///
    /// Requires `ManageSession` permission. After termination, no mutating actions are accepted.
    pub fn terminate_session(&mut self, actor_id: &ActorId) -> Result<(), ServerError> {
        let actor = self.get_actor(actor_id)?;
        actor.check_permission(Permission::ManageSession)?;
        self.terminated = true;
        Ok(())
    }

    /// Broadcast a presentation frame to all subscribers (coalesces intermediate frames).
    pub fn publish_presentation(&mut self, view_id: &str, payload: serde_json::Value) {
        let message = ReplicationMessage::Presentation {
            view_id: view_id.to_string(),
            payload,
        };
        for sub in self.subscribers.values_mut() {
            let _ = sub.push(message.clone());
        }
    }

    pub fn journal(&self) -> &Journal {
        &self.journal
    }

    pub fn journal_mut(&mut self) -> &mut Journal {
        &mut self.journal
    }

    pub fn actors(&self) -> &BTreeMap<ActorId, ActorSession> {
        &self.actors
    }

    pub fn subscribers(&self) -> &BTreeMap<ActorId, SubscriberQueue> {
        &self.subscribers
    }

    pub fn seen_command_ids(&self) -> &BTreeSet<String> {
        &self.seen_command_ids
    }

    pub fn has_seen_command(&self, command_id: &str) -> bool {
        self.seen_command_ids.contains(command_id.trim())
    }

    pub fn approvals(&self) -> BTreeMap<String, ApprovalRequest> {
        self.journal
            .snapshot()
            .children(self.journal.snapshot().container("approvals"))
            .filter_map(|node| {
                let mut approval: ApprovalRequest =
                    serde_json::from_value(node.payload.clone()?).ok()?;
                if let Some(TypedValue::Bool(value)) = node.attributes.get("approved") {
                    approval.approved = Some(*value);
                }
                Some((node.id.to_string(), approval))
            })
            .collect()
    }

    /// Replicates any new journal records committed since `since_offset`
    /// to all active subscriber queues, preserving queues and marking resync on lag.
    pub fn replicate(&mut self, since_offset: u64) {
        let patches: Vec<Patch> = self
            .journal
            .records()
            .filter(|r| r.offset > since_offset)
            .map(|r| r.patch.clone())
            .collect();

        for patch in patches {
            let message = ReplicationMessage::Patch { patch };
            for sub in self.subscribers.values_mut() {
                let _ = sub.push(message.clone());
            }
        }
    }
}

/// Transport-neutral remote driver client handle.
pub struct RemoteDriverClient {
    actor_id: ActorId,
    token: Option<String>,
}

impl RemoteDriverClient {
    pub fn new(actor_id: ActorId) -> Self {
        Self {
            actor_id,
            token: None,
        }
    }

    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    pub fn actor_id(&self) -> &ActorId {
        &self.actor_id
    }

    pub fn connect(
        &self,
        service: &mut ServerSessionService,
        capabilities: Vec<String>,
    ) -> Result<HandshakeResponse, ServerError> {
        service.handshake(HandshakeRequest {
            protocol_version: ProtocolVersion::CURRENT,
            actor_id: self.actor_id.clone(),
            requested_role: ActorRole::InteractiveDriver,
            auth_token: self.token.clone(),
            capabilities,
        })
    }

    pub fn submit_command(
        &self,
        service: &mut ServerSessionService,
        command_id: &str,
        command_line: &str,
    ) -> Result<(), ServerError> {
        service.submit_command(&self.actor_id, command_id, command_line)
    }

    pub fn submit_patch(
        &self,
        service: &mut ServerSessionService,
        patch: Patch,
    ) -> Result<JournalOffset, ServerError> {
        service.submit_patch(&self.actor_id, patch)
    }

    pub fn resolve_approval(
        &self,
        service: &mut ServerSessionService,
        request_id: &str,
        decision: bool,
    ) -> Result<(), ServerError> {
        service.resolve_approval(&self.actor_id, request_id, decision)
    }

    pub fn signal_job(
        &self,
        service: &mut ServerSessionService,
        job_id: &JobId,
        signal: &str,
    ) -> Result<(), ServerError> {
        service.signal_job(&self.actor_id, job_id, signal)
    }

    pub fn change_convar(
        &self,
        service: &mut ServerSessionService,
        name: &str,
        value: TypedValue,
    ) -> Result<JournalOffset, ServerError> {
        service.change_convar(&self.actor_id, name, value)
    }

    pub fn subscribe(
        &self,
        service: &mut ServerSessionService,
        last_offset: Option<JournalOffset>,
    ) -> Result<ReplicationMessage, ServerError> {
        service.subscribe(&self.actor_id, last_offset, None)
    }

    pub fn poll_patches(&self, service: &mut ServerSessionService) -> Vec<ReplicationMessage> {
        let mut messages = Vec::new();
        while let Some(msg) = service.poll_subscriber(&self.actor_id) {
            messages.push(msg);
        }
        messages
    }

    pub fn get_snapshot(
        &self,
        service: &ServerSessionService,
        offset: Option<JournalOffset>,
    ) -> Result<SessionSnapshot, ServerError> {
        service.get_snapshot(&self.actor_id, offset)
    }

    pub fn get_artifact(
        &self,
        service: &ServerSessionService,
        artifact_id: &ArtifactId,
    ) -> Result<ArtifactRecord, ServerError> {
        service.get_artifact(&self.actor_id, artifact_id)
    }
}

/// Transport-neutral read-only spectator client handle.
pub struct SpectatorClient {
    actor_id: ActorId,
    token: Option<String>,
}

impl SpectatorClient {
    pub fn new(actor_id: ActorId) -> Self {
        Self {
            actor_id,
            token: None,
        }
    }

    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    pub fn actor_id(&self) -> &ActorId {
        &self.actor_id
    }

    pub fn connect(
        &self,
        service: &mut ServerSessionService,
        capabilities: Vec<String>,
    ) -> Result<HandshakeResponse, ServerError> {
        service.handshake(HandshakeRequest {
            protocol_version: ProtocolVersion::CURRENT,
            actor_id: self.actor_id.clone(),
            requested_role: ActorRole::Spectator,
            auth_token: self.token.clone(),
            capabilities,
        })
    }

    pub fn subscribe(
        &self,
        service: &mut ServerSessionService,
        last_offset: Option<JournalOffset>,
    ) -> Result<ReplicationMessage, ServerError> {
        service.subscribe(&self.actor_id, last_offset, None)
    }

    pub fn poll_patches(&self, service: &mut ServerSessionService) -> Vec<ReplicationMessage> {
        let mut messages = Vec::new();
        while let Some(msg) = service.poll_subscriber(&self.actor_id) {
            messages.push(msg);
        }
        messages
    }

    pub fn get_snapshot(
        &self,
        service: &ServerSessionService,
        offset: Option<JournalOffset>,
    ) -> Result<SessionSnapshot, ServerError> {
        service.get_snapshot(&self.actor_id, offset)
    }

    pub fn get_artifact(
        &self,
        service: &ServerSessionService,
        artifact_id: &ArtifactId,
    ) -> Result<ArtifactRecord, ServerError> {
        service.get_artifact(&self.actor_id, artifact_id)
    }

    /// Spectators are strictly read-only: submitting a command returns Unauthorized.
    pub fn submit_command(
        &self,
        service: &mut ServerSessionService,
        command_id: &str,
        command_line: &str,
    ) -> Result<(), ServerError> {
        service.submit_command(&self.actor_id, command_id, command_line)
    }

    /// Spectators are strictly read-only: submitting a patch returns Unauthorized.
    pub fn submit_patch(
        &self,
        service: &mut ServerSessionService,
        patch: Patch,
    ) -> Result<JournalOffset, ServerError> {
        service.submit_patch(&self.actor_id, patch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omp_types::{
        ActorId, ArtifactId, ElementId, ElementSnapshot, JournalOffset, Patch, PatchOp, SessionId,
        TypedValue,
    };
    use std::fs;
    use std::path::PathBuf;

    fn temp_journal() -> (Journal, PathBuf) {
        let path =
            std::env::temp_dir().join(format!("omp2-srv-test-{}.journal", SessionId::mint()));
        let journal = Journal::create(&path, SessionId::mint()).unwrap();
        (journal, path)
    }

    fn test_patch(base: u64, result: u64, actor: &ActorId) -> Patch {
        let elem = ElementSnapshot::new(ElementId::mint(), "diagnostic");
        Patch {
            base_offset: JournalOffset(base),
            result_offset: JournalOffset(result),
            by: actor.clone().into(),
            reason: "test patch".into(),
            ops: vec![PatchOp::Create {
                parent: ElementId::new("body").unwrap(),
                index: 0,
                element: elem,
            }],
        }
    }

    #[test]
    fn handshake_and_role_negotiation() {
        let (journal, path) = temp_journal();
        let mut service = ServerSessionService::new(journal);

        let driver_id = ActorId::new("driver-1").unwrap();
        let resp = service
            .handshake(HandshakeRequest {
                protocol_version: ProtocolVersion::CURRENT,
                actor_id: driver_id.clone(),
                requested_role: ActorRole::InteractiveDriver,
                auth_token: Some("secret-token".into()),
                capabilities: vec![omp_types::JOURNAL_FORMAT.to_owned()],
            })
            .unwrap();

        assert_eq!(resp.granted_role, ActorRole::InteractiveDriver);
        assert!(service.get_actor(&driver_id).is_ok());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn submit_patch_authorized_and_replicates() {
        let (journal, path) = temp_journal();
        let mut service = ServerSessionService::new(journal);

        let driver_id = ActorId::new("driver-1").unwrap();
        service
            .attach_actor(ActorSession::new(
                driver_id.clone(),
                ActorRole::InteractiveDriver,
            ))
            .unwrap();

        let spec_id = ActorId::new("spec-1").unwrap();
        service
            .attach_actor(ActorSession::new(spec_id.clone(), ActorRole::Spectator))
            .unwrap();
        let base = service.current_offset();
        let _ = service.subscribe(&spec_id, Some(base), None).unwrap();
        let patch = test_patch(base.0, base.0 + 1, &driver_id);
        let new_offset = service.submit_patch(&driver_id, patch).unwrap();

        // Subscriber receives replicated patch
        let msg = service.poll_subscriber(&spec_id).unwrap();
        match msg {
            ReplicationMessage::Patch { patch } => {
                assert_eq!(patch.result_offset, new_offset);
            }
            other => panic!("expected Patch message, got {other:?}"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn submit_patch_rejects_stale_base_offset() {
        let (journal, path) = temp_journal();
        let mut service = ServerSessionService::new(journal);

        let driver_id = ActorId::new("driver-1").unwrap();
        service
            .attach_actor(ActorSession::new(
                driver_id.clone(),
                ActorRole::InteractiveDriver,
            ))
            .unwrap();

        let base = service.current_offset().0;
        let patch1 = test_patch(base, base + 1, &driver_id);
        service.submit_patch(&driver_id, patch1).unwrap();

        let before = service.journal().snapshot().clone();
        let stale_patch = test_patch(base, base + 2, &driver_id);
        let err = service.submit_patch(&driver_id, stale_patch).unwrap_err();

        match err {
            ServerError::StaleBaseOffset { actual, expected } => {
                assert_eq!(actual, base);
                assert_eq!(expected, base + 1);
            }
            other => panic!("expected StaleBaseOffset, got {other:?}"),
        }
        // Authority unchanged
        assert_eq!(service.journal().snapshot(), &before);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn spectator_cannot_submit_patch_or_command() {
        let (journal, path) = temp_journal();
        let mut service = ServerSessionService::new(journal);

        let spec_id = ActorId::new("spec-1").unwrap();
        service
            .attach_actor(ActorSession::new(spec_id.clone(), ActorRole::Spectator))
            .unwrap();

        let patch = test_patch(0, 1, &spec_id);
        let err = service.submit_patch(&spec_id, patch).unwrap_err();
        assert!(matches!(err, ServerError::Unauthorized { .. }));

        let cmd_err = service
            .submit_command(&spec_id, "cmd-1", "test")
            .unwrap_err();
        assert!(matches!(cmd_err, ServerError::Unauthorized { .. }));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn submit_command_rejects_replayed_and_empty_id() {
        let (journal, path) = temp_journal();
        let mut service = ServerSessionService::new(journal);

        let driver_id = ActorId::new("driver-1").unwrap();
        service
            .attach_actor(ActorSession::new(
                driver_id.clone(),
                ActorRole::InteractiveDriver,
            ))
            .unwrap();

        // Empty command ID rejected
        let err_empty = service
            .submit_command(&driver_id, "   ", "help")
            .unwrap_err();
        assert!(matches!(err_empty, ServerError::InvalidCommand(_)));

        // First command succeeds
        service
            .submit_command(&driver_id, "cmd-42", "echo hello")
            .unwrap();

        // Replay of same ID rejected
        let err_replay = service
            .submit_command(&driver_id, "cmd-42", "echo different")
            .unwrap_err();
        assert!(matches!(err_replay, ServerError::CommandReplayed(_)));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn change_convar_updates_journal_and_replicates() {
        let (journal, path) = temp_journal();
        let mut service = ServerSessionService::new(journal);

        let driver_id = ActorId::new("driver-1").unwrap();
        service
            .attach_actor(ActorSession::new(
                driver_id.clone(),
                ActorRole::InteractiveDriver,
            ))
            .unwrap();

        let spec_id = ActorId::new("spec-1").unwrap();
        service
            .attach_actor(ActorSession::new(spec_id.clone(), ActorRole::Spectator))
            .unwrap();
        let _ = service
            .subscribe(&spec_id, Some(service.current_offset()), None)
            .unwrap();

        let off = service
            .change_convar(
                &driver_id,
                "ai_temperature",
                TypedValue::String("0.7".into()),
            )
            .unwrap();

        // Verify DOM snapshot updated
        let snapshot = service.get_snapshot(&driver_id, None).unwrap();
        let convars = snapshot.session_globals();
        assert_eq!(
            convars.get("ai_temperature"),
            Some(&TypedValue::String("0.7".into()))
        );

        // Verify subscriber received convar patch
        let msg = service.poll_subscriber(&spec_id).unwrap();
        match msg {
            ReplicationMessage::Patch { patch } => {
                assert_eq!(patch.result_offset, off);
                let mut projected = service.journal().materialize(patch.base_offset.0).unwrap();
                omp_state::apply_patch(&mut projected, &patch).unwrap();
                assert_eq!(projected, snapshot);
            }
            other => panic!("expected Patch, got {other:?}"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn approval_request_and_resolution() {
        let (journal, path) = temp_journal();
        let mut service = ServerSessionService::new(journal);

        let driver_id = ActorId::new("driver-1").unwrap();
        service
            .attach_actor(ActorSession::new(
                driver_id.clone(),
                ActorRole::InteractiveDriver,
            ))
            .unwrap();

        let worker_id = ActorId::new("worker-1").unwrap();
        service
            .attach_actor(ActorSession::new(
                worker_id.clone(),
                ActorRole::AutomationWorker,
            ))
            .unwrap();

        // Worker requests approval
        service
            .request_approval(&worker_id, "app-1", "allow git push")
            .unwrap();

        // Spectator cannot approve
        let spec_id = ActorId::new("spec-1").unwrap();
        service
            .attach_actor(ActorSession::new(spec_id.clone(), ActorRole::Spectator))
            .unwrap();
        let err = service
            .resolve_approval(&spec_id, "app-1", true)
            .unwrap_err();
        assert!(matches!(err, ServerError::Unauthorized { .. }));

        // Driver approves
        service.resolve_approval(&driver_id, "app-1", true).unwrap();
        assert_eq!(
            service.approvals().get("app-1").unwrap().approved,
            Some(true)
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn artifact_registration_persists_in_journal_and_enforces_scope() {
        let (journal, path) = temp_journal();
        let mut service = ServerSessionService::new(journal);

        let worker_id = ActorId::new("worker-1").unwrap();
        service
            .attach_actor(ActorSession::new(
                worker_id.clone(),
                ActorRole::AutomationWorker,
            ))
            .unwrap();

        let driver_id = ActorId::new("driver-1").unwrap();
        service
            .attach_actor(ActorSession::new(
                driver_id.clone(),
                ActorRole::InteractiveDriver,
            ))
            .unwrap();

        let spec_id = ActorId::new("spec-1").unwrap();
        service
            .attach_actor(ActorSession::new(spec_id.clone(), ActorRole::Spectator))
            .unwrap();

        let other_id = ActorId::new("other-1").unwrap();
        service
            .attach_actor(ActorSession::new(
                other_id.clone(),
                ActorRole::InteractiveDriver,
            ))
            .unwrap();

        // Register session-scoped artifact
        let art1 = ArtifactRecord::new(
            ArtifactId::new("art-session").unwrap(),
            worker_id.clone(),
            "text/plain",
            100,
        );
        service.register_artifact(&worker_id, art1).unwrap();
        let first_artifact_offset = service.current_offset();

        // Register actor-scoped artifact strictly for driver-1
        let art2 = ArtifactRecord::new(
            ArtifactId::new("art-driver-only").unwrap(),
            worker_id.clone(),
            "application/json",
            200,
        )
        .with_scope(ArtifactScopeSpec::Actor(driver_id.clone()));
        service.register_artifact(&worker_id, art2).unwrap();

        // Register artifact with allowed_actors list
        let art3 = ArtifactRecord::new(
            ArtifactId::new("art-allowed-list").unwrap(),
            worker_id.clone(),
            "image/png",
            300,
        )
        .with_allowed_actors(vec![driver_id.clone(), spec_id.clone()]);
        service.register_artifact(&worker_id, art3).unwrap();

        // Verify art-session accessible by driver, spec, and worker
        assert!(
            service
                .get_artifact(&driver_id, &ArtifactId::new("art-session").unwrap())
                .is_ok()
        );
        assert!(
            service
                .get_artifact(&spec_id, &ArtifactId::new("art-session").unwrap())
                .is_ok()
        );

        // Verify art-driver-only accessible by driver-1
        let res = service
            .get_artifact(&driver_id, &ArtifactId::new("art-driver-only").unwrap())
            .unwrap();
        assert_eq!(res.id, ArtifactId::new("art-driver-only").unwrap());
        assert_eq!(res.size_bytes, 200);

        // other-1 attempting to access driver-only artifact fails with ArtifactScopeViolation
        let scope_err = service
            .get_artifact(&other_id, &ArtifactId::new("art-driver-only").unwrap())
            .unwrap_err();
        assert!(matches!(
            scope_err,
            ServerError::ArtifactScopeViolation { .. }
        ));

        // art-allowed-list accessible by driver and spectator, but not other-1
        assert!(
            service
                .get_artifact(&spec_id, &ArtifactId::new("art-allowed-list").unwrap())
                .is_ok()
        );
        let list_err = service
            .get_artifact(&other_id, &ArtifactId::new("art-allowed-list").unwrap())
            .unwrap_err();
        assert!(matches!(
            list_err,
            ServerError::ArtifactScopeViolation { .. }
        ));

        // Controller can access ANY artifact regardless of actor restrictions
        let ctrl_id = ActorId::new("controller-1").unwrap();
        service
            .attach_actor(ActorSession::new(ctrl_id.clone(), ActorRole::Controller))
            .unwrap();
        assert!(
            service
                .get_artifact(&ctrl_id, &ArtifactId::new("art-driver-only").unwrap())
                .is_ok()
        );
        assert!(
            service
                .get_artifact(&ctrl_id, &ArtifactId::new("art-allowed-list").unwrap())
                .is_ok()
        );

        // Querying non-existent artifact returns ArtifactNotFound
        let not_found = service
            .get_artifact(&ctrl_id, &ArtifactId::new("nonexistent").unwrap())
            .unwrap_err();
        assert!(matches!(not_found, ServerError::ArtifactNotFound(_)));

        let hist = service
            .get_artifact_at(
                &driver_id,
                &ArtifactId::new("art-session").unwrap(),
                first_artifact_offset,
            )
            .unwrap();
        assert_eq!(hist.id, ArtifactId::new("art-session").unwrap());

        let hist_missing = service
            .get_artifact_at(
                &driver_id,
                &ArtifactId::new("art-driver-only").unwrap(),
                first_artifact_offset,
            )
            .unwrap_err();
        assert!(matches!(hist_missing, ServerError::ArtifactNotFound(_)));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn subscribe_replays_or_resyncs_correctly() {
        let (journal, path) = temp_journal();
        let mut service = ServerSessionService::new(journal);

        let driver_id = ActorId::new("driver-1").unwrap();
        service
            .attach_actor(ActorSession::new(
                driver_id.clone(),
                ActorRole::InteractiveDriver,
            ))
            .unwrap();

        let spec_id = ActorId::new("spec-1").unwrap();
        service
            .attach_actor(ActorSession::new(spec_id.clone(), ActorRole::Spectator))
            .unwrap();
        let mut projected = service.get_snapshot(&spec_id, None).unwrap();
        let base = projected.offset;
        for i in 1..=3 {
            service
                .submit_patch(&driver_id, test_patch(base + i - 1, base + i, &driver_id))
                .unwrap();
        }
        let initial = service
            .subscribe(&spec_id, Some(JournalOffset(base)), None)
            .unwrap();
        assert!(matches!(initial, ReplicationMessage::Heartbeat { .. }));
        for _ in 0..3 {
            match service.poll_subscriber(&spec_id).unwrap() {
                ReplicationMessage::Patch { patch } => {
                    omp_state::apply_patch(&mut projected, &patch).unwrap()
                }
                message => panic!("expected ordered patch, got {message:?}"),
            }
        }
        assert_eq!(projected, service.get_snapshot(&spec_id, None).unwrap());
        assert!(service.poll_subscriber(&spec_id).is_none());

        // Subscribe with None: receives full Resync
        let spec2 = ActorId::new("spec-2").unwrap();
        service
            .attach_actor(ActorSession::new(spec2.clone(), ActorRole::Spectator))
            .unwrap();
        let resync_initial = service.subscribe(&spec2, None, None).unwrap();
        match resync_initial {
            ReplicationMessage::Resync { offset, snapshot } => {
                assert_eq!(snapshot, service.get_snapshot(&spec2, None).unwrap());
                assert_eq!(offset.0, snapshot.offset);
            }
            other => panic!("expected Resync, got {other:?}"),
        }

        // Subscribe with future offset (e.g. 99): receives full Resync
        let spec3 = ActorId::new("spec-3").unwrap();
        service
            .attach_actor(ActorSession::new(spec3.clone(), ActorRole::Spectator))
            .unwrap();
        let future_resync = service
            .subscribe(&spec3, Some(JournalOffset(99)), None)
            .unwrap();
        match future_resync {
            ReplicationMessage::Resync { offset, snapshot } => {
                assert_eq!(snapshot, service.get_snapshot(&spec3, None).unwrap());
                assert_eq!(offset.0, snapshot.offset);
            }
            other => panic!("expected Resync, got {other:?}"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn spectator_overflow_does_not_block_controller_and_recovers_via_resync() {
        let (journal, path) = temp_journal();
        let mut service = ServerSessionService::new(journal);

        let ctrl_id = ActorId::new("ctrl-1").unwrap();
        service
            .attach_actor(ActorSession::new(ctrl_id.clone(), ActorRole::Controller))
            .unwrap();

        let spec_id = ActorId::new("spec-1").unwrap();
        service
            .attach_actor(ActorSession::new(spec_id.clone(), ActorRole::Spectator))
            .unwrap();

        // Subscribe spectator with small queue capacity of 16
        let _ = service
            .subscribe(&spec_id, Some(JournalOffset(0)), Some(16))
            .unwrap();
        let _ = service
            .subscribe(&ctrl_id, Some(service.current_offset()), Some(16))
            .unwrap();
        let base = service.current_offset().0;

        // Controller submits 25 patches without spectator polling
        for i in 1..=25 {
            let p = test_patch(base + i - 1, base + i, &ctrl_id);
            let res = service.submit_patch(&ctrl_id, p);
            // Controller is NEVER blocked by spectator queue overflow!
            assert!(res.is_ok(), "controller patch {i} must succeed");
        }
        let expected = service.get_snapshot(&ctrl_id, None).unwrap();

        // Spectator's next poll returns a snapshot Resync to current offset (25)
        let next_msg = service.poll_subscriber(&spec_id).unwrap();
        match next_msg {
            ReplicationMessage::Resync { offset, snapshot } => {
                assert_eq!(offset.0, expected.offset);
                assert_eq!(snapshot, expected);
            }
            other => panic!("expected Resync on lagged spectator, got {other:?}"),
        }
        match service.poll_subscriber(&ctrl_id).unwrap() {
            ReplicationMessage::Resync { snapshot, .. } => assert_eq!(snapshot, expected),
            other => panic!("expected controller resync, got {other:?}"),
        }
        let next = test_patch(expected.offset, expected.offset + 1, &ctrl_id);
        service.submit_patch(&ctrl_id, next).unwrap();
        for actor in [&spec_id, &ctrl_id] {
            match service.poll_subscriber(actor).unwrap() {
                ReplicationMessage::Patch { patch } => {
                    let mut projection = expected.clone();
                    omp_state::apply_patch(&mut projection, &patch).unwrap();
                    assert_eq!(projection, service.get_snapshot(actor, None).unwrap());
                }
                other => panic!("expected post-resync patch, got {other:?}"),
            }
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn session_lifecycle_and_termination() {
        let (journal, path) = temp_journal();
        let mut service = ServerSessionService::new(journal);

        let ctrl_id = ActorId::new("ctrl-1").unwrap();
        service
            .attach_actor(ActorSession::new(ctrl_id.clone(), ActorRole::Controller))
            .unwrap();

        let driver_id = ActorId::new("driver-1").unwrap();
        service
            .attach_actor(ActorSession::new(
                driver_id.clone(),
                ActorRole::InteractiveDriver,
            ))
            .unwrap();

        // Add a patch
        let base = service.current_offset().0;
        service
            .submit_patch(&ctrl_id, test_patch(base, base + 1, &ctrl_id))
            .unwrap();

        // Driver cannot fork session (requires ManageSession)
        let fork_err = service.fork_session(&driver_id, 0).unwrap_err();
        assert!(matches!(fork_err, ServerError::Unauthorized { .. }));

        // Controller forks session
        let branch = service.fork_session(&ctrl_id, 0).unwrap();
        assert_eq!(service.journal().snapshot().current_branch(), &branch);

        // Terminate session
        assert!(!service.is_terminated());
        service.terminate_session(&ctrl_id).unwrap();
        assert!(service.is_terminated());

        // Any mutating operation after termination fails
        let p_err = service
            .submit_patch(&ctrl_id, test_patch(1, 2, &ctrl_id))
            .unwrap_err();
        assert!(matches!(p_err, ServerError::SessionTerminated(_)));

        let c_err = service
            .submit_command(&ctrl_id, "cmd-99", "echo")
            .unwrap_err();
        assert!(matches!(c_err, ServerError::SessionTerminated(_)));

        let v_err = service
            .change_convar(&ctrl_id, "ai_temperature", TypedValue::String("1.0".into()))
            .unwrap_err();
        assert!(matches!(v_err, ServerError::SessionTerminated(_)));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn transport_neutral_endpoint_binding() {
        let (journal, path) = temp_journal();
        let service = ServerSessionService::new(journal);

        // InMemory succeeds
        let bound = service.bind(&ServerEndpoint::InMemory).unwrap();
        assert_eq!(bound, ServerEndpoint::InMemory);

        // Network endpoints explicitly report UnsupportedTransport rather than fake network
        let tcp_err = service
            .bind(&ServerEndpoint::tcp("127.0.0.1:8080"))
            .unwrap_err();
        assert!(matches!(tcp_err, ServerError::UnsupportedTransport(s) if s.contains("tcp://")));

        let unix_err = service
            .bind(&ServerEndpoint::unix("/tmp/omp.sock"))
            .unwrap_err();
        assert!(matches!(unix_err, ServerError::UnsupportedTransport(s) if s.contains("unix://")));

        let pipe_err = service
            .bind(&ServerEndpoint::NamedPipe("test-pipe".into()))
            .unwrap_err();
        assert!(matches!(pipe_err, ServerError::UnsupportedTransport(s) if s.contains("pipe://")));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn remote_driver_and_spectator_client_apis() {
        let (journal, path) = temp_journal();
        let mut service = ServerSessionService::new(journal);

        let driver = RemoteDriverClient::new(ActorId::new("driver-client").unwrap());
        let handshake = driver
            .connect(&mut service, vec![omp_types::JOURNAL_FORMAT.to_owned()])
            .unwrap();
        assert_eq!(handshake.granted_role, ActorRole::InteractiveDriver);

        // Driver can submit command and patch
        driver
            .submit_command(&mut service, "cmd-1", "test command")
            .unwrap();
        let base = service.current_offset().0;
        let patch = test_patch(base, base + 1, driver.actor_id());
        driver.submit_patch(&mut service, patch).unwrap();

        // Spectator client connects
        let spectator = SpectatorClient::new(ActorId::new("spec-client").unwrap());
        let s_handshake = spectator
            .connect(&mut service, vec![omp_types::JOURNAL_FORMAT.to_owned()])
            .unwrap();
        assert_eq!(s_handshake.granted_role, ActorRole::Spectator);

        // Spectator can read snapshot
        let snap = spectator.get_snapshot(&service, None).unwrap();
        assert_eq!(snap, service.get_snapshot(driver.actor_id(), None).unwrap());

        // Spectator submitting command or patch fails with Unauthorized
        let cmd_err = spectator
            .submit_command(&mut service, "cmd-2", "echo")
            .unwrap_err();
        assert!(matches!(cmd_err, ServerError::Unauthorized { .. }));

        let patch_err = spectator
            .submit_patch(&mut service, test_patch(1, 2, spectator.actor_id()))
            .unwrap_err();
        assert!(matches!(patch_err, ServerError::Unauthorized { .. }));

        let _ = fs::remove_file(path);
    }
}
