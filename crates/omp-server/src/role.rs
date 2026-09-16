use omp_types::ActorId;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum ActorRole {
    Controller,
    InteractiveDriver,
    Spectator,
    SubagentInspector,
    AutomationWorker,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Permission {
    ReadSnapshot,
    SubscribePatches,
    SubmitCommand,
    Approve,
    SignalJob,
    ChangeConVar,
    WriteWorkspace,
    SubmitPatch,
    ManageSession,
    AccessArtifact,
    AttachSubagent,
}

impl ActorRole {
    pub fn has_permission(&self, perm: Permission) -> bool {
        match self {
            ActorRole::Controller => true,
            ActorRole::InteractiveDriver => matches!(
                perm,
                Permission::ReadSnapshot
                    | Permission::SubscribePatches
                    | Permission::SubmitCommand
                    | Permission::Approve
                    | Permission::SignalJob
                    | Permission::ChangeConVar
                    | Permission::SubmitPatch
                    | Permission::AccessArtifact
                    | Permission::AttachSubagent
            ),
            ActorRole::Spectator => matches!(
                perm,
                Permission::ReadSnapshot
                    | Permission::SubscribePatches
                    | Permission::AccessArtifact
            ),
            ActorRole::SubagentInspector => matches!(
                perm,
                Permission::ReadSnapshot
                    | Permission::SubscribePatches
                    | Permission::AccessArtifact
            ),
            ActorRole::AutomationWorker => matches!(
                perm,
                Permission::ReadSnapshot
                    | Permission::SubscribePatches
                    | Permission::SubmitCommand
                    | Permission::SignalJob
                    | Permission::WriteWorkspace
                    | Permission::SubmitPatch
                    | Permission::AccessArtifact
            ),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActorSession {
    pub actor_id: ActorId,
    pub role: ActorRole,
    #[serde(skip)]
    pub token: Option<String>,
    pub capabilities: Vec<String>,
    pub attached_at_epoch_ms: u64,
}

impl ActorSession {
    pub fn new(actor_id: ActorId, role: ActorRole) -> Self {
        Self {
            actor_id,
            role,
            token: None,
            capabilities: Vec::new(),
            attached_at_epoch_ms: 0,
        }
    }

    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    pub fn with_capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn check_permission(&self, perm: Permission) -> Result<(), crate::error::ServerError> {
        if self.role.has_permission(perm) {
            Ok(())
        } else {
            Err(crate::error::ServerError::Unauthorized {
                actor: self.actor_id.to_string(),
                role: self.role,
                attempted_action: format!("{perm:?}"),
            })
        }
    }

    pub fn has_capability(&self, cap: &str) -> bool {
        self.capabilities.iter().any(|c| c == cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_has_all_permissions() {
        let perms = [
            Permission::ReadSnapshot,
            Permission::SubscribePatches,
            Permission::SubmitCommand,
            Permission::Approve,
            Permission::SignalJob,
            Permission::ChangeConVar,
            Permission::WriteWorkspace,
            Permission::SubmitPatch,
            Permission::ManageSession,
            Permission::AccessArtifact,
            Permission::AttachSubagent,
        ];
        for perm in perms {
            assert!(ActorRole::Controller.has_permission(perm));
        }
    }

    #[test]
    fn spectator_is_strictly_readonly() {
        assert!(ActorRole::Spectator.has_permission(Permission::ReadSnapshot));
        assert!(ActorRole::Spectator.has_permission(Permission::SubscribePatches));
        assert!(ActorRole::Spectator.has_permission(Permission::AccessArtifact));

        assert!(!ActorRole::Spectator.has_permission(Permission::SubmitCommand));
        assert!(!ActorRole::Spectator.has_permission(Permission::Approve));
        assert!(!ActorRole::Spectator.has_permission(Permission::SignalJob));
        assert!(!ActorRole::Spectator.has_permission(Permission::ChangeConVar));
        assert!(!ActorRole::Spectator.has_permission(Permission::WriteWorkspace));
        assert!(!ActorRole::Spectator.has_permission(Permission::SubmitPatch));
        assert!(!ActorRole::Spectator.has_permission(Permission::ManageSession));
        assert!(!ActorRole::Spectator.has_permission(Permission::AttachSubagent));
    }

    #[test]
    fn interactive_driver_permissions() {
        assert!(ActorRole::InteractiveDriver.has_permission(Permission::SubmitCommand));
        assert!(ActorRole::InteractiveDriver.has_permission(Permission::Approve));
        assert!(ActorRole::InteractiveDriver.has_permission(Permission::SignalJob));
        assert!(ActorRole::InteractiveDriver.has_permission(Permission::ChangeConVar));
        assert!(ActorRole::InteractiveDriver.has_permission(Permission::SubmitPatch));
        assert!(ActorRole::InteractiveDriver.has_permission(Permission::AttachSubagent));
        assert!(!ActorRole::InteractiveDriver.has_permission(Permission::ManageSession));
        assert!(!ActorRole::InteractiveDriver.has_permission(Permission::WriteWorkspace));
    }

    #[test]
    fn automation_worker_permissions() {
        assert!(ActorRole::AutomationWorker.has_permission(Permission::WriteWorkspace));
        assert!(ActorRole::AutomationWorker.has_permission(Permission::SubmitCommand));
        assert!(ActorRole::AutomationWorker.has_permission(Permission::SignalJob));
        assert!(ActorRole::AutomationWorker.has_permission(Permission::SubmitPatch));
        assert!(!ActorRole::AutomationWorker.has_permission(Permission::Approve));
        assert!(!ActorRole::AutomationWorker.has_permission(Permission::ChangeConVar));
        assert!(!ActorRole::AutomationWorker.has_permission(Permission::ManageSession));
    }

    #[test]
    fn check_permission_reports_unauthorized_action() {
        let session = ActorSession::new(ActorId::new("spectator-1").unwrap(), ActorRole::Spectator);
        assert!(session.check_permission(Permission::ReadSnapshot).is_ok());
        let err = session
            .check_permission(Permission::SubmitCommand)
            .unwrap_err();
        match &err {
            crate::error::ServerError::Unauthorized {
                actor,
                role,
                attempted_action,
            } => {
                assert_eq!(actor, "spectator-1");
                assert_eq!(*role, ActorRole::Spectator);
                assert!(attempted_action.contains("SubmitCommand"));
            }
            other => panic!("expected Unauthorized, got {other:?}"),
        }
    }

    #[test]
    fn actor_session_does_not_serialize_token() {
        let session = ActorSession::new(ActorId::new("owner").unwrap(), ActorRole::Controller)
            .with_token("secret-owner-token");
        let value = serde_json::to_value(&session).unwrap();
        assert!(value.get("token").is_none());
        assert_eq!(value["role"], "Controller");
    }
}
