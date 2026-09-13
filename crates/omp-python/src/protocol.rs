use omp_types::{Patch, PatchOp};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LifecycleEvent {
    Load {
        extension_id: String,
        manifest_json: String,
    },
    Unload {
        extension_id: String,
    },
    Reload {
        extension_id: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolDeclaration {
    pub name: String,
    pub version: String,
    pub description: String,
    pub parameter_schema: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DirectorDeclaration {
    pub name: String,
    pub priority: i32,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentDeclaration {
    pub name: String,
    pub schema: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HostQuery {
    GetConVar { name: String },
    GetSnapshot { offset: Option<u64> },
    GetArtifact { id: String },
    QueryDOM { selector: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostPatchRequest {
    pub ops: Vec<PatchOp>,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmitJobRequest {
    pub job_id: String,
    pub operation: String,
    pub capabilities: Vec<String>,
    pub payload: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HostToExtension {
    Lifecycle(LifecycleEvent),
    InvokeTool {
        call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
    },
    InvokeDirector {
        director_id: String,
        turn_id: String,
        event: String,
        context: serde_json::Value,
    },
    NotifyPatch {
        patch: Patch,
    },
    ExecuteRemote {
        request_id: String,
        function_name: String,
        source_hash: String,
        arguments: serde_json::Value,
    },
    Cancel {
        request_id: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ExtensionToHost {
    RegisterDeclarations {
        extension_id: String,
        tools: Vec<ToolDeclaration>,
        directors: Vec<DirectorDeclaration>,
        components: Vec<ComponentDeclaration>,
    },
    QueryHost {
        query_id: String,
        query: HostQuery,
    },
    SubmitPatch {
        request_id: String,
        patch: HostPatchRequest,
    },
    SubmitJob {
        request: SubmitJobRequest,
    },
    ToolResult {
        call_id: String,
        result: Result<serde_json::Value, String>,
        artifacts: Vec<String>,
    },
    DirectorResult {
        director_id: String,
        decision: serde_json::Value,
    },
    RemoteResult {
        request_id: String,
        result: Result<serde_json::Value, String>,
        artifact_id: Option<String>,
    },
    Heartbeat,
    Error {
        code: String,
        message: String,
    },
}
