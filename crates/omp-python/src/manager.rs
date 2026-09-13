use crate::error::PythonExtensionError;
use crate::protocol::{
    ComponentDeclaration, DirectorDeclaration, HostPatchRequest, HostQuery, LifecycleEvent,
    ToolDeclaration,
};
use omp_types::Patch;
use std::collections::{BTreeMap, BTreeSet};

pub type WorkerTerminator = Box<dyn FnMut() -> Result<(), String> + Send>;

pub struct LoadedExtension {
    pub id: String,
    pub manifest: serde_json::Value,
    pub tools: Vec<ToolDeclaration>,
    pub directors: Vec<DirectorDeclaration>,
    pub components: Vec<ComponentDeclaration>,
    pub active_requests: BTreeSet<String>,
    pub workers: Vec<WorkerTerminator>,
}

impl std::fmt::Debug for LoadedExtension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedExtension")
            .field("id", &self.id)
            .field("manifest", &self.manifest)
            .field("tools", &self.tools)
            .field("directors", &self.directors)
            .field("components", &self.components)
            .field("active_requests", &self.active_requests)
            .field("workers_count", &self.workers.len())
            .finish()
    }
}

pub struct ExtensionManager {
    extensions: BTreeMap<String, LoadedExtension>,
}

impl ExtensionManager {
    pub fn new() -> Self {
        Self {
            extensions: BTreeMap::new(),
        }
    }

    /// Load an extension into the manager with its manifest.
    pub fn load(
        &mut self,
        extension_id: String,
        manifest_json: &str,
    ) -> Result<LifecycleEvent, PythonExtensionError> {
        let manifest: serde_json::Value = serde_json::from_str(manifest_json)
            .map_err(|e| PythonExtensionError::Serialization(e.to_string()))?;
        if self.extensions.contains_key(&extension_id) {
            self.unload(&extension_id)?;
        }

        self.extensions.insert(
            extension_id.clone(),
            LoadedExtension {
                id: extension_id.clone(),
                manifest,
                tools: Vec::new(),
                directors: Vec::new(),
                components: Vec::new(),
                active_requests: BTreeSet::new(),
                workers: Vec::new(),
            },
        );

        Ok(LifecycleEvent::Load {
            extension_id,
            manifest_json: manifest_json.to_string(),
        })
    }

    /// Register an active worker process terminator associated with an extension.
    pub fn register_worker<F>(
        &mut self,
        extension_id: &str,
        terminator: F,
    ) -> Result<(), PythonExtensionError>
    where
        F: FnMut() -> Result<(), String> + Send + 'static,
    {
        let ext = self.extensions.get_mut(extension_id).ok_or_else(|| {
            PythonExtensionError::ExtensionTerminated {
                extension_id: extension_id.to_string(),
                reason: "extension not loaded".into(),
            }
        })?;
        ext.workers.push(Box::new(terminator));
        Ok(())
    }

    /// Register declarative tools, directors, and components declared by the extension.
    pub fn register_declarations(
        &mut self,
        extension_id: &str,
        tools: Vec<ToolDeclaration>,
        directors: Vec<DirectorDeclaration>,
        components: Vec<ComponentDeclaration>,
    ) -> Result<(), PythonExtensionError> {
        let ext = self.extensions.get_mut(extension_id).ok_or_else(|| {
            PythonExtensionError::ExtensionTerminated {
                extension_id: extension_id.to_string(),
                reason: "extension not loaded".into(),
            }
        })?;

        ext.tools = tools;
        ext.directors = directors;
        ext.components = components;
        Ok(())
    }

    /// Unload an extension, invalidating active requests and terminating worker processes.
    pub fn unload(&mut self, extension_id: &str) -> Result<LifecycleEvent, PythonExtensionError> {
        let mut ext = self.extensions.remove(extension_id).ok_or_else(|| {
            PythonExtensionError::ExtensionTerminated {
                extension_id: extension_id.to_string(),
                reason: "extension not loaded".into(),
            }
        })?;

        let mut failures = Vec::new();
        for mut worker in ext.workers.drain(..) {
            if let Err(error) = worker() {
                failures.push(error);
            }
        }
        if !failures.is_empty() {
            return Err(PythonExtensionError::ExtensionTerminated {
                extension_id: extension_id.into(),
                reason: failures.join("; "),
            });
        }

        Ok(LifecycleEvent::Unload {
            extension_id: extension_id.to_string(),
        })
    }

    /// Reload an extension. Terminates active in-flight requests, kills active workers, and resets declarations.
    pub fn reload(&mut self, extension_id: &str) -> Result<LifecycleEvent, PythonExtensionError> {
        let ext = self.extensions.get_mut(extension_id).ok_or_else(|| {
            PythonExtensionError::ExtensionTerminated {
                extension_id: extension_id.to_string(),
                reason: "extension not loaded".into(),
            }
        })?;

        // Forcibly terminate all active worker processes associated with this extension
        let mut failures = Vec::new();
        for mut worker in ext.workers.drain(..) {
            if let Err(error) = worker() {
                failures.push(error);
            }
        }

        // Invalidate active request handles
        ext.active_requests.clear();
        ext.tools.clear();
        ext.directors.clear();
        ext.components.clear();
        if !failures.is_empty() {
            return Err(PythonExtensionError::ExtensionTerminated {
                extension_id: extension_id.into(),
                reason: failures.join("; "),
            });
        }

        Ok(LifecycleEvent::Reload {
            extension_id: extension_id.to_string(),
        })
    }

    /// Track a new in-flight request ID.
    pub fn track_request(
        &mut self,
        extension_id: &str,
        request_id: String,
    ) -> Result<(), PythonExtensionError> {
        let ext = self.extensions.get_mut(extension_id).ok_or_else(|| {
            PythonExtensionError::ExtensionTerminated {
                extension_id: extension_id.to_string(),
                reason: "extension not loaded".into(),
            }
        })?;
        ext.active_requests.insert(request_id);
        Ok(())
    }

    /// Conclude or cancel a tracked request.
    pub fn complete_request(&mut self, extension_id: &str, request_id: &str) {
        if let Some(ext) = self.extensions.get_mut(extension_id) {
            ext.active_requests.remove(request_id);
        }
    }

    /// Check whether a request handle is active and valid.
    pub fn is_request_active(&self, extension_id: &str, request_id: &str) -> bool {
        self.extensions
            .get(extension_id)
            .map(|ext| ext.active_requests.contains(request_id))
            .unwrap_or(false)
    }

    /// Validates an incoming request handle; returns an error if invalidated by reload or unknown.
    pub fn validate_request_handle(
        &self,
        extension_id: &str,
        request_id: &str,
    ) -> Result<(), PythonExtensionError> {
        let ext = self.extensions.get(extension_id).ok_or_else(|| {
            PythonExtensionError::ExtensionTerminated {
                extension_id: extension_id.to_string(),
                reason: "extension not loaded".into(),
            }
        })?;

        if !ext.active_requests.contains(request_id) {
            return Err(PythonExtensionError::ProtocolViolation(format!(
                "request handle '{request_id}' is invalid or was cancelled by reload"
            )));
        }
        Ok(())
    }

    /// Validate a host query submitted by an extension.
    ///
    /// Presence of the extension is required; query arguments are
    /// shape-checked (nonempty names/selectors/ids) so a compromised worker
    /// cannot exfiltrate via unbounded selectors. Unknown query kinds are
    /// rejected explicitly — notably `set_convar` / `execute_command`, which
    /// the Python wrappers historically sent but `HostQuery` cannot represent.
    pub fn validate_query(
        &self,
        extension_id: &str,
        query: &HostQuery,
    ) -> Result<(), PythonExtensionError> {
        if !self.extensions.contains_key(extension_id) {
            return Err(PythonExtensionError::ExtensionTerminated {
                extension_id: extension_id.to_string(),
                reason: "extension not loaded".into(),
            });
        }
        let arg = match query {
            HostQuery::GetConVar { name } => name.as_str(),
            HostQuery::GetSnapshot { .. } => return Ok(()),
            HostQuery::GetArtifact { id } => id.as_str(),
            HostQuery::QueryDOM { selector } => selector.as_str(),
        };
        if arg.trim().is_empty() || arg.len() > 4096 {
            return Err(PythonExtensionError::ProtocolViolation(
                "host query argument must be 1..=4096 characters".into(),
            ));
        }
        Ok(())
    }

    /// Validate a patch request submitted by an extension by delegating to
    /// the journal's `Patch::validate` (reason length, op count, element
    /// kinds, per-op text bounds), stamped with a synthetic offset pair.
    pub fn validate_patch(
        &self,
        extension_id: &str,
        patch: &HostPatchRequest,
    ) -> Result<(), PythonExtensionError> {
        if !self.extensions.contains_key(extension_id) {
            return Err(PythonExtensionError::ExtensionTerminated {
                extension_id: extension_id.to_string(),
                reason: "extension not loaded".into(),
            });
        }

        if patch.ops.is_empty() {
            return Err(PythonExtensionError::ProtocolViolation(
                "patch operations list cannot be empty".into(),
            ));
        }

        // Reuse the authoritative validator with a synthetic 0->1 offset pair
        // (the host re-stamps real offsets at journal-append time).
        let synthetic = Patch {
            base_offset: omp_types::JournalOffset(0),
            result_offset: omp_types::JournalOffset(1),
            by: omp_types::ElementId::new("extension")
                .map(omp_types::PatchAuthor::Element)
                .unwrap_or_else(|_| {
                    omp_types::PatchAuthor::Actor(
                        omp_types::ActorId::new("extension").unwrap_or_else(|_| {
                            // "extension" is charset-valid; mint only if the
                            // impossible happens.
                            omp_types::ActorId::mint()
                        }),
                    )
                }),
            reason: patch.reason.clone(),
            ops: patch.ops.clone(),
        };
        synthetic.validate().map_err(|error| {
            PythonExtensionError::ProtocolViolation(format!(
                "extension patch rejected: {}: {}",
                error.code, error.message
            ))
        })?;
        Ok(())
    }

    /// Look up a tool by name across extensions. Duplicate registrations are
    /// a loud error at lookup time so shadowing can never pass silently.
    pub fn get_tool(&self, name: &str) -> Result<Option<(&str, &ToolDeclaration)>, PythonExtensionError> {
        let mut found = None;
        for (ext_id, ext) in &self.extensions {
            for tool in &ext.tools {
                if tool.name == name {
                    if found.is_some() {
                        return Err(PythonExtensionError::ProtocolViolation(format!(
                            "duplicate tool registration for '{name}'"
                        )));
                    }
                    found = Some((ext_id.as_str(), tool));
                }
            }
        }
        Ok(found)
    }

    pub fn get_director(
        &self,
        name: &str,
    ) -> Result<Option<(&str, &DirectorDeclaration)>, PythonExtensionError> {
        let mut found = None;
        for (ext_id, ext) in &self.extensions {
            for director in &ext.directors {
                if director.name == name {
                    if found.is_some() {
                        return Err(PythonExtensionError::ProtocolViolation(format!(
                            "duplicate director registration for '{name}'"
                        )));
                    }
                    found = Some((ext_id.as_str(), director));
                }
            }
        }
        Ok(found)
    }

    pub fn active_extensions(&self) -> Vec<String> {
        self.extensions.keys().cloned().collect()
    }
}

impl Default for ExtensionManager {
    fn default() -> Self {
        Self::new()
    }
}
