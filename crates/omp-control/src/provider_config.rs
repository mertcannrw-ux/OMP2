use crate::{ConVarStore, ProviderAction, SessionHost};
use omp_inference::{ProviderClient, ProviderMetadata};
use omp_state::Journal;
use omp_types::{
    ElementId, ElementSnapshot, JournalOffset, Patch, PatchOp, StructuredError, TypedValue,
};

fn setting(store: &ConVarStore, name: &str, envs: &[&str]) -> String {
    let value = store.get_typed::<String>(name).unwrap_or_default();
    if !value.trim().is_empty() && value != "default" {
        return value;
    }
    envs.iter()
        .find_map(|name| std::env::var(name).ok().filter(|v| !v.trim().is_empty()))
        .unwrap_or_default()
}

/// Bounded one-screen summary of provider catalog metadata: model count,
/// selected model, and the first few model ids. Never dumps the full JSON.
fn summarize_provider_metadata(meta: &serde_json::Value) -> String {
    const MAX_LISTED: usize = 8;
    let models = meta
        .get("models")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let selected = meta
        .get("selected")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .or_else(|| {
            // `active_model` is a `ModelMetadata` object, not a string.
            meta.get("active_model").and_then(|v| {
                v.as_str().map(str::to_owned).or_else(|| {
                    v.get("id")
                        .or_else(|| v.get("name"))
                        .and_then(|id| id.as_str())
                        .map(str::to_owned)
                })
            })
        })
        .unwrap_or("(none)".into());
    let mut out = format!("{} models advertised; selected: {selected}", models.len());
    for model in models.iter().take(MAX_LISTED) {
        let id = model
            .get("id")
            .or_else(|| model.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        out.push_str(&format!("\n  - {id}"));
    }
    if models.len() > MAX_LISTED {
        out.push_str(&format!("\n  ... and {} more", models.len() - MAX_LISTED));
    }
    out
}

impl SessionHost {
    pub(crate) fn provider_from_config(
        &self,
        store: &ConVarStore,
    ) -> Result<ProviderClient, StructuredError> {
        let endpoint = setting(
            store,
            "ai_endpoint",
            &["OMP_ENDPOINT", "AI_ENDPOINT", "OPENAI_BASE_URL"],
        );
        let mut adapter = setting(store, "ai_provider", &["AI_PROVIDER", "OMP_PROVIDER"]);
        let model = setting(store, "ai_model", &["AI_MODEL", "OMP_MODEL"]);
        if endpoint.is_empty() && adapter.is_empty() && model.is_empty() {
            return Ok(ProviderClient::unconfigured());
        }
        if adapter.is_empty() {
            adapter = if endpoint.starts_with("https://api.anthropic.com/") {
                "anthropic"
            } else {
                "openai_compatible"
            }
            .into();
        }
        let mut client =
            ProviderClient::new(adapter, model, (!endpoint.is_empty()).then_some(endpoint))?;
        let key_env = store
            .get_typed::<String>("ai_api_key_env")
            .unwrap_or_default();
        if !key_env.is_empty() {
            if !key_env
                .bytes()
                .enumerate()
                .all(|(i, b)| b == b'_' || b.is_ascii_alphabetic() || (i > 0 && b.is_ascii_digit()))
            {
                return Err(StructuredError::new(
                    "invalid_key_reference",
                    "Use a host environment variable name, not an API key",
                    false,
                ));
            }
            client.api_key = Some(
                std::env::var(&key_env)
                    .ok()
                    .filter(|key| !key.is_empty())
                    .ok_or_else(|| {
                        StructuredError::new(
                            "missing_credentials",
                            "The configured API key environment variable is empty or absent",
                            false,
                        )
                    })?,
            );
        }
        Ok(client)
    }

    // Rebuild transient state from the branch, without fetching on every turn/command.
    pub(crate) fn refresh_provider(&mut self) -> Result<(), StructuredError> {
        let mut client = self.provider_from_config(&self.convars)?;
        if self.provider_injected {
            // Explicitly injected clients win until journal config takes over
            // (commit_provider clears the flag). An env- or convar-derived
            // client must never silently replace an injected one — even when
            // it is non-empty, since it may lack a model.
            return Ok(());
        }
        if client.provider == self.provider.provider
            && client.endpoint == self.provider.endpoint
            && client.model == self.provider.model
        {
            client.metadata = self.provider.metadata.clone();
            if let Some(metadata) = client.active_model_metadata().cloned() {
                client.apply_model_metadata(&metadata);
            }
            client.caps = self.provider.caps.clone();
            client.tool_force = self.provider.tool_force.clone();
        }
        self.provider = client;
        Ok(())
    }

    pub(crate) fn restore_provider_metadata(
        &mut self,
        snapshot: &omp_state::SessionSnapshot,
    ) -> Result<(), StructuredError> {
        if self.provider_injected {
            return Ok(());
        }
        self.provider.metadata = None;
        if let Some(TypedValue::Json(value)) = snapshot
            .element(snapshot.container("capabilities"))
            .and_then(|node| node.attributes.get("provider_metadata"))
        {
            let metadata: ProviderMetadata =
                serde_json::from_value(value.clone()).map_err(|_| {
                    StructuredError::new(
                        "invalid_provider_metadata",
                        "Journal provider metadata is invalid",
                        false,
                    )
                })?;
            if metadata.endpoint == self.provider.endpoint
                && metadata.provider == self.provider.provider
            {
                if let Some(model) = metadata
                    .models
                    .iter()
                    .find(|model| model.id == self.provider.model)
                {
                    self.provider.apply_model_metadata(model);
                }
                self.provider.metadata = Some(metadata);
            }
        }
        Ok(())
    }
    /// Refresh the provider catalog once at each run/resume/serve startup.
    /// Failure clears advertised limits rather than silently reusing stale metadata.
    pub fn initialize_provider(&mut self, journal: &mut Journal) -> Result<(), StructuredError> {
        self.convars.hydrate_from_dom(journal.snapshot());
        let mut client = match self.provider_from_config(&self.convars) {
            Ok(client) => client,
            Err(error) => {
                let mut unavailable = ProviderClient::unconfigured();
                unavailable.model = self
                    .convars
                    .get_typed::<String>("ai_model")
                    .unwrap_or_default();
                unavailable.provider = self
                    .convars
                    .get_typed::<String>("ai_provider")
                    .unwrap_or_default();
                unavailable.endpoint = self
                    .convars
                    .get_typed::<String>("ai_endpoint")
                    .ok()
                    .filter(|v| !v.is_empty());
                let key_env = self
                    .convars
                    .get_typed::<String>("ai_api_key_env")
                    .unwrap_or_default();
                self.commit_provider(journal, unavailable, &key_env, None, Some(&error))?;
                return Err(error);
            }
        };
        if client.provider.is_empty() {
            self.provider = client;
            return Ok(());
        }
        let key_env = self
            .convars
            .get_typed::<String>("ai_api_key_env")
            .unwrap_or_default();
        match Self::fetch_selection(&mut client, false) {
            Ok(meta) => self.commit_provider(journal, client, &key_env, Some(meta), None),
            Err(error) => {
                self.commit_provider(journal, client, &key_env, None, Some(&error))?;
                Err(error)
            }
        }
    }

    fn fetch_selection(
        client: &mut ProviderClient,
        explicit: bool,
    ) -> Result<ProviderMetadata, StructuredError> {
        let mut meta = client.refresh_metadata()?;
        meta.models.sort_by(|a, b| a.id.cmp(&b.id));
        let selected = meta.models.iter().find(|m| m.id == client.model);
        if explicit && selected.is_none() {
            return Err(StructuredError::new(
                "model_not_available",
                "The requested model is absent from the provider catalog",
                false,
            ));
        }
        let selected = selected
            .or_else(|| meta.models.first())
            .ok_or_else(|| {
                StructuredError::new(
                    "empty_model_catalog",
                    "Provider returned no available models",
                    false,
                )
            })?
            .clone();
        client.apply_model_metadata(&selected);
        meta.active_model = Some(selected);
        client.metadata = Some(meta.clone());
        Ok(meta)
    }

    pub(crate) fn execute_provider_command(
        &mut self,
        journal: &mut Journal,
        action: ProviderAction,
    ) -> Result<(), StructuredError> {
        match action {
            ProviderAction::Show => {
                let caps = journal
                    .snapshot()
                    .element(journal.snapshot().container("capabilities"));
                let metadata = caps.and_then(|node| node.attributes.get("provider_metadata"));
                let text = match metadata {
                    Some(TypedValue::Json(meta)) => {
                        // Summarize instead of pretty-printing the full
                        // catalog: unbounded model lists would bloat the
                        // journal on every `/provider show`.
                        let summary = summarize_provider_metadata(meta);
                        format!("{summary}\nModel selection: /provider select <id>. Refresh: /provider refresh. Null metadata means the provider did not advertise it.")
                    }
                    _ => "No provider catalog. Set OMP_API_KEY in the host environment, then /provider <endpoint>. Models and advertised settings refresh on every startup. Never paste API keys into commands.".into(),
                };
                let mut node = ElementSnapshot::new(ElementId::mint(), "diagnostic");
                node.text = text;
                journal
                    .append_patch(Patch {
                        base_offset: JournalOffset(journal.snapshot().offset),
                        result_offset: journal.next_offset(),
                        by: self.owner.clone().into(),
                        reason: "show provider configuration".into(),
                        ops: vec![PatchOp::Create {
                            parent: journal.snapshot().container("body").clone(),
                            index: journal.snapshot().get_visible_body().count() as u32,
                            element: node,
                        }],
                    })
                    .map_err(|e| e.structured())?;
                Ok(())
            }
            ProviderAction::Refresh => self.initialize_provider(journal),
            ProviderAction::Configure {
                endpoint,
                adapter,
                key_env,
                model,
            } => {
                let mut staged = self.convars.clone();
                for (name, value) in [
                    ("ai_endpoint", endpoint),
                    ("ai_provider", adapter),
                    ("ai_api_key_env", key_env.clone()),
                ] {
                    staged.set_from_str(name, &value).map_err(|e| {
                        StructuredError::new("provider_config", e.to_string(), false)
                    })?;
                }
                staged
                    .set_from_str("ai_model", model.as_deref().unwrap_or(""))
                    .map_err(|e| StructuredError::new("provider_config", e.to_string(), false))?;
                let mut client = self.provider_from_config(&staged)?;
                // Explicit endpoint-only setup must not inherit an unrelated AI_MODEL environment override.
                client.model = model.clone().unwrap_or_default();
                let meta = Self::fetch_selection(&mut client, model.is_some())?;
                self.commit_provider(journal, client, &key_env, Some(meta), None)
            }
            ProviderAction::Select { model } => {
                let mut client = self.provider_from_config(&self.convars)?;
                client.model = model;
                let meta = Self::fetch_selection(&mut client, true)?;
                let key_env = self
                    .convars
                    .get_typed::<String>("ai_api_key_env")
                    .unwrap_or_default();
                self.commit_provider(journal, client, &key_env, Some(meta), None)
            }
        }
    }

    fn commit_provider(
        &mut self,
        journal: &mut Journal,
        client: ProviderClient,
        key_env: &str,
        metadata: Option<ProviderMetadata>,
        error: Option<&StructuredError>,
    ) -> Result<(), StructuredError> {
        let selected = metadata
            .as_ref()
            .and_then(|meta| meta.active_model.as_ref());
        let number = |value: Option<usize>| {
            value
                .and_then(|v| i64::try_from(v).ok())
                .map(TypedValue::Integer)
                .unwrap_or(TypedValue::Null)
        };
        let mut ops = Vec::new();
        let convars = journal.snapshot().container("convars").clone();
        for (name, value) in [
            ("ai_provider", TypedValue::String(client.provider.clone())),
            ("ai_model", TypedValue::String(client.model.clone())),
            (
                "ai_endpoint",
                TypedValue::String(client.endpoint.clone().unwrap_or_default()),
            ),
            ("ai_api_key_env", TypedValue::String(key_env.into())),
            (
                "ai_context_length",
                number(selected.and_then(|m| m.context_length)),
            ),
            (
                "ai_max_tokens",
                number(selected.and_then(|m| m.max_output_tokens)),
            ),
            (
                "ai_thinking",
                TypedValue::String(
                    selected
                        .and_then(|m| m.thinking_default.clone())
                        .unwrap_or_else(|| "auto".into()),
                ),
            ),
            (
                "ai_thinking_levels",
                selected
                    .and_then(|m| m.thinking_levels.as_ref())
                    .map(|levels| TypedValue::Json(serde_json::json!(levels)))
                    .unwrap_or(TypedValue::Null),
            ),
        ] {
            ops.push(PatchOp::SetAttribute {
                element: convars.clone(),
                name: name.into(),
                value,
            });
        }
        let capabilities = journal.snapshot().container("capabilities").clone();
        ops.push(PatchOp::SetAttribute {
            element: capabilities.clone(),
            name: "provider_metadata".into(),
            value: metadata
                .as_ref()
                .map(|m| TypedValue::Json(serde_json::to_value(m).unwrap()))
                .unwrap_or(TypedValue::Null),
        });
        ops.push(PatchOp::SetAttribute {
            element: capabilities,
            name: "provider_refresh_error".into(),
            value: error
                .map(|e| TypedValue::Json(serde_json::to_value(e).unwrap()))
                .unwrap_or(TypedValue::Null),
        });
        let mut node = ElementSnapshot::new(ElementId::mint(), "diagnostic");
        node.text = if let Some(error) = error {
            format!(
                "Provider catalog refresh failed ({}). Advertised settings cleared; /provider refresh retries.",
                error.code
            )
        } else {
            match (metadata.as_ref(), selected) {
                (Some(meta), Some(m)) => {
                    let context_desc = match (&m.context_length, &m.context_provenance) {
                        (Some(len), Some(prov)) => format!("{len} ({prov})"),
                        (Some(len), None) => len.to_string(),
                        (None, _) => "unknown".into(),
                    };
                    format!(
                        "Provider catalog refreshed: {} models. Selected {}. Context: {}; max output: {}; thinking levels: {}. Missing fields remain unknown. /provider lists models; /provider select <id> changes selection.",
                        meta.models.len(),
                        m.id,
                        context_desc,
                        m.max_output_tokens
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "unknown".into()),
                        m.thinking_levels
                            .as_ref()
                            .map(|v| v.join(", "))
                            .unwrap_or_else(|| "unknown".into())
                    )
                }
                // (None, None) or partial metadata without an error: never unwrap,
                // report the degraded state instead.
                _ => "Provider configuration updated without catalog metadata; run /provider refresh to discover models."
                    .to_string(),
            }
        };
        ops.push(PatchOp::Create {
            parent: journal.snapshot().container("body").clone(),
            index: journal.snapshot().get_visible_body().count() as u32,
            element: node,
        });
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason: "refresh provider catalog and effective settings".into(),
                ops,
            })
            .map_err(|e| e.structured())?;
        self.convars.hydrate_from_dom(journal.snapshot());
        self.provider = client;
        self.provider_injected = false;
        Ok(())
    }
}
