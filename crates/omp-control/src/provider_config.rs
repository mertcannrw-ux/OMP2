use crate::{ConVarStore, ProviderAction, SessionHost};
use omp_inference::{ProviderClient, ProviderMetadata};
use omp_state::Journal;
use omp_types::{
    ElementId, ElementSnapshot, JournalOffset, Patch, PatchOp, PROVIDERS_CONTAINER,
    PROVIDER_MODELS_ATTRIBUTE, ProviderRecord, StructuredError, TypedValue,
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
    /// Reasoning effort levels the active model advertises, as journaled by the
    /// last provider refresh.
    fn advertised_thinking_levels(&self) -> Vec<String> {
        match self.convars.get("ai_thinking_levels") {
            Some(TypedValue::Json(value)) => value
                .as_array()
                .map(|levels| {
                    levels
                        .iter()
                        .filter_map(|level| level.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    /// Effort the provider uses when nothing is requested, if it advertises one.
    fn advertised_thinking_default(&self, snapshot: &omp_state::SessionSnapshot) -> Option<String> {
        snapshot
            .element(snapshot.container("capabilities"))
            .and_then(|node| node.attributes.get("provider_metadata"))
            .and_then(|value| match value {
                TypedValue::Json(metadata) => metadata
                    .pointer("/active_model/thinking_default")
                    .and_then(|value| value.as_str())
                    .map(str::to_string),
                _ => None,
            })
    }

    /// Reports or sets the reasoning effort for the active model.
    ///
    /// The level is checked against what the model advertises *before* it is
    /// written, so a typo is answered at the console instead of failing a turn
    /// with `unsupported_thinking_level`.
    pub(crate) fn execute_effort_command(
        &mut self,
        journal: &mut Journal,
        level: Option<String>,
    ) -> Result<(), StructuredError> {
        let advertised = self.advertised_thinking_levels();
        let default = self.advertised_thinking_default(journal.snapshot());
        let current = self
            .convars
            .get_typed::<String>("ai_thinking")
            .unwrap_or_else(|_| "auto".into());
        let model = self
            .convars
            .get_typed::<String>("ai_model")
            .unwrap_or_default();

        let Some(level) = level else {
            let mut report = format!(
                "Reasoning effort: {}  ·  model: {}",
                if current.trim().is_empty() {
                    "auto"
                } else {
                    current.trim()
                },
                if model.is_empty() {
                    "(none selected)"
                } else {
                    &model
                }
            );
            if advertised.is_empty() {
                report.push_str(
                    "\nThis model advertises no effort levels; only 'auto' and 'off' apply.",
                );
            } else {
                report.push_str(&format!("\nAdvertised levels: {}", advertised.join(", ")));
            }
            if let Some(default) = &default {
                report.push_str(&format!("  ·  provider default: {default}"));
            }
            report.push_str(
                "\nSet with /effort <level>; 'auto' leaves the choice to the provider, 'off' disables reasoning.",
            );
            return self.emit_provider_diagnostic(journal, report);
        };

        let normalized = level.trim().to_ascii_lowercase();
        let value: String = match normalized.as_str() {
            "auto" | "default" => "auto".to_string(),
            "off" | "none" | "0" | "disabled" | "no" => "off".to_string(),
            other => {
                if !advertised
                    .iter()
                    .any(|known| known.eq_ignore_ascii_case(other))
                {
                    return Err(StructuredError::new(
                        "unsupported_thinking_level",
                        format!(
                            "'{level}' is not a reasoning effort this model advertises; available: {}",
                            if advertised.is_empty() {
                                "auto, off".to_string()
                            } else {
                                format!("auto, off, {}", advertised.join(", "))
                            }
                        ),
                        false,
                    ));
                }
                // Use the provider's own spelling so the value round-trips.
                advertised
                    .iter()
                    .find(|known| known.eq_ignore_ascii_case(other))
                    .cloned()
                    .unwrap_or_else(|| other.to_string())
            }
        };
        let value = value.as_str();
        // Writing through the command engine keeps journaling and validation
        // identical to `/ai_thinking <value>`.
        self.execute_command(journal, &format!("ai_thinking {value}"))?;
        let mut report = format!("Reasoning effort set to '{value}'");
        if let Some(default) = &default {
            report.push_str(&format!(" (provider default: {default})"));
        }
        report.push('.');
        self.emit_provider_diagnostic(journal, report)
    }

    /// Journals one diagnostic element, which is how the console reports state.
    fn emit_provider_diagnostic(
        &mut self,
        journal: &mut Journal,
        text: String,
    ) -> Result<(), StructuredError> {
        let mut node = ElementSnapshot::new(ElementId::mint(), "diagnostic");
        node.text = text;
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason: "provider diagnostic".into(),
                ops: vec![PatchOp::Create {
                    parent: journal.snapshot().container("body").clone(),
                    index: journal.snapshot().get_visible_body().count() as u32,
                    element: node,
                }],
            })
            .map_err(|error| error.structured())
            .map(|_| ())
    }

    /// Every provider the session has been told about, in insertion order.
    pub fn providers(&self, snapshot: &omp_state::SessionSnapshot) -> Vec<ProviderRecord> {
        snapshot
            .children(snapshot.container(PROVIDERS_CONTAINER))
            .filter_map(ProviderRecord::from_element)
            .collect()
    }

    /// Models cached for one registered provider, as last fetched.
    pub fn provider_models(
        &self,
        snapshot: &omp_state::SessionSnapshot,
        record: &ProviderRecord,
    ) -> Vec<serde_json::Value> {
        snapshot
            .children(snapshot.container(PROVIDERS_CONTAINER))
            .find(|element| {
                ProviderRecord::from_element(element)
                    .is_some_and(|existing| existing.name == record.name)
            })
            .and_then(|element| element.attributes.get(PROVIDER_MODELS_ATTRIBUTE))
            .and_then(|value| match value {
                TypedValue::Json(models) => models.as_array().cloned(),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// Registers a provider, replacing any record with the same name.
    pub fn record_provider(
        &mut self,
        journal: &mut Journal,
        record: &ProviderRecord,
        models: Option<serde_json::Value>,
    ) -> Result<(), StructuredError> {
        let container = journal.snapshot().container(PROVIDERS_CONTAINER).clone();
        let existing = journal
            .snapshot()
            .children(&container)
            .find(|element| {
                ProviderRecord::from_element(element)
                    .is_some_and(|current| current.name == record.name)
            })
            .map(|element| element.id.clone());
        let mut ops = match existing.clone() {
            Some(id) => ["name", "adapter", "endpoint", "key_env", "model"]
                .into_iter()
                .map(|attribute| {
                    let value = match attribute {
                        "name" => record.name.clone(),
                        "adapter" => record.adapter.clone(),
                        "endpoint" => record.endpoint.clone(),
                        "key_env" => record.key_env.clone(),
                        _ => record.model.clone(),
                    };
                    PatchOp::SetAttribute {
                        element: id.clone(),
                        name: attribute.into(),
                        value: TypedValue::String(value),
                    }
                })
                .collect::<Vec<_>>(),
            None => {
                let id = ElementId::new(format!("provider-{}", record.name)).map_err(|_| {
                    StructuredError::new(
                        "invalid_provider_name",
                        "Provider names use letters, digits, dots, underscores or hyphens",
                        false,
                    )
                })?;
                vec![PatchOp::Create {
                    parent: container.clone(),
                    index: journal.snapshot().children(&container).count() as u32,
                    element: record.to_element(id),
                }]
            }
        };
        // Two records for one endpoint would both look active; the name just
        // declared wins, and the record it replaces is forgotten.
        let replaced: Vec<ElementId> = journal
            .snapshot()
            .children(&container)
            .filter(|element| {
                ProviderRecord::from_element(element).is_some_and(|existing| {
                    existing.name != record.name && existing.matches_endpoint(&record.adapter, &record.endpoint)
                })
            })
            .map(|element| element.id.clone())
            .collect();
        ops.extend(replaced.into_iter().map(|element| PatchOp::Delete { element }));
        if let Some(models) = models {
            let id = existing.unwrap_or_else(|| {
                ElementId::new(format!("provider-{}", record.name)).expect("validated name")
            });
            ops.push(PatchOp::SetAttribute {
                element: id,
                name: PROVIDER_MODELS_ATTRIBUTE.into(),
                value: TypedValue::Json(models),
            });
        }
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason: format!("register provider {}", record.name),
                ops,
            })
            .map_err(|error| error.structured())
            .map(|_| ())
    }

    /// Forgets a registered provider.
    pub fn forget_provider(
        &mut self,
        journal: &mut Journal,
        name: &str,
    ) -> Result<bool, StructuredError> {
        let container = journal.snapshot().container(PROVIDERS_CONTAINER).clone();
        let Some(id) = journal
            .snapshot()
            .children(&container)
            .find(|element| {
                ProviderRecord::from_element(element).is_some_and(|record| record.name == name)
            })
            .map(|element| element.id.clone())
        else {
            return Ok(false);
        };
        journal
            .append_patch(Patch {
                base_offset: JournalOffset(journal.snapshot().offset),
                result_offset: journal.next_offset(),
                by: self.owner.clone().into(),
                reason: format!("forget provider {name}"),
                ops: vec![PatchOp::Delete { element: id }],
            })
            .map_err(|error| error.structured())?;
        Ok(true)
    }

    /// Client for one registered provider, without making it active.
    fn client_for_record(&self, record: &ProviderRecord) -> Result<ProviderClient, StructuredError> {
        let adapter = if record.adapter.is_empty() {
            "openai_compatible".to_string()
        } else {
            record.adapter.clone()
        };
        let mut client = ProviderClient::new(
            adapter,
            record.model.clone(),
            (!record.endpoint.is_empty()).then(|| record.endpoint.clone()),
        )?;
        if !record.key_env.is_empty() {
            let key = std::env::var(&record.key_env)
                .ok()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    StructuredError::new(
                        "missing_credentials",
                        format!(
                            "Environment variable {} is empty or absent; set it before using provider '{}'",
                            record.key_env, record.name
                        ),
                        false,
                    )
                })?;
            client = client.with_api_key(key);
        }
        Ok(client)
    }

    /// Fetches one provider's catalog and caches it on its record.
    fn refresh_registered_provider(
        &mut self,
        journal: &mut Journal,
        record: &ProviderRecord,
    ) -> Result<usize, StructuredError> {
        let mut client = self.client_for_record(record)?;
        let metadata = client.refresh_metadata()?;
        let models = serde_json::to_value(&metadata.models).map_err(|error| {
            StructuredError::new("provider_registry", error.to_string(), false)
        })?;
        let count = metadata.models.len();
        self.record_provider(journal, record, Some(models))?;
        Ok(count)
    }

    /// Human-readable provider inventory: what is configured, what is active,
    /// and how many models each one has.
    pub fn provider_report(&self, snapshot: &omp_state::SessionSnapshot) -> String {
        let records = self.providers(snapshot);
        let adapter = self.convars.get_typed::<String>("ai_provider").unwrap_or_default();
        let endpoint = self.convars.get_typed::<String>("ai_endpoint").unwrap_or_default();
        let active = omp_types::label_for_endpoint(&records, &adapter, &endpoint);
        let mut out = if records.is_empty() {
            format!(
                "No providers recorded yet; active endpoint: {} ({}). Add one: /provider add <name> <endpoint> [--key-env ENV]",
                if endpoint.is_empty() { "(none)" } else { &endpoint },
                active
            )
        } else {
            format!(
                "{} provider(s) recorded; active: {active} ({})",
                records.len(),
                if endpoint.is_empty() { "no endpoint" } else { &endpoint }
            )
        };
        for record in &records {
            let models = self.provider_models(snapshot, record).len();
            let registered = record.matches_endpoint(&adapter, &endpoint);
            out.push_str(&format!(
                "\n  {} {} · {} · {}{}{}{}",
                if registered { '*' } else { '-' },
                record.name,
                if record.adapter.is_empty() { "openai_compatible" } else { &record.adapter },
                record.endpoint,
                if models == 0 { String::new() } else { format!(" · {models} models") },
                if record.model.is_empty() { String::new() } else { format!(" · model {}", record.model) },
                if record.key_env.is_empty() { String::new() } else { format!(" · key ${}", record.key_env) },
            ));
        }
        out.push_str(
            "\nSwitch: /provider use <name>. Add: /provider add <name> <endpoint> [--key-env ENV]. Forget: /provider remove <name>. Catalogs: /provider refresh.",
        );
        out
    }

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
        // A stream that produces nothing for this long is treated as stalled;
        // reasoning models can pause for minutes, so it is a session setting.
        if let Ok(timeout) = store.get_typed::<i64>("ai_request_timeout_secs")
            && timeout > 0
        {
            client = client.with_timeout(timeout as u64);
        }
        let key_env = store
            .get_typed::<String>("ai_api_key_env")
            .unwrap_or_default();
        if !key_env.is_empty() {
            if !crate::command::is_allowed_key_env(&key_env) {
                return Err(StructuredError::new(
                    "invalid_key_reference",
                    "Use a known host API key environment variable (OMP_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENAI_COMPATIBLE_API_KEY)",
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
                            format!(
                                "Environment variable {key_env} is empty or absent; set it before starting omp2 (config files store the variable name, never the key)"
                            ),
                            false,
                        )
                    })?,
            );
        }
        Ok(client)
    }

    // Rebuild transient state from the branch, without fetching on every turn/command.
    pub(crate) fn refresh_provider(&mut self) -> Result<(), StructuredError> {
        let mut client = match self.provider_from_config(&self.convars) {
            Ok(client) => client,
            // No credential means nothing to refresh here; the failure belongs
            // to the turn that needs the provider, not to starting a session.
            Err(error) if Self::is_missing_credential(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
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
    /// Fetches catalogs for registered providers. `only_empty` keeps startup
    /// cheap: a provider whose catalog is already cached waits for an explicit
    /// refresh.
    pub(crate) fn refresh_registered_catalogs(
        &mut self,
        journal: &mut Journal,
        only_empty: bool,
    ) -> Result<(), StructuredError> {
        let records = self.providers(journal.snapshot());
        let active_endpoint = self
            .convars
            .get_typed::<String>("ai_endpoint")
            .unwrap_or_default();
        let mut failures = Vec::new();
        let mut refreshed = 0usize;
        for record in records {
            if omp_types::normalize_endpoint(&record.endpoint)
                == omp_types::normalize_endpoint(&active_endpoint)
            {
                continue;
            }
            if only_empty && !self.provider_models(journal.snapshot(), &record).is_empty() {
                continue;
            }
            match self.refresh_registered_provider(journal, &record) {
                Ok(_) => refreshed += 1,
                Err(error) => failures.push(format!("{}: {}", record.name, error.message)),
            }
        }
        if !failures.is_empty() {
            self.emit_provider_diagnostic(
                journal,
                format!(
                    "Some provider catalogs could not be fetched: {}",
                    failures.join("; ")
                ),
            )?;
        } else if refreshed > 0 {
            self.emit_provider_diagnostic(
                journal,
                format!("Fetched {refreshed} registered provider catalog(s)."),
            )?;
        }
        Ok(())
    }

    /// True when the only thing wrong with the provider configuration is a
    /// credential that has not been exported yet.
    ///
    /// Configuring a provider must not require its key to be present: the key
    /// lives in the host environment and can be set afterwards, and the session
    /// stays usable meanwhile. Work that actually needs the provider still fails.
    pub fn is_missing_credential(error: &StructuredError) -> bool {
        error.code == "missing_credentials"
    }

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
            // No active provider is configured, but declared providers still
            // deserve their catalogs.
            self.refresh_registered_catalogs(journal, true)?;
            return Ok(());
        }
        let key_env = self
            .convars
            .get_typed::<String>("ai_api_key_env")
            .unwrap_or_default();
        let outcome = match Self::fetch_selection(&mut client, false) {
            Ok(meta) => self.commit_provider(journal, client, &key_env, Some(meta), None),
            Err(error) => {
                self.commit_provider(journal, client, &key_env, None, Some(&error))?;
                Err(error)
            }
        };
        // Providers a profile or config file declared have no catalog yet; fetch
        // those now so the model list is complete from the first prompt.
        let _ = self.refresh_registered_catalogs(journal, true);
        outcome
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
                self.emit_provider_diagnostic(journal, text)
            }
            ProviderAction::List => {
                let report = self.provider_report(journal.snapshot());
                self.emit_provider_diagnostic(journal, report)
            }
            ProviderAction::Add {
                name,
                endpoint,
                adapter,
                key_env,
                model,
            } => {
                let record = ProviderRecord {
                    name: name.clone(),
                    adapter,
                    endpoint: endpoint.clone(),
                    key_env: key_env.clone(),
                    model: model.clone().unwrap_or_default(),
                };
                // The name doubles as a durable element id; reject the ones that
                // cannot be one before anything is written.
                ElementId::new(format!("provider-{name}")).map_err(|_| {
                    StructuredError::new(
                        "invalid_provider_name",
                        "Provider names use 1..128 letters, digits, dots, underscores or hyphens",
                        false,
                    )
                })?;
                self.record_provider(journal, &record, None)?;
                // Registration is pure configuration: no fetch happens here, so a
                // profile or user config may declare providers. Catalogs are
                // fetched at startup and by /provider refresh.
                self.emit_provider_diagnostic(
                    journal,
                    format!(
                        "Registered provider '{name}' ({endpoint}). Its catalog is fetched at startup; /provider refresh fetches now. Switch with /provider use {name}."
                    ),
                )
            }
            ProviderAction::Use { name } => {
                let Some(record) = self
                    .providers(journal.snapshot())
                    .into_iter()
                    .find(|record| record.name == name)
                else {
                    let known: Vec<String> = self
                        .providers(journal.snapshot())
                        .into_iter()
                        .map(|record| record.name)
                        .collect();
                    return Err(StructuredError::new(
                        "provider_not_registered",
                        format!(
                            "No provider named '{name}'; configured: {}",
                            if known.is_empty() { "(none)".to_string() } else { known.join(", ") }
                        ),
                        false,
                    ));
                };
                let mut staged = self.convars.clone();
                for (attribute, value) in [
                    ("ai_provider", record.adapter.clone()),
                    ("ai_endpoint", record.endpoint.clone()),
                    ("ai_api_key_env", record.key_env.clone()),
                    ("ai_model", record.model.clone()),
                    ("ai_provider_name", record.name.clone()),
                ] {
                    staged.set_from_str(attribute, &value).map_err(|error| {
                        StructuredError::new("provider_config", error.to_string(), false)
                    })?;
                }
                let mut client = self.provider_from_config(&staged)?;
                client.model = record.model.clone();
                let meta = Self::fetch_selection(&mut client, !record.model.is_empty())?;
                self.commit_provider(journal, client, &record.key_env, Some(meta.clone()), None)?;
                if let Ok(models) = serde_json::to_value(&meta.models) {
                    self.record_provider(journal, &record, Some(models))?;
                }
                self.emit_provider_diagnostic(
                    journal,
                    format!(
                        "Active provider is now '{name}' ({}); {} models advertised.",
                        record.endpoint,
                        meta.models.len()
                    ),
                )
            }
            ProviderAction::Remove { name } => {
                let existed = self.forget_provider(journal, &name)?;
                let active = self.convars.get_typed::<String>("ai_endpoint").unwrap_or_default();
                let text = if existed {
                    format!(
                        "Forgot provider '{name}'. The active settings still point at {active} until /provider use <name> picks another."
                    )
                } else {
                    format!("No provider named '{name}' was recorded.")
                };
                self.emit_provider_diagnostic(journal, text)
            }
            ProviderAction::Refresh => {
                self.initialize_provider(journal)?;
                // Every recorded provider gets a fresh catalog, so the annotated
                // model list never shows a stale count for an inactive gateway.
                self.refresh_registered_catalogs(journal, false)
            }
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

    /// Records the active provider when nothing registered covers it yet, so a
    /// provider configured through the environment or a profile still appears in
    /// `/provider list` and can be switched back to by name.
    fn record_active_provider(
        &mut self,
        journal: &mut Journal,
        client: &ProviderClient,
        key_env: &str,
        metadata: Option<&ProviderMetadata>,
    ) -> Result<(), StructuredError> {
        let Some(endpoint) = client.endpoint.clone() else {
            return Ok(());
        };
        let declared = self
            .convars
            .get_typed::<String>("ai_provider_name")
            .unwrap_or_default();
        let existing = self.providers(journal.snapshot());
        let already = existing
            .iter()
            .find(|record| record.matches_endpoint(&client.provider, &endpoint));
        if let Some(record) = already {
            // Keep the user's name; refresh the catalog the list shows.
            if let Some(models) = metadata.and_then(|meta| serde_json::to_value(&meta.models).ok()) {
                self.record_provider(journal, &record.clone(), Some(models))?;
            }
            return Ok(());
        }
        let name = if declared.trim().is_empty() {
            omp_types::endpoint_host(&endpoint)
        } else {
            declared.trim().to_string()
        };
        if name.is_empty() {
            return Ok(());
        }
        let record = ProviderRecord {
            name,
            adapter: client.provider.clone(),
            endpoint,
            key_env: key_env.to_string(),
            model: client.model.clone(),
        };
        let models = metadata.and_then(|meta| serde_json::to_value(&meta.models).ok());
        self.record_provider(journal, &record, models)
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
        let registered = self.record_active_provider(
            journal,
            &client,
            key_env,
            metadata.as_ref(),
        );
        self.provider = client;
        self.provider_injected = false;
        // A registry failure must never invalidate a working provider.
        let _ = registered;
        Ok(())
    }
}
