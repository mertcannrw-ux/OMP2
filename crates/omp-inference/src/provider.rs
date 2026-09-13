use crate::capability::{CapabilityProfile, ProviderCapability, TriState};
use crate::compat::CompatTable;
use crate::corrective;
use crate::model_taxonomy::{ModelIdentifier, ProviderRoute};
use crate::repair;
use crate::request::{
    InferenceRequest, ProviderRequest, ToolCallSpec, WireFormat, translate_request,
};
use crate::tool_force::ToolForcePolicy;
use omp_types::{StructuredError, ToolCallId};
use serde::de::DeserializeSeed;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read};
use std::net::{Ipv4Addr, Ipv6Addr};
use std::time::Duration;
/// Streaming delta emitted incrementally during inference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InferenceDelta<'a> {
    Text(&'a str),
    Thinking(&'a str),
}

/// Canonical turn produced by the inference layer independent of provider wire details.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanonicalTurn {
    pub text: String,
    pub thinking: Option<String>,
    pub tool_calls: Vec<ToolCallSpec>,
    pub usage: serde_json::Value,
    pub finish_reason: String,
}

/// Accumulator for streaming tool call deltas across SSE chunks.
#[derive(Clone, Debug, Default)]
struct StreamingToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments_buffer: String,
}
/// Advertised metadata for a specific model returned by a provider's catalog.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub thinking_supported: Option<bool>,
    pub thinking_levels: Option<Vec<String>>,
    pub thinking_default: Option<String>,
    pub context_length: Option<usize>,
    pub max_output_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_provenance: Option<String>,
}

impl ModelMetadata {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            thinking_supported: None,
            thinking_levels: None,
            thinking_default: None,
            context_length: None,
            max_output_tokens: None,
            context_provenance: None,
        }
    }
}

/// Advertised metadata for a provider and its catalog of models.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProviderMetadata {
    pub provider: String,
    pub endpoint: Option<String>,
    pub models: Vec<ModelMetadata>,
    pub active_model: Option<ModelMetadata>,
    pub refreshed_at_unix_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<String>,
}

/// The client identifies itself to providers: a generic HTTP-library user
/// agent gets refused by gateways that route agent traffic (OpenCode Go asks
/// clients to name themselves for exactly this reason).
pub const CLIENT_USER_AGENT: &str = "omp2/0.1.0";

/// Hosts that require a stable per-conversation id on inference requests.
///
/// OpenCode Go rejects a request without `x-opencode-session` outright
/// (HTTP 400 `MissingSessionID`) and uses the value for routing and prompt
/// caching, so the header carries the session id the journal already assigns.
const CONVERSATION_HEADER_HOSTS: &[&str] = &["opencode.ai"];
/// Header carrying the conversation id for [`CONVERSATION_HEADER_HOSTS`].
const CONVERSATION_HEADER: &str = "x-opencode-session";

/// Idle budget for a provider request: the time a stream may produce nothing
/// before it is treated as stalled. Reasoning models routinely pause for
/// minutes between tokens on a hard prompt, and a gateway may hold the stream
/// while it queues, so this is minutes rather than seconds.
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 300;
/// Fast failure for an unreachable endpoint: connecting is never slow.
pub const CONNECT_TIMEOUT_SECS: u64 = 15;
pub const MIN_REQUEST_TIMEOUT_SECS: u64 = 10;
pub const MAX_REQUEST_TIMEOUT_SECS: u64 = 1800;

/// Shared provider client contract for semantic request to canonical turn.
#[derive(Clone, Debug)]
pub struct ProviderClient {
    pub provider: String,
    pub model: String,
    pub ident: ModelIdentifier,
    pub caps: CapabilityProfile,
    pub api_key: Option<String>,
    pub endpoint: Option<String>,
    pub timeout_secs: u64,
    pub max_response_bytes: usize,
    pub tool_force: Option<ToolForcePolicy>,
    pub attempt_count: u32,
    pub metadata: Option<ProviderMetadata>,
    /// Stable per-conversation id sent to hosts that require one. Set from the
    /// session the client serves, so replicas and subagents stay distinct.
    pub conversation_id: Option<String>,
}

impl ProviderClient {
    /// Returns an unconfigured client placeholder.
    pub fn unconfigured() -> Self {
        Self {
            provider: String::new(),
            model: String::new(),
            ident: ModelIdentifier::infer("unknown", ProviderRoute::OpenAiDirect),
            caps: CapabilityProfile::default(),
            api_key: None,
            endpoint: None,
            timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
            max_response_bytes: 10 * 1024 * 1024,
            tool_force: None,
            attempt_count: 0,
            metadata: None,
            conversation_id: None,
        }
    }

    /// Checks if provider and model have been explicitly configured.
    pub fn is_configured(&self) -> bool {
        !self.provider.is_empty()
            && !self.model.is_empty()
            && self.provider != "default"
            && self.model != "default"
    }
    /// Creates a provider client resolving model and provider from environment and explicit parameters.
    /// Pure constructor with no HTTP operations.
    /// Allows endpoint-only configuration with no model for catalog lookup; infer errors if none.
    /// Default adapter when only endpoint given is openai_compatible.
    pub fn from_env(provider: &str, model: &str) -> Result<Self, StructuredError> {
        let p_trimmed = provider.trim();
        let m_trimmed = model.trim();

        let endpoint = std::env::var("OMP_ENDPOINT")
            .or_else(|_| std::env::var("AI_ENDPOINT"))
            .or_else(|_| std::env::var("OPENAI_BASE_URL"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let resolved_provider = if !p_trimmed.is_empty() && p_trimmed != "default" {
            p_trimmed.to_string()
        } else {
            std::env::var("OMP_PROVIDER")
                .or_else(|_| std::env::var("AI_PROVIDER"))
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s != "default")
                .unwrap_or_else(|| {
                    if endpoint.is_some() {
                        "openai_compatible".to_string()
                    } else {
                        String::new()
                    }
                })
        };

        if resolved_provider.is_empty() && endpoint.is_none() {
            return Err(StructuredError::new(
                "missing_provider",
                "Explicit provider is required; no silent fallback to default provider is permitted",
                false,
            ));
        }

        let resolved_model = if !m_trimmed.is_empty() && m_trimmed != "default" {
            m_trimmed.to_string()
        } else {
            std::env::var("OMP_MODEL")
                .or_else(|_| std::env::var("AI_MODEL"))
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s != "default")
                .unwrap_or_default()
        };

        Self::new(resolved_provider, resolved_model, endpoint)
    }

    /// Explicit constructor for provider, model, and optional endpoint.
    /// Pure constructor with no HTTP operations.
    /// Allows endpoint-only configuration with no model for catalog lookup.
    /// Credentials default to OMP_API_KEY precedence.
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        endpoint: Option<String>,
    ) -> Result<Self, StructuredError> {
        let mut p_str = provider.into().trim().to_lowercase();
        let mut m_str = model.into().trim().to_string();
        if m_str == "default" {
            m_str.clear();
        }

        if (p_str.is_empty() || p_str == "default") && endpoint.is_some() {
            p_str = "openai_compatible".to_string();
        }

        if p_str.is_empty() || p_str == "default" {
            return Err(StructuredError::new(
                "missing_provider",
                "Explicit provider is required; no silent fallback is allowed",
                false,
            ));
        }

        let validated_endpoint = if let Some(ep) = endpoint {
            let valid = validate_endpoint_url(&ep)?;
            Some(valid)
        } else {
            None
        };

        let route = match p_str.as_str() {
            "anthropic" => ProviderRoute::AnthropicDirect,
            "openai" => ProviderRoute::OpenAiDirect,
            "azure" | "azure_openai" => ProviderRoute::AzureOpenAi {
                resource_name: "default".into(),
                deployment: m_str.clone(),
            },
            "ollama" => ProviderRoute::Ollama {
                base_url: validated_endpoint
                    .clone()
                    .unwrap_or_else(|| "http://localhost:11434".into()),
            },
            _ => {
                if let Some(ep) = &validated_endpoint {
                    ProviderRoute::OpenAiCompatible {
                        base_url: ep.clone(),
                    }
                } else {
                    return Err(StructuredError::new(
                        "missing_endpoint",
                        format!("Explicit endpoint is required for provider '{p_str}'"),
                        false,
                    ));
                }
            }
        };

        let ident = ModelIdentifier::infer(&m_str, route);
        let compat_table = CompatTable::standard();
        let caps = compat_table.resolve(&ident);

        // Load credentials from environment: OMP_API_KEY takes precedence
        let api_key = std::env::var("OMP_API_KEY").ok().or_else(|| {
            if p_str == "anthropic" {
                std::env::var("ANTHROPIC_API_KEY").ok()
            } else {
                std::env::var("OPENAI_API_KEY")
                    .or_else(|_| std::env::var("OPENAI_COMPATIBLE_API_KEY"))
                    .ok()
            }
        });

        Ok(Self {
            conversation_id: None,
            provider: p_str,
            model: m_str,
            ident,
            caps,
            api_key,
            endpoint: validated_endpoint,
            timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
            max_response_bytes: 10 * 1024 * 1024,
            tool_force: None,
            attempt_count: 0,
            metadata: None,
        })
    }

    /// Sets an injected or custom endpoint (e.g. local test mock or private proxy).
    /// The endpoint is validated and never stored unvalidated: invalid input
    /// keeps the previous endpoint and returns an error.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Result<Self, StructuredError> {
        let ep = endpoint.into();
        let valid = validate_endpoint_url(&ep)?;
        if let ProviderRoute::OpenAiCompatible { base_url } = &mut self.ident.route {
            *base_url = valid.clone();
        } else if let ProviderRoute::Ollama { base_url } = &mut self.ident.route {
            *base_url = valid.clone();
        }
        self.endpoint = Some(valid);
        Ok(self)
    }

    /// Updates the active model and taxonomy limits from advertised ModelMetadata.
    pub fn apply_model_metadata(&mut self, metadata: &ModelMetadata) {
        self.model = metadata.id.clone();
        self.ident = ModelIdentifier::infer(&self.model, self.ident.route.clone());
        self.ident.taxonomy.context_window = metadata.context_length;
        self.ident.taxonomy.max_output_tokens = metadata.max_output_tokens;
        if let Some(meta) = &mut self.metadata {
            meta.active_model = Some(metadata.clone());
        }
    }

    /// Sets the idle read/write timeout in seconds, clamped to
    /// `MIN_REQUEST_TIMEOUT_SECS..=MAX_REQUEST_TIMEOUT_SECS`.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs.clamp(MIN_REQUEST_TIMEOUT_SECS, MAX_REQUEST_TIMEOUT_SECS);
        self
    }

    /// Sets the maximum response bytes limit before streaming truncation.
    pub fn with_max_response_bytes(mut self, bytes: usize) -> Self {
        self.max_response_bytes = bytes;
        self
    }

    /// Sets or overrides the API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the stable id sent to hosts that require a conversation header.
    pub fn with_conversation_id(mut self, id: impl Into<String>) -> Self {
        self.conversation_id = Some(id.into());
        self
    }

    /// Names this client and its conversation on an outgoing request.
    ///
    /// Gateways that route coding-agent traffic reject generic library user
    /// agents, and some (OpenCode Go) reject a request that carries no
    /// conversation id outright.
    fn apply_client_identity(&self, wire_req: &mut ProviderRequest) {
        wire_req
            .headers
            .push(("user-agent".into(), CLIENT_USER_AGENT.to_string()));
        if Self::requires_conversation_header(&wire_req.endpoint)
            && let Some(conversation) = &self.conversation_id
        {
            wire_req
                .headers
                .push((CONVERSATION_HEADER.into(), conversation.clone()));
        }
    }

    /// True when this endpoint belongs to a host that requires the
    /// conversation header on inference requests.
    fn requires_conversation_header(endpoint: &str) -> bool {
        let host = endpoint
            .split("://")
            .nth(1)
            .unwrap_or(endpoint)
            .split(['/', '?'])
            .next()
            .unwrap_or("")
            .rsplit('@')
            .next()
            .unwrap_or("")
            .split(':')
            .next()
            .unwrap_or("");
        CONVERSATION_HEADER_HOSTS
            .iter()
            .any(|known| host.eq_ignore_ascii_case(known) || host.ends_with(&format!(".{known}")))
    }

    /// Attaches a cost-aware tool force policy to this client session.
    pub fn with_tool_force(mut self, policy: ToolForcePolicy) -> Self {
        self.tool_force = Some(policy);
        self
    }

    /// Refreshes/validates host credentials.
    pub fn authenticate_refresh(&mut self) -> Result<(), StructuredError> {
        let refreshed = Self::new(&self.provider, &self.model, self.endpoint.clone())?;
        self.api_key = refreshed.api_key;
        if self.endpoint.is_none() && self.api_key.is_none() {
            return Err(StructuredError::new(
                "unauthenticated",
                "No host credential is available for refresh",
                false,
            ));
        }
        Ok(())
    }

    /// Returns the currently cached provider metadata, if available.
    pub fn metadata(&self) -> Option<&ProviderMetadata> {
        self.metadata.as_ref()
    }

    /// Returns the active model's advertised metadata, if available.
    pub fn active_model_metadata(&self) -> Option<&ModelMetadata> {
        self.metadata.as_ref().and_then(|m| m.active_model.as_ref())
    }

    /// Fetches model catalog from the configured endpoint, updates cached metadata,
    /// and represents unavailable fields as None rather than hardcoding limits.
    /// Clears stale metadata on refresh failure.
    pub fn refresh_metadata(&mut self) -> Result<ProviderMetadata, StructuredError> {
        self.metadata = None;
        self.ident.taxonomy.context_window = None;
        self.ident.taxonomy.max_output_tokens = None;
        let raw_endpoint = self.endpoint.as_deref().ok_or_else(|| {
            StructuredError::new(
                "missing_endpoint",
                format!(
                    "No endpoint is configured for provider '{}' to refresh metadata",
                    self.provider
                ),
                false,
            )
        })?;

        match self.do_refresh_metadata(raw_endpoint) {
            Ok(metadata) => {
                self.caps
                    .set(ProviderCapability::ModelDiscovery, TriState::Supported);
                self.metadata = Some(metadata.clone());
                Ok(metadata)
            }
            Err(err) => {
                self.metadata = None;
                Err(err)
            }
        }
    }

    fn do_refresh_metadata(&self, raw_endpoint: &str) -> Result<ProviderMetadata, StructuredError> {
        validate_endpoint_url(raw_endpoint)?;

        let timeout = Duration::from_secs(self.timeout_secs.clamp(1, 10));
        let agent = ureq::AgentBuilder::new()
            .timeout(timeout)
            .redirects(0)
            .build();

        let primary_url = normalize_models_endpoint(raw_endpoint);
        let mut models = self.fetch_all_model_pages(&agent, &primary_url)?;
        if models.len() > 4096
            || serde_json::to_vec(&models)
                .map(|data| data.len())
                .unwrap_or(usize::MAX)
                > 512 * 1024
        {
            return Err(StructuredError::new(
                "catalog_limit",
                "Provider model catalog exceeds the bounded session metadata limit",
                false,
            ));
        }

        let provenance = self.enrich_models_from_registry(raw_endpoint, &mut models);

        // Match active model against advertised catalog by exact match only (no fuzzy/synthesized)
        let active_model = if !self.model.is_empty() && self.model != "default" {
            models.iter().find(|m| m.id == self.model).cloned()
        } else {
            None
        };

        let refreshed_at_unix_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs());

        Ok(ProviderMetadata {
            provider: self.provider.clone(),
            endpoint: self.endpoint.clone(),
            models,
            active_model,
            refreshed_at_unix_secs,
            provenance,
        })
    }

    /// Enriches missing context length on discovered models from the public models registry (models.dev),
    /// bounded in time and size, without overriding direct advertised limits or leaking credentials.
    pub fn enrich_models_from_registry(
        &self,
        raw_endpoint: &str,
        models: &mut [ModelMetadata],
    ) -> Option<String> {
        if !models.iter().any(|m| m.context_length.is_none()) {
            return None;
        }

        let registry_override = std::env::var("OMP_REGISTRY_URL")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let can_query = registry_override.is_some() || is_public_https_endpoint(raw_endpoint);
        if !can_query {
            return None;
        }

        let registry_url = registry_override
            .as_deref()
            .unwrap_or("https://models.dev/api.json");

        let timeout = Duration::from_secs(self.timeout_secs.clamp(1, 10));
        let agent = ureq::AgentBuilder::new()
            .timeout(timeout)
            .redirects(0)
            .build();

        let req = agent
            .get(registry_url)
            .set("accept", "application/json")
            .set("user-agent", CLIENT_USER_AGENT);

        let resp = match req.call() {
            Ok(resp) => resp,
            Err(_) => return None,
        };

        let mut bytes = Vec::new();
        const MAX_REGISTRY_BYTES: usize = 8 * 1024 * 1024;
        if resp
            .into_reader()
            .take((MAX_REGISTRY_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .is_err()
            || bytes.len() > MAX_REGISTRY_BYTES
        {
            return None;
        }

        let target_api = normalize_api_base(raw_endpoint);
        let matched = match parse_registry_for_endpoint(&bytes, &target_api) {
            Ok(Some(matched)) => matched,
            _ => return None,
        };

        let mut provenance = None;
        for model in models.iter_mut() {
            if model.context_length.is_none()
                && let Some(Some(context_len)) = matched.models.get(&model.id) {
                    model.context_length = Some(*context_len);
                    let prov = format!("models.dev:{}", matched.key);
                    model.context_provenance = Some(prov.clone());
                    provenance = Some(prov);
                }
        }

        provenance
    }

    fn fetch_all_model_pages(
        &self,
        agent: &ureq::Agent,
        initial_url: &str,
    ) -> Result<Vec<ModelMetadata>, StructuredError> {
        let mut all_models: Vec<ModelMetadata> = Vec::new();
        let mut seen_ids = HashSet::new();
        let mut current_url = initial_url.to_string();
        let max_pages = 10;
        let mut page_count = 0;

        let deadline =
            std::time::Instant::now() + Duration::from_secs(self.timeout_secs.clamp(1, 10));
        while page_count < max_pages {
            page_count += 1;
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Err(StructuredError::new(
                    "catalog_timeout",
                    "Catalog refresh deadline exceeded",
                    true,
                ));
            }
            let val = self.execute_models_get(agent, &current_url, remaining)?;
            let page_models = parse_models_catalog(&val)?;

            if page_models.is_empty() && page_count > 1 {
                break;
            }

            for model in page_models {
                if !seen_ids.insert(model.id.clone()) {
                    return Err(StructuredError::new(
                        "duplicate_model_id",
                        format!(
                            "Provider catalog contains duplicate model ID: '{}'",
                            model.id
                        ),
                        false,
                    ));
                }
                all_models.push(model);
            }

            let next_url_opt = extract_next_page_url(&val, &current_url, initial_url)?;
            match next_url_opt {
                Some(next_url) => {
                    if next_url == current_url || page_count == max_pages {
                        return Err(StructuredError::new(
                            "catalog_page_limit",
                            "Catalog pagination did not finish within the page limit",
                            false,
                        ));
                    }
                    current_url = next_url;
                }
                None => break,
            }
        }

        if all_models.is_empty() {
            return Err(StructuredError::new(
                "empty_catalog",
                "Provider returned an empty model catalog",
                false,
            ));
        }

        Ok(all_models)
    }

    fn execute_models_get(
        &self,
        agent: &ureq::Agent,
        url: &str,
        remaining: Duration,
    ) -> Result<serde_json::Value, StructuredError> {
        let mut req = agent.get(url).timeout(remaining);
        req = req.set("accept", "application/json");
        req = req.set("user-agent", CLIENT_USER_AGENT);
        if let Some(key) = &self.api_key {
            if self.provider == "anthropic" {
                req = req
                    .set("x-api-key", key)
                    .set("anthropic-version", "2023-06-01");
            } else {
                req = req.set("authorization", &format!("Bearer {key}"));
            }
        }

        let resp = req.call().map_err(|err| match err {
            ureq::Error::Status(code, _resp) => StructuredError::new(
                "http_status_error",
                format!("HTTP {code}"),
                code == 429 || code >= 500,
            ),
            ureq::Error::Transport(err) => {
                StructuredError::new("network_error", err.to_string(), true)
            }
        })?;

        let mut bytes = Vec::new();
        resp.into_reader()
            .take(2 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| {
                StructuredError::new("catalog_read", "Failed to read catalog response", true)
            })?;
        if bytes.len() > 2 * 1024 * 1024 {
            return Err(StructuredError::new(
                "catalog_limit",
                "Catalog response exceeds 2 MiB",
                false,
            ));
        }
        serde_json::from_slice(&bytes).map_err(|_| {
            StructuredError::new(
                "invalid_json_catalog",
                "Provider catalog is not valid JSON",
                false,
            )
        })
    }
    /// Returns streaming capability tri-state explicitly.
    pub fn stream_capable(&self) -> TriState {
        self.caps.get(ProviderCapability::Streaming)
    }

    /// Queries token count for text. Returns explicit error when unsupported.
    pub fn token_count(&self, text: &str) -> Result<usize, StructuredError> {
        // Fail before touching the network when no credential is configured,
        // mirroring the `missing_credentials` path in `infer()`.
        if self.api_key.as_deref().is_none_or(|key| key.trim().is_empty()) {
            return Err(StructuredError::new(
                "missing_credentials",
                "No API key configured for token counting",
                false,
            ));
        }
        match self.caps.get(ProviderCapability::TokenCount) {
            TriState::Supported => {
                if self.provider != "anthropic" {
                    return Err(StructuredError::new(
                        "unsupported_capability",
                        "No exact tokenizer adapter is configured for this route",
                        false,
                    ));
                }
                let api_key = self.api_key.as_deref().unwrap_or("");
                let response = ureq::AgentBuilder::new().timeout(Duration::from_secs(self.timeout_secs)).redirects(0).build()
                    .post(&count_tokens_endpoint(self.endpoint.as_deref()))
                    .set("anthropic-version", "2023-06-01")
                    .set("x-api-key", api_key)
                    .send_json(serde_json::json!({"model":self.model,"messages":[{"role":"user","content":text}]}))
                    .map_err(|error| StructuredError::new("token_count_failed", error.to_string(), true))?;
                let value: serde_json::Value = serde_json::from_reader(
                    response.into_reader().take(65536),
                )
                .map_err(|error| {
                    StructuredError::new("token_count_response", error.to_string(), false)
                })?;
                value["input_tokens"]
                    .as_u64()
                    .map(|value| value as usize)
                    .ok_or_else(|| {
                        StructuredError::new(
                            "token_count_response",
                            "Provider omitted input_tokens",
                            false,
                        )
                    })
            }
            TriState::Unsupported => Err(StructuredError::new(
                "unsupported_capability",
                format!(
                    "Token counting is unsupported for provider '{}'",
                    self.provider
                ),
                false,
            )),
            TriState::Unknown => Err(StructuredError::new(
                "unknown_capability",
                format!(
                    "Token counting support is unknown for provider '{}'",
                    self.provider
                ),
                false,
            )),
        }
    }

    /// Queries usage status. Returns explicit error when unsupported.
    pub fn usage_query(&self) -> Result<serde_json::Value, StructuredError> {
        match self.caps.get(ProviderCapability::UsageQuery) {
            TriState::Supported => Err(StructuredError::new(
                "unsupported_capability",
                "Account usage requires an administrative provider adapter; per-turn measured usage is returned by infer",
                false,
            )),
            TriState::Unsupported => Err(StructuredError::new(
                "unsupported_capability",
                format!(
                    "Usage query is unsupported for provider '{}'",
                    self.provider
                ),
                false,
            )),
            TriState::Unknown => Err(StructuredError::new(
                "unknown_capability",
                format!(
                    "Usage query support is unknown for provider '{}'",
                    self.provider
                ),
                false,
            )),
        }
    }

    /// Returns explicit tri-state for any provider capability.
    pub fn capability_state(&self, cap: ProviderCapability) -> TriState {
        self.caps.get(cap)
    }

    /// Executes a semantic inference request producing a canonical turn.
    /// Performs bounded network streaming, thinking token normalization,
    /// leaked tool-call synthesis, and schema argument repair.
    pub fn infer(
        &mut self,
        request: &InferenceRequest,
        on_delta: &mut dyn FnMut(InferenceDelta<'_>) -> Result<(), StructuredError>,
    ) -> Result<CanonicalTurn, StructuredError> {
        if !self.is_configured() {
            if self.model.is_empty() || self.model == "default" {
                return Err(StructuredError::new(
                    "missing_model",
                    "Cannot execute inference: model is not configured",
                    false,
                ));
            }
            return Err(StructuredError::new(
                "unconfigured_provider",
                "Cannot execute inference: provider and model are not configured",
                false,
            ));
        }

        // Validate credentials if not using a custom local endpoint
        if self.endpoint.is_none() && self.api_key.is_none() {
            return Err(StructuredError::new(
                "missing_credentials",
                format!(
                    "No API key found in host environment for provider '{}'. Please set {}_API_KEY",
                    self.provider,
                    self.provider.to_uppercase()
                ),
                false,
            ));
        }

        // 1. Prepare request under cost-aware forced tool policy (if configured)
        let mut effective_req = request.clone();
        if let Some(policy) = &mut self.tool_force {
            policy.prepare_request(&mut effective_req, &self.caps)?;
        }

        // 2. Translate semantic request to provider wire representation
        let mut wire_req = translate_request(&effective_req, &self.ident, &self.caps)?;

        // Override endpoint if an injected or custom endpoint was set
        if let Some(ep) = &self.endpoint {
            wire_req.endpoint = resolve_turn_endpoint(ep, wire_req.wire_format);
        }
        self.apply_client_identity(&mut wire_req);
        // Inject authentication headers from environment
        if let Some(key) = &self.api_key {
            match wire_req.wire_format {
                WireFormat::AnthropicMessages => {
                    wire_req.headers.push(("x-api-key".into(), key.clone()));
                }
                _ => {
                    wire_req
                        .headers
                        .push(("authorization".into(), format!("Bearer {key}")));
                }
            }
        }

        // Unknown routes may negotiate SSE; a JSON response remains valid. Do not
        // promote unknown metadata to supported or silently retry a failed request.
        if self.caps.get(ProviderCapability::Streaming) != TriState::Unsupported {
            wire_req.body["stream"] = serde_json::json!(true);
        }

        // 3. Execute bounded HTTP request.
        //
        // A stall before the first token is retried once: nothing has reached the
        // session yet, so re-issuing the same request cannot duplicate output, and
        // a provider that stalls once usually answers the second attempt. After
        // the first delta a retry would duplicate text, so the failure surfaces.
        let mut emitted = false;
        let mut stall_retried = false;
        let (raw_text, mut thinking, raw_tool_calls, usage, finish_reason) = loop {
            let mut observer = |delta: InferenceDelta<'_>| {
                emitted = true;
                on_delta(delta)
            };
            match execute_wire_request(
                &wire_req,
                self.timeout_secs,
                self.max_response_bytes,
                &mut observer,
            ) {
                Ok(parsed) => break parsed,
                Err(error) => {
                    let retryable = !emitted
                        && !stall_retried
                        && matches!(error.code.as_str(), "stream_stalled" | "network_error");
                    if !retryable {
                        return Err(error);
                    }
                    stall_retried = true;
                }
            }
        };

        // 4. Corrective: normalize thinking tokens if not already separated
        let mut text = raw_text;
        if thinking.is_none() {
            let norm = corrective::normalize_thinking_tokens(&text);
            if norm.has_thinking {
                thinking = norm.thinking;
                text = norm.content;
            }
        }

        // 5. Corrective: detect repetition loops in output
        if let Some(repetition) = corrective::detect_repetition_loop(&text, 12, 3) {
            let mut error = StructuredError::new(
                "repetition_loop",
                "Provider output entered a repetition loop",
                true,
            );
            // `RepetitionDetection` serializes infallibly in practice; fall back
            // to the raw phrase so diagnostics are never lost to a panic.
            error.diagnostics = serde_json::to_value(&repetition).ok().or_else(|| {
                Some(serde_json::json!({ "repeated_phrase": repetition.repeated_phrase }))
            });
            return Err(error);
        }

        // 6. Repair: extract leaked tool calls from text if none came over wire
        let mut initial_calls = raw_tool_calls;
        if initial_calls.is_empty() {
            let leaked = repair::extract_leaked_tool_calls(&text);
            if !leaked.is_empty() {
                initial_calls = leaked;
            }
        }

        // 7. Repair: validate and repair arguments against tool schemas
        let mut canonical_tool_calls = Vec::new();
        for tc in initial_calls {
            let mut args = tc.arguments.clone();

            // If arguments arrived as a JSON string, attempt malformed repair
            if let serde_json::Value::String(s) = &args {
                match corrective::repair_malformed_json(s, 5) {
                    Ok(parsed) => args = parsed,
                    Err(e) => {
                        return Err(e);
                    }
                }
            }

            // If active tools contain a schema for this tool, validate and repair
            if let Some(tool_schema) = effective_req
                .active_tools
                .iter()
                .find(|t| t.name == tc.name)
            {
                match repair::repair_arguments(&tc.name, &args, &tool_schema.parameters) {
                    Ok(outcome) => {
                        args = outcome.arguments;
                    }
                    Err(e) => {
                        // Ambiguous or unrepairable schema mismatch produces retryable error
                        return Err(e);
                    }
                }
            }

            canonical_tool_calls.push(ToolCallSpec {
                id: tc.id,
                name: tc.name,
                arguments: args,
            });
        }

        // 8. Update tool force policy attempt state
        self.attempt_count += 1;
        if let Some(policy) = &mut self.tool_force {
            let tool_names: Vec<String> = canonical_tool_calls
                .iter()
                .map(|t| t.name.clone())
                .collect();
            policy.record_turn_result(&tool_names)?;
        }

        Ok(CanonicalTurn {
            text,
            thinking,
            tool_calls: canonical_tool_calls,
            usage,
            finish_reason,
        })
    }
}

/// Normalizes an endpoint URL to target the OpenAI-compatible `/models` resource.
pub fn normalize_models_endpoint(endpoint: &str) -> String {
    format!("{}/models", endpoint_base(endpoint))
}

/// Normalizes turn endpoint based on wire format.
fn resolve_turn_endpoint(endpoint: &str, wire_format: WireFormat) -> String {
    let base = endpoint_base(endpoint);
    match wire_format {
        WireFormat::AnthropicMessages => format!("{base}/messages"),
        _ => format!("{base}/chat/completions"),
    }
}

fn endpoint_base(endpoint: &str) -> String {
    let mut base = endpoint.trim().trim_end_matches('/');
    for suffix in ["/chat/completions", "/messages", "/models", "/completions"] {
        if let Some(value) = base.strip_suffix(suffix) {
            base = value;
            break;
        }
    }
    let authority_end = base.find("://").map(|index| index + 3).unwrap_or(0);
    if !base[authority_end..].contains('/') {
        format!("{base}/v1")
    } else {
        base.into()
    }
}

/// Composes the Anthropic count_tokens endpoint using the same base
/// normalization as turn requests, so a bare host, a `/v1` base, or a
/// full `/v1/messages` endpoint all resolve to `<base>/messages/count_tokens`.
fn count_tokens_endpoint(endpoint: Option<&str>) -> String {
    format!(
        "{}/messages/count_tokens",
        endpoint_base(endpoint.unwrap_or("https://api.anthropic.com"))
    )
}

/// Validates that an endpoint URL uses HTTPS or loopback HTTP,
/// contains no credentials in userinfo or query string, and contains no URL fragment.
///
/// Implementation note: the URL is parsed with a throwaway `ureq` request
/// builder purely as a standards-compliant parser — no network I/O occurs
/// (the request is never sent).
pub fn validate_endpoint_url(raw: &str) -> Result<String, StructuredError> {
    let trimmed = raw.trim();
    let agent = ureq::builder().build();
    let req = agent.get(trimmed);
    let req_url = req.request_url().map_err(|e| {
        StructuredError::new(
            "invalid_endpoint_url",
            format!("Invalid endpoint URL: {e}"),
            false,
        )
    })?;
    let u = req_url.as_url();

    let scheme = u.scheme();
    if scheme != "https" && scheme != "http" {
        return Err(StructuredError::new(
            "unsupported_scheme",
            format!(
                "Endpoint scheme '{scheme}' is unsupported; must be https or http (loopback only)"
            ),
            false,
        ));
    }

    let host = u.host_str().ok_or_else(|| {
        StructuredError::new(
            "invalid_endpoint_url",
            "Endpoint URL is missing a host",
            false,
        )
    })?;

    if scheme == "http" && !is_loopback_host(host) {
        return Err(StructuredError::new(
            "insecure_endpoint",
            format!(
                "Insecure HTTP is only permitted for loopback addresses (localhost, 127.0.0.1, [::1]); host '{host}' requires HTTPS"
            ),
            false,
        ));
    }

    if !u.username().is_empty() || u.password().is_some() {
        return Err(StructuredError::new(
            "credential_in_url",
            "Userinfo credentials are not permitted in endpoint URLs",
            false,
        ));
    }

    if u.fragment().is_some() {
        return Err(StructuredError::new(
            "fragment_in_url",
            "URL fragments are not permitted in endpoint URLs",
            false,
        ));
    }

    if u.query().is_some() {
        return Err(StructuredError::new(
            "credential_in_url",
            "Endpoint query strings are not allowed; configure credentials separately",
            false,
        ));
    }

    Ok(trimmed.to_string())
}

fn is_loopback_host(host: &str) -> bool {
    let h = host.trim().trim_start_matches('[').trim_end_matches(']');
    if h.eq_ignore_ascii_case("localhost") {
        return true;
    }
    if let Ok(ip) = h.parse::<Ipv4Addr>() {
        return ip.is_loopback();
    }
    if let Ok(ip) = h.parse::<Ipv6Addr>() {
        return ip.is_loopback();
    }
    false
}

fn is_private_host(host: &str) -> bool {
    let h = host.trim().trim_start_matches('[').trim_end_matches(']');
    if h.eq_ignore_ascii_case("localhost") || h.ends_with(".local") || h.ends_with(".localhost") {
        return true;
    }
    if let Ok(ip) = h.parse::<Ipv4Addr>() {
        return ip.is_loopback() || ip.is_private() || ip.is_link_local();
    }
    if let Ok(ip) = h.parse::<Ipv6Addr>() {
        let segs = ip.segments();
        let is_unique_local = (segs[0] & 0xfe00) == 0xfc00;
        let is_link_local = (segs[0] & 0xffc0) == 0xfe80;
        return ip.is_loopback() || is_unique_local || is_link_local;
    }
    false
}

/// Determines whether an endpoint URL is an HTTPS URL targeting a public (non-loopback, non-private) host.
pub fn is_public_https_endpoint(raw: &str) -> bool {
    let trimmed = raw.trim();
    if !trimmed.starts_with("https://") {
        return false;
    }
    if let Some(rest) = trimmed.strip_prefix("https://") {
        let host = rest.split(['/', ':', '?', '#']).next().unwrap_or("");
        let host = host.trim().trim_start_matches('[').trim_end_matches(']');
        if host.is_empty() || is_loopback_host(host) || is_private_host(host) {
            return false;
        }
        return true;
    }
    false
}

/// Normalizes an API base URL for exact matching against public model registries (such as models.dev).
/// Strips query params, URL fragments, trailing slashes, and common endpoints (`/models`, `/chat/completions`, etc.),
/// while lowercasing the scheme and host.
pub fn normalize_api_base(raw: &str) -> String {
    let mut base = raw.trim();
    if let Some(pos) = base.find(['?', '#']) {
        base = &base[..pos];
    }
    let mut base = base.trim_end_matches('/');
    for suffix in ["/chat/completions", "/messages", "/models", "/completions"] {
        if let Some(stripped) = base.strip_suffix(suffix) {
            base = stripped.trim_end_matches('/');
            break;
        }
    }
    if let Some((scheme, rest)) = base.split_once("://") {
        if let Some((host_port, path)) = rest.split_once('/') {
            let clean_path = path.trim_matches('/');
            if clean_path.is_empty() {
                format!(
                    "{}://{}/v1",
                    scheme.to_ascii_lowercase(),
                    host_port.to_ascii_lowercase()
                )
            } else {
                format!(
                    "{}://{}/{}",
                    scheme.to_ascii_lowercase(),
                    host_port.to_ascii_lowercase(),
                    clean_path
                )
            }
        } else {
            format!(
                "{}://{}/v1",
                scheme.to_ascii_lowercase(),
                rest.trim_matches('/').to_ascii_lowercase()
            )
        }
    } else {
        base.to_string()
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RegistryLimitEntry {
    #[serde(default)]
    context: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct RegistryModelEntry {
    #[serde(default)]
    limit: Option<RegistryLimitEntry>,
}

/// Matched provider entry from the public models registry.
#[derive(Debug, Clone)]
pub struct MatchedRegistryProvider {
    pub key: String,
    pub models: HashMap<String, Option<usize>>,
}

struct RegistryLookup<'a> {
    target_endpoint_base: &'a str,
}

impl<'de, 'a> DeserializeSeed<'de> for RegistryLookup<'a> {
    type Value = Option<MatchedRegistryProvider>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        deserializer.deserialize_map(RegistryRootVisitor {
            target_endpoint_base: self.target_endpoint_base,
        })
    }
}

struct RegistryRootVisitor<'a> {
    target_endpoint_base: &'a str,
}

impl<'de, 'a> serde::de::Visitor<'de> for RegistryRootVisitor<'a> {
    type Value = Option<MatchedRegistryProvider>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a registry map of provider objects")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: serde::de::MapAccess<'de>,
    {
        let mut matched: Option<MatchedRegistryProvider> = None;

        while let Some(key) = map.next_key::<String>()? {
            let provider_res = map.next_value_seed(ProviderFieldVisitor {
                target_endpoint_base: self.target_endpoint_base,
            })?;

            if let Some(models) = provider_res {
                if let Some(existing) = &mut matched {
                    existing.models.extend(models);
                } else {
                    matched = Some(MatchedRegistryProvider { key, models });
                }
            }
        }

        Ok(matched)
    }
}

struct ProviderFieldVisitor<'a> {
    target_endpoint_base: &'a str,
}

impl<'de, 'a> DeserializeSeed<'de> for ProviderFieldVisitor<'a> {
    type Value = Option<HashMap<String, Option<usize>>>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        deserializer.deserialize_map(self)
    }
}

impl<'de, 'a> serde::de::Visitor<'de> for ProviderFieldVisitor<'a> {
    type Value = Option<HashMap<String, Option<usize>>>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a provider object with api and models")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: serde::de::MapAccess<'de>,
    {
        let mut api_matches = None;
        let mut parsed_models = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "api" => {
                    let api_str = map.next_value::<String>()?;
                    let matches = normalize_api_base(&api_str) == self.target_endpoint_base;
                    api_matches = Some(matches);
                    if !matches {
                        parsed_models = None;
                    }
                }
                "models" => match api_matches {
                    Some(false) => {
                        map.next_value::<serde::de::IgnoredAny>()?;
                    }
                    Some(true) => {
                        parsed_models =
                            Some(map.next_value::<HashMap<String, RegistryModelEntry>>()?);
                    }
                    None => {
                        parsed_models =
                            Some(map.next_value::<HashMap<String, RegistryModelEntry>>()?);
                    }
                },
                _ => {
                    map.next_value::<serde::de::IgnoredAny>()?;
                }
            }
        }

        if api_matches == Some(true) {
            let model_map = parsed_models
                .unwrap_or_default()
                .into_iter()
                .map(|(id, m)| (id, m.limit.and_then(|l| l.context)))
                .collect();
            Ok(Some(model_map))
        } else {
            Ok(None)
        }
    }
}

/// Parses the public models registry JSON, extracting model limits only for the provider
/// whose normalized API base URL matches `target_api_base`. Uses typed narrowed deserialization
/// and `IgnoredAny` to avoid allocating unused model catalogs.
pub fn parse_registry_for_endpoint(
    bytes: &[u8],
    target_api_base: &str,
) -> Result<Option<MatchedRegistryProvider>, StructuredError> {
    let mut de = serde_json::Deserializer::from_slice(bytes);
    let seed = RegistryLookup {
        target_endpoint_base: target_api_base,
    };
    seed.deserialize(&mut de).map_err(|e| {
        StructuredError::new(
            "invalid_registry_json",
            format!("Failed to deserialize public models registry: {e}"),
            false,
        )
    })
}

fn is_same_origin(base: &str, candidate: &str) -> bool {
    let agent = ureq::builder().build();
    let base_req = match agent.get(base).request_url() {
        Ok(r) => r,
        Err(_) => return false,
    };
    let cand_req = match agent.get(candidate).request_url() {
        Ok(r) => r,
        Err(_) => return false,
    };
    let b = base_req.as_url();
    let c = cand_req.as_url();
    b.scheme() == c.scheme()
        && b.host_str() == c.host_str()
        && b.port_or_known_default() == c.port_or_known_default()
}

fn set_query_param(base_url: &str, key: &str, value: &str) -> Result<String, StructuredError> {
    let request = ureq::get(base_url);
    let parsed = request
        .request_url()
        .map_err(|_| StructuredError::new("invalid_url", "Invalid catalog URL", false))?;
    let mut url = parsed.as_url().clone();
    url.set_query(None);
    url.query_pairs_mut().append_pair(key, value);
    Ok(url.to_string())
}

fn extract_next_page_url(
    val: &serde_json::Value,
    current_url: &str,
    initial_url: &str,
) -> Result<Option<String>, StructuredError> {
    if val.get("has_more").and_then(|v| v.as_bool()) == Some(true) {
        if let Some(last_id) = val.get("last_id").and_then(|v| v.as_str()) {
            let next_url = set_query_param(initial_url, "after_id", last_id)?;
            return Ok(Some(next_url));
        }
        if let Some(cursor) = val
            .get("next_cursor")
            .or_else(|| val.get("after"))
            .and_then(|v| v.as_str())
        {
            let next_url = set_query_param(initial_url, "after", cursor)?;
            return Ok(Some(next_url));
        }
    }

    if let Some(cursor) = val.get("next_cursor").and_then(|v| v.as_str())
        && !cursor.is_empty() {
            let next_url = set_query_param(initial_url, "after", cursor)?;
            return Ok(Some(next_url));
        }

    if let Some(next_str) = val
        .get("next")
        .or_else(|| val.get("next_page"))
        .and_then(|v| v.as_str())
        .filter(|v| !v.is_empty())
    {
        let request = ureq::get(current_url);
        let parsed = request
            .request_url()
            .map_err(|_| StructuredError::new("invalid_url", "Invalid catalog URL", false))?;
        let next = parsed.as_url().join(next_str).map_err(|_| {
            StructuredError::new("invalid_pagination", "Invalid next page URL", false)
        })?;
        if !is_same_origin(initial_url, next.as_str())
            || !next.username().is_empty()
            || next.password().is_some()
            || next.fragment().is_some()
        {
            return Err(StructuredError::new(
                "insecure_pagination",
                "Catalog pagination crossed origins or contained credentials",
                false,
            ));
        }
        return Ok(Some(next.to_string()));
    }
    if val.get("has_more").and_then(|v| v.as_bool()) == Some(true) {
        return Err(StructuredError::new(
            "invalid_pagination",
            "Provider advertised more models without a cursor",
            false,
        ));
    }

    Ok(None)
}

/// Parses models catalog from provider response into structured ModelMetadata entries.
pub fn parse_models_catalog(
    val: &serde_json::Value,
) -> Result<Vec<ModelMetadata>, StructuredError> {
    let raw_list = if let Some(arr) = val.get("data").and_then(|v| v.as_array()) {
        arr
    } else if let Some(arr) = val.get("models").and_then(|v| v.as_array()) {
        arr
    } else if let Some(arr) = val.as_array() {
        arr
    } else {
        return Err(StructuredError::new(
            "malformed_catalog",
            "Models catalog response does not contain a data or models array",
            false,
        ));
    };

    raw_list
        .iter()
        .map(|entry| {
            parse_model_metadata(entry).ok_or_else(|| {
                StructuredError::new(
                    "malformed_catalog",
                    "Catalog model entry has no usable ID",
                    false,
                )
            })
        })
        .collect()
}

fn parse_model_metadata(val: &serde_json::Value) -> Option<ModelMetadata> {
    let id = val
        .get("id")
        .and_then(|v| v.as_str())
        .or_else(|| val.get("name").and_then(|v| v.as_str()))
        .or_else(|| val.get("model").and_then(|v| v.as_str()))
        .or_else(|| val.as_str())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && s.len() <= 512 && !s.chars().any(char::is_control))?
        .to_string();

    let context_length = val
        .get("context_length")
        .or_else(|| val.get("context_window"))
        .or_else(|| val.get("max_context_tokens"))
        .or_else(|| val.get("max_model_len"))
        .or_else(|| val.get("max_position_embeddings"))
        .or_else(|| val.pointer("/limits/context_window"))
        .or_else(|| val.pointer("/limits/max_context_tokens"))
        .or_else(|| val.pointer("/limits/context_length"))
        .or_else(|| val.pointer("/top_provider/context_length"))
        .and_then(|v| v.as_u64())
        .filter(|v| *v > 0)
        .and_then(|v| usize::try_from(v).ok());

    let max_output_tokens = val
        .pointer("/top_provider/max_completion_tokens")
        .or_else(|| val.pointer("/top_provider/max_output_tokens"))
        .or_else(|| val.get("max_output_tokens"))
        .or_else(|| val.get("max_completion_tokens"))
        .or_else(|| val.get("max_tokens"))
        .or_else(|| val.pointer("/limits/max_output_tokens"))
        .or_else(|| val.pointer("/limits/max_completion_tokens"))
        .or_else(|| val.pointer("/limits/max_tokens"))
        .or_else(|| val.pointer("/per_request_limits/max_output_tokens"))
        .and_then(|v| v.as_u64())
        .filter(|v| *v > 0 && *v <= u32::MAX as u64)
        .and_then(|v| usize::try_from(v).ok());

    let (thinking_supported, thinking_levels, thinking_default) = extract_thinking_metadata(val);

    Some(ModelMetadata {
        id,
        thinking_supported,
        thinking_levels,
        thinking_default,
        context_length,
        max_output_tokens,
        context_provenance: None,
    })
}

fn extract_thinking_metadata(
    val: &serde_json::Value,
) -> (Option<bool>, Option<Vec<String>>, Option<String>) {
    let mut supported = None;
    let mut levels = None;
    let mut default = None;

    if let Some(s) = val
        .pointer("/default_parameters/reasoning_effort")
        .or_else(|| val.pointer("/default_parameters/thinking/level"))
        .or_else(|| val.pointer("/default_parameters/thinking_level"))
        .or_else(|| val.pointer("/thinking/default"))
        .or_else(|| val.pointer("/thinking/default_level"))
        .or_else(|| val.get("thinking_default"))
        .or_else(|| val.get("reasoning_effort_default"))
        .and_then(|v| v.as_str())
        && !s.trim().is_empty() {
            default = Some(s.trim().to_string());
        }

    if let Some(lvl_arr) = val
        .pointer("/thinking/levels")
        .or_else(|| val.pointer("/capabilities/thinking/levels"))
        .or_else(|| val.pointer("/capabilities/reasoning_effort"))
        .or_else(|| val.get("thinking_levels"))
        .or_else(|| val.get("reasoning_effort"))
        .or_else(|| val.pointer("/supported_parameters/reasoning_effort"))
        .or_else(|| val.pointer("/parameters/reasoning_effort/enum"))
        .and_then(|v| v.as_array())
    {
        let lvls: Vec<String> = lvl_arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
            .filter(|s| !s.is_empty())
            .collect();
        if !lvls.is_empty() {
            levels = Some(lvls);
        }
    }

    if let Some(t) = val
        .get("thinking")
        .or_else(|| val.pointer("/capabilities/thinking"))
    {
        if let Some(b) = t.as_bool() {
            supported = Some(b);
        } else if let Some(obj) = t.as_object() {
            supported = obj.get("supported").and_then(|v| v.as_bool());
        }
    }

    if supported.is_none()
        && let Some(b) = val.get("thinking_supported").and_then(|v| v.as_bool()) {
            supported = Some(b);
        }

    if supported.is_none() && (levels.is_some() || default.is_some()) {
        supported = Some(true);
    }

    (supported, levels, default)
}

/// Canonical turn components produced by the wire parsers.
type ParsedTurn = (
    String,
    Option<String>,
    Vec<ToolCallSpec>,
    serde_json::Value,
    String,
);
type ParsedTurnResult = Result<ParsedTurn, StructuredError>;

/// Bounded chunk emitter shared by the streaming parsers.
fn emit_bounded_delta(
    delta: InferenceDelta<'_>,
    on_delta: &mut dyn FnMut(InferenceDelta<'_>) -> Result<(), StructuredError>,
) -> Result<(), StructuredError> {
    const MAX_CHUNK: usize = 8192;
    let (raw, is_thinking) = match delta {
        InferenceDelta::Text(s) => (s, false),
        InferenceDelta::Thinking(s) => (s, true),
    };
    if raw.is_empty() {
        return Ok(());
    }
    let mut remaining = raw;
    while !remaining.is_empty() {
        let chunk = if remaining.len() <= MAX_CHUNK {
            remaining
        } else {
            let mut end = MAX_CHUNK;
            while end > 0 && !remaining.is_char_boundary(end) {
                end -= 1;
            }
            if end == 0 {
                end = remaining
                    .chars()
                    .next()
                    .map(|c| c.len_utf8())
                    .unwrap_or(remaining.len());
            }
            &remaining[..end]
        };
        let d = if is_thinking {
            InferenceDelta::Thinking(chunk)
        } else {
            InferenceDelta::Text(chunk)
        };
        on_delta(d)?;
        remaining = &remaining[chunk.len()..];
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
struct InlineThinkingParser {
    in_think: bool,
    buffer: String,
}

impl InlineThinkingParser {
    fn feed(
        &mut self,
        chunk: &str,
        on_delta: &mut dyn FnMut(InferenceDelta<'_>) -> Result<(), StructuredError>,
    ) -> Result<(), StructuredError> {
        self.buffer.push_str(chunk);

        loop {
            if !self.in_think {
                let found_open = self
                    .buffer
                    .find("<think>")
                    .map(|idx| (idx, "<think>"))
                    .or_else(|| self.buffer.find("<thought>").map(|idx| (idx, "<thought>")));

                if let Some((pos, tag)) = found_open {
                    if pos > 0 {
                        emit_bounded_delta(InferenceDelta::Text(&self.buffer[..pos]), on_delta)?;
                    }
                    self.buffer.drain(..pos + tag.len());
                    self.in_think = true;
                } else {
                    let mut prefix_start = None;
                    if let Some(actual_idx) = self.buffer.rfind('<') {
                        let suffix = &self.buffer[actual_idx..];
                        if "<think>".starts_with(suffix) || "<thought>".starts_with(suffix) {
                            prefix_start = Some(actual_idx);
                        }
                    }

                    if let Some(idx) = prefix_start {
                        if idx > 0 {
                            emit_bounded_delta(
                                InferenceDelta::Text(&self.buffer[..idx]),
                                on_delta,
                            )?;
                            self.buffer.drain(..idx);
                        }
                        break;
                    } else {
                        if !self.buffer.is_empty() {
                            emit_bounded_delta(InferenceDelta::Text(&self.buffer), on_delta)?;
                            self.buffer.clear();
                        }
                        break;
                    }
                }
            } else {
                let found_close = self
                    .buffer
                    .find("</think>")
                    .map(|idx| (idx, "</think>"))
                    .or_else(|| {
                        self.buffer
                            .find("</thought>")
                            .map(|idx| (idx, "</thought>"))
                    });

                if let Some((pos, tag)) = found_close {
                    if pos > 0 {
                        emit_bounded_delta(
                            InferenceDelta::Thinking(&self.buffer[..pos]),
                            on_delta,
                        )?;
                    }
                    self.buffer.drain(..pos + tag.len());
                    self.in_think = false;
                } else {
                    let mut prefix_start = None;
                    if let Some(actual_idx) = self.buffer.rfind('<') {
                        let suffix = &self.buffer[actual_idx..];
                        if "</think>".starts_with(suffix) || "</thought>".starts_with(suffix) {
                            prefix_start = Some(actual_idx);
                        }
                    }

                    if let Some(idx) = prefix_start {
                        if idx > 0 {
                            emit_bounded_delta(
                                InferenceDelta::Thinking(&self.buffer[..idx]),
                                on_delta,
                            )?;
                            self.buffer.drain(..idx);
                        }
                        break;
                    } else {
                        if !self.buffer.is_empty() {
                            emit_bounded_delta(InferenceDelta::Thinking(&self.buffer), on_delta)?;
                            self.buffer.clear();
                        }
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    fn flush(
        &mut self,
        on_delta: &mut dyn FnMut(InferenceDelta<'_>) -> Result<(), StructuredError>,
    ) -> Result<(), StructuredError> {
        if !self.buffer.is_empty() {
            if self.in_think {
                emit_bounded_delta(InferenceDelta::Thinking(&self.buffer), on_delta)?;
            } else {
                emit_bounded_delta(InferenceDelta::Text(&self.buffer), on_delta)?;
            }
            self.buffer.clear();
        }
        Ok(())
    }
}

/// One transport for HTTP and HTTPS, with an overall deadline and bounded body readers.
fn execute_wire_request(
    req: &ProviderRequest,
    timeout_secs: u64,
    max_response_bytes: usize,
    on_delta: &mut dyn FnMut(InferenceDelta<'_>) -> Result<(), StructuredError>,
) -> ParsedTurnResult {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(CONNECT_TIMEOUT_SECS))
        .timeout_read(Duration::from_secs(
            timeout_secs.clamp(MIN_REQUEST_TIMEOUT_SECS, MAX_REQUEST_TIMEOUT_SECS),
        ))
        .timeout_write(Duration::from_secs(
            timeout_secs.clamp(MIN_REQUEST_TIMEOUT_SECS, MAX_REQUEST_TIMEOUT_SECS),
        ))
        .redirects(0)
        .build();
    let mut request = agent.post(&req.endpoint);
    for (name, value) in &req.headers {
        request = request.set(name, value);
    }
    let response = request.send_json(&req.body).map_err(|error| match error {
        ureq::Error::Status(code, _response) => StructuredError::new(
            "http_status_error",
            format!("HTTP {code}"),
            code == 429 || code >= 500,
        ),
        ureq::Error::Transport(error) => {
            StructuredError::new("network_error", error.to_string(), true)
        }
    })?;
    let streaming = response
        .header("content-type")
        .is_some_and(|value| value.starts_with("text/event-stream"));
    let mut reader = BufReader::new(response.into_reader());
    if streaming {
        parse_sse_stream(&mut reader, req.wire_format, max_response_bytes, on_delta)
    } else {
        parse_json_stream(&mut reader, req.wire_format, max_response_bytes)
    }
}

/// Parses a bounded line-delimited SSE stream into canonical turn components.
#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_sse_stream<R: BufRead>(
    reader: &mut R,
    wire_format: WireFormat,
    max_response_bytes: usize,
    on_delta: &mut dyn FnMut(InferenceDelta<'_>) -> Result<(), StructuredError>,
) -> ParsedTurnResult {
    let mut total_bytes = 0;
    let mut text_acc = String::new();
    let mut thinking_acc: Option<String> = None;
    let mut tool_calls_acc: Vec<StreamingToolCall> = Vec::new();
    let mut usage_val = serde_json::json!({});
    let mut finish_reason: Option<String> = None;
    let mut seen_terminal = false;
    let mut inline_parser = InlineThinkingParser::default();

    let mut line_buf = String::new();
    let mut current_event_type = String::new();

    loop {
        line_buf.clear();
        let remaining = max_response_bytes
            .saturating_sub(total_bytes)
            .saturating_add(1) as u64;
        let bytes_read = reader
            .by_ref()
            .take(remaining)
            .read_line(&mut line_buf)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::TimedOut {
                    StructuredError::new(
                        "stream_stalled",
                        format!(
                            "The provider sent nothing for the whole request budget ({e}); the model may still be reasoning, or the gateway stopped forwarding. Raise ai_request_timeout_secs if this model needs longer pauses."
                        ),
                        true,
                    )
                } else {
                    StructuredError::new(
                        "stream_read_failed",
                        format!("Error reading SSE stream: {e}"),
                        true,
                    )
                }
            })?;
        if bytes_read == 0 {
            break;
        }

        total_bytes += bytes_read;
        if total_bytes > max_response_bytes {
            return Err(StructuredError::new(
                "response_limit_exceeded",
                format!("Stream exceeded maximum configured limit of {max_response_bytes} bytes"),
                false,
            ));
        }

        let trimmed = line_buf.trim();
        if trimmed.is_empty() || trimmed.starts_with(':') {
            continue;
        }

        if let Some(event) = trimmed.strip_prefix("event:") {
            current_event_type = event.trim().to_string();
            if current_event_type == "message_stop" {
                seen_terminal = true;
            }
            continue;
        }

        if let Some(data_str) = trimmed.strip_prefix("data:") {
            let payload = data_str.trim();
            if payload.is_empty() {
                continue;
            }
            if payload == "[DONE]" {
                seen_terminal = true;
                break;
            }

            let val: serde_json::Value = serde_json::from_str(payload).map_err(|e| {
                StructuredError::new(
                    "malformed_sse_data",
                    format!("Malformed SSE JSON payload: {e}. Payload: '{payload}'"),
                    false,
                )
            })?;

            // Check for explicit provider stream errors
            if current_event_type == "error"
                || val.get("error").is_some()
                || val.get("type").and_then(|t| t.as_str()) == Some("error")
            {
                let err_obj = val.get("error").unwrap_or(&val);
                let msg = err_obj
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("Provider stream error");
                let code = err_obj
                    .get("type")
                    .or_else(|| err_obj.get("code"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("provider_stream_error");
                return Err(StructuredError::new(
                    code,
                    format!("Provider stream error: {msg}"),
                    false,
                ));
            }

            match wire_format {
                WireFormat::AnthropicMessages => {
                    process_anthropic_sse_event(
                        &current_event_type,
                        &val,
                        &mut text_acc,
                        &mut thinking_acc,
                        &mut tool_calls_acc,
                        &mut usage_val,
                        &mut finish_reason,
                        &mut seen_terminal,
                        &mut inline_parser,
                        on_delta,
                    )?;
                }
                _ => {
                    process_openai_sse_chunk(
                        &val,
                        &mut text_acc,
                        &mut thinking_acc,
                        &mut tool_calls_acc,
                        &mut usage_val,
                        &mut finish_reason,
                        &mut inline_parser,
                        on_delta,
                    )?;
                }
            }
        } else if trimmed.starts_with('{') && trimmed.ends_with('}') {
            let val: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
                StructuredError::new(
                    "malformed_sse_data",
                    format!("Malformed JSON line in SSE stream: {e}. Line: '{trimmed}'"),
                    false,
                )
            })?;

            if val.get("error").is_some()
                || val.get("type").and_then(|t| t.as_str()) == Some("error")
            {
                let err_obj = val.get("error").unwrap_or(&val);
                let msg = err_obj
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("Provider stream error");
                let code = err_obj
                    .get("type")
                    .or_else(|| err_obj.get("code"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("provider_stream_error");
                return Err(StructuredError::new(
                    code,
                    format!("Provider stream error: {msg}"),
                    false,
                ));
            }

            match wire_format {
                WireFormat::AnthropicMessages => {
                    process_anthropic_sse_event(
                        &current_event_type,
                        &val,
                        &mut text_acc,
                        &mut thinking_acc,
                        &mut tool_calls_acc,
                        &mut usage_val,
                        &mut finish_reason,
                        &mut seen_terminal,
                        &mut inline_parser,
                        on_delta,
                    )?;
                }
                _ => {
                    process_openai_sse_chunk(
                        &val,
                        &mut text_acc,
                        &mut thinking_acc,
                        &mut tool_calls_acc,
                        &mut usage_val,
                        &mut finish_reason,
                        &mut inline_parser,
                        on_delta,
                    )?;
                }
            }
        }
    }

    if !seen_terminal && finish_reason.is_none() {
        return Err(StructuredError::new(
            "premature_stream_termination",
            "Stream terminated prematurely without completion signal or finish reason",
            true,
        ));
    }

    if !tool_calls_acc.is_empty()
        && matches!(finish_reason.as_deref(), Some("length" | "max_tokens"))
    {
        return Err(StructuredError::new(
            "partial_tool_arguments",
            "Output limit interrupted tool arguments; no tool was executed",
            true,
        ));
    }
    inline_parser.flush(on_delta)?;

    let final_tools = tool_calls_acc
        .into_iter()
        .map(|t| {
            let id =
                t.id.map(ToolCallId::new)
                    .transpose()?
                    .unwrap_or_else(ToolCallId::mint);
            let name = t.name.unwrap_or_else(|| "unknown_tool".into());
            let arguments = if t.arguments_buffer.trim().is_empty() {
                serde_json::json!({})
            } else {
                // A completed stream may still need the existing bounded dialect repair.
                serde_json::from_str::<serde_json::Value>(&t.arguments_buffer)
                    .unwrap_or(serde_json::Value::String(t.arguments_buffer))
            };
            Ok(ToolCallSpec {
                id,
                name,
                arguments,
            })
        })
        .collect::<Result<Vec<_>, StructuredError>>()?;

    let final_finish_reason = finish_reason.unwrap_or_else(|| "stop".to_string());

    Ok((
        text_acc,
        thinking_acc,
        final_tools,
        usage_val,
        final_finish_reason,
    ))
}

/// Parses a non-SSE JSON response payload.
fn parse_json_stream<R: BufRead>(
    reader: &mut R,
    wire_format: WireFormat,
    max_response_bytes: usize,
) -> ParsedTurnResult {
    let mut body = String::new();
    reader
        .take(max_response_bytes as u64)
        .read_to_string(&mut body)
        .map_err(|e| {
            StructuredError::new(
                "read_failed",
                format!("Failed to read response body: {e}"),
                true,
            )
        })?;

    let val: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
        StructuredError::new(
            "json_parse_error",
            format!("Failed to parse response JSON: {e}. Body: '{body}'"),
            true,
        )
    })?;

    let mut text = String::new();
    let mut thinking = None;
    let mut tools = Vec::new();
    let mut usage = serde_json::json!({});
    let mut finish_reason = "stop".to_string();

    match wire_format {
        WireFormat::AnthropicMessages => {
            process_anthropic_json(
                &val,
                &mut text,
                &mut thinking,
                &mut tools,
                &mut usage,
                &mut finish_reason,
            );
        }
        _ => {
            process_openai_json(
                &val,
                &mut text,
                &mut thinking,
                &mut tools,
                &mut usage,
                &mut finish_reason,
            );
        }
    }

    let final_tools = tools
        .into_iter()
        .map(|t| {
            let id =
                t.id.map(ToolCallId::new)
                    .transpose()?
                    .unwrap_or_else(ToolCallId::mint);
            let name = t.name.unwrap_or_else(|| "unknown_tool".into());
            let arguments = serde_json::from_str::<serde_json::Value>(&t.arguments_buffer)
                .unwrap_or(serde_json::Value::String(t.arguments_buffer));
            Ok(ToolCallSpec {
                id,
                name,
                arguments,
            })
        })
        .collect::<Result<Vec<_>, StructuredError>>()?;

    Ok((text, thinking, final_tools, usage, finish_reason))
}

#[allow(clippy::too_many_arguments)]
fn process_openai_sse_chunk(
    val: &serde_json::Value,
    text_acc: &mut String,
    thinking_acc: &mut Option<String>,
    tool_calls_acc: &mut Vec<StreamingToolCall>,
    usage_val: &mut serde_json::Value,
    finish_reason: &mut Option<String>,
    inline_parser: &mut InlineThinkingParser,
    on_delta: &mut dyn FnMut(InferenceDelta<'_>) -> Result<(), StructuredError>,
) -> Result<(), StructuredError> {
    if let Some(choices) = val.get("choices").and_then(|c| c.as_array()) {
        for choice in choices {
            if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str())
                && !reason.is_empty() && reason != "null" {
                    *finish_reason = Some(reason.to_string());
                }

            let delta_opt = choice.get("delta");
            let message_opt = choice.get("message");

            // Extract thinking / reasoning from aliases
            let reasoning_opt = delta_opt
                .and_then(|d| {
                    d.get("reasoning_content")
                        .or_else(|| d.get("reasoning"))
                        .or_else(|| d.get("thinking"))
                })
                .or_else(|| {
                    message_opt.and_then(|m| {
                        m.get("reasoning_content")
                            .or_else(|| m.get("reasoning"))
                            .or_else(|| m.get("thinking"))
                    })
                })
                .or_else(|| {
                    choice
                        .get("reasoning_content")
                        .or_else(|| choice.get("reasoning"))
                        .or_else(|| choice.get("thinking"))
                })
                .and_then(|r| r.as_str());

            if let Some(reasoning) = reasoning_opt {
                let th = thinking_acc.get_or_insert_with(String::new);
                th.push_str(reasoning);
                emit_bounded_delta(InferenceDelta::Thinking(reasoning), on_delta)?;
            }

            // Extract content
            let content_opt = delta_opt
                .and_then(|d| d.get("content"))
                .or_else(|| message_opt.and_then(|m| m.get("content")))
                .and_then(|c| c.as_str());

            if let Some(content) = content_opt {
                text_acc.push_str(content);
                inline_parser.feed(content, on_delta)?;
            }

            // Extract tool calls
            let tool_calls_opt = delta_opt
                .and_then(|d| d.get("tool_calls"))
                .or_else(|| message_opt.and_then(|m| m.get("tool_calls")))
                .and_then(|t| t.as_array());

            if let Some(tcs) = tool_calls_opt {
                const MAX_STREAMING_TOOLS: usize = 256;
                for tc in tcs {
                    let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                    if index >= MAX_STREAMING_TOOLS as u64 {
                        return Err(StructuredError::new(
                            "stream_tool_limit",
                            "Tool call index exceeds stream limit",
                            false,
                        ));
                    }
                    let idx = index as usize;
                    while tool_calls_acc.len() <= idx {
                        tool_calls_acc.push(StreamingToolCall::default());
                    }

                    if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                        tool_calls_acc[idx].id = Some(id.to_string());
                    }

                    if let Some(func) = tc.get("function") {
                        if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                            tool_calls_acc[idx].name = Some(name.to_string());
                        }
                        if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                            tool_calls_acc[idx].arguments_buffer.push_str(args);
                        }
                    }
                }
            }
        }
    }

    if let Some(u) = val.get("usage") {
        *usage_val = u.clone();
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_anthropic_sse_event(
    event_type: &str,
    val: &serde_json::Value,
    text_acc: &mut String,
    thinking_acc: &mut Option<String>,
    tool_calls_acc: &mut Vec<StreamingToolCall>,
    usage_val: &mut serde_json::Value,
    finish_reason: &mut Option<String>,
    seen_terminal: &mut bool,
    inline_parser: &mut InlineThinkingParser,
    on_delta: &mut dyn FnMut(InferenceDelta<'_>) -> Result<(), StructuredError>,
) -> Result<(), StructuredError> {
    match event_type {
        "content_block_start" => {
            if let Some(cb) = val.get("content_block") {
                let block_type = cb.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if block_type == "tool_use" {
                    const MAX_STREAMING_TOOLS: usize = 256;
                    if tool_calls_acc.len() >= MAX_STREAMING_TOOLS {
                        return Err(StructuredError::new(
                            "stream_tool_limit",
                            "Tool call count exceeds stream limit",
                            false,
                        ));
                    }
                    if tool_calls_acc.len() < MAX_STREAMING_TOOLS {
                        let mut stc = StreamingToolCall::default();
                        if let Some(id) = cb.get("id").and_then(|i| i.as_str()) {
                            stc.id = Some(id.to_string());
                        }
                        if let Some(name) = cb.get("name").and_then(|n| n.as_str()) {
                            stc.name = Some(name.to_string());
                        }
                        tool_calls_acc.push(stc);
                    }
                }
                if block_type == "text" {
                    if let Some(text) = cb.get("text").and_then(|value| value.as_str()) {
                        text_acc.push_str(text);
                        inline_parser.feed(text, on_delta)?;
                    }
                } else if block_type == "thinking"
                    && let Some(text) = cb.get("thinking").and_then(|value| value.as_str()) {
                        thinking_acc.get_or_insert_with(String::new).push_str(text);
                        emit_bounded_delta(InferenceDelta::Thinking(text), on_delta)?;
                    }
            }
        }
        "content_block_delta" => {
            if let Some(delta) = val.get("delta") {
                let delta_type = delta.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match delta_type {
                    "text_delta" => {
                        if let Some(t) = delta.get("text").and_then(|s| s.as_str()) {
                            text_acc.push_str(t);
                            inline_parser.feed(t, on_delta)?;
                        }
                    }
                    "thinking_delta" | "reasoning_delta" => {
                        if let Some(t) = delta
                            .get("thinking")
                            .or_else(|| delta.get("reasoning_content"))
                            .or_else(|| delta.get("reasoning"))
                            .and_then(|s| s.as_str())
                        {
                            let th = thinking_acc.get_or_insert_with(String::new);
                            th.push_str(t);
                            emit_bounded_delta(InferenceDelta::Thinking(t), on_delta)?;
                        }
                    }
                    "input_json_delta" => {
                        if let Some(partial) = delta.get("partial_json").and_then(|p| p.as_str())
                            && let Some(last) = tool_calls_acc.last_mut() {
                                last.arguments_buffer.push_str(partial);
                            }
                    }
                    _ => {}
                }
            }
        }
        "message_delta" => {
            if let Some(delta) = val.get("delta")
                && let Some(stop) = delta.get("stop_reason").and_then(|s| s.as_str()) {
                    *finish_reason = Some(stop.to_string());
                }
            if let Some(u) = val.get("usage") {
                *usage_val = u.clone();
            }
        }
        "message_stop" => {
            *seen_terminal = true;
        }
        "message_start" => {
            if let Some(msg) = val.get("message")
                && let Some(u) = msg.get("usage") {
                    *usage_val = u.clone();
                }
        }
        _ => {}
    }
    Ok(())
}

fn process_openai_json(
    val: &serde_json::Value,
    text_acc: &mut String,
    thinking_acc: &mut Option<String>,
    tool_calls_acc: &mut Vec<StreamingToolCall>,
    usage_val: &mut serde_json::Value,
    finish_reason: &mut String,
) {
    if let Some(choice) = val.get("choices").and_then(|c| c.get(0)) {
        if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
            *finish_reason = reason.to_string();
        }

        if let Some(msg) = choice.get("message") {
            if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                text_acc.push_str(content);
            }

            if let Some(reasoning) = msg
                .get("reasoning_content")
                .or_else(|| msg.get("reasoning"))
                .or_else(|| msg.get("thinking"))
                .or_else(|| choice.get("reasoning_content"))
                .or_else(|| choice.get("reasoning"))
                .or_else(|| choice.get("thinking"))
                .and_then(|r| r.as_str())
            {
                *thinking_acc = Some(reasoning.to_string());
            }

            if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                for tc in tcs {
                    let mut stc = StreamingToolCall::default();
                    if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                        stc.id = Some(id.to_string());
                    }
                    if let Some(func) = tc.get("function") {
                        if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                            stc.name = Some(name.to_string());
                        }
                        if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                            stc.arguments_buffer = args.to_string();
                        }
                    }
                    tool_calls_acc.push(stc);
                }
            }
        }
    }

    if let Some(u) = val.get("usage") {
        *usage_val = u.clone();
    }
}

fn process_anthropic_json(
    val: &serde_json::Value,
    text_acc: &mut String,
    thinking_acc: &mut Option<String>,
    tool_calls_acc: &mut Vec<StreamingToolCall>,
    usage_val: &mut serde_json::Value,
    finish_reason: &mut String,
) {
    if let Some(stop) = val.get("stop_reason").and_then(|s| s.as_str()) {
        *finish_reason = stop.to_string();
    }

    if let Some(content_blocks) = val.get("content").and_then(|c| c.as_array()) {
        for block in content_blocks {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
            match block_type {
                "text" => {
                    if let Some(t) = block.get("text").and_then(|s| s.as_str()) {
                        text_acc.push_str(t);
                    }
                }
                "thinking" | "reasoning" => {
                    if let Some(t) = block
                        .get("thinking")
                        .or_else(|| block.get("reasoning"))
                        .or_else(|| block.get("reasoning_content"))
                        .and_then(|s| s.as_str())
                    {
                        *thinking_acc = Some(t.to_string());
                    }
                }
                "tool_use" => {
                    let mut stc = StreamingToolCall::default();
                    if let Some(id) = block.get("id").and_then(|i| i.as_str()) {
                        stc.id = Some(id.to_string());
                    }
                    if let Some(name) = block.get("name").and_then(|n| n.as_str()) {
                        stc.name = Some(name.to_string());
                    }
                    if let Some(input) = block.get("input") {
                        stc.arguments_buffer = input.to_string();
                    }
                    tool_calls_acc.push(stc);
                }
                _ => {}
            }
        }
    }

    if let Some(u) = val.get("usage") {
        *usage_val = u.clone();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    #[test]
    fn test_canonical_turn_serde() {
        let turn = CanonicalTurn {
            text: "Hello world".into(),
            thinking: Some("Let me think".into()),
            tool_calls: vec![ToolCallSpec {
                id: ToolCallId::new("call_1").unwrap(),
                name: "Read".into(),
                arguments: serde_json::json!({ "path": "test.txt" }),
            }],
            usage: serde_json::json!({ "prompt_tokens": 10, "completion_tokens": 20 }),
            finish_reason: "stop".into(),
        };

        let serialized = serde_json::to_string(&turn).unwrap();
        let deserialized: CanonicalTurn = serde_json::from_str(&serialized).unwrap();
        assert_eq!(turn, deserialized);
    }

    #[test]
    fn test_provider_client_endpoint_only_no_model() {
        // Endpoint-only configuration with no model is accepted for catalog lookup
        let client = ProviderClient::new(
            "openai_compatible",
            "",
            Some("http://127.0.0.1:8000/v1".into()),
        )
        .unwrap();
        assert_eq!(client.provider, "openai_compatible");
        assert_eq!(client.model, "");
        assert!(!client.is_configured());

        // Inference without model must fail with missing_model
        let mut client_mut = client;
        let req = InferenceRequest::default();
        let err = client_mut.infer(&req, &mut |_| Ok(())).unwrap_err();
        assert_eq!(err.code, "missing_model");
    }

    #[test]
    fn test_count_tokens_endpoint_composition() {
        // Default (no configured endpoint)
        assert_eq!(
            count_tokens_endpoint(None),
            "https://api.anthropic.com/v1/messages/count_tokens"
        );
        // Bare host
        assert_eq!(
            count_tokens_endpoint(Some("https://api.anthropic.com")),
            "https://api.anthropic.com/v1/messages/count_tokens"
        );
        // /v1 base
        assert_eq!(
            count_tokens_endpoint(Some("https://api.anthropic.com/v1")),
            "https://api.anthropic.com/v1/messages/count_tokens"
        );
        // Full /v1/messages endpoint (previously produced a bare /count_tokens)
        assert_eq!(
            count_tokens_endpoint(Some("https://api.anthropic.com/v1/messages")),
            "https://api.anthropic.com/v1/messages/count_tokens"
        );
        // Trailing slash
        assert_eq!(
            count_tokens_endpoint(Some("https://api.anthropic.com/v1/")),
            "https://api.anthropic.com/v1/messages/count_tokens"
        );
    }

    #[test]
    fn test_endpoint_url_validation() {
        assert!(validate_endpoint_url("https://api.openai.com/v1").is_ok());
        assert!(validate_endpoint_url("http://localhost:11434").is_ok());
        assert!(validate_endpoint_url("http://127.0.0.1:8000/v1").is_ok());
        assert!(validate_endpoint_url("http://[::1]:8000/v1").is_ok());

        // Remote HTTP is insecure and rejected
        let err_remote = validate_endpoint_url("http://api.openai.com/v1").unwrap_err();
        assert_eq!(err_remote.code, "insecure_endpoint");

        // Userinfo credentials in URL rejected
        let err_userinfo =
            validate_endpoint_url("https://user:pass@api.openai.com/v1").unwrap_err();
        assert_eq!(err_userinfo.code, "credential_in_url");

        // Fragments rejected
        let err_frag = validate_endpoint_url("https://api.openai.com/v1#section").unwrap_err();
        assert_eq!(err_frag.code, "fragment_in_url");

        // Query credentials rejected
        let err_key =
            validate_endpoint_url("https://api.openai.com/v1?api_key=secret").unwrap_err();
        assert_eq!(err_key.code, "credential_in_url");
    }

    #[test]
    fn test_catalog_parsing_documented_fields() {
        let json_val = serde_json::json!({
            "data": [
                {
                    "id": "anthropic/claude-3.7-sonnet",
                    "context_length": 200000,
                    "top_provider": {
                        "max_completion_tokens": 64000
                    },
                    "default_parameters": {
                        "reasoning_effort": "medium"
                    },
                    "thinking": {
                        "supported": true,
                        "levels": ["low", "medium", "high"]
                    }
                },
                {
                    "id": "minimal-model"
                }
            ]
        });

        let catalog = parse_models_catalog(&json_val).unwrap();
        assert_eq!(catalog.len(), 2);

        let claude = &catalog[0];
        assert_eq!(claude.id, "anthropic/claude-3.7-sonnet");
        assert_eq!(claude.context_length, Some(200000));
        assert_eq!(claude.max_output_tokens, Some(64000));
        assert_eq!(claude.thinking_supported, Some(true));
        assert_eq!(claude.thinking_default.as_deref(), Some("medium"));
        assert_eq!(
            claude.thinking_levels,
            Some(vec![
                "low".to_string(),
                "medium".to_string(),
                "high".to_string()
            ])
        );

        let minimal = &catalog[1];
        assert_eq!(minimal.id, "minimal-model");
        assert_eq!(minimal.context_length, None);
        assert_eq!(minimal.max_output_tokens, None);
        assert_eq!(minimal.thinking_supported, None);
        assert_eq!(minimal.thinking_default, None);
        assert_eq!(minimal.thinking_levels, None);
    }

    #[test]
    fn test_catalog_rejects_malformed_response() {
        // Malformed catalog (no data or models array)
        let malformed = serde_json::json!({ "error": "not found" });
        let err_malformed = parse_models_catalog(&malformed).unwrap_err();
        assert_eq!(err_malformed.code, "malformed_catalog");
    }

    #[test]
    fn test_openai_sse_parsing() {
        let sse_data = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello \",\"reasoning_content\":\"Thinking step 1... \"}}]}\n\n\
                        data: {\"choices\":[{\"delta\":{\"content\":\"world!\",\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"function\":{\"name\":\"Bash\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}}]}}]}\n\n\
                        data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}],\"usage\":{\"total_tokens\":42}}\n\n\
                        data: [DONE]\n\n";

        let mut cursor = Cursor::new(sse_data.as_bytes());
        let (text, thinking, tool_calls, usage, finish_reason) = parse_sse_stream(
            &mut cursor,
            WireFormat::OpenAiChatCompletions,
            1024 * 1024,
            &mut |_| Ok(()),
        )
        .unwrap();

        assert_eq!(text, "Hello world!");
        assert_eq!(thinking, Some("Thinking step 1... ".into()));
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "Bash");
        assert_eq!(tool_calls[0].arguments["command"], "ls");
        assert_eq!(finish_reason, "tool_calls");
        assert_eq!(usage["total_tokens"], 42);
    }

    #[test]
    fn test_anthropic_sse_parsing() {
        let sse_data = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":25}}}\n\n\
                        event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\"}}\n\n\
                        event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Reasoning...\"}}\n\n\
                        event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
                        event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Output text\"}}\n\n\
                        event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_ant_1\",\"name\":\"Read\"}}\n\n\
                        event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\": \\\"src/lib.rs\\\"}\"}}\n\n\
                        event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":50}}\n\n\
                        event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

        let mut cursor = Cursor::new(sse_data.as_bytes());
        let (text, thinking, tool_calls, usage, finish_reason) = parse_sse_stream(
            &mut cursor,
            WireFormat::AnthropicMessages,
            1024 * 1024,
            &mut |_| Ok(()),
        )
        .unwrap();

        assert_eq!(text, "Output text");
        assert_eq!(thinking, Some("Reasoning...".into()));
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "Read");
        assert_eq!(tool_calls[0].arguments["path"], "src/lib.rs");
        assert_eq!(finish_reason, "tool_use");
        assert_eq!(usage["output_tokens"], 50);
    }

    #[test]
    fn test_stream_limit_enforcement() {
        let huge_chunk = "data: {\"choices\":[{\"delta\":{\"content\":\"".to_string()
            + &"A".repeat(2000)
            + "\"}}]}\n\n";
        let mut cursor = Cursor::new(huge_chunk.as_bytes());
        let err = parse_sse_stream(
            &mut cursor,
            WireFormat::OpenAiChatCompletions,
            500,
            &mut |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(err.code, "response_limit_exceeded");
    }

    #[test]
    fn test_early_callback_before_eof() {
        use std::cell::Cell;
        struct GatedReader<'a> {
            first: &'a [u8],
            rest: Cursor<&'a [u8]>,
            delivered: &'a Cell<bool>,
        }
        impl Read for GatedReader<'_> {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                if !self.first.is_empty() {
                    return self.first.read(buffer);
                }
                assert!(
                    self.delivered.get(),
                    "parser requested the rest before delivering thinking"
                );
                self.rest.read(buffer)
            }
        }
        let delivered = Cell::new(false);
        let mut reader = BufReader::new(GatedReader {
            first:
                b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"First thought\"}}]}\n\n",
            rest: Cursor::new(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"Answer\"}}]}\n\ndata: [DONE]\n\n"
                    .as_slice(),
            ),
            delivered: &delivered,
        });
        let mut answer = String::new();
        let (text, thinking, _, _, _) = parse_sse_stream(
            &mut reader,
            WireFormat::OpenAiChatCompletions,
            1024 * 1024,
            &mut |delta| {
                match delta {
                    InferenceDelta::Thinking(value) => {
                        assert_eq!(value, "First thought");
                        delivered.set(true);
                    }
                    InferenceDelta::Text(value) => answer.push_str(value),
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(answer, "Answer");
        assert_eq!(text, answer);
        assert_eq!(thinking.as_deref(), Some("First thought"));
    }

    #[test]
    fn test_utf8_chunking_bounded_slices() {
        let content = format!("{}ı", "日本語abc".repeat(1000));
        let sse_data = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}}}}]}}\n\ndata: [DONE]\n\n",
            serde_json::to_string(&content).unwrap()
        );
        let mut cursor = Cursor::new(sse_data.as_bytes());
        let mut delivered = String::new();
        let (text, _, _, _, _) = parse_sse_stream(
            &mut cursor,
            WireFormat::OpenAiChatCompletions,
            1024 * 1024,
            &mut |delta| {
                if let InferenceDelta::Text(chunk) = delta {
                    assert!(chunk.len() <= 8192);
                    delivered.push_str(chunk);
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(delivered, content);
        assert_eq!(text, content);
    }

    #[test]
    fn test_end_of_stream_errors_and_malformed() {
        // 1. Premature stream termination (cut off before terminal signal or finish_reason)
        let truncated_sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Incomplete...\"}}]}\n\n";
        let mut cursor = Cursor::new(truncated_sse.as_bytes());
        let err = parse_sse_stream(
            &mut cursor,
            WireFormat::OpenAiChatCompletions,
            1024 * 1024,
            &mut |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(err.code, "premature_stream_termination");

        // 2. Explicit provider error in stream
        let error_sse =
            "data: {\"error\":{\"message\":\"Engine overloaded\",\"type\":\"server_error\"}}\n\n";
        let mut cursor_err = Cursor::new(error_sse.as_bytes());
        let err2 = parse_sse_stream(
            &mut cursor_err,
            WireFormat::OpenAiChatCompletions,
            1024 * 1024,
            &mut |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(err2.code, "server_error");

        // 3. Malformed SSE data
        let malformed_sse = "data: {not valid json\n\n";
        let mut cursor_mal = Cursor::new(malformed_sse.as_bytes());
        let err3 = parse_sse_stream(
            &mut cursor_mal,
            WireFormat::OpenAiChatCompletions,
            1024 * 1024,
            &mut |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(err3.code, "malformed_sse_data");
    }

    #[test]
    fn test_callback_failure_terminates_inference() {
        let sse_data = "data: {\"choices\":[{\"delta\":{\"content\":\"First chunk\"}}]}\n\n\
                        data: {\"choices\":[{\"delta\":{\"content\":\"Second chunk\"}}]}\n\n\
                        data: [DONE]\n\n";
        let mut cursor = Cursor::new(sse_data.as_bytes());
        let mut calls = 0;
        let err = parse_sse_stream(
            &mut cursor,
            WireFormat::OpenAiChatCompletions,
            1024 * 1024,
            &mut |_| {
                calls += 1;
                Err(StructuredError::new(
                    "client_abort",
                    "User aborted stream",
                    false,
                ))
            },
        )
        .unwrap_err();

        assert_eq!(err.code, "client_abort");
        assert_eq!(calls, 1);
    }

    #[test]
    fn test_inline_think_tag_streaming() {
        let sse_data = "data: {\"choices\":[{\"delta\":{\"content\":\"Prefix <th\"}}]}\n\n\
                        data: {\"choices\":[{\"delta\":{\"content\":\"ink>Reasoning inside</th\"}}]}\n\n\
                        data: {\"choices\":[{\"delta\":{\"content\":\"ink> Suffix\"}}]}\n\n\
                        data: [DONE]\n\n";
        let mut cursor = Cursor::new(sse_data.as_bytes());
        let mut deltas = Vec::new();
        let (text, _thinking, _, _, _) = parse_sse_stream(
            &mut cursor,
            WireFormat::OpenAiChatCompletions,
            1024 * 1024,
            &mut |delta| {
                match delta {
                    InferenceDelta::Text(t) => deltas.push(("text", t.to_string())),
                    InferenceDelta::Thinking(th) => deltas.push(("thinking", th.to_string())),
                }
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(
            deltas,
            vec![
                ("text", "Prefix ".to_string()),
                ("thinking", "Reasoning inside".to_string()),
                ("text", " Suffix".to_string()),
            ]
        );
        assert_eq!(text, "Prefix <think>Reasoning inside</think> Suffix");
        let norm = corrective::normalize_thinking_tokens(&text);
        assert_eq!(norm.thinking, Some("Reasoning inside".into()));
        assert_eq!(norm.content, "Prefix  Suffix");
    }

    #[test]
    fn test_normalize_api_base_routes_and_suffixes() {
        assert_eq!(
            normalize_api_base("https://opencode.ai/zen/go/v1"),
            "https://opencode.ai/zen/go/v1"
        );
        assert_eq!(
            normalize_api_base("https://opencode.ai/zen/go/v1/"),
            "https://opencode.ai/zen/go/v1"
        );
        assert_eq!(
            normalize_api_base("https://opencode.ai/zen/go/v1/models"),
            "https://opencode.ai/zen/go/v1"
        );
        assert_eq!(
            normalize_api_base("https://opencode.ai/zen/go/v1/chat/completions"),
            "https://opencode.ai/zen/go/v1"
        );
        assert_eq!(
            normalize_api_base("https://OPENCODE.AI/zen/go/v1"),
            "https://opencode.ai/zen/go/v1"
        );
        assert_eq!(
            normalize_api_base("https://opencode.ai/zen/v1"),
            "https://opencode.ai/zen/v1"
        );
        // Zen /zen/v1 and Go /zen/go/v1 must not collide
        assert_ne!(
            normalize_api_base("https://opencode.ai/zen/go/v1"),
            normalize_api_base("https://opencode.ai/zen/v1")
        );
        assert_eq!(
            normalize_api_base("https://api.openai.com/v1"),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            normalize_api_base("https://api.openai.com/"),
            "https://api.openai.com/v1"
        );
    }

    #[test]
    fn test_registry_parsing_exact_route_and_model() {
        let registry_json = serde_json::json!({
            "opencode": {
                "api": "https://opencode.ai/zen/v1",
                "models": {
                    "hy3": { "limit": { "context": 128000 } }
                }
            },
            "opencode-go": {
                "api": "https://opencode.ai/zen/go/v1",
                "models": {
                    "hy3": { "limit": { "context": 256000 } },
                    "other": { "limit": { "context": 64000 } }
                }
            },
            "openai": {
                "api": "https://api.openai.com/v1",
                "models": {
                    "gpt-4": { "limit": { "context": 8192 } }
                }
            }
        });
        let bytes = serde_json::to_vec(&registry_json).unwrap();

        // 1. Exact match for opencode-go
        let matched_go = parse_registry_for_endpoint(&bytes, "https://opencode.ai/zen/go/v1")
            .unwrap()
            .expect("should match opencode-go");
        assert_eq!(matched_go.key, "opencode-go");
        assert_eq!(matched_go.models.get("hy3"), Some(&Some(256000)));
        assert_eq!(matched_go.models.get("other"), Some(&Some(64000)));
        assert_eq!(matched_go.models.get("gpt-4"), None);

        // 2. Exact match for opencode (route isolation)
        let matched_zen = parse_registry_for_endpoint(&bytes, "https://opencode.ai/zen/v1")
            .unwrap()
            .expect("should match opencode");
        assert_eq!(matched_zen.key, "opencode");
        assert_eq!(matched_zen.models.get("hy3"), Some(&Some(128000)));

        // 3. Unmatched route returns None
        let unmatched = parse_registry_for_endpoint(&bytes, "https://unmatched.ai/v1").unwrap();
        assert!(unmatched.is_none());
    }

    #[test]
    fn test_registry_enrichment_scenarios() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        fn drain_request(stream: &mut std::net::TcpStream) {
            let mut buf = [0u8; 1024];
            let mut read_bytes = 0;
            while read_bytes < buf.len() {
                match stream.read(&mut buf[read_bytes..]) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        read_bytes += n;
                        if buf[..read_bytes].windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                }
            }
        }
        let client = ProviderClient::new(
            "openai_compatible",
            "hy3",
            Some("https://opencode.ai/zen/go/v1".into()),
        )
        .unwrap();

        // Scenario 1: Precedence of direct limit, enrichment of missing context, and unknown models
        {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let url = format!("http://127.0.0.1:{port}/api.json");

            let server_thread = std::thread::spawn(move || {
                if let Ok((mut stream, _)) = listener.accept() {
                    drain_request(&mut stream);
                    let registry_data = serde_json::json!({
                        "opencode-go": {
                            "api": "https://opencode.ai/zen/go/v1",
                            "models": {
                                "advertised-model": { "limit": { "context": 200000 } },
                                "hy3": { "limit": { "context": 256000 } }
                            }
                        }
                    });
                    let body = serde_json::to_string(&registry_data).unwrap();
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(resp.as_bytes());
                    let _ = stream.flush();
                }
            });

            unsafe {
                std::env::set_var("OMP_REGISTRY_URL", &url);
            }

            let mut m_advertised = ModelMetadata::new("advertised-model");
            m_advertised.context_length = Some(100000);

            let m_hy3 = ModelMetadata::new("hy3");
            let m_unknown = ModelMetadata::new("unknown-model");

            let mut models = vec![m_advertised, m_hy3, m_unknown];
            let prov =
                client.enrich_models_from_registry("https://opencode.ai/zen/go/v1", &mut models);

            unsafe {
                std::env::remove_var("OMP_REGISTRY_URL");
            }
            let _ = server_thread.join();

            assert_eq!(prov.as_deref(), Some("models.dev:opencode-go"));
            // Direct advertised limit maintained (precedence)
            assert_eq!(models[0].context_length, Some(100000));
            assert_eq!(models[0].context_provenance, None);
            // Missing limit enriched
            assert_eq!(models[1].context_length, Some(256000));
            assert_eq!(
                models[1].context_provenance.as_deref(),
                Some("models.dev:opencode-go")
            );
            // Unknown model remains unknown
            assert_eq!(models[2].context_length, None);
            assert_eq!(models[2].context_provenance, None);
        }

        // Scenario 2: Unavailable registry (HTTP 500) fails gracefully without error
        {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let url = format!("http://127.0.0.1:{port}/api.json");

            let server_thread = std::thread::spawn(move || {
                if let Ok((mut stream, _)) = listener.accept() {
                    drain_request(&mut stream);
                    let resp = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    let _ = stream.write_all(resp.as_bytes());
                    let _ = stream.flush();
                }
            });

            unsafe {
                std::env::set_var("OMP_REGISTRY_URL", &url);
            }

            let mut models = vec![ModelMetadata::new("hy3")];
            let prov =
                client.enrich_models_from_registry("https://opencode.ai/zen/go/v1", &mut models);

            unsafe {
                std::env::remove_var("OMP_REGISTRY_URL");
            }
            let _ = server_thread.join();

            assert_eq!(prov, None);
            assert_eq!(models[0].context_length, None);
        }

        // Scenario 3: Oversized registry (>8 MiB) fails gracefully without error
        {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let url = format!("http://127.0.0.1:{port}/api.json");

            let server_thread = std::thread::spawn(move || {
                if let Ok((mut stream, _)) = listener.accept() {
                    drain_request(&mut stream);
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        9 * 1024 * 1024
                    );
                    let _ = stream.write_all(header.as_bytes());
                    let chunk = vec![b' '; 64 * 1024];
                    for _ in 0..(9 * 1024 * 1024 / chunk.len()) {
                        if stream.write_all(&chunk).is_err() {
                            break;
                        }
                    }
                    let _ = stream.flush();
                }
            });

            unsafe {
                std::env::set_var("OMP_REGISTRY_URL", &url);
            }

            let mut models = vec![ModelMetadata::new("hy3")];
            let prov =
                client.enrich_models_from_registry("https://opencode.ai/zen/go/v1", &mut models);

            unsafe {
                std::env::remove_var("OMP_REGISTRY_URL");
            }
            let _ = server_thread.join();

            assert_eq!(prov, None);
            assert_eq!(models[0].context_length, None);
        }

        // Scenario 4: Local HTTP service is skipped without OMP_REGISTRY_URL
        {
            let mut models = vec![ModelMetadata::new("hy3")];
            let prov = client.enrich_models_from_registry("http://127.0.0.1:8000/v1", &mut models);
            assert_eq!(prov, None);
            assert_eq!(models[0].context_length, None);
        }
    }

    #[test]
    fn client_identity_headers_follow_the_endpoint() {
        let mut wire = ProviderRequest {
            endpoint: "https://opencode.ai/zen/go/v1/chat/completions".into(),
            headers: Vec::new(),
            body: serde_json::json!({}),
            native_tool_choice_used: false,
            soft_instruction_injected: false,
            wire_format: WireFormat::OpenAiChatCompletions,
        };

        let client = ProviderClient::new(
            "openai_compatible",
            "glm-5.3-flash",
            Some("https://opencode.ai/zen/go/v1".into()),
        )
        .unwrap()
        .with_conversation_id("session-abc");
        client.apply_client_identity(&mut wire);
        let header = |name: &str| {
            wire.headers
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
        };
        assert_eq!(header("user-agent").as_deref(), Some(CLIENT_USER_AGENT));
        assert_eq!(header("x-opencode-session").as_deref(), Some("session-abc"));

        // A host that does not ask for a conversation id never sees one.
        let mut elsewhere = ProviderRequest {
            endpoint: "https://api.example.com/v1/chat/completions".into(),
            ..wire.clone()
        };
        elsewhere.headers.clear();
        client.apply_client_identity(&mut elsewhere);
        assert!(
            !elsewhere
                .headers
                .iter()
                .any(|(key, _)| key == "x-opencode-session")
        );
        assert!(
            elsewhere
                .headers
                .iter()
                .any(|(key, value)| key == "user-agent" && value == CLIENT_USER_AGENT)
        );

        // Missing conversation id: identity only.
        let mut anonymous = ProviderRequest {
            endpoint: "https://opencode.ai/zen/go/v1/chat/completions".into(),
            ..wire.clone()
        };
        anonymous.headers.clear();
        ProviderClient::new(
            "openai_compatible",
            "glm-5.3-flash",
            Some("https://opencode.ai/zen/go/v1".into()),
        )
        .unwrap()
        .apply_client_identity(&mut anonymous);
        assert!(
            !anonymous
                .headers
                .iter()
                .any(|(key, _)| key == "x-opencode-session")
        );
    }

    /// Reads one full HTTP request off `stream`, so a server that closes right
    /// after replying never resets a connection with unread input.
    fn read_full_request(stream: &mut std::net::TcpStream) -> String {
        use std::io::{BufRead as _, Read as _};
        let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
        let mut length = 0usize;
        let mut head = String::new();
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                length = value.trim().parse().unwrap_or(0);
            }
            head.push_str(&line);
        }
        let mut body = vec![0u8; length];
        let _ = reader.read_exact(&mut body);
        head
    }

    #[test]
    fn a_stall_before_any_output_is_retried_once() {
        use std::io::Write as _;
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let mut accepted = 0;
            // First connection: accept and drop without answering, which is what
            // a gateway does when it gives up on a queued request.
            if let Ok((mut stream, _)) = listener.accept() {
                accepted += 1;
                read_full_request(&mut stream);
                drop(stream);
            }
            // Second connection: a normal completion.
            if let Ok((mut stream, _)) = listener.accept() {
                accepted += 1;
                read_full_request(&mut stream);
                let body = serde_json::json!({
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "recovered"}, "finish_reason": "stop"}],
                    "usage": {"total_tokens": 3}
                })
                .to_string();
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
            }
            accepted
        });

        let mut client = ProviderClient::new(
            "openai_compatible",
            "test-model",
            Some(format!("http://127.0.0.1:{port}/v1")),
        )
        .unwrap()
        .with_api_key("test-key");
        client
            .caps
            .set(ProviderCapability::Streaming, TriState::Unsupported);

        let mut request = InferenceRequest::default();
        request
            .messages
            .push(crate::request::SemanticMessage::user("hello"));
        let turn = client
            .infer(&request, &mut |_| Ok(()))
            .expect("a dropped connection before any output is retried");
        assert_eq!(turn.text, "recovered");
        assert_eq!(server.join().unwrap(), 2, "the request was re-issued exactly once");
    }

    #[test]
    fn a_stall_after_output_is_not_retried() {
        use std::io::Write as _;
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let mut accepted = 0;
            if let Ok((mut stream, _)) = listener.accept() {
                accepted += 1;
                read_full_request(&mut stream);
                // Emit one delta, then drop: retrying here would duplicate text.
                let body = "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n";
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
            }
            accepted
        });

        let mut client = ProviderClient::new(
            "openai_compatible",
            "test-model",
            Some(format!("http://127.0.0.1:{port}/v1")),
        )
        .unwrap()
        .with_api_key("test-key");

        let mut request = InferenceRequest::default();
        request
            .messages
            .push(crate::request::SemanticMessage::user("hello"));
        let error = client
            .infer(&request, &mut |_| Ok(()))
            .expect_err("a truncated stream is an error, not a silent success");
        assert!(
            matches!(error.code.as_str(), "premature_stream_termination" | "stream_read_failed"),
            "unexpected code {}",
            error.code
        );
        assert_eq!(
            server.join().unwrap(),
            1,
            "output already reached the session; the request must not be replayed"
        );
    }

    #[test]
    fn an_idle_timeout_names_the_budget_that_expired() {
        struct Stalling;
        impl std::io::Read for Stalling {
            fn read(&mut self, _buffer: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "timed out reading response",
                ))
            }
        }
        let mut reader = std::io::BufReader::new(Stalling);
        let error = parse_sse_stream(
            &mut reader,
            WireFormat::OpenAiChatCompletions,
            1024,
            &mut |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(error.code, "stream_stalled");
        assert!(
            error.message.contains("ai_request_timeout_secs"),
            "the error must name the setting that can raise the budget: {}",
            error.message
        );
    }

    #[test]
    fn request_timeout_defaults_and_clamps() {
        assert_eq!(DEFAULT_REQUEST_TIMEOUT_SECS, 300);
        let client = ProviderClient::new(
            "openai_compatible",
            "m",
            Some("http://127.0.0.1:8000/v1".into()),
        )
        .unwrap();
        assert_eq!(client.timeout_secs, DEFAULT_REQUEST_TIMEOUT_SECS);
        assert_eq!(client.clone().with_timeout(5).timeout_secs, MIN_REQUEST_TIMEOUT_SECS);
        assert_eq!(client.clone().with_timeout(9_000).timeout_secs, MAX_REQUEST_TIMEOUT_SECS);
        assert_eq!(client.with_timeout(600).timeout_secs, 600);
    }
}
