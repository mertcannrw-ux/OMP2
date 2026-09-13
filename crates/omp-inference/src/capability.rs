use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// Capability tri-state. Unsupported and unknown capabilities must remain explicit;
/// do not coerce unknown to false.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriState {
    Supported,
    Unsupported,
    Unknown,
}

impl TriState {
    #[inline]
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported)
    }

    #[inline]
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported)
    }

    #[inline]
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }

    /// Converts explicitly to Option<bool>, where Unknown is None.
    /// This prevents accidental boolean coercion while allowing explicit unwrapping.
    #[inline]
    pub fn to_opt_bool(&self) -> Option<bool> {
        match self {
            Self::Supported => Some(true),
            Self::Unsupported => Some(false),
            Self::Unknown => None,
        }
    }
}

impl fmt::Display for TriState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Supported => write!(f, "supported"),
            Self::Unsupported => write!(f, "unsupported"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

impl FromStr for TriState {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "supported" | "true" | "yes" | "1" => Ok(Self::Supported),
            "unsupported" | "false" | "no" | "0" => Ok(Self::Unsupported),
            "unknown" | "unverified" | "?" => Ok(Self::Unknown),
            other => Err(format!("invalid TriState value: '{other}'")),
        }
    }
}

/// Provider capabilities beyond simple streaming.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderCapability {
    /// Authentication token or credential refresh capability.
    AuthRefresh,
    /// SSE / chunked token streaming.
    Streaming,
    /// Simple streaming without multiplexed events.
    SimpleStreaming,
    /// Offline or endpoint-based token counting before request dispatch.
    TokenCount,
    /// Detailed token and quota usage query.
    UsageQuery,
    /// Dynamic model discovery and catalog listing.
    ModelDiscovery,
    /// Text / multimodal embeddings generation.
    Embeddings,
    /// Image or video generation generation endpoints.
    ImageVideoGeneration,
    /// Web search or external retrieval grounding.
    Search,
    /// Remote provider-side context compaction or caching.
    RemoteCompaction,
    /// Constrained sampling / grammar-guided decoding.
    ConstrainedSampling,
    /// Provider-native tool choice / function calling parameter enforcement.
    NativeToolChoice,
    /// Dedicated `developer` message role distinct from `system` or `user`.
    DeveloperRole,
    /// Mid-session system prompt injections accepted without error.
    MidSessionSystemPrompts,
    /// Dialect parsing for provider-specific structured blocks.
    DialectParsing,
}

impl ProviderCapability {
    pub const ALL: &'static [ProviderCapability] = &[
        Self::AuthRefresh,
        Self::Streaming,
        Self::SimpleStreaming,
        Self::TokenCount,
        Self::UsageQuery,
        Self::ModelDiscovery,
        Self::Embeddings,
        Self::ImageVideoGeneration,
        Self::Search,
        Self::RemoteCompaction,
        Self::ConstrainedSampling,
        Self::NativeToolChoice,
        Self::DeveloperRole,
        Self::MidSessionSystemPrompts,
        Self::DialectParsing,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AuthRefresh => "auth_refresh",
            Self::Streaming => "streaming",
            Self::SimpleStreaming => "simple_streaming",
            Self::TokenCount => "token_count",
            Self::UsageQuery => "usage_query",
            Self::ModelDiscovery => "model_discovery",
            Self::Embeddings => "embeddings",
            Self::ImageVideoGeneration => "image_video_generation",
            Self::Search => "search",
            Self::RemoteCompaction => "remote_compaction",
            Self::ConstrainedSampling => "constrained_sampling",
            Self::NativeToolChoice => "native_tool_choice",
            Self::DeveloperRole => "developer_role",
            Self::MidSessionSystemPrompts => "mid_session_system_prompts",
            Self::DialectParsing => "dialect_parsing",
        }
    }
}

impl fmt::Display for ProviderCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for ProviderCapability {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let norm = s.trim().to_lowercase().replace('-', "_");
        match norm.as_str() {
            "auth_refresh" => Ok(Self::AuthRefresh),
            "streaming" => Ok(Self::Streaming),
            "simple_streaming" => Ok(Self::SimpleStreaming),
            "token_count" => Ok(Self::TokenCount),
            "usage_query" => Ok(Self::UsageQuery),
            "model_discovery" => Ok(Self::ModelDiscovery),
            "embeddings" => Ok(Self::Embeddings),
            "image_video_generation" => Ok(Self::ImageVideoGeneration),
            "search" => Ok(Self::Search),
            "remote_compaction" => Ok(Self::RemoteCompaction),
            "constrained_sampling" => Ok(Self::ConstrainedSampling),
            "native_tool_choice" => Ok(Self::NativeToolChoice),
            "developer_role" => Ok(Self::DeveloperRole),
            "mid_session_system_prompts" => Ok(Self::MidSessionSystemPrompts),
            "dialect_parsing" => Ok(Self::DialectParsing),
            other => Err(format!("unknown provider capability: '{other}'")),
        }
    }
}

/// Cost impact of native tool choice enforcement on prompt caching or tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum NativeToolCost {
    /// Native tool choice is side-effect-free and does not break prompt prefix caching.
    Free,
    /// Native tool choice incurs prefix cache invalidation, pricing surcharge, or extra latency.
    Costly,
    /// Cost impact is unknown or unverified.
    #[default]
    Unknown,
}


impl fmt::Display for NativeToolCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Free => write!(f, "free"),
            Self::Costly => write!(f, "costly"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Resolved capability profile for a model route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityProfile {
    capabilities: HashMap<ProviderCapability, TriState>,
    pub native_tool_cost: NativeToolCost,
}

impl Default for CapabilityProfile {
    fn default() -> Self {
        let mut capabilities = HashMap::new();
        for cap in ProviderCapability::ALL {
            capabilities.insert(*cap, TriState::Unknown);
        }
        Self {
            capabilities,
            native_tool_cost: NativeToolCost::Unknown,
        }
    }
}

impl CapabilityProfile {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, cap: ProviderCapability) -> TriState {
        self.capabilities
            .get(&cap)
            .copied()
            .unwrap_or(TriState::Unknown)
    }

    pub fn set(&mut self, cap: ProviderCapability, state: TriState) {
        self.capabilities.insert(cap, state);
    }

    /// Returns true only if explicitly Supported.
    pub fn supports(&self, cap: ProviderCapability) -> bool {
        self.get(cap).is_supported()
    }

    /// Returns true if capability is explicit Unsupported.
    pub fn is_unsupported(&self, cap: ProviderCapability) -> bool {
        self.get(cap).is_unsupported()
    }

    /// Returns true if capability is Unknown.
    pub fn is_unknown(&self, cap: ProviderCapability) -> bool {
        self.get(cap).is_unknown()
    }

    pub fn all_capabilities(&self) -> &HashMap<ProviderCapability, TriState> {
        &self.capabilities
    }
}
