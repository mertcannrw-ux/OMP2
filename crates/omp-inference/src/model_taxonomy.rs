use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Canonical model family lineage.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    Claude,
    Gpt,
    Gemini,
    DeepSeek,
    Llama,
    Qwen,
    Mistral,
    Custom(String),
}

impl ModelFamily {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Claude => "claude",
            Self::Gpt => "gpt",
            Self::Gemini => "gemini",
            Self::DeepSeek => "deepseek",
            Self::Llama => "llama",
            Self::Qwen => "qwen",
            Self::Mistral => "mistral",
            Self::Custom(name) => name.as_str(),
        }
    }

    pub fn matches_name(&self, name: &str) -> bool {
        let norm = name.trim().to_lowercase();
        self.as_str().eq_ignore_ascii_case(&norm)
    }
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for ModelFamily {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let norm = s.trim().to_lowercase();
        if norm.contains("claude") {
            Ok(Self::Claude)
        } else if norm.contains("gpt")
            || norm.starts_with('o')
                && (norm.starts_with("o1") || norm.starts_with("o3") || norm.starts_with("o4"))
        {
            Ok(Self::Gpt)
        } else if norm.contains("gemini") {
            Ok(Self::Gemini)
        } else if norm.contains("deepseek") {
            Ok(Self::DeepSeek)
        } else if norm.contains("llama") {
            Ok(Self::Llama)
        } else if norm.contains("qwen") {
            Ok(Self::Qwen)
        } else if norm.contains("mistral") || norm.contains("mixtral") {
            Ok(Self::Mistral)
        } else {
            Ok(Self::Custom(norm))
        }
    }
}

/// Capability class denoting the architectural lineage and expected performance profile.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelClass {
    /// High-capacity, general-purpose frontier model (e.g. Claude 3.7 Sonnet, GPT-4o).
    Flagship,
    /// Low-latency, cost-efficient model (e.g. Claude 3.5 Haiku, GPT-4o-mini, Gemini 2.0 Flash).
    Fast,
    /// Extended test-time compute / reasoning model (e.g. o1, o3-mini, DeepSeek-R1).
    Reasoning,
    /// Code-specialized checkpoint (e.g. Qwen 2.5 Coder).
    Coding,
    /// Representation and semantic search model.
    Embedding,
    /// Specialized vision-only encoder.
    VisionOnly,
    /// User-defined class.
    Custom(String),
}

impl ModelClass {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Flagship => "flagship",
            Self::Fast => "fast",
            Self::Reasoning => "reasoning",
            Self::Coding => "coding",
            Self::Embedding => "embedding",
            Self::VisionOnly => "vision_only",
            Self::Custom(name) => name.as_str(),
        }
    }
}

impl fmt::Display for ModelClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Provider and hosting route.
/// Separates what the route changes (endpoint, auth, encapsulation)
/// from semantic request building call sites.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderRoute {
    AnthropicDirect,
    OpenAiDirect,
    Bedrock {
        region: String,
    },
    Vertex {
        project: String,
        location: String,
    },
    AzureOpenAi {
        resource_name: String,
        deployment: String,
    },
    OpenAiCompatible {
        base_url: String,
    },
    Ollama {
        base_url: String,
    },
    LocalEmbedded,
    Custom {
        provider_name: String,
        host: String,
    },
}

impl ProviderRoute {
    pub fn provider_name(&self) -> &str {
        match self {
            Self::AnthropicDirect => "anthropic",
            Self::OpenAiDirect => "openai",
            Self::Bedrock { .. } => "bedrock",
            Self::Vertex { .. } => "vertex",
            Self::AzureOpenAi { .. } => "azure",
            Self::OpenAiCompatible { .. } => "openai_compatible",
            Self::Ollama { .. } => "ollama",
            Self::LocalEmbedded => "local_embedded",
            Self::Custom { provider_name, .. } => provider_name.as_str(),
        }
    }

    pub fn host_name(&self) -> &str {
        match self {
            Self::AnthropicDirect => "api.anthropic.com",
            Self::OpenAiDirect => "api.openai.com",
            Self::Bedrock { region } => region.as_str(),
            Self::Vertex { location, .. } => location.as_str(),
            Self::AzureOpenAi { resource_name, .. } => resource_name.as_str(),
            Self::OpenAiCompatible { base_url } => base_url.as_str(),
            Self::Ollama { base_url } => base_url.as_str(),
            Self::LocalEmbedded => "localhost",
            Self::Custom { host, .. } => host.as_str(),
        }
    }

    pub fn is_local(&self) -> bool {
        matches!(self, Self::LocalEmbedded | Self::Ollama { .. })
    }
}

/// What the model ID denotes: family, revision, and nominal context limits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelTaxonomy {
    pub model_id: String,
    pub family: ModelFamily,
    pub revision: Option<String>,
    pub display_name: String,
    pub context_window: Option<usize>,
    pub max_output_tokens: Option<usize>,
}

impl ModelTaxonomy {
    pub fn from_model_id(id: &str) -> Self {
        let trimmed = id.trim();
        let lower = trimmed.to_lowercase();
        let family: ModelFamily = lower.parse().unwrap_or(ModelFamily::Custom(lower.clone()));

        // Extract revision suffix if present (e.g. "20250219", "2024-08-06", "preview", "0613")
        let revision = extract_revision(&lower);
        let display_name = format_display_name(trimmed);

        Self {
            model_id: trimmed.to_string(),
            family,
            revision,
            display_name,
            context_window: None,
            max_output_tokens: None,
        }
    }
}

fn extract_revision(lower: &str) -> Option<String> {
    // Check for date-stamped revision like "20241022", "20250219" or "2024-08-06"
    for part in lower.split(['-', ':', '.']) {
        if part.len() == 8 && part.chars().all(|c| c.is_ascii_digit()) {
            return Some(part.to_string());
        }
    }
    // Check for YYYY-MM-DD pattern
    let parts: Vec<&str> = lower.split('-').collect();
    for window in parts.windows(3) {
        if window[0].len() == 4
            && window[0].chars().all(|c| c.is_ascii_digit())
            && window[1].len() == 2
            && window[1].chars().all(|c| c.is_ascii_digit())
            && window[2].len() == 2
            && window[2].chars().all(|c| c.is_ascii_digit())
        {
            return Some(format!("{}-{}-{}", window[0], window[1], window[2]));
        }
    }
    None
}

fn format_display_name(id: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    for seg in id.split(['-', '_', '/']) {
        if seg.is_empty() {
            continue;
        }
        let mut chars = seg.chars();
        if let Some(first) = chars.next() {
            let capitalized = first.to_uppercase().collect::<String>() + chars.as_str();
            parts.push(capitalized);
        }
    }
    if parts.is_empty() {
        id.to_string()
    } else {
        parts.join(" ")
    }
}

/// Composite model identifier decoupling taxonomy, capability class, and provider route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentifier {
    pub taxonomy: ModelTaxonomy,
    pub class: ModelClass,
    pub route: ProviderRoute,
}

impl ModelIdentifier {
    pub fn new(taxonomy: ModelTaxonomy, class: ModelClass, route: ProviderRoute) -> Self {
        Self {
            taxonomy,
            class,
            route,
        }
    }

    /// Automatically infers taxonomy and capability class from model ID string.
    pub fn infer(model_id: &str, route: ProviderRoute) -> Self {
        let taxonomy = ModelTaxonomy::from_model_id(model_id);
        let class = infer_class(&taxonomy.model_id, &taxonomy.family);
        Self {
            taxonomy,
            class,
            route,
        }
    }

    pub fn model_id(&self) -> &str {
        &self.taxonomy.model_id
    }

    pub fn family(&self) -> &ModelFamily {
        &self.taxonomy.family
    }

    pub fn revision(&self) -> Option<&str> {
        self.taxonomy.revision.as_deref()
    }

    pub fn provider_name(&self) -> &str {
        self.route.provider_name()
    }

    pub fn host_name(&self) -> &str {
        self.route.host_name()
    }
}

/// Heuristic classifier: substring matching, not a model registry. Known
/// over-matches (e.g. an id containing "r1" such as "author1", or starting
/// with "o1") classify as Reasoning; prefer explicit `ModelIdentifier`
/// construction when the class matters for capability gating.
fn infer_class(model_id: &str, family: &ModelFamily) -> ModelClass {
    let lower = model_id.to_lowercase();
    if lower.contains("embed") {
        return ModelClass::Embedding;
    }
    if lower.contains("reason")
        || lower.contains("r1")
        || lower.starts_with("o1")
        || lower.starts_with("o3")
    {
        return ModelClass::Reasoning;
    }
    if lower.contains("coder") || lower.contains("coding") {
        return ModelClass::Coding;
    }
    if lower.contains("haiku")
        || lower.contains("mini")
        || lower.contains("flash")
        || lower.contains("small")
    {
        return ModelClass::Fast;
    }
    match family {
        ModelFamily::Claude | ModelFamily::Gpt | ModelFamily::Gemini => ModelClass::Flagship,
        ModelFamily::DeepSeek => ModelClass::Reasoning,
        _ => ModelClass::Flagship,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_taxonomy_no_guessed_limits() {
        let claude = ModelTaxonomy::from_model_id("claude-3-7-sonnet-20250219");
        assert_eq!(claude.family, ModelFamily::Claude);
        assert_eq!(claude.revision, Some("20250219".to_string()));
        assert_eq!(claude.context_window, None);
        assert_eq!(claude.max_output_tokens, None);

        let gpt = ModelTaxonomy::from_model_id("gpt-4o");
        assert_eq!(gpt.family, ModelFamily::Gpt);
        assert_eq!(gpt.context_window, None);
        assert_eq!(gpt.max_output_tokens, None);

        let custom = ModelTaxonomy::from_model_id("custom-model-x");
        assert_eq!(custom.context_window, None);
        assert_eq!(custom.max_output_tokens, None);
    }
}
