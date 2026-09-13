use crate::capability::{CapabilityProfile, ProviderCapability};
use crate::model_taxonomy::{ModelFamily, ModelIdentifier, ProviderRoute};
use omp_types::{ArtifactId, StructuredError, ToolCallId};
use serde::{Deserialize, Serialize};

/// Role of a semantic message in conversation history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

impl MessageRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// Content part within a rich multimodal or artifact-backed message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        media_type: String,
        data_base64: String,
    },
    ArtifactRef {
        artifact_id: ArtifactId,
        byte_range: Option<(u64, u64)>,
    },
}

/// Message payload: single text or structured multimodal parts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Flattens text parts, joining with newlines.
    /// Returns an error for Image/ArtifactRef parts instead of silently dropping
    /// them, so unsupported multimodal content can never vanish on the wire.
    pub fn as_text(&self) -> Result<String, StructuredError> {
        match self {
            Self::Text(s) => Ok(s.clone()),
            Self::Parts(parts) => {
                let mut out = String::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str(text);
                        }
                        ContentPart::Image { .. } => {
                            return Err(StructuredError::new(
                                "unsupported_content_part",
                                "Image content parts require a multimodal-capable wire format and cannot be flattened to text",
                                false,
                            ));
                        }
                        ContentPart::ArtifactRef { .. } => {
                            return Err(StructuredError::new(
                                "unsupported_content_part",
                                "ArtifactRef content parts must be resolved to text before request translation",
                                false,
                            ));
                        }
                    }
                }
                Ok(out)
            }
        }
    }
}

/// Tool call specification from assistant response or projected turn.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallSpec {
    pub id: ToolCallId,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Semantic message in the conversation chain.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticMessage {
    pub role: MessageRole,
    pub content: MessageContent,
    pub tool_calls: Vec<ToolCallSpec>,
    pub tool_call_id: Option<ToolCallId>,
    pub name: Option<String>,
}

impl SemanticMessage {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: MessageContent::Text(text.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: MessageContent::Text(text.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: MessageContent::Text(text.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn tool_result(tool_call_id: ToolCallId, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: MessageContent::Text(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id),
            name: None,
        }
    }
}

/// Transcript fold representing speculative compaction or context pruning.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MessageFold {
    /// Summarized previous turns preserving semantic handoff state.
    Handoff {
        summary: String,
        folded_count: usize,
        original_token_count: usize,
    },
    /// Dropped old turns, preserving exact prefix and recent window.
    Shake {
        kept_prefix_count: usize,
        dropped_count: usize,
    },
    /// Provider remote context cache handle.
    RemoteState { handle: String, token_count: usize },
}

/// Tool definition schema exposed to the model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Desired thinking / extended test-time compute mode.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingMode {
    #[default]
    None,
    Disabled,
    Auto,
    Enabled {
        budget_tokens: Option<u32>,
    },
    Adaptive,
    Effort {
        level: String,
    },
}
/// Tool choice constraint requested by Director or user.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoiceRequirement {
    None,
    Auto,
    Required,
    Forced { tool: String },
}

/// Sampling parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stop_sequences: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            top_p: None,
            max_tokens: None,
            stop_sequences: Vec::new(),
        }
    }
}

/// Constrained grammar specification for output decoding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarSpec {
    pub kind: String,
    pub definition: String,
}

/// Compaction strategy for context bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionStrategy {
    Handoff,
    Shake,
    Remote,
}

/// Compaction policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompactionPolicy {
    pub threshold_tokens: usize,
    pub strategy: CompactionStrategy,
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            threshold_tokens: 100_000,
            strategy: CompactionStrategy::Handoff,
        }
    }
}

/// Semantic inference request representing agent intent independent of provider wire details.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub messages: Vec<SemanticMessage>,
    pub folds: Vec<MessageFold>,
    pub active_tools: Vec<ToolSchema>,
    pub desired_thinking: ThinkingMode,
    pub tool_choice: ToolChoiceRequirement,
    pub strict_schema: bool,
    pub grammar: Option<GrammarSpec>,
    pub sampling: SamplingParams,
    pub usage_request: bool,
    pub compaction_policy: CompactionPolicy,
}

impl Default for InferenceRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            folds: Vec::new(),
            active_tools: Vec::new(),
            desired_thinking: ThinkingMode::None,
            tool_choice: ToolChoiceRequirement::Auto,
            strict_schema: false,
            grammar: None,
            sampling: SamplingParams::default(),
            usage_request: true,
            compaction_policy: CompactionPolicy::default(),
        }
    }
}

/// Wire protocol format expected by provider endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireFormat {
    AnthropicMessages,
    OpenAiChatCompletions,
    GeminiGenerateContent,
    OllamaChat,
    GenericJson,
}

/// Provider-specific wire request ready for HTTP transmission.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderRequest {
    pub endpoint: String,
    pub headers: Vec<(String, String)>,
    pub body: serde_json::Value,
    pub native_tool_choice_used: bool,
    pub soft_instruction_injected: bool,
    pub wire_format: WireFormat,
}

/// Translates a semantic InferenceRequest into a provider wire request.
/// Respects capability profile without hardcoded provider branches at request sites.
pub fn translate_request(
    req: &InferenceRequest,
    ident: &ModelIdentifier,
    caps: &CapabilityProfile,
) -> Result<ProviderRequest, StructuredError> {
    let wire_format = match &ident.route {
        ProviderRoute::AnthropicDirect => WireFormat::AnthropicMessages,
        ProviderRoute::OpenAiDirect
        | ProviderRoute::AzureOpenAi { .. }
        | ProviderRoute::OpenAiCompatible { .. } => WireFormat::OpenAiChatCompletions,
        ProviderRoute::Ollama { .. } => WireFormat::OllamaChat,
        _ => match ident.family() {
            ModelFamily::Claude => WireFormat::AnthropicMessages,
            ModelFamily::Gemini => WireFormat::GeminiGenerateContent,
            _ => WireFormat::OpenAiChatCompletions,
        },
    };

    match wire_format {
        WireFormat::AnthropicMessages => translate_anthropic(req, ident, caps),
        WireFormat::OpenAiChatCompletions => translate_openai(req, ident, caps),
        // Intentionally lossy fallback: unknown/custom-family routes (including
        // GeminiGenerateContent and OllamaChat, which have no dedicated
        // translators yet) go out as OpenAI-compatible chat completions, the
        // widest-supported dialect. Native Gemini/Ollama extras are dropped.
        _ => translate_openai(req, ident, caps),
    }
}

fn translate_anthropic(
    req: &InferenceRequest,
    ident: &ModelIdentifier,
    caps: &CapabilityProfile,
) -> Result<ProviderRequest, StructuredError> {
    let mut system_prompt = String::new();
    let mut soft_instruction_injected = false;

    // Apply folds
    for fold in &req.folds {
        if let MessageFold::Handoff { summary, .. } = fold {
            if !system_prompt.is_empty() {
                system_prompt.push_str("\n\n");
            }
            system_prompt.push_str("[Context Summary from Earlier Turns:\n");
            system_prompt.push_str(summary);
            system_prompt.push_str("\n]");
        }
    }

    // Process system and normal messages
    let mut wire_messages = Vec::new();
    for msg in &req.messages {
        match msg.role {
            MessageRole::System | MessageRole::Developer => {
                let text = msg.content.as_text()?;
                if !system_prompt.is_empty() {
                    system_prompt.push_str("\n\n");
                }
                system_prompt.push_str(&text);
            }
            MessageRole::User => {
                wire_messages.push(serde_json::json!({
                    "role": "user",
                    "content": msg.content.as_text()?,
                }));
            }
            MessageRole::Assistant => {
                let mut content_parts = Vec::new();
                let text = msg.content.as_text()?;
                if !text.is_empty() {
                    content_parts.push(serde_json::json!({
                        "type": "text",
                        "text": text,
                    }));
                }
                for tc in &msg.tool_calls {
                    content_parts.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id.as_str(),
                        "name": tc.name,
                        "input": tc.arguments,
                    }));
                }
                wire_messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": content_parts,
                }));
            }
            MessageRole::Tool => {
                let tool_call_id = msg
                    .tool_call_id
                    .as_ref()
                    .map(|id| id.as_str())
                    .unwrap_or("unknown_tool_call");
                wire_messages.push(serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": msg.content.as_text()?,
                    }]
                }));
            }
        }
    }

    // Tool choice and tool definitions
    let mut native_tool_choice_used = false;
    let mut wire_tools = Vec::new();
    for tool in &req.active_tools {
        wire_tools.push(serde_json::json!({
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }));
    }

    let mut tool_choice_json = None;
    match &req.tool_choice {
        ToolChoiceRequirement::Forced { tool } => {
            if caps.supports(ProviderCapability::NativeToolChoice) {
                native_tool_choice_used = true;
                tool_choice_json = Some(serde_json::json!({
                    "type": "tool",
                    "name": tool,
                }));
            } else {
                // Soft instruction fallback
                soft_instruction_injected = true;
                let soft = format!(
                    "\n[System Directive: You MUST call tool '{tool}' with valid parameters next.]"
                );
                system_prompt.push_str(&soft);
            }
        }
        ToolChoiceRequirement::Required => {
            if caps.supports(ProviderCapability::NativeToolChoice) {
                native_tool_choice_used = true;
                tool_choice_json = Some(serde_json::json!({ "type": "any" }));
            }
        }
        ToolChoiceRequirement::Auto => {
            if !wire_tools.is_empty() {
                tool_choice_json = Some(serde_json::json!({ "type": "auto" }));
            }
        }
        ToolChoiceRequirement::None => {}
    }

    let max_tokens = match req.sampling.max_tokens {
        Some(tokens) => tokens,
        None => match ident.taxonomy.max_output_tokens.and_then(|v| u32::try_from(v).ok()) {
            Some(tokens) => tokens,
            None => {
                return Err(StructuredError::new(
                    "model_metadata_missing",
                    format!(
                        "Anthropic messages API requires max_tokens, but none was specified in sampling params or model metadata for model '{}'",
                        ident.model_id()
                    ),
                    false,
                ));
            }
        },
    };

    let mut body = serde_json::json!({
        "model": ident.model_id(),
        "messages": wire_messages,
        "max_tokens": max_tokens,
    });
    if !system_prompt.is_empty() {
        body["system"] = serde_json::Value::String(system_prompt);
    }
    if !wire_tools.is_empty() {
        body["tools"] = serde_json::Value::Array(wire_tools);
    }
    if let Some(tc) = tool_choice_json {
        body["tool_choice"] = tc;
    }
    if let Some(temp) = req.sampling.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(top_p) = req.sampling.top_p {
        body["top_p"] = serde_json::json!(top_p);
    }
    if !req.sampling.stop_sequences.is_empty() {
        body["stop_sequences"] = serde_json::json!(req.sampling.stop_sequences);
    }
    if let Some(grammar) = &req.grammar {
        let grammar_desc = format!(
            "\n\n[Output Constraint: You must respond strictly conforming to {}:\n{}\n]",
            grammar.kind, grammar.definition
        );
        if let Some(sys) = body.get_mut("system") {
            if let Some(s) = sys.as_str() {
                let mut updated = s.to_string();
                updated.push_str(&grammar_desc);
                *sys = serde_json::Value::String(updated);
            }
        } else {
            body["system"] = serde_json::Value::String(grammar_desc);
        }
    }
    // Thinking mode
    match &req.desired_thinking {
        ThinkingMode::None | ThinkingMode::Disabled => {}
        ThinkingMode::Enabled {
            budget_tokens: Some(budget),
        } => {
            body["thinking"] = serde_json::json!({
                "type": "enabled",
                "budget_tokens": budget,
            });
        }
        ThinkingMode::Enabled {
            budget_tokens: None,
        } => {
            return Err(StructuredError::new(
                "thinking_budget_missing",
                "ThinkingMode::Enabled requires an explicit budget_tokens when auto/adaptive thinking is not selected",
                false,
            ));
        }
        ThinkingMode::Auto => {}
        ThinkingMode::Adaptive => {
            body["thinking"] = serde_json::json!({
                "type": "adaptive",
            });
        }
        ThinkingMode::Effort { level } => {
            body["thinking"] = serde_json::json!({
                "type": "adaptive",
            });
            body["output_config"] = serde_json::json!({
                "effort": level,
            });
        }
    }
    Ok(ProviderRequest {
        endpoint: format!("https://{}/v1/messages", ident.host_name()),
        headers: vec![
            ("content-type".into(), "application/json".into()),
            ("anthropic-version".into(), "2023-06-01".into()),
        ],
        body,
        native_tool_choice_used,
        soft_instruction_injected,
        wire_format: WireFormat::AnthropicMessages,
    })
}

fn translate_openai(
    req: &InferenceRequest,
    ident: &ModelIdentifier,
    caps: &CapabilityProfile,
) -> Result<ProviderRequest, StructuredError> {
    let mut soft_instruction_injected = false;
    let mut native_tool_choice_used = false;

    let supports_developer_role = caps.supports(ProviderCapability::DeveloperRole);
    let mut wire_messages = Vec::new();

    // Process folds
    for fold in &req.folds {
        if let MessageFold::Handoff { summary, .. } = fold {
            let role = if supports_developer_role {
                "developer"
            } else {
                "system"
            };
            wire_messages.push(serde_json::json!({
                "role": role,
                "content": format!("[Context Summary:\n{summary}\n]"),
            }));
        }
    }

    for msg in &req.messages {
        match msg.role {
            MessageRole::Developer => {
                let role = if supports_developer_role {
                    "developer"
                } else {
                    "system"
                };
                wire_messages.push(serde_json::json!({
                    "role": role,
                    "content": msg.content.as_text()?,
                }));
            }
            MessageRole::System => {
                wire_messages.push(serde_json::json!({
                    "role": "system",
                    "content": msg.content.as_text()?,
                }));
            }
            MessageRole::User => {
                wire_messages.push(serde_json::json!({
                    "role": "user",
                    "content": msg.content.as_text()?,
                }));
            }
            MessageRole::Assistant => {
                let mut wire_msg = serde_json::json!({
                    "role": "assistant",
                    "content": msg.content.as_text()?,
                });
                if !msg.tool_calls.is_empty() {
                    let calls: Vec<serde_json::Value> = msg
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": tc.id.as_str(),
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments.to_string(),
                                }
                            })
                        })
                        .collect();
                    wire_msg["tool_calls"] = serde_json::Value::Array(calls);
                }
                wire_messages.push(wire_msg);
            }
            MessageRole::Tool => {
                let tool_call_id = msg
                    .tool_call_id
                    .as_ref()
                    .map(|id| id.as_str())
                    .unwrap_or("unknown_tool_call");
                wire_messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": msg.content.as_text()?,
                }));
            }
        }
    }

    let mut wire_tools = Vec::new();
    for tool in &req.active_tools {
        wire_tools.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        }));
    }

    let mut tool_choice_json = None;
    match &req.tool_choice {
        ToolChoiceRequirement::Forced { tool } => {
            if caps.supports(ProviderCapability::NativeToolChoice) {
                native_tool_choice_used = true;
                tool_choice_json = Some(serde_json::json!({
                    "type": "function",
                    "function": { "name": tool }
                }));
            } else {
                soft_instruction_injected = true;
                wire_messages.push(serde_json::json!({
                    "role": "user",
                    "content": format!("[Directive: You MUST call tool '{tool}' with valid parameters next.]"),
                }));
            }
        }
        ToolChoiceRequirement::Required => {
            if caps.supports(ProviderCapability::NativeToolChoice) {
                native_tool_choice_used = true;
                tool_choice_json = Some(serde_json::json!("required"));
            }
        }
        ToolChoiceRequirement::Auto => {
            if !wire_tools.is_empty() {
                tool_choice_json = Some(serde_json::json!("auto"));
            }
        }
        ToolChoiceRequirement::None => {
            tool_choice_json = Some(serde_json::json!("none"));
        }
    }

    let mut body = serde_json::json!({
        "model": ident.model_id(),
        "messages": wire_messages,
    });

    if !wire_tools.is_empty() {
        body["tools"] = serde_json::Value::Array(wire_tools);
    }
    if let Some(tc) = tool_choice_json {
        body["tool_choice"] = tc;
    }
    if let Some(temp) = req.sampling.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(max_tokens) = req.sampling.max_tokens {
        body["max_tokens"] = serde_json::json!(max_tokens);
    }
    if let Some(top_p) = req.sampling.top_p {
        body["top_p"] = serde_json::json!(top_p);
    }
    if !req.sampling.stop_sequences.is_empty() {
        body["stop"] = serde_json::json!(req.sampling.stop_sequences);
    }
    if let Some(grammar) = &req.grammar {
        if caps.supports(ProviderCapability::ConstrainedSampling) {
            if grammar.kind == "json_schema" {
                let schema: serde_json::Value =
                    serde_json::from_str(&grammar.definition).map_err(|e| {
                        StructuredError::new(
                            "invalid_grammar_schema",
                            format!("json_schema grammar definition is not valid JSON: {e}"),
                            false,
                        )
                    })?;
                body["response_format"] = serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "constrained_output",
                        "strict": req.strict_schema,
                        "schema": schema
                    }
                });
            } else if grammar.kind == "json_object" {
                body["response_format"] = serde_json::json!({ "type": "json_object" });
            }
        } else {
            let constraint = format!(
                "[Constraint: Respond in valid {}. Strict schema enforcement.]",
                grammar.kind
            );
            wire_messages.push(serde_json::json!({
                "role": if supports_developer_role { "developer" } else { "system" },
                "content": constraint,
            }));
            body["messages"] = serde_json::Value::Array(wire_messages);
        }
    }

    // Thinking mode / reasoning effort
    match &req.desired_thinking {
        ThinkingMode::None | ThinkingMode::Disabled => {}
        ThinkingMode::Effort { level } => {
            body["reasoning_effort"] = serde_json::json!(level);
        }
        ThinkingMode::Adaptive | ThinkingMode::Auto => {}
        ThinkingMode::Enabled {
            budget_tokens: Some(_),
        } => {
            return Err(StructuredError::new(
                "unsupported_thinking_budget",
                "This wire dialect does not advertise a token-budget thinking control; use an advertised effort level",
                false,
            ));
        }
        ThinkingMode::Enabled {
            budget_tokens: None,
        } => {
            return Err(StructuredError::new(
                "thinking_budget_missing",
                "ThinkingMode::Enabled requires an explicit budget_tokens when auto/adaptive thinking is not selected",
                false,
            ));
        }
    }

    Ok(ProviderRequest {
        endpoint: format!("https://{}/v1/chat/completions", ident.host_name()),
        headers: vec![("content-type".into(), "application/json".into())],
        body,
        native_tool_choice_used,
        soft_instruction_injected,
        wire_format: WireFormat::OpenAiChatCompletions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapabilityProfile;
    use crate::model_taxonomy::{ModelClass, ModelTaxonomy, ProviderRoute};

    fn test_ident(model_id: &str, route: ProviderRoute) -> ModelIdentifier {
        ModelIdentifier::new(
            ModelTaxonomy::from_model_id(model_id),
            ModelClass::Flagship,
            route,
        )
    }

    #[test]
    fn test_anthropic_missing_max_tokens_error() {
        let req = InferenceRequest {
            messages: vec![SemanticMessage::user("hello")],
            sampling: SamplingParams::default(), // max_tokens: None
            ..Default::default()
        };
        let ident = test_ident("claude-3-7-sonnet", ProviderRoute::AnthropicDirect);
        let caps = CapabilityProfile::new();
        let err = translate_request(&req, &ident, &caps).unwrap_err();
        assert_eq!(err.code, "model_metadata_missing");
    }

    #[test]
    fn test_anthropic_thinking_effort() {
        let req = InferenceRequest {
            messages: vec![SemanticMessage::user("hello")],
            sampling: SamplingParams {
                max_tokens: Some(4096),
                ..Default::default()
            },
            desired_thinking: ThinkingMode::Effort {
                level: "high".to_string(),
            },
            ..Default::default()
        };
        let ident = test_ident("claude-3-7-sonnet", ProviderRoute::AnthropicDirect);
        let caps = CapabilityProfile::new();
        let p_req = translate_request(&req, &ident, &caps).unwrap();
        assert_eq!(p_req.body["thinking"]["type"], "adaptive");
        assert_eq!(p_req.body["output_config"]["effort"], "high");
    }

    #[test]
    fn test_anthropic_thinking_enabled_missing_budget() {
        let req = InferenceRequest {
            messages: vec![SemanticMessage::user("hello")],
            sampling: SamplingParams {
                max_tokens: Some(4096),
                ..Default::default()
            },
            desired_thinking: ThinkingMode::Enabled {
                budget_tokens: None,
            },
            ..Default::default()
        };
        let ident = test_ident("claude-3-7-sonnet", ProviderRoute::AnthropicDirect);
        let caps = CapabilityProfile::new();
        let err = translate_request(&req, &ident, &caps).unwrap_err();
        assert_eq!(err.code, "thinking_budget_missing");
    }

    #[test]
    fn test_anthropic_thinking_enabled_explicit_budget() {
        let req = InferenceRequest {
            messages: vec![SemanticMessage::user("hello")],
            sampling: SamplingParams {
                max_tokens: Some(4096),
                ..Default::default()
            },
            desired_thinking: ThinkingMode::Enabled {
                budget_tokens: Some(1024),
            },
            ..Default::default()
        };
        let ident = test_ident("claude-3-7-sonnet", ProviderRoute::AnthropicDirect);
        let caps = CapabilityProfile::new();
        let p_req = translate_request(&req, &ident, &caps).unwrap();
        assert_eq!(p_req.body["thinking"]["type"], "enabled");
        assert_eq!(p_req.body["thinking"]["budget_tokens"], 1024);
    }

    #[test]
    fn test_openai_reasoning_effort() {
        let req = InferenceRequest {
            messages: vec![SemanticMessage::user("hello")],
            desired_thinking: ThinkingMode::Effort {
                level: "medium".to_string(),
            },
            ..Default::default()
        };
        let ident = test_ident("o3-mini", ProviderRoute::OpenAiDirect);
        let caps = CapabilityProfile::new();
        let p_req = translate_request(&req, &ident, &caps).unwrap();
        assert_eq!(p_req.body["reasoning_effort"], "medium");
    }

    #[test]
    fn test_openai_thinking_enabled_missing_budget() {
        let req = InferenceRequest {
            messages: vec![SemanticMessage::user("hello")],
            desired_thinking: ThinkingMode::Enabled {
                budget_tokens: None,
            },
            ..Default::default()
        };
        let ident = test_ident("o3-mini", ProviderRoute::OpenAiDirect);
        let caps = CapabilityProfile::new();
        let err = translate_request(&req, &ident, &caps).unwrap_err();
        assert_eq!(err.code, "thinking_budget_missing");
    }
}
