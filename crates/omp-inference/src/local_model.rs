use omp_types::StructuredError;
use serde::{Deserialize, Serialize};

/// Structured judgment returned by the local model for internal gating.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalJudgment {
    pub passed: bool,
    pub score: f32,
    pub rationale: String,
}

/// Internal cheap local model capability for lightweight classification,
/// auto-titling, translation, judging, and TTS/STT policy hooks.
///
/// Selected strictly by host policy; never a second user-visible agent.
pub trait LocalModelEngine: Send + Sync {
    /// Classifies an input text into one of the candidate category labels.
    fn classify(&self, text: &str, candidate_labels: &[String]) -> Result<String, StructuredError>;

    /// Generates a concise title (<= 8 words) for a conversation summary or prompt.
    fn generate_title(&self, conversation_summary: &str) -> Result<String, StructuredError>;

    /// Translates a short text into the target language code (e.g. "en", "es", "tr").
    fn translate(&self, text: &str, target_language: &str) -> Result<String, StructuredError>;

    /// Evaluates a candidate output against validation criteria.
    fn judge(&self, criteria: &str, candidate: &str) -> Result<LocalJudgment, StructuredError>;

    /// Synthesizes speech audio from text input (TTS hook).
    fn synthesize_speech(&self, text: &str) -> Result<Vec<u8>, StructuredError>;

    /// Transcribes speech audio to text output (STT hook).
    fn transcribe_speech(&self, audio_bytes: &[u8]) -> Result<String, StructuredError>;
}

/// Deterministic, fast, offline local engine requiring no external network or API keys.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeuristicLocalEngine;

impl HeuristicLocalEngine {
    pub fn new() -> Self {
        Self
    }
}

impl LocalModelEngine for HeuristicLocalEngine {
    fn classify(&self, text: &str, candidate_labels: &[String]) -> Result<String, StructuredError> {
        if candidate_labels.is_empty() {
            return Err(StructuredError::new(
                "empty_candidate_labels",
                "Cannot classify text against an empty label set",
                false,
            ));
        }

        let lower_text = text.to_lowercase();
        let mut best_label = &candidate_labels[0];
        let mut best_score = 0;

        for label in candidate_labels {
            let lower_label = label.to_lowercase();
            let mut score = 0;
            for token in lower_label.split([' ', '_', '-']) {
                if !token.is_empty() && lower_text.contains(token) {
                    score += 1;
                }
            }
            if score > best_score {
                best_score = score;
                best_label = label;
            }
        }

        Ok(best_label.clone())
    }

    fn generate_title(&self, conversation_summary: &str) -> Result<String, StructuredError> {
        let first_line = conversation_summary
            .lines()
            .map(|l| l.trim())
            .find(|l| !l.is_empty())
            .unwrap_or("New Session");

        // Clean out prompt markers
        let clean = first_line
            .trim_start_matches('#')
            .trim_start_matches('-')
            .trim_start_matches('*')
            .trim();

        let words: Vec<&str> = clean.split_whitespace().take(6).collect();
        if words.is_empty() {
            return Ok("New Session".into());
        }

        let mut capitalized_words = Vec::new();
        for word in words {
            let mut chars = word.chars();
            if let Some(first) = chars.next() {
                capitalized_words.push(first.to_uppercase().collect::<String>() + chars.as_str());
            }
        }

        Ok(capitalized_words.join(" "))
    }

    fn translate(&self, _text: &str, _target_language: &str) -> Result<String, StructuredError> {
        Err(StructuredError::new(
            "unsupported_capability",
            "Translation requires a configured model engine",
            false,
        ))
    }

    fn judge(&self, criteria: &str, candidate: &str) -> Result<LocalJudgment, StructuredError> {
        let lower_crit = criteria.to_lowercase();
        let lower_cand = candidate.to_lowercase();

        let mut matching_keywords = 0;
        let mut total_keywords = 0;

        for word in lower_crit.split_whitespace() {
            let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
            if trimmed.len() > 3 {
                total_keywords += 1;
                if lower_cand.contains(trimmed) {
                    matching_keywords += 1;
                }
            }
        }

        let score = if total_keywords > 0 {
            matching_keywords as f32 / total_keywords as f32
        } else {
            1.0
        };

        let passed = score >= 0.5;
        let rationale = format!(
            "Candidate matched {}/{} criteria keywords (score: {:.2})",
            matching_keywords, total_keywords, score
        );

        Ok(LocalJudgment {
            passed,
            score,
            rationale,
        })
    }

    fn synthesize_speech(&self, _text: &str) -> Result<Vec<u8>, StructuredError> {
        Err(StructuredError::new(
            "unsupported_in_heuristic",
            "TTS speech synthesis requires a bundled or external local model engine",
            false,
        ))
    }

    fn transcribe_speech(&self, _audio_bytes: &[u8]) -> Result<String, StructuredError> {
        Err(StructuredError::new(
            "unsupported_in_heuristic",
            "STT speech transcription requires a bundled or external local model engine",
            false,
        ))
    }
}

/// Selection policy for internal cheap local model engine.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum LocalEnginePolicy {
    /// Deterministic offline heuristics (zero runtime dependencies).
    #[default]
    Heuristic,
    /// Bundled local model runtime with weights located on local filesystem.
    Bundled { model_path: Option<String> },
    /// External local inference server (e.g. Ollama or local endpoint).
    External { endpoint: String, model: String },
    /// Local model policy hooks explicitly disabled.
    Disabled,
}


/// Engine that interfaces with a bundled local model runtime on the host.
/// Fails cleanly with structured error if runtime or model weights are absent (no fake success).
#[derive(Clone, Debug)]
pub struct BundledLocalEngine {
    pub model_path: Option<String>,
}

impl BundledLocalEngine {
    pub fn new(model_path: Option<String>) -> Self {
        Self { model_path }
    }

    fn check_runtime_available(&self) -> Result<(), StructuredError> {
        Err(StructuredError::new(
            "bundled_runtime_absent",
            "No bundled model inference runtime is installed; a weights path alone is not an executable model",
            false,
        ))
    }
}

impl LocalModelEngine for BundledLocalEngine {
    fn classify(&self, text: &str, candidate_labels: &[String]) -> Result<String, StructuredError> {
        self.check_runtime_available()?;
        HeuristicLocalEngine::new().classify(text, candidate_labels)
    }

    fn generate_title(&self, conversation_summary: &str) -> Result<String, StructuredError> {
        self.check_runtime_available()?;
        HeuristicLocalEngine::new().generate_title(conversation_summary)
    }

    fn translate(&self, text: &str, target_language: &str) -> Result<String, StructuredError> {
        self.check_runtime_available()?;
        HeuristicLocalEngine::new().translate(text, target_language)
    }

    fn judge(&self, criteria: &str, candidate: &str) -> Result<LocalJudgment, StructuredError> {
        self.check_runtime_available()?;
        HeuristicLocalEngine::new().judge(criteria, candidate)
    }

    fn synthesize_speech(&self, _text: &str) -> Result<Vec<u8>, StructuredError> {
        self.check_runtime_available()?;
        Err(StructuredError::new(
            "bundled_audio_unimplemented",
            "Bundled audio synthesis runtime is not initialized",
            false,
        ))
    }

    fn transcribe_speech(&self, _audio_bytes: &[u8]) -> Result<String, StructuredError> {
        self.check_runtime_available()?;
        Err(StructuredError::new(
            "bundled_audio_unimplemented",
            "Bundled audio transcription runtime is not initialized",
            false,
        ))
    }
}

/// Engine that interfaces with an external local inference endpoint (e.g. Ollama, llama.cpp).
/// Fails cleanly if the endpoint is absent/unreachable (no fake success).
#[derive(Clone, Debug)]
pub struct ExternalLocalEngine {
    pub endpoint: String,
    pub model: String,
}

impl ExternalLocalEngine {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            model: model.into(),
        }
    }

    fn infer_text(&self, instruction: &str, input: &str) -> Result<String, StructuredError> {
        use crate::request::{InferenceRequest, SamplingParams, SemanticMessage};
        let mut client = crate::provider::ProviderClient::new(
            "openai_compatible",
            &self.model,
            Some(self.endpoint.clone()),
        )?;
        client.api_key = None;
        client.max_response_bytes = 65_536;
        client.timeout_secs = 30;
        let request = InferenceRequest {
            messages: vec![
                SemanticMessage::system(instruction),
                SemanticMessage::user(input),
            ],
            sampling: SamplingParams {
                max_tokens: Some(512),
                ..Default::default()
            },
            ..Default::default()
        };
        Ok(client.infer(&request, &mut |_| Ok(()))?.text)
    }
}

impl LocalModelEngine for ExternalLocalEngine {
    fn classify(&self, text: &str, candidate_labels: &[String]) -> Result<String, StructuredError> {
        if candidate_labels.is_empty() {
            return Err(StructuredError::new(
                "empty_candidate_labels",
                "Classification requires candidate labels",
                false,
            ));
        }
        let instruction = format!(
            "Classify the input. Return exactly one label from this JSON array: {}",
            serde_json::to_string(candidate_labels).unwrap_or_else(|_| "[]".into())
        );
        let answer = self.infer_text(&instruction, text)?;
        candidate_labels
            .iter()
            .find(|label| label.as_str() == answer.trim())
            .cloned()
            .ok_or_else(|| {
                StructuredError::new(
                    "invalid_classification",
                    "Model returned a label outside the candidate set",
                    true,
                )
            })
    }

    fn generate_title(&self, conversation_summary: &str) -> Result<String, StructuredError> {
        let answer = self.infer_text(
            "Return only a concise title of at most eight words for this input.",
            conversation_summary,
        )?;
        let title = answer.trim();
        if title.is_empty() || title.split_whitespace().count() > 8 {
            return Err(StructuredError::new(
                "invalid_title",
                "Model title must contain one to eight words",
                true,
            ));
        }
        Ok(title.into())
    }

    fn translate(&self, text: &str, target_language: &str) -> Result<String, StructuredError> {
        self.infer_text(
            &format!(
                "Translate the input into {}. Return only the translation.",
                serde_json::to_string(target_language)
                    .unwrap_or_else(|_| "\"unknown\"".into())
            ),
            text,
        )
    }

    fn judge(&self, criteria: &str, candidate: &str) -> Result<LocalJudgment, StructuredError> {
        let response = self.infer_text("Evaluate candidate against criteria. Return JSON with passed (boolean), score (number 0..1), rationale (string).", &serde_json::json!({"criteria":criteria,"candidate":candidate}).to_string())?;
        let judgment: LocalJudgment = serde_json::from_str(&response)
            .map_err(|error| StructuredError::new("invalid_judgment", error.to_string(), true))?;
        if !judgment.score.is_finite() || !(0.0..=1.0).contains(&judgment.score) {
            return Err(StructuredError::new(
                "invalid_judgment",
                "Judgment score must be within 0..1",
                true,
            ));
        }
        Ok(judgment)
    }

    fn synthesize_speech(&self, _text: &str) -> Result<Vec<u8>, StructuredError> {
        Err(StructuredError::new(
            "external_audio_unimplemented",
            "External local audio synthesis runtime not configured",
            false,
        ))
    }

    fn transcribe_speech(&self, _audio_bytes: &[u8]) -> Result<String, StructuredError> {
        Err(StructuredError::new(
            "external_audio_unimplemented",
            "External local audio transcription runtime not configured",
            false,
        ))
    }
}

/// Router dispatching local inference requests according to host policy.
#[derive(Clone, Debug)]
pub struct PolicyLocalEngine {
    policy: LocalEnginePolicy,
    heuristic: HeuristicLocalEngine,
    bundled: Option<BundledLocalEngine>,
    external: Option<ExternalLocalEngine>,
}

impl PolicyLocalEngine {
    pub fn from_policy(policy: LocalEnginePolicy) -> Self {
        let bundled = match &policy {
            LocalEnginePolicy::Bundled { model_path } => {
                Some(BundledLocalEngine::new(model_path.clone()))
            }
            _ => None,
        };
        let external = match &policy {
            LocalEnginePolicy::External { endpoint, model } => {
                Some(ExternalLocalEngine::new(endpoint.clone(), model.clone()))
            }
            _ => None,
        };
        Self {
            policy,
            heuristic: HeuristicLocalEngine::new(),
            bundled,
            external,
        }
    }

    pub fn from_env() -> Self {
        let policy = if let Ok(endpoint) = std::env::var("OMP_LOCAL_MODEL_ENDPOINT") {
            let model = std::env::var("OMP_LOCAL_MODEL_NAME").unwrap_or_else(|_| "llama3".into());
            LocalEnginePolicy::External { endpoint, model }
        } else if let Ok(path) = std::env::var("OMP_BUNDLED_MODEL_PATH") {
            LocalEnginePolicy::Bundled {
                model_path: Some(path),
            }
        } else {
            LocalEnginePolicy::Heuristic
        };
        Self::from_policy(policy)
    }

    fn active_engine(&self) -> Result<&dyn LocalModelEngine, StructuredError> {
        match &self.policy {
            LocalEnginePolicy::Heuristic => Ok(&self.heuristic),
            LocalEnginePolicy::Bundled { .. } => {
                if let Some(b) = &self.bundled {
                    Ok(b)
                } else {
                    Err(StructuredError::new(
                        "bundled_runtime_absent",
                        "Bundled engine not initialized",
                        false,
                    ))
                }
            }
            LocalEnginePolicy::External { .. } => {
                if let Some(e) = &self.external {
                    Ok(e)
                } else {
                    Err(StructuredError::new(
                        "external_runtime_absent",
                        "External engine not initialized",
                        false,
                    ))
                }
            }
            LocalEnginePolicy::Disabled => Err(StructuredError::new(
                "local_model_disabled",
                "Local model capabilities are disabled by policy",
                false,
            )),
        }
    }
}

impl LocalModelEngine for PolicyLocalEngine {
    fn classify(&self, text: &str, candidate_labels: &[String]) -> Result<String, StructuredError> {
        self.active_engine()?.classify(text, candidate_labels)
    }

    fn generate_title(&self, conversation_summary: &str) -> Result<String, StructuredError> {
        self.active_engine()?.generate_title(conversation_summary)
    }

    fn translate(&self, text: &str, target_language: &str) -> Result<String, StructuredError> {
        self.active_engine()?.translate(text, target_language)
    }

    fn judge(&self, criteria: &str, candidate: &str) -> Result<LocalJudgment, StructuredError> {
        self.active_engine()?.judge(criteria, candidate)
    }

    fn synthesize_speech(&self, text: &str) -> Result<Vec<u8>, StructuredError> {
        self.active_engine()?.synthesize_speech(text)
    }

    fn transcribe_speech(&self, audio_bytes: &[u8]) -> Result<String, StructuredError> {
        self.active_engine()?.transcribe_speech(audio_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, Read, Write};

    #[test]
    fn external_translation_consumes_model_response() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!(
            "http://{}/v1/chat/completions",
            listener.local_addr().unwrap()
        );
        listener.set_nonblocking(true).unwrap();
        let server = std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(peer) => break peer,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(started.elapsed() < std::time::Duration::from_secs(5));
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    Err(error) => panic!("{error}"),
                }
            };
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(request["model"], "translation-engine");
            let response = serde_json::json!({"choices":[{"message":{"content":"Merhaba dünya"},"finish_reason":"stop"}]}).to_string();
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", response.len(), response).unwrap();
        });
        let result =
            ExternalLocalEngine::new(endpoint, "translation-engine").translate("Hello world", "tr");
        server.join().unwrap();
        assert_eq!(result.unwrap(), "Merhaba dünya");
        assert!(HeuristicLocalEngine.translate("Hello world", "tr").is_err());
    }
}
