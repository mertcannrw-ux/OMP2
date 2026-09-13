use crate::capability::{CapabilityProfile, NativeToolCost, ProviderCapability};
use crate::request::{
    InferenceRequest, MessageContent, MessageRole, SemanticMessage, ToolChoiceRequirement,
};
use omp_types::StructuredError;
use serde::{Deserialize, Serialize};

/// Action taken during turn preparation under tool force policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolForceAction {
    /// Soft instruction injected; native tool force omitted (either costly or unsupported).
    SoftInstructionOnly { tool: String, attempt: u32 },
    /// Native tool choice enforced because it is side-effect-free.
    NativeForceFree { tool: String },
    /// Non-compliance observed on earlier attempt; escalated to native force despite cost.
    NativeForceEscalated { tool: String, attempt: u32 },
}

/// Outcome after inspecting assistant response tool calls.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolForceOutcome {
    /// Target tool was successfully called by the model.
    Satisfied { tool: String, attempts: u32 },
    /// Target tool was not called; bounded retry needed.
    RetryRequired {
        tool: String,
        attempt: u32,
        max_attempts: u32,
        escalate_native: bool,
    },
}

/// Cost-aware escalation state machine for forced tool requirements.
///
/// Invariants:
/// 1. Always injects a clear soft instruction into conversation context.
/// 2. Passes native force initially ONLY when supported AND side-effect-free (Free).
/// 3. If bounded retries show non-compliance, escalates to native force even if costly.
/// 4. If still unsupported or non-compliant after bounded retries, returns a structured error.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolForcePolicy {
    pub tool: String,
    pub max_attempts: u32,
    pub current_attempt: u32,
    pub escalated_to_costly: bool,
    pub custom_reminder: Option<String>,
}

impl ToolForcePolicy {
    pub fn new(tool: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            max_attempts: 3,
            current_attempt: 0,
            escalated_to_costly: false,
            custom_reminder: None,
        }
    }

    pub fn with_max_attempts(mut self, max: u32) -> Self {
        self.max_attempts = max.max(1);
        self
    }

    pub fn with_reminder(mut self, reminder: impl Into<String>) -> Self {
        self.custom_reminder = Some(reminder.into());
        self
    }

    /// Prepares the inference request under the cost-aware escalation policy.
    ///
    /// NOTE: each call pushes one `User` directive message, so repeated
    /// retries grow the history by one message per attempt. This is bounded
    /// by `max_attempts` (default 3) and intentional — the escalation ladder
    /// needs the reminder visible in-context on every retry.
    pub fn prepare_request(
        &mut self,
        req: &mut InferenceRequest,
        caps: &CapabilityProfile,
    ) -> Result<ToolForceAction, StructuredError> {
        if self.current_attempt >= self.max_attempts {
            let mut err = StructuredError::new(
                "exhausted_tool_requirement",
                format!(
                    "Forced tool requirement for '{}' exhausted after {} attempts",
                    self.tool, self.max_attempts
                ),
                false,
            );
            err.diagnostics = Some(serde_json::json!({
                "tool": self.tool,
                "attempts": self.current_attempt,
                "max_attempts": self.max_attempts,
                "escalated": self.escalated_to_costly,
            }));
            return Err(err);
        }

        // 1. Always inject a soft instruction
        let prompt_text = match &self.custom_reminder {
            Some(rem) => format!(
                "[Directive: You MUST call tool '{}' next. Instruction: {}]",
                self.tool, rem
            ),
            None => format!(
                "[Directive: You MUST call tool '{}' with valid parameters in your next response.]",
                self.tool
            ),
        };
        req.messages.push(SemanticMessage {
            role: MessageRole::User,
            content: MessageContent::Text(prompt_text),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: Some("tool_force_directive".into()),
        });

        // 2. Determine native tool choice escalation
        let supports_native = caps.supports(ProviderCapability::NativeToolChoice);
        let cost = caps.native_tool_cost;

        if !supports_native {
            // Cannot escalate to native; rely purely on soft instruction
            req.tool_choice = ToolChoiceRequirement::Auto;
            Ok(ToolForceAction::SoftInstructionOnly {
                tool: self.tool.clone(),
                attempt: self.current_attempt,
            })
        } else if cost == NativeToolCost::Free {
            // Side-effect-free: pass native force immediately
            req.tool_choice = ToolChoiceRequirement::Forced {
                tool: self.tool.clone(),
            };
            Ok(ToolForceAction::NativeForceFree {
                tool: self.tool.clone(),
            })
        } else {
            // Costly or unknown: soft instruction on first attempt; escalate on subsequent attempts
            if self.current_attempt == 0 {
                req.tool_choice = ToolChoiceRequirement::Auto;
                Ok(ToolForceAction::SoftInstructionOnly {
                    tool: self.tool.clone(),
                    attempt: 0,
                })
            } else {
                self.escalated_to_costly = true;
                req.tool_choice = ToolChoiceRequirement::Forced {
                    tool: self.tool.clone(),
                };
                Ok(ToolForceAction::NativeForceEscalated {
                    tool: self.tool.clone(),
                    attempt: self.current_attempt,
                })
            }
        }
    }

    /// Evaluates the turn results against the forced tool requirement.
    pub fn record_turn_result(
        &mut self,
        tools_called: &[String],
    ) -> Result<ToolForceOutcome, StructuredError> {
        if tools_called.iter().any(|t| t == &self.tool) {
            return Ok(ToolForceOutcome::Satisfied {
                tool: self.tool.clone(),
                attempts: self.current_attempt + 1,
            });
        }

        // Model did not comply
        self.current_attempt += 1;

        if self.current_attempt >= self.max_attempts {
            let mut err = StructuredError::new(
                "exhausted_tool_requirement",
                format!(
                    "Model failed to invoke required tool '{}' within {} attempts",
                    self.tool, self.max_attempts
                ),
                false,
            );
            err.diagnostics = Some(serde_json::json!({
                "tool": self.tool,
                "attempts": self.current_attempt,
                "max_attempts": self.max_attempts,
                "escalated": self.escalated_to_costly,
                "tools_actually_called": tools_called,
            }));
            return Err(err);
        }

        Ok(ToolForceOutcome::RetryRequired {
            tool: self.tool.clone(),
            attempt: self.current_attempt,
            max_attempts: self.max_attempts,
            escalate_native: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::TriState;

    #[test]
    fn test_forced_tool_free_native_immediate() {
        let mut policy = ToolForcePolicy::new("Write");
        let mut req = InferenceRequest::default();
        let mut caps = CapabilityProfile::new();
        caps.set(ProviderCapability::NativeToolChoice, TriState::Supported);
        caps.native_tool_cost = NativeToolCost::Free;

        let action = policy.prepare_request(&mut req, &caps).unwrap();
        assert_eq!(
            action,
            ToolForceAction::NativeForceFree {
                tool: "Write".into()
            }
        );
        assert_eq!(
            req.tool_choice,
            ToolChoiceRequirement::Forced {
                tool: "Write".into()
            }
        );
        assert!(
            req.messages
                .iter()
                .any(|m| m.name.as_deref() == Some("tool_force_directive"))
        );
    }

    #[test]
    fn test_forced_tool_costly_escalation() {
        let mut policy = ToolForcePolicy::new("Read").with_max_attempts(2);
        let mut req = InferenceRequest::default();
        let mut caps = CapabilityProfile::new();
        caps.set(ProviderCapability::NativeToolChoice, TriState::Supported);
        caps.native_tool_cost = NativeToolCost::Costly;

        // Attempt 0: Soft instruction only
        let action0 = policy.prepare_request(&mut req, &caps).unwrap();
        assert_eq!(
            action0,
            ToolForceAction::SoftInstructionOnly {
                tool: "Read".into(),
                attempt: 0
            }
        );
        assert_eq!(req.tool_choice, ToolChoiceRequirement::Auto);

        // Non-compliance
        let outcome = policy.record_turn_result(&[]).unwrap();
        assert!(matches!(
            outcome,
            ToolForceOutcome::RetryRequired {
                escalate_native: true,
                ..
            }
        ));

        // Attempt 1: Escalated to native force despite cost
        let mut req2 = InferenceRequest::default();
        let action1 = policy.prepare_request(&mut req2, &caps).unwrap();
        assert_eq!(
            action1,
            ToolForceAction::NativeForceEscalated {
                tool: "Read".into(),
                attempt: 1
            }
        );
        assert_eq!(
            req2.tool_choice,
            ToolChoiceRequirement::Forced {
                tool: "Read".into()
            }
        );

        // Second failure -> exhaustion
        let err = policy.record_turn_result(&[]).unwrap_err();
        assert_eq!(err.code, "exhausted_tool_requirement");
    }

    #[test]
    fn test_forced_tool_unsupported_native() {
        let mut policy = ToolForcePolicy::new("Edit").with_max_attempts(2);
        let mut req = InferenceRequest::default();
        let mut caps = CapabilityProfile::new();
        caps.set(ProviderCapability::NativeToolChoice, TriState::Unsupported);

        // Attempt 0: Soft only
        let action0 = policy.prepare_request(&mut req, &caps).unwrap();
        assert_eq!(
            action0,
            ToolForceAction::SoftInstructionOnly {
                tool: "Edit".into(),
                attempt: 0
            }
        );

        policy.record_turn_result(&[]).unwrap();

        // Attempt 1: Still soft only because native is unsupported
        let mut req2 = InferenceRequest::default();
        let action1 = policy.prepare_request(&mut req2, &caps).unwrap();
        assert_eq!(
            action1,
            ToolForceAction::SoftInstructionOnly {
                tool: "Edit".into(),
                attempt: 1
            }
        );

        // Exhaustion
        let err = policy.record_turn_result(&[]).unwrap_err();
        assert_eq!(err.code, "exhausted_tool_requirement");
    }

    #[test]
    fn test_forced_tool_satisfied() {
        let mut policy = ToolForcePolicy::new("Bash");
        let outcome = policy.record_turn_result(&["Bash".to_string()]).unwrap();
        assert_eq!(
            outcome,
            ToolForceOutcome::Satisfied {
                tool: "Bash".into(),
                attempts: 1
            }
        );
    }
}
