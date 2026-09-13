use omp_types::{ActorId, DirectorId, ElementId, SessionId, StructuredError, ToolCallId};
use serde::{Deserialize, Serialize};

pub use omp_inference::request::{
    InferenceRequest, MessageRole, SemanticMessage, ToolChoiceRequirement, ToolSchema,
};

pub type ControlResult<T> = Result<T, StructuredError>;

/// Semantic message alias for backwards compatibility.
pub type InferenceMessage = SemanticMessage;
/// Tool schema alias for backwards compatibility.
pub type ToolDefinitionSpec = ToolSchema;
/// Tool choice requirement alias for backwards compatibility.
pub type ToolChoiceIntent = ToolChoiceRequirement;

/// Read-only snapshot of agent execution context.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentView {
    pub actor_id: ActorId,
    pub session_id: SessionId,
    pub model: String,
    pub active_tools: Vec<String>,
    pub workspace_view: Option<String>,
    #[serde(default)]
    pub pending_todos: Vec<String>,
}

/// Read-only snapshot of the candidate turn being completed.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TurnView {
    pub turn_index: u32,
    pub assistant_text: Option<String>,
    pub tool_calls: Vec<ToolCallInfo>,
    #[serde(default)]
    pub executed_tools: Vec<ToolCallInfo>,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallInfo {
    pub id: ToolCallId,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Durable spec / snapshot of a Director's state in the session DOM.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DirectorSpec {
    pub id: DirectorId,
    pub kind: String,
    pub state: serde_json::Value,
    pub element_id: Option<ElementId>,
}

/// Decision returned by a Director when an agent is about to yield.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum YieldDecision {
    /// Pass control to the next outer Director.
    Pass,
    /// Instruct the agent to continue iterating with an optional prompt injection.
    Continue { prompt: Option<String> },
    /// Allow the agent to yield turn to the user / caller.
    Yield,
    /// Push a new child Director onto the stack.
    Push(DirectorSpec),
    /// Current Director has satisfied its goals and should be popped.
    Done,
    /// Halt execution with a structured error.
    Fail(StructuredError),
}

/// The core Director trait for cross-turn policy, reminders, and control ownership.
pub trait Director: Send + Sync {
    /// Unique identifier for this director instance.
    fn id(&self) -> &DirectorId;

    /// Kind name of the director.
    fn kind(&self) -> &str;

    /// Serialize current state to a durable DirectorSpec.
    fn to_spec(&self) -> DirectorSpec;

    /// Called outer-to-inner before submitting an inference request.
    fn prepare_inference(&mut self, request: InferenceRequest) -> ControlResult<InferenceRequest>;

    /// Called inner-to-outer when the agent loop considers yielding.
    fn on_yield(&mut self, agent: &AgentView, turn: &TurnView) -> ControlResult<YieldDecision>;
}

/// Failure behavior configuration when a ForceTool director exhausts attempts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Default)]
pub enum ForceToolFailureBehavior {
    #[default]
    Fail,
    YieldWithWarning,
}

/// Built-in Director enforcing semantic tool execution with bounded retries.
#[derive(Clone, Debug)]
pub struct ForceTool {
    pub id: DirectorId,
    pub tool: String,
    pub until_predicate_id: Option<String>,
    pub reminder: String,
    pub attempt_count: u32,
    pub max_attempts: u32,
    pub failure_behavior: ForceToolFailureBehavior,
}

impl ForceTool {
    pub fn new(
        id: DirectorId,
        tool: impl Into<String>,
        reminder: impl Into<String>,
        max_attempts: u32,
    ) -> Self {
        Self {
            id,
            tool: tool.into(),
            until_predicate_id: None,
            reminder: reminder.into(),
            attempt_count: 0,
            max_attempts: max_attempts.max(1),
            failure_behavior: ForceToolFailureBehavior::Fail,
        }
    }

    pub fn with_predicate(mut self, predicate_id: impl Into<String>) -> Self {
        self.until_predicate_id = Some(predicate_id.into());
        self
    }

    pub fn with_failure_behavior(mut self, behavior: ForceToolFailureBehavior) -> Self {
        self.failure_behavior = behavior;
        self
    }

    pub fn from_spec(spec: &DirectorSpec) -> ControlResult<Self> {
        let tool = spec
            .state
            .get("tool")
            .and_then(|v| v.as_str())
            .filter(|tool| !tool.trim().is_empty())
            .ok_or_else(|| {
                StructuredError::new(
                    "director_spec",
                    "ForceTool director requires a nonempty 'tool'",
                    false,
                )
            })?
            .to_string();
        let until_predicate_id = spec
            .state
            .get("until_predicate_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let reminder = spec
            .state
            .get("reminder")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let attempt_count = spec
            .state
            .get("attempt_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let max_attempts = spec
            .state
            .get("max_attempts")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;
        let failure_behavior = spec
            .state
            .get("failure_behavior")
            .and_then(|v| serde_json::from_value::<ForceToolFailureBehavior>(v.clone()).ok())
            .unwrap_or_default();

        Ok(Self {
            id: spec.id.clone(),
            tool,
            until_predicate_id,
            reminder,
            attempt_count,
            max_attempts,
            failure_behavior,
        })
    }
}

impl Director for ForceTool {
    fn id(&self) -> &DirectorId {
        &self.id
    }

    fn kind(&self) -> &str {
        "ForceTool"
    }

    fn to_spec(&self) -> DirectorSpec {
        DirectorSpec {
            id: self.id.clone(),
            kind: self.kind().to_string(),
            state: serde_json::json!({
                "tool": self.tool,
                "until_predicate_id": self.until_predicate_id,
                "reminder": self.reminder,
                "attempt_count": self.attempt_count,
                "max_attempts": self.max_attempts,
                "failure_behavior": self.failure_behavior,
            }),
            element_id: None,
        }
    }

    fn prepare_inference(
        &mut self,
        mut request: InferenceRequest,
    ) -> ControlResult<InferenceRequest> {
        // Enforce semantic tool choice intent
        request.tool_choice = ToolChoiceRequirement::Forced {
            tool: self.tool.clone(),
        };

        // Escalate with explicit instruction if this is a retry attempt
        if self.attempt_count > 0 {
            request.messages.push(SemanticMessage::system(format!(
                "[Director Reminder (attempt {}/{})]: You MUST invoke tool '{}'. {}",
                self.attempt_count + 1,
                self.max_attempts,
                self.tool,
                self.reminder
            )));
        }
        Ok(request)
    }

    fn on_yield(&mut self, _agent: &AgentView, turn: &TurnView) -> ControlResult<YieldDecision> {
        // Check if required tool was invoked in the candidate turn, including settled/executed tools
        let tool_invoked = turn.executed_tools.iter().any(|tc| tc.name == self.tool)
            || turn.tool_calls.iter().any(|tc| tc.name == self.tool);

        if tool_invoked {
            // Predicate / condition satisfied
            Ok(YieldDecision::Done)
        } else {
            self.attempt_count += 1;
            if self.attempt_count < self.max_attempts {
                Ok(YieldDecision::Continue {
                    prompt: Some(format!(
                        "Tool '{}' was required but not called. Attempt {}/{}. {}",
                        self.tool, self.attempt_count, self.max_attempts, self.reminder
                    )),
                })
            } else {
                match self.failure_behavior {
                    ForceToolFailureBehavior::Fail => {
                        Ok(YieldDecision::Fail(StructuredError::new(
                            "force_tool_exhausted",
                            format!(
                                "ForceTool exhausted: tool '{}' was not invoked after {} attempts",
                                self.tool, self.max_attempts
                            ),
                            false,
                        )))
                    }
                    ForceToolFailureBehavior::YieldWithWarning => Ok(YieldDecision::Yield),
                }
            }
        }
    }
}

/// Built-in Director for todo list reminders.
#[derive(Clone, Debug)]
pub struct TodoReminderDirector {
    pub id: DirectorId,
    pub pending_todos: Vec<String>,
    pub reminded: bool,
}

impl TodoReminderDirector {
    pub fn new(id: DirectorId, todos: Vec<String>) -> Self {
        Self {
            id,
            pending_todos: todos,
            reminded: false,
        }
    }

    pub fn from_spec(spec: &DirectorSpec) -> ControlResult<Self> {
        let pending_todos = spec
            .state
            .get("pending_todos")
            .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok())
            .unwrap_or_default();
        let reminded = spec
            .state
            .get("reminded")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        Ok(Self {
            id: spec.id.clone(),
            pending_todos,
            reminded,
        })
    }
}

impl Director for TodoReminderDirector {
    fn id(&self) -> &DirectorId {
        &self.id
    }

    fn kind(&self) -> &str {
        "TodoReminder"
    }

    fn to_spec(&self) -> DirectorSpec {
        DirectorSpec {
            id: self.id.clone(),
            kind: self.kind().to_string(),
            state: serde_json::json!({
                "pending_todos": self.pending_todos,
                "reminded": self.reminded,
            }),
            element_id: None,
        }
    }

    fn prepare_inference(&mut self, request: InferenceRequest) -> ControlResult<InferenceRequest> {
        Ok(request)
    }

    fn on_yield(&mut self, agent: &AgentView, _turn: &TurnView) -> ControlResult<YieldDecision> {
        // Prefer the live agent view; fall back to the persisted spec list so
        // a rehydrated director (fresh `AgentView::default()`) still reminds.
        let todos = if agent.pending_todos.is_empty() {
            &self.pending_todos
        } else {
            // Keep the persisted copy in sync for the next rehydration.
            self.pending_todos = agent.pending_todos.clone();
            &agent.pending_todos
        };

        if todos.is_empty() {
            return Ok(YieldDecision::Done);
        }

        if !self.reminded {
            self.reminded = true;
            let items = todos.join("; ");
            Ok(YieldDecision::Continue {
                prompt: Some(format!("Reminder: you have pending todos: {}", items)),
            })
        } else {
            Ok(YieldDecision::Pass)
        }
    }
}

/// Built-in Director for plan mode.
/// Plan mode requires reading/creating the plan file and either proposing a plan
/// or asking a clarifying question before yielding.
#[derive(Clone, Debug)]
pub struct PlanModeDirector {
    pub id: DirectorId,
    pub plan_path: String,
    pub plan_file_inspected: bool,
    pub proposed_plan: bool,
    pub user_question_asked: bool,
}

impl PlanModeDirector {
    pub fn new(id: DirectorId, plan_path: impl Into<String>) -> Self {
        Self {
            id,
            plan_path: plan_path.into(),
            plan_file_inspected: false,
            proposed_plan: false,
            user_question_asked: false,
        }
    }

    pub fn mark_plan_inspected(&mut self) {
        self.plan_file_inspected = true;
    }

    pub fn mark_plan_proposed(&mut self) {
        self.proposed_plan = true;
    }

    pub fn mark_question_asked(&mut self) {
        self.user_question_asked = true;
    }

    pub fn from_spec(spec: &DirectorSpec) -> ControlResult<Self> {
        let plan_path = spec
            .state
            .get("plan_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let plan_file_inspected = spec
            .state
            .get("plan_file_inspected")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let proposed_plan = spec
            .state
            .get("proposed_plan")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let user_question_asked = spec
            .state
            .get("user_question_asked")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        Ok(Self {
            id: spec.id.clone(),
            plan_path,
            plan_file_inspected,
            proposed_plan,
            user_question_asked,
        })
    }
}

impl Director for PlanModeDirector {
    fn id(&self) -> &DirectorId {
        &self.id
    }

    fn kind(&self) -> &str {
        "PlanMode"
    }

    fn to_spec(&self) -> DirectorSpec {
        DirectorSpec {
            id: self.id.clone(),
            kind: self.kind().to_string(),
            state: serde_json::json!({
                "plan_path": self.plan_path,
                "plan_file_inspected": self.plan_file_inspected,
                "proposed_plan": self.proposed_plan,
                "user_question_asked": self.user_question_asked,
            }),
            element_id: None,
        }
    }

    fn prepare_inference(
        &mut self,
        mut request: InferenceRequest,
    ) -> ControlResult<InferenceRequest> {
        request.messages.push(SemanticMessage::system(format!(
            "[PlanMode Director]: Plan mode is ACTIVE for '{}'. You must inspect the plan file and either present a plan or ask a user question before yielding.",
            self.plan_path
        )));
        Ok(request)
    }

    fn on_yield(&mut self, _agent: &AgentView, turn: &TurnView) -> ControlResult<YieldDecision> {
        // Inspect text in turn for question or proposal using word-boundary
        // matching so e.g. "questionnaire" or "airplane" do not satisfy the gate.
        if let Some(text) = &turn.assistant_text {
            if contains_word(&text.to_lowercase(), "question") || text.contains('?') {
                self.user_question_asked = true;
            }
            if contains_word(&text.to_lowercase(), "plan") {
                self.proposed_plan = true;
            }
        }

        // Check if any tool inspected the exact plan file path: require an exact
        // match or a separator-bounded path suffix so "plans2.md" or
        // "myplan.md" cannot satisfy a gate on "plan.md".
        let matches_plan = |args: &serde_json::Value| -> bool {
            if let Some(path_val) = args.get("path").and_then(|v| v.as_str()) {
                path_suffix_match(path_val, &self.plan_path)
                    || path_suffix_match(&self.plan_path, path_val)
            } else {
                false
            }
        };

        for tc in turn.executed_tools.iter().chain(turn.tool_calls.iter()) {
            let n = tc.name.to_ascii_lowercase();
            if (n == "read" || n == "write" || n == "edit") && matches_plan(&tc.arguments) {
                self.plan_file_inspected = true;
            }
        }

        if !self.plan_file_inspected {
            return Ok(YieldDecision::Continue {
                prompt: Some(format!(
                    "Plan mode requires accessing the plan file '{}' before yielding.",
                    self.plan_path
                )),
            });
        }

        if !self.proposed_plan && !self.user_question_asked {
            return Ok(YieldDecision::Continue {
                prompt: Some(
                    "Plan mode requires either proposing a plan or asking the user a clarifying question before yielding.".into(),
                ),
            });
        }

        // Requirements satisfied; pass to outer directors
        Ok(YieldDecision::Pass)
    }
}

/// Returns true when `word` appears in `text` as a standalone word: bounded on
/// both sides by a non-alphanumeric character or string edge.
/// Word-boundary substring match. Implemented over `match_indices` (always
/// char-boundary safe) instead of raw byte indexing, so multibyte assistant
/// text can never panic the director gate.
fn contains_word(text: &str, word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    for (start, _) in text.match_indices(word) {
        let left_ok = text[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric());
        let right_ok = text[start + word.len()..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric());
        if left_ok && right_ok {
            return true;
        }
    }
    false
}

/// Returns true when `candidate` equals `expected` or ends with it at a path
/// separator boundary (`/` or `\`), so unrelated names sharing a string suffix
/// do not match. Both directions are checked by the caller.
fn path_suffix_match(candidate: &str, expected: &str) -> bool {
    if candidate == expected {
        return true;
    }
    if candidate.len() <= expected.len() {
        return false;
    }
    if !candidate.ends_with(expected) {
        return false;
    }
    matches!(
        candidate.as_bytes()[candidate.len() - expected.len() - 1],
        b'/' | b'\\'
    )
}

/// Built-in Director for goal completion checks.
#[derive(Clone, Debug)]
pub struct GoalCompletionDirector {
    pub id: DirectorId,
    pub goal_description: String,
    pub completed: bool,
}

impl GoalCompletionDirector {
    pub fn new(id: DirectorId, goal: impl Into<String>) -> Self {
        Self {
            id,
            goal_description: goal.into(),
            completed: false,
        }
    }
    pub fn mark_completed(&mut self) {
        self.completed = true;
    }
    pub fn from_spec(spec: &DirectorSpec) -> ControlResult<Self> {
        let goal_description = spec
            .state
            .get("goal_description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let completed = spec
            .state
            .get("completed")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        Ok(Self {
            id: spec.id.clone(),
            goal_description,
            completed,
        })
    }
}

impl Director for GoalCompletionDirector {
    fn id(&self) -> &DirectorId {
        &self.id
    }

    fn kind(&self) -> &str {
        "GoalCompletion"
    }

    fn to_spec(&self) -> DirectorSpec {
        DirectorSpec {
            id: self.id.clone(),
            kind: self.kind().to_string(),
            state: serde_json::json!({
                "goal_description": self.goal_description,
                "completed": self.completed,
            }),
            element_id: None,
        }
    }

    fn prepare_inference(&mut self, request: InferenceRequest) -> ControlResult<InferenceRequest> {
        Ok(request)
    }

    fn on_yield(&mut self, _agent: &AgentView, _turn: &TurnView) -> ControlResult<YieldDecision> {
        if self.completed {
            Ok(YieldDecision::Done)
        } else {
            Ok(YieldDecision::Pass)
        }
    }
}

/// Create a Director instance from a durable journal-derived DirectorSpec.
pub fn create_director_from_spec(spec: &DirectorSpec) -> ControlResult<Box<dyn Director>> {
    match spec.kind.as_str() {
        "ForceTool" => Ok(Box::new(ForceTool::from_spec(spec)?)),
        "TodoReminder" => Ok(Box::new(TodoReminderDirector::from_spec(spec)?)),
        "PlanMode" => Ok(Box::new(PlanModeDirector::from_spec(spec)?)),
        "GoalCompletion" => Ok(Box::new(GoalCompletionDirector::from_spec(spec)?)),
        other => Err(StructuredError::new(
            "unknown_director_kind",
            format!("unknown director kind: '{}'", other),
            false,
        )),
    }
}
