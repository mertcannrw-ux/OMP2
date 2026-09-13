use crate::director::{
    AgentView, ControlResult, Director, DirectorSpec, InferenceRequest, ToolCallInfo, TurnView,
    YieldDecision,
};
use omp_types::{
    ActorId, ElementId, ElementSnapshot, JournalOffset, Patch, PatchOp, StructuredError, TypedValue,
};
use std::collections::BTreeMap;

/// Stack of active Directors representing cross-turn policy and control constraints.
/// Directors walk outer-to-inner on `prepare_inference` and inner-to-outer on `on_yield`.
#[derive(Default)]
pub struct DirectorStack {
    directors: Vec<Box<dyn Director>>,
}

impl DirectorStack {
    pub fn new() -> Self {
        Self {
            directors: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.directors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.directors.is_empty()
    }

    pub fn push(&mut self, director: Box<dyn Director>) {
        self.directors.push(director);
    }

    pub fn pop(&mut self) -> Option<Box<dyn Director>> {
        self.directors.pop()
    }

    pub fn clear(&mut self) {
        self.directors.clear();
    }

    /// Prepare inference request by walking Directors outer-to-inner (index 0 to n-1).
    pub fn prepare_inference(
        &mut self,
        mut request: InferenceRequest,
    ) -> ControlResult<InferenceRequest> {
        for director in &mut self.directors {
            request = director.prepare_inference(request)?;
        }
        Ok(request)
    }

    /// Evaluate candidate yield by walking Directors inner-to-outer (index n-1 down to 0).
    pub fn handle_yield(
        &mut self,
        agent: &AgentView,
        turn: &TurnView,
    ) -> ControlResult<YieldDecision> {
        let mut idx = self.directors.len();
        while idx > 0 {
            idx -= 1;
            let decision = self.directors[idx].on_yield(agent, turn)?;
            match decision {
                YieldDecision::Pass => {
                    // Continue to next outer director
                    continue;
                }
                YieldDecision::Done => {
                    // Pop completed director and continue yielding to outer directors
                    self.directors.remove(idx);
                    continue;
                }
                other => {
                    // Inner director made an active decision (Continue, Yield, Push, Fail)
                    return Ok(other);
                }
            }
        }

        // All directors passed or stack is empty: allow yield
        Ok(YieldDecision::Yield)
    }

    /// Collect durable specs for all active directors on the stack.
    pub fn to_specs(&self) -> Vec<DirectorSpec> {
        self.directors.iter().map(|d| d.to_spec()).collect()
    }

    /// Reconstruct DirectorStack from journal-derived DirectorSpec list.
    pub fn from_specs(specs: &[DirectorSpec]) -> ControlResult<Self> {
        let mut stack = Self::new();
        for spec in specs {
            stack.push(crate::director::create_director_from_spec(spec)?);
        }
        Ok(stack)
    }

    /// Hydrate DirectorStack directly from an authoritative SessionSnapshot.
    pub fn from_session_snapshot(snapshot: &omp_state::SessionSnapshot) -> ControlResult<Self> {
        let mut specs = Vec::new();
        for elem in snapshot.active_directors() {
            let kind = elem
                .attributes
                .get("kind")
                .and_then(|v| match v {
                    TypedValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| elem.kind.clone());

            let dir_id = omp_types::DirectorId::new(elem.id.as_str()).unwrap_or_else(|_| {
                omp_types::DirectorId::new("director").expect("valid director id")
            });

            specs.push(DirectorSpec {
                id: dir_id,
                kind,
                state: elem.payload.clone().unwrap_or(serde_json::Value::Null),
                element_id: Some(elem.id.clone()),
            });
        }
        Self::from_specs(&specs)
    }
}

/// Action decision for the host agent loop.
#[derive(Clone, Debug)]
pub enum AgentTurnAction {
    /// Tool calls were emitted; host loop must execute them before yielding.
    ExecuteTools(Vec<ToolCallInfo>),
    /// Director instructed to continue turn with an optional prompt injection.
    ContinueInference { prompt: Option<String> },
    /// Turn completed and approved to yield to the user.
    YieldToUser,
    /// A new Director was pushed onto the stack.
    PushDirector(DirectorSpec),
    /// Loop must halt due to a structured error.
    HaltWithError(StructuredError),
}

/// The boring host loop decision point:
/// 1. Execute tool calls if any remain.
/// 2. Call `on_yield` on the Director stack only when no tool calls remain.
/// 3. Map the yield decision into an actionable AgentTurnAction.
pub fn evaluate_agent_turn(
    stack: &mut DirectorStack,
    agent: &AgentView,
    turn: &TurnView,
) -> ControlResult<AgentTurnAction> {
    // If tool calls were returned by the model, execute them first.
    // A candidate yield is never evaluated until tool calls have finished.
    if !turn.tool_calls.is_empty() {
        return Ok(AgentTurnAction::ExecuteTools(turn.tool_calls.clone()));
    }

    // No tool calls remain: candidate yield evaluation
    let decision = stack.handle_yield(agent, turn)?;
    match decision {
        YieldDecision::Continue { prompt } => Ok(AgentTurnAction::ContinueInference { prompt }),
        YieldDecision::Yield | YieldDecision::Pass | YieldDecision::Done => {
            Ok(AgentTurnAction::YieldToUser)
        }
        YieldDecision::Push(spec) => Ok(AgentTurnAction::PushDirector(spec)),
        YieldDecision::Fail(err) => Ok(AgentTurnAction::HaltWithError(err)),
    }
}

/// Helper to generate DOM patch operations representing Director stack state.
/// Maps journal-derived director ids to StructuredError instead of panicking,
/// so a corrupt journal id fails the patch instead of the process.
pub fn director_spec_to_patch_op(
    parent_element: &ElementId,
    index: u32,
    spec: &DirectorSpec,
) -> Result<PatchOp, StructuredError> {
    let mut attributes = BTreeMap::new();
    attributes.insert("kind".into(), TypedValue::String(spec.kind.clone()));

    let elem_id = match spec.element_id.clone() {
        Some(id) => id,
        None => ElementId::new(format!("director-{}", spec.id)).map_err(|e| {
            StructuredError::new(
                "invalid_director_id",
                format!("Journal-derived director id is not a valid element id: {e}"),
                false,
            )
        })?,
    };

    Ok(PatchOp::Create {
        parent: parent_element.clone(),
        index,
        element: ElementSnapshot {
            id: elem_id,
            schema_version: 1,
            kind: "director".into(),
            attributes,
            text: String::new(),
            payload: Some(spec.state.clone()),
        },
    })
}

/// Helper to create a patch reflecting director push.
pub fn create_director_push_patch(
    base_offset: JournalOffset,
    result_offset: JournalOffset,
    by: ActorId,
    parent_element: &ElementId,
    index: u32,
    spec: &DirectorSpec,
) -> Result<Patch, StructuredError> {
    Ok(Patch {
        base_offset,
        result_offset,
        by: by.into(),
        reason: format!("push director {}", spec.kind),
        ops: vec![director_spec_to_patch_op(parent_element, index, spec)?],
    })
}
