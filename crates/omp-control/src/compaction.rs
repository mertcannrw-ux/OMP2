//! Engine-side, lossless context compaction.
//!
//! # Why this shape
//!
//! Deterministic elision, not an LLM summary pass, is the default because the
//! controlled evidence does not support summarization as an automatic win:
//! observation masking matched or beat LLM summarization at roughly half the
//! cost across five model configurations on SWE-bench Verified, and on one of
//! them summarization measurably *hurt* (40.4% → 31.4% solve rate, arXiv
//! 2508.21433); a frontier endpoint told its exact workload still answered
//! set-membership queries at random-guess accuracy after compacting (0.505 vs a
//! 0.5 coin flip, control 0.02 — arXiv 2608.01326). Generation is provably the
//! stronger primitive in the worst case, so the DAG here is built to carry model
//! written text later — a node's text is one `ReplaceText` patch away — without
//! making automatic compaction depend on a second model call today.
//!
//! # What makes it lossless
//!
//! Reachability, not the summary text. The journal keeps every element; a
//! [`omp_types::SummaryNode`] records only which body elements the provider
//! projection leaves out. `Read summary://<id>` renders those originals back,
//! byte for byte, so the inventory text is allowed to be lossy while the
//! history is not.
//!
//! # Bounds
//!
//! Three stages run in order, each strictly reducing the projection: elision
//! (drop old elements, keep an inventory and a marker), condensation (fold the
//! oldest nodes into one parent, which loses text but keeps reachability), then
//! reduction of what is left — tool observations first (their arguments, then
//! their output), then assistant text, then the user's own words, and never
//! pinned instructions. Every stage makes progress or the loop stops, so
//! `plan()` always terminates. Nothing ever elides pinned instructions or the
//! turn in flight.
//!
//! Every reduction the planner decides is also one the projection realizes: a
//! saving the wire does not take would let the loop exit above the hard bound.

use omp_inference::compaction::{CompactionBudget, estimate_tokens};
use omp_inference::request::MessageFold;
use omp_state::{Journal, SessionSnapshot};
use omp_types::{
    ActorId, ElementId, ElementSnapshot, JournalOffset, Patch, PatchOp, StructuredError,
    SUMMARIES_CONTAINER, SummaryKind, SummaryNode, ToolCallId, TypedValue, expand_covered,
};
use std::collections::{BTreeMap, BTreeSet};

/// Completed turns kept verbatim below a compaction pass.
pub const DEFAULT_KEEP_RECENT_TURNS: usize = 2;
/// Root nodes tolerated before the oldest ones are condensed into one parent.
pub const DEFAULT_MAX_ROOT_NODES: usize = 8;
/// Root nodes folded together by one condensation.
pub const CONDENSE_FAN_IN: usize = 4;
/// Upper bound on a node's inventory text before it is capped.
pub const MAX_NODE_TEXT_BYTES: usize = 8_000;
/// Preview budget for one inventory line.
pub const MAX_PREVIEW_CHARS: usize = 96;
/// Conservative inventory cost of one covered element, including its preview
/// line and the node header. Covering an element that projects to fewer tokens
/// than this costs more than it saves, so such an element is left alone until
/// the hard bound demands otherwise.
pub const INVENTORY_LINE_TOKENS: usize = 40;
/// Header cost of one node's projection (marker plus range description).
pub const NODE_HEADER_TOKENS: usize = 24;
/// Characters a reduced (kept, not elided) element preserves from each end.
pub const TRUNCATED_HEAD_CHARS: usize = 400;
pub const TRUNCATED_TAIL_CHARS: usize = 200;
/// Bytes a reduced element may project to, marker included.
pub const TRUNCATED_BYTES: usize = TRUNCATED_HEAD_CHARS + TRUNCATED_TAIL_CHARS + 96;
/// The smallest tool-call argument payload worth replacing with a marker.
pub const MIN_OMITTED_ARGUMENT_TOKENS: usize = 64;

/// Everything the planner needs that the document tree does not carry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionConfig {
    pub budget: CompactionBudget,
    /// Turns kept verbatim below a compaction pass. At least one turn — the one
    /// in flight — is always kept, whatever this says.
    pub keep_recent_turns: usize,
    /// Root nodes tolerated before condensing.
    pub max_root_nodes: usize,
    /// Tokens spent on the tool roster and other fixed request scaffolding that
    /// the planner cannot see from the body.
    pub fixed_overhead_tokens: usize,
}

/// One reduction applied to a kept element's projection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Reduction {
    /// Output/text capped to the head-and-tail budget.
    capped_text: bool,
    /// Tool-call arguments replaced by a marker object.
    omitted_arguments: bool,
}

/// What the host must do to its projection this turn.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompactionOutcome {
    /// Fold markers, oldest node first, ready to prepend to the request.
    pub folds: Vec<MessageFold>,
    /// Body elements left out of the projection.
    pub covered: BTreeSet<ElementId>,
    reductions: BTreeMap<ElementId, Reduction>,
}

impl CompactionOutcome {
    /// Applies this outcome's cap to one element's projected text.
    pub fn project_text(&self, id: &ElementId, text: &str) -> String {
        match self.reductions.get(id) {
            Some(reduction) if reduction.capped_text => truncate_for_projection(text),
            _ => text.to_string(),
        }
    }

    /// True when the element's tool-call arguments must be replaced by a marker.
    pub fn omits_arguments(&self, id: &ElementId) -> bool {
        self.reductions
            .get(id)
            .is_some_and(|reduction| reduction.omitted_arguments)
    }
}

/// Caps `text`, keeping both ends and saying what happened to the middle.
pub fn truncate_for_projection(text: &str) -> String {
    let limit = TRUNCATED_HEAD_CHARS + TRUNCATED_TAIL_CHARS;
    if text.len() <= limit {
        return text.to_string();
    }
    let head = text_prefix(text, TRUNCATED_HEAD_CHARS);
    let tail = text_suffix(text, TRUNCATED_TAIL_CHARS);
    let dropped = text.len().saturating_sub(head.len() + tail.len());
    format!(
        "{head}\n[…{dropped} chars truncated to fit the context window; the journal keeps the full text]\n{tail}"
    )
}

/// Projects `text` to its first `limit` characters on a UTF-8 boundary.
pub(crate) fn text_prefix(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_string();
    }
    let mut end = limit;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

/// Projects `text` to its last `limit` characters on a UTF-8 boundary.
pub(crate) fn text_suffix(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_string();
    }
    let mut start = text.len() - limit;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_string()
}

/// Upper bound on what a reduced element projects to. The cap is a byte budget
/// and the estimator charges at most one token per byte, so the budget itself is
/// the bound.
fn reduced_tokens() -> usize {
    TRUNCATED_BYTES
}

/// How one body element participates in the projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EntryRole {
    User,
    Assistant,
    ToolCall,
    Instructions,
}

/// One body element as the projection sees it.
#[derive(Clone, Debug)]
struct BodyEntry {
    id: ElementId,
    role: EntryRole,
    /// Tokens of the tool-call arguments, which the projection sends verbatim
    /// unless the reduction below replaces them.
    args_tokens: usize,
    /// Tokens of the element's own text: the message body, or a tool result.
    text_tokens: usize,
    /// Instructions and other pinned content that must always be sent.
    pinned: bool,
    /// Part of the turn in flight, which is never elided.
    in_flight: bool,
    /// One-line inventory preview used when the element is covered.
    preview: String,
}

impl BodyEntry {
    fn eligible(&self) -> bool {
        !self.pinned && !self.in_flight
    }

    fn tokens(&self) -> usize {
        let overhead = if self.role == EntryRole::ToolCall { 8 } else { 0 };
        self.args_tokens + self.text_tokens + overhead
    }

    /// The next reduction this element can take, with the tokens it saves.
    fn next_reduction(&self, applied: Option<&Reduction>) -> Option<(Reduction, usize)> {
        let applied = applied.copied().unwrap_or_default();
        let effective_text = if applied.capped_text {
            reduced_tokens().min(self.text_tokens)
        } else {
            self.text_tokens
        };
        // Oversized tool arguments go before anything else: they are input the
        // model already acted on, and the call is still legible without them.
        if self.role == EntryRole::ToolCall
            && !applied.omitted_arguments
            && self.args_tokens > MIN_OMITTED_ARGUMENT_TOKENS
            && self.args_tokens > effective_text
        {
            return Some((
                Reduction {
                    omitted_arguments: true,
                    ..applied
                },
                self.args_tokens,
            ));
        }
        if !applied.capped_text && self.text_tokens > reduced_tokens() {
            return Some((
                Reduction {
                    capped_text: true,
                    ..applied
                },
                self.text_tokens - reduced_tokens(),
            ));
        }
        None
    }
}

/// The summary DAG as loaded from one snapshot.
#[derive(Clone, Debug, Default)]
struct SummaryDag {
    nodes: BTreeMap<ElementId, SummaryNode>,
    texts: BTreeMap<ElementId, String>,
    order: Vec<ElementId>,
    children: BTreeSet<ElementId>,
    /// Nodes whose payload could not be decoded. The planner refuses to elide
    /// while any exist, because their covered elements are unaccounted for.
    unreadable: usize,
}

impl SummaryDag {
    fn load(snapshot: &SessionSnapshot) -> Self {
        let mut dag = Self::default();
        for element in snapshot.children(snapshot.container(SUMMARIES_CONTAINER)) {
            let Ok(node) = SummaryNode::from_element(element) else {
                dag.unreadable += 1;
                continue;
            };
            dag.order.push(element.id.clone());
            dag.children.extend(node.children.iter().cloned());
            dag.texts.insert(element.id.clone(), element.text.clone());
            dag.nodes.insert(element.id.clone(), node);
        }
        dag
    }

    /// Nodes the projection renders: everything that is nobody's child.
    fn roots(&self) -> Vec<ElementId> {
        self.order
            .iter()
            .filter(|id| !self.children.contains(*id))
            .cloned()
            .collect()
    }

    fn covered(&self) -> BTreeSet<ElementId> {
        self.nodes
            .values()
            .flat_map(|node| node.covered.iter().cloned())
            .collect()
    }

    fn token_index(&self) -> BTreeMap<ElementId, usize> {
        self.texts
            .iter()
            .map(|(id, text)| (id.clone(), estimate_tokens(text)))
            .collect()
    }

    fn projection_tokens(&self, tokens: &BTreeMap<ElementId, usize>) -> usize {
        self.roots()
            .iter()
            .map(|id| tokens.get(id).copied().unwrap_or(0) + NODE_HEADER_TOKENS)
            .sum()
    }

    fn folds(&self) -> Vec<MessageFold> {
        self.roots()
            .iter()
            .map(|id| {
                let node = self.nodes.get(id).cloned().unwrap_or_default();
                MessageFold::Handoff {
                    summary: render_fold(id, self, &node),
                    folded_count: expand_covered(&self.nodes, id).len(),
                    original_token_count: node.covered_tokens as usize,
                }
            })
            .collect()
    }
}

/// Plans and durably records this turn's compaction.
///
/// The journaled nodes make the decision itself replayable, forkable and
/// replicable; the returned outcome only tells the caller how to project the
/// request right now.
pub fn plan(
    journal: &mut Journal,
    owner: &ActorId,
    config: &CompactionConfig,
) -> Result<CompactionOutcome, StructuredError> {
    let entries = collect_entries(journal.snapshot());
    let mut dag = SummaryDag::load(journal.snapshot());
    let mut node_tokens = dag.token_index();
    let mut covered = dag.covered();
    let mut reductions: BTreeMap<ElementId, Reduction> = BTreeMap::new();
    let overhead = config.fixed_overhead_tokens;

    // An unreadable node means some elements were elided by a decision this
    // build cannot see; adding coverage on top could double-hide them.
    if dag.unreadable > 0 {
        return Ok(CompactionOutcome {
            folds: dag.folds(),
            covered,
            reductions,
        });
    }

    let mut projected =
        projected_tokens(&entries, &covered, &dag, &node_tokens, &reductions, overhead);

    // --- Stages 1 and 2 accumulate one coverage set for this pass. ----------
    // Both stages decide *what* to elide; only the commit below journals, so a
    // pass that needs both a soft pass and the hard bound still produces one
    // node instead of two.
    let mut pending: Vec<ElementId> = Vec::new();
    let mut simulated = projected;
    if config.budget.should_trigger(projected) {
        let target = config.budget.target_tokens();
        let boundary = recent_turn_boundary(&entries, config.keep_recent_turns);
        for entry in entries.iter().take(boundary) {
            if simulated <= target {
                break;
            }
            if !entry.eligible() || covered.contains(&entry.id) {
                continue;
            }
            // Only cover what pays for its own inventory line and node header.
            if entry.tokens() <= INVENTORY_LINE_TOKENS + NODE_HEADER_TOKENS {
                continue;
            }
            pending.push(entry.id.clone());
            simulated = simulated
                .saturating_sub(entry.tokens() - (INVENTORY_LINE_TOKENS + NODE_HEADER_TOKENS));
        }
    }
    if config.budget.exceeds_hard_bound(simulated) {
        // The whole completed history has to go; only the turn in flight, pinned
        // instructions and what is already covered stay.
        let boundary = final_turn_boundary(&entries);
        for entry in entries.iter().take(boundary) {
            if entry.eligible() && !covered.contains(&entry.id) && !pending.contains(&entry.id) {
                pending.push(entry.id.clone());
            }
        }
    }
    if !pending.is_empty() {
        commit_elision(journal, owner, &entries, &pending, &mut dag)?;
        covered.extend(pending);
        node_tokens = dag.token_index();
        projected =
            projected_tokens(&entries, &covered, &dag, &node_tokens, &reductions, overhead);
    }

    // --- Stage 2b: condense the oldest roots so fold cost stays bounded. -----
    while dag.roots().len() > config.max_root_nodes.max(1) {
        let fold_set: Vec<ElementId> = dag.roots().into_iter().take(CONDENSE_FAN_IN).collect();
        if fold_set.len() < 2 {
            break;
        }
        let covered_tokens = fold_set
            .iter()
            .map(|id| dag.nodes.get(id).map(|node| node.covered_tokens).unwrap_or(0))
            .sum();
        let text = render_condensed_inventory(&fold_set, &dag);
        let node = SummaryNode {
            covered: Vec::new(),
            children: fold_set,
            covered_tokens,
            created_offset: journal.snapshot().offset,
        };
        create_node(journal, owner, node, SummaryKind::Condensed, text)?;
        dag = SummaryDag::load(journal.snapshot());
        node_tokens = dag.token_index();
        projected =
            projected_tokens(&entries, &covered, &dag, &node_tokens, &reductions, overhead);
    }

    // --- Stage 3: reduce what is left until the bound holds. -----------------
    // Tool observations first (arguments, then output), then assistant text,
    // then the user's own words. Pinned instructions are never reduced: an
    // instruction capped to a marker is an instruction broken, and the
    // governance evidence says surviving policies are what long sessions lose
    // first.
    while config.budget.exceeds_hard_bound(projected) {
        let Some((entry, reduction, savings)) = entries
            .iter()
            .filter(|entry| !covered.contains(&entry.id))
            .filter(|entry| entry.role != EntryRole::Instructions)
            .filter_map(|entry| {
                entry
                    .next_reduction(reductions.get(&entry.id))
                    .map(|(reduction, savings)| (entry, reduction, savings))
            })
            .min_by_key(|(entry, _, savings)| {
                (truncation_rank(entry.role), std::cmp::Reverse(*savings))
            })
        else {
            break;
        };
        debug_assert!(savings > 0);
        reductions.insert(entry.id.clone(), reduction);
        projected =
            projected_tokens(&entries, &covered, &dag, &node_tokens, &reductions, overhead);
    }

    // Node text is the last thing to shrink: it stays expandable either way.
    while config.budget.exceeds_hard_bound(projected) {
        let Some(id) = dag
            .roots()
            .into_iter()
            .find(|id| dag.texts.get(id).map(String::len).unwrap_or(0) > MAX_NODE_TEXT_BYTES)
        else {
            break;
        };
        let capped = cap_node_text(&dag.texts[&id]);
        replace_node_text(journal, owner, &id, &capped)?;
        dag.texts.insert(id.clone(), capped.clone());
        node_tokens.insert(id, estimate_tokens(&capped));
        projected =
            projected_tokens(&entries, &covered, &dag, &node_tokens, &reductions, overhead);
    }

    Ok(CompactionOutcome {
        folds: dag.folds(),
        covered,
        reductions,
    })
}

/// Order in which kept elements are reduced once the hard bound is at risk:
/// tool observations first, then assistant text, then the user's own words.
fn truncation_rank(role: EntryRole) -> usize {
    match role {
        EntryRole::ToolCall => 0,
        EntryRole::Assistant => 1,
        EntryRole::User => 2,
        EntryRole::Instructions => 3,
    }
}

/// Tokens the projection of this pass costs, counting every reduction the
/// planner decided — a saving the wire does not take would let the loop exit
/// above the hard bound.
fn projected_tokens(
    entries: &[BodyEntry],
    covered: &BTreeSet<ElementId>,
    dag: &SummaryDag,
    node_tokens: &BTreeMap<ElementId, usize>,
    reductions: &BTreeMap<ElementId, Reduction>,
    overhead: usize,
) -> usize {
    let body: usize = entries
        .iter()
        .filter(|entry| !covered.contains(&entry.id))
        .map(|entry| {
            let Some(reduction) = reductions.get(&entry.id) else {
                return entry.tokens();
            };
            let mut cost = entry.tokens();
            if reduction.omitted_arguments {
                cost = cost.saturating_sub(entry.args_tokens);
            }
            if reduction.capped_text {
                cost = cost.saturating_sub(entry.text_tokens.saturating_sub(reduced_tokens()));
            }
            cost
        })
        .sum();
    overhead + body + dag.projection_tokens(node_tokens)
}

/// Renders the `summary://` marker plus inventory for one node.
fn render_fold(id: &ElementId, dag: &SummaryDag, node: &SummaryNode) -> String {
    let elements = expand_covered(&dag.nodes, id).len().max(node.covered.len());
    format!(
        "[compacted {elements} earlier elements, ~{} tokens. Expand the originals with Read summary://{id}]\n{}",
        node.covered_tokens,
        dag.texts.get(id).map(String::as_str).unwrap_or("")
    )
}

/// Creates the leaf node covering `ids`, if any are new.
fn commit_elision(
    journal: &mut Journal,
    owner: &ActorId,
    entries: &[BodyEntry],
    ids: &[ElementId],
    dag: &mut SummaryDag,
) -> Result<Option<ElementId>, StructuredError> {
    if ids.is_empty() {
        return Ok(None);
    }
    let index: BTreeMap<&ElementId, &BodyEntry> =
        entries.iter().map(|entry| (&entry.id, entry)).collect();
    let covered_tokens: u64 = ids
        .iter()
        .filter_map(|id| index.get(id))
        .map(|entry| entry.tokens() as u64)
        .sum();
    let node = SummaryNode {
        covered: ids.to_vec(),
        children: Vec::new(),
        covered_tokens,
        created_offset: journal.snapshot().offset,
    };
    let text = render_inventory(ids, &index);
    let id = create_node(journal, owner, node.clone(), SummaryKind::Leaf, text.clone())?;
    dag.order.push(id.clone());
    dag.nodes.insert(id.clone(), node);
    dag.texts.insert(id.clone(), text);
    Ok(Some(id))
}

fn create_node(
    journal: &mut Journal,
    owner: &ActorId,
    node: SummaryNode,
    kind: SummaryKind,
    text: String,
) -> Result<ElementId, StructuredError> {
    let container = journal.snapshot().container(SUMMARIES_CONTAINER).clone();
    let index = journal.snapshot().children(&container).count() as u32;
    let id = ElementId::new(format!("sum-{}", ElementId::mint()))?;
    let element = node.to_element(id.clone(), kind, text);
    journal
        .append_patch(Patch {
            base_offset: JournalOffset(journal.snapshot().offset),
            result_offset: journal.next_offset(),
            by: owner.clone().into(),
            reason: format!("compact context ({})", kind.as_str()),
            ops: vec![PatchOp::Create {
                parent: container,
                index,
                element,
            }],
        })
        .map_err(|error| error.structured())?;
    Ok(id)
}

fn replace_node_text(
    journal: &mut Journal,
    owner: &ActorId,
    id: &ElementId,
    text: &str,
) -> Result<(), StructuredError> {
    journal
        .append_patch(Patch {
            base_offset: JournalOffset(journal.snapshot().offset),
            result_offset: journal.next_offset(),
            by: owner.clone().into(),
            reason: "cap compacted inventory".into(),
            ops: vec![PatchOp::ReplaceText {
                element: id.clone(),
                text: text.to_string(),
            }],
        })
        .map_err(|error| error.structured())?;
    Ok(())
}

fn render_inventory(ids: &[ElementId], index: &BTreeMap<&ElementId, &BodyEntry>) -> String {
    let mut out = String::from("Elided transcript elements (originals retrievable by id):\n");
    for id in ids {
        let preview = index
            .get(id)
            .map(|entry| entry.preview.as_str())
            .unwrap_or("(element no longer in the transcript)");
        out.push_str("- ");
        out.push_str(preview);
        out.push('\n');
    }
    out
}

fn render_condensed_inventory(children: &[ElementId], dag: &SummaryDag) -> String {
    let mut out =
        String::from("Condensed summary of earlier compactions (children keep the originals):\n");
    for id in children {
        let node = dag.nodes.get(id).cloned().unwrap_or_default();
        out.push_str(&format!(
            "- summary://{id} — {} elements, ~{} tokens\n",
            expand_covered(&dag.nodes, id).len().max(node.covered.len()),
            node.covered_tokens
        ));
    }
    out
}

fn cap_node_text(text: &str) -> String {
    let head = text_prefix(text, MAX_NODE_TEXT_BYTES / 2);
    format!("{head}\n[…] inventory capped; expand individual nodes with Read summary://<id>")
}

/// Entries the projection would carry, in body order.
///
/// This must mirror `SessionHost::derive_inference_request` exactly: an element
/// the wire never sends must not be charged to the budget, and an element the
/// wire always sends must be.
fn collect_entries(snapshot: &SessionSnapshot) -> Vec<BodyEntry> {
    let mut entries = Vec::new();
    for element in snapshot.get_visible_body() {
        let pinned = is_pinned(element);
        let (role, args_tokens, text_tokens, in_flight, preview) = match element.kind.as_str() {
            "user" => (
                EntryRole::User,
                0,
                estimate_tokens(&element.text),
                false,
                format!("user: {}", preview_line(&element.text)),
            ),
            "assistant" => {
                // Interrupted output is not a completed model turn, so the wire
                // drops it and the budget must not carry it either.
                if matches!(
                    element.attributes.get("status"),
                    Some(TypedValue::String(status))
                        if matches!(status.as_str(), "running" | "failed" | "cancelled")
                ) || element.attributes.get("streaming") == Some(&TypedValue::Bool(true))
                {
                    continue;
                }
                (
                    EntryRole::Assistant,
                    0,
                    estimate_tokens(&element.text),
                    false,
                    format!("assistant: {}", preview_line(&element.text)),
                )
            }
            "tool_call" => {
                // Without a usable call id the whole element is skipped by the
                // projection, so it is invisible to the budget too.
                match element.attributes.get("call_id") {
                    Some(TypedValue::String(call_id)) if ToolCallId::new(call_id).is_ok() => {}
                    _ => continue,
                }
                let result = snapshot
                    .children(&element.id)
                    .find(|child| child.kind == "result");
                let in_flight = result.is_none();
                let args = snapshot
                    .children(&element.id)
                    .find(|child| child.kind == "input")
                    .and_then(|child| child.payload.as_ref())
                    .map(|payload| payload.to_string())
                    .unwrap_or_default();
                let tool = match element.attributes.get("tool") {
                    Some(TypedValue::String(name)) => name.clone(),
                    _ => "tool".to_string(),
                };
                let result_text = result.map(|child| child.text.as_str()).unwrap_or("");
                (
                    EntryRole::ToolCall,
                    estimate_tokens(&args),
                    estimate_tokens(result_text),
                    in_flight,
                    format!(
                        "tool {tool}({}) -> {}",
                        preview_line(&args),
                        preview_line(result_text)
                    ),
                )
            }
            "system" | "steering" => (
                EntryRole::Instructions,
                0,
                estimate_tokens(&element.text),
                true,
                format!("instructions: {}", preview_line(&element.text)),
            ),
            _ => continue,
        };
        entries.push(BodyEntry {
            id: element.id.clone(),
            role,
            args_tokens,
            text_tokens,
            pinned,
            in_flight,
            preview,
        });
    }
    entries
}

fn is_pinned(element: &ElementSnapshot) -> bool {
    element.attributes.get("pinned") == Some(&TypedValue::Bool(true))
}

fn preview_line(text: &str) -> String {
    let line = text
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    let mut collapsed: String = line.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.len() > MAX_PREVIEW_CHARS {
        collapsed = format!("{}…", text_prefix(&collapsed, MAX_PREVIEW_CHARS));
    }
    if collapsed.is_empty() {
        collapsed.push_str("(empty)");
    }
    collapsed
}

/// Index of the first entry belonging to the last `keep` user turns.
///
/// Everything before it belongs to completed turns and may be elided; the turn
/// in flight (`final_turn_boundary`) never is, so at least one turn is kept.
fn recent_turn_boundary(entries: &[BodyEntry], keep: usize) -> usize {
    let keep = keep.max(1);
    let users: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.role == EntryRole::User)
        .map(|(index, _)| index)
        .collect();
    if users.len() <= keep {
        return 0;
    }
    users[users.len() - keep]
}

/// Index of the entry that starts the turn in flight.
fn final_turn_boundary(entries: &[BodyEntry]) -> usize {
    entries
        .iter()
        .rposition(|entry| entry.role == EntryRole::User)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use omp_state::SessionSnapshot;

    struct Fixture {
        dir: std::path::PathBuf,
        journal: Journal,
        owner: ActorId,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = std::env::temp_dir()
                .join(format!("omp-compaction-{}", omp_types::SessionId::mint()));
            std::fs::create_dir_all(&dir).unwrap();
            let journal =
                Journal::create(dir.join("session.journal"), omp_types::SessionId::mint()).unwrap();
            Self {
                dir,
                journal,
                owner: ActorId::new("owner").unwrap(),
            }
        }

        fn push(&mut self, kind: &str, text: &str) -> ElementId {
            self.push_with(kind, text, &[])
        }

        fn push_with(
            &mut self,
            kind: &str,
            text: &str,
            attributes: &[(&str, TypedValue)],
        ) -> ElementId {
            let id = ElementId::mint();
            let mut element = ElementSnapshot::new(id.clone(), kind);
            for (name, value) in attributes {
                element.attributes.insert((*name).into(), value.clone());
            }
            element.text = text.to_string();
            self.attach(self.journal.snapshot().container("body").clone(), element);
            id
        }

        fn push_tool_call(&mut self, args: serde_json::Value, result: &str) -> ElementId {
            let id = self.push_with(
                "tool_call",
                "",
                &[
                    ("tool", TypedValue::String("Bash".into())),
                    (
                        "call_id",
                        TypedValue::String(omp_types::ToolCallId::mint().to_string()),
                    ),
                ],
            );
            let mut input = ElementSnapshot::new(ElementId::mint(), "input");
            input.payload = Some(args);
            self.attach(id.clone(), input);
            let mut output = ElementSnapshot::new(ElementId::mint(), "result");
            output.text = result.to_string();
            self.attach(id.clone(), output);
            id
        }

        fn push_corrupt_summary_node(&mut self) {
            let mut element = ElementSnapshot::new(ElementId::mint(), "summary_node");
            element.payload = Some(serde_json::json!({"covered": [], "surprise": true}));
            self.attach(self.journal.snapshot().container("summaries").clone(), element);
        }

        fn attach(&mut self, parent: ElementId, element: ElementSnapshot) {
            let index = self.journal.snapshot().children(&parent).count() as u32;
            self.journal
                .append_patch(Patch {
                    base_offset: JournalOffset(self.journal.snapshot().offset),
                    result_offset: self.journal.next_offset(),
                    by: self.owner.clone().into(),
                    reason: "test element".into(),
                    ops: vec![PatchOp::Create {
                        parent,
                        index,
                        element,
                    }],
                })
                .unwrap();
        }

        fn nodes(&self) -> Vec<(ElementId, SummaryNode)> {
            let snapshot = self.journal.snapshot();
            snapshot
                .children(snapshot.container(SUMMARIES_CONTAINER))
                .filter_map(|element| {
                    SummaryNode::from_element(element)
                        .ok()
                        .map(|node| (element.id.clone(), node))
                })
                .collect()
        }

        fn node_map(&self) -> BTreeMap<ElementId, SummaryNode> {
            self.nodes().into_iter().collect()
        }

        fn plan_with(&mut self, config: &CompactionConfig) -> CompactionOutcome {
            plan(&mut self.journal, &self.owner, config).unwrap()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn config(soft: usize, hard: usize) -> CompactionConfig {
        CompactionConfig {
            budget: CompactionBudget {
                soft_tokens: soft,
                hard_tokens: hard,
            },
            keep_recent_turns: DEFAULT_KEEP_RECENT_TURNS,
            max_root_nodes: DEFAULT_MAX_ROOT_NODES,
            fixed_overhead_tokens: 0,
        }
    }

    fn big(tag: &str, chars: usize) -> String {
        format!("{tag} {}", "x".repeat(chars))
    }

    /// Projects the whole body the way the host does, so a test can check the
    /// bound the planner claims rather than the planner's own arithmetic.
    fn projected_after(entries: &[BodyEntry], outcome: &CompactionOutcome) -> usize {
        let visible: usize = entries
            .iter()
            .filter(|entry| !outcome.covered.contains(&entry.id))
            .map(|entry| {
                let mut cost = if outcome.omits_arguments(&entry.id) {
                    entry.text_tokens
                } else {
                    entry.tokens()
                };
                if outcome.project_text(&entry.id, &"x".repeat(entry.text_tokens * 4)) != "x".repeat(entry.text_tokens * 4) {
                    cost -= entry.text_tokens.saturating_sub(reduced_tokens());
                }
                cost
            })
            .sum();
        visible + outcome.folds.len() * NODE_HEADER_TOKENS
    }

    #[test]
    fn compaction_elides_old_turns_and_spares_pinned_instructions() {
        let mut fixture = Fixture::new();
        let instructions = fixture.push_with(
            "system",
            &big("policy", 3_000),
            &[("pinned", TypedValue::Bool(true))],
        );
        let mut turns = Vec::new();
        for index in 0..4 {
            let user = fixture.push("user", &big(&format!("question {index}"), 4_000));
            let assistant = fixture.push("assistant", &big(&format!("answer {index}"), 4_000));
            turns.push((user, assistant));
        }

        let outcome = fixture.plan_with(&config(1_000, 1_200));

        assert!(
            !outcome.covered.contains(&instructions),
            "pinned instructions are never elided"
        );
        assert!(!outcome.folds.is_empty(), "compaction produced a fold");
        assert!(outcome.covered.contains(&turns[0].0));
        assert!(outcome.covered.contains(&turns[1].1));
        assert!(
            !outcome.covered.contains(&turns[3].0) && !outcome.covered.contains(&turns[3].1),
            "the turn in flight stays verbatim"
        );
        for (user, assistant) in &turns {
            assert!(fixture.journal.snapshot().element(user).is_some());
            assert!(fixture.journal.snapshot().element(assistant).is_some());
        }

        let nodes = fixture.nodes();
        assert_eq!(nodes.len(), 1, "one leaf node covers the elided prefix");
        let (id, node) = &nodes[0];
        assert!(node.children.is_empty());
        assert_eq!(
            node.covered.len(),
            outcome.covered.len(),
            "the node accounts for every elided element"
        );
        assert!(node.covered.contains(&turns[0].0) && node.covered.contains(&turns[2].1));
        let text = fixture.journal.snapshot().element(id).unwrap().text.clone();
        assert!(text.contains("question 0"), "inventory names what was elided");
        match &outcome.folds[0] {
            MessageFold::Handoff { summary, .. } => {
                assert!(summary.contains(&format!("summary://{id}")));
            }
            other => panic!("expected a handoff fold, got {other:?}"),
        }
    }

    #[test]
    fn planning_twice_does_not_duplicate_nodes() {
        let mut fixture = Fixture::new();
        for index in 0..4 {
            fixture.push("user", &big(&format!("question {index}"), 4_000));
            fixture.push("assistant", &big(&format!("answer {index}"), 4_000));
        }
        let config = config(1_000, 1_200);
        let first = fixture.plan_with(&config);
        let offset_after_first = fixture.journal.snapshot().offset;
        let second = fixture.plan_with(&config);

        assert_eq!(fixture.nodes().len(), 1, "the second pass reuses the first node");
        assert_eq!(first.covered, second.covered);
        assert_eq!(first.folds.len(), second.folds.len());
        assert_eq!(
            fixture.journal.snapshot().offset,
            offset_after_first,
            "a pass that changes nothing journals nothing"
        );
    }

    #[test]
    fn later_turns_extend_the_dag_and_condensation_keeps_reachability() {
        let mut fixture = Fixture::new();
        let config = CompactionConfig {
            max_root_nodes: 2,
            ..config(1_000, 1_200)
        };
        let mut oldest = None;
        for index in 0..8 {
            let user = fixture.push("user", &big(&format!("question {index}"), 4_000));
            fixture.push("assistant", &big(&format!("answer {index}"), 4_000));
            if index == 0 {
                oldest = Some(user);
            }
            fixture.plan_with(&config);
        }

        let nodes = fixture.node_map();
        let dag = SummaryDag::load(fixture.journal.snapshot());
        assert!(
            dag.roots().len() <= 2,
            "condensation bounds the projected roots, got {}",
            dag.roots().len()
        );
        let oldest = oldest.unwrap();
        assert!(
            dag.roots()
                .iter()
                .any(|root| expand_covered(&nodes, root).contains(&oldest)),
            "the oldest elided element stays reachable from a root"
        );
    }

    #[test]
    fn reductions_start_with_tool_observations_and_never_touch_instructions() {
        let mut fixture = Fixture::new();
        let policy = big("policy", 200);
        let instructions = fixture.push_with(
            "system",
            &policy,
            &[("pinned", TypedValue::Bool(true))],
        );
        fixture.push("user", &big("earlier", 4_000));
        fixture.push("assistant", &big("earlier answer", 4_000));
        let question = big("do the thing", 4_000);
        let user = fixture.push("user", &question);
        // A tool result large enough that capping it alone brings the
        // projection back under the bound, leaving the user's turn intact.
        let tool = fixture.push_tool_call(
            serde_json::json!({"command": "ls"}),
            &"observation ".repeat(4_000),
        );

        let budget = config(1_600, 1_900);
        let outcome = fixture.plan_with(&budget);

        let tool_text = "observation ".repeat(4_000);
        assert!(
            outcome.project_text(&tool, &tool_text).len() < tool_text.len(),
            "the observation is capped"
        );
        assert_eq!(
            outcome.project_text(&user, &question).len(),
            question.len(),
            "the user's own words survive"
        );
        assert!(!outcome.covered.contains(&instructions));
        assert_eq!(
            outcome.project_text(&instructions, &policy).len(),
            policy.len(),
            "pinned instructions are never reduced"
        );

        let entries = collect_entries(fixture.journal.snapshot());
        assert!(
            projected_after(&entries, &outcome) <= budget.budget.hard_tokens,
            "the projection the host builds lands under the hard bound"
        );
    }

    #[test]
    fn pinned_instructions_are_never_reduced_even_when_the_bound_cannot_be_met() {
        let mut fixture = Fixture::new();
        let policy = big("policy", 8_000);
        let instructions = fixture.push_with("system", &policy, &[]);
        let tool = fixture.push_tool_call(
            serde_json::json!({"command": "ls"}),
            &"observation ".repeat(4_000),
        );

        let outcome = fixture.plan_with(&config(600, 700));

        assert_eq!(
            outcome.project_text(&instructions, &policy).len(),
            policy.len(),
            "an instruction capped to a marker is a broken instruction"
        );
        assert!(
            outcome.project_text(&tool, &"observation ".repeat(4_000)).len() < 48_000,
            "the observation still gives way first"
        );
    }

    #[test]
    fn oversized_tool_arguments_are_omitted_before_the_output() {
        let mut fixture = Fixture::new();
        fixture.push("user", &big("earlier", 4_000));
        fixture.push("assistant", &big("earlier answer", 4_000));
        let user = fixture.push("user", "carry on");
        let arguments = serde_json::json!({"content": "y".repeat(40_000)});
        let tool = fixture.push_tool_call(arguments, "ok");

        let outcome = fixture.plan_with(&config(600, 700));

        assert!(outcome.omits_arguments(&tool), "the payload is replaced by a marker");
        let tiny = "ok";
        assert_eq!(outcome.project_text(&tool, tiny), tiny, "the tiny output is left alone");
        assert!(!outcome.omits_arguments(&user));

        let entries = collect_entries(fixture.journal.snapshot());
        assert!(projected_after(&entries, &outcome) <= config(600, 700).budget.hard_tokens);
    }

    #[test]
    fn a_projection_that_is_already_small_is_left_alone() {
        let mut fixture = Fixture::new();
        fixture.push("user", "hello");
        fixture.push("assistant", "hi");
        let before = fixture.journal.snapshot().offset;
        let outcome = fixture.plan_with(&config(10_000, 12_000));
        assert!(outcome.covered.is_empty());
        assert!(outcome.folds.is_empty());
        assert_eq!(fixture.journal.snapshot().offset, before);
    }

    #[test]
    fn hard_bound_is_met_when_the_session_leaves_room_for_it() {
        let mut fixture = Fixture::new();
        for index in 0..6 {
            fixture.push("user", &big(&format!("question {index}"), 4_000));
            fixture.push("assistant", &big(&format!("answer {index}"), 4_000));
        }
        let config = config(2_000, 2_400);
        let outcome = fixture.plan_with(&config);
        let entries = collect_entries(fixture.journal.snapshot());
        assert!(
            projected_after(&entries, &outcome) <= config.budget.hard_tokens,
            "the projection lands under the hard bound"
        );
    }

    #[test]
    fn unreadable_nodes_stop_further_elision_instead_of_double_hiding() {
        let mut fixture = Fixture::new();
        for index in 0..4 {
            fixture.push("user", &big(&format!("question {index}"), 4_000));
            fixture.push("assistant", &big(&format!("answer {index}"), 4_000));
        }
        fixture.push_corrupt_summary_node();
        let before = fixture.journal.snapshot().offset;
        let outcome = fixture.plan_with(&config(1_000, 1_200));
        assert!(outcome.covered.is_empty());
        assert!(outcome.folds.is_empty());
        assert_eq!(
            fixture.journal.snapshot().offset,
            before,
            "no node is created while the DAG is unreadable"
        );
    }

    #[test]
    fn zero_kept_turns_is_a_configuration_not_a_panic() {
        let mut fixture = Fixture::new();
        for index in 0..3 {
            fixture.push("user", &big(&format!("question {index}"), 4_000));
            fixture.push("assistant", &big(&format!("answer {index}"), 4_000));
        }
        let outcome = fixture.plan_with(&CompactionConfig {
            keep_recent_turns: 0,
            ..config(1_000, 1_200)
        });
        assert!(
            outcome.covered.len() < 6,
            "at least the turn in flight is kept"
        );
    }

    #[test]
    fn expansion_returns_every_elided_element_in_body_order() {
        let mut fixture = Fixture::new();
        let mut expected = Vec::new();
        for index in 0..4 {
            expected.push(fixture.push("user", &big(&format!("question {index}"), 4_000)));
            expected.push(fixture.push("assistant", &big(&format!("answer {index}"), 4_000)));
        }
        fixture.plan_with(&config(1_000, 1_200));
        let nodes = fixture.node_map();
        let (id, _) = fixture.nodes()[0].clone();
        let expanded = expand_covered(&nodes, &id);
        assert_eq!(expanded, expected[..expanded.len()].to_vec());
        assert!(expanded.len() >= 4, "at least two turns were elided");
    }

    #[test]
    fn snapshot_keeps_every_covered_element_addressable() {
        let mut fixture = Fixture::new();
        let user = fixture.push("user", &big("original question", 4_000));
        fixture.push("assistant", &big("answer", 4_000));
        fixture.push("user", &big("second question", 4_000));
        fixture.push("assistant", &big("second answer", 4_000));
        fixture.plan_with(&config(1_000, 1_200));
        let snapshot: &SessionSnapshot = fixture.journal.snapshot();
        let element = snapshot.element(&user).expect("covered element stays in the DOM");
        assert!(element.text.contains("original question"));
    }

    #[test]
    fn truncation_respects_utf8_boundaries() {
        let text = "λ".repeat(1_000);
        let projected = truncate_for_projection(&text);
        assert!(projected.len() < text.len());
        assert!(projected.starts_with("λ"));
        assert!(projected.ends_with("λ"));
        assert!(std::str::from_utf8(projected.as_bytes()).is_ok());
        assert_eq!(truncate_for_projection("short"), "short");
    }
}

#[cfg(test)]
mod reduction_sequencing_tests {
    use super::*;

    fn entry(role: EntryRole, args_tokens: usize, text_tokens: usize) -> BodyEntry {
        BodyEntry {
            id: ElementId::mint(),
            role,
            args_tokens,
            text_tokens,
            pinned: false,
            in_flight: true,
            preview: String::new(),
        }
    }

    #[test]
    fn a_tool_call_can_be_reduced_twice() {
        let call = entry(EntryRole::ToolCall, 5_000, 6_000);
        let (first, saved_first) = call.next_reduction(None).unwrap();
        assert!(first.capped_text && !first.omitted_arguments);
        assert_eq!(saved_first, 6_000 - reduced_tokens());

        // Once the output is capped, the arguments dominate and are next.
        let (second, saved_second) = call.next_reduction(Some(&first)).unwrap();
        assert!(second.omitted_arguments && second.capped_text);
        assert_eq!(saved_second, 5_000);
        assert!(call.next_reduction(Some(&second)).is_none());
    }

    #[test]
    fn a_small_tool_call_is_left_alone_and_instructions_are_excluded() {
        let small = entry(EntryRole::ToolCall, 10, 100);
        assert!(small.next_reduction(None).is_none());
        let instruction = entry(EntryRole::Instructions, 0, 40_000);
        assert!(instruction.next_reduction(None).is_some());
        let user = entry(EntryRole::User, 0, 40_000);
        assert_eq!(truncation_rank(user.role), 2);
    }
}
