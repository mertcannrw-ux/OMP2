use omp_types::{
    BranchId, ElementId, ElementSnapshot, Patch, PatchOp, SessionId, Status, TypedValue,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub element: ElementSnapshot,
    pub parent: Option<ElementId>,
    pub children: Vec<ElementId>,
}

/// Every durable feature is a node; indexes and live handles are not persisted here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionSnapshot {
    pub session_id: SessionId,
    pub offset: u64,
    pub selected_branch: BranchId,
    nodes: BTreeMap<ElementId, Node>,
    used_ids: BTreeSet<ElementId>,
}

#[derive(Debug, thiserror::Error)]
pub enum StateError {
    #[error("patch base {actual} does not match {expected}")]
    WrongBase { actual: u64, expected: u64 },
    #[error("element not found: {0}")]
    Missing(String),
    #[error("invalid state: {0}")]
    Invalid(String),
    #[error("journal I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("journal encoding: {0}")]
    Encoding(#[from] serde_json::Error),
    #[error("journal is damaged or failed; reopen required")]
    WriteFailure,
}
impl StateError {
    pub fn structured(&self) -> omp_types::StructuredError {
        let code = match self {
            Self::WrongBase { .. } => "stale_base",
            Self::Missing(_) => "missing_element",
            Self::Invalid(_) => "invalid_state",
            Self::Io(_) => "journal_io",
            Self::Encoding(_) => "journal_encoding",
            Self::WriteFailure => "write_failure",
        };
        omp_types::StructuredError::new(
            code,
            self.to_string(),
            matches!(self, Self::WrongBase { .. }),
        )
    }
}

fn id(value: &str) -> ElementId {
    ElementId::new(value).expect("static DOM id")
}
const ROOTS: &[&str] = &[
    "root",
    "meta",
    "body",
    "queues",
    "views",
    "convars",
    "todo",
    "jobs",
    "actors",
    "directors",
    "branches",
    "artifacts",
    "capabilities",
    "tools",
    "steering",
    "prompts",
    "approvals",
    "scheduler",
    "summaries",
];

impl SessionSnapshot {
    pub fn empty(session_id: SessionId) -> Self {
        let mut value = Self {
            session_id,
            offset: 0,
            selected_branch: BranchId::new("main").unwrap(),
            nodes: BTreeMap::new(),
            used_ids: BTreeSet::new(),
        };
        for name in ROOTS {
            let parent = match *name {
                "root" => None,
                "meta" | "body" | "queues" | "views" => Some(id("root")),
                "steering" | "prompts" | "approvals" | "scheduler" => Some(id("queues")),
                _ => Some(id("meta")),
            };
            let key = id(name);
            value.nodes.insert(
                key.clone(),
                Node {
                    element: ElementSnapshot::new(key.clone(), *name),
                    parent: parent.clone(),
                    children: vec![],
                },
            );
            value.used_ids.insert(key.clone());
            if let Some(parent) = parent
                && let Some(node) = value.nodes.get_mut(&parent) {
                    // Parents are inserted before children in ROOTS order, so
                    // this only skips on a mis-ordered ROOTS table edit.
                    node.children.push(key);
                }
        }
        value
    }
    pub fn node(&self, element: &ElementId) -> Option<&Node> {
        self.nodes.get(element)
    }
    pub fn element(&self, element: &ElementId) -> Option<&ElementSnapshot> {
        self.node(element).map(|n| &n.element)
    }
    pub fn children(&self, parent: &ElementId) -> impl Iterator<Item = &ElementSnapshot> {
        self.nodes
            .get(parent)
            .into_iter()
            .flat_map(|n| n.children.iter())
            .filter_map(|key| self.element(key))
    }
    /// Look up a structural container id. Returns `Invalid` instead of
    /// panicking so corrupt snapshots surface as patch rejections.
    pub fn try_container(&self, name: &str) -> Result<&ElementId, StateError> {
        let key = ElementId::new(name)
            .map_err(|_| StateError::Invalid(format!("invalid container name '{name}'")))?;
        self.nodes
            .get_key_value(&key)
            .map(|(key, _)| key)
            .ok_or_else(|| StateError::Invalid(format!("missing container '{name}'")))
    }
    pub fn container(&self, name: &str) -> &ElementId {
        // `SessionSnapshot::empty` always seeds every ROOTS entry, so this
        // only fires on programmer error constructing a snapshot by hand.
        self.try_container(name)
            .expect("required container missing: snapshot was not built via SessionSnapshot::empty")
    }
    pub fn current_branch(&self) -> &BranchId {
        &self.selected_branch
    }
    pub fn get_visible_body(&self) -> impl Iterator<Item = &ElementSnapshot> {
        self.children(self.container("body"))
    }
    pub fn get_last_visible_message(&self) -> Option<&ElementSnapshot> {
        self.get_visible_body()
            .filter(|e| matches!(e.kind.as_str(), "user" | "assistant" | "system" | "message"))
            .last()
    }
    pub fn active_tool_roster(&self) -> impl Iterator<Item = &ElementSnapshot> {
        self.children(self.container("tools"))
    }
    pub fn active_directors(&self) -> impl Iterator<Item = &ElementSnapshot> {
        self.children(self.container("directors"))
    }
    pub fn active_jobs(&self) -> impl Iterator<Item = &ElementSnapshot> {
        self.children(self.container("jobs"))
            .filter(|e| !is_terminal(e))
    }
    pub fn active_todos(&self) -> impl Iterator<Item = &ElementSnapshot> {
        self.children(self.container("todo"))
            .filter(|e| !is_terminal(e))
    }
    pub fn try_session_globals(&self) -> Result<&BTreeMap<String, TypedValue>, StateError> {
        self.element(self.try_container("convars")?)
            .map(|node| &node.attributes)
            .ok_or_else(|| StateError::Invalid("missing convars container".into()))
    }
    pub fn session_globals(&self) -> &BTreeMap<String, TypedValue> {
        self.try_session_globals().expect(
            "convars container missing: snapshot was not built via SessionSnapshot::empty",
        )
    }
    pub fn turn_count(&self) -> usize {
        self.get_visible_body().filter(|e| e.kind == "user").count()
    }
    pub fn all_elements(&self) -> impl Iterator<Item = &ElementSnapshot> {
        self.nodes.values().map(|n| &n.element)
    }
    pub fn reconcile_handles(&self, live: &BTreeSet<ElementId>) -> Vec<HandleAction> {
        let wanted: BTreeSet<_> = self
            .active_jobs()
            .chain(
                self.children(self.container("actors"))
                    .filter(|e| !is_terminal(e)),
            )
            .map(|e| e.id.clone())
            .collect();
        wanted
            .difference(live)
            .cloned()
            .map(HandleAction::ResumeOrSpawn)
            .chain(
                live.difference(&wanted)
                    .cloned()
                    .map(HandleAction::Terminate),
            )
            .collect()
    }
    pub fn inspect_xml(&self) -> String {
        fn escape(text: &str) -> String {
            text.replace('&', "&amp;")
                .replace('<', "&lt;")
                .replace('>', "&gt;")
                .replace('"', "&quot;")
        }
        fn visit(snapshot: &SessionSnapshot, key: &ElementId, depth: usize, out: &mut String) {
            let Some(node) = snapshot.nodes.get(key) else {
                // Dangling child reference: record it as a comment instead of
                // panicking the host on corrupt state.
                out.push_str(&format!("{}<!-- dangling node {key} -->\n", "  ".repeat(depth)));
                return;
            };
            let indent = "  ".repeat(depth);
            out.push_str(&format!(
                "{indent}<{} id=\"{}\" schema=\"{}\">\n",
                node.element.kind, key, node.element.schema_version
            ));
            for (name, value) in &node.element.attributes {
                let rendered =
                    serde_json::to_string(value).unwrap_or_else(|_| "\"<unserializable>\"".into());
                out.push_str(&format!(
                    "{indent}  <attribute name=\"{}\">{}</attribute>\n",
                    escape(name),
                    escape(&rendered)
                ));
            }
            if !node.element.text.is_empty() {
                out.push_str(&format!(
                    "{indent}  <text>{}</text>\n",
                    escape(&node.element.text)
                ));
            }
            if let Some(payload) = &node.element.payload {
                out.push_str(&format!(
                    "{indent}  <payload>{}</payload>\n",
                    escape(&payload.to_string())
                ));
            }
            for child in &node.children {
                visit(snapshot, child, depth + 1, out);
            }
            out.push_str(&format!("{indent}</{}>\n", node.element.kind));
        }
        let mut out = format!(
            "<!-- session={} offset={} branch={} -->\n",
            self.session_id, self.offset, self.selected_branch
        );
        visit(self, &id("root"), 0, &mut out);
        out
    }
}
fn is_terminal(e: &ElementSnapshot) -> bool {
    matches!(e.attributes.get("status"),Some(TypedValue::String(s)) if Status::from_dom_str(s).is_some_and(|status| status.is_terminal_session()))
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HandleAction {
    ResumeOrSpawn(ElementId),
    Terminate(ElementId),
}

/// Undo retains only touched values/subtrees, not a second full-document copy.
enum Undo {
    Create(ElementId),
    Delete(Vec<(ElementId, Node)>, ElementId, usize),
    Move(ElementId, ElementId, usize),
    Attribute(ElementId, String, Option<TypedValue>),
    Text(ElementId, String),
    Append(ElementId, usize),
    Payload(ElementId, Option<serde_json::Value>),
}
impl SessionSnapshot {
    fn require(&self, key: &ElementId) -> Result<&Node, StateError> {
        self.node(key)
            .ok_or_else(|| StateError::Missing(key.to_string()))
    }
    fn mutable_location(&self, key: &ElementId) -> Result<(ElementId, usize), StateError> {
        if ROOTS.contains(&key.as_str()) {
            return Err(StateError::Invalid(
                "cannot remove or move structural container".into(),
            ));
        }
        let parent = self
            .require(key)?
            .parent
            .clone()
            .ok_or_else(|| StateError::Invalid("orphan node".into()))?;
        // A child missing from its parent's list means a desynced snapshot:
        // reject the patch instead of panicking the host.
        let index = self
            .nodes
            .get(&parent)
            .ok_or_else(|| StateError::Invalid("parent/child desync: missing parent".into()))?
            .children
            .iter()
            .position(|v| v == key)
            .ok_or_else(|| StateError::Invalid("parent/child desync".into()))?;
        Ok((parent, index))
    }
    fn apply_one(&mut self, op: &PatchOp) -> Result<Undo, StateError> {
        Ok(match op {
            PatchOp::Create {
                parent,
                index,
                element,
            } => {
                if self.used_ids.contains(&element.id) {
                    return Err(StateError::Invalid(
                        "element identifier already used".into(),
                    ));
                }
                if *index as usize > self.require(parent)?.children.len() {
                    return Err(StateError::Invalid("child index outside parent".into()));
                }
                self.nodes
                    .get_mut(parent)
                    .ok_or_else(|| StateError::Invalid("create parent missing".into()))?
                    .children
                    .insert(*index as usize, element.id.clone());
                self.nodes.insert(
                    element.id.clone(),
                    Node {
                        element: element.clone(),
                        parent: Some(parent.clone()),
                        children: vec![],
                    },
                );
                self.used_ids.insert(element.id.clone());
                Undo::Create(element.id.clone())
            }
            PatchOp::Delete { element } => {
                let (parent, index) = self.mutable_location(element)?;
                self.nodes
                    .get_mut(&parent)
                    .ok_or_else(|| StateError::Invalid("delete parent missing".into()))?
                    .children
                    .remove(index);
                let mut stack = vec![element.clone()];
                let mut removed = vec![];
                while let Some(key) = stack.pop() {
                    // A dangling child reference means a desynced snapshot:
                    // reject the patch instead of panicking the host.
                    let node = self
                        .nodes
                        .remove(&key)
                        .ok_or_else(|| StateError::Invalid("dangling child reference".into()))?;
                    stack.extend(node.children.iter().cloned());
                    removed.push((key, node));
                }
                Undo::Delete(removed, parent, index)
            }
            PatchOp::Move {
                element,
                parent,
                index,
            } => {
                let (old_parent, old_index) = self.mutable_location(element)?;
                let target_len =
                    self.require(parent)?.children.len() - usize::from(&old_parent == parent);
                if *index as usize > target_len {
                    return Err(StateError::Invalid(
                        "move index outside post-removal parent".into(),
                    ));
                }
                let mut cursor = Some(parent);
                while let Some(key) = cursor {
                    if key == element {
                        return Err(StateError::Invalid("move creates cycle".into()));
                    }
                    cursor = self
                        .nodes
                        .get(key)
                        .ok_or_else(|| StateError::Invalid("move target missing".into()))?
                        .parent
                        .as_ref();
                }
                self.nodes
                    .get_mut(&old_parent)
                    .ok_or_else(|| StateError::Invalid("move source missing".into()))?
                    .children
                    .remove(old_index);
                self.nodes
                    .get_mut(parent)
                    .ok_or_else(|| StateError::Invalid("move target missing".into()))?
                    .children
                    .insert(*index as usize, element.clone());
                self.nodes
                    .get_mut(element)
                    .ok_or_else(|| StateError::Invalid("move element missing".into()))?
                    .parent = Some(parent.clone());
                Undo::Move(element.clone(), old_parent, old_index)
            }
            PatchOp::SetAttribute {
                element,
                name,
                value,
            } => {
                self.require(element)?;
                let old = self
                    .nodes
                    .get_mut(element)
                    .ok_or_else(|| StateError::Invalid("attribute target missing".into()))?
                    .element
                    .attributes
                    .insert(name.clone(), value.clone());
                Undo::Attribute(element.clone(), name.clone(), old)
            }
            PatchOp::RemoveAttribute { element, name } => {
                self.require(element)?;
                let old = self
                    .nodes
                    .get_mut(element)
                    .ok_or_else(|| StateError::Invalid("attribute target missing".into()))?
                    .element
                    .attributes
                    .remove(name);
                Undo::Attribute(element.clone(), name.clone(), old)
            }
            PatchOp::ReplaceText { element, text } => {
                self.require(element)?;
                let old = std::mem::replace(
                    &mut self
                        .nodes
                        .get_mut(element)
                        .ok_or_else(|| StateError::Invalid("text target missing".into()))?
                        .element
                        .text,
                    text.clone(),
                );
                Undo::Text(element.clone(), old)
            }
            PatchOp::AppendText { element, text } => {
                self.require(element)?;
                let dst = &mut self
                    .nodes
                    .get_mut(element)
                    .ok_or_else(|| StateError::Invalid("text target missing".into()))?
                    .element
                    .text;
                let len = dst.len();
                dst.push_str(text);
                Undo::Append(element.clone(), len)
            }
            PatchOp::ReplacePayload { element, payload } => {
                self.require(element)?;
                let old = self
                    .nodes
                    .get_mut(element)
                    .ok_or_else(|| StateError::Invalid("payload target missing".into()))?
                    .element
                    .payload
                    .replace(payload.clone());
                Undo::Payload(element.clone(), old)
            }
        })
    }
    /// Roll back one applied op. The undo stack is built only by successful
    /// `apply_one` calls on this same snapshot, so every referenced node must
    /// exist; missing nodes are skipped defensively instead of panicking.
    fn undo(&mut self, undo: Undo) {
        match undo {
            Undo::Create(key) => {
                if let Some(node) = self.nodes.remove(&key) {
                    if let Some(parent) = node.parent
                        && let Some(parent_node) = self.nodes.get_mut(&parent) {
                            parent_node.children.retain(|v| v != &key);
                        }
                    self.used_ids.remove(&key);
                }
            }
            Undo::Delete(nodes, parent, index) => {
                if nodes.is_empty() {
                    return;
                }
                let key = nodes[0].0.clone();
                self.nodes.extend(nodes);
                if let Some(parent_node) = self.nodes.get_mut(&parent) {
                    let at = index.min(parent_node.children.len());
                    parent_node.children.insert(at, key);
                }
            }
            Undo::Move(key, parent, index) => {
                let current = self.nodes.get(&key).and_then(|n| n.parent.clone());
                if let Some(current) = current
                    && let Some(current_node) = self.nodes.get_mut(&current) {
                        current_node.children.retain(|v| v != &key);
                    }
                if let Some(parent_node) = self.nodes.get_mut(&parent) {
                    let at = index.min(parent_node.children.len());
                    parent_node.children.insert(at, key.clone());
                }
                if let Some(node) = self.nodes.get_mut(&key) {
                    node.parent = Some(parent);
                }
            }
            Undo::Attribute(key, name, value) => {
                if let Some(node) = self.nodes.get_mut(&key) {
                    let attrs = &mut node.element.attributes;
                    match value {
                        Some(v) => {
                            attrs.insert(name, v);
                        }
                        None => {
                            attrs.remove(&name);
                        }
                    }
                }
            }
            Undo::Text(key, value) => {
                if let Some(node) = self.nodes.get_mut(&key) {
                    node.element.text = value;
                }
            }
            Undo::Append(key, len) => {
                if let Some(node) = self.nodes.get_mut(&key) {
                    // `len` was captured pre-append, so it is always in range;
                    // clamp defensively against logic errors elsewhere.
                    let at = len.min(node.element.text.len());
                    node.element.text.truncate(at);
                }
            }
            Undo::Payload(key, value) => {
                if let Some(node) = self.nodes.get_mut(&key) {
                    node.element.payload = value;
                }
            }
        }
    }
}
pub fn apply_patch(snapshot: &mut SessionSnapshot, patch: &Patch) -> Result<(), StateError> {
    patch
        .validate()
        .map_err(|e| StateError::Invalid(e.to_string()))?;
    if patch.base_offset.0 != snapshot.offset {
        return Err(StateError::WrongBase {
            actual: patch.base_offset.0,
            expected: snapshot.offset,
        });
    }
    let mut undo = Vec::with_capacity(patch.ops.len());
    for op in &patch.ops {
        match snapshot.apply_one(op) {
            Ok(value) => undo.push(value),
            Err(error) => {
                for value in undo.into_iter().rev() {
                    snapshot.undo(value);
                }
                return Err(error);
            }
        }
    }
    snapshot.offset = patch.result_offset.0;
    Ok(())
}
