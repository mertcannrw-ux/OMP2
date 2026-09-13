//! Durable schema for lossless context compaction.
//!
//! Compaction never destroys session state: the journal keeps every element, and
//! a [`SummaryNode`] only records which body elements are left out of the
//! projection sent to the provider. Because the covered elements stay
//! addressable by id in the DOM, expansion is a read rather than a
//! reconstruction, so the generated inventory text is allowed to be lossy while
//! the history itself stays exact.

use crate::ids::ElementId;
use crate::patch::ElementSnapshot;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Container element that holds the summary DAG.
pub const SUMMARIES_CONTAINER: &str = "summaries";
/// Element kind of one DAG node.
pub const SUMMARY_NODE_KIND: &str = "summary_node";
/// Attribute carrying the [`SummaryKind`] of a node.
pub const SUMMARY_KIND_ATTRIBUTE: &str = "node_kind";

/// Whether a node elides body elements directly or condenses child nodes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryKind {
    Leaf,
    Condensed,
}

impl SummaryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Leaf => "leaf",
            Self::Condensed => "condensed",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "leaf" => Some(Self::Leaf),
            "condensed" => Some(Self::Condensed),
            _ => None,
        }
    }
}

/// One node of the summary DAG.
///
/// `covered` names the body elements this node removes from the provider
/// projection, oldest first. `children` names the nodes a condensed node
/// replaces; their covered elements are reachable through the parent, so
/// condensing loses text but never reachability.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SummaryNode {
    pub covered: Vec<ElementId>,
    pub children: Vec<ElementId>,
    pub covered_tokens: u64,
    pub created_offset: u64,
}

impl SummaryNode {
    /// Total number of body elements reachable through this node.
    pub fn covered_count(&self) -> usize {
        self.covered.len()
    }

    pub fn encode(&self) -> serde_json::Value {
        serde_json::json!({
            "covered": self.covered.iter().map(|id| id.as_str()).collect::<Vec<_>>(),
            "children": self.children.iter().map(|id| id.as_str()).collect::<Vec<_>>(),
            "covered_tokens": self.covered_tokens,
            "created_offset": self.created_offset,
        })
    }

    pub fn decode(payload: &serde_json::Value) -> Result<Self, String> {
        let object = payload
            .as_object()
            .ok_or_else(|| "summary node payload must be an object".to_string())?;
        for key in object.keys() {
            if !matches!(
                key.as_str(),
                "covered" | "children" | "covered_tokens" | "created_offset"
            ) {
                return Err(format!("unknown summary node field: {key}"));
            }
        }
        let mut covered = Vec::new();
        for value in object
            .get("covered")
            .and_then(|value| value.as_array())
            .ok_or_else(|| "summary node payload requires a covered array".to_string())?
        {
            let raw = value
                .as_str()
                .ok_or_else(|| "covered entries must be strings".to_string())?;
            covered.push(ElementId::new(raw).map_err(|error| error.message)?);
        }
        let mut children = Vec::new();
        if let Some(entries) = object.get("children").and_then(|value| value.as_array()) {
            for value in entries {
                let raw = value
                    .as_str()
                    .ok_or_else(|| "children entries must be strings".to_string())?;
                children.push(ElementId::new(raw).map_err(|error| error.message)?);
            }
        }
        let covered_tokens = object
            .get("covered_tokens")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        let created_offset = object
            .get("created_offset")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        Ok(Self {
            covered,
            children,
            covered_tokens,
            created_offset,
        })
    }

    /// Reads a node back out of a DOM element, rejecting foreign or malformed
    /// payloads instead of guessing.
    pub fn from_element(element: &ElementSnapshot) -> Result<Self, String> {
        if element.kind != SUMMARY_NODE_KIND {
            return Err(format!(
                "element {} is a {}, not a {SUMMARY_NODE_KIND}",
                element.id, element.kind
            ));
        }
        let payload = element
            .payload
            .as_ref()
            .ok_or_else(|| format!("summary node {} carries no payload", element.id))?;
        Self::decode(payload)
    }

    /// Materializes this node as a DOM element.
    pub fn to_element(&self, id: ElementId, kind: SummaryKind, text: String) -> ElementSnapshot {
        let mut element = ElementSnapshot::new(id, SUMMARY_NODE_KIND);
        element.attributes.insert(
            SUMMARY_KIND_ATTRIBUTE.into(),
            crate::patch::TypedValue::String(kind.as_str().to_string()),
        );
        element.attributes.insert(
            "covered_count".into(),
            crate::patch::TypedValue::Integer(self.covered.len() as i64),
        );
        element.attributes.insert(
            "covered_tokens".into(),
            crate::patch::TypedValue::Integer(self.covered_tokens as i64),
        );
        element.attributes.insert(
            "created_offset".into(),
            crate::patch::TypedValue::Integer(self.created_offset as i64),
        );
        element.text = text;
        element.payload = Some(self.encode());
        element
    }

    pub fn kind_of(element: &ElementSnapshot) -> Option<SummaryKind> {
        match element.attributes.get(SUMMARY_KIND_ATTRIBUTE) {
            Some(crate::patch::TypedValue::String(value)) => SummaryKind::parse(value),
            _ => None,
        }
    }
}

/// Expands `root` into the de-duplicated list of covered body elements.
///
/// The walk is breadth-first from the root, following `children` and `covered`
/// in stored order, so the same DAG always expands to the same sequence.
/// Callers that need transcript order re-order the result against the body
/// (see `Read summary://<id>`); traversal is iterative and cycle-safe, so a
/// corrupted or hand-edited DAG cannot make this recurse forever.
pub fn expand_covered(nodes: &BTreeMap<ElementId, SummaryNode>, root: &ElementId) -> Vec<ElementId> {
    let mut visited: BTreeSet<&ElementId> = BTreeSet::new();
    let mut expanded: Vec<ElementId> = Vec::new();
    let mut seen: BTreeSet<ElementId> = BTreeSet::new();
    let mut queue = std::collections::VecDeque::from([root]);
    while let Some(id) = queue.pop_front() {
        if !visited.insert(id) {
            continue;
        }
        let Some(node) = nodes.get(id) else {
            continue;
        };
        queue.extend(node.children.iter());
        for covered in &node.covered {
            if seen.insert(covered.clone()) {
                expanded.push(covered.clone());
            }
        }
    }
    expanded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::TypedValue;

    fn element_id(value: &str) -> ElementId {
        ElementId::new(value).unwrap()
    }

    #[test]
    fn node_payload_round_trips_through_a_dom_element() {
        let node = SummaryNode {
            covered: vec![element_id("e-1"), element_id("e-2")],
            children: vec![element_id("sum-child")],
            covered_tokens: 4096,
            created_offset: 17,
        };
        let element = node.to_element(element_id("sum-1"), SummaryKind::Leaf, "inv".into());
        assert_eq!(element.kind, SUMMARY_NODE_KIND);
        assert_eq!(SummaryNode::kind_of(&element), Some(SummaryKind::Leaf));
        assert_eq!(element.attributes["covered_count"], TypedValue::Integer(2));
        assert_eq!(SummaryNode::from_element(&element).unwrap(), node);
    }

    #[test]
    fn malformed_payloads_are_rejected_not_guessed() {
        let mut element = ElementSnapshot::new(element_id("sum-1"), SUMMARY_NODE_KIND);
        assert!(SummaryNode::from_element(&element).is_err());
        element.payload = Some(serde_json::json!({"covered": "not-an-array"}));
        assert!(SummaryNode::from_element(&element).is_err());
        element.payload = Some(serde_json::json!({"covered": [], "surprise": 1}));
        assert!(SummaryNode::from_element(&element).is_err());
        element.payload = Some(serde_json::json!({"covered": ["bad id!"]}));
        assert!(SummaryNode::from_element(&element).is_err());
        element.payload = Some(serde_json::json!({"covered": ["sum-1"]}));
        assert!(SummaryNode::from_element(&element).is_ok());
    }

    #[test]
    fn expansion_walks_children_and_survives_cycles() {
        let mut nodes = BTreeMap::new();
        nodes.insert(
            element_id("leaf-1"),
            SummaryNode {
                covered: vec![element_id("e-1"), element_id("e-2")],
                children: vec![],
                ..Default::default()
            },
        );
        nodes.insert(
            element_id("leaf-2"),
            SummaryNode {
                covered: vec![element_id("e-2"), element_id("e-3")],
                children: vec![],
                ..Default::default()
            },
        );
        nodes.insert(
            element_id("condensed"),
            SummaryNode {
                covered: vec![],
                children: vec![element_id("leaf-1"), element_id("leaf-2")],
                ..Default::default()
            },
        );
        assert_eq!(
            expand_covered(&nodes, &element_id("condensed")),
            vec![element_id("e-1"), element_id("e-2"), element_id("e-3")]
        );

        // A cycle must terminate and still report every reachable element.
        nodes.insert(
            element_id("leaf-1"),
            SummaryNode {
                covered: vec![element_id("e-1")],
                children: vec![element_id("condensed")],
                ..Default::default()
            },
        );
        assert_eq!(
            expand_covered(&nodes, &element_id("condensed")),
            vec![element_id("e-1"), element_id("e-2"), element_id("e-3")]
        );
    }
}
