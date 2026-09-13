use crate::{ActorId, ElementId, JournalOffset, MAX_PATCH_OPS, StructuredError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum TypedValue {
    Null,
    Bool(bool),
    Integer(i64),
    Number(f64),
    String(String),
    Json(serde_json::Value),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElementSnapshot {
    pub id: ElementId,
    pub schema_version: u16,
    pub kind: String,
    pub attributes: BTreeMap<String, TypedValue>,
    pub text: String,
    pub payload: Option<serde_json::Value>,
}
impl ElementSnapshot {
    pub fn new(id: ElementId, kind: impl Into<String>) -> Self {
        Self {
            id,
            schema_version: 1,
            kind: kind.into(),
            attributes: BTreeMap::new(),
            text: String::new(),
            payload: None,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum PatchOp {
    Create {
        parent: ElementId,
        index: u32,
        element: ElementSnapshot,
    },
    Delete {
        element: ElementId,
    },
    Move {
        element: ElementId,
        parent: ElementId,
        index: u32,
    },
    SetAttribute {
        element: ElementId,
        name: String,
        value: TypedValue,
    },
    RemoveAttribute {
        element: ElementId,
        name: String,
    },
    ReplaceText {
        element: ElementId,
        text: String,
    },
    AppendText {
        element: ElementId,
        text: String,
    },
    ReplacePayload {
        element: ElementId,
        payload: serde_json::Value,
    },
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum PatchAuthor {
    Actor(ActorId),
    Element(ElementId),
}
impl From<ActorId> for PatchAuthor {
    fn from(id: ActorId) -> Self {
        Self::Actor(id)
    }
}
impl From<ElementId> for PatchAuthor {
    fn from(id: ElementId) -> Self {
        Self::Element(id)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Patch {
    pub base_offset: JournalOffset,
    pub result_offset: JournalOffset,
    pub by: PatchAuthor,
    pub reason: String,
    pub ops: Vec<PatchOp>,
}
impl Patch {
    /// Bound on any single text/attribute-value/payload field inside one op.
    ///
    /// Set at the wire cap: the serialized-patch check below already bounds
    /// the patch *total*, so this changes no accepted behavior today — it
    /// pins the per-op share explicitly so a future wire-cap increase cannot
    /// silently let one op bloat DOM memory per patch. (Job output and stream
    /// deltas legitimately approach this size; do NOT lower it without
    /// chunking those producers first.)
    pub const MAX_OP_TEXT_BYTES: usize = crate::MAX_WIRE_BYTES;
    pub fn validate(&self) -> Result<(), StructuredError> {
        if self.result_offset <= self.base_offset
            || self.reason.trim().is_empty()
            || self.reason.len() > 4096
            || self.ops.len() > MAX_PATCH_OPS
        {
            return Err(StructuredError::new(
                "invalid_patch",
                "Invalid offset progression, intent or operation count",
                false,
            ));
        }
        for op in &self.ops {
            match op {
                PatchOp::Create { element, .. } => {
                    if element.schema_version != 1
                        || element.kind.is_empty()
                        || element.kind.len() > 128
                        || !element
                            .kind
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
                    {
                        return Err(StructuredError::new(
                            "invalid_element",
                            "Unsupported element schema or invalid kind",
                            false,
                        ));
                    }
                    if element.text.len() > Self::MAX_OP_TEXT_BYTES {
                        return Err(StructuredError::new(
                            "invalid_element",
                            "Element text exceeds per-operation limit",
                            false,
                        ));
                    }
                }
                PatchOp::SetAttribute { name, value, .. } => {
                    if name.is_empty() || name.len() > 256 {
                        return Err(StructuredError::new(
                            "invalid_patch",
                            "Attribute name must be 1..=256 bytes",
                            false,
                        ));
                    }
                    if let TypedValue::Number(value) = value
                        && !value.is_finite() {
                            // NaN/Inf are not valid JSON and would fail
                            // serialization late, after partial application.
                            return Err(StructuredError::new(
                                "invalid_patch",
                                "Attribute number must be finite",
                                false,
                            ));
                        }
                    if let TypedValue::String(value) = value
                        && value.len() > Self::MAX_OP_TEXT_BYTES {
                            return Err(StructuredError::new(
                                "invalid_patch",
                                "Attribute value exceeds per-operation limit",
                                false,
                            ));
                        }
                }
                PatchOp::ReplaceText { text, .. } | PatchOp::AppendText { text, .. } => {
                    if text.len() > Self::MAX_OP_TEXT_BYTES {
                        return Err(StructuredError::new(
                            "invalid_patch",
                            "Text payload exceeds per-operation limit",
                            false,
                        ));
                    }
                }
                PatchOp::ReplacePayload { payload, .. } => {
                    if serde_json::to_string(payload)
                        .is_ok_and(|rendered| rendered.len() > Self::MAX_OP_TEXT_BYTES)
                    {
                        return Err(StructuredError::new(
                            "invalid_patch",
                            "Payload exceeds per-operation limit",
                            false,
                        ));
                    }
                }
                PatchOp::RemoveAttribute { name, .. } => {
                    if name.is_empty() || name.len() > 256 {
                        return Err(StructuredError::new(
                            "invalid_patch",
                            "Attribute name must be 1..=256 bytes",
                            false,
                        ));
                    }
                }
                PatchOp::Delete { .. } | PatchOp::Move { .. } => {}
            }
        }
        Ok(())
    }
}
