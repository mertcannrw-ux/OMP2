use crate::StructuredError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PROTOCOL_MAJOR: u16 = 1;
pub const PROTOCOL_MINOR: u16 = 0;
pub const JOURNAL_FORMAT: &str = "patch@1";
pub const TOOL_PROTOCOL: &str = "tool@1";
pub const JOB_PROTOCOL: &str = "job@1";
pub const COMPONENT_PROTOCOL: &str = "component@1";
pub const MAX_WIRE_BYTES: usize = 1_048_576;
pub const MAX_PATCH_OPS: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

impl ProtocolVersion {
    pub const CURRENT: Self = Self {
        major: PROTOCOL_MAJOR,
        minor: PROTOCOL_MINOR,
    };
    /// Minor versions are forward-compatible by policy: a `1.0` host accepts
    /// a `1.99` peer (unknown minor features are ignored via
    /// `deny_unknown_fields`-free extension points), but a newer-minor host
    /// talking to an older-minor peer must not assume new fields. Major
    /// equality is the only hard gate.
    pub fn negotiate(
        self,
        peer: Self,
        advertised: &BTreeSet<String>,
        required: &BTreeSet<String>,
    ) -> Result<(), StructuredError> {
        if self.major != peer.major {
            return Err(StructuredError::new(
                "protocol_major_mismatch",
                "Protocol major versions differ",
                false,
            ));
        }
        if !required.is_subset(advertised) {
            return Err(StructuredError::new(
                "protocol_missing_feature",
                "Peer does not advertise every required feature",
                false,
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Handshake {
    pub version: ProtocolVersion,
    pub features: BTreeSet<String>,
    pub required_features: BTreeSet<String>,
}
impl Handshake {
    pub fn current() -> Self {
        Self {
            version: ProtocolVersion::CURRENT,
            features: [
                JOURNAL_FORMAT,
                TOOL_PROTOCOL,
                JOB_PROTOCOL,
                COMPONENT_PROTOCOL,
            ]
            .map(str::to_owned)
            .into(),
            required_features: BTreeSet::new(),
        }
    }
    pub fn accept(&self, peer: &Self) -> Result<(), StructuredError> {
        self.version
            .negotiate(peer.version, &peer.features, &self.required_features)?;
        peer.version
            .negotiate(self.version, &self.features, &peer.required_features)
    }
}

pub fn decode_wire<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, StructuredError> {
    if bytes.len() > MAX_WIRE_BYTES {
        return Err(StructuredError::new(
            "wire_size_limit",
            "Wire message exceeds one MiB",
            false,
        ));
    }
    serde_json::from_slice(bytes)
        .map_err(|e| StructuredError::new("invalid_wire", e.to_string(), false))
}
