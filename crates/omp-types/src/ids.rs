use serde::{Deserialize, Serialize};
use std::fmt;

/// Wire-safe opaque name. Generation uses OS entropy, never branch-local indexes.
macro_rules! ids {
    ($($name:ident),* $(,)?) => { $(
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
        #[serde(try_from = "String", into = "String")]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, crate::StructuredError> {
                let value = value.into();
                // Byte length is correct here (not `chars().count()`): the
                // charset below is ASCII-only, so bytes == characters.
                if value.is_empty() || value.len() > 128 || !value.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)) {
                    return Err(crate::StructuredError::new("invalid_id", "IDs require 1..128 ASCII letters, digits, dots, underscores or hyphens", false));
                }
                Ok(Self(value))
            }
            pub fn mint() -> Self { Self(uuid::Uuid::new_v4().to_string()) }
            pub fn as_str(&self) -> &str { &self.0 }
            pub fn as_ref_str(&self) -> &str { &self.0 }
        }
        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str { &self.0 }
        }
        impl std::borrow::Borrow<str> for $name {
            fn borrow(&self) -> &str { &self.0 }
        }
        impl TryFrom<String> for $name {
            type Error = crate::StructuredError;
            fn try_from(value: String) -> Result<Self, Self::Error> { Self::new(value) }
        }
        impl From<$name> for String { fn from(id: $name) -> Self { id.0 } }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
        }
    )* };
}
ids!(
    SessionId,
    ElementId,
    JobId,
    DirectorId,
    ToolCallId,
    ArtifactId,
    ActorId,
    BranchId,
    WorkspaceViewId
);

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize, Ord, PartialOrd,
)]
#[serde(transparent)]
pub struct JournalOffset(pub u64);
