use omp_types::{ActorId, ElementId, JournalOffset, Patch, PatchOp, TypedValue};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use thiserror::Error;

/// ConVar declaration flags controlling scope, persistence, and replication.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Default)]
pub struct ConVarFlags(pub u32);

impl ConVarFlags {
    pub const NONE: Self = Self(0);
    /// Session-scoped value stored as DOM nodes and journaled.
    pub const SESSION: Self = Self(1 << 0);
    /// Replicated to remote clients and spectators.
    pub const REPLICATED: Self = Self(1 << 1);
    /// Client/userinfo value saved in user profile/settings.
    pub const USERINFO: Self = Self(1 << 2);
    /// Archived to persistent configuration files.
    pub const ARCHIVE: Self = Self(1 << 3);
    /// Privileged/cheat command flag.
    pub const CHEAT: Self = Self(1 << 4);
    /// Emits a notification when changed.
    pub const NOTIFY: Self = Self(1 << 5);
    /// Read-only value; cannot be modified by user commands directly.
    pub const READONLY: Self = Self(1 << 6);
    /// Automatically inherited by child subagent processes/sessions.
    pub const INHERIT: Self = Self(1 << 7);

    pub const fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn intersects(&self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }
}

impl std::ops::BitOr for ConVarFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for ConVarFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// TriState value supporting Unknown capability/setting without false coercion.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Default)]
pub enum TriState {
    False,
    True,
    #[default]
    Unknown,
}

impl fmt::Display for TriState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::False => write!(f, "0"),
            Self::True => write!(f, "1"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

impl FromStr for TriState {
    type Err = ConVarError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(Self::True),
            "0" | "false" | "no" | "off" => Ok(Self::False),
            "unknown" | "auto" | "default" | "?" | "null" | "none" => Ok(Self::Unknown),
            other => Err(ConVarError::InvalidValue {
                name: "tristate".into(),
                reason: format!("expected boolean or 'unknown', got '{}'", other),
            }),
        }
    }
}

#[derive(Debug, Error)]
pub enum ConVarError {
    #[error("convar '{0}' not found")]
    NotFound(String),
    #[error("convar '{0}' is read-only")]
    ReadOnly(String),
    #[error("convar '{0}' is cheat-protected and cheats are not enabled")]
    CheatProtected(String),
    #[error("invalid value for convar '{name}': {reason}")]
    InvalidValue { name: String, reason: String },
    #[error("type mismatch for convar '{name}': expected {expected}, got {actual}")]
    TypeMismatch {
        name: String,
        expected: &'static str,
        actual: String,
    },
}

pub type ValidatorFn = Arc<dyn Fn(&TypedValue) -> Result<(), String> + Send + Sync>;
pub type SerializerFn = Arc<dyn Fn(&TypedValue) -> String + Send + Sync>;
pub type ChangeHookFn = Arc<dyn Fn(&TypedValue, &TypedValue) + Send + Sync>;
pub type ParserFn = Arc<dyn Fn(&str) -> Result<TypedValue, String> + Send + Sync>;

/// Complete definition of a ConVar with declaration metadata, flags, and hooks.
#[derive(Clone)]
pub struct ConVarDefinition {
    pub name: String,
    pub help: String,
    pub flags: ConVarFlags,
    pub default_value: TypedValue,
    pub validator: Option<ValidatorFn>,
    pub serializer: Option<SerializerFn>,
    pub change_hook: Option<ChangeHookFn>,
    pub parser: Option<ParserFn>,
}

impl ConVarDefinition {
    pub fn new(
        name: impl Into<String>,
        default_value: TypedValue,
        help: impl Into<String>,
        flags: ConVarFlags,
    ) -> Self {
        Self {
            name: name.into(),
            help: help.into(),
            flags,
            default_value,
            validator: None,
            serializer: None,
            change_hook: None,
            parser: None,
        }
    }

    pub fn with_validator(
        mut self,
        validator: impl Fn(&TypedValue) -> Result<(), String> + Send + Sync + 'static,
    ) -> Self {
        self.validator = Some(Arc::new(validator));
        self
    }

    pub fn with_serializer(
        mut self,
        serializer: impl Fn(&TypedValue) -> String + Send + Sync + 'static,
    ) -> Self {
        self.serializer = Some(Arc::new(serializer));
        self
    }

    pub fn with_change_hook(
        mut self,
        hook: impl Fn(&TypedValue, &TypedValue) + Send + Sync + 'static,
    ) -> Self {
        self.change_hook = Some(Arc::new(hook));
        self
    }

    pub fn with_parser(
        mut self,
        parser: impl Fn(&str) -> Result<TypedValue, String> + Send + Sync + 'static,
    ) -> Self {
        self.parser = Some(Arc::new(parser));
        self
    }
}

/// Helper trait for converting between Rust types and TypedValue.
pub trait ConVarType: Sized {
    fn to_typed_value(&self) -> TypedValue;
    fn from_typed_value(val: &TypedValue) -> Result<Self, String>;
    fn parse_str(raw: &str) -> Result<Self, String>;
}

impl ConVarType for bool {
    fn to_typed_value(&self) -> TypedValue {
        TypedValue::Bool(*self)
    }
    fn from_typed_value(val: &TypedValue) -> Result<Self, String> {
        match val {
            TypedValue::Bool(b) => Ok(*b),
            TypedValue::Integer(i) => Ok(*i != 0),
            TypedValue::String(s) => match s.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Ok(true),
                "0" | "false" | "no" | "off" => Ok(false),
                "unknown" | "auto" | "default" | "?" => Err(
                    "cannot convert 'unknown' tristate to boolean; explicit value required".into(),
                ),
                _ => Err(format!("cannot convert '{}' to bool", s)),
            },
            _ => Err("expected boolean".into()),
        }
    }
    fn parse_str(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            "unknown" | "auto" | "default" | "?" => {
                Err("cannot convert 'unknown' to boolean; explicit value required".into())
            }
            _ => Err(format!("invalid boolean: '{}'", raw)),
        }
    }
}

impl ConVarType for i64 {
    fn to_typed_value(&self) -> TypedValue {
        TypedValue::Integer(*self)
    }
    fn from_typed_value(val: &TypedValue) -> Result<Self, String> {
        match val {
            TypedValue::Integer(i) => Ok(*i),
            TypedValue::Number(n) => Ok(*n as i64),
            TypedValue::String(s) => s
                .trim()
                .parse::<i64>()
                .map_err(|e| format!("cannot parse integer '{}': {}", s, e)),
            _ => Err("expected integer".into()),
        }
    }
    fn parse_str(raw: &str) -> Result<Self, String> {
        raw.trim()
            .parse::<i64>()
            .map_err(|e| format!("cannot parse integer '{}': {}", raw, e))
    }
}

impl ConVarType for f64 {
    fn to_typed_value(&self) -> TypedValue {
        TypedValue::Number(*self)
    }
    fn from_typed_value(val: &TypedValue) -> Result<Self, String> {
        match val {
            TypedValue::Number(n) => Ok(*n),
            TypedValue::Integer(i) => Ok(*i as f64),
            TypedValue::String(s) => s
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("cannot parse float '{}': {}", s, e)),
            _ => Err("expected number".into()),
        }
    }
    fn parse_str(raw: &str) -> Result<Self, String> {
        raw.trim()
            .parse::<f64>()
            .map_err(|e| format!("cannot parse float '{}': {}", raw, e))
    }
}

impl ConVarType for String {
    fn to_typed_value(&self) -> TypedValue {
        TypedValue::String(self.clone())
    }
    fn from_typed_value(val: &TypedValue) -> Result<Self, String> {
        match val {
            TypedValue::String(s) => Ok(s.clone()),
            TypedValue::Bool(b) => Ok(b.to_string()),
            TypedValue::Integer(i) => Ok(i.to_string()),
            TypedValue::Number(n) => Ok(n.to_string()),
            TypedValue::Null => Ok(String::new()),
            TypedValue::Json(j) => Ok(j.to_string()),
        }
    }
    fn parse_str(raw: &str) -> Result<Self, String> {
        Ok(raw.to_string())
    }
}

impl ConVarType for TriState {
    fn to_typed_value(&self) -> TypedValue {
        match self {
            Self::False => TypedValue::Bool(false),
            Self::True => TypedValue::Bool(true),
            Self::Unknown => TypedValue::String("unknown".into()),
        }
    }
    fn from_typed_value(val: &TypedValue) -> Result<Self, String> {
        match val {
            TypedValue::Bool(b) => Ok(if *b { Self::True } else { Self::False }),
            TypedValue::String(s) => s.parse::<TriState>().map_err(|e| e.to_string()),
            // Only 0/1 map; anything else is a type error rather than a
            // silent `Unknown` (so `2` cannot masquerade as unset).
            TypedValue::Integer(0) => Ok(Self::False),
            TypedValue::Integer(1) => Ok(Self::True),
            TypedValue::Integer(other) => {
                Err(format!("tristate integer must be 0 or 1, got {other}"))
            }
            TypedValue::Null => Ok(Self::Unknown),
            _ => Err("expected tristate value".into()),
        }
    }
    fn parse_str(raw: &str) -> Result<Self, String> {
        raw.parse::<TriState>().map_err(|e| e.to_string())
    }
}

impl ConVarType for Option<i64> {
    fn to_typed_value(&self) -> TypedValue {
        match self {
            Some(i) => TypedValue::Integer(*i),
            None => TypedValue::Null,
        }
    }
    fn from_typed_value(val: &TypedValue) -> Result<Self, String> {
        match val {
            TypedValue::Null => Ok(None),
            TypedValue::Integer(i) => Ok(Some(*i)),
            TypedValue::Number(n) => Ok(Some(*n as i64)),
            TypedValue::String(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty()
                    || trimmed.eq_ignore_ascii_case("null")
                    || trimmed.eq_ignore_ascii_case("none")
                {
                    Ok(None)
                } else {
                    trimmed
                        .parse::<i64>()
                        .map(Some)
                        .map_err(|e| format!("cannot parse integer '{}': {}", s, e))
                }
            }
            _ => Err("expected nullable integer".into()),
        }
    }
    fn parse_str(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty()
            || trimmed.eq_ignore_ascii_case("null")
            || trimmed.eq_ignore_ascii_case("none")
        {
            Ok(None)
        } else {
            trimmed
                .parse::<i64>()
                .map(Some)
                .map_err(|e| format!("cannot parse integer '{}': {}", raw, e))
        }
    }
}

impl ConVarType for serde_json::Value {
    fn to_typed_value(&self) -> TypedValue {
        if self.is_null() {
            TypedValue::Null
        } else {
            TypedValue::Json(self.clone())
        }
    }
    fn from_typed_value(val: &TypedValue) -> Result<Self, String> {
        match val {
            TypedValue::Null => Ok(serde_json::Value::Null),
            TypedValue::Json(j) => Ok(j.clone()),
            TypedValue::String(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") {
                    Ok(serde_json::Value::Null)
                } else {
                    serde_json::from_str(trimmed)
                        .map_err(|e| format!("cannot parse JSON '{}': {}", s, e))
                }
            }
            TypedValue::Bool(b) => Ok(serde_json::Value::Bool(*b)),
            TypedValue::Integer(i) => Ok(serde_json::json!(*i)),
            TypedValue::Number(n) => Ok(serde_json::json!(*n)),
        }
    }
    fn parse_str(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") {
            Ok(serde_json::Value::Null)
        } else {
            serde_json::from_str(trimmed).map_err(|e| format!("cannot parse JSON '{}': {}", raw, e))
        }
    }
}

#[derive(Clone)]
pub struct ConVarStore {
    definitions: BTreeMap<String, ConVarDefinition>,
    values: BTreeMap<String, TypedValue>,
    dirty_session: BTreeSet<String>,
    cheats_enabled: bool,
    defer_hooks: bool,
    pending_hooks: Vec<(ChangeHookFn, TypedValue, TypedValue)>,
}

/// Typed ConVar declaration.
#[derive(Clone)]
pub struct ConVar<T: ConVarType> {
    pub name: String,
    pub default: T,
    pub help: String,
    pub flags: ConVarFlags,
    pub validator: Option<ValidatorFn>,
    pub serializer: Option<SerializerFn>,
    pub change_hook: Option<ChangeHookFn>,
}

impl<T: ConVarType + Clone + Send + Sync + 'static> ConVar<T> {
    pub fn new(
        name: impl Into<String>,
        default: T,
        help: impl Into<String>,
        flags: ConVarFlags,
    ) -> Self {
        Self {
            name: name.into(),
            default,
            help: help.into(),
            flags,
            validator: None,
            serializer: None,
            change_hook: None,
        }
    }

    pub fn with_validator(
        mut self,
        validator: impl Fn(&TypedValue) -> Result<(), String> + Send + Sync + 'static,
    ) -> Self {
        self.validator = Some(Arc::new(validator));
        self
    }

    pub fn with_serializer(
        mut self,
        serializer: impl Fn(&TypedValue) -> String + Send + Sync + 'static,
    ) -> Self {
        self.serializer = Some(Arc::new(serializer));
        self
    }

    pub fn with_change_hook(
        mut self,
        hook: impl Fn(&TypedValue, &TypedValue) + Send + Sync + 'static,
    ) -> Self {
        self.change_hook = Some(Arc::new(hook));
        self
    }

    pub fn to_definition(&self) -> ConVarDefinition {
        let mut def = ConVarDefinition::new(
            &self.name,
            self.default.to_typed_value(),
            &self.help,
            self.flags,
        );
        def.validator = self.validator.clone();
        def.serializer = self.serializer.clone();
        def.change_hook = self.change_hook.clone();
        def.parser = Some(Arc::new(|s| T::parse_str(s).map(|v| v.to_typed_value())));
        def
    }
}
impl Default for ConVarStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ConVarStore {
    pub fn new() -> Self {
        Self {
            definitions: BTreeMap::new(),
            values: BTreeMap::new(),
            dirty_session: BTreeSet::new(),
            cheats_enabled: false,
            defer_hooks: false,
            pending_hooks: Vec::new(),
        }
    }

    pub fn cheats_enabled(&self) -> bool {
        self.cheats_enabled
    }

    pub fn set_cheats_enabled(&mut self, enabled: bool) {
        self.cheats_enabled = enabled;
    }

    pub fn begin_transaction(&mut self) {
        self.defer_hooks = true;
        self.pending_hooks.clear();
    }

    pub(crate) fn transaction_active(&self) -> bool {
        self.defer_hooks
    }
    pub fn commit_transaction(&mut self) {
        self.defer_hooks = false;
        let hooks = std::mem::take(&mut self.pending_hooks);
        for (hook, old_val, new_val) in hooks {
            hook(&old_val, &new_val);
        }
    }

    /// Register a ConVar declaration.
    pub fn register<T: ConVarType + Clone + Send + Sync + 'static>(&mut self, convar: ConVar<T>) {
        self.register_def(convar.to_definition());
    }

    /// Register a ConVar definition directly.
    pub fn register_def(&mut self, def: ConVarDefinition) {
        let name = def.name.clone();
        if !self.values.contains_key(&name) {
            self.values.insert(name.clone(), def.default_value.clone());
        }
        self.definitions.insert(name, def);
    }

    /// Retrieve definition by name.
    pub fn get_def(&self, name: &str) -> Option<&ConVarDefinition> {
        self.definitions.get(name)
    }

    /// Retrieve raw effective TypedValue by name.
    pub fn get(&self, name: &str) -> Option<&TypedValue> {
        self.values.get(name)
    }

    /// Retrieve typed effective value.
    pub fn get_typed<T: ConVarType>(&self, name: &str) -> Result<T, ConVarError> {
        let val = self
            .values
            .get(name)
            .ok_or_else(|| ConVarError::NotFound(name.to_string()))?;
        T::from_typed_value(val).map_err(|e| ConVarError::TypeMismatch {
            name: name.to_string(),
            expected: std::any::type_name::<T>(),
            actual: e,
        })
    }

    /// Set value with validation, flag checks, change hooks, and dirty session tracking.
    pub fn set(&mut self, name: &str, new_val: TypedValue) -> Result<TypedValue, ConVarError> {
        let def = self
            .definitions
            .get(name)
            .ok_or_else(|| ConVarError::NotFound(name.to_string()))?
            .clone();

        if def.flags.contains(ConVarFlags::READONLY) {
            return Err(ConVarError::ReadOnly(name.to_string()));
        }

        if def.flags.contains(ConVarFlags::CHEAT) && !self.cheats_enabled {
            return Err(ConVarError::CheatProtected(name.to_string()));
        }

        let compatible = match (&def.default_value, &new_val) {
            (TypedValue::Null, _) => true,
            (TypedValue::Bool(_), TypedValue::Bool(_))
            | (TypedValue::Integer(_), TypedValue::Integer(_))
            | (TypedValue::Number(_), TypedValue::Number(_))
            | (TypedValue::String(_), TypedValue::String(_))
            | (TypedValue::Json(_), TypedValue::Json(_)) => true,
            // TriState serializes unknown as a string and known values as booleans.
            (TypedValue::String(default), TypedValue::Bool(_)) if default == "unknown" => true,
            _ => false,
        };
        if !compatible {
            return Err(ConVarError::InvalidValue {
                name: name.into(),
                reason: "value does not match the declared type".into(),
            });
        }
        if let Some(validator) = &def.validator
            && let Err(reason) = validator(&new_val) {
                return Err(ConVarError::InvalidValue {
                    name: name.to_string(),
                    reason,
                });
            }

        let old_val = self.values.get(name).cloned().unwrap_or(TypedValue::Null);

        if let Some(hook) = &def.change_hook {
            if self.defer_hooks {
                self.pending_hooks
                    .push((hook.clone(), old_val.clone(), new_val.clone()));
            } else {
                hook(&old_val, &new_val);
            }
        }

        if def.flags.contains(ConVarFlags::SESSION) {
            self.dirty_session.insert(name.to_string());
        }
        if name == "sv_cheats" {
            // Normalize truthy spellings: direct `set("sv_cheats", 1)` must
            // behave like the string parser (`set_from_str`), not silently
            // store `1` without enabling cheats.
            self.cheats_enabled = match &new_val {
                TypedValue::Bool(b) => *b,
                TypedValue::Integer(i) => *i != 0,
                TypedValue::String(s) => {
                    matches!(s.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
                }
                _ => false,
            };
        }

        self.values.insert(name.to_string(), new_val.clone());
        Ok(new_val)
    }

    /// Set value by parsing a string representation.
    pub fn set_from_str(&mut self, name: &str, raw: &str) -> Result<TypedValue, ConVarError> {
        let def = self
            .definitions
            .get(name)
            .ok_or_else(|| ConVarError::NotFound(name.to_string()))?
            .clone();

        if def.flags.contains(ConVarFlags::READONLY) {
            return Err(ConVarError::ReadOnly(name.to_string()));
        }

        if def.flags.contains(ConVarFlags::CHEAT) && !self.cheats_enabled {
            return Err(ConVarError::CheatProtected(name.to_string()));
        }

        let typed_val = if let Some(parser) = &def.parser {
            parser(raw).map_err(|e| ConVarError::InvalidValue {
                name: name.to_string(),
                reason: e,
            })?
        } else {
            match &def.default_value {
                TypedValue::Bool(_) => TypedValue::Bool(bool::parse_str(raw).map_err(|e| {
                    ConVarError::InvalidValue {
                        name: name.to_string(),
                        reason: e,
                    }
                })?),
                TypedValue::Integer(_) => {
                    TypedValue::Integer(i64::parse_str(raw).map_err(|e| {
                        ConVarError::InvalidValue {
                            name: name.to_string(),
                            reason: e,
                        }
                    })?)
                }
                TypedValue::Number(_) => TypedValue::Number(f64::parse_str(raw).map_err(|e| {
                    ConVarError::InvalidValue {
                        name: name.to_string(),
                        reason: e,
                    }
                })?),
                TypedValue::String(_) => {
                    if let Ok(ts) = raw.parse::<TriState>() {
                        if raw.trim().eq_ignore_ascii_case("unknown") {
                            ts.to_typed_value()
                        } else {
                            TypedValue::String(raw.to_string())
                        }
                    } else {
                        TypedValue::String(raw.to_string())
                    }
                }
                TypedValue::Null => TypedValue::String(raw.to_string()),
                TypedValue::Json(_) => serde_json::from_str::<serde_json::Value>(raw)
                    .map(TypedValue::Json)
                    .map_err(|e| ConVarError::InvalidValue {
                        name: name.to_string(),
                        reason: e.to_string(),
                    })?,
            }
        };
        self.set(name, typed_val)
    }

    /// Reset convar to its default value.
    pub fn reset(&mut self, name: &str) -> Result<TypedValue, ConVarError> {
        let def = self
            .definitions
            .get(name)
            .ok_or_else(|| ConVarError::NotFound(name.to_string()))?
            .clone();

        if def.flags.contains(ConVarFlags::READONLY) {
            return Err(ConVarError::ReadOnly(name.to_string()));
        }

        if def.flags.contains(ConVarFlags::CHEAT) && !self.cheats_enabled {
            return Err(ConVarError::CheatProtected(name.to_string()));
        }

        self.set(name, def.default_value.clone())
    }

    /// Iterate over all registered convars.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ConVarDefinition, &TypedValue)> {
        self.definitions.iter().map(move |(k, def)| {
            let val = self.values.get(k).unwrap_or(&def.default_value);
            (k, def, val)
        })
    }

    /// Hydrate session-scoped ConVar values directly from the authoritative DOM snapshot.
    pub fn hydrate_from_dom(&mut self, snapshot: &omp_state::SessionSnapshot) {
        let globals = snapshot.session_globals();
        for (name, definition) in &self.definitions {
            if definition.flags.contains(ConVarFlags::SESSION) {
                self.values.insert(
                    name.clone(),
                    globals
                        .get(name)
                        .map(|value| (*value).clone())
                        .unwrap_or_else(|| definition.default_value.clone()),
                );
            }
        }
        self.cheats_enabled = match self.values.get("sv_cheats") {
            Some(TypedValue::Bool(b)) => *b,
            Some(TypedValue::Integer(i)) => *i != 0,
            Some(TypedValue::String(s)) => {
                matches!(s.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
            }
            _ => false,
        };
        self.pending_hooks.clear();
        self.defer_hooks = false;
        self.dirty_session.clear();
    }

    /// Drain dirty session-scoped convars into a journal patch.
    pub fn drain_session_patch(
        &mut self,
        base_offset: JournalOffset,
        result_offset: JournalOffset,
        by: ActorId,
        meta_convars_elem: &ElementId,
    ) -> Option<Patch> {
        if self.dirty_session.is_empty() {
            return None;
        }

        let mut ops = Vec::new();
        for key in std::mem::take(&mut self.dirty_session) {
            if let Some(val) = self.values.get(&key) {
                ops.push(PatchOp::SetAttribute {
                    element: meta_convars_elem.clone(),
                    name: key,
                    value: val.clone(),
                });
            }
        }

        Some(Patch {
            base_offset,
            result_offset,
            by: by.into(),
            reason: "convar session mutation".into(),
            ops,
        })
    }

    /// Seed a child process/session store with current effective values.
    /// All effective values are seeded so child starts with complete configuration.
    pub fn seed_child(&self) -> Self {
        Self {
            definitions: self.definitions.clone(),
            values: self.values.clone(),
            dirty_session: self
                .definitions
                .iter()
                .filter(|(_, def)| def.flags.contains(ConVarFlags::SESSION))
                .map(|(name, _)| name.clone())
                .collect(),
            cheats_enabled: self.cheats_enabled,
            defer_hooks: false,
            pending_hooks: Vec::new(),
        }
    }

    /// Apply a configuration stream (e.g. subagent.cfg or <agent>.cfg overrides)
    /// to this child store.
    pub fn apply_cfg_stream(
        &mut self,
        engine: &mut crate::command::CommandEngine,
        script: &str,
    ) -> Result<Vec<crate::command::CommandEffect>, crate::command::CommandError> {
        engine.execute(script, self)
    }
}

/// Register the standard built-in ConVars specified by the harness plan.
pub fn register_builtin_convars(store: &mut ConVarStore) {
    let session_replicated = ConVarFlags::SESSION | ConVarFlags::REPLICATED;
    let client_archived = ConVarFlags::SESSION | ConVarFlags::USERINFO | ConVarFlags::ARCHIVE;

    store.register(ConVar::new(
        "ai_model",
        "".to_string(),
        "Active inference model identifier",
        session_replicated,
    ));

    store.register(ConVar::new(
        "ai_provider",
        "".to_string(),
        "Active inference provider adapter",
        session_replicated,
    ));

    store.register(ConVar::new(
        "ai_endpoint",
        "".to_string(),
        "Custom API endpoint for active inference provider",
        session_replicated,
    ));

    store.register(ConVar::new(
        "ai_api_key_env",
        "".to_string(),
        "Host environment variable name referencing provider API key",
        session_replicated,
    ));

    store.register(ConVar::new(
        "ai_thinking",
        "auto".to_string(),
        "Reasoning effort: 'auto' (provider default), 'off', or a level the model advertises (/effort)",
        session_replicated,
    ));

    store.register(
        ConVar::new(
            "ai_thinking_levels",
            serde_json::Value::Null,
            "Supported thinking/reasoning effort levels advertised by provider metadata",
            session_replicated | ConVarFlags::READONLY,
        )
        .with_validator(|v| match v {
            TypedValue::Null | TypedValue::Json(_) => Ok(()),
            _ => Err("ai_thinking_levels must be JSON or null".into()),
        }),
    );
    store.register(ConVar::new(
        "ai_fastmode",
        false,
        "Fast mode inference toggle",
        session_replicated,
    ));

    store.register(
        ConVar::new(
            "ai_temperature",
            0.7_f64,
            "Sampling temperature",
            session_replicated,
        )
        .with_validator(|v| match v {
            TypedValue::Number(n) if *n >= 0.0 && *n <= 2.0 => Ok(()),
            _ => Err("ai_temperature must be between 0.0 and 2.0".into()),
        }),
    );

    store.register(
        ConVar::new(
            "ai_max_tokens",
            None::<i64>,
            "Maximum completion tokens per turn (positive integer or null)",
            session_replicated,
        )
        .with_validator(|v| match v {
            TypedValue::Null => Ok(()),
            TypedValue::Integer(i) if *i > 0 => Ok(()),
            _ => Err("ai_max_tokens must be positive or null".into()),
        }),
    );

    store.register(
        ConVar::new(
            "ai_context_length",
            None::<i64>,
            "Nominal context window token length advertised by provider metadata",
            session_replicated | ConVarFlags::READONLY,
        )
        .with_validator(|v| match v {
            TypedValue::Null => Ok(()),
            TypedValue::Integer(i) if *i > 0 => Ok(()),
            _ => Err("ai_context_length must be positive or null".into()),
        }),
    );
    store.register(
        ConVar::new(
            "ai_compaction_threshold",
            0.8_f64,
            "Fraction of the advertised context window at which older turns are elided",
            session_replicated,
        )
        .with_validator(|v| match v {
            TypedValue::Number(n) if *n >= 0.0 && *n <= 1.0 => Ok(()),
            _ => Err("ai_compaction_threshold must be between 0.0 and 1.0".into()),
        }),
    );
    store.register(ConVar::new(
        "cl_showthinking",
        true,
        "Show thinking/reasoning blocks in the client interface",
        client_archived,
    ));

    store.register(ConVar::new(
        "cl_theme",
        "default".to_string(),
        "Active client color theme",
        client_archived,
    ));

    store.register(
        ConVar::new(
            "cl_icon_mode",
            "unicode".to_string(),
            "Icon presentation mode (unicode/nerd/ascii)",
            client_archived,
        )
        .with_validator(|v| match v {
            TypedValue::String(s) if ["unicode", "nerd", "ascii"].contains(&s.as_str()) => Ok(()),
            _ => Err("cl_icon_mode must be 'unicode', 'nerd', or 'ascii'".into()),
        }),
    );

    store.register(
        ConVar::new(
            "cl_resize_policy",
            "rebuild".to_string(),
            "Terminal resize policy (rebuild/preserve/append)",
            client_archived,
        )
        .with_validator(|v| match v {
            TypedValue::String(s) if ["rebuild", "preserve", "append"].contains(&s.as_str()) => {
                Ok(())
            }
            _ => Err("cl_resize_policy must be 'rebuild', 'preserve', or 'append'".into()),
        }),
    );
    store.register(ConVar::new(
        "sandbox_network",
        false,
        "Permit network access in sandbox jobs",
        session_replicated,
    ));

    store.register(
        ConVar::new(
            "sandbox_write_scope",
            "workspace".to_string(),
            "Filesystem write boundary for sandbox jobs (workspace/none)",
            session_replicated,
        )
        .with_validator(|value| match value {
            TypedValue::String(scope) if matches!(scope.as_str(), "workspace" | "none") => Ok(()),
            _ => Err("sandbox_write_scope must be workspace or none".into()),
        }),
    );

    store.register(
        ConVar::new(
            "job_max_concurrency",
            4_i64,
            "Maximum concurrent background jobs",
            session_replicated,
        )
        .with_validator(|v| match v {
            TypedValue::Integer(i) if *i > 0 => Ok(()),
            _ => Err("job_max_concurrency must be positive".into()),
        }),
    );

    store.register(
        ConVar::new(
            "tool_max_output_bytes",
            1_048_576_i64,
            "Maximum allowed output bytes for a tool before truncation",
            session_replicated,
        )
        .with_validator(|v| match v {
            TypedValue::Integer(i) if *i > 0 => Ok(()),
            _ => Err("tool_max_output_bytes must be positive".into()),
        }),
    );

    store.register(ConVar::new(
        "ai_provider_name",
        String::new(),
        "Name of the active provider in the session's provider registry",
        session_replicated,
    ));

    store.register(
        ConVar::new(
            "ai_request_timeout_secs",
            300_i64,
            "Idle budget in seconds for a provider request before the stream is treated as stalled",
            session_replicated,
        )
        .with_validator(|v| match v {
            TypedValue::Integer(i) if (10..=1800).contains(i) => Ok(()),
            _ => Err("ai_request_timeout_secs must be between 10 and 1800".into()),
        }),
    );

    store.register(
        ConVar::new(
            "tool_max_runtime_ms",
            300_000_i64,
            "Maximum runtime in milliseconds for a tool before timeout",
            session_replicated,
        )
        .with_validator(|v| match v {
            TypedValue::Integer(i) if *i > 0 => Ok(()),
            _ => Err("tool_max_runtime_ms must be positive".into()),
        }),
    );

    store.register(ConVar::new(
        "sv_cheats",
        false,
        "Enable privileged/cheat commands and convars",
        ConVarFlags::SESSION | ConVarFlags::NOTIFY,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_max_tokens_convar() {
        let mut store = ConVarStore::new();
        register_builtin_convars(&mut store);

        assert_eq!(store.get("ai_max_tokens"), Some(&TypedValue::Null));
        assert_eq!(
            store.get_typed::<Option<i64>>("ai_max_tokens").unwrap(),
            None
        );

        store.set_from_str("ai_max_tokens", "8192").unwrap();
        assert_eq!(
            store.get_typed::<Option<i64>>("ai_max_tokens").unwrap(),
            Some(8192)
        );
        assert_eq!(store.get_typed::<i64>("ai_max_tokens").unwrap(), 8192);

        store.set_from_str("ai_max_tokens", "null").unwrap();
        assert_eq!(store.get("ai_max_tokens"), Some(&TypedValue::Null));
        assert_eq!(
            store.get_typed::<Option<i64>>("ai_max_tokens").unwrap(),
            None
        );

        // Validator rejects negative and zero
        assert!(store.set_from_str("ai_max_tokens", "0").is_err());
        assert!(store.set_from_str("ai_max_tokens", "-100").is_err());
    }

    #[test]
    fn test_ai_context_length_readonly() {
        let mut store = ConVarStore::new();
        register_builtin_convars(&mut store);

        assert_eq!(store.get("ai_context_length"), Some(&TypedValue::Null));
        assert_eq!(
            store.get_typed::<Option<i64>>("ai_context_length").unwrap(),
            None
        );

        // Read-only cannot be modified via set or set_from_str
        assert!(matches!(
            store.set_from_str("ai_context_length", "128000"),
            Err(ConVarError::ReadOnly(_))
        ));
        assert!(matches!(
            store.set("ai_context_length", TypedValue::Integer(128000)),
            Err(ConVarError::ReadOnly(_))
        ));
    }

    #[test]
    fn test_ai_thinking_levels_readonly() {
        let mut store = ConVarStore::new();
        register_builtin_convars(&mut store);

        assert_eq!(store.get("ai_thinking_levels"), Some(&TypedValue::Null));
        assert!(matches!(
            store.set_from_str("ai_thinking_levels", r#"["low", "high"]"#),
            Err(ConVarError::ReadOnly(_))
        ));
    }
}
