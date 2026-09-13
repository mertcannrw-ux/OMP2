use crate::capability::{CapabilityProfile, NativeToolCost, ProviderCapability, TriState};
use crate::model_taxonomy::{ModelClass, ModelFamily, ModelIdentifier};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CompatError {
    #[error("unknown directive: '{0}'")]
    UnknownDirective(String),
    #[error("unknown value '{value}' for key '{key}'")]
    UnknownValue { key: String, value: String },
    #[error("invalid revision range: {0}")]
    InvalidRevisionRange(String),
    #[error("unreachable rule '{id}': shadowed by rule '{shadowed_by}'")]
    UnreachableRule { id: String, shadowed_by: String },
    #[error(
        "conflicting equal-specificity rules for capability '{cap}': rule '{rule_a}' specifies {val_a}, but rule '{rule_b}' specifies {val_b}"
    )]
    ConflictingEqualSpecificity {
        cap: String,
        rule_a: String,
        val_a: String,
        rule_b: String,
        val_b: String,
    },
    #[error("syntax error at line {line}: {message}")]
    SyntaxError { line: usize, message: String },
}

/// Pattern for matching revision strings.
///
/// Range bounds compare with [`compare_revisions`]: numeric-aware per
/// dot-separated segment (so `10.0 > 9.0`), falling back to lexicographic
/// order for non-numeric segments (dates like `2024-01` still order sanely).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RevisionPattern {
    Exact(String),
    Prefix(String),
    Range {
        min: Option<String>,
        max: Option<String>,
    },
    Wildcard,
}

/// Compare two revision strings segment-wise: numeric segments compare as
/// integers (`10 > 9`), other segments compare lexicographically. Missing
/// trailing segments count as zero (`1.2 == 1.2.0`).
pub fn compare_revisions(left: &str, right: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let mut left_parts = left.split(['.', '-', '_']);
    let mut right_parts = right.split(['.', '-', '_']);
    loop {
        match (left_parts.next(), right_parts.next()) {
            (None, None) => return Ordering::Equal,
            (None, Some(first)) => {
                // Left ran out: equal only if every remaining right segment
                // is zero/empty (`1.2 == 1.2.0`), else left is smaller.
                let rest_zero = first.parse::<u64>().is_ok_and(|n| n == 0) || first.is_empty();
                let all_zero = rest_zero
                    && right_parts.all(|s| s.parse::<u64>().is_ok_and(|n| n == 0) || s.is_empty());
                return if all_zero { Ordering::Equal } else { Ordering::Less };
            }
            (Some(_), None) => {
                // Mirror of the above: re-split is cheap; compare recursively
                // by swapping.
                return compare_revisions(right, left).reverse();
            }
            (Some(l), Some(r)) => {
                let ord = match (l.parse::<u64>(), r.parse::<u64>()) {
                    (Ok(a), Ok(b)) => a.cmp(&b),
                    _ => l.cmp(r),
                };
                if ord != Ordering::Equal {
                    return ord;
                }
            }
        }
    }
}

impl RevisionPattern {
    pub fn matches(&self, rev: Option<&str>) -> bool {
        match (self, rev) {
            (Self::Wildcard, _) => true,
            (Self::Exact(expected), Some(actual)) => expected == actual,
            (Self::Prefix(prefix), Some(actual)) => actual.starts_with(prefix),
            (Self::Range { min, max }, Some(actual)) => {
                use std::cmp::Ordering;
                if let Some(min_val) = min
                    && compare_revisions(actual, min_val.as_str()) == Ordering::Less {
                        return false;
                    }
                if let Some(max_val) = max
                    && compare_revisions(actual, max_val.as_str()) == Ordering::Greater {
                        return false;
                    }
                true
            }
            _ => false,
        }
    }

    pub fn overlaps(&self, other: &RevisionPattern) -> bool {
        match (self, other) {
            (Self::Wildcard, _) | (_, Self::Wildcard) => true,
            (Self::Exact(a), Self::Exact(b)) => a == b,
            (Self::Exact(a), Self::Prefix(p)) | (Self::Prefix(p), Self::Exact(a)) => {
                a.starts_with(p)
            }
            (Self::Prefix(p1), Self::Prefix(p2)) => p1.starts_with(p2) || p2.starts_with(p1),
            (Self::Exact(a), Self::Range { min, max })
            | (Self::Range { min, max }, Self::Exact(a)) => {
                use std::cmp::Ordering;
                let above_min = min.as_ref().is_none_or(|m| {
                    compare_revisions(a.as_str(), m.as_str()) != Ordering::Less
                });
                let below_max = max.as_ref().is_none_or(|m| {
                    compare_revisions(a.as_str(), m.as_str()) != Ordering::Greater
                });
                above_min && below_max
            }
            (
                Self::Range {
                    min: min1,
                    max: max1,
                },
                Self::Range {
                    min: min2,
                    max: max2,
                },
            ) => {
                use std::cmp::Ordering;
                if let (Some(a), Some(b)) = (min1, max2)
                    && compare_revisions(a.as_str(), b.as_str()) == Ordering::Greater {
                        return false;
                    }
                if let (Some(a), Some(b)) = (min2, max1)
                    && compare_revisions(a.as_str(), b.as_str()) == Ordering::Greater {
                        return false;
                    }
                true
            }
            (Self::Prefix(p), Self::Range { min, max })
            | (Self::Range { min, max }, Self::Prefix(p)) => {
                use std::cmp::Ordering;
                if let Some(max_val) = max
                    && compare_revisions(p.as_str(), max_val.as_str()) == Ordering::Greater && !max_val.starts_with(p.as_str()) {
                        return false;
                    }
                if let Some(min_val) = min
                    && compare_revisions(p.as_str(), min_val.as_str()) == Ordering::Less && !min_val.starts_with(p.as_str()) {
                        return false;
                    }
                true
            }
        }
    }

    pub fn covers(&self, other: &RevisionPattern) -> bool {
        match (self, other) {
            (Self::Wildcard, _) => true,
            (Self::Exact(a), Self::Exact(b)) => a == b,
            (Self::Prefix(p), Self::Exact(b)) => b.starts_with(p),
            (Self::Prefix(p1), Self::Prefix(p2)) => p2.starts_with(p1),
            (
                Self::Range {
                    min: min1,
                    max: max1,
                },
                Self::Exact(b),
            ) => {
                use std::cmp::Ordering;
                let above_min = min1.as_ref().is_none_or(|m| {
                    compare_revisions(b.as_str(), m.as_str()) != Ordering::Less
                });
                let below_max = max1.as_ref().is_none_or(|m| {
                    compare_revisions(b.as_str(), m.as_str()) != Ordering::Greater
                });
                above_min && below_max
            }
            (
                Self::Range {
                    min: min1,
                    max: max1,
                },
                Self::Range {
                    min: min2,
                    max: max2,
                },
            ) => {
                let min_covered = match (min1, min2) {
                    (None, _) => true,
                    (Some(_), None) => false,
                    (Some(m1), Some(m2)) => {
                        compare_revisions(m2.as_str(), m1.as_str()) != std::cmp::Ordering::Less
                    }
                };
                let max_covered = match (max1, max2) {
                    (None, _) => true,
                    (Some(_), None) => false,
                    (Some(m1), Some(m2)) => {
                        compare_revisions(m2.as_str(), m1.as_str()) != std::cmp::Ordering::Greater
                    }
                };
                min_covered && max_covered
            }
            _ => false,
        }
    }
}

/// Declarative compatibility rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompatRule {
    pub id: String,
    pub host: Option<String>,
    pub provider: Option<String>,
    pub family: Option<ModelFamily>,
    pub revision: Option<RevisionPattern>,
    pub class: Option<ModelClass>,
    pub capabilities: HashMap<ProviderCapability, TriState>,
    pub native_tool_cost: Option<NativeToolCost>,
    pub priority: i32,
}

impl CompatRule {
    /// Determines the structural specificity level (0..=5).
    /// Precedence order:
    /// 5: exact host + exact model revision
    /// 4: host + family
    /// 3: provider + revision
    /// 2: provider + family
    /// 1: class defaults
    /// 0: unknown / global wildcard
    pub fn specificity_level(&self) -> u8 {
        if self.host.is_some() && matches!(self.revision, Some(RevisionPattern::Exact(_))) {
            5
        } else if self.host.is_some() && self.family.is_some() {
            4
        } else if self.provider.is_some() && self.revision.is_some() {
            3
        } else if self.provider.is_some() && self.family.is_some() {
            2
        } else if self.class.is_some() {
            1
        } else {
            0
        }
    }

    /// Evaluates if this rule applies to the model identifier.
    pub fn matches(&self, ident: &ModelIdentifier) -> bool {
        if let Some(host) = &self.host
            && !ident.host_name().eq_ignore_ascii_case(host) {
                return false;
            }
        if let Some(provider) = &self.provider
            && !ident.provider_name().eq_ignore_ascii_case(provider) {
                return false;
            }
        if let Some(family) = &self.family
            && ident.family() != family {
                return false;
            }
        if let Some(rev_pat) = &self.revision
            && !rev_pat.matches(ident.revision()) {
                return false;
            }
        if let Some(class) = &self.class
            && &ident.class != class {
                return false;
            }
        true
    }

    /// Checks if the selector criteria of `self` and `other` overlap.
    pub fn overlaps_criteria(&self, other: &CompatRule) -> bool {
        if self.specificity_level() != other.specificity_level() {
            return false;
        }

        let host_match = match (&self.host, &other.host) {
            (Some(a), Some(b)) => a.eq_ignore_ascii_case(b),
            _ => true,
        };
        let provider_match = match (&self.provider, &other.provider) {
            (Some(a), Some(b)) => a.eq_ignore_ascii_case(b),
            _ => true,
        };
        let family_match = match (&self.family, &other.family) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        };
        let class_match = match (&self.class, &other.class) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        };
        let rev_match = match (&self.revision, &other.revision) {
            (Some(a), Some(b)) => a.overlaps(b),
            _ => true,
        };

        host_match && provider_match && family_match && class_match && rev_match
    }

    /// Checks whether `self` strictly shadows `other`, rendering `other` unreachable.
    pub fn shadows(&self, other: &CompatRule) -> bool {
        if self.id == other.id {
            return false;
        }

        // Self must have strictly higher precedence
        let higher_prec = self.specificity_level() > other.specificity_level()
            || (self.specificity_level() == other.specificity_level()
                && self.priority > other.priority);
        if !higher_prec {
            return false;
        }

        // Self must match every model that other matches
        if let Some(sh) = &self.host {
            match &other.host {
                Some(oh) if sh.eq_ignore_ascii_case(oh) => {}
                _ => return false,
            }
        }
        if let Some(sp) = &self.provider {
            match &other.provider {
                Some(op) if sp.eq_ignore_ascii_case(op) => {}
                _ => return false,
            }
        }
        if let Some(sf) = &self.family {
            match &other.family {
                Some(of) if sf == of => {}
                _ => return false,
            }
        }
        if let Some(sc) = &self.class {
            match &other.class {
                Some(oc) if sc == oc => {}
                _ => return false,
            }
        }
        if let Some(sr) = &self.revision {
            match &other.revision {
                Some(or) if sr.covers(or) => {}
                _ => return false,
            }
        }

        // Self must cover every capability declared by other
        for cap in other.capabilities.keys() {
            if !self.capabilities.contains_key(cap) {
                return false;
            }
        }

        // If other defines native_tool_cost, self must also define it
        if other.native_tool_cost.is_some() && self.native_tool_cost.is_none() {
            return false;
        }

        true
    }
}

/// Compiler that parses, validates, and indexes compatibility rules into a deterministic table.
pub struct CompatCompiler {
    rules: Vec<CompatRule>,
}

impl Default for CompatCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl CompatCompiler {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: CompatRule) -> Result<(), CompatError> {
        // Validate revision range if present (numeric-aware comparison).
        if let Some(RevisionPattern::Range { min, max }) = &rule.revision
            && let (Some(a), Some(b)) = (min, max)
                && compare_revisions(a.as_str(), b.as_str()) == std::cmp::Ordering::Greater {
                    return Err(CompatError::InvalidRevisionRange(format!(
                        "min '{a}' is greater than max '{b}'"
                    )));
                }

        // Validate equal-specificity conflicts against existing rules
        for existing in &self.rules {
            if existing.id != rule.id
                && existing.specificity_level() == rule.specificity_level()
                && existing.priority == rule.priority
                && existing.overlaps_criteria(&rule)
            {
                // Check capabilities conflict
                for (cap, val) in &rule.capabilities {
                    if let Some(existing_val) = existing.capabilities.get(cap)
                        && val != existing_val {
                            return Err(CompatError::ConflictingEqualSpecificity {
                                cap: cap.to_string(),
                                rule_a: existing.id.clone(),
                                val_a: existing_val.to_string(),
                                rule_b: rule.id.clone(),
                                val_b: val.to_string(),
                            });
                        }
                }
                // Check native tool cost conflict
                if let (Some(cost_a), Some(cost_b)) =
                    (existing.native_tool_cost, rule.native_tool_cost)
                    && cost_a != cost_b {
                        return Err(CompatError::ConflictingEqualSpecificity {
                            cap: "native_tool_cost".to_string(),
                            rule_a: existing.id.clone(),
                            val_a: cost_a.to_string(),
                            rule_b: rule.id.clone(),
                            val_b: cost_b.to_string(),
                        });
                    }
            }

            // Check unreachable / shadowed rules
            if existing.shadows(&rule) {
                return Err(CompatError::UnreachableRule {
                    id: rule.id.clone(),
                    shadowed_by: existing.id.clone(),
                });
            }
            if rule.shadows(existing) {
                return Err(CompatError::UnreachableRule {
                    id: existing.id.clone(),
                    shadowed_by: rule.id.clone(),
                });
            }
        }

        self.rules.push(rule);
        Ok(())
    }

    pub fn parse_kdl_text(&mut self, text: &str) -> Result<(), CompatError> {
        for (line_idx, line) in text.lines().enumerate() {
            let line_num = line_idx + 1;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
                continue;
            }

            if trimmed.starts_with("rule ") {
                let rule = parse_single_rule_line(trimmed, line_num)?;
                self.add_rule(rule)?;
            } else {
                return Err(CompatError::UnknownDirective(trimmed.to_string()));
            }
        }
        Ok(())
    }

    /// Compiles all rules into a deterministic indexed table.
    pub fn compile(mut self) -> Result<CompatTable, CompatError> {
        // Sort deterministically:
        // Highest specificity first (5 down to 0),
        // then highest priority first,
        // then stable alphabetical ID for complete determinism (file order never wins).
        self.rules.sort_by(|a, b| {
            b.specificity_level()
                .cmp(&a.specificity_level())
                .then_with(|| b.priority.cmp(&a.priority))
                .then_with(|| a.id.cmp(&b.id))
        });

        Ok(CompatTable { rules: self.rules })
    }
}

fn parse_revision_pattern(v: &str) -> Result<RevisionPattern, CompatError> {
    let clean = v
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim();
    if clean == "*" {
        Ok(RevisionPattern::Wildcard)
    } else if clean.contains("..") {
        let parts: Vec<&str> = clean.split("..").collect();
        if parts.len() != 2 {
            return Err(CompatError::InvalidRevisionRange(format!(
                "invalid revision range syntax '{v}'"
            )));
        }
        let min = if parts[0].trim().is_empty() {
            None
        } else {
            Some(parts[0].trim().to_string())
        };
        let max = if parts[1].trim().is_empty() {
            None
        } else {
            Some(parts[1].trim().to_string())
        };
        if let (Some(a), Some(b)) = (&min, &max)
            && compare_revisions(a.as_str(), b.as_str()) == std::cmp::Ordering::Greater {
                return Err(CompatError::InvalidRevisionRange(format!(
                    "min '{a}' is greater than max '{b}'"
                )));
            }
        if min.is_none() && max.is_none() {
            Ok(RevisionPattern::Wildcard)
        } else {
            Ok(RevisionPattern::Range { min, max })
        }
    } else if clean.ends_with('*') {
        Ok(RevisionPattern::Prefix(
            clean.trim_end_matches('*').to_string(),
        ))
    } else {
        Ok(RevisionPattern::Exact(clean.to_string()))
    }
}

fn parse_single_rule_line(line: &str, line_num: usize) -> Result<CompatRule, CompatError> {
    // Format: rule "<id>" host="<host>" provider="<provider>" family="<family>" revision="<rev>" class="<class>" priority=<num> caps="..." cost="..."
    let mut host = None;
    let mut provider = None;
    let mut family = None;
    let mut revision = None;
    let mut class = None;
    let mut priority = 0;
    let mut capabilities = HashMap::new();
    let mut native_tool_cost = None;

    let tokens = tokenize_directive(line);
    if tokens.len() < 2 || tokens[0] != "rule" {
        return Err(CompatError::SyntaxError {
            line: line_num,
            message: "expected rule <id> [key=value...]".into(),
        });
    }

    let id = tokens[1].clone();

    for token in &tokens[2..] {
        if let Some((k, v)) = token.split_once('=') {
            let k = k.trim();
            let v = v.trim().trim_matches('"');
            match k {
                "host" => host = Some(v.to_string()),
                "provider" => provider = Some(v.to_string()),
                "family" => match v.to_lowercase().as_str() {
                    "claude" => family = Some(ModelFamily::Claude),
                    "gpt" => family = Some(ModelFamily::Gpt),
                    "gemini" => family = Some(ModelFamily::Gemini),
                    "deepseek" => family = Some(ModelFamily::DeepSeek),
                    "llama" => family = Some(ModelFamily::Llama),
                    "qwen" => family = Some(ModelFamily::Qwen),
                    "mistral" | "mixtral" => family = Some(ModelFamily::Mistral),
                    _ => {
                        return Err(CompatError::UnknownValue {
                            key: "family".into(),
                            value: v.to_string(),
                        });
                    }
                },
                "revision" => {
                    let rev = parse_revision_pattern(v)?;
                    revision = Some(rev);
                }
                "class" => match v.to_lowercase().as_str() {
                    "flagship" => class = Some(ModelClass::Flagship),
                    "fast" => class = Some(ModelClass::Fast),
                    "reasoning" => class = Some(ModelClass::Reasoning),
                    "coding" => class = Some(ModelClass::Coding),
                    "embedding" => class = Some(ModelClass::Embedding),
                    "vision_only" => class = Some(ModelClass::VisionOnly),
                    _ => {
                        return Err(CompatError::UnknownValue {
                            key: "class".into(),
                            value: v.to_string(),
                        });
                    }
                },
                "priority" => {
                    priority = v.parse().map_err(|_| CompatError::UnknownValue {
                        key: "priority".into(),
                        value: v.to_string(),
                    })?;
                }
                "cost" => match v.to_lowercase().as_str() {
                    "free" => native_tool_cost = Some(NativeToolCost::Free),
                    "costly" => native_tool_cost = Some(NativeToolCost::Costly),
                    "unknown" => native_tool_cost = Some(NativeToolCost::Unknown),
                    _ => {
                        return Err(CompatError::UnknownValue {
                            key: "cost".into(),
                            value: v.to_string(),
                        });
                    }
                },
                _ => {
                    if let Ok(pcap) = k.parse::<ProviderCapability>() {
                        match v.parse::<TriState>() {
                            Ok(cap) => {
                                capabilities.insert(pcap, cap);
                            }
                            Err(_) => {
                                return Err(CompatError::UnknownValue {
                                    key: k.to_string(),
                                    value: v.to_string(),
                                });
                            }
                        }
                    } else {
                        return Err(CompatError::UnknownDirective(k.to_string()));
                    }
                }
            }
        } else {
            return Err(CompatError::UnknownDirective(token.to_string()));
        }
    }

    Ok(CompatRule {
        id,
        host,
        provider,
        family,
        revision,
        class,
        capabilities,
        native_tool_cost,
        priority,
    })
}

fn tokenize_directive(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    for ch in line.chars() {
        if ch == '"' {
            in_quotes = !in_quotes;
            current.push(ch);
        } else if ch.is_whitespace() && !in_quotes {
            if !current.is_empty() {
                tokens.push(current.trim_matches('"').to_string());
                current.clear();
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        tokens.push(current.trim_matches('"').to_string());
    }
    tokens
}

/// Compiled, deterministic indexed compatibility table.
#[derive(Clone, Debug)]
pub struct CompatTable {
    rules: Vec<CompatRule>,
}

impl Default for CompatTable {
    fn default() -> Self {
        Self::standard()
    }
}

impl CompatTable {
    pub fn new(mut rules: Vec<CompatRule>) -> Self {
        // Sort identically to CompatCompiler::compile(): highest specificity
        // first, then highest priority, then alphabetical id for determinism.
        rules.sort_by(|a, b| {
            b.specificity_level()
                .cmp(&a.specificity_level())
                .then_with(|| b.priority.cmp(&a.priority))
                .then_with(|| a.id.cmp(&b.id))
        });
        Self { rules }
    }

    /// Creates standard default compatibility table for frontier providers.
    ///
    /// The rules below are hardcoded and validated; `expect` (not silent
    /// fallback) is deliberate so a programmer error editing them fails fast
    /// at startup instead of shipping a wrong capability table.
    pub fn standard() -> Self {
        let mut compiler = CompatCompiler::new();

        // 1. Anthropic Direct
        let mut anthropic_caps = HashMap::new();
        anthropic_caps.insert(ProviderCapability::Streaming, TriState::Supported);
        anthropic_caps.insert(ProviderCapability::NativeToolChoice, TriState::Supported);
        anthropic_caps.insert(ProviderCapability::TokenCount, TriState::Supported);
        anthropic_caps.insert(ProviderCapability::DeveloperRole, TriState::Unsupported);
        anthropic_caps.insert(
            ProviderCapability::MidSessionSystemPrompts,
            TriState::Unsupported,
        );
        anthropic_caps.insert(ProviderCapability::RemoteCompaction, TriState::Supported);
        anthropic_caps.insert(ProviderCapability::UsageQuery, TriState::Supported);

        compiler
            .add_rule(CompatRule {
                id: "anthropic-direct-default".into(),
                host: Some("api.anthropic.com".into()),
                provider: Some("anthropic".into()),
                family: Some(ModelFamily::Claude),
                revision: None,
                class: None,
                capabilities: anthropic_caps,
                native_tool_cost: Some(NativeToolCost::Free),
                priority: 10,
            })
            .expect("hardcoded standard rule must be valid");

        // 2. OpenAI Direct
        let mut openai_caps = HashMap::new();
        openai_caps.insert(ProviderCapability::Streaming, TriState::Supported);
        openai_caps.insert(ProviderCapability::NativeToolChoice, TriState::Supported);
        openai_caps.insert(ProviderCapability::DeveloperRole, TriState::Supported);
        openai_caps.insert(ProviderCapability::ConstrainedSampling, TriState::Supported);
        openai_caps.insert(ProviderCapability::UsageQuery, TriState::Supported);

        compiler
            .add_rule(CompatRule {
                id: "openai-direct-default".into(),
                host: Some("api.openai.com".into()),
                provider: Some("openai".into()),
                family: Some(ModelFamily::Gpt),
                revision: None,
                class: None,
                capabilities: openai_caps,
                native_tool_cost: Some(NativeToolCost::Costly), // OpenAI forced tool choice can break prompt cache
                priority: 10,
            })
            .expect("hardcoded standard rule must be valid");

        // 3. Class defaults (Level 1)
        let mut reasoning_caps = HashMap::new();
        reasoning_caps.insert(
            ProviderCapability::ConstrainedSampling,
            TriState::Unsupported,
        );

        compiler
            .add_rule(CompatRule {
                id: "class-reasoning-default".into(),
                host: None,
                provider: None,
                family: None,
                revision: None,
                class: Some(ModelClass::Reasoning),
                capabilities: reasoning_caps,
                native_tool_cost: Some(NativeToolCost::Unknown),
                priority: 1,
            })
            .expect("hardcoded standard rule must be valid");

        compiler
            .compile()
            .expect("standard rules must compile cleanly")
    }

    /// Resolves capabilities for a model identifier following strict precedence ladder:
    /// 1. Exact host + exact model revision (specificity 5)
    /// 2. Host + family (specificity 4)
    /// 3. Provider + revision (specificity 3)
    /// 4. Provider + family (specificity 2)
    /// 5. Class defaults (specificity 1)
    /// 6. Unknown fallback (specificity 0)
    pub fn resolve(&self, ident: &ModelIdentifier) -> CapabilityProfile {
        let mut profile = CapabilityProfile::new();

        // Collect all matching rules in deterministic precedence order (highest specificity first)
        let matched_rules: Vec<&CompatRule> =
            self.rules.iter().filter(|r| r.matches(ident)).collect();

        // For each capability, the highest specificity rule defining it sets it (honoring explicit Unknown)
        for cap in ProviderCapability::ALL {
            let mut resolved = TriState::Unknown;
            for rule in &matched_rules {
                if let Some(state) = rule.capabilities.get(cap) {
                    resolved = *state;
                    break;
                }
            }
            profile.set(*cap, resolved);
        }

        // Resolve native tool cost (first rule defining it sets it)
        for rule in &matched_rules {
            if let Some(cost) = rule.native_tool_cost {
                profile.native_tool_cost = cost;
                break;
            }
        }

        profile
    }

    pub fn resolve_capability(&self, ident: &ModelIdentifier, cap: ProviderCapability) -> TriState {
        for rule in &self.rules {
            if rule.matches(ident)
                && let Some(state) = rule.capabilities.get(&cap) {
                    return *state;
                }
        }
        TriState::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reject_unknown_directive() {
        let mut compiler = CompatCompiler::new();
        let kdl = r#"rule "test" host="api.example.com" unknown_directive="true""#;
        let err = compiler.parse_kdl_text(kdl).unwrap_err();
        assert_eq!(
            err,
            CompatError::UnknownDirective("unknown_directive".into())
        );
    }

    #[test]
    fn test_reject_unknown_value_for_known_key() {
        let mut compiler = CompatCompiler::new();
        let kdl = r#"rule "test" family="non_existent_alien_family""#;
        let err = compiler.parse_kdl_text(kdl).unwrap_err();
        assert!(matches!(err, CompatError::UnknownValue { key, .. } if key == "family"));

        let mut compiler2 = CompatCompiler::new();
        let kdl2 = r#"rule "test" class="quantum_computer""#;
        let err2 = compiler2.parse_kdl_text(kdl2).unwrap_err();
        assert!(matches!(err2, CompatError::UnknownValue { key, .. } if key == "class"));

        let mut compiler3 = CompatCompiler::new();
        let kdl3 = r#"rule "test" streaming="maybe""#;
        let err3 = compiler3.parse_kdl_text(kdl3).unwrap_err();
        assert!(matches!(err3, CompatError::UnknownValue { key, .. } if key == "streaming"));
    }

    #[test]
    fn test_revision_range_validation() {
        let mut compiler = CompatCompiler::new();
        let kdl = r#"rule "test" revision="2.0..1.0""#;
        let err = compiler.parse_kdl_text(kdl).unwrap_err();
        assert!(matches!(err, CompatError::InvalidRevisionRange(_)));

        let mut compiler2 = CompatCompiler::new();
        let kdl2 = r#"rule "test" revision="1.0..2.0""#;
        assert!(compiler2.parse_kdl_text(kdl2).is_ok());
    }

    #[test]
    fn test_revision_compare_is_numeric_aware() {
        use std::cmp::Ordering;
        // Lexicographic order would say "10.0" < "9.0"; numeric wins.
        assert_eq!(compare_revisions("10.0", "9.0"), Ordering::Greater);
        assert_eq!(compare_revisions("9.0", "10.0"), Ordering::Less);
        assert_eq!(compare_revisions("1.2", "1.2.0"), Ordering::Equal);
        assert_eq!(compare_revisions("1.2", "1.2.0.1"), Ordering::Less);
        assert_eq!(compare_revisions("1.2.0.1", "1.2"), Ordering::Greater);
        assert_eq!(compare_revisions("2024-01", "2024-02"), Ordering::Less);

        // Ranges honor numeric order too.
        let range = RevisionPattern::Range {
            min: Some("9.0".into()),
            max: Some("11.0".into()),
        };
        assert!(range.matches(Some("10.0")));
        assert!(!range.matches(Some("8.9")));
        assert!(!range.matches(Some("12.0")));
    }

    #[test]
    fn test_conflicting_equal_specificity() {
        let mut compiler = CompatCompiler::new();
        let kdl = r#"
rule "rule-a" host="api.test.com" family="claude" priority=10 streaming="supported"
rule "rule-b" host="api.test.com" family="claude" priority=10 streaming="unsupported"
"#;
        let err = compiler.parse_kdl_text(kdl).unwrap_err();
        assert!(matches!(
            err,
            CompatError::ConflictingEqualSpecificity { .. }
        ));
    }

    #[test]
    fn test_unreachable_shadowed_rule() {
        let mut compiler = CompatCompiler::new();
        let kdl = r#"
rule "rule-broad" host="api.test.com" family="claude" priority=10 streaming="supported"
rule "rule-shadowed" host="api.test.com" family="claude" priority=5 streaming="unsupported"
"#;
        let err = compiler.parse_kdl_text(kdl).unwrap_err();
        assert!(
            matches!(err, CompatError::UnreachableRule { id, shadowed_by } if id == "rule-shadowed" && shadowed_by == "rule-broad")
        );
    }

    #[test]
    fn test_explicit_unknown_capability_resolution() {
        let mut compiler = CompatCompiler::new();
        let kdl = r#"
rule "specific-host" host="api.test.com" family="claude" priority=10 streaming="unknown"
rule "generic-class" class="fast" priority=1 streaming="supported"
"#;
        compiler.parse_kdl_text(kdl).unwrap();
        let table = compiler.compile().unwrap();
        let ident = ModelIdentifier::new(
            crate::model_taxonomy::ModelTaxonomy::from_model_id("claude-3-haiku"),
            ModelClass::Fast,
            crate::model_taxonomy::ProviderRoute::Custom {
                provider_name: "anthropic".into(),
                host: "api.test.com".into(),
            },
        );
        let profile = table.resolve(&ident);
        // Higher precedence rule specified streaming="unknown", so it must be TriState::Unknown, not overridden by fast class
        assert_eq!(
            profile.get(ProviderCapability::Streaming),
            TriState::Unknown
        );
    }
}
