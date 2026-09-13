use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::definition::{
    HostGateway, HostRequest, ToolCall,
    ToolDefinition, ToolDiagnostic, ToolError, ToolExecutionResult, ToolExecutor, ToolLimits,
    ToolOutput,
};

/// Selector types supported by the Read primitive.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LineSelector {
    /// Single line `:N`
    Single(usize),
    /// From line `:N-`
    FromLine(usize),
    /// Inclusive range `:N-M`
    Inclusive(usize, usize),
    /// Offset count `:N+Count`
    Count(usize, usize),
    /// Disjoint ranges `:5-16,960-973`
    Disjoint(Vec<(usize, usize)>),
    /// Last N lines `:-N`
    Tail(usize),
}

impl LineSelector {
    /// Applies the selector to a slice of lines (1-indexed).
    pub fn select_lines<'a>(&self, lines: &'a [&'a str]) -> Vec<(usize, &'a str)> {
        let (res, _) = self.select_lines_bounded(lines, usize::MAX, usize::MAX);
        res
    }

    /// Applies the selector to a slice of lines (1-indexed) with strict count and byte bounds.
    /// Returns the selected (1-indexed line_number, line_content) pairs and whether output was truncated.
    pub fn select_lines_bounded<'a>(
        &self,
        lines: &'a [&'a str],
        max_lines: usize,
        max_bytes: usize,
    ) -> (Vec<(usize, &'a str)>, bool) {
        let total = lines.len();
        let mut result = Vec::new();
        let mut total_bytes = 0;
        let mut truncated = false;

        let mut push_line = |num: usize, text: &'a str| -> bool {
            if result.len() >= max_lines {
                truncated = true;
                return false;
            }
            let bytes = text.len() + 1;
            if total_bytes + bytes > max_bytes {
                truncated = true;
                return false;
            }
            total_bytes += bytes;
            result.push((num, text));
            true
        };

        match self {
            LineSelector::Single(n) => {
                if *n >= 1 && *n <= total {
                    push_line(*n, lines[*n - 1]);
                }
            }
            LineSelector::FromLine(start) => {
                let s = (*start).max(1);
                for i in s..=total {
                    if !push_line(i, lines[i - 1]) {
                        break;
                    }
                }
            }
            LineSelector::Inclusive(start, end) => {
                let s = (*start).max(1);
                let e = (*end).min(total);
                if s <= e {
                    for i in s..=e {
                        if !push_line(i, lines[i - 1]) {
                            break;
                        }
                    }
                }
            }
            LineSelector::Count(start, count) => {
                let s = (*start).max(1);
                let e = (s + count.saturating_sub(1)).min(total);
                for i in s..=e {
                    if !push_line(i, lines[i - 1]) {
                        break;
                    }
                }
            }
            LineSelector::Disjoint(ranges) => {
                for &(s, e) in ranges {
                    let start = s.max(1);
                    let end = e.min(total);
                    if start <= end {
                        for i in start..=end {
                            if !push_line(i, lines[i - 1]) {
                                return (result, truncated);
                            }
                        }
                    }
                }
            }
            LineSelector::Tail(count) => {
                let s = total.saturating_sub(*count) + 1;
                for i in s..=total {
                    if !push_line(i, lines[i - 1]) {
                        break;
                    }
                }
            }
        }

        (result, truncated)
    }
}

/// Parsed selector on a resource path.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReadSelector {
    pub raw: bool,
    pub lines: Option<LineSelector>,
    pub query: Option<String>,
    pub subpath: Option<String>,
    pub is_img: bool,
    pub conflicts_only: bool,
}

impl ReadSelector {
    /// Parses an inline selector string like `:50-200`, `:raw:50-200`, `:50-200:raw`, `:raw`, `?q=...`.
    pub fn parse(spec: &str) -> Self {
        let mut selector = Self::default();
        let (path_part, query_part) = match spec.split_once('?') {
            Some((p, q)) => (p, Some(q)),
            None => (spec, None),
        };

        if let Some(q) = query_part {
            if let Some(rest) = q.strip_prefix("q=") {
                selector.query = Some(rest.to_string());
            } else {
                selector.query = Some(q.to_string());
            }
        }

        let parts: Vec<&str> = path_part.split(':').filter(|s| !s.is_empty()).collect();
        for part in parts {
            if part == "raw" {
                selector.raw = true;
            } else if part == "img" {
                selector.is_img = true;
            } else if part == "conflicts" {
                selector.conflicts_only = true;
            } else if selector.subpath.is_some() {
                selector.subpath.as_mut().unwrap().push(':');
                selector.subpath.as_mut().unwrap().push_str(part);
            } else if let Some(lines) = Self::parse_line_spec(part) {
                selector.lines = Some(lines);
            } else {
                selector.subpath = Some(part.to_string());
            }
        }

        selector
    }

    pub fn parse_line_spec(spec: &str) -> Option<LineSelector> {
        if spec.contains(',') {
            let mut ranges = Vec::new();
            for item in spec.split(',') {
                let trimmed = item.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Some((s, e)) = trimmed.split_once('-') {
                    if let (Ok(start), Ok(end)) =
                        (s.trim().parse::<usize>(), e.trim().parse::<usize>())
                    {
                        ranges.push((start, end));
                    }
                } else if let Ok(line) = trimmed.parse::<usize>() {
                    ranges.push((line, line));
                }
            }
            if !ranges.is_empty() {
                return Some(LineSelector::Disjoint(ranges));
            }
        }

        if let Some(rest) = spec.strip_prefix('-')
            && let Ok(count) = rest.parse::<usize>() {
                return Some(LineSelector::Tail(count));
            }

        if let Some((s, e)) = spec.split_once('-') {
            if e.is_empty() {
                if let Ok(start) = s.parse::<usize>() {
                    return Some(LineSelector::FromLine(start));
                }
            } else if let (Ok(start), Ok(end)) = (s.parse::<usize>(), e.parse::<usize>()) {
                return Some(LineSelector::Inclusive(start, end));
            }
        }

        if let Some((s, c)) = spec.split_once('+')
            && let (Ok(start), Ok(count)) = (s.parse::<usize>(), c.parse::<usize>()) {
                return Some(LineSelector::Count(start, count));
            }

        if let Ok(line) = spec.parse::<usize>() {
            return Some(LineSelector::FromLine(line));
        }

        None
    }
}

/// Resource schemes recognized by the Read primitive.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ResourceScheme {
    File,
    Artifact,
    Agent,
    History,
    Skill,
    Rule,
    Issue,
    Pr,
    Ssh,
    Http,
    Https,
    Sqlite,
    Archive,
    Document,
    Custom(String),
}

impl ResourceScheme {
    pub fn from_uri(uri: &str) -> (Self, &str) {
        if let Some(rest) = uri.strip_prefix("artifact://") {
            (ResourceScheme::Artifact, rest)
        } else if let Some(rest) = uri.strip_prefix("agent://") {
            (ResourceScheme::Agent, rest)
        } else if let Some(rest) = uri.strip_prefix("history://") {
            (ResourceScheme::History, rest)
        } else if let Some(rest) = uri.strip_prefix("skill://") {
            (ResourceScheme::Skill, rest)
        } else if let Some(rest) = uri.strip_prefix("rule://") {
            (ResourceScheme::Rule, rest)
        } else if let Some(rest) = uri.strip_prefix("issue://") {
            (ResourceScheme::Issue, rest)
        } else if let Some(rest) = uri.strip_prefix("pr://") {
            (ResourceScheme::Pr, rest)
        } else if let Some(rest) = uri.strip_prefix("ssh://") {
            (ResourceScheme::Ssh, rest)
        } else if let Some(rest) = uri.strip_prefix("http://") {
            (ResourceScheme::Http, rest)
        } else if let Some(rest) = uri.strip_prefix("https://") {
            (ResourceScheme::Https, rest)
        } else if let Some(rest) = uri.strip_prefix("file://") {
            (ResourceScheme::File, rest)
        } else {
            let lower = uri.to_ascii_lowercase();
            if lower.ends_with(".sqlite") || lower.ends_with(".sqlite3") || lower.ends_with(".db") {
                (ResourceScheme::Sqlite, uri)
            } else if lower.ends_with(".zip")
                || lower.ends_with(".tar")
                || lower.ends_with(".tar.gz")
                || lower.ends_with(".tgz")
                || lower.ends_with(".jar")
                || lower.ends_with(".whl")
                || lower.ends_with(".asar")
            {
                (ResourceScheme::Archive, uri)
            } else if lower.ends_with(".pdf")
                || lower.ends_with(".docx")
                || lower.ends_with(".pptx")
                || lower.ends_with(".xlsx")
                || lower.ends_with(".epub")
            {
                (ResourceScheme::Document, uri)
            } else {
                if let Some((scheme, target)) = uri.split_once("://") {
                    return (ResourceScheme::Custom(scheme.into()), target);
                }
                (ResourceScheme::File, uri)
            }
        }
    }
}

/// Handler for a resource projection.
pub trait ProjectionHandler: Send + Sync {
    fn handle(
        &self,
        path: &str,
        selector: &ReadSelector,
        gateway: &dyn HostGateway,
    ) -> Result<ToolOutput, ToolError>;
}

/// Registry of URI projections for custom URL schemes and extensions.
pub struct ProjectionRegistry {
    handlers: RwLock<BTreeMap<String, Arc<dyn ProjectionHandler>>>,
}

impl Default for ProjectionRegistry {
    fn default() -> Self {
        Self {
            handlers: RwLock::new(BTreeMap::new()),
        }
    }
}

impl ProjectionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, scheme: impl Into<String>, handler: Arc<dyn ProjectionHandler>) {
        self.handlers.write().insert(scheme.into(), handler);
    }

    pub fn get(&self, scheme: &str) -> Option<Arc<dyn ProjectionHandler>> {
        self.handlers.read().get(scheme).cloned()
    }
}

/// Executor for the permanent `Read` tool.
pub struct ReadExecutor {
    pub projections: Arc<ProjectionRegistry>,
}

impl Default for ReadExecutor {
    fn default() -> Self {
        Self {
            projections: Arc::new(ProjectionRegistry::new()),
        }
    }
}

impl ToolExecutor for ReadExecutor {
    fn execute(
        &self,
        call: &ToolCall,
        gateway: &dyn HostGateway,
    ) -> Result<ToolExecutionResult, ToolError> {
        let path_val = call
            .input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::Validation {
                message: "missing required 'path' parameter".into(),
                details: Some(call.input.clone()),
            })?;

        // Split base path and selector (e.g., "src/lib.rs:50-200" or "db.sqlite:users")
        let (base_path, selector_str) = extract_path_and_selector(path_val);
        let selector = ReadSelector::parse(&selector_str);
        let (scheme, clean_target) = ResourceScheme::from_uri(base_path);

        // Check if custom projection handler registered
        if let ResourceScheme::Custom(name) = &scheme
            && let Some(handler) = self.projections.get(name) {
                let output = handler.handle(clean_target, &selector, gateway)?;
                return Ok(ToolExecutionResult {
                    output,
                    diagnostics: Vec::new(),
                    usage: None,
                    artifacts: Vec::new(),
                });
            }

        // Host request for materialization
        let req = HostRequest::ReadResource {
            path: base_path.to_string(),
            selector: if selector_str.is_empty() {
                None
            } else {
                Some(selector_str)
            },
            raw: selector.raw,
        };

        let resp = gateway.request(req)?;
        let mut result = ToolExecutionResult::from_host(resp);

        // Strict bounded output enforcement
        const MAX_OUTPUT_BYTES: usize = 100_000;
        if result.output.content.len() > MAX_OUTPUT_BYTES {
            let mut cut_idx = MAX_OUTPUT_BYTES;
            while cut_idx > 0 && !result.output.content.is_char_boundary(cut_idx) {
                cut_idx -= 1;
            }
            result.output.content.truncate(cut_idx);
            result.output.truncated = true;
            result.diagnostics.push(ToolDiagnostic::warning(format!(
                "Output exceeded limit of {} bytes and was truncated.",
                MAX_OUTPUT_BYTES
            )));
        } else if result.output.truncated {
            result.diagnostics.push(ToolDiagnostic::warning(
                "Output exceeded limit and was truncated.",
            ));
        }

        Ok(result)
    }
}

fn is_selector_token(token: &str) -> bool {
    let t = token.trim();
    if t.is_empty() {
        return false;
    }
    if t == "raw" || t == "img" || t == "conflicts" {
        return true;
    }
    ReadSelector::parse_line_spec(t).is_some()
}

/// Splits a raw path string into the target path and any trailing selector.
/// Correctly handles combinations like `:raw:50-200`, `:50-200:raw`, Windows drive letters,
/// schemes (e.g. `artifact://`), and queries (`?q=...`).
pub fn extract_path_and_selector(raw: &str) -> (&str, String) {
    let (path_part, query_opt) = match raw.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (raw, None),
    };
    if path_part.starts_with("http://") || path_part.starts_with("https://") {
        if let Some(base) = raw.strip_suffix(":raw") {
            return (base, "raw".into());
        }
        return (raw, String::new());
    }

    // Resource members are selectors even when they are not line-range tokens.
    let lower = path_part.to_ascii_lowercase();
    for extension in [
        ".sqlite3:",
        ".sqlite:",
        ".db:",
        ".tar.gz:",
        ".tgz:",
        ".zip:",
        ".tar:",
        ".jar:",
        ".whl:",
        ".asar:",
    ] {
        if let Some(index) = lower.find(extension) {
            let end = index + extension.len() - 1;
            let mut selector = path_part[end + 1..].to_string();
            if let Some(query) = query_opt {
                selector.push('?');
                selector.push_str(query);
            }
            return (&path_part[..end], selector);
        }
    }
    let mut remaining = path_part;
    let mut selector_tokens: Vec<&str> = Vec::new();

    while let Some(colon_idx) = remaining.rfind(':') {
        // Windows drive letter: C:\ or C:/ at index 1
        if colon_idx == 1
            && remaining
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic())
        {
            break;
        }
        // Scheme prefix: scheme:// where colon is followed by //
        if remaining[colon_idx..].starts_with("://") {
            break;
        }

        let token = &remaining[colon_idx + 1..];
        if is_selector_token(token) {
            selector_tokens.push(token);
            remaining = &remaining[..colon_idx];
        } else {
            break;
        }
    }

    selector_tokens.reverse();
    let mut selector_str = selector_tokens.join(":");

    if let Some(q) = query_opt {
        if selector_str.is_empty() {
            selector_str = format!("?{q}");
        } else {
            selector_str = format!("{selector_str}:?{q}");
        }
    }

    (remaining, selector_str)
}

/// Creates the permanent `Read` tool definition.
pub fn read_tool_definition(executor: Arc<ReadExecutor>) -> ToolDefinition {
    ToolDefinition::new(
        "Read",
        "1.0.0",
        json!({
            "type": "string",
            "description": "Intent describing the resource read operation"
        }),
        json!({
            "type": "object",
            "required": ["path", "i"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Local path, internal URI (e.g. skill://, artifact://), or URL with optional inline selectors (:50, :50-200, :raw)"
                },
                "i": {
                    "type": "string",
                    "description": "Present-participle intent describing what is being read"
                }
            }
        }),
        vec!["read_fs".into(), "read_resource".into()],
        ToolLimits {
            max_output_bytes: 100_000,
            max_runtime_ms: 30_000,
            max_artifacts: 10,
            max_concurrent_jobs: 4,
        },
        executor,
        "read",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::HostResponse;
    use crate::definition::DirectHostGateway;

    #[test]
    fn test_extract_path_and_selector() {
        let (p, s) = extract_path_and_selector("src/lib.rs:50-200");
        assert_eq!(p, "src/lib.rs");
        assert_eq!(s, "50-200");

        let (p, s) = extract_path_and_selector("src/lib.rs:raw:50-200");
        assert_eq!(p, "src/lib.rs");
        assert_eq!(s, "raw:50-200");

        let (p, s) = extract_path_and_selector("src/lib.rs:50-200:raw");
        assert_eq!(p, "src/lib.rs");
        assert_eq!(s, "50-200:raw");

        let (p, s) = extract_path_and_selector("C:\\project\\file.rs:raw:10-20");
        assert_eq!(p, "C:\\project\\file.rs");
        assert_eq!(s, "raw:10-20");

        let (p, s) = extract_path_and_selector("artifact://art_1:raw");
        assert_eq!(p, "artifact://art_1");
        assert_eq!(s, "raw");

        let (p, s) = extract_path_and_selector("image.png?q=describe");
        assert_eq!(p, "image.png");
        assert_eq!(ReadSelector::parse(&s).query.as_deref(), Some("describe"));
    }

    #[test]
    fn test_disjoint_range_with_single_lines() {
        let sel = ReadSelector::parse_line_spec("5-10,42,90-95").unwrap();
        assert_eq!(
            sel,
            LineSelector::Disjoint(vec![(5, 10), (42, 42), (90, 95)])
        );
    }

    #[test]
    fn test_line_selector_bounded() {
        let lines = vec!["line 1", "line 2", "line 3", "line 4", "line 5"];
        let sel = LineSelector::Inclusive(1, 5);

        // Bound to 2 lines
        let (selected, truncated) = sel.select_lines_bounded(&lines, 2, 1000);
        assert_eq!(selected.len(), 2);
        assert!(truncated);

        // Bound to 15 bytes
        let (selected_bytes, truncated_bytes) = sel.select_lines_bounded(&lines, 10, 15);
        assert!(selected_bytes.len() < 5);
        assert!(truncated_bytes);
    }

    #[test]
    fn test_read_executor_strict_output_bounding() {
        let executor = ReadExecutor::default();
        let gateway = DirectHostGateway::new();

        // Push oversized response (>100_000 bytes)
        let huge_content = "X".repeat(150_000);
        gateway.push_response(HostResponse::success(huge_content));

        let call = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "Read",
            "1.0.0",
            json!({ "path": "big.txt", "i": "Reading large file" }),
        );
        let result = executor.execute(&call, &gateway).unwrap();
        assert!(result.output.content.len() <= 100_000);
        assert!(result.output.truncated);
        assert!(
            result
                .diagnostics
                .iter()
                .any(|d| d.message.contains("truncated"))
        );
    }
}
