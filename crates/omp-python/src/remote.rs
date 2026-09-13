use crate::error::PythonExtensionError;
use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_PAYLOAD_BYTES: usize = 1024 * 1024; // 1 MB
pub const DEFAULT_REMOTE_TIMEOUT_MS: u64 = 30_000;
/// Upper bound mirroring `LimitPolicy::default().max_wall_time` (120 s) so a
/// remote job can never outlive the host execution budget.
pub const MAX_REMOTE_TIMEOUT_MS: u64 = 120_000;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RemoteFunctionSpec {
    pub name: String,
    pub source_code: String,
    pub source_hash: String,
    pub declared_capabilities: Vec<String>,
    pub imports: Vec<String>,
    pub max_payload_bytes: usize,
    pub timeout_ms: u64,
}

impl RemoteFunctionSpec {
    pub fn new(name: impl Into<String>, source_code: impl Into<String>) -> Self {
        let code = source_code.into();
        let source_hash = sha256_hex(code.as_bytes());
        Self {
            name: name.into(),
            source_code: code,
            source_hash,
            declared_capabilities: Vec::new(),
            imports: Vec::new(),
            max_payload_bytes: DEFAULT_MAX_PAYLOAD_BYTES,
            timeout_ms: DEFAULT_REMOTE_TIMEOUT_MS,
        }
    }

    pub fn with_capabilities(mut self, caps: Vec<String>) -> Self {
        self.declared_capabilities = caps;
        self
    }

    pub fn with_imports(mut self, imports: Vec<String>) -> Self {
        self.imports = imports;
        self
    }
}

pub struct RemoteValidator;

impl RemoteValidator {
    pub const PROHIBITED_MODULES: &'static [&'static str] = &[
        "os",
        "subprocess",
        "sys",
        "socket",
        "ctypes",
        "pty",
        "posix",
        "nt",
        "shutil",
        "builtins",
        "importlib",
        "signal",
        "multiprocessing",
        "threading",
    ];

    pub const FORBIDDEN_CONSTRUCTS: &'static [&'static str] = &[
        "eval(",
        "eval (",
        "exec(",
        "exec (",
        "__import__",
        "compile(",
        "compile (",
        "globals()",
        "globals ()",
        "locals()",
        "locals ()",
        "__subclasses__",
        "__globals__",
        "__getattribute__",
        "getattr(",
        "getattr (",
        "setattr(",
        "setattr (",
        "delattr(",
        "delattr (",
        "importlib.",
        "importlib(",
        "importlib (",
    ];

    /// Validates the remote function specification prior to execution.
    ///
    /// The host is the single enforcement point: imports are derived from
    /// `source_code` itself (the client-supplied `imports` list is treated as
    /// advisory and merged in), and `source_hash` must match
    /// `sha256(source_code)` unless it is empty (legacy callers).
    pub fn validate(spec: &RemoteFunctionSpec) -> Result<(), PythonExtensionError> {
        // 0. Verify source integrity when the caller supplies a hash.
        if !spec.source_hash.trim().is_empty() {
            let actual = sha256_hex(spec.source_code.as_bytes());
            if actual != spec.source_hash {
                return Err(PythonExtensionError::ValidationFailed(
                    "remote function source_hash does not match sha256(source_code)".into(),
                ));
            }
        }

        // 0b. Enforce host-side budgets (client values are untrusted).
        if spec.max_payload_bytes == 0 || spec.max_payload_bytes > DEFAULT_MAX_PAYLOAD_BYTES {
            return Err(PythonExtensionError::OversizedPayload {
                size: spec.max_payload_bytes,
                limit: DEFAULT_MAX_PAYLOAD_BYTES,
            });
        }
        if spec.timeout_ms == 0 || spec.timeout_ms > MAX_REMOTE_TIMEOUT_MS {
            return Err(PythonExtensionError::Timeout {
                timeout_ms: spec.timeout_ms,
                context: format!(
                    "remote timeout must be within 1..={MAX_REMOTE_TIMEOUT_MS} ms"
                ),
            });
        }

        // 1. Check for empty or oversized source
        if spec.source_code.trim().is_empty() {
            return Err(PythonExtensionError::ValidationFailed(
                "remote function source code cannot be empty".into(),
            ));
        }

        if spec.source_code.len() > spec.max_payload_bytes {
            return Err(PythonExtensionError::OversizedPayload {
                size: spec.source_code.len(),
                limit: spec.max_payload_bytes,
            });
        }

        // 2. Validate imports against prohibited list. Imports are extracted
        // from the source itself so a caller cannot bypass the check by
        // sending `imports: []` alongside `import os` in the source body.
        let mut all_imports = extract_source_imports(&spec.source_code);
        all_imports.extend(spec.imports.iter().cloned());
        for import_name in &all_imports {
            let base_module = import_name.split('.').next().unwrap_or(import_name);
            if Self::PROHIBITED_MODULES.contains(&base_module) {
                // Check if an explicit capability overrides the prohibition
                let capability_needed = format!("import_{base_module}");
                if !spec
                    .declared_capabilities
                    .iter()
                    .any(|c| c == &capability_needed || c == "system_all")
                {
                    return Err(PythonExtensionError::ProhibitedImport {
                        module: import_name.clone(),
                    });
                }
            }
        }

        // 3. Scan for forbidden dynamic code execution constructs
        for construct in Self::FORBIDDEN_CONSTRUCTS {
            if spec.source_code.contains(construct) {
                return Err(PythonExtensionError::DynamicCodeForbidden(format!(
                    "forbidden construct '{construct}' detected in function '{}'",
                    spec.name
                )));
            }
        }

        // 4. Check for undeclared capability requirements
        if spec.source_code.contains("open(") {
            let has_fs = spec
                .declared_capabilities
                .iter()
                .any(|c| c == "fs_read" || c == "fs_write" || c == "fs_all");
            if !has_fs {
                return Err(PythonExtensionError::UndeclaredCapability {
                    capability: "fs_read/fs_write".into(),
                });
            }
        }

        if spec.source_code.contains("urllib")
            || spec.source_code.contains("requests")
            || mentions_network_identifier(&spec.source_code)
        {
            let has_net = spec
                .declared_capabilities
                .iter()
                .any(|c| c == "net" || c == "net_all");
            if !has_net {
                return Err(PythonExtensionError::UndeclaredCapability {
                    capability: "net".into(),
                });
            }
        }

        Ok(())
    }

    /// Validates an input payload size against the configured limit.
    pub fn validate_payload(
        payload_bytes: usize,
        limit: usize,
    ) -> Result<(), PythonExtensionError> {
        if payload_bytes > limit {
            Err(PythonExtensionError::OversizedPayload {
                size: payload_bytes,
                limit,
            })
        } else {
            Ok(())
        }
    }
}

/// sha256 hex digest used for `source_hash` verification.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

/// Best-effort extraction of top-level module names from Python source.
///
/// This is a line-oriented tokenizer, not a full parser: it strips `#`
/// comments, then recognizes `import a, b.c` and `from a.b import ...`
/// statements (including leading whitespace). It exists so the host does not
/// trust the client-supplied `imports` list; the Python SDK's own
/// `ASTValidator` remains the stricter client-side check.
fn extract_source_imports(source: &str) -> std::collections::BTreeSet<String> {
    let mut imports = std::collections::BTreeSet::new();
    for line in source.lines() {
        // Strip trailing comments; a `#` inside a string literal may cause an
        // over-approximation, which is safe (fail-closed) here.
        let code = match line.find('#') {
            Some(index) => &line[..index],
            None => line,
        };
        let trimmed = code.trim();
        if let Some(rest) = trimmed.strip_prefix("import ") {
            for part in rest.split(',') {
                let name = part.split_whitespace().next().unwrap_or("");
                let name = name.trim_matches(|c| c == '(' || c == ')' || c == ';');
                if !name.is_empty() {
                    imports.insert(name.to_string());
                }
            }
        } else if let Some(rest) = trimmed.strip_prefix("from ")
            && let Some((module, _)) = rest.split_once(" import ")
        {
            let module = module.trim();
            if !module.is_empty() && !module.starts_with('.') {
                imports.insert(module.to_string());
            }
        }
    }
    imports
}

/// Token-aware network-identifier check: matches `http`, `https`, `socket`,
/// `urlopen`, and `request(` as whole identifiers, so a variable named
/// `http_server` no longer triggers a false positive while `import socket`
/// style usage is still gated.
fn mentions_network_identifier(source: &str) -> bool {
    const MARKERS: &[&str] = &[
        "http", "https", "socket", "urlopen", "urllib3", "httpx", "aiohttp",
    ];
    source
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|token| MARKERS.contains(&token))
}
