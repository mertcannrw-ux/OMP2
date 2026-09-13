use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::definition::{
    HostGateway, HostRequest, ToolCall, ToolDefinition,
    ToolDiagnostic, ToolError, ToolExecutionResult, ToolExecutor, ToolLimits,
};

/// Sensitive operation types that require capability approval.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SensitiveOperation {
    NetworkAccess(String),
    GitPush,
    ProcessControl(String),
    WriteOutsideScope(String),
    DestructiveFs(String),
}

/// Analyzer that inspects bash commands for sensitive operations.
///
/// ADVISORY ONLY: this is substring matching, not a security boundary. It is
/// trivially bypassed (`cu$RL`, `/usr/bin/curl`, `python -c`, `bash -c`,
/// `rm -rf /tmp/../`, env indirection). Real enforcement happens in the
/// sandbox layer (`SandboxCapability`, `sandbox_network=false` by default,
/// workspace-scoped writes). Detections here trigger approval prompts; a
/// clean scan MUST NOT be treated as proof the command is safe.
pub struct CommandPolicyAnalyzer;

impl CommandPolicyAnalyzer {
    pub fn analyze(command: &str) -> Vec<SensitiveOperation> {
        let mut ops = Vec::new();
        let trimmed = command.trim();

        // Check git push
        if trimmed.starts_with("git push") || trimmed.contains(" git push") {
            ops.push(SensitiveOperation::GitPush);
        }

        // Check network access (advisory substring signals; see struct docs).
        let net_bins = [
            "curl",
            "wget",
            "nc",
            "netcat",
            "ssh",
            "scp",
            "sftp",
            "telnet",
            "git fetch",
            "git pull",
            "git clone",
        ];
        // Lowercased copy for case-insensitive matching.
        let lowered = trimmed.to_ascii_lowercase();
        for bin in &net_bins {
            if lowered.starts_with(bin)
                || lowered.contains(&format!(" {bin} "))
                || lowered.contains(&format!("| {bin}"))
                || lowered.contains(&format!("/{bin}"))
            {
                ops.push(SensitiveOperation::NetworkAccess(bin.to_string()));
            }
        }
        // Shell-indirection signals: `python -c`, `perl -e`, `bash -c`, and
        // `$VAR` expansion are classic analyzer bypasses — flag for review.
        for marker in ["python -c", "python3 -c", "perl -e", "bash -c", "sh -c", "$("]
        {
            if lowered.contains(marker) {
                ops.push(SensitiveOperation::NetworkAccess(format!(
                    "shell-indirection ({marker})"
                )));
            }
        }

        // Check process kill / signal
        let proc_bins = ["kill", "pkill", "killall", "fuser"];
        for bin in &proc_bins {
            if trimmed.starts_with(bin) || trimmed.contains(&format!(" {} ", bin)) {
                ops.push(SensitiveOperation::ProcessControl(bin.to_string()));
            }
        }

        // Check destructive filesystem operations
        if trimmed.contains("rm -rf /")
            || trimmed.contains("rm -rf /*")
            || trimmed.contains("rmdir /")
        {
            ops.push(SensitiveOperation::DestructiveFs(command.to_string()));
        }

        // Check out of scope writes
        let out_of_scope_prefixes = [
            "/etc",
            "/usr",
            "/boot",
            "/sys",
            "/dev",
            "C:\\Windows",
            "C:\\Program Files",
        ];
        for prefix in &out_of_scope_prefixes {
            if trimmed.contains(&format!("> {}", prefix))
                || trimmed.contains(&format!(">> {}", prefix))
            {
                ops.push(SensitiveOperation::WriteOutsideScope(prefix.to_string()));
            }
        }

        ops
    }
}

/// Executor for the policy-aware `Bash` tool.
#[derive(Default)]
pub struct BashExecutor;

impl BashExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl ToolExecutor for BashExecutor {
    fn execute(
        &self,
        call: &ToolCall,
        gateway: &dyn HostGateway,
    ) -> Result<ToolExecutionResult, ToolError> {
        let command = call
            .input
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::Validation {
                message: "missing required 'command' parameter".into(),
                details: Some(call.input.clone()),
            })?;

        let cwd = call
            .input
            .get("cwd")
            .and_then(|v| v.as_str())
            .map(str::to_owned);

        let timeout = Some(crate::definition::parse_timeout_ms(
            &call.input,
            30_000,
            120_000,
        ));
        let pty = call
            .input
            .get("pty")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let is_async = call
            .input
            .get("async")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Analyze command policy and detect sensitive operations
        let sensitive = CommandPolicyAnalyzer::analyze(command);
        let mut capability_demands = Vec::new();
        let mut diags = Vec::new();

        for op in &sensitive {
            match op {
                SensitiveOperation::NetworkAccess(bin) => {
                    capability_demands.push("network".to_string());
                    diags.push(ToolDiagnostic::info(format!(
                        "Command requires network capability: {}",
                        bin
                    )));
                }
                SensitiveOperation::GitPush => {
                    capability_demands.push("git_push".to_string());
                    diags.push(ToolDiagnostic::warning(
                        "Command requests remote repository mutation (git push)",
                    ));
                }
                SensitiveOperation::ProcessControl(bin) => {
                    capability_demands.push("process_control".to_string());
                    diags.push(ToolDiagnostic::warning(format!(
                        "Command requests process signaling: {}",
                        bin
                    )));
                }
                SensitiveOperation::WriteOutsideScope(target) => {
                    capability_demands.push("write_outside_scope".to_string());
                    diags.push(ToolDiagnostic::error(format!(
                        "Command targets out-of-scope system path: {}",
                        target
                    )));
                }
                SensitiveOperation::DestructiveFs(cmd) => {
                    return Err(ToolError::CapabilityDenied {
                        capability: format!(
                            "destructive root filesystem operation blocked: {}",
                            cmd
                        ),
                    });
                }
            }
        }

        let mut env = BTreeMap::new();

        // Also merge any call-specified env variables
        if let Some(extra_env) = call.input.get("env").and_then(|v| v.as_object()) {
            for (k, v) in extra_env {
                if let Some(s) = v.as_str() {
                    env.insert(k.clone(), s.to_string());
                }
            }
        }

        let req = HostRequest::ExecuteProcess {
            command: command.to_string(),
            cwd,
            env,
            timeout_ms: timeout,
            pty,
            is_async,
            capability_demands,
        };

        let resp = gateway.request(req)?;
        let mut result = ToolExecutionResult::from_host(resp);
        result.diagnostics.extend(diags);

        // Strict bounded output enforcement (100,000 bytes)
        const MAX_OUTPUT_BYTES: usize = 100_000;
        if result.output.content.len() > MAX_OUTPUT_BYTES {
            let mut cut_idx = MAX_OUTPUT_BYTES;
            while cut_idx > 0 && !result.output.content.is_char_boundary(cut_idx) {
                cut_idx -= 1;
            }
            result.output.content.truncate(cut_idx);
            result.output.truncated = true;
            result.diagnostics.push(ToolDiagnostic::warning(format!(
                "Bash output exceeded limit of {} bytes and was truncated.",
                MAX_OUTPUT_BYTES
            )));
        }

        Ok(result)
    }
}

/// Creates the permanent `Bash` tool definition.
pub fn bash_tool_definition(executor: Arc<BashExecutor>) -> ToolDefinition {
    ToolDefinition::new(
        "Bash",
        "1.0.0",
        json!({
            "type": "string",
            "description": "Present-participle intent describing what the command computes or effects"
        }),
        json!({
            "type": "object",
            "required": ["command", "i"],
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command or pipeline to execute under policy limits"
                },
                "i": {
                    "type": "string",
                    "description": "Present-participle intent"
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory"
                },
                "env": {
                    "type": "object",
                    "description": "Environment variables to inject"
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (0 = disabled)"
                },
                "pty": {
                    "type": "boolean",
                    "description": "Allocate interactive PTY for terminal commands"
                },
                "async": {
                    "type": "boolean",
                    "description": "Run asynchronously in background"
                }
            }
        }),
        vec!["process_spawn".into(), "job_control".into()],
        ToolLimits {
            max_output_bytes: 100_000,
            max_runtime_ms: 60_000,
            max_artifacts: 10,
            max_concurrent_jobs: 4,
        },
        executor,
        "bash",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::HostResponse;
    use crate::definition::DirectHostGateway;

    #[test]
    fn test_command_policy_analyzer() {
        let sensitive = CommandPolicyAnalyzer::analyze("git push origin main");
        assert!(
            sensitive
                .iter()
                .any(|s| matches!(s, SensitiveOperation::GitPush))
        );

        let sensitive_net = CommandPolicyAnalyzer::analyze("curl https://example.com");
        assert!(
            sensitive_net
                .iter()
                .any(|s| matches!(s, SensitiveOperation::NetworkAccess(_)))
        );
    }

    #[test]
    fn test_bash_output_bounding() {
        let executor = BashExecutor;
        let gateway = DirectHostGateway::new();

        let huge = "A".repeat(120_000);
        gateway.push_response(HostResponse::success(huge));

        let call = ToolCall::new(
            omp_types::ToolCallId::mint(),
            "Bash",
            "1.0.0",
            json!({
                "command": "cat huge.txt",
                "i": "Reading huge text"
            }),
        );
        let res = executor.execute(&call, &gateway).unwrap();
        assert!(res.output.content.len() <= 100_000);
        assert!(res.output.truncated);
    }
}
