use crate::component::{Component, ComponentProps, ComponentValidationError};
use crate::out::{AnsiOutSink, StringOutSink};
use crate::richtext::{Color, RichText, SemanticColor, Style};
use omp_types::{ElementId, ElementSnapshot, Status};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

fn status_color(status: &Status) -> SemanticColor {
    match status {
        Status::Queued => SemanticColor::Muted,
        Status::Active | Status::Running => SemanticColor::Primary,
        Status::Finalized | Status::Committed | Status::Succeeded => SemanticColor::Success,
        Status::CancelRequested | Status::Cancelled | Status::Detached => SemanticColor::Warning,
        Status::Failed | Status::WriteFailure => SemanticColor::Error,
        Status::Truncated => SemanticColor::Warning,
        Status::Unknown => SemanticColor::Muted,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadComponent {
    pub path: String,
    pub resolved_path: Option<String>,
    pub status: Status,
    pub head_preview: Option<String>,
    pub blob: Option<String>,
    pub diagnostics: Vec<String>,
    pub usage: Option<(u64, u64)>,
}

impl ReadComponent {
    pub fn to_component(&self) -> Result<Component, ComponentValidationError> {
        let mut rows = Vec::new();

        // 1. Title row + status badge
        let mut title_rt = RichText::new();
        title_rt.push(
            Style::new()
                .bold()
                .fg(Color::Semantic(SemanticColor::Primary)),
            format!("Read: {}", self.path),
        );
        if let Some(res) = &self.resolved_path
            && res != &self.path {
                title_rt.push(Style::semantic(SemanticColor::Muted), format!(" -> {res}"));
            }
        let title_comp = Component::text(title_rt);
        let badge_comp = Component::badge(status_color(&self.status), format!("{:?}", self.status));
        rows.push(Component::row(vec![title_comp, badge_comp])?);

        // 2. Optional head/pre block
        if let Some(head) = &self.head_preview {
            rows.push(Component::pre(head));
        }

        // 3. Expanded blob
        if let Some(blob) = &self.blob {
            rows.push(Component::pre(blob));
        }

        // 4. Diagnostics
        for diag in &self.diagnostics {
            rows.push(Component::callout(
                SemanticColor::Warning,
                "Diagnostic",
                diag,
            )?);
        }

        // 5. Usage
        if let Some((prompt, comp)) = self.usage {
            rows.push(Component::usage(prompt, comp, None));
        }

        let mut props = ComponentProps::new();
        props = props.set("border", true);
        props = props.set("title", "Read Tool");
        Component::box_container(props, rows)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BashComponent {
    pub command: String,
    pub cwd: Option<String>,
    pub status: Status,
    pub exit_code: Option<i32>,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
    pub truncated: bool,
}

impl BashComponent {
    pub fn to_component(&self) -> Result<Component, ComponentValidationError> {
        let mut children = Vec::new();

        let mut cmd_rt = RichText::new();
        cmd_rt.push(
            Style::new()
                .bold()
                .fg(Color::Semantic(SemanticColor::Primary)),
            format!("$ {}", self.command),
        );
        if let Some(cwd) = &self.cwd {
            cmd_rt.push(
                Style::semantic(SemanticColor::Muted),
                format!(" [in {cwd}]"),
            );
        }
        let cmd_comp = Component::text(cmd_rt);
        let badge_comp = Component::badge(status_color(&self.status), format!("{:?}", self.status));
        children.push(Component::row(vec![cmd_comp, badge_comp])?);

        if let Some(code) = self.exit_code {
            let _code_color = if code == 0 {
                SemanticColor::Success
            } else {
                SemanticColor::Error
            };
            children.push(Component::plain_text(format!("exit code: {code}")));
        }

        if let Some(out) = &self.stdout
            && !out.is_empty() {
                children.push(Component::pre(out));
            }

        if let Some(err) = &self.stderr
            && !err.is_empty() {
                children.push(Component::callout(SemanticColor::Error, "stderr", err)?);
            }

        if self.truncated {
            children.push(Component::callout(
                SemanticColor::Warning,
                "Truncation Notice",
                "Output exceeded limit and was truncated. Full log preserved in artifact.",
            )?);
        }

        let mut props = ComponentProps::new();
        props = props.set("border", true);
        props = props.set("title", "Bash Execution");
        Component::box_container(props, children)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditComponent {
    pub path: String,
    pub status: Status,
    pub diff_stats: Option<(usize, usize)>,
    pub hunks: Vec<String>,
}

impl EditComponent {
    pub fn to_component(&self) -> Result<Component, ComponentValidationError> {
        let mut children = Vec::new();

        let mut title_rt = RichText::new();
        title_rt.push(
            Style::new()
                .bold()
                .fg(Color::Semantic(SemanticColor::Primary)),
            format!("Edit: {}", self.path),
        );
        if let Some((added, removed)) = self.diff_stats {
            title_rt.push(
                Style::semantic(SemanticColor::Success),
                format!(" +{added}"),
            );
            title_rt.push(
                Style::semantic(SemanticColor::Error),
                format!(" -{removed}"),
            );
        }
        let title_comp = Component::text(title_rt);
        let badge_comp = Component::badge(status_color(&self.status), format!("{:?}", self.status));
        children.push(Component::row(vec![title_comp, badge_comp])?);

        for hunk in &self.hunks {
            children.push(Component::pre(hunk));
        }

        let mut props = ComponentProps::new();
        props = props.set("border", true);
        props = props.set("title", "File Edit");
        Component::box_container(props, children)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MessageComponent {
    pub role: String,
    pub author: Option<String>,
    pub content: String,
    pub thinking: Option<String>,
    pub tokens: Option<(u64, u64)>,
    #[serde(default)]
    pub status: Option<Status>,
    #[serde(default)]
    pub diagnostics: Vec<String>,
}

impl MessageComponent {
    pub fn to_component(&self) -> Result<Component, ComponentValidationError> {
        let mut children = Vec::new();

        if self.role != "assistant" {
            let mut header_rt = RichText::new();
            let role_color = match self.role.as_str() {
                "user" => SemanticColor::Primary,
                "system" => SemanticColor::Muted,
                "tool" => SemanticColor::Warning,
                _ => SemanticColor::Info,
            };
            header_rt.push(
                Style::new().bold().fg(Color::Semantic(role_color)),
                self.role.to_uppercase(),
            );
            if let Some(author) = &self.author {
                header_rt.push(
                    Style::semantic(SemanticColor::Muted),
                    format!(" ({author})"),
                );
            }
            let header_comp = if let Some(status) = &self.status {
                if matches!(status, Status::Running | Status::Failed | Status::Cancelled) {
                    let badge = Component::badge(status_color(status), format!("{status:?}"));
                    Component::row(vec![Component::text(header_rt), badge])?
                } else {
                    Component::text(header_rt)
                }
            } else {
                Component::text(header_rt)
            };
            children.push(header_comp);
        } else if let Some(status) = &self.status
            && matches!(status, Status::Running | Status::Failed | Status::Cancelled) {
                let badge = Component::badge(status_color(status), format!("{status:?}"));
                children.push(badge);
            }

        if let Some(think) = &self.thinking {
            children.push(Component::callout(SemanticColor::Muted, "Thinking", think)?);
        }

        if !self.content.is_empty() || self.thinking.is_none() {
            children.push(Component::plain_text(&self.content));
        }

        for diag in &self.diagnostics {
            children.push(Component::callout(
                SemanticColor::Warning,
                "Diagnostic",
                diag,
            )?);
        }

        if let Some((p, c)) = self.tokens {
            children.push(Component::usage(p, c, None));
        }

        Component::col(children)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubagentComponent {
    pub actor_id: String,
    pub status: Status,
    pub task_prompt: String,
    pub diff_summary: Option<String>,
}

impl SubagentComponent {
    pub fn to_component(&self) -> Result<Component, ComponentValidationError> {
        let mut children = Vec::new();

        let title_rt = RichText::from_styled(
            Style::new()
                .bold()
                .fg(Color::Semantic(SemanticColor::Primary)),
            format!("Subagent: {}", self.actor_id),
        );
        let badge_comp = Component::badge(status_color(&self.status), format!("{:?}", self.status));
        children.push(Component::row(vec![Component::text(title_rt), badge_comp])?);

        children.push(Component::callout(
            SemanticColor::Info,
            "Task",
            &self.task_prompt,
        )?);

        if let Some(diff) = &self.diff_summary {
            children.push(Component::pre(diff));
        }

        let mut props = ComponentProps::new();
        props = props.set("border", true);
        props = props.set("title", "Subagent Actor");
        Component::box_container(props, children)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApprovalComponent {
    pub id: String,
    pub action: String,
    pub status: Status,
    pub description: String,
    pub options: Vec<String>,
}

impl ApprovalComponent {
    pub fn to_component(&self) -> Result<Component, ComponentValidationError> {
        let mut children = Vec::new();

        let title_rt = RichText::from_styled(
            Style::new()
                .bold()
                .fg(Color::Semantic(SemanticColor::Warning)),
            format!("Approval Required: {}", self.action),
        );
        let badge = Component::badge(status_color(&self.status), format!("{:?}", self.status));
        children.push(Component::row(vec![Component::text(title_rt), badge])?);

        children.push(Component::plain_text(&self.description));

        let mut opt_row = Vec::new();
        for opt in &self.options {
            opt_row.push(Component::badge(SemanticColor::Primary, opt));
        }
        if !opt_row.is_empty() {
            children.push(Component::row(opt_row)?);
        }

        let mut props = ComponentProps::new();
        props = props.set("border", true);
        props = props.set("title", "Capability Approval");
        Component::box_container(props, children)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DirectorComponent {
    pub name: String,
    pub attempts: u32,
    pub max_attempts: u32,
    pub status: Status,
    pub prompt: Option<String>,
}

impl DirectorComponent {
    pub fn to_component(&self) -> Result<Component, ComponentValidationError> {
        let mut children = Vec::new();

        let title_rt = RichText::from_styled(
            Style::new()
                .bold()
                .fg(Color::Semantic(SemanticColor::Accent)),
            format!(
                "Director: {} ({}/{})",
                self.name, self.attempts, self.max_attempts
            ),
        );
        let badge = Component::badge(status_color(&self.status), format!("{:?}", self.status));
        children.push(Component::row(vec![Component::text(title_rt), badge])?);

        if let Some(p) = &self.prompt {
            children.push(Component::plain_text(p));
        }

        Component::col(children)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactComponent {
    pub artifact_id: String,
    pub media_type: String,
    pub byte_size: usize,
    pub origin_job: Option<String>,
}

impl ArtifactComponent {
    pub fn to_component(&self) -> Result<Component, ComponentValidationError> {
        let mut children = Vec::new();

        let title_rt = RichText::from_styled(
            Style::new()
                .bold()
                .fg(Color::Semantic(SemanticColor::Primary)),
            format!("Artifact: {}", self.artifact_id),
        );
        let size_badge =
            Component::badge(SemanticColor::Muted, format!("{} bytes", self.byte_size));
        children.push(Component::row(vec![Component::text(title_rt), size_badge])?);

        children.push(Component::plain_text(format!("Media: {}", self.media_type)));
        if let Some(job) = &self.origin_job {
            children.push(Component::plain_text(format!("Origin: {job}")));
        }

        Component::col(children)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoQAComponent {
    pub tool_name: String,
    pub tool_version: String,
    pub severity: String,
    pub expected: String,
    pub observed: String,
    pub reproduction: Option<String>,
}

impl AutoQAComponent {
    pub fn to_component(&self) -> Result<Component, ComponentValidationError> {
        let mut children = Vec::new();

        let sev_color = match self.severity.to_lowercase().as_str() {
            "high" | "critical" => SemanticColor::Error,
            "medium" => SemanticColor::Warning,
            _ => SemanticColor::Info,
        };

        let title_rt = RichText::from_styled(
            Style::new().bold().fg(Color::Semantic(sev_color)),
            format!("AutoQA Report: {} v{}", self.tool_name, self.tool_version),
        );
        let badge = Component::badge(sev_color, &self.severity);
        children.push(Component::row(vec![Component::text(title_rt), badge])?);

        children.push(Component::callout(
            SemanticColor::Info,
            "Expected",
            &self.expected,
        )?);
        children.push(Component::callout(
            SemanticColor::Error,
            "Observed",
            &self.observed,
        )?);

        if let Some(repro) = &self.reproduction {
            children.push(Component::pre(repro));
        }

        let mut props = ComponentProps::new();
        props = props.set("border", true);
        props = props.set("title", "Quality Assurance");
        Component::box_container(props, children)
    }
}

pub type ElementRendererFn =
    Arc<dyn Fn(&ElementSnapshot) -> Result<Component, String> + Send + Sync>;

#[derive(Default, Clone)]
pub struct SemanticRegistry {
    renderers: BTreeMap<String, ElementRendererFn>,
}

impl SemanticRegistry {
    pub fn new() -> Self {
        let mut reg = Self {
            renderers: BTreeMap::new(),
        };
        reg.register_defaults();
        reg
    }

    pub fn register(
        &mut self,
        kind: impl Into<String>,
        renderer: impl Fn(&ElementSnapshot) -> Result<Component, String> + Send + Sync + 'static,
    ) {
        self.renderers.insert(kind.into(), Arc::new(renderer));
    }

    fn register_defaults(&mut self) {
        self.register("message", |elem| {
            let role = elem
                .attributes
                .get("role")
                .and_then(|v| match v {
                    omp_types::TypedValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "user".to_string());

            let author = elem.attributes.get("author").and_then(|v| match v {
                omp_types::TypedValue::String(s) => Some(s.clone()),
                _ => None,
            });

            let comp = MessageComponent {
                role,
                author,
                content: elem.text.clone(),
                thinking: None,
                tokens: None,
                status: None,
                diagnostics: Vec::new(),
            };
            comp.to_component().map_err(|e| e.to_string())
        });

        self.register("assistant", |elem| {
            let author = elem.attributes.get("author").and_then(|v| match v {
                omp_types::TypedValue::String(s) => Some(s.clone()),
                _ => None,
            });
            let thinking = elem.attributes.get("thinking").and_then(|v| match v {
                omp_types::TypedValue::String(s) => Some(s.clone()),
                _ => None,
            });
            let comp = MessageComponent {
                role: "assistant".into(),
                author,
                content: elem.text.clone(),
                thinking,
                tokens: None,
                status: None,
                diagnostics: Vec::new(),
            };
            comp.to_component().map_err(|e| e.to_string())
        });

        self.register("tool_call", |elem| {
            let name = elem
                .attributes
                .get("name")
                .and_then(|v| match v {
                    omp_types::TypedValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "tool".to_string());

            let status_str = elem
                .attributes
                .get("status")
                .and_then(|v| match v {
                    omp_types::TypedValue::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or("Running");

            let status = match status_str {
                "Succeeded" => Status::Succeeded,
                "Failed" => Status::Failed,
                "Cancelled" => Status::Cancelled,
                _ => Status::Running,
            };

            if name == "read" {
                let path = elem
                    .attributes
                    .get("path")
                    .and_then(|v| match v {
                        omp_types::TypedValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| elem.text.clone());

                let read_comp = ReadComponent {
                    path,
                    resolved_path: None,
                    status,
                    head_preview: None,
                    blob: if elem.text.is_empty() {
                        None
                    } else {
                        Some(elem.text.clone())
                    },
                    diagnostics: Vec::new(),
                    usage: None,
                };
                return read_comp.to_component().map_err(|e| e.to_string());
            }

            if name == "bash" {
                let cmd = elem
                    .attributes
                    .get("command")
                    .and_then(|v| match v {
                        omp_types::TypedValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| elem.text.clone());

                let bash_comp = BashComponent {
                    command: cmd,
                    cwd: None,
                    status,
                    exit_code: None,
                    stdout: if elem.text.is_empty() {
                        None
                    } else {
                        Some(elem.text.clone())
                    },
                    stderr: None,
                    truncated: false,
                };
                return bash_comp.to_component().map_err(|e| e.to_string());
            }

            // Generic fallback tool
            let mut children = Vec::new();
            let title = RichText::from_styled(
                Style::new()
                    .bold()
                    .fg(Color::Semantic(SemanticColor::Primary)),
                format!("Tool Call: {name}"),
            );
            let badge = Component::badge(status_color(&status), format!("{status:?}"));
            children.push(
                Component::row(vec![Component::text(title), badge]).map_err(|e| e.to_string())?,
            );
            if !elem.text.is_empty() {
                children.push(Component::pre(&elem.text));
            }

            let mut props = ComponentProps::new();
            props = props.set("border", true);
            props = props.set("title", "Tool Call");
            Component::box_container(props, children).map_err(|e| e.to_string())
        });
    }

    /// Project the authoritative element and its structured children for every client.
    pub fn render_session_element(
        &self,
        snapshot: &omp_state::SessionSnapshot,
        id: &ElementId,
    ) -> Result<Component, String> {
        let element = snapshot
            .element(id)
            .ok_or_else(|| format!("Missing element {id}"))?;
        let text_attr = |name: &str| match element.attributes.get(name) {
            Some(omp_types::TypedValue::String(value)) => Some(value.as_str()),
            _ => None,
        };
        let status = text_attr("status")
            .and_then(|value| {
                serde_json::from_value::<Status>(serde_json::Value::String(value.into())).ok()
            })
            .unwrap_or(Status::Unknown);
        if matches!(
            element.kind.as_str(),
            "user" | "assistant" | "system" | "steering" | "message"
        ) {
            let resolved_status = if status == Status::Unknown
                && element.attributes.get("streaming") == Some(&omp_types::TypedValue::Bool(true))
            {
                Status::Running
            } else {
                status
            };
            let role = text_attr("role").unwrap_or(&element.kind);
            let author = text_attr("author").map(str::to_owned);

            let show_thinking = snapshot
                .element(snapshot.container("convars"))
                .and_then(|e| e.attributes.get("cl_showthinking"))
                .map(|v| match v {
                    omp_types::TypedValue::Bool(b) => *b,
                    omp_types::TypedValue::String(s) => !matches!(s.as_str(), "false" | "0"),
                    omp_types::TypedValue::Integer(i) => *i != 0,
                    _ => true,
                })
                .unwrap_or(true);

            let effective_thinking = if show_thinking {
                snapshot
                    .children(id)
                    .find(|child| child.kind == "think")
                    .map(|child| child.text.clone())
                    .or_else(|| text_attr("thinking").map(str::to_owned))
            } else {
                None
            };

            let diagnostics: Vec<String> = snapshot
                .children(id)
                .filter(|child| child.kind == "diag")
                .map(|child| child.text.clone())
                .collect();

            let message_status = match resolved_status {
                Status::Running | Status::Failed | Status::Cancelled => Some(resolved_status),
                _ => None,
            };

            let tokens = element.attributes.get("usage").and_then(|val| match val {
                omp_types::TypedValue::Json(v) => {
                    let prompt = v
                        .get("prompt_tokens")
                        .or_else(|| v.get("input_tokens"))
                        .and_then(|v| v.as_u64());
                    let completion = v
                        .get("completion_tokens")
                        .or_else(|| v.get("output_tokens"))
                        .and_then(|v| v.as_u64());
                    match (prompt, completion) {
                        (Some(p), Some(c)) => Some((p, c)),
                        _ => None,
                    }
                }
                _ => None,
            });

            return MessageComponent {
                role: role.into(),
                author,
                content: element.text.clone(),
                thinking: effective_thinking,
                tokens,
                status: message_status,
                diagnostics,
            }
            .to_component()
            .map_err(|error| error.to_string());
        }
        let data = element.payload.as_ref();
        let field = |key: &str| {
            data.and_then(|value| value.get(key))
                .and_then(|value| value.as_str())
        };
        let number = |key: &str| {
            data.and_then(|value| value.get(key))
                .and_then(|value| value.as_u64())
                .unwrap_or(0)
        };
        let semantic = match element.kind.as_str() {
            "subagent" => Some(
                SubagentComponent {
                    actor_id: text_attr("actor_id").unwrap_or(element.id.as_str()).into(),
                    status: status.clone(),
                    task_prompt: field("prompt").unwrap_or(&element.text).into(),
                    diff_summary: data
                        .and_then(|value| value.get("diff"))
                        .map(|value| value.to_string()),
                }
                .to_component(),
            ),
            "approval" => Some(
                ApprovalComponent {
                    id: element.id.to_string(),
                    action: text_attr("action")
                        .or(field("action"))
                        .unwrap_or("Capability request")
                        .into(),
                    status: status.clone(),
                    description: field("action_description").unwrap_or(&element.text).into(),
                    options: vec!["approve".into(), "deny".into()],
                }
                .to_component(),
            ),
            "director" => Some(
                DirectorComponent {
                    name: text_attr("kind").unwrap_or("Director").into(),
                    attempts: number("attempts") as u32,
                    max_attempts: number("max_attempts") as u32,
                    status: status.clone(),
                    prompt: field("reminder").map(str::to_owned),
                }
                .to_component(),
            ),
            "artifact" => Some(
                ArtifactComponent {
                    artifact_id: field("id").unwrap_or(element.id.as_str()).into(),
                    media_type: field("media_type")
                        .unwrap_or("application/octet-stream")
                        .into(),
                    byte_size: number("byte_length") as usize,
                    origin_job: field("origin_job").map(str::to_owned),
                }
                .to_component(),
            ),
            "job" => Some(Component::col(vec![
                Component::row(vec![
                    Component::plain_text(format!("Job: {}", element.id)),
                    Component::badge(status_color(&status), format!("{status:?}")),
                ])
                .map_err(|error| error.to_string())?,
                Component::pre(
                    data.map(|value| value.to_string())
                        .unwrap_or_else(|| element.text.clone()),
                ),
            ])),
            _ => None,
        };
        if let Some(component) = semantic {
            return component.map_err(|error| error.to_string());
        }
        if element.kind != "tool_call" {
            return self.render_component(element);
        }
        let name = text_attr("tool").ok_or("Tool name is missing")?;
        let input = snapshot
            .children(id)
            .find(|child| child.kind == "input")
            .and_then(|child| child.payload.as_ref());
        let input_text = |key: &str| {
            input
                .and_then(|value| value.get(key))
                .and_then(|value| value.as_str())
        };
        let result = snapshot.children(id).find(|child| child.kind == "result");
        let output = result.map(|child| child.text.clone());
        let payload = result.and_then(|child| child.payload.as_ref());
        let diagnostics: Vec<String> = snapshot
            .children(id)
            .filter(|child| child.kind == "diag")
            .map(|child| child.text.clone())
            .collect();
        let primary = match name {
            "Read" => ReadComponent {
                path: input_text("path").unwrap_or("").into(),
                resolved_path: payload
                    .and_then(|value| value.get("resolved_path"))
                    .and_then(|value| value.as_str())
                    .map(str::to_owned),
                status: status.clone(),
                head_preview: None,
                blob: output,
                diagnostics: Vec::new(),
                usage: None,
            }
            .to_component(),
            "Bash" => BashComponent {
                command: input_text("command").unwrap_or("").into(),
                cwd: input_text("cwd").map(str::to_owned),
                status: status.clone(),
                exit_code: payload
                    .and_then(|value| value.get("exit_code"))
                    .and_then(|value| value.as_i64())
                    .map(|value| value as i32),
                stdout: output,
                stderr: None,
                truncated: result.is_some_and(|child| {
                    child.attributes.get("truncated") == Some(&omp_types::TypedValue::Bool(true))
                }),
            }
            .to_component(),
            "Edit" => EditComponent {
                path: input_text("path").unwrap_or("").into(),
                status: status.clone(),
                diff_stats: None,
                hunks: output.into_iter().collect(),
            }
            .to_component(),
            "AutoQA" => AutoQAComponent {
                tool_name: input_text("tool").unwrap_or("").into(),
                tool_version: input_text("tool_version").unwrap_or("").into(),
                severity: input_text("severity").unwrap_or("info").into(),
                expected: input_text("expected_behavior").unwrap_or("").into(),
                observed: input_text("observed_behavior").unwrap_or("").into(),
                reproduction: input
                    .and_then(|value| value.get("reproduction_data"))
                    .map(|value| value.to_string()),
            }
            .to_component(),
            _ => {
                let mut rows = vec![
                    Component::row(vec![
                        Component::plain_text(format!(
                            "{name}: {}",
                            text_attr("intent").unwrap_or("")
                        )),
                        Component::badge(status_color(&status), format!("{status:?}")),
                    ])
                    .map_err(|error| error.to_string())?,
                ];
                if let Some(output) = output {
                    rows.push(Component::pre(output));
                }
                Component::col(rows)
            }
        }
        .map_err(|error| error.to_string())?;
        let mut rows = vec![primary];
        if let Some(image) = payload.and_then(|value| value.get("image"))
            && let Some(src) = image.get("src").and_then(|value| value.as_str()) {
                rows.push(Component::image(
                    src,
                    input_text("path").unwrap_or("Image"),
                    image
                        .get("width")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0) as u32,
                    image
                        .get("height")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0) as u32,
                ));
            }
        for diagnostic in diagnostics {
            rows.push(
                Component::callout(SemanticColor::Warning, "Diagnostic", diagnostic)
                    .map_err(|error| error.to_string())?,
            );
        }
        for child in snapshot.children(id) {
            match child.kind.as_str() {
                "artifact_ref" => rows.push(Component::plain_text(&child.text)),
                "usage" => {
                    if let Some(usage) = &child.payload {
                        let input = usage
                            .get("input_tokens")
                            .and_then(|value| value.as_u64())
                            .unwrap_or(0);
                        let output = usage
                            .get("output_tokens")
                            .and_then(|value| value.as_u64())
                            .unwrap_or(0);
                        rows.push(Component::usage(input, output, None));
                    }
                }
                _ => {}
            }
        }
        Component::col(rows).map_err(|error| error.to_string())
    }

    pub fn render_session(
        &self,
        snapshot: &omp_state::SessionSnapshot,
    ) -> Result<Component, String> {
        Component::col(
            snapshot
                .get_visible_body()
                .map(|element| self.render_session_element(snapshot, &element.id))
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(|error| error.to_string())
    }

    pub fn render_component(&self, element: &ElementSnapshot) -> Result<Component, String> {
        if let Some(renderer) = self.renderers.get(&element.kind) {
            renderer(element)
        } else {
            // Default generic fallback
            let mut rt = RichText::new();
            rt.push(
                Style::new()
                    .bold()
                    .fg(Color::Semantic(SemanticColor::Primary)),
                format!("<{}> ", element.kind),
            );
            rt.push_plain(&element.text);
            Ok(Component::text(rt))
        }
    }

    pub fn render_plain(&self, element: &ElementSnapshot) -> Result<String, String> {
        let comp = self.render_component(element)?;
        let mut sink = StringOutSink::new();
        comp.render_to_sink(&mut sink, 0)
            .map_err(|e| e.to_string())?;
        Ok(sink.into_string())
    }

    pub fn render_tui(&self, element: &ElementSnapshot) -> Result<String, String> {
        let comp = self.render_component(element)?;
        let mut sink = AnsiOutSink::new();
        comp.render_to_sink(&mut sink, 0)
            .map_err(|e| e.to_string())?;
        Ok(sink.into_string())
    }

    pub fn render_debug_json(
        &self,
        element: &ElementSnapshot,
    ) -> Result<serde_json::Value, String> {
        let comp = self.render_component(element)?;
        serde_json::to_value(comp).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omp_state::SessionSnapshot;
    use omp_types::{
        ElementId, ElementSnapshot, JournalOffset, Patch, PatchOp, SessionId, TypedValue,
    };

    #[test]
    fn test_render_assistant_streaming_with_think_child() {
        let mut snapshot = SessionSnapshot::empty(SessionId::mint());
        let asst_id = ElementId::mint();
        let think_id = ElementId::mint();

        let mut asst = ElementSnapshot::new(asst_id.clone(), "assistant");
        asst.attributes
            .insert("status".into(), TypedValue::String("running".into()));
        asst.attributes
            .insert("streaming".into(), TypedValue::Bool(true));
        asst.text = "Partial answer text".into();

        let mut think = ElementSnapshot::new(think_id.clone(), "think");
        think.text = "Internal thought stream".into();

        let body_container = snapshot.container("body").clone();
        let patch = Patch {
            base_offset: JournalOffset(snapshot.offset),
            result_offset: JournalOffset(snapshot.offset + 1),
            by: omp_types::ActorId::new("owner").unwrap().into(),
            reason: "assistant stream".into(),
            ops: vec![
                PatchOp::Create {
                    parent: body_container,
                    index: 0,
                    element: asst,
                },
                PatchOp::Create {
                    parent: asst_id.clone(),
                    index: 0,
                    element: think,
                },
            ],
        };
        omp_state::apply_patch(&mut snapshot, &patch).unwrap();

        let registry = SemanticRegistry::new();
        let comp = registry
            .render_session_element(&snapshot, &asst_id)
            .unwrap();
        let mut sink = StringOutSink::new();
        comp.render_to_sink(&mut sink, 0).unwrap();
        let out = sink.into_string();

        assert!(!out.contains("ASSISTANT"), "unexpected ASSISTANT: {out}");
        assert!(out.contains("[Running]"), "missing [Running] badge: {out}");
        assert!(out.contains("Thinking"), "missing Thinking callout: {out}");
        assert!(!out.contains("[Muted]"), "unexpected [Muted]: {out}");
        assert!(
            out.contains("Internal thought stream"),
            "missing thought: {out}"
        );
        assert!(
            out.contains("Partial answer text"),
            "missing partial answer: {out}"
        );
    }

    #[test]
    fn test_render_assistant_hides_thinking_when_cl_showthinking_false() {
        let mut snapshot = SessionSnapshot::empty(SessionId::mint());
        let asst_id = ElementId::mint();
        let think_id = ElementId::mint();

        let mut asst = ElementSnapshot::new(asst_id.clone(), "assistant");
        asst.attributes
            .insert("status".into(), TypedValue::String("running".into()));
        asst.attributes
            .insert("streaming".into(), TypedValue::Bool(true));
        asst.text = "Visible response".into();

        let mut think = ElementSnapshot::new(think_id.clone(), "think");
        think.text = "Secret thought".into();

        let body_container = snapshot.container("body").clone();
        let convars_container = snapshot.container("convars").clone();
        let patch = Patch {
            base_offset: JournalOffset(snapshot.offset),
            result_offset: JournalOffset(snapshot.offset + 1),
            by: omp_types::ActorId::new("owner").unwrap().into(),
            reason: "setup".into(),
            ops: vec![
                PatchOp::SetAttribute {
                    element: convars_container,
                    name: "cl_showthinking".into(),
                    value: TypedValue::String("0".into()),
                },
                PatchOp::Create {
                    parent: body_container,
                    index: 0,
                    element: asst,
                },
                PatchOp::Create {
                    parent: asst_id.clone(),
                    index: 0,
                    element: think,
                },
            ],
        };
        omp_state::apply_patch(&mut snapshot, &patch).unwrap();

        let registry = SemanticRegistry::new();
        let comp = registry
            .render_session_element(&snapshot, &asst_id)
            .unwrap();
        let mut sink = StringOutSink::new();
        comp.render_to_sink(&mut sink, 0).unwrap();
        let out = sink.into_string();

        assert!(!out.contains("ASSISTANT"), "unexpected ASSISTANT: {out}");
        assert!(out.contains("[Running]"), "missing [Running] badge: {out}");
        assert!(out.contains("Visible response"), "missing response: {out}");
        assert!(!out.contains("Secret thought"), "leaked thought: {out}");
        assert!(!out.contains("Thinking"), "thinking callout shown: {out}");
    }

    #[test]
    fn test_render_assistant_failed_and_cancelled_distinguished() {
        let mut snapshot = SessionSnapshot::empty(SessionId::mint());
        let failed_id = ElementId::mint();
        let cancelled_id = ElementId::mint();
        let diag_id = ElementId::mint();

        let mut failed = ElementSnapshot::new(failed_id.clone(), "assistant");
        failed
            .attributes
            .insert("status".into(), TypedValue::String("failed".into()));
        failed.text = "Incomplete response".into();

        let mut diag = ElementSnapshot::new(diag_id.clone(), "diag");
        diag.text = "Context length exceeded".into();

        let mut cancelled = ElementSnapshot::new(cancelled_id.clone(), "assistant");
        cancelled
            .attributes
            .insert("status".into(), TypedValue::String("cancelled".into()));
        cancelled.text = "Interrupted response".into();

        let body_container = snapshot.container("body").clone();
        let patch = Patch {
            base_offset: JournalOffset(snapshot.offset),
            result_offset: JournalOffset(snapshot.offset + 1),
            by: omp_types::ActorId::new("owner").unwrap().into(),
            reason: "failures".into(),
            ops: vec![
                PatchOp::Create {
                    parent: body_container.clone(),
                    index: 0,
                    element: failed,
                },
                PatchOp::Create {
                    parent: failed_id.clone(),
                    index: 0,
                    element: diag,
                },
                PatchOp::Create {
                    parent: body_container,
                    index: 1,
                    element: cancelled,
                },
            ],
        };
        omp_state::apply_patch(&mut snapshot, &patch).unwrap();

        let registry = SemanticRegistry::new();

        // Failed
        let comp_failed = registry
            .render_session_element(&snapshot, &failed_id)
            .unwrap();
        let mut sink_f = StringOutSink::new();
        comp_failed.render_to_sink(&mut sink_f, 0).unwrap();
        let out_f = sink_f.into_string();
        assert!(
            out_f.contains("[Failed]"),
            "missing [Failed] badge: {out_f}"
        );
        assert!(
            out_f.contains("Incomplete response"),
            "missing text: {out_f}"
        );
        assert!(
            out_f.contains("Context length exceeded"),
            "lost provider diagnostic: {out_f}"
        );

        // Cancelled
        let comp_cancelled = registry
            .render_session_element(&snapshot, &cancelled_id)
            .unwrap();
        let mut sink_c = StringOutSink::new();
        comp_cancelled.render_to_sink(&mut sink_c, 0).unwrap();
        let out_c = sink_c.into_string();
        assert!(
            out_c.contains("[Cancelled]"),
            "missing [Cancelled] badge: {out_c}"
        );
        assert!(
            out_c.contains("Interrupted response"),
            "missing text: {out_c}"
        );
    }

    #[test]
    fn test_render_legacy_completed_assistant() {
        let mut snapshot = SessionSnapshot::empty(SessionId::mint());
        let asst_id = ElementId::mint();

        let mut asst = ElementSnapshot::new(asst_id.clone(), "assistant");
        asst.attributes.insert(
            "thinking".into(),
            TypedValue::String("Completed thoughts".into()),
        );
        asst.text = "Final completed response".into();

        let body_container = snapshot.container("body").clone();
        let patch = Patch {
            base_offset: JournalOffset(snapshot.offset),
            result_offset: JournalOffset(snapshot.offset + 1),
            by: omp_types::ActorId::new("owner").unwrap().into(),
            reason: "legacy".into(),
            ops: vec![PatchOp::Create {
                parent: body_container,
                index: 0,
                element: asst,
            }],
        };
        omp_state::apply_patch(&mut snapshot, &patch).unwrap();

        let registry = SemanticRegistry::new();
        let comp = registry
            .render_session_element(&snapshot, &asst_id)
            .unwrap();
        let mut sink = StringOutSink::new();
        comp.render_to_sink(&mut sink, 0).unwrap();
        let out = sink.into_string();

        assert!(!out.contains("ASSISTANT"), "unexpected ASSISTANT: {out}");
        assert!(!out.contains("[Running]"), "unexpected running: {out}");
        assert!(!out.contains("[Failed]"), "unexpected failed: {out}");
        assert!(!out.contains("[Cancelled]"), "unexpected cancelled: {out}");
        assert!(out.contains("Thinking"), "missing Thinking callout: {out}");
        assert!(!out.contains("[Muted]"), "unexpected [Muted]: {out}");
        assert!(
            out.contains("Completed thoughts"),
            "missing thinking text: {out}"
        );
        assert!(
            out.contains("Final completed response"),
            "missing response: {out}"
        );
    }
}
