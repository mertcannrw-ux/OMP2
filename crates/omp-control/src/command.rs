use crate::convar::{ConVarError, ConVarFlags, ConVarStore};
use omp_types::{ActorId, ElementId, JournalOffset, Patch, PatchOp, StructuredError, TypedValue};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

/// Provider setup contains references to host credentials, never credential values.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ProviderAction {
    Show,
    Refresh,
    Configure {
        endpoint: String,
        adapter: String,
        key_env: String,
        model: Option<String>,
    },
    Select {
        model: String,
    },
    /// List every configured provider and mark the active one.
    List,
    /// Register a provider without switching to it.
    Add {
        name: String,
        endpoint: String,
        adapter: String,
        key_env: String,
        model: Option<String>,
    },
    /// Make a registered provider the active one.
    Use {
        name: String,
    },
    /// Forget a registered provider.
    Remove {
        name: String,
    },
}

/// Dialect assumed for an endpoint nobody described.
fn endpoint_default_adapter(endpoint: &str) -> String {
    if endpoint.starts_with("https://api.anthropic.com/") {
        "anthropic".to_string()
    } else {
        "openai_compatible".to_string()
    }
}

/// Parses `--adapter/--key-env/--model` pairs shared by `add` and bare endpoint
/// configuration. Credentials are never accepted, only the variable name.
fn parse_provider_options(
    default_adapter: String,
    options: &[String],
) -> Result<(String, String, Option<String>), CommandError> {
    let mut adapter = default_adapter;
    let mut key_env = "OMP_API_KEY".to_string();
    let mut model = None;
    let mut seen = std::collections::BTreeSet::new();
    let mut pairs = options.chunks_exact(2);
    for pair in &mut pairs {
        if !seen.insert(pair[0].as_str()) {
            return Err(CommandError::Parse("duplicate provider option".into()));
        }
        match pair[0].as_str() {
            "--adapter" => adapter = pair[1].clone(),
            "--key-env" => key_env = pair[1].clone(),
            "--model" => model = Some(pair[1].clone()),
            _ => {
                return Err(CommandError::Parse(
                    "unknown provider option; use /provider --help".into(),
                ));
            }
        }
    }
    if !pairs.remainder().is_empty() {
        return Err(CommandError::Parse("provider option needs a value".into()));
    }
    Ok((adapter, key_env, model))
}

impl ProviderAction {
    /// True when the action only edits the provider registry.
    ///
    /// Registry edits are configuration: they must be allowed inside a `exec`
    /// stream (a profile or user config declaring providers) and must not force
    /// the "run this alone" rule that exists for actions performing network
    /// work.
    pub fn is_pure_configuration(&self) -> bool {
        matches!(self, Self::Add { .. } | Self::Remove { .. } | Self::List)
    }
}

/// Representation of a parsed command.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Command {
    SetConVar {
        name: String,
        value: String,
    },
    GetConVar {
        name: String,
    },
    Bind {
        key: String,
        command: String,
    },
    Unbind {
        key: String,
    },
    Toggle {
        name: String,
        values: Vec<String>,
    },
    Alias {
        name: String,
        command: String,
    },
    Unalias {
        name: String,
    },
    Exec {
        path: String,
    },
    Echo {
        message: String,
    },
    Force {
        tool: String,
        reminder: Option<String>,
        max_attempts: Option<u32>,
    },
    ToolCall {
        name: String,
        input: serde_json::Value,
    },
    DynDiscovery {
        query: Option<String>,
        action: Option<String>,
        help: bool,
    },
    CancelJob {
        job_id: String,
    },
    Provider {
        action: ProviderAction,
    },
    Custom {
        name: String,
        args: Vec<String>,
    },
}
/// Recorded outcome of an executed command.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CommandEffect {
    ConVarUpdated {
        name: String,
        old_value: TypedValue,
        new_value: TypedValue,
        flags: ConVarFlags,
    },
    ConVarQueried {
        name: String,
        value: TypedValue,
    },
    Bound {
        key: String,
        command: String,
    },
    Unbound {
        key: String,
    },
    Aliased {
        name: String,
        command: String,
    },
    Unaliased {
        name: String,
    },
    ExecRequested {
        path: String,
    },
    Output {
        message: String,
    },
    ForcePushed {
        tool: String,
        reminder: String,
        max_attempts: u32,
    },
    ToolExecuted {
        name: String,
        input: serde_json::Value,
    },
    DynDiscovered {
        query: Option<String>,
    },
    JobCancelled {
        job_id: String,
    },
    Provider {
        action: ProviderAction,
    },
}

#[derive(Debug, Error)]
pub enum CommandError {
    #[error("parse error: {0}")]
    Parse(String),
    #[error("convar error: {0}")]
    ConVar(#[from] ConVarError),
    #[error("alias recursion limit exceeded (depth > {depth}) for alias '{name}'")]
    RecursionLimit { name: String, depth: usize },
    #[error("command expansion byte limit exceeded: {bytes} > {limit}")]
    ExpansionLimit { bytes: usize, limit: usize },
    #[error("unknown command: '{0}'")]
    UnknownCommand(String),
    #[error("missing required argument for command '{cmd}': {arg}")]
    MissingArgument { cmd: String, arg: String },
}

impl CommandError {
    pub fn to_structured_error(&self) -> StructuredError {
        match self {
            Self::Parse(msg) => StructuredError::new("cmd_parse_error", msg, false),
            Self::ConVar(err) => {
                let code = match err {
                    ConVarError::NotFound(_) => "cmd_convar_not_found",
                    ConVarError::ReadOnly(_) => "cmd_convar_read_only",
                    ConVarError::CheatProtected(_) => "cmd_cheat_protected",
                    ConVarError::InvalidValue { .. } => "cmd_convar_invalid_value",
                    ConVarError::TypeMismatch { .. } => "cmd_convar_type_mismatch",
                };
                StructuredError::new(code, err.to_string(), false)
            }
            Self::RecursionLimit { name, depth } => StructuredError::new(
                "cmd_recursion_limit",
                format!(
                    "alias recursion limit exceeded at depth {} on '{}'",
                    depth, name
                ),
                false,
            ),
            Self::ExpansionLimit { bytes, limit } => StructuredError::new(
                "cmd_expansion_limit",
                format!("expansion budget exceeded: {} bytes > {} max", bytes, limit),
                false,
            ),
            Self::UnknownCommand(cmd) => {
                StructuredError::new("cmd_unknown", format!("unknown command: {}", cmd), false)
            }
            Self::MissingArgument { cmd, arg } => StructuredError::new(
                "cmd_missing_arg",
                format!("command '{}' missing argument: {}", cmd, arg),
                false,
            ),
        }
    }
}

/// Tokenizer for splitting commands deterministically while respecting quotes and escapes.
pub struct CommandTokenizer;

impl CommandTokenizer {
    /// Tokenize a raw script string into individual commands separated by `;` or unescaped newlines.
    pub fn split_statements(input: &str) -> Vec<String> {
        let mut statements = Vec::new();
        let mut current = String::new();
        let mut in_quotes = false;
        let mut chars = input.chars().peekable();

        while let Some(c) = chars.next() {
            match c {
                '\\' => {
                    current.push('\\');
                    if let Some(next) = chars.next() {
                        current.push(next);
                    }
                }
                '"' => {
                    in_quotes = !in_quotes;
                    current.push('"');
                }
                '/' if !in_quotes
                    && chars.peek() == Some(&'/')
                    && current.chars().last().is_none_or(char::is_whitespace) =>
                {
                    // Skip single-line comment to newline
                    chars.next();
                    for next_ch in chars.by_ref() {
                        if next_ch == '\n' {
                            break;
                        }
                    }
                    if !current.trim().is_empty() {
                        statements.push(current.trim().to_string());
                        current.clear();
                    }
                }
                '#' if !in_quotes => {
                    // Skip shell-style comment to newline
                    for next_ch in chars.by_ref() {
                        if next_ch == '\n' {
                            break;
                        }
                    }
                    if !current.trim().is_empty() {
                        statements.push(current.trim().to_string());
                        current.clear();
                    }
                }
                ';' | '\n' | '\r' if !in_quotes => {
                    let trimmed = current.trim();
                    if !trimmed.is_empty() {
                        statements.push(trimmed.to_string());
                    }
                    current.clear();
                }
                other => {
                    current.push(other);
                }
            }
        }

        let trimmed = current.trim();
        if !trimmed.is_empty() {
            statements.push(trimmed.to_string());
        }

        statements
    }

    /// Tokenize a single command line into word tokens, respecting quotes and escapes.
    pub fn tokenize_line(line: &str) -> Result<Vec<String>, CommandError> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let mut in_quotes = false;
        let mut chars = line.chars().peekable();

        while let Some(c) = chars.next() {
            match c {
                '\\' => {
                    if let Some(next) = chars.next() {
                        match next {
                            'n' => current.push('\n'),
                            't' => current.push('\t'),
                            'r' => current.push('\r'),
                            '"' => current.push('"'),
                            '\\' => current.push('\\'),
                            other => {
                                current.push('\\');
                                current.push(other);
                            }
                        }
                    } else {
                        current.push('\\');
                    }
                }
                '"' => {
                    in_quotes = !in_quotes;
                }
                c if c.is_whitespace() && !in_quotes => {
                    if !current.is_empty() {
                        tokens.push(std::mem::take(&mut current));
                    }
                }
                other => {
                    current.push(other);
                }
            }
        }

        if in_quotes {
            return Err(CommandError::Parse("unclosed quote in command".into()));
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        Ok(tokens)
    }
}

/// Deterministic command parser.
pub struct CommandParser;

impl CommandParser {
    /// Parse a single statement into a `Command`.
    pub fn parse_statement(statement: &str) -> Result<Command, CommandError> {
        let tokens = CommandTokenizer::tokenize_line(statement)?;
        if tokens.is_empty() {
            return Err(CommandError::Parse("empty statement".into()));
        }

        let verb = &tokens[0];
        match verb.to_ascii_lowercase().as_str() {
            "bind" => {
                if tokens.len() < 2 {
                    return Err(CommandError::MissingArgument {
                        cmd: "bind".into(),
                        arg: "key".into(),
                    });
                }
                let key = tokens[1].clone();
                let command = if tokens.len() > 2 {
                    tokens[2..].join(" ")
                } else {
                    String::new()
                };
                Ok(Command::Bind { key, command })
            }
            "unbind" => {
                if tokens.len() < 2 {
                    return Err(CommandError::MissingArgument {
                        cmd: "unbind".into(),
                        arg: "key".into(),
                    });
                }
                Ok(Command::Unbind {
                    key: tokens[1].clone(),
                })
            }
            "toggle" => {
                if tokens.len() < 2 {
                    return Err(CommandError::MissingArgument {
                        cmd: "toggle".into(),
                        arg: "name".into(),
                    });
                }
                let name = tokens[1].clone();
                let values = tokens[2..].to_vec();
                Ok(Command::Toggle { name, values })
            }
            "alias" => {
                if tokens.len() < 2 {
                    return Err(CommandError::MissingArgument {
                        cmd: "alias".into(),
                        arg: "name".into(),
                    });
                }
                let name = tokens[1].clone();
                let command = if tokens.len() > 2 {
                    tokens[2..].join(" ")
                } else {
                    String::new()
                };
                Ok(Command::Alias { name, command })
            }
            "unalias" => {
                if tokens.len() < 2 {
                    return Err(CommandError::MissingArgument {
                        cmd: "unalias".into(),
                        arg: "name".into(),
                    });
                }
                Ok(Command::Unalias {
                    name: tokens[1].clone(),
                })
            }
            "exec" => {
                if tokens.len() < 2 {
                    return Err(CommandError::MissingArgument {
                        cmd: "exec".into(),
                        arg: "path".into(),
                    });
                }
                Ok(Command::Exec {
                    path: tokens[1].clone(),
                })
            }
            "echo" => Ok(Command::Echo {
                message: tokens[1..].join(" "),
            }),
            "force" | "/force" => {
                if tokens.len() < 2 {
                    return Err(CommandError::MissingArgument {
                        cmd: "force".into(),
                        arg: "tool".into(),
                    });
                }
                let tool = tokens[1].clone();
                // Reminder may contain spaces: join all tokens between the
                // tool name and an optional trailing integer max-attempts.
                // `force write Please call write 3` -> reminder "Please call
                // write", max_attempts 3.
                let (reminder, max_attempts) = match tokens.len() {
                    2 => (None, None),
                    _ => {
                        if tokens.len() > 3
                            && let Ok(attempts) = tokens[tokens.len() - 1].parse::<u32>()
                        {
                            (
                                Some(tokens[2..tokens.len() - 1].join(" ")),
                                Some(attempts),
                            )
                        } else {
                            (Some(tokens[2..].join(" ")), None)
                        }
                    }
                };
                Ok(Command::Force {
                    tool,
                    reminder,
                    max_attempts,
                })
            }
            "tool" | "call" => {
                if tokens.len() < 2 {
                    return Err(CommandError::MissingArgument {
                        cmd: "tool".into(),
                        arg: "name".into(),
                    });
                }
                let name = tokens[1].clone();
                let input_str = if tokens.len() > 2 {
                    tokens[2..].join(" ")
                } else {
                    "{}".to_string()
                };
                // Reject malformed JSON with a hint instead of masking the
                // error as `{"raw": ...}` (which tools would misinterpret).
                let input = serde_json::from_str(&input_str).map_err(|error| {
                    CommandError::Parse(format!(
                        "tool input is not valid JSON: {error}; quote the argument or pass {{}}"
                    ))
                })?;
                Ok(Command::ToolCall { name, input })
            }
            "dyn" => {
                let query = if tokens.len() > 1 && !tokens[1].starts_with("--") {
                    Some(tokens[1].clone())
                } else {
                    None
                };
                let help = tokens.iter().any(|t| t == "--help" || t == "-h");
                Ok(Command::DynDiscovery {
                    query,
                    action: None,
                    help,
                })
            }
            "cancel" => {
                if tokens.len() < 2 {
                    return Err(CommandError::MissingArgument {
                        cmd: "cancel".into(),
                        arg: "job_id".into(),
                    });
                }
                Ok(Command::CancelJob {
                    job_id: tokens[1].clone(),
                })
            }
            "job" if tokens.len() >= 3 && tokens[1].eq_ignore_ascii_case("cancel") => {
                Ok(Command::CancelJob {
                    job_id: tokens[2].clone(),
                })
            }
            "provider" | "/provider" => {
                let action = match tokens.get(1).map(String::as_str) {
                    None => ProviderAction::Show,
                    Some("refresh") if tokens.len() == 2 => ProviderAction::Refresh,
                    Some("select") if tokens.len() == 3 => ProviderAction::Select { model: tokens[2].clone() },
                    Some("list") if tokens.len() == 2 => ProviderAction::List,
                    Some("use") if tokens.len() == 3 => ProviderAction::Use { name: tokens[2].clone() },
                    Some("remove") if tokens.len() == 3 => ProviderAction::Remove { name: tokens[2].clone() },
                    Some("add") if tokens.len() >= 4 => {
                        let (adapter, key_env, model) = parse_provider_options(
                            endpoint_default_adapter(&tokens[3]),
                            &tokens[4..],
                        )?;
                        ProviderAction::Add {
                            name: tokens[2].clone(),
                            endpoint: tokens[3].clone(),
                            adapter,
                            key_env,
                            model,
                        }
                    }
                    Some("--help" | "help") => return Ok(Command::Echo { message: "Usage: /provider [list | add <name> <endpoint> [--adapter <dialect>] [--key-env <ENV>] [--model <id>] | use <name> | remove <name> | refresh | select <id> | <endpoint> [options]]. Providers are recorded in the session, so switching costs no retyping; the model list names the provider each model came from. Default dialect: OpenAI-compatible; key: OMP_API_KEY. Never paste API keys into commands.".into() }),
                    Some(endpoint) => {
                        let (adapter, key_env, model) =
                            parse_provider_options(endpoint_default_adapter(endpoint), &tokens[2..])?;
                        ProviderAction::Configure { endpoint: endpoint.into(), adapter, key_env, model }
                    }
                };
                Ok(Command::Provider { action })
            }
            _ => {
                if tokens.len() == 1 {
                    Ok(Command::GetConVar {
                        name: tokens[0].clone(),
                    })
                } else if tokens.len() == 2 {
                    Ok(Command::SetConVar {
                        name: tokens[0].clone(),
                        value: tokens[1].clone(),
                    })
                } else {
                    Ok(Command::Custom {
                        name: tokens[0].clone(),
                        args: tokens[1..].to_vec(),
                    })
                }
            }
        }
    }

    /// Parse full multi-statement command string.
    pub fn parse_script(script: &str) -> Result<Vec<Command>, CommandError> {
        let statements = CommandTokenizer::split_statements(script);
        let mut commands = Vec::with_capacity(statements.len());
        for stmt in statements {
            commands.push(Self::parse_statement(&stmt)?);
        }
        Ok(commands)
    }
}

/// Central budget limits to prevent alias recursion and unbounded expansion.
#[derive(Clone, Copy, Debug)]
pub struct ExpansionLimits {
    pub max_depth: usize,
    pub max_bytes: usize,
}

impl Default for ExpansionLimits {
    fn default() -> Self {
        Self {
            max_depth: 16,
            max_bytes: 65_536,
        }
    }
}

/// Command engine managing binds, aliases, and atomic execution against ConVarStore.
#[derive(Clone, Default)]
pub struct CommandEngine {
    binds: BTreeMap<String, String>,
    aliases: BTreeMap<String, String>,
    limits: ExpansionLimits,
}

impl CommandEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_limits(limits: ExpansionLimits) -> Self {
        Self {
            binds: BTreeMap::new(),
            aliases: BTreeMap::new(),
            limits,
        }
    }

    pub fn binds(&self) -> &BTreeMap<String, String> {
        &self.binds
    }

    pub fn aliases(&self) -> &BTreeMap<String, String> {
        &self.aliases
    }

    pub fn get_bind(&self, key: &str) -> Option<&String> {
        self.binds.get(key)
    }

    pub fn get_alias(&self, name: &str) -> Option<&String> {
        self.aliases.get(name)
    }

    pub fn hydrate_from_dom(&mut self, snapshot: &omp_state::SessionSnapshot) {
        self.binds.clear();
        self.aliases.clear();
        for (name, value) in snapshot.session_globals() {
            if let TypedValue::String(command) = value {
                if let Some(key) = name.strip_prefix("bind:") {
                    self.binds.insert(key.into(), command.clone());
                } else if let Some(alias) = name.strip_prefix("alias:") {
                    self.aliases.insert(alias.into(), command.clone());
                }
            }
        }
    }

    /// Expand a command string resolving any aliases up to limits.
    pub fn expand_command(
        &self,
        command_str: &str,
        depth: usize,
        visited_aliases: &mut Vec<String>,
        total_bytes: &mut usize,
    ) -> Result<Vec<Command>, CommandError> {
        *total_bytes += command_str.len();
        if *total_bytes > self.limits.max_bytes {
            return Err(CommandError::ExpansionLimit {
                bytes: *total_bytes,
                limit: self.limits.max_bytes,
            });
        }

        if depth > self.limits.max_depth {
            return Err(CommandError::RecursionLimit {
                name: visited_aliases.last().cloned().unwrap_or_default(),
                depth,
            });
        }

        let parsed = CommandParser::parse_script(command_str)?;
        let mut result = Vec::new();

        for cmd in parsed {
            match &cmd {
                Command::GetConVar { name } => {
                    if let Some(aliased_cmd) = self.aliases.get(name) {
                        if visited_aliases.contains(name) {
                            return Err(CommandError::RecursionLimit {
                                name: name.clone(),
                                depth,
                            });
                        }
                        visited_aliases.push(name.clone());
                        let mut sub = self.expand_command(
                            aliased_cmd,
                            depth + 1,
                            visited_aliases,
                            total_bytes,
                        )?;
                        result.append(&mut sub);
                        visited_aliases.pop();
                    } else {
                        result.push(cmd);
                    }
                }
                Command::Custom { name, args } => {
                    if let Some(aliased_cmd) = self.aliases.get(name) {
                        if visited_aliases.contains(name) {
                            return Err(CommandError::RecursionLimit {
                                name: name.clone(),
                                depth,
                            });
                        }
                        visited_aliases.push(name.clone());
                        let expanded_text = if args.is_empty() {
                            aliased_cmd.clone()
                        } else {
                            format!("{} {}", aliased_cmd, args.join(" "))
                        };
                        let mut sub = self.expand_command(
                            &expanded_text,
                            depth + 1,
                            visited_aliases,
                            total_bytes,
                        )?;
                        result.append(&mut sub);
                        visited_aliases.pop();
                    } else {
                        result.push(cmd);
                    }
                }
                _ => {
                    result.push(cmd);
                }
            }
        }

        Ok(result)
    }

    /// Execute a command script atomically against the ConVarStore.
    /// If any command fails, state changes are rolled back, ensuring no partial mutation.
    pub fn execute(
        &mut self,
        script: &str,
        convars: &mut ConVarStore,
    ) -> Result<Vec<CommandEffect>, CommandError> {
        self.execute_loaded(script, convars, &mut |path| {
            Err(CommandError::Parse(format!(
                "exec requires a host cfg loader: {path}"
            )))
        })
    }

    pub fn execute_loaded(
        &mut self,
        script: &str,
        convars: &mut ConVarStore,
        load: &mut dyn FnMut(&str) -> Result<String, CommandError>,
    ) -> Result<Vec<CommandEffect>, CommandError> {
        let convars_snapshot = convars.clone();
        let engine_snapshot = self.clone();
        let outer_transaction = convars.transaction_active();
        if !outer_transaction {
            convars.begin_transaction();
        }
        let mut effects = Vec::new();
        if let Err(error) = self.execute_stream(script, convars, load, 0, &mut 0, &mut effects) {
            *convars = convars_snapshot;
            *self = engine_snapshot;
            return Err(error);
        }
        if !outer_transaction {
            convars.commit_transaction();
        }
        Ok(effects)
    }

    fn execute_stream(
        &mut self,
        script: &str,
        convars: &mut ConVarStore,
        load: &mut dyn FnMut(&str) -> Result<String, CommandError>,
        depth: usize,
        bytes: &mut usize,
        effects: &mut Vec<CommandEffect>,
    ) -> Result<(), CommandError> {
        for statement in CommandTokenizer::split_statements(script) {
            let commands = self.expand_command(&statement, depth, &mut Vec::new(), bytes)?;
            for command in commands {
                if let Command::Exec { path } = command {
                    if depth >= self.limits.max_depth {
                        return Err(CommandError::RecursionLimit { name: path, depth });
                    }
                    let source = load(&path)?;
                    self.execute_stream(&source, convars, load, depth + 1, bytes, effects)?;
                    effects.push(CommandEffect::ExecRequested { path });
                } else {
                    effects.extend(self.execute_one(command, convars)?);
                }
            }
        }
        Ok(())
    }
    fn execute_one(
        &mut self,
        cmd: Command,
        convars: &mut ConVarStore,
    ) -> Result<Vec<CommandEffect>, CommandError> {
        match cmd {
            Command::SetConVar { name, value } => {
                let old_value = convars.get(&name).cloned().unwrap_or(TypedValue::Null);
                let def_flags = convars.get_def(&name).map(|d| d.flags).unwrap_or_default();
                let new_value = convars.set_from_str(&name, &value)?;
                Ok(vec![CommandEffect::ConVarUpdated {
                    name,
                    old_value,
                    new_value,
                    flags: def_flags,
                }])
            }
            Command::GetConVar { name } => {
                let val = convars
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| ConVarError::NotFound(name.clone()))?;
                Ok(vec![CommandEffect::ConVarQueried { name, value: val }])
            }
            Command::Bind { key, command } => {
                self.binds.insert(key.clone(), command.clone());
                Ok(vec![CommandEffect::Bound { key, command }])
            }
            Command::Unbind { key } => {
                self.binds.remove(&key);
                Ok(vec![CommandEffect::Unbound { key }])
            }
            Command::Toggle { name, values } => {
                let current = convars
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| ConVarError::NotFound(name.clone()))?;
                let def_flags = convars.get_def(&name).map(|d| d.flags).unwrap_or_default();

                let next_str = if values.is_empty() {
                    // Default toggle for boolean or tristate
                    match &current {
                        TypedValue::Bool(b) => (!b).to_string(),
                        TypedValue::Integer(i) => (if *i == 0 { 1 } else { 0 }).to_string(),
                        TypedValue::String(s) => {
                            if let Ok(ts) = s.parse::<crate::convar::TriState>() {
                                match ts {
                                    crate::convar::TriState::Unknown
                                    | crate::convar::TriState::False => "1".to_string(),
                                    crate::convar::TriState::True => "0".to_string(),
                                }
                            } else {
                                return Err(CommandError::ConVar(ConVarError::InvalidValue {
                                    name: name.clone(),
                                    reason: "cannot toggle non-boolean convar without explicit target values"
                                        .into(),
                                }));
                            }
                        }
                        TypedValue::Null => "1".to_string(),
                        _ => {
                            return Err(CommandError::ConVar(ConVarError::InvalidValue {
                                name: name.clone(),
                                reason: "cannot toggle non-boolean convar without explicit target values"
                                    .into(),
                            }));
                        }
                    }
                } else {
                    // Compare parsed values, so 0/false and 1/true are equivalent.
                    let definition =
                        convars.get_def(&name).ok_or_else(|| {
                            CommandError::ConVar(ConVarError::NotFound(name.clone()))
                        })?;
                    let typed_values: Result<Vec<_>, _> = values
                        .iter()
                        .map(|value| {
                            if let Some(parser) = &definition.parser {
                                parser(value).map_err(|reason| {
                                    CommandError::ConVar(ConVarError::InvalidValue {
                                        name: name.clone(),
                                        reason,
                                    })
                                })
                            } else {
                                Ok(TypedValue::String(value.clone()))
                            }
                        })
                        .collect();
                    let typed_values = typed_values?;
                    let idx = typed_values
                        .iter()
                        .position(|value| value == &current)
                        .unwrap_or(values.len() - 1);
                    let next_idx = (idx + 1) % values.len();
                    values[next_idx].clone()
                };

                let new_value = convars.set_from_str(&name, &next_str)?;
                Ok(vec![CommandEffect::ConVarUpdated {
                    name,
                    old_value: current,
                    new_value,
                    flags: def_flags,
                }])
            }
            Command::Alias { name, command } => {
                self.aliases.insert(name.clone(), command.clone());
                Ok(vec![CommandEffect::Aliased { name, command }])
            }
            Command::Unalias { name } => {
                self.aliases.remove(&name);
                Ok(vec![CommandEffect::Unaliased { name }])
            }
            Command::Exec { path } => Ok(vec![CommandEffect::ExecRequested { path }]),
            Command::Echo { message } => Ok(vec![CommandEffect::Output { message }]),
            Command::Force {
                tool,
                reminder,
                max_attempts,
            } => {
                let rem = reminder.unwrap_or_else(|| format!("invoke tool {}", tool));
                let max_att = max_attempts.unwrap_or(3);
                Ok(vec![CommandEffect::ForcePushed {
                    tool,
                    reminder: rem,
                    max_attempts: max_att,
                }])
            }
            Command::ToolCall { name, input } => {
                Ok(vec![CommandEffect::ToolExecuted { name, input }])
            }
            Command::DynDiscovery { query, .. } => Ok(vec![CommandEffect::DynDiscovered { query }]),
            Command::CancelJob { job_id } => Ok(vec![CommandEffect::JobCancelled { job_id }]),
            Command::Provider { action } => Ok(vec![CommandEffect::Provider { action }]),
            Command::Custom { name, args } => {
                // If it looks like convar set:
                if let Some(_def) = convars.get_def(&name) {
                    let old_value = convars.get(&name).cloned().unwrap_or(TypedValue::Null);
                    let def_flags = convars.get_def(&name).map(|d| d.flags).unwrap_or_default();
                    let new_value = convars.set_from_str(&name, &args.join(" "))?;
                    Ok(vec![CommandEffect::ConVarUpdated {
                        name,
                        old_value,
                        new_value,
                        flags: def_flags,
                    }])
                } else {
                    Err(CommandError::UnknownCommand(name))
                }
            }
        }
    }
}

/// Helper to convert session-scoped command effects into a journal patch.
pub fn command_effects_to_patch(
    effects: &[CommandEffect],
    base_offset: JournalOffset,
    result_offset: JournalOffset,
    by: ActorId,
    meta_convars_elem: &ElementId,
) -> Option<Patch> {
    let mut ops = Vec::new();
    for effect in effects {
        match effect {
            CommandEffect::ConVarUpdated {
                name,
                new_value,
                flags,
                ..
            } if flags.contains(ConVarFlags::SESSION) => {
                ops.push(PatchOp::SetAttribute {
                    element: meta_convars_elem.clone(),
                    name: name.clone(),
                    value: new_value.clone(),
                });
            }
            CommandEffect::Bound { key, command } => ops.push(PatchOp::SetAttribute {
                element: meta_convars_elem.clone(),
                name: format!("bind:{key}"),
                value: TypedValue::String(command.clone()),
            }),
            CommandEffect::Aliased { name, command } => ops.push(PatchOp::SetAttribute {
                element: meta_convars_elem.clone(),
                name: format!("alias:{name}"),
                value: TypedValue::String(command.clone()),
            }),
            CommandEffect::Unbound { key } => ops.push(PatchOp::RemoveAttribute {
                element: meta_convars_elem.clone(),
                name: format!("bind:{key}"),
            }),
            CommandEffect::Unaliased { name } => ops.push(PatchOp::RemoveAttribute {
                element: meta_convars_elem.clone(),
                name: format!("alias:{name}"),
            }),
            _ => {}
        }
    }

    if ops.is_empty() {
        None
    } else {
        Some(Patch {
            base_offset,
            result_offset,
            by: by.into(),
            reason: "command effects".into(),
            ops,
        })
    }
}
