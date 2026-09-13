# OMP2

**Oh My Pi 2** — a systems-oriented agent harness.

OMP2 runs an LLM agent against a local workspace behind an explicit operating-system
security boundary. Every session is an append-only journal of document patches, so any
run can be inspected, replayed, forked, or resumed. Tool execution is contained in a
Windows AppContainer governed by a Job Object, remote clients attach over a role-scoped
TCP protocol, and the streaming transcript protocol is backed by a machine-checked TLA+
specification.

**Rust** (edition 2024) · ~46,200 lines · 10 crates · 228 tests · Windows-first

---

## Contents

- [What OMP2 is](#what-omp2-is)
- [Architecture](#architecture)
- [Core concepts](#core-concepts)
- [Tools](#tools)
- [Command line](#command-line)
- [Configuration](#configuration)
- [Provider setup](#provider-setup)
- [Python extensions](#python-extensions)
- [Formal specification](#formal-specification)
- [Building](#building)
- [Testing](#testing)
- [Repository layout](#repository-layout)

---

## What OMP2 is

OMP2 is a durable, journal-backed agent harness. The design rests on four decisions:

1. **All durable state is a document.** Session state lives in a tree of typed element
   nodes (`SessionSnapshot`). There is no separate database — the journal is the state.
2. **All mutation is a patch.** Every turn, tool call, command, and configuration change
   is appended to the journal as a `Patch` of `PatchOp` operations. Replay reconstructs
   any prior state exactly.
3. **The host is trusted, the sandbox is not.** Model-directed execution happens across a
   boundary that cannot select tools, mutate session state, or escalate capabilities.
4. **Presentation is specified, not improvised.** The streaming transcript protocol is
   defined in TLA+ and model-checked, then mapped construct-for-construct onto Rust.

Because state is a replayable journal, `inspect`, `fork`, `doctor`, and `replay` are
first-class operations rather than debugging afterthoughts.

---

## Architecture

| Crate | Responsibility |
| --- | --- |
| `omp-types` | Shared vocabulary: ids (`SessionId`, `ActorId`, `ElementId`, `BranchId`, `JournalOffset`), `Patch`/`PatchOp`/`PatchAuthor`, `TypedValue`, `ElementSnapshot`, `Status`, `StructuredError`, `LimitPolicy`, and the sandbox capability types. |
| `omp-state` | Durable state. The `SessionSnapshot` DOM (`dom.rs`) and the append-only `Journal` (`journal.rs`) with checksummed framing, branches, and replay recovery. |
| `omp-runtime` | The trusted-host / sandbox boundary. Killable `Job`, `BoundedStream`, the artifact store, copy-on-write `WorkspaceView` with diffing, and the Windows AppContainer + Job Object implementation. |
| `omp-inference` | Provider adapters. Semantic request → wire translation, bounded HTTP/SSE streaming, capability and compatibility gating, malformed-output repair, and local-model engines. |
| `omp-control` | The session authority. `SessionHost` turn loop, the director policy stack, convar store, command engine, provider catalog, and subagent child sessions. |
| `omp-tools` | The tool surface. Permanent roster plus dynamic discovery, the hashline `Edit` language, artifacts, and the Python bridge. |
| `omp-render` | Presentation. Typed component tree, `RichText`, streaming sinks, and the `TranscriptScheduler` implementing ElasticSlots. |
| `omp-server` | Remote sessions. Role/permission model, newline-delimited JSON over TCP, token authentication, and replication. |
| `omp-cli` | The `omp2` binary. Subcommand dispatch, interactive TUI (ratatui + crossterm), and the UI worker process. |
| `omp-python` | Embedded CPython 3.11.9 runtime and the Python extension host. |

Internal dependencies form a strict DAG rooted at `omp-types`, which depends on nothing
else in the workspace:

| Crate | Depends on |
| --- | --- |
| `omp-types` | — |
| `omp-state`, `omp-inference` | `omp-types` |
| `omp-runtime`, `omp-render` | `omp-types`, `omp-state` |
| `omp-python` | `omp-types`, `omp-state`, `omp-runtime` |
| `omp-tools` | `omp-types`, `omp-state`, `omp-runtime`, `omp-python` |
| `omp-control` | `omp-types`, `omp-state`, `omp-runtime`, `omp-inference`, `omp-tools` |
| `omp-server` | `omp-types`, `omp-state`, `omp-control`, `omp-render` |
| `omp-cli` | `omp-types`, `omp-state`, `omp-runtime`, `omp-control`, `omp-render`, `omp-server` |

---

## Core concepts

### The journal

A session's journal is a single append-only file. The format is deliberately small:

```
"OMP2J01\n"   magic, 8 bytes
frame         Header        { session_id, format, version }
frame         JournalRecord
frame         JournalRecord
              …
```

Each frame is `u32` length (little-endian) + JSON payload + a 32-byte SHA-256 of the
payload. Frames are capped at `MAX_WIRE_BYTES` (1 MiB). The writer holds an OS file lock
for the lifetime of the `Journal`, so a session has exactly one writer; every record is
checksummed, so a torn or corrupted suffix is detectable and `doctor --repair` can
truncate back to the last valid record.

A `JournalRecord` carries `offset`, `timestamp_ms`, `session_id`, `branch_id`,
`parent_offset`, the `patch`, its `actor` (`PatchAuthor::Actor` or `::Element`), the
`protocol_version`, and a `checksum`. Because records name their `parent_offset` and
`branch_id`, a journal is a tree: `fork --offset <n>` branches it, and `replay` prints
the ancestry chain of the selected branch.

### The document tree

`SessionSnapshot` holds `Node` values — an `ElementSnapshot` (`id`, `schema_version`,
`kind`, `attributes`, `text`, optional JSON `payload`) plus parent and children. Every
durable feature is a node; indexes and live handles are caches, never persisted.

The eight patch operations are `Create`, `Delete`, `Move`, `SetAttribute`,
`RemoveAttribute`, `ReplaceText`, `AppendText`, and `ReplacePayload`. `apply_patch` is
transactional — it undoes partial work on error — and undo tracks only the touched
values and subtrees rather than a second full copy of the document.

### Sandboxing

On Windows, model-directed execution runs inside an AppContainer (Lowbox) with:

- **No network by default.** Network capability is marked `Unsupported` unless granted.
- **An ACL-granted isolated `cwd`.** The host filesystem and the journal sit outside it
  and are denied at the kernel level.
- **A Low Mandatory Integrity Label** (`S:(ML;OICI;NW;;;LW)`) on the isolated `cwd`, so
  Lowbox processes can write without being blocked by Mandatory Integrity Control.
- **A Job Object** with `KILL_ON_JOB_CLOSE`, memory limits, CPU-time limits, an active
  process limit, and UI restrictions (clipboard, window handles, desktop switches).
- **Suspended creation**, assigned to the Job Object before the main thread resumes,
  eliminating uncontained spawn races.
- **Handle inheritance restricted** to the stdio pipes via
  `PROC_THREAD_ATTRIBUTE_HANDLE_LIST`, and a **sanitized environment block** that strips
  host tokens and API keys, retaining only what an AppContainer requires.

Temporary DACL grants on external binaries are tracked and restored on drop, so host
filesystem permissions are never permanently altered. The sandbox receives a
`SandboxRequest` of explicit `SandboxCapability` grants and can only return
`SandboxEvent`s — it cannot name tools or touch session state.

### Workspace isolation

Jobs and subagents may run against a copy-on-write `WorkspaceView`. Allocation captures a
`FileBaseline` of the workspace (path, size, SHA-256), and completion produces a
`WorkspaceDiff` of added/modified/deleted `FileDiffEntry` values computed against that
immutable baseline rather than the mutable live parent. Views are handed out under an
OS-locked `WorkspaceLease`, so two jobs can never mutate the same view.

Heavy dependency directories are **not** junctioned into a view by default: writes
through a junction land in the host workspace and are invisible to the diff. They are
created empty instead. Opting into junctioning is explicit:

```bash
OMP_ALLOW_HEAVY_JUNCTION_WRITE=1
```

### ElasticSlots

ElasticSlots is the streaming transcript protocol, decoupling three layers: semantic
block state, a width-independent logical history ledger, and the physical terminal.
Blocks move `Queued → Active → Finalized → Committed`; a block's mode (`Mutable` or
`AppendOnly`) is fixed at admission. Only the single append-only head block may stream
into history, and it stops at the commit frontier — mutable speculation never leaks into
the ledger. Resizes change presentation geometry only and never touch logical history.

The protocol is specified in [`spec/ElasticSlots.tla`](spec/ElasticSlots.tla) and
implemented in `crates/omp-render/src/transcript.rs`. See
[Formal specification](#formal-specification).

---

## Tools

The permanent roster is fixed at seven tools, plus dynamic discovery.

| Tool | Purpose |
| --- | --- |
| `Read` | Reads files, internal URIs (`skill://`, `artifact://`), and URLs, with inline selectors (`:50`, `:50-200`, `:raw`). |
| `Bash` | Runs a shell command or pipeline under policy limits. |
| `Write` | Creates or overwrites a file. |
| `Edit` | Applies surgical edits in the hashline patch language. |
| `Eval` | Runs code in a persistent Python or JavaScript kernel. |
| `Agent` | Delegates a batch of tasks to subagents. |
| `AutoQA` | Files a structured defect report when a tool misbehaves. |
| `dyn` | Discovers and invokes dynamically registered tools. |

### The hashline edit language

`Edit` is not a string-replace tool. It consumes a patch of sections addressed by
`[PATH#TAG]`, where the tag identifies the revision the patch was authored against:

```
EditOp::PutRange      PUT N.=M:   replace inclusive line range
EditOp::PutBlock      PUT N*:     replace syntactic block
EditOp::InsertBefore  PUT <N:     insert before line N
EditOp::InsertAfter   PUT >N:     insert after line N
EditOp::PasteBefore   PUT <N @reg paste register before line N
EditOp::PasteAfter    PUT >N @reg paste register after line N
EditOp::CutRange      CUT N.=M    cut inclusive lines
EditOp::CutBlock      CUT N*      cut syntactic block
EditOp::RemoveFile    REM         delete target file
EditOp::MoveFile      MV DEST     move/rename target file
```

Registers make multi-file restructures a single atomic patch. The parser is
`HashlineParser`; a patch whose section does not match the target path produces a
conflict diagnostic instead of a rewrite.

### Command policy

`CommandPolicyAnalyzer` inspects shell commands for `NetworkAccess`, `GitPush`,
`ProcessControl`, `WriteOutsideScope`, and `DestructiveFs`. It is **advisory** —
substring matching, trivially bypassed by shell indirection — and its own documentation
says a clean scan is not proof of safety. Real enforcement lives in the sandbox layer
(`SandboxCapability`, `sandbox_network`, workspace-scoped writes).

---

## Command line

The binary is `omp2`.

```
omp2 - Systems-oriented agent harness

LAUNCH:
    omp2                  Start the interactive terminal in the current workspace
    omp2 run              Equivalent explicit form

SUBCOMMANDS:
    run       Create a durable session with optional deterministic --message
    resume    Resume latest session from workspace state directory
    fork      Fork a session branch at --offset <u64>
    inspect   Inspect materialized session state as JSON or XML
    doctor    Validate journal protocol compatibility and recovery diagnostics
    replay    Display branch ancestry chain and selected branch
    serve     Run host-owned local service abstraction over TCP

OPTIONS:
    --journal <path>     Explicit journal file path
    --workspace <path>   Explicit workspace root
    --offset <u64>       Journal offset (required for fork)
    --message <text>     Deterministic initial user message (for run)
    --profile <name>     Profile cfg to load (e.g. default, factory, remote)
    --cfg <path>         Custom .cfg script to execute
    --format <json|xml>  Inspect output format (default: json)
    --repair             Repair damaged journal suffix in doctor
    --bind <addr>        Service bind address (default: 127.0.0.1:0)
    --transport <tcp>    Transport protocol (default: tcp)
    --once               Handle one connection then exit (serve)
    --json               Output structured JSON
    -h, --help           Print help information
```

### Interactive terminal

`run` and `resume` open the styled interface when stdin and stdout are terminals.
`/` opens the command menu (type to filter, arrows to select, Tab to fill, Enter to
accept). Responses and provider-exposed thinking stream live; `/thinking` toggles
reasoning visibility. Enter sends, `Ctrl+J` inserts a newline, and bracketed paste stays
an editable draft. Wheel, PgUp and PgDn scroll history; `Ctrl+Home`/`Ctrl+End` jump to the
ends. `/new` starts a clean session while preserving the previous journal. `Esc` stops
running work; `/exit` closes the session. Redirected streams and `TERM=dumb` fall back to
the plain line-oriented interface.

### Remote sessions

`serve` publishes the session over the TCP transport (newline-delimited JSON, 1 MiB
frames, 32 concurrent connections, 30 s read / 10 s write timeouts). Clients
authenticate into an `ActorRole`, and every operation is checked against `Permission`:

| `ActorRole` | Notes |
| --- | --- |
| `Controller` | Full session authority. |
| `InteractiveDriver` | Drives turns interactively. |
| `Spectator` | Read-only observer. |
| `SubagentInspector` | Observes delegated subagents. |
| `AutomationWorker` | Non-interactive execution. |

Auth tokens are written to a sibling file next to the journal with owner-only
permissions and are never printed to stdout. The wire contract for connection setup is
[`schemas/handshake.v1.schema.json`](schemas/handshake.v1.schema.json).

---

## Configuration

Configuration is a Source-engine-style convar store. Settings are `SetConVar` commands,
which means they are journaled like everything else — and therefore rewound, forked, and
replicated with the session.

| Convar | Purpose |
| --- | --- |
| `ai_endpoint`, `ai_provider`, `ai_model`, `ai_api_key_env` | Provider routing. |
| `ai_temperature`, `ai_max_tokens`, `ai_context_length`, `ai_compaction_threshold` | Sampling and context management. |
| `ai_thinking`, `ai_thinking_levels`, `ai_fastmode` | Reasoning controls. |
| `cl_theme`, `cl_icon_mode`, `cl_showthinking`, `cl_resize_policy` | Terminal presentation. |
| `sandbox_network`, `sandbox_write_scope` | Sandbox grants. |
| `job_max_concurrency`, `tool_max_output_bytes`, `tool_max_runtime_ms` | Resource limits. |
| `sv_cheats` | Gate for session manipulation commands. |

Profiles are `.cfg` scripts in [`profiles/`](profiles):

| Profile | Intent |
| --- | --- |
| `default` | Interactive developer. |
| `factory` | Autonomous software factory — untrusted repository, hostile isolation, network off, 2 concurrent jobs. |
| `remote` | Remote interactive driver. |
| `spectator` | Read-only observer. |

### Provider compatibility rules

`configs/compat/default.kdl` declares provider and model-family capabilities as
single-rule directives, so behaviour is data rather than hardcoded branches:

```
rule "anthropic-claude" host="api.anthropic.com" provider="anthropic" family="claude" priority=10 \
     cost="free" streaming="supported" native_tool_choice="supported" token_count="supported" \
     developer_role="unsupported" mid_session_system_prompts="unsupported" \
     remote_compaction="supported" usage_query="supported"
```

`CompatCompiler` turns these into a `CompatTable`; `omp-inference` consults it through
`TriState` capability profiles before every request. Rules may match on `host`,
`provider`, `family`, or model `class`, with `priority` resolving conflicts.

---

## Provider setup

Set the endpoint and key in the host environment before starting:

```bash
export OMP_ENDPOINT="https://api.anthropic.com"
export OMP_API_KEY="…"
omp2 run
```

Startup fetches the available models and advertised settings from the endpoint; no model
preset is selected for you. In the console, `/provider` shows the catalog and effective
settings, `/provider select <id>` chooses a discovered model, and `/provider refresh`
refetches after a discovery failure. Missing provider metadata stays `unknown` — model
limits are never guessed.

| Environment variable | Purpose |
| --- | --- |
| `OMP_ENDPOINT` (or `AI_ENDPOINT`, `OPENAI_BASE_URL`) | Provider endpoint. |
| `OMP_API_KEY` | Credentials. Falls back to `ANTHROPIC_API_KEY` for the Anthropic adapter. |
| `OMP_PROVIDER` (or `AI_PROVIDER`) | Adapter selection. |
| `OMP_MODEL` (or `AI_MODEL`) | Model selection. |
| `OMP_REGISTRY_URL` | Override the model registry source. |
| `OMP_LOCAL_MODEL_ENDPOINT`, `OMP_LOCAL_MODEL_NAME`, `OMP_BUNDLED_MODEL_PATH` | Local model engines. |
| `OMP_ALLOW_HEAVY_JUNCTION_WRITE` | Opt into junctioning heavy dirs into workspace views. |

API keys are read from the host environment and are never placed in console commands or
config files.

---

## Python extensions

OMP2 ships an embedded CPython 3.11.9 runtime, so extensions need no system Python. The
runtime is the pinned official Windows embeddable distribution; `PythonRuntime::bundled()`
verifies `assets/python-3.11.9-embed-amd64.zip` against the SHA-256 recorded in
[`crates/omp-python/assets/checksums.txt`](crates/omp-python/assets/checksums.txt) before
use, and each `PythonSession` runs in its own `WorkspaceView` under a `LimitPolicy`.

The SDK lives in [`python/omp_sdk/`](python/omp_sdk) and has one rule: **durable state
belongs in the session DOM, never in module-level Python state.** Extensions implement
`on_load`, `on_unload`, `on_reload`, and `on_patch`, and register tools, directors, and
components with the host:

```python
from omp_sdk import Extension, ExtensionContext, Tool

class TodoCounterTool(Tool):
    name = "todo_counter"
    version = "1.0.0"
    description = "Increments and reports todo items stored in the host session DOM"
    parameter_schema = {
        "type": "object",
        "properties": {"title": {"type": "string", "description": "Title of todo item to add"}},
        "required": ["title"],
    }

    def execute(self, arguments, context):
        self.context.dom.create_element(
            parent="todo",
            element_id=f"todo_{uuid.uuid4().hex[:12]}",
            kind="todo_item",
            attributes={"title": arguments["title"]},
            text=arguments["title"],
        )
        return {"status": "success"}


class StatefulTodoExtension(Extension):
    def __init__(self, transport=None):
        super().__init__(extension_id="stateful_todo_example", transport=transport)

    def on_load(self, context):
        super().on_load(context)
        self.register_tool(TodoCounterTool(context))
        self.sync_declarations()
```

Writes go through the host as journaled patches, so extension state participates in
rewind, fork, resume, and replication like any other session mutation. The full working
example is [`python/omp_sdk/examples.py`](python/omp_sdk/examples.py); the host-side
clients are `SessionClient`, `DOMClient`, `JobClient`, `ArtifactClient`,
`ConVarWrapper`, `CommandClient`, `Tool`, `Director`, and `Component`.

---

## Formal specification

[`spec/`](spec) contains the TLA+ specification for ElasticSlots, based on the Playbook
Appendix B specification. It is not documentation of intent — it is the source of truth
that `crates/omp-render/src/transcript.rs` is written against, and
[`spec/README`](spec/README) carries a construct-by-construct trace mapping (TLA+
`c` → `TranscriptScheduler.commit_frontier()`, `want[i]` → `TranscriptBlock.snapshot`,
`replayMode` → `TranscriptScheduler.replay_gate`, and so on).

The model is finite and executable under TLC, and verifies **11 state invariants**
including `ExactCommittedHistory` (the ledger is exactly committed rows followed by the
head's stable prefix — no duplicates, gaps, or reordering), `NoPrematureHistory`
(speculation never leaks), `NativeSourceSafety` (physical row provenance tags never lie),
and `FailedWriteStops` (a write failure halts the scheduler). It also checks **9 temporal
properties** including history monotonicity, final immutability, append-only prefix
monotonicity, and the resize invariant that geometry changes never touch logical history.

```bash
# Full check, including temporal action and liveness properties
tlc spec/ElasticSlots.tla -config spec/ElasticSlots.cfg

# Safety-only check (state invariants)
tlc spec/ElasticSlots.tla -config spec/ElasticSlots_safety.cfg
```

Six counterexample traces — out-of-order retirement, mutable speculative leakage,
non-monotonic append-only mutation, duplicate rows on retire-after-streaming, rebuild
resize without an epoch increment, and blind retry after a partial write — are recorded in
[`tests/fixtures/transcript_counterexamples.json`](tests/fixtures/transcript_counterexamples.json)
and exercised by `crates/omp-render/tests/`.

---

## Building

Requires the Rust toolchain pinned in [`rust-toolchain.toml`](rust-toolchain.toml)
(**1.97.0**, edition 2024, with `rustfmt` and `clippy`); `rustup` installs it
automatically.

```bash
cargo build            # debug
cargo build --release  # release
```

The binary is `target/debug/omp2` (`omp2.exe` on Windows). Build output lands in
`target/` per [`.cargo/config.toml`](.cargo/config.toml).

The embedded runtimes are vendored under `crates/*/assets/` (the Python embeddable
distribution and busybox), so no separate runtime download or system Python install is
required. Third-party crates are resolved from crates.io in the usual way; `Cargo.lock`
is committed, so a fresh clone resolves the same versions.

---

## Testing

```bash
cargo test --workspace          # unit + integration tests
cargo clippy --workspace --all-targets
```

228 tests across 13 integration test files and the crate unit suites. Notable suites:

| Suite | Covers |
| --- | --- |
| `crates/omp-state/tests/replay.rs` | Journal replay, branching, damaged-suffix recovery. |
| `crates/omp-control/tests/directors.rs` | Director policy: cfg exec and rewind, force-tool exhaustion across resume, compaction guard, child config inheritance. |
| `crates/omp-render/tests/transcript.rs` | ElasticSlots block lifecycle against the counterexample fixtures. |
| `crates/omp-runtime/tests/jobs.rs` | Job lifecycle, cancellation, and containment. |
| `crates/omp-types/tests/protocol.rs` | Wire types and the `patch.v1` schema surface. |
| `tests/smoke/cli_smoke.rs` | End-to-end exercise of all seven subcommands. |

Shared fixtures live in [`fixtures/`](fixtures) — branch rewind, convar command
sequences, journal patches, provider malformed tool calls, resize replay, sandbox
truncation/cancellation, component trees, and compatibility rules.

---

## Repository layout

```
crates/          Rust workspace (10 crates)
  omp-types/       shared types and wire vocabulary
  omp-state/       DOM + append-only journal
  omp-runtime/     host/sandbox boundary, jobs, artifacts, workspace views
  omp-inference/   provider adapters and streaming
  omp-control/     session host, directors, convars, commands
  omp-tools/       tool roster, hashline Edit, Python bridge
  omp-render/      component tree, RichText, ElasticSlots transcript
  omp-server/      TCP transport, roles, replication
  omp-cli/         the `omp2` binary and TUI
  omp-python/      embedded CPython runtime   (assets/ — vendored, checksummed)
python/          Python extension SDK and tests
spec/            ElasticSlots TLA+ specification and model configs
schemas/         JSON Schema for the wire patch and handshake formats
configs/         Default configuration and KDL compatibility rules
profiles/        Session profiles (default, factory, remote, spectator)
fixtures/        Shared cross-crate test fixtures
tests/           Workspace-level smoke tests and counterexample fixtures
```

---

## License

MIT — declared in `[workspace.package]` in [`Cargo.toml`](Cargo.toml).

Vendored third-party assets retain their own licenses: the CPython embeddable
distribution under `crates/omp-python/assets/` (see its bundled `LICENSE.txt`) and
busybox under `crates/omp-tools/assets/`.
