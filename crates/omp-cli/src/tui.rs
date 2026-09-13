use crate::{
    CliError,
    editor::{Editor, completions},
    view::{self, DrawState, Panel, Transcript},
    worker::{Worker, WorkerEvent},
};
#[cfg(not(windows))]
use crossterm::terminal::disable_raw_mode;
use crossterm::{
    event::{
        DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, enable_raw_mode},
};
use omp_state::SessionSnapshot;
use ratatui::{Terminal, backend::CrosstermBackend};
use std::{
    io::{self, IsTerminal},
    path::Path,
    time::{Duration, Instant},
};

pub fn interactive() -> bool {
    io::stdin().is_terminal()
        && io::stdout().is_terminal()
        && std::env::var("TERM").as_deref() != Ok("dumb")
}

struct TerminalGuard {
    #[cfg(windows)]
    modes: [(windows_sys::Win32::Foundation::HANDLE, u32); 2],
}
impl TerminalGuard {
    fn enter() -> io::Result<Self> {
        #[cfg(windows)]
        let modes = {
            use windows_sys::Win32::System::Console::{
                GetConsoleMode, GetStdHandle, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
            };
            let mut modes = [(std::ptr::null_mut(), 0); 2];
            for (slot, kind) in modes.iter_mut().zip([STD_INPUT_HANDLE, STD_OUTPUT_HANDLE]) {
                slot.0 = unsafe { GetStdHandle(kind) };
                if unsafe { GetConsoleMode(slot.0, &mut slot.1) } == 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            modes
        };
        let guard = Self {
            #[cfg(windows)]
            modes,
        };
        enable_raw_mode()?;
        execute!(
            io::stdout(),
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture
        )?;
        Ok(guard)
    }
}
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = execute!(
            io::stdout(),
            DisableMouseCapture,
            DisableBracketedPaste,
            LeaveAlternateScreen,
            crossterm::cursor::Show
        );
        #[cfg(not(windows))]
        let _ = disable_raw_mode();
        #[cfg(windows)]
        for (handle, mode) in self.modes {
            unsafe {
                windows_sys::Win32::System::Console::SetConsoleMode(handle, mode);
            }
        }
    }
}

const HELP: &str = "Write a message and press Enter to ask the agent to work.\n\nINPUT\n  /                 Open the command menu; keep typing to filter\n  Up / Down         Select a command, move through input, or recall history\n  Tab               Complete the selected command without running it\n  Enter             Accept a command / send a message\n  Ctrl+J            New line (Shift+Enter also works where supported)\n  Left / Right      Move the cursor; Home / End move within a line\n  Ctrl+W            Delete the previous word\n  Ctrl+U / Ctrl+K   Delete to start / end of the line\n  Esc               Close a menu, or stop a running operation\n  Ctrl+C            Stop work; clear draft; press again on empty input to exit\n  Ctrl+D            Exit on empty input\n  Wheel / PgUp/PgDn Scroll the conversation\n  Ctrl+Home         Jump to the oldest message\n  Ctrl+End          Return to the latest message\n  Ctrl+L            Redraw the terminal\n\nCOMMANDS\n  /new              Start a clean session; keep settings and save the old chat\n  /model            Open the discovered-model picker\n  /provider         Show provider catalog and configuration\n  /provider refresh Refetch models and settings\n  /settings         Inspect effective session settings\n  /status           Inspect this session and provider status\n  /thinking         Toggle reasoning visibility\n  /jobs             Inspect session jobs\n  /actors           Inspect child actors\n  /inspect          Inspect the complete materialized session\n  /fork <offset>    Fork from a journal offset\n  /force <tool>     Require a tool through the host Director\n  /tool <name> JSON Execute a tool through the host capability boundary\n  /dyn              Discover dynamic tools\n  /cancel <job-id>  Cancel a background job\n  /exit             Save and exit\n  /detach           Detach the console\n\nProvider keys belong in host environment variables, never in chat.\nThe terminal is a view of the journal. Resize and redraw do not change history.";

fn local_panel(line: &str, snapshot: &SessionSnapshot) -> Result<Option<Panel>, CliError> {
    let (title, text) = match line {
        "/help" => ("Keyboard & commands", HELP.to_string()),
        "/settings" => (
            "Session settings",
            snapshot
                .session_globals()
                .iter()
                .map(|(name, value)| {
                    format!(
                        "{name} = {}",
                        serde_json::to_string(value).unwrap_or_default()
                    )
                })
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        "/status" => (
            "Session status",
            format!(
                "Session: {}\nBranch: {}\nJournal offset: {}\nTurns: {}\nProvider: {}\nEndpoint: {}\nModel: {}\nContext: {}\nOutput limit: {}\nActive jobs: {}\n\n/provider refresh updates the catalog.\nUnknown limits were not advertised by the provider.",
                snapshot.session_id,
                snapshot.current_branch(),
                snapshot.offset,
                snapshot.turn_count(),
                view::setting(snapshot, "ai_provider"),
                view::setting(snapshot, "ai_endpoint"),
                view::setting(snapshot, "ai_model"),
                view::setting(snapshot, "ai_context_length"),
                view::setting(snapshot, "ai_max_tokens"),
                snapshot.active_jobs().count()
            ),
        ),
        "/inspect" => (
            "Materialized session",
            serde_json::to_string_pretty(snapshot)?,
        ),
        "/actors" | "/jobs" => {
            let container = if line == "/actors" { "actors" } else { "jobs" };
            let entries = snapshot
                .children(snapshot.container(container))
                .collect::<Vec<_>>();
            let text = if entries.is_empty() {
                format!("No {container} in this session.")
            } else {
                serde_json::to_string_pretty(&entries)?
            };
            (
                if line == "/actors" {
                    "Session actors"
                } else {
                    "Session jobs"
                },
                text,
            )
        }
        _ => return Ok(None),
    };
    Ok(Some(Panel {
        title: title.into(),
        text,
        scroll: 0,
    }))
}

pub fn run(
    workspace: &Path,
    journal_path: &Path,
    mut snapshot: SessionSnapshot,
    refresh: bool,
    initial_message: Option<&str>,
) -> Result<(), CliError> {
    let mut worker = Worker::start(workspace, journal_path)?;
    let _guard = TerminalGuard::enter()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    terminal.clear()?;
    let mut input = crate::input::Input::new()?;
    let mut editor = Editor::default();
    let mut transcript = Transcript::default();
    let mut selected = 0usize;
    let mut dismissed = false;
    let mut busy = refresh;
    let mut ready = false;
    let mut started = Instant::now();
    let mut pending_initial = initial_message.map(str::to_owned);
    let mut notice = if refresh {
        "Connecting to provider…".to_string()
    } else {
        String::new()
    };
    let mut notice_error = false;
    let mut scroll = 0usize;
    let mut panel: Option<Panel> = None;
    let mut last_interrupt: Option<Instant> = None;
    let mut last_frame = Instant::now() - Duration::from_secs(1);
    let mut dirty = true;
    let mut detach = false;
    if refresh {
        worker.submit("/provider refresh")?;
    }
    loop {
        for event in worker.poll()? {
            match event {
                WorkerEvent::Snapshot { snapshot: next }
                | WorkerEvent::SessionChanged { snapshot: next, .. } => {
                    if snapshot.session_id != next.session_id {
                        editor = Editor::default();
                        transcript = Transcript::default();
                        scroll = 0;
                        selected = 0;
                        dismissed = false;
                        panel = None;
                        last_interrupt = None;
                        terminal.clear()?;
                    }
                    snapshot = next;
                    ready = true;
                    dirty = true;
                    if !busy && let Some(message) = pending_initial.take() {
                        worker.submit(&message)?;
                        busy = true;
                        started = Instant::now();
                    }
                }
                WorkerEvent::Patch { patch, branch } => {
                    omp_state::apply_patch(&mut snapshot, &patch)?;
                    snapshot.selected_branch = branch;
                    dirty = true;
                }
                WorkerEvent::Settled { error } => {
                    busy = false;
                    dirty = true;
                    notice_error = error.is_some();
                    notice = error
                        .map(|error| format!("{}: {}", error.code, error.message))
                        .unwrap_or_default();
                    if !notice_error && let Some(message) = pending_initial.take() {
                        worker.submit(&message)?;
                        busy = true;
                        started = Instant::now();
                    }
                }
            }
        }
        let candidates = if dismissed {
            vec![]
        } else {
            completions(&editor.text, &snapshot)
        };
        let menu_open = !dismissed
            && editor.text.starts_with('/')
            && !editor.text.contains('\n')
            && (!candidates.is_empty()
                || !editor.text.contains(' ')
                || editor.text.starts_with("/provider select "));
        selected = selected.min(candidates.len().saturating_sub(1));
        if dirty || (busy && last_frame.elapsed() >= Duration::from_millis(200)) {
            let mut draw_state = DrawState {
                snapshot: &snapshot,
                workspace,
                editor: &editor,
                candidates: &candidates,
                selected,
                menu_open,
                busy,
                elapsed: started.elapsed().as_secs(),
                notice: &notice,
                notice_error,
                scroll,
                panel: panel.as_ref(),
            };
            terminal.draw(|frame| {
                view::draw(frame, &mut draw_state, &mut transcript);
            })?;
            scroll = draw_state.scroll;
            last_frame = Instant::now();
            dirty = false;
        }
        let Some(event) = input.next(Duration::from_millis(25))? else {
            continue;
        };
        dirty = true;
        let mut submit = false;
        match event {
            Event::Resize(_, _) => {
                terminal.autoresize()?;
            }
            Event::Paste(text) => {
                if panel.is_none() {
                    if text.len() > 65_536 || !editor.insert(&text) {
                        notice = "Input limit: 65,536 bytes. Paste a smaller selection.".into();
                        notice_error = true;
                    }
                    dismissed = true;
                    selected = 0;
                }
            }
            Event::Mouse(mouse) => {
                if let Some(open) = &mut panel {
                    let panel_lines = open.text.lines().count();
                    let max_panel_scroll = panel_lines.saturating_sub(1) as u16;
                    match mouse.kind {
                        MouseEventKind::ScrollDown => {
                            open.scroll = open.scroll.saturating_add(3).min(max_panel_scroll);
                        }
                        MouseEventKind::ScrollUp => {
                            open.scroll = open.scroll.saturating_sub(3);
                        }
                        _ => {}
                    }
                } else {
                    match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            scroll = scroll.saturating_add(3).min(transcript.lines.len());
                        }
                        MouseEventKind::ScrollDown => {
                            scroll = scroll.saturating_sub(3);
                        }
                        _ => {}
                    }
                }
            }
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                let control = key.modifiers.contains(KeyModifiers::CONTROL);
                if let Some(open) = &mut panel {
                    let panel_lines = open.text.lines().count();
                    let max_panel_scroll = panel_lines.saturating_sub(1) as u16;
                    match key.code {
                        KeyCode::Esc => panel = None,
                        KeyCode::Char('c') if control => panel = None,
                        KeyCode::PageDown | KeyCode::Down => {
                            open.scroll = open
                                .scroll
                                .saturating_add(if key.code == KeyCode::Down { 1 } else { 10 })
                                .min(max_panel_scroll);
                        }
                        KeyCode::PageUp | KeyCode::Up => {
                            open.scroll = open.scroll.saturating_sub(if key.code == KeyCode::Up {
                                1
                            } else {
                                10
                            });
                        }
                        KeyCode::Home => open.scroll = 0,
                        KeyCode::End => open.scroll = max_panel_scroll,
                        _ => {}
                    }
                    continue;
                }
                match key.code {
                    KeyCode::Char('c') if control => {
                        if busy {
                            worker.interrupt()?;
                            busy = false;
                            ready = false;
                            pending_initial = None;
                            notice = "Stopped. Committed work is preserved; you can send another message.".into();
                            notice_error = false;
                        } else if !editor.text.is_empty() {
                            editor.clear();
                            dismissed = false;
                            notice.clear();
                            last_interrupt = Some(Instant::now());
                        } else if last_interrupt
                            .is_some_and(|time| time.elapsed() < Duration::from_secs(2))
                        {
                            break;
                        } else {
                            last_interrupt = Some(Instant::now());
                            notice = "Press Ctrl+C again to exit, or keep typing.".into();
                            notice_error = false;
                        }
                    }
                    KeyCode::Char('d') if control && editor.text.is_empty() => break,
                    KeyCode::Char('l') if control => terminal.clear()?,
                    KeyCode::Home if control => scroll = transcript.lines.len(),
                    KeyCode::End if control => scroll = 0,
                    KeyCode::PageUp => {
                        scroll = scroll.saturating_add(10).min(transcript.lines.len());
                    }
                    KeyCode::PageDown => scroll = scroll.saturating_sub(10),
                    KeyCode::Esc => {
                        if menu_open {
                            dismissed = true;
                        } else if busy {
                            worker.interrupt()?;
                            busy = false;
                            ready = false;
                            pending_initial = None;
                            notice="Stopped. Committed work is preserved; you can send another message.".into();
                            notice_error = false;
                        }
                    }
                    KeyCode::Up if menu_open && !candidates.is_empty() => {
                        selected = selected.checked_sub(1).unwrap_or(candidates.len() - 1)
                    }
                    KeyCode::Down if menu_open && !candidates.is_empty() => {
                        selected = (selected + 1) % candidates.len()
                    }
                    KeyCode::Tab if menu_open && !candidates.is_empty() => {
                        editor.set(candidates[selected].insert.clone());
                        selected = 0;
                    }
                    KeyCode::Enter
                        if !key
                            .modifiers
                            .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
                    {
                        if menu_open && !candidates.is_empty() {
                            let choice = &candidates[selected];
                            editor.set(choice.insert.clone());
                            submit = choice.execute;
                            selected = 0;
                        } else {
                            submit = true;
                        }
                    }
                    _ => {
                        if editor.handle_key(key) {
                            dismissed = false;
                            selected = 0;
                            notice.clear();
                            notice_error = false;
                        }
                    }
                }
            }
            _ => {}
        }
        if submit && !editor.text.trim().is_empty() {
            let command = editor.text.trim();
            if command == "/new" {
                // Reconcile stopped work in the old journal before switching; never run exit hooks.
                worker.interrupt()?;
                busy = false;
                ready = false;
                pending_initial = Some("/new".into());
                editor.clear();
                dismissed = false;
                selected = 0;
                notice = "Starting a clean session…".into();
                notice_error = false;
                continue;
            }
            if command == "/detach" && (busy || snapshot.active_jobs().count() > 0) {
                notice="Stop active work before detaching; local worker jobs cannot outlive their host.".into();
                notice_error = true;
                continue;
            }
            if command == "/exit" || command == "/detach" {
                detach = command == "/detach";
                break;
            }
            if command == "/model" {
                editor.set("/provider select ".into());
                dismissed = false;
                selected = 0;
                continue;
            }
            if let Some(local) = local_panel(command, &snapshot)? {
                let _ = editor.submit();
                panel = Some(local);
                dismissed = false;
                continue;
            }
            if busy || !ready {
                notice="Wait for the current operation, or press Esc to stop it. Your draft is preserved.".into();
                notice_error = false;
                continue;
            }
            if let Some(mut line) = editor.submit() {
                if line == "/thinking" {
                    line = "/toggle cl_showthinking".into();
                }
                worker.submit(&line)?;
                busy = true;
                started = Instant::now();
                scroll = 0;
                dismissed = false;
                selected = 0;
                notice.clear();
                notice_error = false;
            }
        }
    }
    if busy {
        worker.interrupt()?;
    }
    worker.stop(detach)?;
    Ok(())
}
