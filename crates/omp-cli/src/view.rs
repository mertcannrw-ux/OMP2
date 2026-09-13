use crate::editor::{Completion, Editor};
use omp_render::{Out, OutError, Run, SemanticColor};
use omp_state::SessionSnapshot;
use omp_types::TypedValue;
use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Layout, Margin, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap},
};
use std::path::Path;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

#[derive(Clone, Copy)]
pub struct Theme {
    pub background: Color,
    pub text: Color,
    pub muted: Color,
    pub accent: Color,
    pub border: Color,
    pub input: Color,
    pub selected: Color,
    pub error: Color,
}
impl Theme {
    pub fn from_snapshot(snapshot: &SessionSnapshot) -> Self {
        if setting(snapshot, "cl_theme") == "light" {
            Self {
                background: Color::Rgb(247, 248, 250),
                text: Color::Rgb(29, 35, 43),
                muted: Color::Rgb(88, 98, 111),
                accent: Color::Rgb(0, 100, 137),
                border: Color::Rgb(170, 177, 187),
                input: Color::Rgb(132, 85, 38),
                selected: Color::Rgb(218, 236, 241),
                error: Color::Rgb(175, 40, 48),
            }
        } else {
            Self {
                background: Color::Rgb(12, 14, 18),
                text: Color::Rgb(225, 229, 236),
                muted: Color::Rgb(153, 163, 179),
                accent: Color::Rgb(68, 199, 235),
                border: Color::Rgb(67, 77, 91),
                input: Color::Rgb(208, 169, 125),
                selected: Color::Rgb(26, 51, 64),
                error: Color::Rgb(255, 125, 132),
            }
        }
    }
    fn semantic(self, color: SemanticColor) -> Color {
        match color {
            SemanticColor::Info | SemanticColor::Primary => self.accent,
            SemanticColor::Muted => self.muted,
            SemanticColor::Warning | SemanticColor::Accent => self.input,
            SemanticColor::Error => self.error,
            SemanticColor::Success => Color::Rgb(109, 194, 151),
        }
    }
    fn color(self, color: omp_render::Color) -> Color {
        match color {
            omp_render::Color::Semantic(color) => self.semantic(color),
            omp_render::Color::Rgb(r, g, b) => Color::Rgb(r, g, b),
            omp_render::Color::Ansi256(index) => Color::Indexed(index),
            _ => self.text,
        }
    }
    fn run_style(self, style: omp_render::Style) -> Style {
        let mut result = Style::default().fg(style.fg.map(|c| self.color(c)).unwrap_or(self.text));
        if let Some(color) = style.bg {
            result = result.bg(self.color(color));
        }
        for (set, modifier) in [
            (style.modifier.bold, Modifier::BOLD),
            (style.modifier.italic, Modifier::ITALIC),
            (style.modifier.underline, Modifier::UNDERLINED),
            (style.modifier.inverse, Modifier::REVERSED),
            (style.modifier.strikethrough, Modifier::CROSSED_OUT),
        ] {
            if set {
                result = result.add_modifier(modifier);
            }
        }
        result
    }
}

/// `  ·  effort <level>` for the status footer.
///
/// Shows the active level, or `auto` with the provider's default in
/// parentheses so "whatever the model would do" is still visible. Models that
/// advertise no effort levels get no indicator at all.
fn effort_indicator(snapshot: &SessionSnapshot) -> String {
    let configured = setting(snapshot, "ai_thinking");
    let configured = configured.trim();
    let default = snapshot
        .element(snapshot.container("capabilities"))
        .and_then(|node| node.attributes.get("provider_metadata"))
        .and_then(|value| match value {
            TypedValue::Json(metadata) => metadata
                .pointer("/active_model/thinking_default")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            _ => None,
        });
    let advertised = matches!(
        snapshot
            .element(snapshot.container("convars"))
            .and_then(|node| node.attributes.get("ai_thinking_levels")),
        Some(TypedValue::Json(value)) if value.as_array().is_some_and(|levels| !levels.is_empty())
    );
    let label = match configured {
        "" | "auto" => match (default, advertised) {
            (Some(default), _) => format!("auto ({default})"),
            (None, true) => "auto".to_string(),
            // Nothing advertised and nothing asked for: nothing to say.
            (None, false) => return String::new(),
        },
        "off" | "none" | "0" | "disabled" => "off".to_string(),
        level => level.to_string(),
    };
    format!("  ·  effort {label}")
}

/// ` @ host` for the configured endpoint, so the status line says which
/// gateway the model came from. Empty when no endpoint is configured.
fn gateway_suffix(snapshot: &SessionSnapshot) -> String {
    let endpoint = setting(snapshot, "ai_endpoint");
    if endpoint.is_empty() {
        return String::new();
    }
    let host = endpoint
        .split("://")
        .nth(1)
        .unwrap_or(&endpoint)
        .split(['/', '?'])
        .next()
        .unwrap_or("")
        .to_string();
    if host.is_empty() {
        String::new()
    } else {
        format!(" @ {host}")
    }
}

pub fn setting(snapshot: &SessionSnapshot, name: &str) -> String {
    snapshot
        .element(snapshot.container("convars"))
        .and_then(|element| element.attributes.get(name))
        .map(|value| match value {
            TypedValue::String(value) => value.clone(),
            TypedValue::Integer(value) => value.to_string(),
            TypedValue::Bool(value) => value.to_string(),
            TypedValue::Number(value) => value.to_string(),
            TypedValue::Json(value) => value.to_string(),
            TypedValue::Null => "unknown".into(),
        })
        .unwrap_or_default()
}

pub fn safe(text: &str) -> String {
    omp_render::richtext::sanitize_text(text)
}

// Adapter over the existing semantic Out contract: style information never round-trips through ANSI.
// Only the display projection is bounded; the full session remains in the journal and /inspect.
struct TranscriptSink {
    lines: Vec<Line<'static>>,
    column: usize,
    width: usize,
    bytes: usize,
    theme: Theme,
}
impl TranscriptSink {
    fn new(width: u16, theme: Theme) -> Self {
        Self {
            lines: vec![Line::default()],
            column: 0,
            width: usize::from(width.max(1)),
            bytes: 0,
            theme,
        }
    }
    fn push(&mut self, text: &str, style: Style) -> Result<(), OutError> {
        for grapheme in text.graphemes(true) {
            if self.bytes >= 2 * 1024 * 1024 || self.lines.len() >= 50_000 {
                return Err(OutError::BufferOverflow);
            }
            self.bytes += grapheme.len();
            if grapheme == "\n" {
                self.line_break()?;
                continue;
            }
            let width = grapheme.width();
            if self.column + width > self.width && self.column > 0 {
                self.line_break()?;
            }
            let line = self
                .lines
                .last_mut()
                .expect("editor view always holds at least one line");
            if let Some(last) = line.spans.last_mut().filter(|last| last.style == style) {
                last.content.to_mut().push_str(grapheme);
            } else {
                line.spans.push(Span::styled(grapheme.to_string(), style));
            }
            self.column += width;
        }
        Ok(())
    }
}
impl Out for TranscriptSink {
    fn write_run(&mut self, run: &Run) -> Result<(), OutError> {
        self.push(&run.text, self.theme.run_style(run.style))
    }
    fn line_break(&mut self) -> Result<(), OutError> {
        if self.lines.len() >= 50_000 {
            return Err(OutError::BufferOverflow);
        }
        self.lines.push(Line::default());
        self.column = 0;
        Ok(())
    }
    fn flush(&mut self) -> Result<(), OutError> {
        Ok(())
    }
}

pub struct Transcript {
    pub lines: Vec<Line<'static>>,
    offset: u64,
    width: u16,
    session: String,
    theme: String,
    registry: omp_render::SemanticRegistry,
}
impl Default for Transcript {
    fn default() -> Self {
        Self {
            lines: vec![],
            offset: u64::MAX,
            width: 0,
            session: String::new(),
            theme: String::new(),
            registry: omp_render::SemanticRegistry::new(),
        }
    }
}
impl Transcript {
    pub fn update(&mut self, snapshot: &SessionSnapshot, width: u16) -> usize {
        let theme_name = setting(snapshot, "cl_theme");
        if self.offset == snapshot.offset
            && self.width == width
            && self.session == snapshot.session_id.as_str()
            && self.theme == theme_name
        {
            return 0;
        }
        let prev_len = if self.session == snapshot.session_id.as_str() && self.width == width {
            self.lines.len()
        } else {
            0
        };
        self.offset = snapshot.offset;
        self.width = width;
        self.session = snapshot.session_id.to_string();
        self.theme = theme_name;
        let theme = Theme::from_snapshot(snapshot);
        let mut sink = TranscriptSink::new(width, theme);
        for element in snapshot.get_visible_body() {
            let result = (|| {
                let component = self
                    .registry
                    .render_session_element(snapshot, &element.id)
                    .map_err(OutError::Write)?;
                component.render_to_sink(&mut sink, 0)?;
                sink.line_break()
            })();
            if let Err(error) = result {
                let message = match error {
                    OutError::BufferOverflow => "Display limit reached. Use /inspect or omp2 inspect to read the complete journal.".to_string(),
                    error => format!("Could not render entry: {error}"),
                };
                sink.lines
                    .push(Line::styled(message, Style::default().fg(theme.error)));
                break;
            }
        }
        self.lines = sink.lines;
        self.lines.len().saturating_sub(prev_len)
    }
    pub fn max_scroll(&self, body_height: usize) -> usize {
        self.lines.len().saturating_sub(body_height)
    }
}

pub struct Panel {
    pub title: String,
    pub text: String,
    pub scroll: u16,
}

pub struct DrawState<'a> {
    pub snapshot: &'a SessionSnapshot,
    pub workspace: &'a Path,
    pub editor: &'a Editor,
    pub candidates: &'a [Completion],
    pub selected: usize,
    pub menu_open: bool,
    pub busy: bool,
    pub elapsed: u64,
    pub notice: &'a str,
    pub notice_error: bool,
    pub scroll: usize,
    pub panel: Option<&'a Panel>,
}

fn styled(text: impl Into<String>, color: Color) -> Span<'static> {
    Span::styled(safe(&text.into()), Style::default().fg(color))
}

fn fit(text: &str, width: usize) -> String {
    if text.width() <= width {
        return text.into();
    }
    let mut result = String::new();
    let mut used = 0;
    for grapheme in text.graphemes(true) {
        if used + grapheme.width() >= width {
            break;
        }
        result.push_str(grapheme);
        used += grapheme.width();
    }
    result.push('…');
    result
}

fn welcome(frame: &mut Frame, rect: Rect, state: &DrawState, theme: Theme) {
    if rect.height < 5 {
        return;
    }
    let ascii = setting(state.snapshot, "cl_icon_mode") == "ascii";
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(if ascii {
            BorderType::Plain
        } else {
            BorderType::Rounded
        })
        .border_style(Style::default().fg(theme.border))
        .title(Line::from(vec![
            styled("  omp2 ", theme.text),
            styled(format!("v{}  ", env!("CARGO_PKG_VERSION")), theme.muted),
        ]));
    let inner = block.inner(rect);
    frame.render_widget(block, rect);
    let model = setting(state.snapshot, "ai_model");
    let model = if model.is_empty() {
        "No model selected".into()
    } else {
        model
    };
    let provider = setting(state.snapshot, "ai_provider");
    let left = vec![
        Line::default(),
        Line::styled(
            if state.snapshot.turn_count() == 0 {
                "Ready to build."
            } else {
                "Welcome back."
            },
            Style::default().fg(theme.text).bold(),
        ),
        Line::default(),
        Line::styled(
            if ascii {
                "  O M P  "
            } else {
                "  ▄▄▄▄▄▄▄  "
            },
            Style::default().fg(theme.accent),
        ),
        Line::styled(
            if ascii {
                "  [ 2 ]  "
            } else {
                "    █ █    "
            },
            Style::default().fg(theme.accent),
        ),
        Line::styled(
            if ascii {
                "         "
            } else {
                "    █ ▀▄   "
            },
            Style::default().fg(theme.input),
        ),
        Line::default(),
        Line::from(styled(model.clone(), theme.accent)),
        Line::from(styled(
            if provider.is_empty() {
                "Provider not configured".into()
            } else {
                provider
            },
            theme.muted,
        )),
    ];
    let session = state.snapshot.session_id.to_string();
    let right = vec![
        Line::default(),
        Line::styled(
            "Make something useful.",
            Style::default().fg(theme.text).bold(),
        ),
        Line::from(styled(
            "Ask a question, describe a change, or investigate your code.",
            theme.muted,
        )),
        Line::default(),
        Line::from(vec![
            styled("/           ", theme.accent),
            styled("Browse commands", theme.text),
        ]),
        Line::from(vec![
            styled("/model      ", theme.accent),
            styled("Choose a discovered model", theme.text),
        ]),
        Line::from(vec![
            styled("Ctrl+J      ", theme.accent),
            styled("Add a new line", theme.text),
        ]),
        Line::from(vec![
            styled("Esc         ", theme.accent),
            styled("Close a menu / stop a running turn", theme.text),
        ]),
        Line::default(),
        Line::styled("This session", Style::default().fg(theme.accent).bold()),
        Line::from(styled(
            format!(
                "{}  ·  {} turns  ·  journal @{}",
                &session[..session.len().min(8)],
                state.snapshot.turn_count(),
                state.snapshot.offset
            ),
            theme.muted,
        )),
        Line::from(styled(
            format!(
                "{} active jobs  ·  {} actors",
                state.snapshot.active_jobs().count(),
                state
                    .snapshot
                    .children(state.snapshot.container("actors"))
                    .count()
            ),
            theme.muted,
        )),
    ];
    if inner.width >= 76 && inner.height >= 10 {
        let split = Layout::horizontal([Constraint::Length(29), Constraint::Min(0)]).split(inner);
        frame.render_widget(
            Paragraph::new(left)
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: false }),
            split[0],
        );
        let right_area = Rect {
            x: split[1].x.saturating_add(2),
            width: split[1].width.saturating_sub(3),
            ..split[1]
        };
        frame.render_widget(Paragraph::new(right).wrap(Wrap { trim: false }), right_area);
    } else {
        let compact = vec![
            Line::styled("Ready to build.", Style::default().fg(theme.text).bold()),
            Line::from(styled(model, theme.accent)),
            Line::default(),
            Line::from(styled(
                "Type a message.  / commands  ·  Ctrl+J new line",
                theme.muted,
            )),
        ];
        frame.render_widget(
            Paragraph::new(compact).wrap(Wrap { trim: false }),
            inner.inner(Margin {
                horizontal: 1,
                vertical: 0,
            }),
        );
    }
}

pub fn draw(frame: &mut Frame, state: &mut DrawState, transcript: &mut Transcript) {
    let theme = Theme::from_snapshot(state.snapshot);
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(theme.background).fg(theme.text)),
        area,
    );
    if area.width < 20 || area.height < 8 {
        frame.render_widget(
            Paragraph::new("Terminal too small.\nResize to at least 20 x 8.\nCtrl+C exits.")
                .style(Style::default().fg(theme.muted)),
            area,
        );
        return;
    }
    let content = area.inner(Margin {
        horizontal: 1,
        vertical: 0,
    });
    let input_width = content.width.saturating_sub(2).max(1);
    let input_lines = state.editor.visual_lines(input_width);
    let (cursor_col, cursor_row) = state.editor.visual_cursor(input_width);
    let input_height = (input_lines.len() as u16)
        .clamp(1, 6)
        .min(content.height.saturating_sub(6).max(1));
    let menu_height = if state.menu_open {
        (state.candidates.len().max(1) as u16)
            .min(8)
            .saturating_add(2)
            .min(content.height.saturating_sub(input_height + 5))
    } else {
        0
    };
    let layout = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(input_height + 2),
        Constraint::Length(menu_height),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(content);
    let transcript_area = layout[0];
    let welcome_height = if state.snapshot.turn_count() == 0 && state.scroll == 0 {
        transcript_area
            .height
            .min(if transcript_area.width >= 80 { 16 } else { 7 })
    } else {
        0
    };
    if welcome_height > 0 {
        welcome(
            frame,
            Rect {
                height: welcome_height,
                ..transcript_area
            },
            state,
            theme,
        );
    }
    let body_area = Rect {
        y: transcript_area.y + welcome_height,
        height: transcript_area.height.saturating_sub(welcome_height),
        ..transcript_area
    };
    let added = transcript.update(state.snapshot, body_area.width);
    if state.scroll > 0 && added > 0 {
        state.scroll = state.scroll.saturating_add(added);
    }
    let total = transcript.lines.len();
    let body_height = body_area.height as usize;
    let max_scroll = transcript.max_scroll(body_height);
    state.scroll = state.scroll.min(max_scroll);
    let start = total
        .saturating_sub(body_height)
        .saturating_sub(state.scroll);
    if body_area.height > 0 {
        let lines: Vec<Line<'_>> = transcript
            .lines
            .iter()
            .skip(start)
            .take(body_area.height as usize)
            .map(|line| {
                Line::from(
                    line.spans
                        .iter()
                        .map(|span| Span::styled(span.content.as_ref(), span.style))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        frame.render_widget(Paragraph::new(lines), body_area);
    }
    let message = if state.busy {
        format!(
            "{}  Working · {}s · Esc to stop",
            ["·", "•", "●", "•"][(state.elapsed % 4) as usize],
            state.elapsed
        )
    } else if !state.notice.is_empty() {
        safe(state.notice)
    } else if state.scroll > 0 {
        if total > body_height {
            let line_end = (start + body_height).min(total);
            format!(
                "Transcript paused · lines {}-{} of {} · Ctrl+End returns to latest",
                start + 1,
                line_end,
                total
            )
        } else {
            "Transcript paused · Ctrl+End returns to latest".into()
        }
    } else if area.width < 90 {
        "Enter send · Ctrl+J newline · / commands".into()
    } else {
        "Enter send  ·  Ctrl+J new line  ·  / commands  ·  Wheel/PgUp scroll".into()
    };
    frame.render_widget(
        Paragraph::new(message).style(Style::default().fg(if state.notice_error {
            theme.error
        } else if state.busy {
            theme.accent
        } else {
            theme.muted
        })),
        layout[1],
    );
    let input_block = Block::default()
        .borders(Borders::TOP | Borders::BOTTOM)
        .border_style(Style::default().fg(if state.busy {
            theme.border
        } else {
            theme.input
        }));
    let input_area = input_block.inner(layout[2]);
    frame.render_widget(input_block, layout[2]);
    frame.render_widget(
        Paragraph::new(">").style(Style::default().fg(theme.input)),
        Rect {
            width: 2,
            ..input_area
        },
    );
    let edit_area = Rect {
        x: input_area.x + 2,
        width: input_area.width.saturating_sub(2),
        ..input_area
    };
    let input_scroll = cursor_row.saturating_sub(input_height.saturating_sub(1));
    if state.editor.text.is_empty() {
        frame.render_widget(
            Paragraph::new(if state.busy {
                "Compose your next message…"
            } else {
                "Ask omp2 to work on your code…"
            })
            .style(Style::default().fg(theme.muted)),
            edit_area,
        );
    } else {
        let lines: Vec<Line> = input_lines
            .iter()
            .skip(input_scroll as usize)
            .take(input_height as usize)
            .map(|line| Line::raw(line.as_str()))
            .collect();
        frame.render_widget(Paragraph::new(lines), edit_area);
    }
    if state.panel.is_none() {
        frame.set_cursor_position((
            edit_area.x + cursor_col.min(edit_area.width.saturating_sub(1)),
            edit_area.y
                + cursor_row
                    .saturating_sub(input_scroll)
                    .min(edit_area.height.saturating_sub(1)),
        ));
    }
    if state.menu_open && layout[3].height > 0 {
        let menu = layout[3];
        let visible = menu.height.saturating_sub(2) as usize;
        let selected = state.selected.min(state.candidates.len().saturating_sub(1));
        let first = selected.saturating_sub(visible.saturating_sub(1));
        if state.candidates.is_empty() {
            frame.render_widget(
                Paragraph::new("  No matching command. Keep typing, or Esc to close.")
                    .style(Style::default().fg(theme.muted)),
                menu,
            );
        } else {
            for (i, candidate) in state
                .candidates
                .iter()
                .enumerate()
                .skip(first)
                .take(visible)
            {
                let row = Rect {
                    y: menu.y + (i - first) as u16,
                    height: 1,
                    ..menu
                };
                let active = i == selected;
                let label_width = (menu.width / 3).clamp(12, 34) as usize;
                let label = fit(&safe(&candidate.label), label_width);
                let padding = label_width.saturating_sub(label.width());
                let spans = vec![
                    styled(if active { "› " } else { "  " }, theme.accent),
                    styled(label, if active { theme.accent } else { theme.text }),
                    Span::raw(" ".repeat(padding + 2)),
                    styled(
                        &candidate.description,
                        if active { theme.text } else { theme.muted },
                    ),
                ];
                frame.render_widget(
                    Paragraph::new(Line::from(spans)).style(Style::default().bg(if active {
                        theme.selected
                    } else {
                        theme.background
                    })),
                    row,
                );
            }
            let hint = if menu.width < 90 {
                format!(
                    "  ↑↓ select · Tab fill · Enter · Esc    {}/{}",
                    selected + 1,
                    state.candidates.len()
                )
            } else {
                format!(
                    "  ↑↓ select · Tab complete · Enter accept · Esc close    {}/{}",
                    selected + 1,
                    state.candidates.len()
                )
            };
            frame.render_widget(
                Paragraph::new(hint).style(Style::default().fg(theme.muted)),
                Rect {
                    y: menu.bottom().saturating_sub(1),
                    height: 1,
                    ..menu
                },
            );
        }
    }
    let model = setting(state.snapshot, "ai_model");
    let model = if model.is_empty() {
        "No model".into()
    } else {
        model
    };
    let path = state.workspace.to_string_lossy();
    let path = path.strip_prefix("\\\\?\\").unwrap_or(&path);
    let path = fit(
        path,
        content.width.saturating_sub(model.width() as u16 + 16) as usize,
    );
    let status = Line::from(vec![
        styled("omp2  ·  ", theme.muted),
        styled(model, theme.accent),
        styled(gateway_suffix(state.snapshot), theme.muted),
        styled("  ·  ", theme.border),
        styled(path, theme.text),
    ]);
    frame.render_widget(Paragraph::new(status), layout[4]);
    let context = setting(state.snapshot, "ai_context_length");
    let footer = format!(
        "{}  ·  {} turns  ·  context {}{}  ·  /help",
        state.snapshot.current_branch(),
        state.snapshot.turn_count(),
        if context.is_empty() {
            "unknown"
        } else {
            &context
        },
        effort_indicator(state.snapshot)
    );
    frame.render_widget(
        Paragraph::new(safe(&footer)).style(Style::default().fg(theme.muted)),
        layout[5],
    );
    if let Some(panel) = state.panel {
        let panel_area = area.inner(Margin {
            horizontal: if area.width > 80 { 6 } else { 1 },
            vertical: 1,
        });
        frame.render_widget(Clear, panel_area);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme.accent))
            .title(format!(" {} ", safe(&panel.title)))
            .title_bottom(" Esc close · Wheel/PgUp/PgDn scroll ")
            .style(Style::default().bg(theme.background).fg(theme.text));
        frame.render_widget(
            Paragraph::new(safe(&panel.text))
                .wrap(Wrap { trim: false })
                .scroll((panel.scroll, 0))
                .block(block),
            panel_area,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scroll_bounds_and_anchoring_calculation() {
        let transcript = Transcript {
            lines: (0..100).map(|i| Line::raw(format!("line {i}"))).collect(),
            ..Default::default()
        };
        let body_height = 20;
        let max_scroll = transcript.max_scroll(body_height);
        assert_eq!(max_scroll, 80);

        // When scroll is clamped, it cannot exceed max_scroll
        let excessive_scroll = 150usize;
        let clamped = excessive_scroll.min(max_scroll);
        assert_eq!(clamped, 80);
        let start = transcript.lines.len() - body_height - clamped;
        assert_eq!(start, 0);

        // Anchoring: if 5 new lines are added while scrolled up by 30 (viewing lines 50..70)
        let scroll = 30;
        let start_before = transcript.lines.len() - body_height - scroll;
        assert_eq!(start_before, 50);

        let added = 5;
        let new_total = transcript.lines.len() + added;
        let new_scroll = scroll + added;
        let start_after = new_total - body_height - new_scroll;
        assert_eq!(start_after, start_before);
    }

    #[test]
    fn the_footer_says_which_effort_is_in_force() {
        fn snapshot_with(thinking: &str, levels: serde_json::Value, default: Option<&str>) -> SessionSnapshot {
            let mut snapshot = SessionSnapshot::empty(omp_types::SessionId::new("test-effort").unwrap());
            let convars = snapshot.container("convars").clone();
            let capabilities = snapshot.container("capabilities").clone();
            let mut patch = |ops| {
                let base = snapshot.offset;
                omp_state::apply_patch(
                    &mut snapshot,
                    &omp_types::Patch {
                        base_offset: omp_types::JournalOffset(base),
                        result_offset: omp_types::JournalOffset(base + 1),
                        by: omp_types::ActorId::new("test-owner").unwrap().into(),
                        reason: "test".into(),
                        ops,
                    },
                )
                .unwrap();
            };
            patch(vec![
                omp_types::PatchOp::SetAttribute {
                    element: convars.clone(),
                    name: "ai_thinking".into(),
                    value: TypedValue::String(thinking.into()),
                },
                omp_types::PatchOp::SetAttribute {
                    element: convars,
                    name: "ai_thinking_levels".into(),
                    value: TypedValue::Json(levels),
                },
            ]);
            patch(vec![omp_types::PatchOp::SetAttribute {
                element: capabilities,
                name: "provider_metadata".into(),
                value: TypedValue::Json(serde_json::json!({
                    "models": [],
                    "active_model": { "id": "m", "thinking_default": default }
                })),
            }]);
            snapshot
        }

        // An explicit level is shown as itself.
        let snapshot = snapshot_with("high", serde_json::json!(["low", "medium", "high"]), Some("medium"));
        assert_eq!(effort_indicator(&snapshot), "  ·  effort high");

        // "auto" still shows what the provider would do.
        let snapshot = snapshot_with("auto", serde_json::json!(["low", "medium"]), Some("low"));
        assert_eq!(effort_indicator(&snapshot), "  ·  effort auto (low)");

        // Reasoning disabled is explicit.
        let snapshot = snapshot_with("off", serde_json::json!(["low"]), Some("low"));
        assert_eq!(effort_indicator(&snapshot), "  ·  effort off");

        // A model that advertises nothing gets no indicator rather than a guess.
        let snapshot = snapshot_with("auto", serde_json::Value::Null, None);
        assert_eq!(effort_indicator(&snapshot), "");
    }
}
