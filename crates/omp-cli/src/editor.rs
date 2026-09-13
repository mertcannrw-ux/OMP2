use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use omp_render::richtext::sanitize_text;
use omp_state::SessionSnapshot;
use omp_types::TypedValue;
use std::collections::VecDeque;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

const MAX_HISTORY_ENTRIES: usize = 200;
const MAX_BUFFER_BYTES: usize = 65536;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completion {
    pub label: String,
    pub description: String,
    pub insert: String,
    pub execute: bool,
}

pub struct Editor {
    pub text: String,
    pub cursor: usize,
    history: VecDeque<String>,
    history_index: Option<usize>,
    draft: String,
}

impl Default for Editor {
    fn default() -> Self {
        Self::new()
    }
}

impl Editor {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            history: VecDeque::with_capacity(MAX_HISTORY_ENTRIES),
            history_index: None,
            draft: String::new(),
        }
    }

    pub fn set(&mut self, text: String) {
        let sanitized = sanitize_and_normalize(&text);
        let bounded = if sanitized.len() > MAX_BUFFER_BYTES {
            truncate_to_grapheme_budget(&sanitized, MAX_BUFFER_BYTES).to_string()
        } else {
            sanitized
        };
        self.text = bounded;
        self.cursor = self.text.len();
        self.clamp_cursor();
        self.history_index = None;
    }

    pub fn insert(&mut self, input: &str) -> bool {
        if input.is_empty() {
            return false;
        }

        let sanitized = sanitize_and_normalize(input);
        if sanitized.is_empty() {
            return false;
        }

        // Reject entire oversized paste rather than silently truncating
        if self.text.len() + sanitized.len() > MAX_BUFFER_BYTES {
            return false;
        }

        self.clamp_cursor();
        self.text.insert_str(self.cursor, &sanitized);
        self.cursor += sanitized.len();
        // Ensure cursor is clamped forward if graphemes merged to the right
        self.clamp_cursor_forward();
        self.history_index = None;
        true
    }

    /// Clears the input text buffer and resets cursor and draft recall state.
    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.history_index = None;
        self.draft.clear();
    }

    /// Takes the current input text if nonblank, records it in bounded history,
    /// and resets the editor state. Returns None if the input was empty or whitespace-only.
    pub fn submit(&mut self) -> Option<String> {
        if self.text.trim().is_empty() {
            return None;
        }

        let input = std::mem::take(&mut self.text);
        self.cursor = 0;
        self.history_index = None;
        self.draft.clear();

        // Deduplicate against the immediate previous history entry
        if self.history.back().map(|s| s.as_str()) != Some(&input) {
            if self.history.len() >= MAX_HISTORY_ENTRIES {
                self.history.pop_front();
            }
            self.history.push_back(input.clone());
        }

        Some(input)
    }

    /// Handles keyboard events for editing and navigation only.
    ///
    /// Supported actions:
    /// - Character typing (non-control)
    /// - Left / Right arrows (single grapheme)
    /// - Ctrl+Left / Ctrl+Right (word boundary)
    /// - Home / End (line boundary)
    /// - Ctrl+A (start of line) / Ctrl+E (end of line)
    /// - Backspace (delete previous grapheme) / Delete (delete current grapheme)
    /// - Ctrl+K (kill to end of line) / Ctrl+U (kill to start of line)
    /// - Ctrl+W (delete previous word)
    /// - Ctrl+J / Shift+Enter (newline insertion)
    /// - Up / Down (multiline movement first, then history navigation)
    ///
    /// Does not handle Enter submission or Esc cancellation (returns `false`).
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        // Plain Enter and Esc are reserved for the higher-level event loop.
        if key.code == KeyCode::Enter && key.modifiers.is_empty() {
            return false;
        }
        if key.code == KeyCode::Esc {
            return false;
        }

        match key.code {
            // Newline insertions
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.insert("\n")
            }
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => self.insert("\n"),

            // Emacs / Readline line controls
            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.move_to_line_start()
            }
            KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.move_to_line_end()
            }
            KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.kill_to_line_end()
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.kill_to_line_start()
            }
            KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.delete_word_backward()
            }

            // Standard navigation
            KeyCode::Left => {
                if key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                {
                    self.move_word_backward()
                } else {
                    self.move_grapheme_backward()
                }
            }
            KeyCode::Right => {
                if key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                {
                    self.move_word_forward()
                } else {
                    self.move_grapheme_forward()
                }
            }
            KeyCode::Home => self.move_to_line_start(),
            KeyCode::End => self.move_to_line_end(),

            // Deletions
            KeyCode::Backspace => self.delete_grapheme_backward(),
            KeyCode::Delete => self.delete_grapheme_forward(),

            // Multiline then history
            KeyCode::Up => self.handle_up(),
            KeyCode::Down => self.handle_down(),

            // Character typing: plain, Shift, or AltGr (Ctrl+Alt on Windows Turkish/European keyboards)
            KeyCode::Char(c)
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    || key.modifiers.contains(KeyModifiers::ALT) =>
            {
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                self.insert(s)
            }
            _ => false,
        }
    }

    /// Computes visual lines after soft wrapping at `width` columns without splitting Unicode graphemes.
    /// Preserves explicit newlines and returns at least one line (even when buffer is empty).
    pub fn visual_lines(&self, width: u16) -> Vec<String> {
        let width = width.max(1) as usize;
        if self.text.is_empty() {
            return vec![String::new()];
        }

        let mut lines = Vec::new();
        let logical_lines: Vec<&str> = self.text.split('\n').collect();
        let num_logical = logical_lines.len();

        for (idx, logical) in logical_lines.iter().enumerate() {
            if logical.is_empty() {
                lines.push(String::new());
                continue;
            }

            let mut current_line = String::new();
            let mut current_width = 0;

            for grapheme in logical.graphemes(true) {
                let gw = UnicodeWidthStr::width(grapheme);
                if current_width + gw > width && current_width > 0 {
                    lines.push(current_line);
                    current_line = String::new();
                    current_width = 0;
                }
                current_line.push_str(grapheme);
                current_width += gw;
            }

            let is_final = idx + 1 == num_logical;
            let exact_width = current_width == width;
            lines.push(current_line);

            if is_final && exact_width {
                lines.push(String::new());
            }
        }

        lines
    }

    /// Computes the visual cursor (column, row) position inside a viewport of `width` columns.
    /// When the cursor sits at the exact line width boundary, it advances to column 0 on the next row.
    pub fn visual_cursor(&self, width: u16) -> (u16, u16) {
        let width = width.max(1) as usize;
        let target = self.cursor.min(self.text.len());

        let mut current_byte = 0;
        let mut row: usize = 0;
        let mut col: usize = 0;

        let logical_lines: Vec<&str> = self.text.split('\n').collect();
        let num_logical = logical_lines.len();

        for (logical_idx, logical) in logical_lines.into_iter().enumerate() {
            if current_byte == target {
                if col >= width {
                    return (0, (row + 1) as u16);
                }
                return (col as u16, row as u16);
            }

            let mut line_width = 0;

            for g in logical.graphemes(true) {
                let gw = UnicodeWidthStr::width(g);
                if line_width + gw > width && line_width > 0 {
                    row += 1;
                    col = 0;
                    line_width = 0;
                }

                if current_byte == target {
                    return (col as u16, row as u16);
                }

                col += gw;
                line_width += gw;
                current_byte += g.len();
            }

            // Target byte sits at the end of the logical line (before the newline)
            if current_byte == target {
                if col >= width {
                    return (0, (row + 1) as u16);
                }
                return (col as u16, row as u16);
            }

            // Step over the newline character if another logical line follows
            if logical_idx + 1 < num_logical {
                current_byte += 1; // account for '\n'
                row += 1;
                col = 0;
                if current_byte == target {
                    return (col as u16, row as u16);
                }
            }
        }

        if col >= width {
            (0, (row + 1) as u16)
        } else {
            (col as u16, row as u16)
        }
    }

    // -------------------------------------------------------------------------
    // Internal cursor & editing primitives
    // -------------------------------------------------------------------------

    /// Snaps `self.cursor` to the closest valid Unicode grapheme boundary.
    fn clamp_cursor(&mut self) {
        if self.cursor >= self.text.len() {
            self.cursor = self.text.len();
            return;
        }

        let mut nearest = 0;
        for (idx, _) in self.text.grapheme_indices(true) {
            if idx == self.cursor {
                return;
            }
            if idx > self.cursor {
                break;
            }
            nearest = idx;
        }
        self.cursor = nearest;
    }

    /// Clamps cursor forward to the next grapheme cluster boundary if graphemes merged to the right.
    fn clamp_cursor_forward(&mut self) {
        if self.cursor >= self.text.len() {
            self.cursor = self.text.len();
            return;
        }
        for (idx, g) in self.text.grapheme_indices(true) {
            if idx == self.cursor {
                return;
            }
            let end = idx + g.len();
            if self.cursor < end {
                self.cursor = end;
                return;
            }
        }
        self.cursor = self.text.len();
    }

    fn move_grapheme_backward(&mut self) -> bool {
        self.clamp_cursor();
        if self.cursor == 0 {
            return false;
        }
        let mut prev = 0;
        for (idx, _) in self.text.grapheme_indices(true) {
            if idx >= self.cursor {
                break;
            }
            prev = idx;
        }
        self.cursor = prev;
        true
    }

    fn move_grapheme_forward(&mut self) -> bool {
        self.clamp_cursor();
        if self.cursor >= self.text.len() {
            return false;
        }
        for (idx, g) in self.text.grapheme_indices(true) {
            if idx > self.cursor {
                self.cursor = idx;
                return true;
            }
            if idx == self.cursor {
                self.cursor = idx + g.len();
                return true;
            }
        }
        self.cursor = self.text.len();
        true
    }

    fn move_word_backward(&mut self) -> bool {
        self.clamp_cursor();
        if self.cursor == 0 {
            return false;
        }

        let before = &self.text[..self.cursor];
        let mut target = 0;
        let mut in_word = false;

        for (idx, ch) in before.char_indices().rev() {
            if ch.is_whitespace() {
                if in_word {
                    target = idx + ch.len_utf8();
                    break;
                }
            } else {
                in_word = true;
            }
        }

        if self.cursor != target {
            self.cursor = target;
            self.clamp_cursor();
            true
        } else {
            false
        }
    }

    fn move_word_forward(&mut self) -> bool {
        self.clamp_cursor();
        if self.cursor >= self.text.len() {
            return false;
        }

        let after = &self.text[self.cursor..];
        let mut target = self.text.len();
        let mut skipped_ws = false;

        for (offset, ch) in after.char_indices() {
            if ch.is_whitespace() {
                if skipped_ws {
                    target = self.cursor + offset;
                    break;
                }
            } else {
                skipped_ws = true;
            }
        }

        if self.cursor != target {
            self.cursor = target;
            self.clamp_cursor();
            true
        } else {
            false
        }
    }

    fn move_to_line_start(&mut self) -> bool {
        self.clamp_cursor();
        let before = &self.text[..self.cursor];
        let new_cursor = before.rfind('\n').map(|idx| idx + 1).unwrap_or(0);
        if self.cursor != new_cursor {
            self.cursor = new_cursor;
            true
        } else {
            false
        }
    }

    fn move_to_line_end(&mut self) -> bool {
        self.clamp_cursor();
        let after = &self.text[self.cursor..];
        let new_cursor = after
            .find('\n')
            .map(|idx| self.cursor + idx)
            .unwrap_or(self.text.len());
        if self.cursor != new_cursor {
            self.cursor = new_cursor;
            true
        } else {
            false
        }
    }

    fn delete_grapheme_backward(&mut self) -> bool {
        self.clamp_cursor();
        if self.cursor == 0 {
            return false;
        }
        let mut prev = 0;
        for (idx, _) in self.text.grapheme_indices(true) {
            if idx >= self.cursor {
                break;
            }
            prev = idx;
        }
        self.text.drain(prev..self.cursor);
        self.cursor = prev;
        self.history_index = None;
        true
    }

    fn delete_grapheme_forward(&mut self) -> bool {
        self.clamp_cursor();
        if self.cursor >= self.text.len() {
            return false;
        }
        let len = self.text[self.cursor..]
            .graphemes(true)
            .next()
            .map(|g| g.len())
            .unwrap_or(0);
        if len > 0 {
            self.text.drain(self.cursor..self.cursor + len);
            self.history_index = None;
            true
        } else {
            false
        }
    }

    fn kill_to_line_start(&mut self) -> bool {
        self.clamp_cursor();
        let before = &self.text[..self.cursor];
        let line_start = before.rfind('\n').map(|idx| idx + 1).unwrap_or(0);

        if self.cursor > line_start {
            self.text.drain(line_start..self.cursor);
            self.cursor = line_start;
            self.history_index = None;
            true
        } else if line_start > 0 && self.cursor == line_start {
            // Delete the preceding newline to join with the previous line
            let del_pos = line_start - 1;
            self.text.drain(del_pos..del_pos + 1);
            self.cursor = del_pos;
            self.history_index = None;
            true
        } else {
            false
        }
    }

    fn kill_to_line_end(&mut self) -> bool {
        self.clamp_cursor();
        if self.cursor >= self.text.len() {
            return false;
        }
        let after = &self.text[self.cursor..];
        let end_offset = after.find('\n').unwrap_or(after.len());

        if end_offset > 0 {
            self.text.drain(self.cursor..self.cursor + end_offset);
            self.history_index = None;
            true
        } else {
            // Cursor is directly on '\n', delete the newline
            self.text.drain(self.cursor..self.cursor + 1);
            self.history_index = None;
            true
        }
    }

    fn delete_word_backward(&mut self) -> bool {
        self.clamp_cursor();
        if self.cursor == 0 {
            return false;
        }

        let before = &self.text[..self.cursor];
        let mut target = 0;
        let mut in_word = false;

        for (idx, ch) in before.char_indices().rev() {
            if ch.is_whitespace() {
                if in_word {
                    target = idx + ch.len_utf8();
                    break;
                }
            } else {
                in_word = true;
            }
        }

        if self.cursor > target {
            self.text.drain(target..self.cursor);
            self.cursor = target;
            self.history_index = None;
            true
        } else {
            false
        }
    }

    // -------------------------------------------------------------------------
    // Multiline & History Navigation
    // -------------------------------------------------------------------------

    fn handle_up(&mut self) -> bool {
        self.clamp_cursor();

        // Check if there is a line above the cursor
        let before = &self.text[..self.cursor];
        if let Some(prev_nl) = before.rfind('\n') {
            // Cursor is not on the first line. Move to previous line at the same grapheme column.
            let current_line_start = prev_nl + 1;
            let current_col = self.text[current_line_start..self.cursor]
                .graphemes(true)
                .count();

            let line_above_end = prev_nl;
            let line_above_start = self.text[..line_above_end]
                .rfind('\n')
                .map(|idx| idx + 1)
                .unwrap_or(0);

            let line_above = &self.text[line_above_start..line_above_end];
            let mut target_cursor = line_above_start;

            for (g_idx, g) in line_above.graphemes(true).enumerate() {
                if g_idx >= current_col {
                    break;
                }
                target_cursor += g.len();
            }

            self.cursor = target_cursor;
            return true;
        }

        // On the first line: trigger history navigation older
        if self.history.is_empty() {
            return false;
        }

        match self.history_index {
            None => {
                // Save draft and recall newest history entry
                self.draft = self.text.clone();
                let last_idx = self.history.len() - 1;
                self.history_index = Some(last_idx);
                self.text = self.history[last_idx].clone();
                self.cursor = self.text.len();
                true
            }
            Some(idx) => {
                if idx > 0 {
                    let next_idx = idx - 1;
                    self.history_index = Some(next_idx);
                    self.text = self.history[next_idx].clone();
                    self.cursor = self.text.len();
                    true
                } else {
                    false
                }
            }
        }
    }

    fn handle_down(&mut self) -> bool {
        self.clamp_cursor();

        // Check if there is a line below the cursor
        let after = &self.text[self.cursor..];
        if let Some(next_nl_offset) = after.find('\n') {
            // Cursor is not on the last line. Move to next line at the same grapheme column.
            let next_line_start = self.cursor + next_nl_offset + 1;
            let before = &self.text[..self.cursor];
            let current_line_start = before.rfind('\n').map(|idx| idx + 1).unwrap_or(0);
            let current_col = self.text[current_line_start..self.cursor]
                .graphemes(true)
                .count();

            let next_line_end = self.text[next_line_start..]
                .find('\n')
                .map(|idx| next_line_start + idx)
                .unwrap_or(self.text.len());

            let next_line = &self.text[next_line_start..next_line_end];
            let mut target_cursor = next_line_start;

            for (g_idx, g) in next_line.graphemes(true).enumerate() {
                if g_idx >= current_col {
                    break;
                }
                target_cursor += g.len();
            }

            self.cursor = target_cursor;
            return true;
        }

        // On the last line: navigate history newer or restore draft
        match self.history_index {
            Some(idx) => {
                if idx + 1 < self.history.len() {
                    let next_idx = idx + 1;
                    self.history_index = Some(next_idx);
                    self.text = self.history[next_idx].clone();
                    self.cursor = self.text.len();
                } else {
                    // Past the newest history entry: restore draft
                    self.history_index = None;
                    self.text = std::mem::take(&mut self.draft);
                    self.cursor = self.text.len();
                }
                true
            }
            None => false,
        }
    }
}

// -----------------------------------------------------------------------------
// Sanitization and Grapheme Helpers
// -----------------------------------------------------------------------------

/// Sanitizes ANSI escape sequences and C0/C1 control characters using authoritative
/// `omp_render::richtext::sanitize_text`, normalizes CRLF/CR to LF, and expands tabs to 4 spaces.
fn sanitize_and_normalize(input: &str) -> String {
    let mut normalized = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\r' {
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
            normalized.push('\n');
        } else if c == '\t' {
            normalized.push_str("    ");
        } else {
            normalized.push(c);
        }
    }

    sanitize_text(&normalized)
}

/// Truncates string slice to at most `max_bytes` without slicing through grapheme clusters.
fn truncate_to_grapheme_budget(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut boundary = 0;
    for (idx, g) in s.grapheme_indices(true) {
        if idx + g.len() > max_bytes {
            break;
        }
        boundary = idx + g.len();
    }
    &s[..boundary]
}

// -----------------------------------------------------------------------------
// Slash Palette Autocomplete
// -----------------------------------------------------------------------------

/// Computes palette completion suggestions for the given input text based on session state.
///
/// Rules:
/// - Matches only inputs starting with '/'.
/// - Case-insensitive prefix and contains filtering.
/// - When input begins with `"/provider select "`, offers advertised model IDs discovered
///   from `capabilities.provider_metadata.Json.models` without exposing sensitive keys.
/// - Includes declared convars and `alias:` keys from the snapshot.
/// - Unknown slash commands produce an empty vector and remain submittable as typed.
pub fn completions(text: &str, snapshot: &SessionSnapshot) -> Vec<Completion> {
    if !text.starts_with('/') {
        return Vec::new();
    }

    // "/effort <filter>" offers the levels the active model advertises.
    if let Some(level_query) = text.strip_prefix("/effort ") {
        return effort_completions(level_query, snapshot);
    }
    if text == "/effort" {
        return vec![Completion {
            label: "/effort".into(),
            description: "Show or set reasoning effort for the active model".into(),
            insert: "/effort ".into(),
            execute: true,
        }];
    }

    // Special context: "/provider select <filter>" offers discovered models exclusively
    if let Some(model_query) = text.strip_prefix("/provider select ") {
        return model_completions(model_query, snapshot);
    }

    // Do not open a blank menu for completed commands with arguments (Main handles menu logic)
    if text.contains(' ') && !text.starts_with("/provider") {
        return Vec::new();
    }
    let query = text.to_lowercase();
    let mut candidates = base_command_roster();

    // Dynamically include aliases and declared convars from the authoritative snapshot
    populate_snapshot_completions(&mut candidates, snapshot);

    // Filter candidates matching the user's slash input (prefix matches first, then contains)
    let mut exact_matches = Vec::new();
    let mut contains_matches = Vec::new();

    for c in candidates {
        let label_lower = c.label.to_lowercase();
        let insert_lower = c.insert.to_lowercase();

        if label_lower.starts_with(&query) || insert_lower.starts_with(&query) {
            exact_matches.push(c);
        } else if label_lower.contains(&query) || insert_lower.contains(&query) {
            contains_matches.push(c);
        }
    }

    exact_matches.extend(contains_matches);
    exact_matches
}

/// Base permanent slash command catalog matching the host and local CLI capabilities.
fn base_command_roster() -> Vec<Completion> {
    vec![
        Completion {
            label: "/help".into(),
            description: "Display built-in commands, keybindings, and help overview".into(),
            insert: "/help".into(),
            execute: true,
        },
        Completion {
            label: "/new".into(),
            description: "Start a clean session; preserve the previous chat and current settings"
                .into(),
            insert: "/new".into(),
            execute: true,
        },
        Completion {
            label: "/model".into(),
            description: "Switch active inference model (expands to /provider select)".into(),
            insert: "/provider select ".into(),
            execute: false,
        },
        Completion {
            label: "/provider".into(),
            description: "Show active inference provider status, endpoint, and advertised catalog"
                .into(),
            insert: "/provider".into(),
            execute: true,
        },
        Completion {
            label: "/provider refresh".into(),
            description: "Refresh provider catalog and advertised model capabilities from endpoint"
                .into(),
            insert: "/provider refresh".into(),
            execute: true,
        },
        Completion {
            label: "/provider select".into(),
            description: "Select active model from advertised catalog (/provider select <id>)"
                .into(),
            insert: "/provider select ".into(),
            execute: false,
        },
        Completion {
            label: "/settings".into(),
            description: "Inspect session configuration variables and display flags".into(),
            insert: "/settings".into(),
            execute: true,
        },
        Completion {
            label: "/status".into(),
            description: "Display active model, provider endpoint, and session status".into(),
            insert: "/status".into(),
            execute: true,
        },
        Completion {
            label: "/inspect".into(),
            description: "Inspect authoritative session DOM snapshot and element hierarchy".into(),
            insert: "/inspect".into(),
            execute: true,
        },
        Completion {
            label: "/actors".into(),
            description: "List active controllers, peers, and background subagent actors".into(),
            insert: "/actors".into(),
            execute: true,
        },
        Completion {
            label: "/jobs".into(),
            description: "List active background jobs, shell runners, and dev servers".into(),
            insert: "/jobs".into(),
            execute: true,
        },
        Completion {
            label: "/effort".into(),
            description: "Reasoning effort: show it, or set a level the model advertises".into(),
            insert: "/effort ".into(),
            execute: true,
        },
        Completion {
            label: "/thinking".into(),
            description:
                "Toggle reasoning block *visibility* in the transcript (cl_showthinking); use /effort to change how hard the model thinks".into(),
            insert: "/thinking".into(),
            execute: true,
        },
        Completion {
            label: "/fork".into(),
            description: "Fork a new branch from a prior journal offset (requires offset)".into(),
            insert: "/fork ".into(),
            execute: false,
        },
        Completion {
            label: "/force".into(),
            description: "Force next assistant turn to execute specified tool (requires tool name)"
                .into(),
            insert: "/force ".into(),
            execute: false,
        },
        Completion {
            label: "/tool".into(),
            description: "Directly invoke a tool with JSON arguments (requires name JSON)".into(),
            insert: "/tool ".into(),
            execute: false,
        },
        Completion {
            label: "/dyn".into(),
            description: "Discover dynamic tools and subcommands (--q, namespace/action)".into(),
            insert: "/dyn ".into(),
            execute: false,
        },
        Completion {
            label: "/cancel".into(),
            description: "Cancel a running background job or subagent by ID (requires job ID)"
                .into(),
            insert: "/cancel ".into(),
            execute: false,
        },
        Completion {
            label: "/exit".into(),
            description: "Gracefully shutdown session and terminate host process".into(),
            insert: "/exit".into(),
            execute: true,
        },
        Completion {
            label: "/detach".into(),
            description: "Disconnect client terminal interface and leave host running".into(),
            insert: "/detach".into(),
            execute: true,
        },
    ]
}

/// Completion candidates for reasoning effort: the levels the active model
/// advertises, plus the two control values that always apply.
fn effort_completions(query: &str, snapshot: &SessionSnapshot) -> Vec<Completion> {
    let query_lower = query.trim().to_lowercase();
    let advertised: Vec<String> = match snapshot
        .element(snapshot.container("convars"))
        .and_then(|node| node.attributes.get("ai_thinking_levels"))
    {
        Some(TypedValue::Json(value)) => value
            .as_array()
            .map(|levels| {
                levels
                    .iter()
                    .filter_map(|level| level.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    };
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
    let current = crate::view::setting(snapshot, "ai_thinking");

    let mut entries: Vec<(String, String)> = advertised
        .iter()
        .map(|level| {
            let mut description = format!("Model effort level '{level}'");
            if default.as_deref() == Some(level.as_str()) {
                description.push_str(" · provider default");
            }
            (level.clone(), description)
        })
        .collect();
    entries.push(("auto".into(), "Leave effort to the provider".into()));
    entries.push(("off".into(), "Disable reasoning entirely".into()));

    entries
        .into_iter()
        .filter(|(level, description)| {
            query_lower.is_empty()
                || level.to_lowercase().contains(&query_lower)
                || description.to_lowercase().contains(&query_lower)
        })
        .map(|(level, description)| {
            let marker = if current.eq_ignore_ascii_case(&level) {
                " · current"
            } else {
                ""
            };
            Completion {
                label: format!("/effort {level}"),
                description: format!("{description}{marker}"),
                insert: format!("/effort {level}"),
                execute: true,
            }
        })
        .collect()
}

/// Generates completion candidates for advertised models under
/// `"/provider select <query>"`.
///
/// The list covers every recorded provider, not just the active one: each entry
/// names the provider it came from, and selecting a model that belongs to
/// another provider switches to it first.
fn model_completions(query: &str, snapshot: &SessionSnapshot) -> Vec<Completion> {
    let query_lower = query.trim().to_lowercase();
    let records = provider_records(snapshot);
    let adapter = crate::view::setting(snapshot, "ai_provider");
    let endpoint = crate::view::setting(snapshot, "ai_endpoint");
    let registered: Vec<omp_types::ProviderRecord> = records
        .iter()
        .map(|(record, _)| record.clone())
        .collect();
    let active = omp_types::label_for_endpoint(&registered, &adapter, &endpoint);
    let mut results = Vec::new();

    // The active provider's catalog is the authoritative one: it is what the
    // session's limits, compaction budget and requests are built from.
    let caps_id = snapshot.container("capabilities");
    let active_models = snapshot
        .element(caps_id)
        .and_then(|node| node.attributes.get("provider_metadata"))
        .and_then(|value| match value {
            TypedValue::Json(meta) => meta.get("models").and_then(|m| m.as_array()).cloned(),
            _ => None,
        })
        .unwrap_or_default();
    for model in &active_models {
        let Some(id) = model.get("id").and_then(|value| value.as_str()) else {
            continue;
        };
        if !matches_query(&query_lower, id, &active, None) {
            continue;
        }
        results.push(Completion {
            label: format!("/provider select {id}"),
            description: format!("{active} · {}{}", describe_model(model), " · active provider"),
            insert: format!("/provider select {id}"),
            execute: true,
        });
    }

    // Other recorded providers: cached catalogs, annotated with their provider,
    // and selecting one switches the provider before selecting the model.
    for (record, models) in &records {
        if record.matches_endpoint(&adapter, &endpoint) {
            continue;
        }
        for model in models {
            let Some(id) = model.get("id").and_then(|value| value.as_str()) else {
                continue;
            };
            if !matches_query(&query_lower, id, &record.name, Some(&record.name)) {
                continue;
            }
            results.push(Completion {
                label: format!("/provider use {}; /provider select {id}", record.name),
                description: format!(
                    "{} · {} · switches provider",
                    record.name,
                    describe_model(model)
                ),
                insert: format!("/provider use {}; /provider select {id}", record.name),
                execute: true,
            });
        }
    }

    results
}

/// True when `id` or its provider matches what the user typed.
fn matches_query(query: &str, id: &str, provider: &str, provider_filter: Option<&str>) -> bool {
    if query.is_empty() {
        return true;
    }
    let id_lower = id.to_lowercase();
    let provider_lower = provider.to_lowercase();
    let extra = provider_filter.map(str::to_lowercase).unwrap_or_default();
    id_lower.contains(query)
        || query.contains(&id_lower)
        || provider_lower.contains(query)
        || (!extra.is_empty() && extra.contains(query))
}

/// Context and thinking hints for one catalog entry.
fn describe_model(model: &serde_json::Value) -> String {
    let context = model
        .get("context_length")
        .and_then(|value| value.as_u64())
        .map(|context| format!("{}k ctx", context / 1000))
        .unwrap_or_else(|| "context unknown".into());
    let thinking = match model.get("thinking_supported").and_then(|v| v.as_bool()) {
        Some(true) => ", thinking",
        Some(false) => ", no thinking",
        None => "",
    };
    format!("{context}{thinking}")
}

/// Recorded providers with their cached catalogs, in registration order.
fn provider_records(snapshot: &SessionSnapshot) -> Vec<(omp_types::ProviderRecord, Vec<serde_json::Value>)> {
    snapshot
        .children(snapshot.container(omp_types::PROVIDERS_CONTAINER))
        .filter_map(|element| {
            let record = omp_types::ProviderRecord::from_element(element)?;
            let models = match element.attributes.get(omp_types::PROVIDER_MODELS_ATTRIBUTE) {
                Some(TypedValue::Json(value)) => {
                    value.as_array().cloned().unwrap_or_default()
                }
                _ => Vec::new(),
            };
            Some((record, models))
        })
        .collect()
}

/// Discovers aliases and declared convars from `snapshot.session_globals()` and built-in declarations.
fn populate_snapshot_completions(out: &mut Vec<Completion>, snapshot: &SessionSnapshot) {
    const BUILTIN_CONVARS: &[(&str, &str, &str)] = &[
        ("ai_model", "\"\"", "Active inference model identifier"),
        ("ai_provider", "\"\"", "Active inference provider adapter"),
        (
            "ai_endpoint",
            "\"\"",
            "Custom API endpoint for active provider",
        ),
        (
            "ai_thinking",
            "\"auto\"",
            "Model reasoning mode (auto/none/enabled/effort)",
        ),
        ("ai_fastmode", "false", "Fast mode inference toggle"),
        ("ai_temperature", "0.7", "Sampling temperature (0.0 - 2.0)"),
        ("ai_max_tokens", "null", "Max completion tokens per turn"),
        (
            "ai_compaction_threshold",
            "0.8",
            "Context window compaction threshold",
        ),
        (
            "cl_showthinking",
            "true",
            "Show thinking/reasoning blocks in transcript",
        ),
        ("cl_theme", "\"default\"", "Active client color theme"),
        (
            "cl_icon_mode",
            "\"unicode\"",
            "Icon presentation mode: unicode/nerd/ascii",
        ),
        (
            "cl_resize_policy",
            "\"rebuild\"",
            "Terminal resize policy: rebuild/preserve/append",
        ),
        (
            "sandbox_network",
            "false",
            "Permit network access in sandbox jobs",
        ),
        (
            "sandbox_write_scope",
            "\"workspace\"",
            "Filesystem write scope for sandbox jobs",
        ),
    ];

    let mut seen_convars = std::collections::BTreeSet::new();

    for (key, value) in snapshot.session_globals() {
        if let Some(alias_name) = key.strip_prefix("alias:") {
            let target_cmd = match value {
                TypedValue::String(s) => s.as_str(),
                _ => "",
            };
            out.push(Completion {
                label: format!("/{}", alias_name),
                description: format!("Alias for '{}'", target_cmd),
                insert: format!("/{}", alias_name),
                execute: true,
            });
        } else if !key.starts_with("bind:") && key != "source" {
            seen_convars.insert(key.clone());
            let val_desc = match value {
                TypedValue::String(s) => format!("\"{}\"", s),
                TypedValue::Integer(i) => i.to_string(),
                TypedValue::Number(n) => n.to_string(),
                TypedValue::Bool(b) => b.to_string(),
                TypedValue::Null => "null".to_string(),
                TypedValue::Json(j) => j.to_string(),
            };
            let default_note = BUILTIN_CONVARS
                .iter()
                .find(|(name, _, _)| *name == key)
                .map(|(_, def, _)| format!(" (default: {})", def))
                .unwrap_or_default();

            out.push(Completion {
                label: format!("/{}", key),
                description: format!("ConVar {} = {}{}", key, val_desc, default_note),
                insert: format!("/{} ", key),
                execute: false,
            });
        }
    }

    // Add any built-in convars not yet explicitly set in session globals
    for &(name, default_val, help) in BUILTIN_CONVARS {
        if !seen_convars.contains(name) {
            out.push(Completion {
                label: format!("/{}", name),
                description: format!("ConVar {} - {} (default: {})", name, help, default_val),
                insert: format!("/{} ", name),
                execute: false,
            });
        }
    }
}

// -----------------------------------------------------------------------------
// Regression Tests (DO NOT RUN per task instructions)
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use omp_types::{ElementSnapshot, SessionId};

    #[test]
    fn test_utf8_deletion_grapheme_safety() {
        let mut editor = Editor::new();
        // Insert emoji with VS16 and modifier: 👨‍👩‍👦 (family ZWJ)
        editor.insert("A👨‍👩‍👦B");
        assert_eq!(editor.text, "A👨‍👩‍👦B");

        // Position cursor right before 'B' (at end of family grapheme)
        editor.cursor = "A👨‍👩‍👦".len();
        // Backspace must delete the entire family cluster atomically without panicking or splitting UTF-8
        let handled = editor.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::empty()));
        assert!(handled);
        assert_eq!(editor.text, "AB");
        assert_eq!(editor.cursor, 1);
    }

    #[test]
    fn test_paste_sanitization_and_no_escapes() {
        let mut editor = Editor::new();
        // Malicious or styled paste with ANSI CSI color sequences and CRLF
        let raw = "\x1b[31;1mHello\x1b[0m\r\nWorld!\x1b[?25h";
        assert!(editor.insert(raw));
        // ANSI escape codes must be stripped and CRLF converted to LF
        assert_eq!(editor.text, "Hello\nWorld!");
    }

    #[test]
    fn test_draft_recall_and_history_restore() {
        let mut editor = Editor::new();
        editor.insert("cmd1");
        assert_eq!(editor.submit(), Some("cmd1".into()));

        editor.insert("cmd2");
        assert_eq!(editor.submit(), Some("cmd2".into()));

        // User starts typing an unsubmitted draft
        editor.insert("my dra");

        // Pressing Up recalls cmd2 while stashing draft
        assert!(editor.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::empty())));
        assert_eq!(editor.text, "cmd2");

        // Pressing Up again recalls cmd1
        assert!(editor.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::empty())));
        assert_eq!(editor.text, "cmd1");

        // Pressing Down navigates back to cmd2
        assert!(editor.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::empty())));
        assert_eq!(editor.text, "cmd2");

        // Pressing Down again restores the original draft exactly
        assert!(editor.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::empty())));
        assert_eq!(editor.text, "my dra");
    }

    #[test]
    fn test_filter_model_suggestions_no_keys() {
        let session_id = SessionId::new("test-session").unwrap();
        let mut snapshot = SessionSnapshot::empty(session_id);

        let caps_id = snapshot.container("capabilities").clone();
        let mut caps_node = ElementSnapshot::new(caps_id.clone(), "capabilities");
        caps_node.attributes.insert(
            "provider_metadata".into(),
            TypedValue::Json(serde_json::json!({
                "provider": "openrouter",
                "endpoint": "https://openrouter.ai/api/v1",
                "models": [
                    { "id": "anthropic/claude-3.5-sonnet", "context_length": 200000, "thinking_supported": true },
                    { "id": "openai/gpt-4o", "context_length": 128000, "thinking_supported": false }
                ]
            })),
        );
        // Put modified node back
        let patch = omp_types::Patch {
            base_offset: omp_types::JournalOffset(0),
            result_offset: omp_types::JournalOffset(1),
            by: omp_types::ActorId::new("test-owner").unwrap().into(),
            reason: "test".into(),
            ops: vec![omp_types::PatchOp::SetAttribute {
                element: caps_id,
                name: "provider_metadata".into(),
                value: caps_node
                    .attributes
                    .get("provider_metadata")
                    .unwrap()
                    .clone(),
            }],
        };
        omp_state::apply_patch(&mut snapshot, &patch).unwrap();

        // Typing "/provider select " should list only the advertised models
        let comps = completions("/provider select ", &snapshot);
        assert_eq!(comps.len(), 2);
        assert_eq!(
            comps[0].label,
            "/provider select anthropic/claude-3.5-sonnet"
        );
        assert!(comps[0].execute);
        assert_eq!(comps[1].label, "/provider select openai/gpt-4o");

        // Filtering by "gpt"
        let filtered = completions("/provider select gpt", &snapshot);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].label, "/provider select openai/gpt-4o");

        // Non-matching query yields empty
        let unknown = completions("/provider select nonexistent", &snapshot);
        assert!(unknown.is_empty());
    }

    #[test]
    fn test_visual_lines_and_cursor_at_exact_width() {
        let mut editor = Editor::new();
        editor.set("12345".into());
        let lines = editor.visual_lines(5);
        // Final line at exact width must have an extra empty line for cursor autowrap
        assert_eq!(lines, vec!["12345", ""]);
        let (col, row) = editor.visual_cursor(5);
        assert_eq!((col, row), (0, 1));
    }

    #[test]
    fn test_altgr_turkish_character_insertion() {
        let mut editor = Editor::new();
        // AltGr on Windows sends Control + Alt modifiers
        let handled = editor.handle_key(KeyEvent::new(
            KeyCode::Char('@'),
            KeyModifiers::CONTROL | KeyModifiers::ALT,
        ));
        assert!(handled);
        assert_eq!(editor.text, "@");
        assert_eq!(editor.cursor, 1);
    }

    #[test]
    fn test_oversized_insert_rejected_without_silent_truncation() {
        let mut editor = Editor::new();
        let big = "a".repeat(65537);
        assert!(!editor.insert(&big));
        assert_eq!(editor.text, "");
    }

    #[test]
    fn model_list_names_the_provider_each_model_comes_from() {
        fn set_attribute(
            snapshot: &mut SessionSnapshot,
            element: omp_types::ElementId,
            name: &str,
            value: TypedValue,
        ) {
            let base = snapshot.offset;
            omp_state::apply_patch(
                snapshot,
                &omp_types::Patch {
                    base_offset: omp_types::JournalOffset(base),
                    result_offset: omp_types::JournalOffset(base + 1),
                    by: omp_types::ActorId::new("test-owner").unwrap().into(),
                    reason: "test".into(),
                    ops: vec![omp_types::PatchOp::SetAttribute {
                        element,
                        name: name.into(),
                        value,
                    }],
                },
            )
            .unwrap();
        }
        fn create(
            snapshot: &mut SessionSnapshot,
            parent: omp_types::ElementId,
            element: ElementSnapshot,
        ) {
            let base = snapshot.offset;
            omp_state::apply_patch(
                snapshot,
                &omp_types::Patch {
                    base_offset: omp_types::JournalOffset(base),
                    result_offset: omp_types::JournalOffset(base + 1),
                    by: omp_types::ActorId::new("test-owner").unwrap().into(),
                    reason: "test".into(),
                    ops: vec![omp_types::PatchOp::Create {
                        parent,
                        index: 0,
                        element,
                    }],
                },
            )
            .unwrap();
        }

        let mut snapshot = SessionSnapshot::empty(SessionId::new("test-providers").unwrap());
        let convars = snapshot.container("convars").clone();
        let capabilities = snapshot.container("capabilities").clone();
        let providers = snapshot.container(omp_types::PROVIDERS_CONTAINER).clone();

        // Active provider: opencode-go, whose catalog lives in capabilities.
        set_attribute(
            &mut snapshot,
            convars.clone(),
            "ai_provider",
            TypedValue::String("openai_compatible".into()),
        );
        set_attribute(
            &mut snapshot,
            convars,
            "ai_endpoint",
            TypedValue::String("https://opencode.ai/zen/go/v1".into()),
        );
        set_attribute(
            &mut snapshot,
            capabilities,
            "provider_metadata",
            TypedValue::Json(serde_json::json!({
                "provider": "openai_compatible",
                "endpoint": "https://opencode.ai/zen/go/v1",
                "models": [
                    { "id": "glm-5.3-flash", "context_length": 1000000 },
                    { "id": "kimi-k3", "context_length": 1048576 }
                ]
            })),
        );

        // The active provider is itself registered, so its models are labelled
        // with the name the user gave it rather than a host fallback.
        let active_record = omp_types::ProviderRecord {
            name: "opencode-go".into(),
            adapter: "openai_compatible".into(),
            endpoint: "https://opencode.ai/zen/go/v1".into(),
            key_env: "OPENCODE_GO_API_KEY".into(),
            model: String::new(),
        };
        create(
            &mut snapshot,
            providers.clone(),
            active_record.to_element(omp_types::ElementId::new("provider-opencode-go").unwrap()),
        );

        // A second provider is recorded with the catalog it was fetched with.
        let provider = omp_types::ProviderRecord {
            name: "clawbay".into(),
            adapter: "openai_compatible".into(),
            endpoint: "https://api.theclawbay.com/v1".into(),
            key_env: "OPENAI_API_KEY".into(),
            model: String::new(),
        };
        let mut element = provider.to_element(omp_types::ElementId::new("provider-clawbay").unwrap());
        element.attributes.insert(
            omp_types::PROVIDER_MODELS_ATTRIBUTE.into(),
            TypedValue::Json(serde_json::json!([
                { "id": "claude-fable-5", "context_length": 200000 }
            ])),
        );
        create(&mut snapshot, providers, element);

        let comps = completions("/provider select ", &snapshot);
        assert_eq!(comps.len(), 3, "{comps:#?}");

        // The active provider's models are selected directly, and each entry
        // names the provider it came from.
        assert_eq!(comps[0].label, "/provider select glm-5.3-flash");
        assert!(comps[0].description.contains("opencode-go"), "{}", comps[0].description);
        assert!(comps[0].description.contains("1000k ctx"));

        // A model from another provider switches first, and says so.
        assert_eq!(
            comps[2].label,
            "/provider use clawbay; /provider select claude-fable-5"
        );
        assert!(comps[2].description.contains("clawbay"), "{}", comps[2].description);
        assert!(comps[2].execute);

        // Filtering by provider name narrows to that provider's models.
        let by_provider = completions("/provider select clawbay", &snapshot);
        assert_eq!(by_provider.len(), 1);
        assert!(by_provider[0].label.contains("claude-fable-5"));
    }

    #[test]
    fn effort_completions_list_the_models_levels() {
        let mut snapshot = SessionSnapshot::empty(SessionId::new("test-effort-list").unwrap());
        let convars = snapshot.container("convars").clone();
        let capabilities = snapshot.container("capabilities").clone();
        let mut set = |element: omp_types::ElementId, name: &str, value: TypedValue| {
            let base = snapshot.offset;
            omp_state::apply_patch(
                &mut snapshot,
                &omp_types::Patch {
                    base_offset: omp_types::JournalOffset(base),
                    result_offset: omp_types::JournalOffset(base + 1),
                    by: omp_types::ActorId::new("test-owner").unwrap().into(),
                    reason: "test".into(),
                    ops: vec![omp_types::PatchOp::SetAttribute {
                        element,
                        name: name.into(),
                        value,
                    }],
                },
            )
            .unwrap();
        };
        set(
            convars.clone(),
            "ai_thinking_levels",
            TypedValue::Json(serde_json::json!(["low", "medium", "high"])),
        );
        set(convars, "ai_thinking", TypedValue::String("high".into()));
        set(
            capabilities,
            "provider_metadata",
            TypedValue::Json(serde_json::json!({
                "models": [],
                "active_model": { "id": "m", "thinking_default": "medium" }
            })),
        );

        let offered = completions("/effort ", &snapshot);
        let labels: Vec<&str> = offered.iter().map(|entry| entry.label.as_str()).collect();
        assert_eq!(
            labels,
            vec![
                "/effort low",
                "/effort medium",
                "/effort high",
                "/effort auto",
                "/effort off"
            ]
        );
        let medium = &offered[1];
        assert!(medium.description.contains("provider default"), "{}", medium.description);
        let high = &offered[2];
        assert!(high.description.contains("current"), "{}", high.description);

        // Filtering narrows to the matching level.
        let filtered = completions("/effort me", &snapshot);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].label, "/effort medium");
    }
}
