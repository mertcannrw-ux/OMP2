use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum SemanticColor {
    Info,
    Muted,
    Warning,
    Error,
    Success,
    Primary,
    Accent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Color {
    Semantic(SemanticColor),
    Rgb(u8, u8, u8),
    Ansi256(u8),
    Reset,
    Default,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Modifier {
    pub bold: bool,
    pub dim: bool,
    pub italic: bool,
    pub underline: bool,
    pub inverse: bool,
    pub strikethrough: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Style {
    pub fg: Option<Color>,
    pub bg: Option<Color>,
    pub modifier: Modifier,
}

impl Style {
    pub const fn new() -> Self {
        Self {
            fg: None,
            bg: None,
            modifier: Modifier {
                bold: false,
                dim: false,
                italic: false,
                underline: false,
                inverse: false,
                strikethrough: false,
            },
        }
    }

    pub const fn fg(mut self, color: Color) -> Self {
        self.fg = Some(color);
        self
    }

    pub const fn bg(mut self, color: Color) -> Self {
        self.bg = Some(color);
        self
    }

    pub const fn semantic(color: SemanticColor) -> Self {
        Self {
            fg: Some(Color::Semantic(color)),
            bg: None,
            modifier: Modifier {
                bold: false,
                dim: false,
                italic: false,
                underline: false,
                inverse: false,
                strikethrough: false,
            },
        }
    }

    pub const fn bold(mut self) -> Self {
        self.modifier.bold = true;
        self
    }

    pub const fn dim(mut self) -> Self {
        self.modifier.dim = true;
        self
    }

    pub const fn italic(mut self) -> Self {
        self.modifier.italic = true;
        self
    }

    pub const fn underline(mut self) -> Self {
        self.modifier.underline = true;
        self
    }

    pub const fn inverse(mut self) -> Self {
        self.modifier.inverse = true;
        self
    }

    pub const fn strikethrough(mut self) -> Self {
        self.modifier.strikethrough = true;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Run {
    pub style: Style,
    pub text: String,
}

impl Run {
    pub fn new(style: Style, text: impl Into<String>) -> Self {
        Self {
            style,
            text: sanitize_text(&text.into()),
        }
    }

    pub fn plain(text: impl Into<String>) -> Self {
        Self::new(Style::new(), text)
    }

    pub fn visible_width(&self) -> usize {
        str_visible_width(&self.text)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RichText {
    pub runs: Vec<Run>,
}

impl RichText {
    pub const fn new() -> Self {
        Self { runs: Vec::new() }
    }

    pub fn from_plain(text: impl Into<String>) -> Self {
        let mut rt = Self::new();
        rt.push_plain(text);
        rt
    }

    pub fn from_styled(style: Style, text: impl Into<String>) -> Self {
        let mut rt = Self::new();
        rt.push(style, text);
        rt
    }

    pub fn from_semantic(color: SemanticColor, text: impl Into<String>) -> Self {
        let mut rt = Self::new();
        rt.push_semantic(color, text);
        rt
    }

    pub fn push(&mut self, style: Style, text: impl Into<String>) {
        let text = sanitize_text(&text.into());
        if text.is_empty() {
            return;
        }
        if let Some(last) = self.runs.last_mut()
            && last.style == style {
                last.text.push_str(&text);
                return;
            }
        self.runs.push(Run { style, text });
    }

    pub fn push_plain(&mut self, text: impl Into<String>) {
        self.push(Style::new(), text);
    }

    pub fn push_semantic(&mut self, color: SemanticColor, text: impl Into<String>) {
        self.push(Style::semantic(color), text);
    }

    pub fn is_empty(&self) -> bool {
        self.runs.iter().all(|r| r.text.is_empty())
    }

    pub fn plain_text(&self) -> String {
        let mut s = String::new();
        for r in &self.runs {
            s.push_str(&r.text);
        }
        s
    }

    pub fn visible_width(&self) -> usize {
        self.runs.iter().map(|r| r.visible_width()).sum()
    }

    pub fn split_lines(&self) -> Vec<RichText> {
        let mut lines = Vec::new();
        let mut current_line = RichText::new();

        for run in &self.runs {
            let mut parts = run.text.split('\n').peekable();
            while let Some(part) = parts.next() {
                if !part.is_empty() {
                    current_line.push(run.style, part);
                }
                if parts.peek().is_some() {
                    lines.push(current_line);
                    current_line = RichText::new();
                }
            }
        }
        lines.push(current_line);
        lines
    }

    pub fn truncate_width(&self, max_width: usize, ellipsis: Option<&str>) -> RichText {
        let total_width = self.visible_width();
        if total_width <= max_width {
            return self.clone();
        }

        let ellipsis_str = ellipsis.unwrap_or("");
        let ellipsis_width = str_visible_width(ellipsis_str);
        let budget = max_width.saturating_sub(ellipsis_width);

        let mut truncated = RichText::new();
        let mut accumulated_width = 0;

        for run in &self.runs {
            let mut run_buf = String::new();
            for cluster in grapheme_clusters(&run.text) {
                let w = grapheme_cluster_width(cluster);
                if accumulated_width + w > budget {
                    if !run_buf.is_empty() {
                        truncated.push(run.style, run_buf);
                    }
                    if !ellipsis_str.is_empty() {
                        truncated.push(run.style, ellipsis_str);
                    }
                    return truncated;
                }
                run_buf.push_str(cluster);
                accumulated_width += w;
            }
            if !run_buf.is_empty() {
                truncated.push(run.style, run_buf);
            }
        }

        if !ellipsis_str.is_empty() {
            truncated.push(Style::new(), ellipsis_str);
        }
        truncated
    }
}
pub fn char_width(c: char) -> usize {
    let mut buf = [0u8; 4];
    let s = c.encode_utf8(&mut buf);
    grapheme_cluster_width(s)
}

pub fn is_combining_or_extender(u: u32) -> bool {
    (0x0300..=0x036F).contains(&u)
        || (0x1AB0..=0x1AFF).contains(&u)
        || (0x1DC0..=0x1DFF).contains(&u)
        || (0x20D0..=0x20FF).contains(&u)
        || (0xFE20..=0xFE2F).contains(&u)
        || (0xFE00..=0xFE0F).contains(&u)
        || (0xE0100..=0xE01EF).contains(&u)
        || (0x1F3FB..=0x1F3FF).contains(&u)
        || (0xE0020..=0xE007F).contains(&u)
        || u == 0x200D
        || ('\u{200B}'..='\u{200D}').contains(&char::from_u32(u).unwrap_or('\0'))
}

pub fn is_emoji_base(u: u32) -> bool {
    (0x1F300..=0x1FAFF).contains(&u)
        || (0x1F000..=0x1F02F).contains(&u)
        || (0x1F0A0..=0x1F0FF).contains(&u)
        || (0x2600..=0x26FF).contains(&u)
        || (0x2700..=0x27BF).contains(&u)
        || (0x2B50..=0x2B55).contains(&u)
        || (0x231A..=0x231B).contains(&u)
        || (0x23E9..=0x23EC).contains(&u)
        || (0x23F0..=0x23F3).contains(&u)
}

pub fn is_east_asian_wide(u: u32) -> bool {
    (0x1100..=0x115F).contains(&u)
        || (0x2329..=0x232A).contains(&u)
        || (0x2E80..=0xA4CF).contains(&u)
        || (0xAC00..=0xD7A3).contains(&u)
        || (0xF900..=0xFAFF).contains(&u)
        || (0xFE10..=0xFE19).contains(&u)
        || (0xFE30..=0xFE6F).contains(&u)
        || (0xFF00..=0xFF60).contains(&u)
        || (0xFFE0..=0xFFE6).contains(&u)
}

pub fn grapheme_cluster_width(cluster: &str) -> usize {
    if cluster.is_empty() {
        return 0;
    }

    let mut chars = cluster.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return 0,
    };

    if first.is_control() || ('\u{200B}'..='\u{200D}').contains(&first) || first == '\u{FEFF}' {
        return 0;
    }

    let u_first = first as u32;

    // Check if cluster contains emoji presentation selector VS16 (\u{FE0F})
    let has_vs16 = cluster.chars().any(|c| c == '\u{FE0F}');
    // Check if cluster contains Zero-Width Joiner (ZWJ \u{200D})
    let has_zwj = cluster.contains('\u{200D}');

    // Regional Indicator pair (flag emoji)
    if (0x1F1E6..=0x1F1FF).contains(&u_first) {
        let ri_count = cluster
            .chars()
            .filter(|c| (0x1F1E6..=0x1F1FF).contains(&(*c as u32)))
            .count();
        if ri_count >= 2 {
            return 2;
        }
        return 1;
    }

    if is_emoji_base(u_first) || has_vs16 || has_zwj {
        return 2;
    }

    if is_east_asian_wide(u_first) {
        return 2;
    }

    1
}

pub fn grapheme_clusters(s: &str) -> Vec<&str> {
    let mut clusters = Vec::new();
    let mut indices = s.char_indices().peekable();

    while let Some((start, c)) = indices.next() {
        let u = c as u32;

        if (0x1F1E6..=0x1F1FF).contains(&u)
            && let Some(&(_, next_c)) = indices.peek()
                && (0x1F1E6..=0x1F1FF).contains(&(next_c as u32)) {
                    indices.next();
                }

        while let Some(&(_, peek_c)) = indices.peek() {
            let peek_u = peek_c as u32;
            if peek_u == 0x200D {
                indices.next();
                if indices.next().is_none() {
                    break;
                }
                continue;
            }
            if is_combining_or_extender(peek_u) {
                indices.next();
                continue;
            }
            break;
        }

        let end = indices.peek().map(|&(idx, _)| idx).unwrap_or(s.len());
        clusters.push(&s[start..end]);
    }

    clusters
}

pub fn str_visible_width(s: &str) -> usize {
    grapheme_clusters(s)
        .iter()
        .map(|g| grapheme_cluster_width(g))
        .sum()
}

pub fn sanitize_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if let Some(&next) = chars.peek() {
                match next {
                    '[' => {
                        // CSI sequence: ESC [ ... Final (0x40..=0x7E)
                        chars.next();
                        for inner in chars.by_ref() {
                            let u = inner as u32;
                            if (0x40..=0x7E).contains(&u) {
                                break;
                            }
                        }
                    }
                    ']' => {
                        // OSC sequence: ESC ] ... (BEL \x07 or ST ESC \)
                        chars.next();
                        while let Some(inner) = chars.next() {
                            if inner == '\x07' {
                                break;
                            }
                            if inner == '\x1b' {
                                if chars.peek() == Some(&'\\') {
                                    chars.next();
                                }
                                break;
                            }
                        }
                    }
                    'P' | '^' | '_' | 'X' => {
                        // DCS, PM, APC, SOS: ESC P ... ST (ESC \ or BEL)
                        chars.next();
                        while let Some(inner) = chars.next() {
                            if inner == '\x07' {
                                break;
                            }
                            if inner == '\x1b' {
                                if chars.peek() == Some(&'\\') {
                                    chars.next();
                                }
                                break;
                            }
                        }
                    }
                    _ => {
                        chars.next();
                    }
                }
            }
            continue;
        }

        if c == '\n' || c == '\r' || c == '\t' {
            out.push(c);
            continue;
        }

        let u = c as u32;

        // Strip C0 and C1 control characters, DEL
        if (u <= 0x1F) || (0x7F..=0x9F).contains(&u) {
            continue;
        }

        // Strip Unicode Bidirectional Overrides (Trojan Source attacks)
        if (0x202A..=0x202E).contains(&u) || (0x2066..=0x2069).contains(&u) {
            continue;
        }

        out.push(c);
    }
    out
}
