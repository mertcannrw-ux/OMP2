use crate::richtext::{
    Color, RichText, Run, SemanticColor, Style, grapheme_cluster_width, grapheme_clusters,
    str_visible_width,
};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OutError {
    #[error("output write error: {0}")]
    Write(String),
    #[error("buffer overflow: exceeded maximum line/byte capacity")]
    BufferOverflow,
}

pub trait Out {
    fn write_run(&mut self, run: &Run) -> Result<(), OutError>;
    fn line_break(&mut self) -> Result<(), OutError>;
    fn flush(&mut self) -> Result<(), OutError>;

    fn write_rich(&mut self, text: &RichText) -> Result<(), OutError> {
        for run in &text.runs {
            self.write_run(run)?;
        }
        Ok(())
    }

    fn write_str(&mut self, style: Style, s: &str) -> Result<(), OutError> {
        self.write_run(&Run::new(style, s))
    }

    fn write_plain(&mut self, s: &str) -> Result<(), OutError> {
        self.write_run(&Run::plain(s))
    }
}

#[derive(Clone, Debug, Default)]
pub struct StringOutSink {
    buffer: String,
}

impl StringOutSink {
    pub const fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    pub fn into_string(self) -> String {
        self.buffer
    }

    pub fn as_str(&self) -> &str {
        &self.buffer
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

impl Out for StringOutSink {
    fn write_run(&mut self, run: &Run) -> Result<(), OutError> {
        self.buffer.push_str(&run.text);
        Ok(())
    }

    fn line_break(&mut self) -> Result<(), OutError> {
        self.buffer.push('\n');
        Ok(())
    }

    fn flush(&mut self) -> Result<(), OutError> {
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct LineBufferSink {
    lines: Vec<RichText>,
    current_line: RichText,
}

impl LineBufferSink {
    pub const fn new() -> Self {
        Self {
            lines: Vec::new(),
            current_line: RichText::new(),
        }
    }

    pub fn finish(mut self) -> Vec<RichText> {
        if !self.current_line.is_empty() {
            self.lines.push(self.current_line);
        }
        self.lines
    }

    pub fn lines(&self) -> &[RichText] {
        &self.lines
    }

    pub fn current_line(&self) -> &RichText {
        &self.current_line
    }
}

impl Out for LineBufferSink {
    fn write_run(&mut self, run: &Run) -> Result<(), OutError> {
        self.current_line.push(run.style, &run.text);
        Ok(())
    }

    fn line_break(&mut self) -> Result<(), OutError> {
        let line = std::mem::replace(&mut self.current_line, RichText::new());
        self.lines.push(line);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), OutError> {
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct AnsiOutSink {
    buffer: String,
}

impl AnsiOutSink {
    pub const fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    pub fn into_string(self) -> String {
        self.buffer
    }

    pub fn as_str(&self) -> &str {
        &self.buffer
    }

    fn style_to_ansi(style: &Style) -> String {
        let mut codes = Vec::new();

        if style.modifier.bold {
            codes.push("1".to_string());
        }
        if style.modifier.dim {
            codes.push("2".to_string());
        }
        if style.modifier.italic {
            codes.push("3".to_string());
        }
        if style.modifier.underline {
            codes.push("4".to_string());
        }
        if style.modifier.inverse {
            codes.push("7".to_string());
        }
        if style.modifier.strikethrough {
            codes.push("9".to_string());
        }

        if let Some(fg) = style.fg {
            match fg {
                Color::Semantic(SemanticColor::Info) => codes.push("34".to_string()),
                Color::Semantic(SemanticColor::Muted) => codes.push("90".to_string()),
                Color::Semantic(SemanticColor::Warning) => codes.push("33".to_string()),
                Color::Semantic(SemanticColor::Error) => codes.push("31".to_string()),
                Color::Semantic(SemanticColor::Success) => codes.push("32".to_string()),
                Color::Semantic(SemanticColor::Primary) => codes.push("36".to_string()),
                Color::Semantic(SemanticColor::Accent) => codes.push("35".to_string()),
                Color::Rgb(r, g, b) => codes.push(format!("38;2;{r};{g};{b}")),
                Color::Ansi256(c) => codes.push(format!("38;5;{c}")),
                Color::Reset => codes.push("39".to_string()),
                Color::Default => {}
            }
        }

        if let Some(bg) = style.bg {
            match bg {
                Color::Semantic(SemanticColor::Info) => codes.push("44".to_string()),
                Color::Semantic(SemanticColor::Muted) => codes.push("100".to_string()),
                Color::Semantic(SemanticColor::Warning) => codes.push("43".to_string()),
                Color::Semantic(SemanticColor::Error) => codes.push("41".to_string()),
                Color::Semantic(SemanticColor::Success) => codes.push("42".to_string()),
                Color::Semantic(SemanticColor::Primary) => codes.push("46".to_string()),
                Color::Semantic(SemanticColor::Accent) => codes.push("45".to_string()),
                Color::Rgb(r, g, b) => codes.push(format!("48;2;{r};{g};{b}")),
                Color::Ansi256(c) => codes.push(format!("48;5;{c}")),
                Color::Reset => codes.push("49".to_string()),
                Color::Default => {}
            }
        }

        if codes.is_empty() {
            String::new()
        } else {
            format!("\x1b[{}m", codes.join(";"))
        }
    }
}

impl Out for AnsiOutSink {
    fn write_run(&mut self, run: &Run) -> Result<(), OutError> {
        let ansi_start = Self::style_to_ansi(&run.style);
        if !ansi_start.is_empty() {
            self.buffer.push_str(&ansi_start);
        }
        self.buffer.push_str(&run.text);
        if !ansi_start.is_empty() {
            self.buffer.push_str("\x1b[0m");
        }
        Ok(())
    }

    fn line_break(&mut self) -> Result<(), OutError> {
        self.buffer.push('\n');
        Ok(())
    }

    fn flush(&mut self) -> Result<(), OutError> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WrapMode {
    NoWrap,
    WordWrap,
    CharacterWrap,
}

/// Applies terminal geometry while forwarding styled runs, without building an intermediate frame.
pub struct LayoutOut<'a, O: Out> {
    inner: &'a mut O,
    width: usize,
    column: usize,
    wrap: bool,
}

impl<'a, O: Out> LayoutOut<'a, O> {
    pub fn new(inner: &'a mut O, width: usize, wrap: bool) -> Self {
        Self {
            inner,
            width,
            column: 0,
            wrap,
        }
    }
}

impl<O: Out> Out for LayoutOut<'_, O> {
    fn write_run(&mut self, run: &Run) -> Result<(), OutError> {
        for cluster in grapheme_clusters(&run.text) {
            if cluster == "\n" || cluster == "\r\n" {
                self.line_break()?;
                continue;
            }
            if cluster == "\r" {
                continue;
            }
            if cluster == "\t" {
                let spaces = 4 - self.column % 4;
                self.write_run(&Run::new(run.style, " ".repeat(spaces)))?;
                continue;
            }
            let cells = grapheme_cluster_width(cluster);
            if self.width > 0 && self.column + cells > self.width {
                if !self.wrap {
                    continue;
                }
                self.line_break()?;
            }
            if self.width == 0 || cells <= self.width {
                self.inner.write_str(run.style, cluster)?;
                self.column += cells;
            }
        }
        Ok(())
    }

    fn line_break(&mut self) -> Result<(), OutError> {
        self.column = 0;
        self.inner.line_break()
    }

    fn flush(&mut self) -> Result<(), OutError> {
        self.inner.flush()
    }
}

pub fn wrap_rich_text(text: &RichText, max_width: usize, mode: WrapMode) -> Vec<RichText> {
    if max_width == 0 || mode == WrapMode::NoWrap {
        return text.split_lines();
    }

    let raw_lines = text.split_lines();
    let mut wrapped = Vec::new();

    for line in raw_lines {
        if line.visible_width() <= max_width {
            wrapped.push(line);
            continue;
        }

        match mode {
            WrapMode::CharacterWrap => {
                let mut current_line = RichText::new();
                let mut current_width = 0;

                for run in &line.runs {
                    let mut run_buf = String::new();
                    for cluster in grapheme_clusters(&run.text) {
                        let w = grapheme_cluster_width(cluster);
                        if current_width + w > max_width && current_width > 0 {
                            if !run_buf.is_empty() {
                                current_line.push(run.style, run_buf);
                                run_buf = String::new();
                            }
                            wrapped.push(current_line);
                            current_line = RichText::new();
                            current_width = 0;
                        }
                        run_buf.push_str(cluster);
                        current_width += w;
                    }
                    if !run_buf.is_empty() {
                        current_line.push(run.style, run_buf);
                    }
                }
                if !current_line.is_empty() {
                    wrapped.push(current_line);
                }
            }
            WrapMode::WordWrap => {
                let mut current_line = RichText::new();
                let mut current_width = 0;

                for run in &line.runs {
                    let words = run.text.split_inclusive(char::is_whitespace);
                    for word in words {
                        let word_width = str_visible_width(word);
                        if current_width + word_width > max_width && current_width > 0 {
                            wrapped.push(current_line);
                            current_line = RichText::new();
                            current_width = 0;
                        }
                        current_line.push(run.style, word);
                        current_width += word_width;
                    }
                }
                if !current_line.is_empty() {
                    wrapped.push(current_line);
                }
            }
            WrapMode::NoWrap => unreachable!(),
        }
    }

    wrapped
}

#[derive(Clone, Debug)]
pub struct StreamPacingConfig {
    pub min_delay_ms: u64,
    pub chunk_char_target: usize,
    pub burst_limit: usize,
}

impl Default for StreamPacingConfig {
    fn default() -> Self {
        Self {
            min_delay_ms: 15,
            chunk_char_target: 8,
            burst_limit: 128,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StreamPacer {
    config: StreamPacingConfig,
    buffered_text: String,
    last_emit: Option<std::time::Instant>,
}

impl StreamPacer {
    pub fn new(config: StreamPacingConfig) -> Self {
        Self {
            config,
            buffered_text: String::new(),
            last_emit: None,
        }
    }

    /// Returns the accepted UTF-8 prefix. The caller retains the suffix until capacity is available.
    pub fn push_chunk(&mut self, text: &str) -> usize {
        let available = self
            .config
            .burst_limit
            .max(4)
            .saturating_sub(self.buffered_text.len());
        let mut accepted = text.len().min(available);
        while !text.is_char_boundary(accepted) {
            accepted -= 1;
        }
        self.buffered_text.push_str(&text[..accepted]);
        accepted
    }

    pub fn next_drawable_chunk(&mut self) -> Option<String> {
        self.next_drawable_chunk_at(std::time::Instant::now())
    }

    pub fn next_drawable_chunk_at(&mut self, now: std::time::Instant) -> Option<String> {
        if self.buffered_text.is_empty()
            || self.last_emit.is_some_and(|last| {
                now.saturating_duration_since(last)
                    < std::time::Duration::from_millis(self.config.min_delay_ms)
            })
        {
            return None;
        }
        let end = self
            .buffered_text
            .char_indices()
            .nth(self.config.chunk_char_target.max(1))
            .map_or(self.buffered_text.len(), |(index, _)| index);
        let chunk = self.buffered_text.drain(..end).collect();
        self.last_emit = Some(now);
        Some(chunk)
    }

    pub fn drain_all(&mut self) -> Option<String> {
        if self.buffered_text.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.buffered_text))
        }
    }

    pub fn has_buffered(&self) -> bool {
        !self.buffered_text.is_empty()
    }
}
