//! Crossterm's Windows console-record backend cannot preserve bracketed paste.
//! Read VT input on Windows; retain Crossterm's native event source on Unix.
use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use std::{
    collections::VecDeque,
    io,
    time::{Duration, Instant},
};

#[cfg(windows)]
pub struct Input {
    bytes: std::sync::mpsc::Receiver<io::Result<Vec<u8>>>,
    parser: Parser,
    size: (u16, u16),
}
#[cfg(windows)]
impl Input {
    pub fn new() -> io::Result<Self> {
        use windows_sys::Win32::System::Console::{
            ENABLE_VIRTUAL_TERMINAL_INPUT, GetConsoleMode, GetStdHandle, STD_INPUT_HANDLE,
            SetConsoleMode,
        };
        let handle = unsafe { GetStdHandle(STD_INPUT_HANDLE) };
        let mut mode = 0;
        if unsafe { GetConsoleMode(handle, &mut mode) } == 0
            || unsafe { SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_INPUT) } == 0
        {
            return Err(io::Error::last_os_error());
        }
        let (send, bytes) = std::sync::mpsc::sync_channel(8);
        std::thread::spawn(move || {
            use std::io::Read;
            let stdin = io::stdin();
            let mut reader = stdin.lock();
            let mut chunk = [0u8; 4096];
            loop {
                match reader.read(&mut chunk) {
                    Ok(0) => {
                        let _ = send.send(Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "terminal input closed",
                        )));
                        break;
                    }
                    Ok(count) => {
                        if send.send(Ok(chunk[..count].to_vec())).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = send.send(Err(error));
                        break;
                    }
                }
            }
        });
        Ok(Self {
            bytes,
            parser: Parser::default(),
            size: crossterm::terminal::size()?,
        })
    }
    pub fn next(&mut self, timeout: Duration) -> io::Result<Option<Event>> {
        let size = crossterm::terminal::size()?;
        if self.size != size {
            self.size = size;
            return Ok(Some(Event::Resize(size.0, size.1)));
        }
        if let Some(event) = self.parser.next() {
            return Ok(Some(event));
        }
        match self.bytes.recv_timeout(timeout) {
            Ok(bytes) => self.parser.feed(&bytes?),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "terminal input disconnected",
                ));
            }
        }
        Ok(self.parser.next())
    }
}
#[cfg(not(windows))]
pub struct Input;
#[cfg(not(windows))]
impl Input {
    pub fn new() -> io::Result<Self> {
        Ok(Self)
    }
    pub fn next(&mut self, timeout: Duration) -> io::Result<Option<Event>> {
        if crossterm::event::poll(timeout)? {
            Ok(Some(crossterm::event::read()?))
        } else {
            Ok(None)
        }
    }
}

#[derive(Default)]
struct Parser {
    pending: VecDeque<u8>,
    escape_since: Option<Instant>,
    paste: Option<Vec<u8>>,
}
impl Parser {
    fn feed(&mut self, bytes: &[u8]) {
        self.pending.extend(bytes);
    }
    fn next(&mut self) -> Option<Event> {
        loop {
            if let Some(paste) = &mut self.paste {
                const END: &[u8] = b"\x1b[201~";
                while !self.pending.is_empty() {
                    if self.pending[0] == 27 {
                        let matched = self
                            .pending
                            .iter()
                            .take(END.len())
                            .zip(END)
                            .all(|(a, b)| a == b);
                        if matched && self.pending.len() < END.len() {
                            return None;
                        }
                        if matched {
                            self.pending.drain(..END.len());
                            // `take()` always yields `Some` here: we are inside
                            // the `Some` arm of `self.paste`. (`expect`, not
                            // `unwrap`, to document the invariant.)
                            let bytes = self
                                .paste
                                .take()
                                .expect("paste body present while parsing paste");
                            let text = String::from_utf8_lossy(&bytes).into_owned();
                            return Some(Event::Paste(text));
                        }
                    }
                    let byte = self
                        .pending
                        .pop_front()
                        .expect("pending nonempty inside non-empty loop");
                    if paste.len() < 65_537 {
                        paste.push(byte);
                    }
                }
                return None;
            }
            let first = *self.pending.front()?;
            if first == 27 {
                let since = *self.escape_since.get_or_insert_with(Instant::now);
                if self.pending.len() == 1 {
                    if since.elapsed() < Duration::from_millis(35) {
                        return None;
                    }
                    self.pending.pop_front();
                    self.escape_since = None;
                    return Some(key(KeyCode::Esc, KeyModifiers::NONE));
                }
                let second = self.pending[1];
                if second == b'[' || second == b'O' {
                    let is_sgr = second == b'[' && self.pending.get(2) == Some(&b'<');
                    let end = (2..self.pending.len()).find(|index| {
                        let byte = self.pending[*index];
                        if is_sgr {
                            if byte == b'M' {
                                true
                            } else if byte == b'm' {
                                !self
                                    .pending
                                    .get(index + 1)
                                    .is_some_and(|next| next.is_ascii_alphabetic())
                            } else {
                                false
                            }
                        } else {
                            (0x40..=0x7e).contains(&byte)
                        }
                    });
                    let Some(end) = end else {
                        if self.pending.len() > 96 || since.elapsed() > Duration::from_millis(200) {
                            self.pending.clear();
                            self.escape_since = None;
                        }
                        return None;
                    };
                    let sequence: Vec<u8> = self.pending.drain(..=end).collect();
                    self.escape_since = None;
                    if sequence == b"\x1b[200~" {
                        self.paste = Some(Vec::new());
                        continue;
                    }
                    if let Some(event) = decode_sequence(&sequence) {
                        return Some(event);
                    }
                    continue;
                }
                self.pending.pop_front();
                self.escape_since = None;
                if let Some(Event::Key(mut event)) = self.next() {
                    event.modifiers.insert(KeyModifiers::ALT);
                    return Some(Event::Key(event));
                }
                return None;
            }
            self.escape_since = None;
            let code = match first {
                b'\r' => Some((KeyCode::Enter, KeyModifiers::NONE)),
                b'\n' => Some((KeyCode::Char('j'), KeyModifiers::CONTROL)),
                b'\t' => Some((KeyCode::Tab, KeyModifiers::NONE)),
                8 | 127 => Some((KeyCode::Backspace, KeyModifiers::NONE)),
                1..=26 => Some((
                    KeyCode::Char((b'a' + first - 1) as char),
                    KeyModifiers::CONTROL,
                )),
                0 => Some((KeyCode::Char(' '), KeyModifiers::CONTROL)),
                _ => None,
            };
            if let Some((code, modifiers)) = code {
                self.pending.pop_front();
                return Some(key(code, modifiers));
            }
            let needed = if first < 128 {
                1
            } else if first < 224 {
                2
            } else if first < 240 {
                3
            } else {
                4
            };
            if self.pending.len() < needed {
                return None;
            }
            let mut bytes = [0u8; 4];
            for (slot, byte) in bytes.iter_mut().zip(self.pending.iter()).take(needed) {
                *slot = *byte;
            }
            match std::str::from_utf8(&bytes[..needed])
                .ok()
                .and_then(|s| s.chars().next())
            {
                Some(character) => {
                    self.pending.drain(..needed);
                    return Some(key(KeyCode::Char(character), KeyModifiers::NONE));
                }
                None => {
                    self.pending.pop_front();
                }
            }
        }
    }
}
fn key(code: KeyCode, modifiers: KeyModifiers) -> Event {
    Event::Key(KeyEvent::new(code, modifiers))
}
fn decode_sequence(sequence: &[u8]) -> Option<Event> {
    let last = *sequence.last()?;
    if sequence.starts_with(b"\x1b[<") {
        if last != b'M' && last != b'm' {
            return None;
        }
        let body = std::str::from_utf8(&sequence[3..sequence.len() - 1]).ok()?;
        let mut parts = body.split(';');
        let cb: u32 = parts.next()?.parse().ok()?;
        let cx: u16 = parts.next()?.parse().ok()?;
        let cy: u16 = parts.next()?.parse().ok()?;
        if parts.next().is_some() {
            return None;
        }
        let mut modifiers = KeyModifiers::NONE;
        if cb & 4 != 0 {
            modifiers |= KeyModifiers::SHIFT;
        }
        if cb & 8 != 0 {
            modifiers |= KeyModifiers::ALT;
        }
        if cb & 16 != 0 {
            modifiers |= KeyModifiers::CONTROL;
        }
        let kind = if cb & 64 != 0 {
            match cb & 3 {
                0 => MouseEventKind::ScrollUp,
                1 => MouseEventKind::ScrollDown,
                2 => MouseEventKind::ScrollLeft,
                3 => MouseEventKind::ScrollRight,
                _ => return None,
            }
        } else {
            return None;
        };
        return Some(Event::Mouse(MouseEvent {
            kind,
            column: cx.saturating_sub(1),
            row: cy.saturating_sub(1),
            modifiers,
        }));
    }
    let body = std::str::from_utf8(&sequence[2..sequence.len() - 1]).ok()?;
    let params = body
        .split(';')
        .map(|part| {
            part.split(':')
                .next()
                .unwrap_or("")
                .parse::<u32>()
                .unwrap_or(1)
        })
        .collect::<Vec<_>>();
    let modifier = params.get(1).copied().unwrap_or(1).saturating_sub(1);
    let mut modifiers = KeyModifiers::NONE;
    if modifier & 1 != 0 {
        modifiers |= KeyModifiers::SHIFT;
    }
    if modifier & 2 != 0 {
        modifiers |= KeyModifiers::ALT;
    }
    if modifier & 4 != 0 {
        modifiers |= KeyModifiers::CONTROL;
    }
    let code = match last {
        b'A' => KeyCode::Up,
        b'B' => KeyCode::Down,
        b'C' => KeyCode::Right,
        b'D' => KeyCode::Left,
        b'H' => KeyCode::Home,
        b'F' => KeyCode::End,
        b'Z' => {
            modifiers |= KeyModifiers::SHIFT;
            KeyCode::BackTab
        }
        b'~' => match params[0] {
            1 | 7 => KeyCode::Home,
            3 => KeyCode::Delete,
            4 | 8 => KeyCode::End,
            5 => KeyCode::PageUp,
            6 => KeyCode::PageDown,
            _ => return None,
        },
        b'u' => match params[0] {
            13 => KeyCode::Enter,
            27 => KeyCode::Esc,
            9 => KeyCode::Tab,
            127 => KeyCode::Backspace,
            value => KeyCode::Char(char::from_u32(value)?),
        },
        _ => return None,
    };
    Some(key(code, modifiers))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bracketed_multiline_paste_is_one_event_not_commands() {
        let mut parser = Parser::default();
        parser.feed(b"\x1b[200~hello\n/exit\r\n");
        assert!(parser.next().is_none());
        parser.feed(b"world\x1b[20");
        assert!(parser.next().is_none());
        parser.feed(b"1~\r");
        assert_eq!(
            parser.next(),
            Some(Event::Paste("hello\n/exit\r\nworld".into()))
        );
        assert_eq!(parser.next(), Some(key(KeyCode::Enter, KeyModifiers::NONE)));
    }
    #[test]
    fn split_unicode_and_modified_keys_remain_distinct() {
        let mut parser = Parser::default();
        parser.feed(&[0xc4]);
        assert!(parser.next().is_none());
        parser.feed(&[0xb1]);
        assert_eq!(
            parser.next(),
            Some(key(KeyCode::Char('ı'), KeyModifiers::NONE))
        );
        parser.feed(b"\x1b[13;2u\x1b[1;5D\n");
        assert_eq!(
            parser.next(),
            Some(key(KeyCode::Enter, KeyModifiers::SHIFT))
        );
        assert_eq!(
            parser.next(),
            Some(key(KeyCode::Left, KeyModifiers::CONTROL))
        );
        assert_eq!(
            parser.next(),
            Some(key(KeyCode::Char('j'), KeyModifiers::CONTROL))
        );
    }
    #[test]
    fn sgr_mouse_wheel_and_split_frames() {
        let mut parser = Parser::default();
        parser.feed(b"\x1b[<6");
        assert!(parser.next().is_none());
        parser.feed(b"4;10;2");
        assert!(parser.next().is_none());
        parser.feed(b"0M");
        assert_eq!(
            parser.next(),
            Some(Event::Mouse(MouseEvent {
                kind: MouseEventKind::ScrollUp,
                column: 9,
                row: 19,
                modifiers: KeyModifiers::NONE,
            }))
        );
    }
    #[test]
    fn sgr_mouse_modifiers_and_unsupported_actions_discarded() {
        let mut parser = Parser::default();
        // Wheel down (65) + Control (16) = 81
        parser.feed(b"\x1b[<81;15;25M");
        assert_eq!(
            parser.next(),
            Some(Event::Mouse(MouseEvent {
                kind: MouseEventKind::ScrollDown,
                column: 14,
                row: 24,
                modifiers: KeyModifiers::CONTROL,
            }))
        );
        // Unsupported actions like clicks/releases must not become key events or draft text
        parser.feed(b"\x1b[<0;10;20M\x1b[<0;10;20m");
        assert!(parser.next().is_none());
        parser.feed(b"x");
        assert_eq!(
            parser.next(),
            Some(key(KeyCode::Char('x'), KeyModifiers::NONE))
        );
    }
    #[test]
    fn sgr_invalid_reports_do_not_become_keys() {
        let mut parser = Parser::default();
        parser.feed(b"\x1b[<bad;paramsM\x1b[<64M");
        assert_eq!(parser.next(), None);
        parser.feed(b"\r");
        assert_eq!(parser.next(), Some(key(KeyCode::Enter, KeyModifiers::NONE)));
    }
}
