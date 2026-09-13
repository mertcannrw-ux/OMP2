use crate::definition::{HostResponse, ToolError};
use crate::read::ReadSelector;
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const LIMIT: usize = 100_000;
const POSIX_MARKER: &[u8] = b"omp2-posix-read-v1\n";

fn failure(code: &str, message: impl Into<String>) -> ToolError {
    omp_types::StructuredError::new(code, message, false).into()
}

fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// Parsed independently of the shell: no authority text can become a client option.
struct Target<'a> {
    destination: &'a str,
    port: Option<u16>,
    path: String,
}

impl<'a> Target<'a> {
    fn parse(target: &'a str) -> Result<Self, ToolError> {
        let (authority, path) = target.split_once('/').ok_or_else(|| {
            failure(
                "ssh_path",
                "SSH reads require ssh://[user@]host[:port]/absolute/path",
            )
        })?;
        let (destination, port) = match authority.rsplit_once(':') {
            Some((host, port)) => (
                host,
                Some(
                    port.parse::<u16>()
                        .ok()
                        .filter(|p| *p > 0)
                        .ok_or_else(|| failure("ssh_path", "Invalid SSH port"))?,
                ),
            ),
            None => (authority, None),
        };
        if destination.is_empty()
            || destination.starts_with('-')
            || destination.split('@').count() > 2
            || destination
                .split('@')
                .any(|part| part.is_empty() || part.starts_with('-'))
            || !destination
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_' | b'@'))
        {
            return Err(failure("ssh_path", "Invalid SSH destination"));
        }
        let mut decoded = Vec::with_capacity(path.len() + 1);
        decoded.push(b'/');
        let mut bytes = path.bytes();
        while let Some(byte) = bytes.next() {
            if byte == b'%' {
                let a = bytes.next().and_then(|b| (b as char).to_digit(16));
                let b = bytes.next().and_then(|b| (b as char).to_digit(16));
                decoded.push(match (a, b) {
                    (Some(a), Some(b)) => (a * 16 + b) as u8,
                    _ => return Err(failure("ssh_path", "Invalid path percent escape")),
                });
            } else {
                decoded.push(byte);
            }
        }
        let path = String::from_utf8(decoded)
            .map_err(|_| failure("ssh_path", "SSH path must be UTF-8"))?;
        if path.chars().any(char::is_control) {
            return Err(failure(
                "ssh_path",
                "Control characters are forbidden in SSH paths",
            ));
        }
        Ok(Self {
            destination,
            port,
            path,
        })
    }
}

/// The guard kills/reaps on every error path. This is a fixed host-owned SSH adapter,
/// not a way to execute model-provided local commands or inherit provider credentials.
struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub(crate) fn validate(target: &str) -> Result<(), ToolError> {
    Target::parse(target).map(|_| ())
}

pub(crate) fn read(target: &str, selector: &ReadSelector) -> Result<HostResponse, ToolError> {
    let target = Target::parse(target)?;
    if selector.is_img || selector.query.is_some() || selector.subpath.is_some() {
        return Err(failure(
            "ssh_selector",
            "SSH supports bounded text, directory and line/raw reads",
        ));
    }
    let mut command = Command::new("ssh");
    command.current_dir(std::env::temp_dir());
    command.env_clear();
    // OpenSSH configuration and credentials are host-owned; no workspace configuration is loaded.
    for key in [
        "PATH",
        "SystemRoot",
        "WINDIR",
        "USERPROFILE",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "SSH_AUTH_SOCK",
    ] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    command.args([
        "-T",
        "-oBatchMode=yes",
        "-oStrictHostKeyChecking=yes",
        "-oConnectTimeout=10",
        "-oConnectionAttempts=1",
        "-oServerAliveInterval=5",
        "-oServerAliveCountMax=1",
        "-oPermitLocalCommand=no",
        "-oProxyCommand=none",
        "-oProxyJump=none",
        "-oForwardAgent=no",
        "-oClearAllForwardings=yes",
        "-oControlMaster=no",
        "-oControlPath=none",
    ]);
    if let Some(port) = target.port {
        command.args(["-p", &port.to_string()]);
    }
    // POSIX marker is emitted only after the remote shell accepts POSIX syntax.
    // File type checks reject FIFOs/devices; byte/time limits also bound hostile hosts.
    let script = format!(
        "if test -n x; then printf 'omp2-posix-read-v1\\n'; else exit 65; fi; p={}; if test -f \"$p\"; then dd if=\"$p\" bs=100001 count=1 2>/dev/null; elif test -d \"$p\"; then LC_ALL=C ls -lap \"$p\"; else printf 'Not a readable regular file or directory' >&2; exit 66; fi",
        quote(&target.path)
    );
    command
        .arg(target.destination)
        .arg(script)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut process = Process(
        command
            .spawn()
            .map_err(|e| failure("ssh_spawn", e.to_string()))?,
    );
    // `Stdio::piped()` was set above, so `take()` only fails if the stdio
    // was already taken — a programmer error, not runtime state.
    let stdout = process
        .0
        .stdout
        .take()
        .expect("ssh stdout was piped");
    let stderr = process
        .0
        .stderr
        .take()
        .expect("ssh stderr was piped");
    let (send, receive) = mpsc::sync_channel(2);
    for (kind, stream) in [
        (true, Box::new(stdout) as Box<dyn Read + Send>),
        (false, Box::new(stderr) as Box<dyn Read + Send>),
    ] {
        let send = send.clone();
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let result = stream
                .take((LIMIT + POSIX_MARKER.len() + 1) as u64)
                .read_to_end(&mut bytes);
            let _ = send.send((kind, result.map(|_| bytes)));
        });
    }
    drop(send);
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut out = None;
    let mut err = None;
    let mut exit = None;
    while out.is_none() || err.is_none() || exit.is_none() {
        if Instant::now() >= deadline {
            return Err(failure("ssh_timeout", "SSH read exceeded 20 seconds"));
        }
        if exit.is_none() {
            exit = process
                .0
                .try_wait()
                .map_err(|e| failure("ssh_wait", e.to_string()))?;
        }
        match receive.recv_timeout(Duration::from_millis(10)) {
            Ok((kind, bytes)) => {
                let bytes = bytes.map_err(|e| failure("ssh_read", e.to_string()))?;
                if kind {
                    out = Some(bytes);
                } else {
                    err = Some(bytes);
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                std::thread::sleep(Duration::from_millis(10))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        if out
            .as_ref()
            .is_some_and(|bytes| bytes.len() > LIMIT + POSIX_MARKER.len())
        {
            let _ = process.0.kill();
            exit = Some(
                process
                    .0
                    .wait()
                    .map_err(|e| failure("ssh_wait", e.to_string()))?,
            );
            break;
        }
    }
    let bytes = out.unwrap_or_default();
    let truncated = bytes.len() > LIMIT + POSIX_MARKER.len();
    if !truncated && !exit.is_some_and(|status| status.success()) {
        return Err(failure(
            "ssh_read",
            String::from_utf8_lossy(&err.unwrap_or_default()).into_owned(),
        ));
    }
    let bytes = bytes
        .strip_prefix(POSIX_MARKER)
        .ok_or_else(|| failure("ssh_shell", "Remote endpoint did not confirm a POSIX shell"))?;
    let text = String::from_utf8_lossy(&bytes[..bytes.len().min(LIMIT)]);
    let mut response = HostResponse::success(if let Some(lines) = &selector.lines {
        if truncated {
            return Err(failure(
                "ssh_range",
                "Remote file exceeds the bounded range source budget",
            ));
        }
        let all: Vec<_> = text.lines().collect();
        let (selected, _) = lines.select_lines_bounded(&all, usize::MAX, LIMIT);
        selected
            .into_iter()
            .map(|(n, line)| {
                if selector.raw {
                    line.to_owned()
                } else {
                    format!("{n}:{line}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        text.into_owned()
    });
    response.output.truncated = truncated;
    response.output.payload = Some(
        serde_json::json!({"resolved_path":format!("ssh://{}{}", target.destination, target.path)}),
    );
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn destination_cannot_inject_options_or_shell_syntax() {
        for input in [
            "-oProxyCommand=evil/a",
            "host;touch/a",
            "user@-host/a",
            "user@@host/a",
            "host:0/a",
            "host/a%00b",
        ] {
            assert!(Target::parse(input).is_err(), "{input}");
        }
        let target = Target::parse("reader@localhost:2222/a%27b%20c").unwrap();
        assert_eq!(target.path, "/a'b c");
        assert_eq!(quote(&target.path), "'/a'\\''b c'");
    }
}
