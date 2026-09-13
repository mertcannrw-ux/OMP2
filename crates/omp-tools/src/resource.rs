use crate::definition::{HostResponse, ToolError};
use crate::read::LineSelector;
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::Duration;

const LIMIT: usize = 100_000;
fn execution(error: impl std::fmt::Display) -> ToolError {
    ToolError::Execution {
        message: error.to_string(),
        details: None,
    }
}
fn validation(message: impl Into<String>) -> ToolError {
    ToolError::Validation {
        message: message.into(),
        details: None,
    }
}

/// Reject fetch targets that resolve to link-local, loopback-excepted, or
/// cloud-metadata endpoints before any approval or network I/O.
///
/// This is a syntactic guard, not a DNS-resolution guarantee: hostnames that
/// *resolve* to internal addresses (DNS rebinding) are still the caller's
/// responsibility, which is why `authorize_host_read` approval is required
/// separately at the call site.
pub fn validate_public_url(url: &str) -> Result<(), ToolError> {
    let authority = url
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(url)
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("");
    // Strip userinfo for host extraction (its presence is rejected below).
    let after_userinfo = authority.rsplit('@').next().unwrap_or(authority);
    // Bracketed IPv6 literal, optionally with port: `[::1]:8080`.
    let host = if let Some(rest) = after_userinfo.strip_prefix('[') {
        rest.split(']').next().unwrap_or("").to_ascii_lowercase()
    } else {
        after_userinfo
            .split(':')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase()
    };
    if host.is_empty() {
        return Err(validation("Resource URL has no host"));
    }
    // Userinfo in the URL would leak credentials into logs/journal.
    if url.split_once("://").is_some_and(|(_, rest)| {
        rest.split(['/', '?', '#']).next().is_some_and(|a| a.contains('@'))
    }) {
        return Err(validation("Resource URL must not embed credentials"));
    }
    const BLOCKED_EXACT: &[&str] = &[
        "localhost",
        "metadata.google.internal",
        "metadata.google.com",
    ];
    if BLOCKED_EXACT.contains(&host.as_str()) || host.ends_with(".local") {
        return Err(validation(format!(
            "Resource host '{host}' is not fetchable by tools"
        )));
    }
    // Numeric IPv4: block loopback, link-local, private, CGNAT, and metadata.
    let octets: Option<[u8; 4]> = {
        let parts: Vec<&str> = host.split('.').collect();
        if parts.len() == 4 {
            let mut out = [0u8; 4];
            let mut ok = true;
            for (i, part) in parts.iter().enumerate() {
                match part.parse::<u8>() {
                    Ok(v) => out[i] = v,
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            ok.then_some(out)
        } else {
            None
        }
    };
    if let Some([a, b, ..]) = octets {
        let blocked = a == 127
            || a == 10
            || (a == 172 && (16..32).contains(&b))
            || (a == 192 && b == 168)
            || (a == 169 && b == 254)
            || (a == 100 && (64..128).contains(&b))
            || a == 0;
        if blocked {
            return Err(validation(format!(
                "Resource host '{host}' is link-local or internal and not fetchable by tools"
            )));
        }
    }
    // IPv6 loopback / link-local / unique-local. The `:` guard keeps this
    // from over-blocking public hostnames like `fc2.com`.
    let is_ipv6 = host.contains(':');
    if is_ipv6
        && (host == "::1"
            || host.starts_with("fe80:")
            || host.starts_with("fc")
            || host.starts_with("fd"))
    {
        return Err(validation(format!(
            "Resource host '{host}' is link-local or internal and not fetchable by tools"
        )));
    }
    // IPv4-mapped IPv6 (`::ffff:127.0.0.1`): validate the embedded IPv4 tail
    // with the same rules as a plain IPv4 literal.
    if is_ipv6 && let Some(last_colon) = host.rfind(':') {
        let tail = &host[last_colon + 1..];
        let tail_octets: Option<[u8; 4]> = {
            let parts: Vec<&str> = tail.split('.').collect();
            if parts.len() == 4 {
                let mut out = [0u8; 4];
                let mut ok = true;
                for (i, part) in parts.iter().enumerate() {
                    match part.parse::<u8>() {
                        Ok(v) => out[i] = v,
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                ok.then_some(out)
            } else {
                None
            }
        };
        if let Some([a, b, ..]) = tail_octets {
            let blocked = a == 127
                || a == 10
                || (a == 172 && (16..32).contains(&b))
                || (a == 192 && b == 168)
                || (a == 169 && b == 254)
                || (a == 100 && (64..128).contains(&b))
                || a == 0;
            if blocked {
                return Err(validation(format!(
                    "Resource host '{host}' embeds a link-local or internal IPv4 address"
                )));
            }
        }
    }
    Ok(())
}

pub fn fetch(url: &str, github: bool, raw: bool) -> Result<HostResponse, ToolError> {
    // Defense in depth: direct `fetch` callers (e.g. GitHub issue/PR reads)
    // get the same link-local guard as the tool path.
    if !github {
        validate_public_url(url)?;
    }
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(20))
        .redirects(0)
        .build();
    let mut current = url.to_owned();
    for hop in 0..=3 {
        let mut request = agent.get(&current).set("User-Agent", "omp2/0.1");
        if github {
            request = request.set("Accept", "application/vnd.github+json");
            if let Ok(token) = std::env::var("GITHUB_TOKEN") {
                request = request.set("Authorization", &format!("Bearer {token}"));
            }
        }
        let response = request.call().map_err(execution)?;
        if (300..400).contains(&response.status()) {
            if github || hop == 3 {
                return Err(execution("Resource redirect not permitted"));
            }
            let location = response
                .header("Location")
                .ok_or_else(|| execution("Redirect missing Location"))?;
            let base = ureq::get(&current).url().to_owned();
            current = if location.starts_with("https://") || location.starts_with("http://") {
                // Absolute redirect targets re-enter the public-URL guard so a
                // benign host cannot bounce the fetcher to link-local.
                validate_public_url(location)?;
                location.to_owned()
            } else if location.starts_with('/') && !location.starts_with("//") {
                let Some(scheme_end) = base.find("://") else {
                    return Err(execution("Redirect base URL has no scheme"));
                };
                let authority_end = scheme_end + 3;
                let end = base[authority_end..]
                    .find('/')
                    .map(|index| authority_end + index)
                    .unwrap_or(base.len());
                format!("{}{}", &base[..end], location)
            } else {
                return Err(execution("Unsupported relative resource redirect"));
            };
            continue;
        }
        let html = response
            .header("Content-Type")
            .is_some_and(|value| value.contains("text/html"));
        let mut bytes = Vec::new();
        response
            .into_reader()
            .take(LIMIT as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(execution)?;
        let truncated = bytes.len() > LIMIT;
        bytes.truncate(LIMIT);
        let content = String::from_utf8_lossy(&bytes).into_owned();
        let content = if html && !raw {
            html2text::from_read(content.as_bytes(), 100)
        } else {
            content
        };
        let mut output = HostResponse::success(content);
        output.output.truncated = truncated;
        return Ok(output);
    }
    Err(execution("Resource redirect limit"))
}

/// Hashes and counts without retaining the file, then reads only selected lines.
pub fn text_file(
    path: &Path,
    label: &str,
    selector: Option<&LineSelector>,
    raw: bool,
    conflicts: bool,
) -> Result<HostResponse, ToolError> {
    let mut file = std::fs::File::open(path).map_err(execution)?;
    let mut digest = Sha256::new();
    let mut chunk = [0u8; 65536];
    let mut total = 0usize;
    let mut last = None;
    loop {
        let n = file.read(&mut chunk).map_err(execution)?;
        if n == 0 {
            break;
        }
        digest.update(&chunk[..n]);
        total += chunk[..n].iter().filter(|byte| **byte == b'\n').count();
        last = Some(chunk[n - 1]);
    }
    if last.is_some_and(|byte| byte != b'\n') {
        total += 1;
    }
    let hash = digest.finalize();
    let mut out = if raw {
        String::new()
    } else {
        format!("[{label}#{:02X}{:02X}]\n", hash[0], hash[1])
    };
    let mut reader = BufReader::new(std::fs::File::open(path).map_err(execution)?);
    let mut line = Vec::new();
    let mut in_conflict = false;
    let mut truncated = false;
    for number in 1..=total {
        line.clear();
        let mut oversized = false;
        loop {
            let buffer = reader.fill_buf().map_err(execution)?;
            if buffer.is_empty() {
                break;
            }
            let end = buffer
                .iter()
                .position(|byte| *byte == b'\n')
                .map(|index| index + 1)
                .unwrap_or(buffer.len());
            let complete = buffer[end - 1] == b'\n';
            let keep = end.min((LIMIT + 1).saturating_sub(line.len()));
            line.extend_from_slice(&buffer[..keep]);
            oversized |= keep < end;
            reader.consume(end);
            if complete {
                break;
            }
        }
        let text = String::from_utf8_lossy(&line);
        let selected = match selector {
            None => true,
            Some(LineSelector::Single(n)) => number == *n,
            Some(LineSelector::FromLine(n)) => number >= *n,
            Some(LineSelector::Inclusive(start, end)) => (*start..=*end).contains(&number),
            Some(LineSelector::Count(start, count)) => number >= *start && number - *start < *count,
            Some(LineSelector::Disjoint(ranges)) => ranges
                .iter()
                .any(|(start, end)| (*start..=*end).contains(&number)),
            Some(LineSelector::Tail(n)) => number > total.saturating_sub(*n),
        };
        if text.starts_with("<<<<<<<") {
            in_conflict = true;
        }
        let visible = selected && (!conflicts || in_conflict);
        if text.starts_with(">>>>>>>") {
            in_conflict = false;
        }
        if !visible {
            continue;
        }
        let prefix = if raw {
            String::new()
        } else {
            format!("{number}:")
        };
        let remaining = LIMIT.saturating_sub(out.len() + prefix.len());
        let mut cut = text.len().min(remaining);
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        out.push_str(&prefix);
        out.push_str(&text[..cut]);
        if oversized || cut < text.len() {
            truncated = true;
            break;
        }
    }
    let mut response = HostResponse::success(out);
    response.output.truncated = truncated;
    Ok(response)
}

pub fn preview_image(path: &Path, svg: bool) -> Result<(Vec<u8>, u32, u32), ToolError> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(execution)?
        .take(16_000_001)
        .read_to_end(&mut bytes)
        .map_err(execution)?;
    if bytes.len() > 16_000_000 {
        return Err(execution("Image input exceeds 16 MB"));
    }
    if svg {
        let mut options = resvg::usvg::Options::default();
        options.fontdb_mut().load_system_fonts();
        let tree = resvg::usvg::Tree::from_data(&bytes, &options).map_err(execution)?;
        let size = tree.size();
        let scale = (1600.0 / size.width().max(size.height())).min(1.0);
        let width = (size.width() * scale).ceil().max(1.0) as u32;
        let height = (size.height() * scale).ceil().max(1.0) as u32;
        let mut pixels = resvg::tiny_skia::Pixmap::new(width, height)
            .ok_or_else(|| execution("Image allocation failed"))?;
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::from_scale(scale, scale),
            &mut pixels.as_mut(),
        );
        return Ok((pixels.encode_png().map_err(execution)?, width, height));
    }
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(execution)?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16000);
    limits.max_image_height = Some(16000);
    limits.max_alloc = Some(128_000_000);
    reader.limits(limits);
    let image = reader.decode().map_err(execution)?.thumbnail(1600, 1600);
    let (width, height) = (image.width(), image.height());
    let mut png = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut png, image::ImageFormat::Png)
        .map_err(execution)?;
    Ok((png.into_inner(), width, height))
}

#[cfg(test)]
mod tests {
    use super::validate_public_url;

    #[test]
    fn test_public_urls_pass() {
        assert!(validate_public_url("https://example.com/x").is_ok());
        assert!(validate_public_url("https://api.github.com/repos/a/b").is_ok());
        assert!(validate_public_url("http://example.com:8080/path?q=1").is_ok());
        // Public hostname starting with "fc" must not be blocked.
        assert!(validate_public_url("https://fc2.com/blog").is_ok());
    }

    #[test]
    fn test_internal_urls_rejected() {
        for url in [
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.169.254",
            "http://localhost/x",
            "http://127.0.0.1:8080/x",
            "http://10.0.0.5/x",
            "http://192.168.1.1/x",
            "http://172.16.0.1/x",
            "http://100.64.0.1/x",
            "http://[::1]/x",
            "http://[fe80::1]/x",
            "http://[fd00::1]/x",
            "http://user:pass@example.com/x",
            "http://metadata.google.internal/",
            "http://x.local/y",
            "http://0.0.0.0/x",
            // IPv4-mapped IPv6 must not smuggle loopback past the guard.
            "http://[::ffff:127.0.0.1]/x",
            "http://[::ffff:10.1.2.3]/x",
        ] {
            assert!(validate_public_url(url).is_err(), "should reject {url}");
        }
        // ...while a mapped *public* address stays fetchable.
        assert!(validate_public_url("http://[::ffff:8.8.8.8]/x").is_ok());
    }
}
