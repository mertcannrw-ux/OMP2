use omp_types::StructuredError;
use serde::{Deserialize, Serialize};

/// Detection information for a repetition loop in model output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepetitionDetection {
    pub repeated_phrase: String,
    pub repeat_count: usize,
    pub byte_offset: usize,
}

/// Normalized assistant turn separating extended thinking from final response.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedTurn {
    pub thinking: Option<String>,
    pub content: String,
    pub has_thinking: bool,
}

/// Corrective repair for malformed JSON output with bounded attempts.
///
/// Handles common LLM generation errors:
/// 1. Markdown code fences (` ```json `)
/// 2. Trailing commas before closing braces/brackets
/// 3. Unclosed quotes
/// 4. Missing closing brackets/braces
pub fn repair_malformed_json(
    raw: &str,
    max_attempts: usize,
) -> Result<serde_json::Value, StructuredError> {
    let mut candidate = raw.trim().to_string();

    // Fast path
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&candidate) {
        return Ok(val);
    }

    let bounded_steps = max_attempts.min(10);
    for _ in 0..bounded_steps {
        // Step 1: Strip markdown fences if present
        if candidate.starts_with("```") {
            if let Some(first_newline) = candidate.find('\n') {
                candidate = candidate[first_newline + 1..].to_string();
            }
            if let Some(last_fence) = candidate.rfind("```") {
                candidate = candidate[..last_fence].to_string();
            }
            candidate = candidate.trim().to_string();
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&candidate) {
                return Ok(val);
            }
        }

        // Step 2: Strip trailing commas: , } -> } and , ] -> ]
        let cleaned = remove_trailing_commas(&candidate);
        if cleaned != candidate {
            candidate = cleaned;
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&candidate) {
                return Ok(val);
            }
        }

        // Step 3: Check unclosed quote
        let quote_count = count_unescaped_quotes(&candidate);
        if !quote_count.is_multiple_of(2) {
            candidate.push('"');
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&candidate) {
                return Ok(val);
            }
        }

        // Step 4: Balance unclosed braces and brackets
        let balanced = balance_delimiters(&candidate);
        if balanced != candidate {
            candidate = balanced;
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&candidate) {
                return Ok(val);
            }
        }
    }

    Err(StructuredError::new(
        "malformed_json_unrepairable",
        format!("Failed to repair malformed JSON within {max_attempts} attempts: '{raw}'"),
        true,
    ))
}

fn remove_trailing_commas(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;

    while let Some(ch) = chars.next() {
        if escaped {
            escaped = false;
            out.push(ch);
            continue;
        }

        if ch == '\\' {
            escaped = true;
            out.push(ch);
            continue;
        }

        if ch == '"' {
            in_string = !in_string;
            out.push(ch);
            continue;
        }

        if !in_string && ch == ',' {
            // Peek forward past whitespace to see if next token is } or ]
            let mut forward = chars.clone();
            let mut is_trailing = false;
            while let Some(&next_ch) = forward.peek() {
                if next_ch.is_whitespace() {
                    forward.next();
                } else {
                    if next_ch == '}' || next_ch == ']' {
                        is_trailing = true;
                    }
                    break;
                }
            }
            if is_trailing {
                // Skip the comma
                continue;
            }
        }

        out.push(ch);
    }
    out
}

fn count_unescaped_quotes(s: &str) -> usize {
    let mut count = 0;
    let mut escaped = false;
    for ch in s.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '"' {
            count += 1;
        }
    }
    count
}

fn balance_delimiters(s: &str) -> String {
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for ch in s.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            match ch {
                '{' => stack.push('}'),
                '[' => stack.push(']'),
                '}' => {
                    if stack.last() == Some(&'}') {
                        stack.pop();
                    }
                }
                ']'
                    if stack.last() == Some(&']') => {
                        stack.pop();
                    }
                _ => {}
            }
        }
    }

    let mut out = s.to_string();
    while let Some(closing) = stack.pop() {
        out.push(closing);
    }
    out
}

/// Detects repetitive loops in generated model text.
/// Phrase lengths and reported offsets are in bytes; phrases must be valid UTF-8.
pub fn detect_repetition_loop(
    text: &str,
    min_phrase_len: usize,
    min_repeats: usize,
) -> Option<RepetitionDetection> {
    let min_bytes = min_phrase_len.checked_mul(min_repeats)?;
    if min_bytes == 0 || text.len() < min_bytes {
        return None;
    }

    // Bound scan cost on adversarial inputs: inspect only the first 64 KiB and
    // enforce an inner-iteration budget so worst case stays near-linear.
    const MAX_SCAN_BYTES: usize = 64 * 1024;
    const MAX_ITERATIONS: usize = 2_000_000;
    let scan_len = text.len().min(MAX_SCAN_BYTES);
    // Shrink to a valid UTF-8 boundary since phrase offsets are in bytes.
    let mut scan_end = scan_len;
    while scan_end > 0 && !text.is_char_boundary(scan_end) {
        scan_end -= 1;
    }
    let text = &text[..scan_end];
    let len = text.len();
    let bytes = text.as_bytes();
    // Test phrase lengths from min_phrase_len up to len / min_repeats
    let max_phrase_len = (len / min_repeats).min(100);

    let mut iterations: usize = 0;
    for phrase_len in (min_phrase_len..=max_phrase_len).rev() {
        for (i, _) in text.char_indices() {
            iterations += 1;
            if iterations > MAX_ITERATIONS {
                return None;
            }
            if i > len - phrase_len * min_repeats {
                break;
            }
            let Some(phrase) = text.get(i..i + phrase_len) else {
                continue;
            };
            // Check consecutive matches
            let mut repeats = 1;
            let mut next_pos = i + phrase_len;
            while next_pos + phrase_len <= len
                && &bytes[next_pos..next_pos + phrase_len] == phrase.as_bytes()
            {
                repeats += 1;
                next_pos += phrase_len;
            }

            if repeats >= min_repeats {
                return Some(RepetitionDetection {
                    repeated_phrase: phrase.to_string(),
                    repeat_count: repeats,
                    byte_offset: i,
                });
            }
        }
    }

    None
}

/// Normalizes model output containing `<think>...</think>` or `<thought>...</thought>` tags.
pub fn normalize_thinking_tokens(text: &str) -> NormalizedTurn {
    let mut thinking_parts = Vec::new();
    let mut clean_content = String::new();

    let mut cursor = 0;
    while cursor < text.len() {
        let rest = &text[cursor..];
        let think_start = rest
            .find("<think>")
            .map(|idx| (idx, "<think>", "</think>"))
            .or_else(|| {
                rest.find("<thought>")
                    .map(|idx| (idx, "<thought>", "</thought>"))
            });

        if let Some((start_rel, open_tag, close_tag)) = think_start {
            let abs_start = cursor + start_rel;
            clean_content.push_str(&text[cursor..abs_start]);

            let inner_start = abs_start + open_tag.len();
            if let Some(close_rel) = text[inner_start..].find(close_tag) {
                let inner_end = inner_start + close_rel;
                thinking_parts.push(text[inner_start..inner_end].trim().to_string());
                cursor = inner_end + close_tag.len();
            } else {
                // Unclosed thinking tag
                thinking_parts.push(text[inner_start..].trim().to_string());
                cursor = text.len();
            }
        } else {
            clean_content.push_str(rest);
            break;
        }
    }

    let has_thinking = !thinking_parts.is_empty();
    let thinking = if has_thinking {
        Some(thinking_parts.join("\n\n"))
    } else {
        None
    };

    NormalizedTurn {
        thinking,
        content: clean_content.trim().to_string(),
        has_thinking,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multilingual_reply_is_not_a_repetition_loop() {
        let reply = "Merhaba! Size nasıl yardımcı olabilirim? Bugün hangi projede çalışıyoruz?";
        assert_eq!(detect_repetition_loop(reply, 12, 3), None);
    }

    #[test]
    fn repeated_unicode_phrase_keeps_exact_text_and_byte_offset() {
        let prefix = "前文:";
        let phrase = "Çağrı 日本語 𐐷 ";
        let text = format!("{prefix}{}", phrase.repeat(3));
        assert_eq!(
            detect_repetition_loop(&text, 12, 3),
            Some(RepetitionDetection {
                repeated_phrase: phrase.into(),
                repeat_count: 3,
                byte_offset: prefix.len(),
            })
        );
    }

    #[test]
    fn impossible_repetition_thresholds_return_no_match() {
        for (length, count) in [(0, 3), (12, 0), (usize::MAX, 3)] {
            assert_eq!(detect_repetition_loop("some output", length, count), None);
        }
    }
}
