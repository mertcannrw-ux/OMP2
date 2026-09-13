use omp_render::{
    RichText, Run, Style, grapheme_cluster_width, grapheme_clusters, sanitize_text,
    str_visible_width,
};

#[test]
fn test_ansi_injection_prevention_and_control_sanitization() {
    // 1. CSI escape sequences stripped
    let raw_csi = "\x1b[31;1mDangerous Red\x1b[0m Text";
    let sanitized_csi = sanitize_text(raw_csi);
    assert_eq!(sanitized_csi, "Dangerous Red Text");
    assert!(!sanitized_csi.contains('\x1b'));

    // 2. OSC escape sequences stripped (both BEL and ST endings)
    let raw_osc1 = "\x1b]0;Injected Title\x07Body text";
    assert_eq!(sanitize_text(raw_osc1), "Body text");

    let raw_osc2 = "\x1b]2;Another Title\x1b\\Normal content";
    assert_eq!(sanitize_text(raw_osc2), "Normal content");

    // 3. Dangerous control codes stripped (backspace, bell, null, DEL)
    let raw_ctrl = "Hello\x08\x07\x00\x7fWorld";
    assert_eq!(sanitize_text(raw_ctrl), "HelloWorld");

    // 4. Unicode Bidirectional Overrides stripped (Trojan source prevention)
    let raw_bidi = "safe\u{202E}txt\u{202C}end";
    let sanitized_bidi = sanitize_text(raw_bidi);
    assert_eq!(sanitized_bidi, "safetxtend");

    // 5. Standard formatting whitespace preserved
    let raw_ws = "Line 1\nLine 2\r\n\tIndented";
    assert_eq!(sanitize_text(raw_ws), "Line 1\nLine 2\r\n\tIndented");

    // 6. RichText and Run automatically sanitize on push and construction
    let mut rt = RichText::new();
    rt.push(Style::new(), "\x1b[32mClean\x1b[0m");
    assert_eq!(rt.plain_text(), "Clean");

    let run = Run::new(Style::new(), "\x1b[2JClear screen\x08");
    assert_eq!(run.text, "Clear screen");
    assert!(!run.text.contains('\x1b'));
}

#[test]
fn test_grapheme_cluster_measurement() {
    // ASCII characters: width 1 each
    assert_eq!(str_visible_width("hello"), 5);

    // Combining acute accent: 'e' + '\u{0301}' (é) is 1 visible cell on terminal
    let e_acute = "e\u{0301}";
    let clusters = grapheme_clusters(e_acute);
    assert_eq!(clusters.len(), 1);
    assert_eq!(grapheme_cluster_width(clusters[0]), 1);
    assert_eq!(str_visible_width(e_acute), 1);

    // East Asian Wide characters (CJK): width 2 each
    let cjk = "测试";
    assert_eq!(str_visible_width(cjk), 4);

    // Emoji base character: width 2
    let rocket = "🚀";
    assert_eq!(str_visible_width(rocket), 2);

    // Emoji with skin tone modifier: thumb up + medium skin tone = width 2 (not 4)
    let thumbs_up_skin = "👍🏽";
    let thumb_clusters = grapheme_clusters(thumbs_up_skin);
    assert_eq!(thumb_clusters.len(), 1);
    assert_eq!(str_visible_width(thumbs_up_skin), 2);

    // Emoji with ZWJ sequence (family: man + ZWJ + woman + ZWJ + boy) = width 2 (not 8)
    let family = "👨‍👩‍👦";
    let family_clusters = grapheme_clusters(family);
    assert_eq!(family_clusters.len(), 1);
    assert_eq!(str_visible_width(family), 2);

    // Regional indicator pair (Flag of Japan: JP) = width 2 (not 4)
    let flag_jp = "🇯🇵";
    let flag_clusters = grapheme_clusters(flag_jp);
    assert_eq!(flag_clusters.len(), 1);
    assert_eq!(str_visible_width(flag_jp), 2);
}

#[test]
fn test_grapheme_aware_truncation() {
    let mut rt = RichText::new();
    rt.push(Style::new(), "Status: 🚀 Launched!");

    // Truncate width to budget: ensures graphemes are never sliced in half
    let truncated = rt.truncate_width(12, Some("..."));
    let text = truncated.plain_text();
    assert!(truncated.visible_width() <= 12);
    assert!(text.ends_with("..."));
    assert!(!text.contains('\x1b'));
}
