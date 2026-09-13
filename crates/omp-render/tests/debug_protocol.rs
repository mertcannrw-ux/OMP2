use omp_render::{BlockMode, DebugSession};
use omp_types::ElementSnapshot;

#[test]
fn test_debug_session_multi_instance_off_screen_protocol() {
    let mut session_a = DebugSession::new(80, 24);
    let mut session_b = DebugSession::new(100, 30);

    let mut elem_user = ElementSnapshot::new(omp_types::ElementId::new("elem-1").unwrap(), "user");
    elem_user.text = "Hello from session".to_string();

    let block_a = session_a
        .admit_element_block("blk-a", BlockMode::Mutable, "user", &elem_user)
        .unwrap();

    let block_b = session_b
        .admit_element_block("blk-b", BlockMode::Mutable, "user", &elem_user)
        .unwrap();

    // Verify session A off-screen frame
    let frame_a = session_a.render_frame().unwrap();
    assert!(!frame_a.is_empty());

    // Verify session B off-screen frame
    let frame_b = session_b.render_frame().unwrap();
    assert!(!frame_b.is_empty());

    // Verify assertion helper: no premature mutable history
    session_a
        .assert_no_premature_mutable_history(&block_a)
        .unwrap();
    session_b
        .assert_no_premature_mutable_history(&block_b)
        .unwrap();

    // Verify assertion helper: resize rebuild preserves logical history
    session_a
        .assert_logical_history_unchanged_on_resize(120, 35)
        .unwrap();

    let snap = session_a.snapshot();
    assert_eq!(snap.display_epoch, 2);
    assert!(!snap.fail_stopped);
}

#[test]
fn partial_terminal_write_halts_future_frames() {
    let mut session = DebugSession::new(80, 10);
    let mut element = ElementSnapshot::new(omp_types::ElementId::new("text").unwrap(), "user");
    element.text = "first\nsecond\nthird".into();
    session
        .admit_element_block("block", BlockMode::Mutable, "user", &element)
        .unwrap();
    session.terminal.set_fail_after(1);
    assert!(session.render_frame().is_err());
    assert!(session.snapshot().fail_stopped);
    assert_eq!(session.scheduler.forensic_log.len(), 1);
    let writes = session.terminal.total_writes_attempted;
    assert!(session.render_frame().is_err());
    assert_eq!(session.terminal.total_writes_attempted, writes);
    assert!(session.snapshot().logical_history.is_empty());
}

#[test]
fn title_row_and_grapheme_wrapping_preserve_readable_output() {
    use omp_render::{Component, RichText, SemanticColor, StringOutSink, WrapMode, wrap_rich_text};
    let row = Component::row(vec![
        Component::plain_text("Read file"),
        Component::badge(SemanticColor::Success, "done"),
    ])
    .unwrap();
    let mut out = StringOutSink::new();
    row.render_to_sink(&mut out, 0).unwrap();
    assert_eq!(out.as_str(), "Read file [done]\n");
    let text = RichText::from_plain("abcde\u{301}fg");
    let lines = wrap_rich_text(&text, 3, WrapMode::CharacterWrap);
    assert_eq!(
        lines.iter().map(RichText::plain_text).collect::<Vec<_>>(),
        ["abc", "de\u{301}f", "g"]
    );
}

#[test]
fn pacing_backpressures_without_losing_utf8_or_skipping_delay() {
    use omp_render::{StreamPacer, StreamPacingConfig};
    let mut pacer = StreamPacer::new(StreamPacingConfig {
        min_delay_ms: 10,
        chunk_char_target: 2,
        burst_limit: 8,
    });
    let text = "abcédefghijk";
    let mut remaining = text;
    let now = std::time::Instant::now();
    let accepted = pacer.push_chunk(remaining);
    assert!(accepted < text.len());
    remaining = &remaining[accepted..];
    let mut drawn = pacer.next_drawable_chunk_at(now).unwrap();
    assert!(pacer.next_drawable_chunk_at(now).is_none());
    for tick in 1..20 {
        remaining = &remaining[pacer.push_chunk(remaining)..];
        if let Some(chunk) =
            pacer.next_drawable_chunk_at(now + std::time::Duration::from_millis(tick * 10))
        {
            assert!(chunk.chars().count() <= 2);
            drawn.push_str(&chunk);
        }
    }
    assert_eq!(drawn, text);
    assert!(remaining.is_empty());
    assert!(!pacer.has_buffered());
}
