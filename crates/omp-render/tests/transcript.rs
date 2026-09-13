use omp_render::{
    BlockId, BlockMode, BlockState, ReplayGateState, ResizePolicy, TranscriptError, TranscriptScheduler,
};
use omp_types::StructuredError;

#[test]
fn test_block_lifecycle_and_exact_committed_history() {
    let mut scheduler = TranscriptScheduler::new(80, 24);

    let b1 = BlockId::new("b1");
    scheduler
        .admit_block(b1.clone(), BlockMode::Mutable, "user")
        .unwrap();
    scheduler.activate_block(&b1).unwrap();

    // Mutable snapshot update
    scheduler
        .update_snapshot(&b1, vec!["user turn 1".to_string()])
        .unwrap();

    // While active, mutable block emits ZERO rows to logical history
    assert_eq!(scheduler.logical_history.committed_line_count(), 0);
    assert_eq!(scheduler.logical_history.commit_frontier(), 0);

    // Finalize block (writes nothing to history)
    scheduler.finalize_block(&b1).unwrap();
    assert_eq!(scheduler.logical_history.committed_line_count(), 0);

    // Retire block (commits into logical history exactly once)
    scheduler.retire_block(&b1).unwrap();
    assert_eq!(scheduler.logical_history.committed_line_count(), 1);
    assert_eq!(scheduler.logical_history.commit_frontier(), 1);
    assert_eq!(
        scheduler.logical_history.all_lines(None),
        vec!["user turn 1"]
    );
}

#[test]
fn test_append_only_monotonicity_and_streaming() {
    let mut scheduler = TranscriptScheduler::new(80, 24);

    let b2 = BlockId::new("b2");
    scheduler
        .admit_block(b2.clone(), BlockMode::AppendOnly, "assistant")
        .unwrap();
    scheduler.activate_block(&b2).unwrap();

    // Valid extension
    scheduler
        .update_snapshot(&b2, vec!["line 1".to_string()])
        .unwrap();
    scheduler
        .update_snapshot(&b2, vec!["line 1".to_string(), "line 2".to_string()])
        .unwrap();

    // Violation: trying to shrink or change existing lines
    let err = scheduler
        .update_snapshot(&b2, vec!["line 1 modified".to_string()])
        .unwrap_err();
    assert!(matches!(
        err,
        TranscriptError::PrefixMonotonicityViolated { .. }
    ));

    // Advance streaming prefix
    let prefix = scheduler.stream_stable_prefix(&b2, 1).unwrap();
    assert_eq!(prefix, 1);

    // Logical history projection includes streamed prefix of active append-only head
    let active_blk = scheduler.blocks.iter().find(|b| b.id == b2);
    let all_lines = scheduler.logical_history.all_lines(active_blk);
    assert_eq!(all_lines, vec!["line 1"]);

    // Cannot stream on mutable block
    scheduler.finalize_block(&b2).unwrap();
    scheduler.retire_block(&b2).unwrap();
    let b_mut = BlockId::new("b_mut");
    scheduler
        .admit_block(b_mut.clone(), BlockMode::Mutable, "tool")
        .unwrap();
    scheduler.activate_block(&b_mut).unwrap();
    let stream_err = scheduler.stream_stable_prefix(&b_mut, 1).unwrap_err();
    assert!(matches!(
        stream_err,
        TranscriptError::MutableBlockCannotStream(_)
    ));
}

#[test]
fn test_capacity_and_two_row_bridge_elastic_shrink() {
    let mut scheduler = TranscriptScheduler::new(80, 10);

    let b3 = BlockId::new("b3");
    scheduler
        .admit_block(b3.clone(), BlockMode::Mutable, "tool")
        .unwrap();
    scheduler.activate_block(&b3).unwrap();

    // Initial snapshot of 8 lines
    let lines_8: Vec<String> = (1..=8).map(|i| format!("output {i}")).collect();
    scheduler.update_snapshot(&b3, lines_8).unwrap();

    let blk = scheduler.blocks.iter().find(|b| b.id == b3).unwrap();
    assert_eq!(blk.reserved_rows, 8);
    assert_eq!(blk.last_rendered_height, 8);

    // Deep shrink from 8 to 2 lines -> 2-row bridge prevents visual snap:
    // effective_height = 8 - 2 = 6
    let lines_2: Vec<String> = (1..=2).map(|i| format!("output {i}")).collect();
    scheduler.update_snapshot(&b3, lines_2).unwrap();

    let blk = scheduler.blocks.iter().find(|b| b.id == b3).unwrap();
    assert_eq!(blk.last_rendered_height, 6);
    assert_eq!(blk.reserved_rows, 6);

    // Viewport capacity: native_viewport row count never exceeds viewport_height
    assert!(scheduler.native_viewport.len() <= 10);
}

#[test]
fn test_resize_rebuild_preserves_logical_history() {
    let mut scheduler = TranscriptScheduler::new(80, 24);

    let b = BlockId::new("b");
    scheduler
        .admit_block(b.clone(), BlockMode::Mutable, "user")
        .unwrap();
    scheduler.activate_block(&b).unwrap();
    scheduler
        .update_snapshot(&b, vec!["persistent message".to_string()])
        .unwrap();
    scheduler.finalize_block(&b).unwrap();
    scheduler.retire_block(&b).unwrap();

    let history_before = scheduler.logical_history.all_lines(None);
    assert_eq!(scheduler.display_epoch, 1);

    // Perform Rebuild resize
    scheduler.resize(120, 40, ResizePolicy::Rebuild).unwrap();

    let history_after = scheduler.logical_history.all_lines(None);

    // Invariant: logical history is completely unchanged by resize
    assert_eq!(history_before, history_after);
    assert_eq!(scheduler.display_epoch, 2);
    assert_eq!(scheduler.viewport_width, 120);
    assert_eq!(scheduler.viewport_height, 40);
    assert_eq!(scheduler.replay_gate, ReplayGateState::Idle);
}

#[test]
fn test_fail_stop_write_failure_halts_scheduler() {
    let mut scheduler = TranscriptScheduler::new(80, 24);

    let b = BlockId::new("b_fail");
    scheduler
        .admit_block(b.clone(), BlockMode::Mutable, "user")
        .unwrap();
    scheduler.activate_block(&b).unwrap();
    scheduler
        .update_snapshot(&b, vec!["row 1".to_string(), "row 2".to_string()])
        .unwrap();

    assert_eq!(scheduler.native_viewport.len(), 2);

    // Simulate physical write error where only 1 row was accepted
    let write_err = StructuredError::new("terminal_error", "write failed", false);
    scheduler.handle_write_failure(write_err.clone(), 1);

    assert!(scheduler.fail_stopped);
    assert_eq!(scheduler.native_viewport.len(), 0);
    assert_eq!(scheduler.forensic_log.len(), 1);
    assert!(matches!(
        scheduler.replay_gate,
        ReplayGateState::FailedStop {
            forensic_rows_written: 1,
            ..
        }
    ));

    // Any subsequent operation fails immediately with FailStopHalted
    let err = scheduler
        .admit_block(BlockId::new("b_next"), BlockMode::Mutable, "test")
        .unwrap_err();
    assert!(matches!(err, TranscriptError::FailStopHalted(_)));
}
#[test]
fn test_replay_gate_blocks_operations_until_drained() {
    let mut scheduler = TranscriptScheduler::new(80, 24);

    let b1 = BlockId::new("b1");
    scheduler
        .admit_block(b1.clone(), BlockMode::Mutable, "user")
        .unwrap();
    scheduler.activate_block(&b1).unwrap();
    scheduler
        .update_snapshot(&b1, vec!["initial line".to_string()])
        .unwrap();

    // Step 1: Prepare replay
    scheduler.prepare_rebuild_replay(100, 30).unwrap();
    assert!(matches!(
        scheduler.replay_gate,
        ReplayGateState::PreparingReplay { .. }
    ));

    // Any operational call must be blocked by the replay gate!
    let b2 = BlockId::new("b2");
    let err = scheduler
        .admit_block(b2, BlockMode::Mutable, "system")
        .unwrap_err();
    assert!(matches!(err, TranscriptError::ReplayGateViolation(_)));

    let err2 = scheduler
        .update_snapshot(&b1, vec!["new line".to_string()])
        .unwrap_err();
    assert!(matches!(err2, TranscriptError::ReplayGateViolation(_)));

    let err3 = scheduler.finalize_block(&b1).unwrap_err();
    assert!(matches!(err3, TranscriptError::ReplayGateViolation(_)));

    // Step 2: Execute replay writing
    scheduler.execute_rebuild_replay().unwrap();
    assert_eq!(scheduler.replay_gate, ReplayGateState::Idle);
    assert_eq!(scheduler.viewport_width, 100);
    assert_eq!(scheduler.viewport_height, 30);

    // Operations succeed now that gate is Idle
    scheduler
        .update_snapshot(&b1, vec!["updated line".to_string()])
        .unwrap();
}

#[test]
fn test_exact_once_retirement_no_duplicate_logical_rows() {
    let mut scheduler = TranscriptScheduler::new(80, 24);

    let b1 = BlockId::new("b1");
    scheduler
        .admit_block(b1.clone(), BlockMode::AppendOnly, "assistant")
        .unwrap();
    scheduler.activate_block(&b1).unwrap();

    // Append-only updates
    scheduler
        .update_snapshot(&b1, vec!["row A".to_string(), "row B".to_string()])
        .unwrap();
    scheduler.stream_stable_prefix(&b1, 1).unwrap();

    // Finalize
    scheduler.finalize_block(&b1).unwrap();

    // First retirement succeeds
    scheduler.retire_block(&b1).unwrap();
    assert_eq!(scheduler.logical_history.committed_line_count(), 2);
    assert_eq!(scheduler.logical_history.commit_frontier(), 1);

    // Second retirement on already committed block MUST fail!
    assert!(scheduler.retire_block(&b1).is_err());

    // Committed lines count and commit frontier remain exactly 2 and 1 (no duplicate rows)
    assert_eq!(scheduler.logical_history.committed_line_count(), 2);
    assert_eq!(scheduler.logical_history.commit_frontier(), 1);
    assert_eq!(
        scheduler.logical_history.all_lines(None),
        vec!["row A", "row B"]
    );
}

#[test]
fn test_all_resize_policies_preserve_logical_history() {
    let mut scheduler = TranscriptScheduler::new(80, 24);

    let b1 = BlockId::new("b1");
    scheduler
        .admit_block(b1.clone(), BlockMode::Mutable, "user")
        .unwrap();
    scheduler.activate_block(&b1).unwrap();
    scheduler
        .update_snapshot(&b1, vec!["committed text".to_string()])
        .unwrap();
    scheduler.finalize_block(&b1).unwrap();
    scheduler.retire_block(&b1).unwrap();

    let initial_history = scheduler.logical_history.all_lines(None);

    // Policy 1: Preserve
    scheduler.resize(100, 30, ResizePolicy::Preserve).unwrap();
    assert_eq!(scheduler.logical_history.all_lines(None), initial_history);

    // Policy 2: Append
    scheduler.resize(90, 25, ResizePolicy::Append).unwrap();
    assert_eq!(scheduler.logical_history.all_lines(None), initial_history);

    // Policy 3: Rebuild
    scheduler.resize(120, 40, ResizePolicy::Rebuild).unwrap();
    assert_eq!(scheduler.logical_history.all_lines(None), initial_history);
}
#[test]
fn test_out_of_order_retirement_rejected() {
    let mut scheduler = TranscriptScheduler::new(80, 24);

    let b1 = BlockId::new("b1");
    let b2 = BlockId::new("b2");
    scheduler
        .admit_block(b1.clone(), BlockMode::Mutable, "user")
        .unwrap();
    scheduler
        .admit_block(b2.clone(), BlockMode::Mutable, "assistant")
        .unwrap();

    scheduler.activate_block(&b1).unwrap();
    scheduler.activate_block(&b2).unwrap();

    scheduler
        .update_snapshot(&b1, vec!["user msg".to_string()])
        .unwrap();
    scheduler
        .update_snapshot(&b2, vec!["ai reply".to_string()])
        .unwrap();

    // Finalize b2 first
    scheduler.finalize_block(&b2).unwrap();

    // Attempting to retire b2 before b1 is committed MUST fail with OutOfOrderRetirement
    let err = scheduler.retire_block(&b2).unwrap_err();
    assert!(matches!(
        err,
        TranscriptError::OutOfOrderRetirement { ref id, ref expected }
        if id == &b2 && expected.as_ref() == Some(&b1)
    ));

    // Now finalize and retire b1 in correct commit order
    scheduler.finalize_block(&b1).unwrap();
    scheduler.retire_block(&b1).unwrap();
    assert_eq!(scheduler.commit_frontier(), 1);

    // Now b2 can retire
    scheduler.retire_block(&b2).unwrap();
    assert_eq!(scheduler.commit_frontier(), 2);
}

#[test]
fn test_streaming_and_retirement_provenance_tags_no_duplication() {
    use omp_render::NativeSource;

    let mut scheduler = TranscriptScheduler::new(80, 24);

    let b1 = BlockId::new("b1");
    scheduler
        .admit_block(b1.clone(), BlockMode::AppendOnly, "assistant")
        .unwrap();
    scheduler.activate_block(&b1).unwrap();

    scheduler
        .update_snapshot(&b1, vec!["line 1".to_string(), "line 2".to_string()])
        .unwrap();

    // Stream prefix 1
    scheduler.stream_stable_prefix(&b1, 1).unwrap();
    assert_eq!(scheduler.native_scrollback.len(), 1);
    assert_eq!(scheduler.native_scrollback[0].text, "line 1");
    assert_eq!(scheduler.native_scrollback[0].source, NativeSource::Append);

    // Finalize and retire
    scheduler.finalize_block(&b1).unwrap();
    scheduler.retire_block(&b1).unwrap();

    // Exactly 2 rows in scrollback: line 1 from Append, line 2 from Retire (no duplication)
    assert_eq!(scheduler.native_scrollback.len(), 2);
    assert_eq!(scheduler.native_scrollback[0].text, "line 1");
    assert_eq!(scheduler.native_scrollback[0].source, NativeSource::Append);
    assert_eq!(scheduler.native_scrollback[1].text, "line 2");
    assert_eq!(scheduler.native_scrollback[1].source, NativeSource::Retire);

    assert_eq!(scheduler.commit_frontier(), 1);
    assert_eq!(
        scheduler.logical_history.all_lines(None),
        vec!["line 1", "line 2"]
    );
}

#[test]
fn test_flush_retires_in_strict_fifo_order() {
    let mut scheduler = TranscriptScheduler::new(80, 24);

    let b1 = BlockId::new("b1");
    let b2 = BlockId::new("b2");
    let b3 = BlockId::new("b3");
    scheduler
        .admit_block(b1.clone(), BlockMode::Mutable, "a")
        .unwrap();
    scheduler
        .admit_block(b2.clone(), BlockMode::Mutable, "b")
        .unwrap();
    scheduler
        .admit_block(b3.clone(), BlockMode::Mutable, "c")
        .unwrap();

    scheduler.activate_block(&b1).unwrap();
    scheduler.activate_block(&b2).unwrap();
    scheduler.activate_block(&b3).unwrap();

    scheduler
        .update_snapshot(&b1, vec!["first".to_string()])
        .unwrap();
    scheduler
        .update_snapshot(&b2, vec!["second".to_string()])
        .unwrap();
    scheduler
        .update_snapshot(&b3, vec!["third".to_string()])
        .unwrap();

    // Finalize b1 and b2; leave b3 active
    scheduler.finalize_block(&b1).unwrap();
    scheduler.finalize_block(&b2).unwrap();

    // Flush should retire b1 and b2 in order, and stop before b3
    scheduler.flush().unwrap();
    assert_eq!(scheduler.commit_frontier(), 2);
    assert_eq!(scheduler.blocks[0].state, BlockState::Committed);
    assert_eq!(scheduler.blocks[1].state, BlockState::Committed);
    assert_eq!(scheduler.blocks[2].state, BlockState::Active);
}

#[test]
fn test_rebuild_resize_bumps_display_epoch_on_rows() {
    let mut scheduler = TranscriptScheduler::new(80, 24);

    let b = BlockId::new("b");
    scheduler
        .admit_block(b.clone(), BlockMode::Mutable, "user")
        .unwrap();
    scheduler.activate_block(&b).unwrap();
    scheduler
        .update_snapshot(&b, vec!["hello world".to_string()])
        .unwrap();

    assert_eq!(scheduler.display_epoch, 1);
    assert_eq!(scheduler.native_viewport[0].epoch, 1);

    // Rebuild resize to 100x30
    scheduler.resize(100, 30, ResizePolicy::Rebuild).unwrap();
    assert_eq!(scheduler.display_epoch, 2);
    assert_eq!(scheduler.native_viewport[0].epoch, 2);
}

#[test]
fn finalized_streamed_head_survives_rebuild_without_duplicates() {
    let mut scheduler = TranscriptScheduler::new(80, 8);
    let id = BlockId::new("head");
    scheduler
        .admit_block(id.clone(), BlockMode::AppendOnly, "assistant")
        .unwrap();
    scheduler.activate_block(&id).unwrap();
    scheduler
        .update_snapshot(&id, vec!["stable".into(), "final suffix".into()])
        .unwrap();
    scheduler.stream_stable_prefix(&id, 1).unwrap();
    scheduler.finalize_block(&id).unwrap();
    let before = scheduler
        .logical_history
        .all_lines(scheduler.blocks.first());
    scheduler.resize(20, 4, ResizePolicy::Rebuild).unwrap();
    assert_eq!(
        scheduler
            .native_scrollback
            .iter()
            .map(|row| row.text.as_str())
            .collect::<Vec<_>>(),
        vec!["stable"]
    );
    assert_eq!(
        scheduler
            .logical_history
            .all_lines(scheduler.blocks.first()),
        before
    );
    scheduler.retire_block(&id).unwrap();
    assert_eq!(
        scheduler
            .native_scrollback
            .iter()
            .map(|row| row.text.as_str())
            .collect::<Vec<_>>(),
        vec!["stable", "final suffix"]
    );
}

#[test]
fn queued_streaming_cannot_create_premature_history() {
    let mut scheduler = TranscriptScheduler::new(80, 4);
    let id = BlockId::new("queued");
    scheduler
        .admit_block(id.clone(), BlockMode::AppendOnly, "assistant")
        .unwrap();
    scheduler
        .update_snapshot(&id, vec!["not active".into()])
        .unwrap();
    assert!(scheduler.stream_stable_prefix(&id, 1).is_err());
    assert!(scheduler.native_scrollback.is_empty());
    assert!(
        scheduler
            .logical_history
            .all_lines(scheduler.blocks.first())
            .is_empty()
    );
}

#[test]
fn live_reservations_share_capacity_including_overflow() {
    let mut scheduler = TranscriptScheduler::new(10, 4);
    for id in ["old", "new"] {
        let id = BlockId::new(id);
        scheduler
            .admit_block(id.clone(), BlockMode::Mutable, "tool")
            .unwrap();
        scheduler.activate_block(&id).unwrap();
        scheduler
            .update_snapshot(&id, vec!["row".into(); 8])
            .unwrap();
    }
    let summary = scheduler
        .native_viewport
        .iter()
        .filter(|row| row.is_summary)
        .count();
    assert!(
        scheduler
            .blocks
            .iter()
            .map(|block| block.reserved_rows)
            .sum::<usize>()
            + summary
            <= 4
    );
    assert!(
        scheduler
            .native_viewport
            .iter()
            .any(|row| row.owner.as_deref() == Some("tool")
                && row.source_block.as_ref().map(BlockId::as_str) == Some("new"))
    );
    scheduler.resize(10, 0, ResizePolicy::Preserve).unwrap();
    assert_eq!(
        scheduler
            .blocks
            .iter()
            .map(|block| block.reserved_rows)
            .sum::<usize>(),
        0
    );
}

#[test]
fn test_scrollback_ring_bound_evicts_oldest_first() {
    let mut scheduler = TranscriptScheduler::new(80, 24);
    assert_eq!(TranscriptScheduler::MAX_SCROLLBACK_ROWS, 10_000);

    // Retire one block per line: each retire pushes exactly one row.
    for i in 0..(TranscriptScheduler::MAX_SCROLLBACK_ROWS + 500) {
        let id = BlockId::new(format!("ring-{i}"));
        scheduler
            .admit_block(id.clone(), BlockMode::AppendOnly, "assistant")
            .unwrap();
        scheduler.activate_block(&id).unwrap();
        scheduler
            .update_snapshot(&id, vec![format!("line {i}")])
            .unwrap();
        scheduler.finalize_block(&id).unwrap();
        scheduler.retire_block(&id).unwrap();
    }

    assert_eq!(
        scheduler.native_scrollback.len(),
        TranscriptScheduler::MAX_SCROLLBACK_ROWS
    );
    assert_eq!(scheduler.scrollback_evicted, 500);
    // Oldest retained row is line 500; newest is the last line.
    assert_eq!(scheduler.native_scrollback[0].text, "line 500");
    assert_eq!(
        scheduler.native_scrollback.last().unwrap().text,
        format!("line {}", TranscriptScheduler::MAX_SCROLLBACK_ROWS + 499)
    );
    // Logical history (the commit ledger) is unaffected by the ring bound.
    assert_eq!(
        scheduler.logical_history.commit_frontier(),
        TranscriptScheduler::MAX_SCROLLBACK_ROWS + 500
    );
}
