use omp_state::*;
use omp_types::*;
use std::{collections::BTreeSet, fs};

fn key(s: &str) -> ElementId {
    ElementId::new(s).unwrap()
}
fn temp() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("omp2-{}.journal", SessionId::mint()))
}
fn put(j: &mut Journal, parent: &str, kind: &str, text: &str) -> ElementId {
    let id = ElementId::mint();
    let mut element = ElementSnapshot::new(id.clone(), kind);
    element.text = text.into();
    let index = j.snapshot().children(&key(parent)).count() as u32;
    let patch = Patch {
        base_offset: JournalOffset(j.snapshot().offset),
        result_offset: j.next_offset(),
        by: ActorId::mint().into(),
        reason: "regression entry".into(),
        ops: vec![PatchOp::Create {
            parent: key(parent),
            index,
            element,
        }],
    };
    j.append_patch(patch).unwrap();
    id
}
#[test]
fn atomic_ordered_tree_edits_and_cycle_rejection() {
    let mut s = SessionSnapshot::empty(SessionId::mint());
    let patch = |base, next, ops| Patch {
        base_offset: JournalOffset(base),
        result_offset: JournalOffset(next),
        by: ActorId::mint().into(),
        reason: "edit tree".into(),
        ops,
    };
    apply_patch(
        &mut s,
        &patch(
            0,
            1,
            vec![
                PatchOp::Create {
                    parent: key("body"),
                    index: 0,
                    element: ElementSnapshot::new(key("a"), "tool"),
                },
                PatchOp::Create {
                    parent: key("a"),
                    index: 0,
                    element: ElementSnapshot::new(key("b"), "result"),
                },
                PatchOp::AppendText {
                    element: key("b"),
                    text: "hello".into(),
                },
            ],
        ),
    )
    .unwrap();
    let before = s.clone();
    let rejected = patch(
        1,
        2,
        vec![
            PatchOp::ReplaceText {
                element: key("b"),
                text: "wrong".into(),
            },
            PatchOp::SetAttribute {
                element: key("a"),
                name: "status".into(),
                value: TypedValue::String("running".into()),
            },
            PatchOp::Move {
                element: key("a"),
                parent: key("b"),
                index: 0,
            },
        ],
    );
    assert!(apply_patch(&mut s, &rejected).is_err());
    assert_eq!(s, before);
    apply_patch(
        &mut s,
        &patch(
            1,
            2,
            vec![
                PatchOp::Move {
                    element: key("b"),
                    parent: key("body"),
                    index: 1,
                },
                PatchOp::Delete { element: key("a") },
                PatchOp::ReplacePayload {
                    element: key("b"),
                    payload: serde_json::json!({"ok":true}),
                },
                PatchOp::RemoveAttribute {
                    element: key("b"),
                    name: "missing".into(),
                },
            ],
        ),
    )
    .unwrap();
    assert_eq!(
        s.get_visible_body()
            .map(|e| e.id.clone())
            .collect::<Vec<_>>(),
        vec![key("b")]
    );
    assert_eq!(s.element(&key("b")).unwrap().text, "hello");
    assert!(
        apply_patch(
            &mut s,
            &patch(
                2,
                3,
                vec![PatchOp::Create {
                    parent: key("body"),
                    index: 0,
                    element: ElementSnapshot::new(key("a"), "user")
                }]
            )
        )
        .is_err()
    );
    assert!(s.inspect_xml().contains("&quot;ok&quot;:true"));
}
#[test]
fn selected_branch_restores_all_persistent_state() {
    let path = temp();
    let mut j = Journal::create(&path, SessionId::mint()).unwrap();
    put(&mut j, "body", "user", "turn one");
    let checkpoint = put(&mut j, "meta", "checkpoint", "git ref");
    let before_discovery = j.snapshot().offset;
    let tool = put(&mut j, "tools", "dynamic_tool", "Calculator");
    let plan = put(&mut j, "directors", "plan", "must plan");
    put(&mut j, "body", "user", "turn two");
    let dead_save = put(&mut j, "meta", "save", "snake save");
    put(&mut j, "body", "assistant", "dead branch bookmark");
    assert_eq!(j.snapshot().turn_count(), 2);
    let branch = j.fork_at(before_discovery).unwrap();
    assert!(j.snapshot().active_tool_roster().next().is_none());
    assert!(j.snapshot().active_directors().next().is_none());
    assert!(j.snapshot().element(&checkpoint).is_some());
    assert!(j.snapshot().element(&dead_save).is_none());
    assert!(j.snapshot().element(&plan).is_none());
    assert_eq!(j.snapshot().turn_count(), 1);
    assert_eq!(
        j.snapshot().get_last_visible_message().unwrap().text,
        "turn one"
    );
    put(&mut j, "body", "diagnostic", "mixed entry");
    put(&mut j, "body", "tool", "X move");
    let snapshot = j.resume_latest();
    let off = snapshot.offset;
    drop(j);
    let mut j = Journal::open(&path).unwrap();
    assert_eq!(j.snapshot(), &snapshot);
    assert_eq!(j.snapshot().current_branch(), &branch);
    assert!(j.snapshot().element(&tool).is_none());
    assert_eq!(
        j.materialize_branch(&BranchId::new("main").unwrap())
            .unwrap()
            .turn_count(),
        2
    );
    j.select_branch(&BranchId::new("main").unwrap()).unwrap();
    assert!(j.snapshot().element(&tool).is_some());
    j.rewind_to(off).unwrap();
    assert!(j.snapshot().element(&tool).is_none());
    drop(j);
    fs::remove_file(path).unwrap();
}
#[test]
fn crash_every_record_byte_preserves_last_valid_state() {
    let source = temp();
    let mut j = Journal::create(&source, SessionId::mint()).unwrap();
    let header_size = fs::metadata(&source).unwrap().len() as usize;
    put(&mut j, "body", "user", "stable");
    let boundary = fs::metadata(&source).unwrap().len() as usize;
    let stable = j.resume_latest();
    put(&mut j, "body", "assistant", "last record");
    let final_state = j.resume_latest();
    drop(j);
    let bytes = fs::read(&source).unwrap();
    let damaged = temp();
    for cut in header_size..=bytes.len() {
        fs::write(&damaged, &bytes[..cut]).unwrap();
        let recovered = Journal::open(&damaged).unwrap();
        let expected = if cut < boundary {
            0
        } else if cut < bytes.len() {
            1
        } else {
            2
        };
        assert_eq!(recovered.snapshot().offset, expected, "cut={cut}");
        if expected == 1 {
            assert_eq!(recovered.snapshot(), &stable);
        }
        if expected == 2 {
            assert_eq!(recovered.snapshot(), &final_state);
        }
        if ![header_size, boundary, bytes.len()].contains(&cut) {
            assert!(recovered.recovery().is_some());
        }
    }
    fs::write(&damaged, &bytes[..bytes.len() - 1]).unwrap();
    let mut recovered = Journal::open(&damaged).unwrap();
    let suffix = recovered.repair_suffix().unwrap().unwrap();
    assert!(fs::metadata(&suffix).unwrap().len() > 0);
    put(&mut recovered, "body", "user", "resumed after repair");
    drop(recovered);
    for path in [source, damaged, suffix] {
        fs::remove_file(path).unwrap();
    }
}
#[test]
fn handle_reconciliation_has_no_out_of_band_authority() {
    let path = temp();
    let mut j = Journal::create(&path, SessionId::mint()).unwrap();
    let offset = j.snapshot().offset;
    let job = put(&mut j, "jobs", "job", "long running");
    assert_eq!(
        j.snapshot().reconcile_handles(&BTreeSet::new()),
        vec![HandleAction::ResumeOrSpawn(job.clone())]
    );
    j.rewind_to(offset).unwrap();
    assert_eq!(
        j.snapshot().reconcile_handles(&[job.clone()].into()),
        vec![HandleAction::Terminate(job)]
    );
    drop(j);
    fs::remove_file(path).unwrap();
}
#[test]
fn a_second_writer_cannot_open_the_journal() {
    let path = temp();
    let j = Journal::create(&path, SessionId::mint()).unwrap();
    assert!(Journal::open(&path).is_err());
    drop(j);
    drop(Journal::open(&path).unwrap());
    fs::remove_file(path).unwrap();
}

#[test]
fn open_replays_a_long_journal_incrementally() {
    let path = temp();
    let mut j = Journal::create(&path, SessionId::mint()).unwrap();
    const RECORDS: u64 = 1000;
    for i in 0..RECORDS {
        j.append_patch(Patch {
            base_offset: JournalOffset(j.snapshot().offset),
            result_offset: j.next_offset(),
            by: ActorId::mint().into(),
            reason: "long journal".into(),
            ops: vec![PatchOp::SetAttribute {
                element: key("convars"),
                name: "replay_counter".into(),
                value: TypedValue::Integer(i as i64),
            }],
        })
        .unwrap();
    }
    let expected = j.resume_latest();
    drop(j);
    let reopened = Journal::open(&path).unwrap();
    assert!(reopened.recovery().is_none());
    assert_eq!(reopened.snapshot(), &expected);
    assert_eq!(reopened.snapshot().offset, RECORDS);
    assert_eq!(
        reopened.snapshot().session_globals()["replay_counter"],
        TypedValue::Integer((RECORDS - 1) as i64)
    );
    drop(reopened);
    fs::remove_file(path).unwrap();
}

#[test]
fn oversized_patch_does_not_poison_the_live_journal() {
    let path = temp();
    let mut j = Journal::create(&path, SessionId::mint()).unwrap();
    let mut element = ElementSnapshot::new(ElementId::mint(), "tool");
    element.text = "x".repeat(MAX_WIRE_BYTES);
    let err = j
        .append_patch(Patch {
            base_offset: JournalOffset(j.snapshot().offset),
            result_offset: j.next_offset(),
            by: ActorId::mint().into(),
            reason: "too large for a frame".into(),
            ops: vec![PatchOp::Create {
                parent: key("body"),
                index: 0,
                element,
            }],
        })
        .unwrap_err();
    assert!(matches!(err, StateError::Invalid(_)), "{err}");
    put(&mut j, "body", "tool", "still writable");
    assert_eq!(j.snapshot().offset, 1);
    drop(j);
    fs::remove_file(path).unwrap();
}
