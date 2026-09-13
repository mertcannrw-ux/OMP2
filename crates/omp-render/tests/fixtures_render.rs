use omp_render::{Component, DebugSnapshot, NativeRow, ResizePolicy};
use std::fs;
use std::path::Path;

#[test]
fn test_component_tree_fixture_deserializes() {
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/component_tree.json"
    ));
    let content = fs::read_to_string(path).expect("failed to read component_tree.json fixture");
    let parsed: serde_json::Value =
        serde_json::from_str(&content).expect("failed to parse component_tree.json");

    let root_val = parsed.get("root").expect("missing root field");
    let comp: Component = serde_json::from_value(root_val.clone())
        .expect("failed to deserialize Component from root");

    assert_eq!(comp.kind, omp_render::ComponentKind::Box);
    assert_eq!(comp.children.len(), 3);
}

#[test]
fn test_resize_replay_fixture_deserializes() {
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/resize_replay.json"
    ));
    let content = fs::read_to_string(path).expect("failed to read resize_replay.json fixture");
    let parsed: serde_json::Value =
        serde_json::from_str(&content).expect("failed to parse resize_replay.json");

    let before_snap = parsed
        .get("snapshot_before_resize")
        .expect("missing snapshot_before_resize");
    let snap: DebugSnapshot =
        serde_json::from_value(before_snap.clone()).expect("failed to deserialize DebugSnapshot");
    assert_eq!(snap.display_epoch, 1);
    assert_eq!(snap.commit_frontier, 2);

    let resize_policy = parsed
        .get("resize_event")
        .and_then(|r| r.get("policy"))
        .expect("missing policy");
    let policy: ResizePolicy =
        serde_json::from_value(resize_policy.clone()).expect("failed to deserialize ResizePolicy");
    assert_eq!(policy, ResizePolicy::Rebuild);

    let fore_rows = parsed
        .get("write_failure_scenario")
        .and_then(|w| w.get("forensic_rows"))
        .expect("missing forensic_rows");
    let rows: Vec<NativeRow> =
        serde_json::from_value(fore_rows.clone()).expect("failed to deserialize NativeRow vector");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].epoch, 2);
}
