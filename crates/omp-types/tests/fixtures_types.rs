use omp_types::{Patch, SandboxEventEnvelope, SandboxRequest};
use std::fs;
use std::path::Path;

#[test]
fn test_journal_patches_fixture_deserializes_and_validates() {
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/journal_patches.json"
    ));
    let content = fs::read_to_string(path).expect("failed to read journal_patches.json");
    let patches: Vec<Patch> =
        serde_json::from_str(&content).expect("failed to deserialize patches");

    // Non-brittle: the fixture may grow; every patch must validate.
    assert!(!patches.is_empty(), "fixture must contain at least one patch");
    for patch in &patches {
        patch.validate().expect("patch must pass validation");
    }
}

#[test]
fn test_sandbox_fixture_deserializes_and_validates() {
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/sandbox_truncation_cancellation.json"
    ));
    let content =
        fs::read_to_string(path).expect("failed to read sandbox_truncation_cancellation.json");
    let parsed: serde_json::Value =
        serde_json::from_str(&content).expect("failed to parse sandbox json");

    let req_val = parsed.get("request").expect("missing request");
    let req: SandboxRequest =
        serde_json::from_value(req_val.clone()).expect("failed to deserialize SandboxRequest");
    req.validate().expect("sandbox request must be valid");

    let events_val = parsed.get("events").expect("missing events");
    let envelopes: Vec<SandboxEventEnvelope> = serde_json::from_value(events_val.clone())
        .expect("failed to deserialize SandboxEventEnvelope vector");
    assert!(!envelopes.is_empty(), "fixture must contain events");
    // Sequences must be strictly increasing (out-of-order envelopes are a
    // protocol violation even though the fixture may grow).
    let mut previous = 0u64;
    let mut first = true;
    for envelope in &envelopes {
        if !first {
            assert!(
                envelope.sequence > previous,
                "envelope sequences must be strictly increasing"
            );
        }
        first = false;
        previous = envelope.sequence;
    }
}
