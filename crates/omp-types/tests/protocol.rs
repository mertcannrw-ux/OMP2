use omp_types::*;

#[test]
fn ids_cannot_bypass_validation_via_wire() {
    for value in ["", "../journal", "with spaces", "x\u{0}"] {
        assert!(SessionId::new(value).is_err());
        assert!(serde_json::from_value::<SessionId>(serde_json::json!(value)).is_err());
    }
    let a = SessionId::mint();
    let b = SessionId::mint();
    assert_ne!(a, b);
    assert_eq!(
        serde_json::from_str::<SessionId>(&serde_json::to_string(&a).unwrap()).unwrap(),
        a
    );
}

#[test]
fn handshake_requires_features_in_both_directions() {
    let mut host = Handshake::current();
    host.required_features.insert(JOURNAL_FORMAT.into());
    let mut peer = Handshake::current();
    peer.version.minor = 12;
    host.accept(&peer).unwrap();
    peer.features.remove(JOURNAL_FORMAT);
    assert_eq!(
        host.accept(&peer).unwrap_err().code,
        "protocol_missing_feature"
    );
    peer.version.major = 2;
    assert_eq!(
        host.accept(&peer).unwrap_err().code,
        "protocol_major_mismatch"
    );
    peer = Handshake::current();
    peer.required_features.insert("future-feature".into());
    assert_eq!(
        host.accept(&peer).unwrap_err().code,
        "protocol_missing_feature"
    );
}

#[test]
fn malformed_or_oversized_wire_fails_closed() {
    assert!(decode_wire::<Handshake>(br#"{"version":{"major":1,"minor":0},"features":[],"required_features":[],"unexpected":1}"#).is_err());
    assert_eq!(
        decode_wire::<Handshake>(&vec![b' '; MAX_WIRE_BYTES + 1])
            .unwrap_err()
            .code,
        "wire_size_limit"
    );
    let id = ElementId::new("body").unwrap();
    let patch = Patch {
        base_offset: JournalOffset(0),
        result_offset: JournalOffset(1),
        by: ActorId::mint().into(),
        reason: "write message".into(),
        ops: vec![PatchOp::AppendText {
            element: id,
            text: "hello".into(),
        }],
    };
    let encoded = serde_json::to_vec(&patch).unwrap();
    let decoded: Patch = decode_wire(&encoded).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded, patch);
    assert_eq!(
        serde_json::to_string(&Status::CancelRequested).unwrap(),
        "\"cancel_requested\""
    );
}
