use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use omp_python::{
    ExtensionManager, HostPatchRequest, HostQuery, PythonExtensionError, RemoteFunctionSpec,
    RemoteValidator, ToolDeclaration,
};
use omp_types::PatchOp;

#[test]
fn test_remote_validator_rejects_forbidden_dynamic_constructs() {
    // 1. eval rejection
    let spec_eval = RemoteFunctionSpec::new("bad_eval", "def bad(): eval('1+1')");
    let err = RemoteValidator::validate(&spec_eval).unwrap_err();
    assert!(matches!(err, PythonExtensionError::DynamicCodeForbidden(_)));

    // 2. eval with whitespace
    let spec_eval_ws = RemoteFunctionSpec::new("bad_eval_ws", "def bad(): eval (code)");
    let err_ws = RemoteValidator::validate(&spec_eval_ws).unwrap_err();
    assert!(matches!(
        err_ws,
        PythonExtensionError::DynamicCodeForbidden(_)
    ));

    // 3. getattr rejection
    let spec_getattr = RemoteFunctionSpec::new("bad_getattr", "def bad(): getattr(obj, 'x')");
    let err_getattr = RemoteValidator::validate(&spec_getattr).unwrap_err();
    assert!(matches!(
        err_getattr,
        PythonExtensionError::DynamicCodeForbidden(_)
    ));

    // 4. importlib rejection
    let spec_importlib =
        RemoteFunctionSpec::new("bad_importlib", "def bad(): importlib.import_module('x')");
    let err_importlib = RemoteValidator::validate(&spec_importlib).unwrap_err();
    assert!(matches!(
        err_importlib,
        PythonExtensionError::DynamicCodeForbidden(_)
    ));
}

#[test]
fn test_remote_validator_rejects_oversized_payloads() {
    let mut spec = RemoteFunctionSpec::new("oversized", "def big(): pass");
    spec.max_payload_bytes = 10;
    spec.source_code = "def big_function_exceeding_ten_bytes(): pass".to_string();
    // Re-stamp the hash after mutating the source so the size check (not the
    // integrity check) is what fires.
    spec.source_hash = {
        use sha2::{Digest, Sha256};
        hex::encode(Sha256::digest(spec.source_code.as_bytes()))
    };

    let err = RemoteValidator::validate(&spec).unwrap_err();
    assert!(matches!(err, PythonExtensionError::OversizedPayload { .. }));

    // validate_payload helper
    let err_payload = RemoteValidator::validate_payload(1024, 512).unwrap_err();
    assert!(matches!(
        err_payload,
        PythonExtensionError::OversizedPayload {
            size: 1024,
            limit: 512
        }
    ));
}

#[test]
fn test_remote_validator_enforces_imports_and_capabilities() {
    // Prohibited import without capability
    let mut spec_os = RemoteFunctionSpec::new("bad_os", "def bad(): import os");
    spec_os.imports = vec!["os".to_string()];
    let err_os = RemoteValidator::validate(&spec_os).unwrap_err();
    assert!(matches!(
        err_os,
        PythonExtensionError::ProhibitedImport { .. }
    ));

    // Prohibited import with explicit capability override
    let mut spec_os_cap = RemoteFunctionSpec::new("good_os", "def good(): import os");
    spec_os_cap.imports = vec!["os".to_string()];
    spec_os_cap.declared_capabilities = vec!["import_os".to_string()];
    assert!(RemoteValidator::validate(&spec_os_cap).is_ok());

    // Undeclared filesystem capability
    let spec_open = RemoteFunctionSpec::new("bad_open", "def bad(): open('file.txt')");
    let err_open = RemoteValidator::validate(&spec_open).unwrap_err();
    assert!(matches!(
        err_open,
        PythonExtensionError::UndeclaredCapability { .. }
    ));
}

#[test]
fn test_extension_manager_reload_invalidates_handles() {
    let mut mgr = ExtensionManager::new();
    let ext_id = "test_ext";

    // Load extension
    mgr.load(ext_id.to_string(), "{\"name\":\"test\"}").unwrap();

    // Register a tool
    let tool = ToolDeclaration {
        name: "test_tool".to_string(),
        version: "1.0.0".to_string(),
        description: "A test tool".to_string(),
        parameter_schema: serde_json::json!({}),
    };
    mgr.register_declarations(ext_id, vec![tool], vec![], vec![])
        .unwrap();
    assert!(mgr.get_tool("test_tool").unwrap().is_some());

    // Track an in-flight request handle
    let req_handle = "req_12345".to_string();
    mgr.track_request(ext_id, req_handle.clone()).unwrap();
    assert!(mgr.is_request_active(ext_id, &req_handle));
    assert!(mgr.validate_request_handle(ext_id, &req_handle).is_ok());

    // Reload extension
    mgr.reload(ext_id).unwrap();

    // Verification 1: In-flight handle is invalidated!
    assert!(!mgr.is_request_active(ext_id, &req_handle));
    let err = mgr
        .validate_request_handle(ext_id, &req_handle)
        .unwrap_err();
    assert!(matches!(err, PythonExtensionError::ProtocolViolation(_)));

    // Verification 2: Registered tools are cleared until re-registered
    assert!(mgr.get_tool("test_tool").unwrap().is_none());
}

#[test]
fn test_extension_manager_reload_terminates_active_workers() {
    let mut mgr = ExtensionManager::new();
    let ext_id = "worker_ext";
    mgr.load(ext_id.to_string(), "{}").unwrap();

    let terminated = Arc::new(AtomicBool::new(false));
    let terminated_clone = Arc::clone(&terminated);

    mgr.register_worker(ext_id, move || {
        terminated_clone.store(true, Ordering::SeqCst);
        Ok(())
    })
    .unwrap();

    assert!(!terminated.load(Ordering::SeqCst));

    // Reloading extension must invoke worker terminator
    mgr.reload(ext_id).unwrap();
    assert!(terminated.load(Ordering::SeqCst));
}

#[test]
fn test_extension_manager_host_authority_on_patches_and_queries() {
    let mut mgr = ExtensionManager::new();
    let ext_id = "trusted_ext";
    mgr.load(ext_id.to_string(), "{}").unwrap();

    // Empty patch is rejected
    let empty_patch = HostPatchRequest {
        ops: vec![],
        reason: "empty".into(),
    };
    let err = mgr.validate_patch(ext_id, &empty_patch).unwrap_err();
    assert!(matches!(err, PythonExtensionError::ProtocolViolation(_)));

    // Valid patch with ops is accepted
    let valid_patch = HostPatchRequest {
        ops: vec![PatchOp::Delete {
            element: omp_types::ElementId::new("elem1").unwrap(),
        }],
        reason: "cleanup".into(),
    };
    assert!(mgr.validate_patch(ext_id, &valid_patch).is_ok());

    // Unknown extension cannot query or patch
    assert!(
        mgr.validate_query("unknown_ext", &HostQuery::GetSnapshot { offset: None })
            .is_err()
    );
    assert!(mgr.validate_patch("unknown_ext", &valid_patch).is_err());
}

#[test]
fn test_remote_validator_derives_imports_from_source() {
    // Bypass attempt: `import os` in source with an empty client list must
    // still be rejected — the host derives imports from the source itself.
    let spec = RemoteFunctionSpec::new("sneaky", "def f():\n    import os\n    return os.getcwd()");
    assert!(spec.imports.is_empty());
    let err = RemoteValidator::validate(&spec).unwrap_err();
    assert!(matches!(
        err,
        PythonExtensionError::ProhibitedImport { .. }
    ));

    // `from X import` form is covered too.
    let spec_from = RemoteFunctionSpec::new("sneaky2", "def f():\n    from subprocess import run");
    let err = RemoteValidator::validate(&spec_from).unwrap_err();
    assert!(matches!(
        err,
        PythonExtensionError::ProhibitedImport { .. }
    ));

    // Benign imports pass.
    let spec_ok = RemoteFunctionSpec::new("fine", "def f():\n    import json\n    return json.dumps({})");
    assert!(RemoteValidator::validate(&spec_ok).is_ok());
}

#[test]
fn test_remote_validator_verifies_source_hash() {
    let mut spec = RemoteFunctionSpec::new("hashed", "def f(): return 1");
    // `new` stamps the correct hash: validation passes.
    assert!(RemoteValidator::validate(&spec).is_ok());
    // Tamper with the hash: validation fails.
    spec.source_hash = "0".repeat(64);
    let err = RemoteValidator::validate(&spec).unwrap_err();
    assert!(matches!(
        err,
        PythonExtensionError::ValidationFailed(_)
    ));
    // Tamper with the source after hashing: validation fails.
    let mut spec2 = RemoteFunctionSpec::new("hashed2", "def f(): return 1");
    spec2.source_code = "def f(): import os".to_string();
    let err = RemoteValidator::validate(&spec2).unwrap_err();
    assert!(matches!(
        err,
        PythonExtensionError::ValidationFailed(_)
    ));
}

#[test]
fn test_remote_validator_no_http_false_positive() {
    // A variable named `http_server` is not network access.
    let spec = RemoteFunctionSpec::new("plain", "def f():\n    http_server = 1\n    return http_server");
    assert!(RemoteValidator::validate(&spec).is_ok());

    // Real network identifiers still require the `net` capability.
    let spec_net = RemoteFunctionSpec::new("net", "def f():\n    import socket\n    return socket.gethostname()");
    let err = RemoteValidator::validate(&spec_net).unwrap_err();
    assert!(matches!(
        err,
        PythonExtensionError::ProhibitedImport { .. }
            | PythonExtensionError::UndeclaredCapability { .. }
    ));
}

#[test]
fn test_extension_manager_rejects_duplicate_tools_and_bad_patches() {
    let mut mgr = ExtensionManager::new();
    mgr.load("ext_a".to_string(), "{}").unwrap();
    mgr.load("ext_b".to_string(), "{}").unwrap();
    let tool = |name: &str| ToolDeclaration {
        name: name.to_string(),
        version: "1.0.0".to_string(),
        description: "t".to_string(),
        parameter_schema: serde_json::json!({}),
    };
    mgr.register_declarations("ext_a", vec![tool("dup")], vec![], vec![])
        .unwrap();
    mgr.register_declarations("ext_b", vec![tool("dup")], vec![], vec![])
        .unwrap();
    let err = mgr.get_tool("dup").unwrap_err();
    assert!(matches!(err, PythonExtensionError::ProtocolViolation(_)));

    // Oversized reason is rejected via the shared Patch validator.
    let bad_reason = HostPatchRequest {
        ops: vec![PatchOp::Delete {
            element: omp_types::ElementId::new("elem1").unwrap(),
        }],
        reason: "x".repeat(5000),
    };
    assert!(mgr.validate_patch("ext_a", &bad_reason).is_err());

    // Oversized text op is rejected (bound pinned at the wire cap).
    let big_text = HostPatchRequest {
        ops: vec![PatchOp::ReplaceText {
            element: omp_types::ElementId::new("elem1").unwrap(),
            text: "y".repeat(2_000_000),
        }],
        reason: "big".into(),
    };
    assert!(mgr.validate_patch("ext_a", &big_text).is_err());

    // Empty query names are rejected.
    assert!(
        mgr.validate_query(
            "ext_a",
            &HostQuery::GetConVar { name: "".into() }
        )
        .is_err()
    );
    assert!(
        mgr.validate_query(
            "ext_a",
            &HostQuery::GetConVar { name: "ok".into() }
        )
        .is_ok()
    );
}
