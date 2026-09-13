"""Wrapper patch encoding, validation, and structured error handling."""

import pytest
from omp_sdk.examples import TodoCounterTool
from omp_sdk.extension import Extension, ExtensionContext
from omp_sdk.protocol import HostError, MockTransport, _coerce_host_error, invoke_tool
from omp_sdk.wrappers import CommandClient, DOMClient, wrap_typed


def test_wrap_typed_passthrough_avoids_double_wrap() -> None:
    assert wrap_typed({"String": "hi"}) == {"String": "hi"}
    assert wrap_typed({"Integer": 3}) == {"Integer": 3}
    assert wrap_typed("hi") == {"String": "hi"}
    assert wrap_typed(3) == {"Integer": 3}
    assert wrap_typed(True) == {"Bool": True}
    assert wrap_typed(None) == "Null"
    assert wrap_typed([1]) == {"Json": [1]}


def test_create_element_encodes_exact_types() -> None:
    transport = MockTransport()
    dom = DOMClient(transport)
    dom.create_element(
        parent="todo",
        element_id="todo_1",
        kind="todo_item",
        attributes={"title": "Buy milk", "index": 2},
        text="Buy milk",
    )
    (patch,) = transport.patches
    (op,) = patch["ops"]
    element = op["Create"]["element"]
    assert element["attributes"]["title"] == {"String": "Buy milk"}
    assert element["attributes"]["index"] == {"Integer": 2}


def test_wrappers_reject_bad_ids_and_ops() -> None:
    transport = MockTransport()
    dom = DOMClient(transport)
    with pytest.raises(ValueError):
        dom.create_element(parent="todo", element_id="bad id!", kind="todo_item")
    with pytest.raises(ValueError):
        dom.create_element(parent="todo", element_id="ok", kind="bad kind!")
    with pytest.raises(ValueError):
        dom.delete_element("bad id!")
    with pytest.raises(ValueError):
        dom.apply_patch(ops=[], reason="x")
    with pytest.raises(ValueError):
        dom.apply_patch(ops=[{"Bogus": {}}], reason="x")
    with pytest.raises(ValueError):
        dom.apply_patch(ops=[{"Delete": {"element": "x"}}], reason="   ")
    with pytest.raises(NotImplementedError):
        CommandClient(transport).execute("tool read")


def test_todo_counter_tool_round_trip() -> None:
    transport = MockTransport()
    context = ExtensionContext("ext", transport)
    tool = TodoCounterTool(context)
    result = tool.execute({"title": "Buy milk"}, context)
    assert result["status"] == "success"
    assert result["total_todos"] == 1
    (patch,) = transport.patches
    element = patch["ops"][0]["Create"]["element"]
    assert element["attributes"]["title"] == {"String": "Buy milk"}
    # Fresh session without a todo node must not KeyError.
    empty_snapshot: dict = {}
    assert TodoCounterTool._todo_children(empty_snapshot) == []


def test_host_error_preserves_structure() -> None:
    err = _coerce_host_error(
        {"code": "job_failed", "message": "boom", "retryable": True}
    )
    assert isinstance(err, HostError)
    assert err.code == "job_failed"
    assert err.retryable is True
    with pytest.raises(HostError):
        invoke_tool("missing/tool", {})


def test_extension_rejects_slash_tool_names() -> None:
    transport = MockTransport()
    ext = Extension("ext", transport=transport)
    from omp_sdk.wrappers import Tool

    class Bad(Tool):
        name = "a/b"

    ext.register_tool(Bad())
    with pytest.raises(ValueError):
        ext.sync_declarations()
