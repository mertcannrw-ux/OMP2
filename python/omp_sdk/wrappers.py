"""Typed client wrappers for the omp_sdk."""

import re
from typing import Any, Dict, List, Optional
from .protocol import ComponentSpec, DirectorSpec, HostTransport, ToolSpec

_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_KIND_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_REASON_LEN = 4096

# Patch op variants the host journal accepts (mirrors PatchOp).
_VALID_OPS = frozenset(
    {
        "Create",
        "Delete",
        "Move",
        "SetAttribute",
        "RemoveAttribute",
        "ReplaceText",
        "AppendText",
        "ReplacePayload",
    }
)


def _check_id(value: str, what: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.match(value):
        raise ValueError(
            f"{what} must match [A-Za-z0-9._-]{{1,128}} (got {value!r})"
        )


def _check_kind(kind: str) -> None:
    if not isinstance(kind, str) or not _KIND_PATTERN.match(kind) or len(kind) > 128:
        raise ValueError(f"element kind must match [A-Za-z0-9_-]{{1,128}} (got {kind!r})")


def _check_reason(reason: str) -> None:
    if not isinstance(reason, str) or not reason.strip() or len(reason) > _MAX_REASON_LEN:
        raise ValueError("patch reason must be 1..=4096 characters")


def wrap_typed(value: Any) -> Any:
    """Encode a Python value as a host `TypedValue`.

    Already-tagged dicts (`{"String": ...}`, `{"Integer": ...}`, ...) pass
    through untouched so callers can express exact types; anything else is
    wrapped by inferred type. This fixes the old double-wrap bug where
    `{"String": title}` became `{"Json": {"String": title}}`.
    """
    if isinstance(value, dict) and len(value) == 1:
        tag = next(iter(value))
        if tag in ("Null", "Bool", "Integer", "Number", "String", "Json"):
            return value
    if value is None:
        return "Null"
    if isinstance(value, bool):
        return {"Bool": value}
    if isinstance(value, int):
        return {"Integer": value}
    if isinstance(value, float):
        return {"Number": value}
    if isinstance(value, str):
        return {"String": value}
    return {"Json": value}


class ConVarWrapper:
    """Interface to host-owned configuration variables (read-only).

    NOTE: the host `HostQuery` protocol only supports reads. There is no
    `set_convar` or `execute_command` query kind; the old mutating wrappers
    were removed because they had no server-side handler and failed
    unpredictably. Extensions that need writes must submit patches or jobs.
    """

    def __init__(self, transport: HostTransport) -> None:
        self._transport = transport

    def get(self, name: str) -> Any:
        return self._transport.send_query("get_convar", {"name": name})


class CommandClient:
    """Read-only notice: host command execution is not extension-callable.

    Retained as a stub so existing imports keep working; every call raises
    with guidance instead of sending a query the host cannot represent.
    """

    def __init__(self, transport: HostTransport) -> None:
        self._transport = transport

    def execute(self, command_line: str) -> Any:
        raise NotImplementedError(
            "execute_command is not a host query; extensions mutate state via "
            "DOMClient.apply_patch or JobClient.spawn"
        )


class SessionClient:
    """Client for querying authoritative session snapshots."""

    def __init__(self, transport: HostTransport) -> None:
        self._transport = transport

    def get_snapshot(self, offset: Optional[int] = None) -> Dict[str, Any]:
        return self._transport.send_query("get_snapshot", {"offset": offset})


class DOMClient:
    """Client for submitting DOM mutations to the authoritative host session."""

    def __init__(self, transport: HostTransport) -> None:
        self._transport = transport

    def apply_patch(self, ops: List[Dict[str, Any]], reason: str) -> None:
        _check_reason(reason)
        if not ops:
            raise ValueError("patch ops list cannot be empty")
        for op in ops:
            if not isinstance(op, dict) or len(op) != 1:
                raise ValueError(f"each op must be a single-variant dict (got {op!r})")
            variant = next(iter(op))
            if variant not in _VALID_OPS:
                raise ValueError(f"unknown patch op {variant!r}")
        self._transport.send_patch(ops, reason)

    def create_element(
        self,
        parent: str,
        element_id: str,
        kind: str,
        attributes: Optional[Dict[str, Any]] = None,
        text: str = "",
        payload: Optional[Any] = None,
        index: Optional[int] = None,
    ) -> None:
        _check_id(parent, "parent id")
        _check_id(element_id, "element id")
        _check_kind(kind)
        element_snapshot = {
            "id": element_id,
            "schema_version": 1,
            "kind": kind,
            "attributes": {
                name: wrap_typed(value) for name, value in (attributes or {}).items()
            },
            "text": text,
            "payload": payload,
        }
        op: Dict[str, Any] = {
            "Create": {"parent": parent, "element": element_snapshot}
        }
        if index is not None:
            if index < 0:
                raise ValueError("create index must be >= 0")
            op["Create"]["index"] = index
        else:
            op["Create"]["index"] = 0
        self.apply_patch(
            ops=[op],
            reason=f"Create element {element_id} ({kind})",
        )

    def set_attribute(self, element: str, name: str, value: Any) -> None:
        _check_id(element, "element id")
        if not name or len(name) > 256:
            raise ValueError("attribute name must be 1..=256 characters")
        self.apply_patch(
            ops=[{"SetAttribute": {"element": element, "name": name, "value": wrap_typed(value)}}],
            reason=f"Set attribute {name} on {element}",
        )

    def remove_attribute(self, element: str, name: str) -> None:
        _check_id(element, "element id")
        self.apply_patch(
            ops=[{"RemoveAttribute": {"element": element, "name": name}}],
            reason=f"Remove attribute {name} on {element}",
        )

    def replace_text(self, element: str, text: str) -> None:
        _check_id(element, "element id")
        self.apply_patch(
            ops=[{"ReplaceText": {"element": element, "text": text}}],
            reason=f"Replace text on {element}",
        )

    def append_text(self, element: str, text: str) -> None:
        _check_id(element, "element id")
        self.apply_patch(
            ops=[{"AppendText": {"element": element, "text": text}}],
            reason=f"Append text to {element}",
        )

    def move_element(self, element: str, parent: str, index: int = 0) -> None:
        _check_id(element, "element id")
        _check_id(parent, "parent id")
        if index < 0:
            raise ValueError("move index must be >= 0")
        self.apply_patch(
            ops=[{"Move": {"element": element, "parent": parent, "index": index}}],
            reason=f"Move {element} to {parent}",
        )

    def delete_element(self, element: str) -> None:
        _check_id(element, "element id")
        self.apply_patch(
            ops=[{"Delete": {"element": element}}],
            reason=f"Delete element {element}",
        )


class JobClient:
    """Client for managing background and sandbox jobs."""

    def __init__(self, transport: HostTransport) -> None:
        self._transport = transport

    def spawn(
        self,
        job_id: str,
        operation: str,
        capabilities: Optional[List[str]] = None,
        payload: Optional[Any] = None,
    ) -> Any:
        _check_id(job_id, "job id")
        if not operation or len(operation) > 256:
            raise ValueError("operation name must be 1..=256 characters")
        return self._transport.send_job(
            job_id=job_id,
            operation=operation,
            capabilities=capabilities or [],
            payload=payload,
        )


class ArtifactClient:
    """Client for accessing stored artifacts."""

    def __init__(self, transport: HostTransport) -> None:
        self._transport = transport

    def get(self, artifact_id: str) -> Any:
        _check_id(artifact_id, "artifact id")
        return self._transport.send_query("get_artifact", {"id": artifact_id})


class Tool:
    """Base class for declarative tool definitions."""

    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    parameter_schema: Dict[str, Any] = {}

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name or self.__class__.__name__,
            version=self.version,
            description=self.description,
            parameter_schema=self.parameter_schema,
        )

    def execute(self, arguments: Dict[str, Any], context: Any) -> Any:
        raise NotImplementedError


class Director:
    """Base class for declarative Director definitions."""

    name: str = ""
    priority: int = 0
    description: str = ""

    def to_spec(self) -> DirectorSpec:
        return DirectorSpec(
            name=self.name or self.__class__.__name__,
            priority=self.priority,
            description=self.description,
        )

    def on_yield(self, turn_view: Any) -> Any:
        return {"action": "pass"}


class Component:
    """Base class for declarative Component definitions."""

    name: str = ""
    schema: Dict[str, Any] = {}

    def to_spec(self) -> ComponentSpec:
        return ComponentSpec(
            name=self.name or self.__class__.__name__,
            schema=self.schema,
        )
