"""Protocol message definitions and transport layer for omp_sdk."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Live function handles only; the host's active DOM gates every invocation.
_tool_handles: Dict[str, Any] = {}


class HostError(Exception):
    """Structured host-side failure, preserving code/retryable/diagnostics."""

    def __init__(
        self,
        code: str,
        message: str,
        retryable: bool = False,
        diagnostics: Optional[Any] = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable
        self.diagnostics = diagnostics


def _coerce_host_error(error: Any) -> HostError:
    if isinstance(error, HostError):
        return error
    if isinstance(error, dict):
        return HostError(
            code=str(error.get("code", "host_error")),
            message=str(error.get("message", error)),
            retryable=bool(error.get("retryable", False)),
            diagnostics=error.get("diagnostics"),
        )
    return HostError(code="host_error", message=str(error))


def invoke_tool(name: str, arguments: Dict[str, Any]) -> Any:
    try:
        tool, context = _tool_handles[name]
    except KeyError:
        raise HostError(code="unknown_tool", message=f"no live tool handle for {name!r}")
    return tool.execute(arguments, context)



@dataclass
class ToolSpec:
    name: str
    version: str
    description: str
    parameter_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DirectorSpec:
    name: str
    priority: int = 0
    description: str = ""


@dataclass
class ComponentSpec:
    name: str
    schema: Dict[str, Any] = field(default_factory=dict)


class HostTransport:
    """Interface for communicating with the host session."""

    def send_query(self, query_type: str, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def send_patch(self, ops: List[Dict[str, Any]], reason: str) -> None:
        raise NotImplementedError

    def send_job(self, job_id: str, operation: str, capabilities: List[str], payload: Any) -> Any:
        raise NotImplementedError

    def register_declarations(
        self,
        extension_id: str,
        tools: List[ToolSpec],
        directors: List[DirectorSpec],
        components: List[ComponentSpec],
    ) -> None:
        raise NotImplementedError


class StreamTransport(HostTransport):
    """Synchronous bounded request/reply transport owned by the worker host."""

    def _request(self, request: Dict[str, Any]) -> Any:
        import sys
        encoded = json.dumps({"host_request": request}, allow_nan=False)
        if len(encoded.encode("utf-8")) > 65536:
            raise ValueError("Host request exceeds 65536 bytes")
        sys.__stdout__.write(encoded + "\n")
        sys.__stdout__.flush()
        line = sys.__stdin__.readline(1048577)
        if not line or len(line.encode("utf-8")) > 1048576:
            raise HostError(code="host_response", message="Host response missing or oversized")
        response = json.loads(line)
        if "error" in response:
            raise _coerce_host_error(response["error"])
        return response["result"]

    def send_query(self, query_type: str, params: Dict[str, Any]) -> Any:
        return self._request({"type": "query", "query": query_type, "params": params})

    def send_patch(self, ops: List[Dict[str, Any]], reason: str) -> None:
        self._request({"type": "patch", "ops": ops, "reason": reason})

    def send_job(self, job_id: str, operation: str, capabilities: List[str], payload: Any) -> Any:
        return self._request({"type": "job", "job_id": job_id, "operation": operation, "capabilities": capabilities, "payload": payload})

    def register_declarations(self, extension_id: str, tools: List[ToolSpec], directors: List[DirectorSpec], components: List[ComponentSpec]) -> None:
        from dataclasses import asdict
        self._request({"type": "register", "extension_id": extension_id, "tools": [asdict(tool) for tool in tools], "directors": [asdict(director) for director in directors], "components": [asdict(component) for component in components]})


class MockTransport(HostTransport):
    """In-memory transport for testing and standalone validation.

    Unlike the old pass-through mock, patches are shape-checked (op variants,
    reason bounds) so wrapper tests fail the same way the real host would.
    """

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

    def __init__(self) -> None:
        self.queries: List[Dict[str, Any]] = []
        self.patches: List[Dict[str, Any]] = []
        self.jobs: List[Dict[str, Any]] = []
        self.declarations: List[Dict[str, Any]] = []

    def send_query(self, query_type: str, params: Dict[str, Any]) -> Any:
        self.queries.append({"type": query_type, "params": params})
        return {"status": "ok", "query": query_type}

    def send_patch(self, ops: List[Dict[str, Any]], reason: str) -> None:
        if not reason.strip() or len(reason) > 4096:
            raise ValueError("patch reason must be 1..=4096 characters")
        if not ops:
            raise ValueError("patch ops list cannot be empty")
        for op in ops:
            if not isinstance(op, dict) or len(op) != 1:
                raise ValueError(f"each op must be a single-variant dict (got {op!r})")
            variant = next(iter(op))
            if variant not in self._VALID_OPS:
                raise ValueError(f"unknown patch op {variant!r}")
        self.patches.append({"ops": ops, "reason": reason})

    def send_job(self, job_id: str, operation: str, capabilities: List[str], payload: Any) -> Any:
        self.jobs.append({
            "job_id": job_id,
            "operation": operation,
            "capabilities": capabilities,
            "payload": payload,
        })
        return {"job_id": job_id, "status": "submitted"}

    def register_declarations(
        self,
        extension_id: str,
        tools: List[ToolSpec],
        directors: List[DirectorSpec],
        components: List[ComponentSpec],
    ) -> None:
        self.declarations.append({
            "extension_id": extension_id,
            "tools": tools,
            "directors": directors,
            "components": components,
        })
