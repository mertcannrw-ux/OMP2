"""Implementation of the @remote decorator and client-side AST inspection."""

import ast
import functools
import hashlib
import inspect
import json
import textwrap
from typing import Any, Callable, Dict, List, Optional, Set

from .protocol import HostTransport, StreamTransport

PROHIBITED_MODULES: Set[str] = {
    "os", "subprocess", "sys", "socket", "ctypes", "pty", "posix", "nt", "shutil",
    "builtins", "importlib", "signal", "multiprocessing", "threading",
}

FORBIDDEN_CALLS: Set[str] = {
    "eval", "exec", "compile", "globals", "locals",
    "getattr", "setattr", "delattr", "__subclasses__", "__getattribute__",
}

DYNAMIC_IMPORT_CALLS: Set[str] = {
    "__import__", "import_module",
}

class RemoteValidationError(Exception):
    """Raised when remote function source or arguments fail validation."""
    pass


class ASTValidator(ast.NodeVisitor):
    """Inspects an AST for safety, prohibited imports, and forbidden calls."""

    def __init__(self, declared_capabilities: Set[str]) -> None:
        self.declared_capabilities = declared_capabilities
        self.imported_modules: List[str] = []
        self.errors: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base = alias.name.split(".")[0]
            self.imported_modules.append(base)
            if base in PROHIBITED_MODULES:
                cap_needed = f"import_{base}"
                if cap_needed not in self.declared_capabilities and "system_all" not in self.declared_capabilities:
                    self.errors.append(f"Prohibited import '{base}' without capability '{cap_needed}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base = node.module.split(".")[0]
            self.imported_modules.append(base)
            if base in PROHIBITED_MODULES:
                cap_needed = f"import_{base}"
                if cap_needed not in self.declared_capabilities and "system_all" not in self.declared_capabilities:
                    self.errors.append(f"Prohibited import '{base}' without capability '{cap_needed}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in DYNAMIC_IMPORT_CALLS:
                self.errors.append(f"Dynamic import via '{func_name}()' is forbidden in remote function")
            elif func_name in FORBIDDEN_CALLS:
                self.errors.append(f"Forbidden dynamic call to '{func_name}()' in remote function")
            elif func_name == "open":
                if not any(c in self.declared_capabilities for c in ("fs_read", "fs_write", "fs_all")):
                    self.errors.append("Call to 'open()' requires 'fs_read' or 'fs_write' capability")
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in DYNAMIC_IMPORT_CALLS:
                self.errors.append(f"Dynamic import via '{attr_name}()' is forbidden in remote function")
            elif attr_name in FORBIDDEN_CALLS:
                self.errors.append(f"Forbidden dynamic call to '{attr_name}()' in remote function")
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "importlib":
                self.errors.append(f"Dynamic import via 'importlib.{attr_name}()' is forbidden in remote function")
        else:
            self.errors.append(
                f"Dynamic call expression of type '{type(node.func).__name__}' is forbidden in remote function"
            )
        self.generic_visit(node)


def validate_closure(fn: Callable[..., Any]) -> None:
    """Rejects closures and dynamic source capture of free/nonlocal variables."""
    if getattr(fn, "__closure__", None) is not None:
        raise RemoteValidationError(
            f"Function '{fn.__name__}' captures closure variables; closures are forbidden in @remote functions"
        )
    if hasattr(fn, "__code__") and fn.__code__.co_freevars:
        raise RemoteValidationError(
            f"Function '{fn.__name__}' has free variables {fn.__code__.co_freevars}; closures are forbidden in @remote functions"
        )
    if hasattr(inspect, "getclosurevars"):
        cv = inspect.getclosurevars(fn)
        if cv.nonlocals:
            raise RemoteValidationError(
                f"Function '{fn.__name__}' captures nonlocal variables {list(cv.nonlocals.keys())}; closures are forbidden in @remote functions"
            )
        if cv.globals:
            raise RemoteValidationError(
                f"Function '{fn.__name__}' references global state {list(cv.globals)}; import modules inside the function and pass data as arguments"
            )


def _extract_source(target_fn: Callable[..., Any]) -> str:
    """Capture only inspectable function source; never search caller locals."""
    return textwrap.dedent(inspect.getsource(target_fn))


def remote(
    capabilities: Optional[List[str]] = None,
    max_payload_bytes: int = 1024 * 1024,
    timeout_ms: int = 30000,
    transport: Optional[HostTransport] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator marking a function for bounded remote sandbox execution.

    Inspects AST/source, captures imports, enforces capability declarations,
    rejects dynamic closures, validates payload limits, and submits sandbox jobs.
    """
    declared_caps = set(capabilities or [])
    if max_payload_bytes < 1 or not 1 <= timeout_ms <= 30000:
        raise RemoteValidationError("Remote payload budget must be positive and timeout must be 1..30000 ms")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # 1. Inspect and normalize function source (using textwrap.dedent, not cleandoc)
        try:
            source = _extract_source(fn)
        except (OSError, TypeError) as err:
            raise RemoteValidationError(f"Unable to retrieve source code for {fn.__name__}: {err}")

        source_bytes = len(source.encode("utf-8"))
        if source_bytes > max_payload_bytes:
            raise RemoteValidationError(
                f"Source code size ({source_bytes} bytes) exceeds payload limit ({max_payload_bytes} bytes)"
            )
        try:
            tree = ast.parse(source)
        except SyntaxError as err:
            raise RemoteValidationError(f"Syntax error in remote function {fn.__name__}: {err}")

        validator = ASTValidator(declared_caps)
        validator.visit(tree)

        if validator.errors:
            raise RemoteValidationError(f"Validation failed for {fn.__name__}: {'; '.join(validator.errors)}")

        # 3. Reject dynamic source capture
        validate_closure(fn)

        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Validate serializability of arguments
            try:
                payload = json.dumps({"args": args, "kwargs": kwargs}, allow_nan=False)
            except (TypeError, OverflowError, ValueError) as err:
                raise RemoteValidationError(f"Arguments to {fn.__name__} are not JSON serializable: {err}")

            payload_bytes = len(payload.encode("utf-8"))
            if payload_bytes > max_payload_bytes:
                raise RemoteValidationError(
                    f"Arguments payload size ({payload_bytes} bytes) exceeds limit ({max_payload_bytes} bytes)"
                )
            total_bytes = source_bytes + payload_bytes
            if total_bytes > max_payload_bytes:
                raise RemoteValidationError(
                    f"Total payload size ({total_bytes} bytes) exceeds limit ({max_payload_bytes} bytes)"
                )
            # Submit bounded sandbox request to host. The job id carries a
            # random nonce: the old `remote_{fn}_{hash[:8]}` form collided for
            # repeated calls of the same function.
            import uuid
            active_transport = transport or StreamTransport()
            nonce = uuid.uuid4().hex[:8]
            return active_transport.send_job(
                job_id=f"remote_{fn.__name__}_{source_hash[:8]}_{nonce}",
                operation="execute_remote",
                capabilities=list(declared_caps),
                payload={
                    "function_name": fn.__name__,
                    "source_code": source,
                    "source_hash": source_hash,
                    "imports": validator.imported_modules,
                    "arguments": {"args": args, "kwargs": kwargs},
                    "timeout_ms": timeout_ms,
                },
            )

        wrapper.__remote_spec__ = {  # type: ignore[attr-defined]
            "name": fn.__name__,
            "source_hash": source_hash,
            "capabilities": list(declared_caps),
            "imports": validator.imported_modules,
            "max_payload_bytes": max_payload_bytes,
            "timeout_ms": timeout_ms,
        }
        return wrapper

    return decorator
