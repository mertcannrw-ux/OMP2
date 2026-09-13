"""Reject remote functions that cannot cross the sandbox boundary safely."""

import pytest
from omp_sdk.remote import RemoteValidationError, remote


def test_remote_rejects_closure() -> None:
    outer_var = 42

    def closure_fn(x: int) -> int:
        return x + outer_var

    with pytest.raises(RemoteValidationError):
        remote()(closure_fn)


def test_remote_rejects_dynamic_import() -> None:
    def dynamic_import_fn():
        mod = __import__("math")
        return mod.sqrt(4)

    with pytest.raises(RemoteValidationError):
        remote()(dynamic_import_fn)


def test_remote_rejects_forbidden_calls() -> None:
    def eval_fn(code: str) -> None:
        eval(code)

    with pytest.raises(RemoteValidationError):
        remote()(eval_fn)

    def getattr_fn(obj: object) -> None:
        getattr(obj, "foo")

    with pytest.raises(RemoteValidationError):
        remote()(getattr_fn)


def test_remote_rejects_dynamic_call_expressions() -> None:
    def expr_call_fn(funcs: list) -> None:
        funcs[0]()

    with pytest.raises(RemoteValidationError):
        remote()(expr_call_fn)


def test_remote_rejects_oversized_payload() -> None:
    def sample_fn(x: int) -> int:
        return x * 2

    with pytest.raises(RemoteValidationError):
        remote(max_payload_bytes=10)(sample_fn)

    @remote(max_payload_bytes=1024)
    def args_fn(data: str) -> str:
        return data

    with pytest.raises(RemoteValidationError):
        args_fn("A" * 2048)
