"""Resource-bounded worker for a single code-generation benchmark problem.

This module deliberately uses only stdin/stdout JSON so every dataset adapter
shares the same process and result contract.  OS limits and privilege dropping
are installed by the parent before Python starts; the guards here reduce the
remaining process/network/filesystem surface available to generated code.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import importlib
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import traceback
from decimal import Decimal, InvalidOperation
from typing import Any, Callable


class CaseTimeout(Exception):
    pass


def _alarm(_signum: int, _frame: Any) -> None:
    raise CaseTimeout("candidate timed out")


@contextlib.contextmanager
def _time_limit(seconds: float):
    old_handler = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _blocked(*_args: Any, **_kwargs: Any) -> Any:
    raise PermissionError("operation disabled during code evaluation")


def _install_guards() -> None:
    socket.socket = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    subprocess.Popen = _blocked  # type: ignore[assignment]
    subprocess.run = _blocked  # type: ignore[assignment]
    subprocess.call = _blocked  # type: ignore[assignment]
    subprocess.check_call = _blocked  # type: ignore[assignment]
    subprocess.check_output = _blocked  # type: ignore[assignment]
    for name in (
        "system", "popen", "fork", "forkpty", "kill", "killpg", "putenv",
        "remove", "removedirs", "rmdir", "rename", "renames", "replace",
        "unlink", "truncate", "chmod", "chown", "chroot",
    ):
        if hasattr(os, name):
            setattr(os, name, _blocked)
    for name in dir(os):
        if name.startswith("spawn"):
            setattr(os, name, _blocked)
    if hasattr(os, "open"):
        os.open = _blocked  # type: ignore[assignment]
    for name in ("rmtree", "move", "chown"):
        if hasattr(shutil, name):
            setattr(shutil, name, _blocked)
    importlib.reload = _blocked  # type: ignore[assignment]

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any):
        del file, mode, args, kwargs
        raise PermissionError("filesystem access disabled during code evaluation")

    builtins.open = guarded_open
    io.open = guarded_open  # type: ignore[assignment]


def _namespace() -> dict[str, Any]:
    namespace = {"__name__": "__main__", "__builtins__": builtins.__dict__}
    # LiveCodeBench's reference harness provides this common competitive-
    # programming prelude, so candidates are graded under the same contract.
    exec(
        "from string import *\nfrom re import *\nfrom datetime import *\n"
        "from collections import *\nfrom heapq import *\nfrom bisect import *\n"
        "from copy import *\nfrom math import *\nfrom random import *\n"
        "from statistics import *\nfrom itertools import *\nfrom functools import *\n"
        "from operator import *\nfrom io import *\nfrom typing import *\n",
        namespace,
    )
    return namespace


def _python_harness(job: dict[str, Any], timeout: float) -> None:
    namespace = _namespace()
    with _time_limit(timeout), contextlib.redirect_stdout(io.StringIO()):
        exec(compile(str(job["code"]), "<candidate>", "exec"), namespace)
        exec(compile(str(job["harness"]), "<benchmark-tests>", "exec"), namespace)
        if job.get("invoke_check"):
            check = namespace.get("check")
            candidate = namespace.get(str(job.get("entry_point") or ""))
            if not callable(check) or not callable(candidate):
                raise AssertionError("benchmark check or candidate function is absent")
            check(candidate)


def _parse_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text


def _equivalent(actual: Any, expected: Any) -> bool:
    expected = _parse_value(expected)
    if isinstance(actual, tuple):
        actual = list(actual)
    if isinstance(expected, tuple):
        expected = list(expected)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _equivalent(left, right) for left, right in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _equivalent(actual[key], expected[key]) for key in actual
        )
    return actual == expected


def _stdio_equivalent(actual: str, expected: Any) -> bool:
    left = "\n".join(line.strip() for line in actual.strip().splitlines())
    right = "\n".join(line.strip() for line in str(expected).strip().splitlines())
    if left == right:
        return True
    if _equivalent(_parse_value(left), _parse_value(right)):
        return True
    left_tokens, right_tokens = left.split(), right.split()
    if len(left_tokens) != len(right_tokens):
        return False
    try:
        return all(Decimal(a) == Decimal(b) for a, b in zip(left_tokens, right_tokens))
    except InvalidOperation:
        return False


def _functional(namespace: dict[str, Any], function_name: str) -> Callable[..., Any]:
    solution = namespace.get("Solution")
    if isinstance(solution, type):
        candidate = getattr(solution(), function_name, None)
    else:
        candidate = namespace.get(function_name)
    if not callable(candidate):
        raise AssertionError(f"candidate function {function_name!r} is absent")
    return candidate


def _functional_args(value: Any) -> list[Any]:
    if not isinstance(value, str):
        return list(value) if isinstance(value, tuple) else [value]
    lines = value.strip().splitlines()
    return [_parse_value(line) for line in lines] if lines else []


class _Stdin:
    def __init__(self, value: str):
        self._text = io.StringIO(value)
        self.buffer = io.BytesIO(value.encode("utf-8"))

    def read(self, *args: Any) -> str:
        return self._text.read(*args)

    def readline(self, *args: Any) -> str:
        return self._text.readline(*args)

    def readlines(self, *args: Any) -> list[str]:
        return self._text.readlines(*args)

    def __iter__(self):
        return iter(self._text)


def _livecodebench(job: dict[str, Any], timeout: float) -> None:
    tests = list(job.get("tests") or [])
    if not tests:
        raise ValueError("coding benchmark row contains no decodable test cases")
    function_name = str(job.get("function_name") or "").strip()
    for test in tests:
        if not isinstance(test, dict):
            raise ValueError("LiveCodeBench test case is not an object")
        test_type = str(test.get("testtype") or "").casefold()
        functional = bool(function_name) or test_type == "functional"
        with _time_limit(timeout):
            namespace = _namespace()
            if functional:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(str(job["code"]), "<candidate>", "exec"), namespace)
                    actual = _functional(namespace, function_name)(
                        *_functional_args(test.get("input", ""))
                    )
                if not _equivalent(actual, test.get("output")):
                    raise AssertionError("wrong answer")
            else:
                old_stdin = sys.stdin
                old_open = builtins.open
                output = io.StringIO()
                raw_input = str(test.get("input") or "")
                try:
                    sys.stdin = _Stdin(raw_input)  # type: ignore[assignment]

                    def input_open(
                        file: Any, mode: str = "r", *args: Any, **kwargs: Any,
                    ):
                        if file in (0, "/dev/stdin"):
                            return (
                                io.BytesIO(raw_input.encode("utf-8"))
                                if "b" in mode else io.StringIO(raw_input)
                            )
                        return old_open(file, mode, *args, **kwargs)

                    builtins.open = input_open
                    with contextlib.redirect_stdout(output):
                        try:
                            exec(
                                compile(str(job["code"]), "<candidate>", "exec"),
                                namespace,
                            )
                        except SystemExit:
                            pass
                finally:
                    sys.stdin = old_stdin
                    builtins.open = old_open
                if not _stdio_equivalent(output.getvalue(), test.get("output", "")):
                    raise AssertionError("wrong answer")


def main() -> int:
    try:
        job = json.loads(sys.stdin.read())
        if not isinstance(job, dict):
            raise ValueError("worker input must be a JSON object")
        timeout = float(job.get("case_timeout_seconds") or 6.0)
        if job.get("kind") == "python_harness" and "numpy" in str(job.get("harness")):
            # EvalPlus' trusted test harness imports NumPy. Load it before file
            # APIs are closed; generated code receives only the cached module.
            __import__("numpy")
        _install_guards()
        if job.get("kind") == "python_harness":
            _python_harness(job, timeout)
        elif job.get("kind") == "livecodebench":
            _livecodebench(job, timeout)
        else:
            raise ValueError(f"unknown worker job kind: {job.get('kind')!r}")
        result = {"passed": True, "reason": "passed"}
    except CaseTimeout as exc:
        result = {"passed": False, "reason": "timeout", "detail": str(exc)}
    except (SyntaxError, IndentationError) as exc:
        result = {"passed": False, "reason": "syntax_error", "detail": str(exc)}
    except AssertionError as exc:
        result = {"passed": False, "reason": "wrong_answer", "detail": str(exc)}
    except BaseException as exc:
        result = {
            "passed": False,
            "reason": "runtime_error",
            "detail": "".join(traceback.format_exception_only(type(exc), exc))[-500:],
        }
    sys.__stdout__.write(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
