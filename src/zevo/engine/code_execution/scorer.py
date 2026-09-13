"""Normalize coding benchmarks and score them through one worker protocol."""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import pickle
import re
import resource
import subprocess
import sys
import tempfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from zevo.code_benchmarks import CodeExecutionAdapter, resolve_code_answer
from zevo.engine.artifact_validation import _read_records


_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_SUPPORTED = {"humaneval_plus", "mbpp_plus", "livecodebench", "code_contests"}


class _PrimitiveUnpickler(pickle.Unpickler):
    """Decode LCB's primitive pickle envelope without permitting globals."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"pickle global is forbidden: {module}.{name}")


def extract_python_code(value: Any) -> str:
    """Accept raw code or a model response containing fenced Python code."""
    text = "" if value is None else str(value)
    matches = _FENCE.findall(text)
    if matches:
        return matches[-1].strip("\r\n")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    # HumanEval completions may intentionally start with four spaces because
    # they are appended to the supplied function signature/docstring.
    return text.strip("\r\n")


def _jsonish(value: Any, *, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _lcb_cases(value: Any) -> list[dict[str, Any]]:
    value = resolve_code_answer(value)
    direct = _jsonish(value, default=None)
    if isinstance(direct, list):
        return [dict(item) for item in direct if isinstance(item, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        payload = zlib.decompress(base64.b64decode(value))
        decoded = _PrimitiveUnpickler(io.BytesIO(payload)).load()
        if isinstance(decoded, bytes):
            decoded = decoded.decode("utf-8")
        decoded = _jsonish(decoded, default=[])
    except (ValueError, TypeError, OSError, EOFError, pickle.UnpicklingError):
        return []
    return [dict(item) for item in decoded if isinstance(item, dict)] if isinstance(
        decoded, list
    ) else []


def _code_contest_cases(value: Any) -> list[dict[str, Any]]:
    """Convert CodeContests' parallel input/output arrays to worker cases."""
    decoded = _jsonish(resolve_code_answer(value), default={})
    if not isinstance(decoded, dict):
        return []
    inputs = decoded.get("input") or []
    outputs = decoded.get("output") or []
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return []
    return [
        {"input": source, "output": expected, "testtype": "stdin"}
        for source, expected in zip(inputs, outputs)
    ]


def _job(adapter: CodeExecutionAdapter, row: dict[str, Any], code: str) -> dict[str, Any]:
    if adapter == "humaneval_plus":
        entry_point = str(row.get("entry_point") or "").strip()
        prompt = str(row.get("prompt") or "")
        if entry_point and not re.search(
            rf"\b(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", code,
        ):
            code = prompt + code
        return {
            "kind": "python_harness",
            "code": code,
            "harness": str(row.get("test") or ""),
            "entry_point": entry_point,
            "invoke_check": True,
        }
    if adapter == "mbpp_plus":
        return {
            "kind": "python_harness",
            "code": code,
            "harness": str(row.get("test") or ""),
            "entry_point": "",
            "invoke_check": False,
        }
    if adapter == "livecodebench":
        metadata = _jsonish(row.get("metadata"), default={})
        public = _lcb_cases(row.get("public_test_cases"))
        private = _lcb_cases(row.get("private_test_cases"))
        return {
            "kind": "livecodebench",
            "code": code,
            "function_name": str(
                metadata.get("func_name") or metadata.get("function_name") or ""
            ) if isinstance(metadata, dict) else "",
            "tests": [*public, *private],
        }
    if adapter == "code_contests":
        public = _code_contest_cases(row.get("public_tests"))
        private = _code_contest_cases(row.get("private_tests"))
        generated = _code_contest_cases(row.get("generated_tests"))
        return {
            "kind": "livecodebench",
            "code": code,
            "function_name": "",
            "tests": [*public, *private, *generated],
        }
    raise ValueError(f"unsupported code execution adapter: {adapter!r}")


def _limits(timeout_seconds: float, test_count: int):
    def apply() -> None:
        def set_soft(kind: int, wanted: int) -> None:
            try:
                _soft, hard = resource.getrlimit(kind)
                target = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
                resource.setrlimit(kind, (target, hard))
            except (OSError, ValueError):
                # Limit availability differs across macOS development and the
                # Linux evaluator image. Other independent limits still apply.
                pass

        cpu = max(2, min(300, int(timeout_seconds * max(1, test_count)) + 2))
        set_soft(resource.RLIMIT_CPU, cpu)
        memory = 2 * 1024 * 1024 * 1024
        set_soft(resource.RLIMIT_AS, memory)
        set_soft(resource.RLIMIT_FSIZE, 16 * 1024 * 1024)
        set_soft(resource.RLIMIT_NOFILE, 64)
        if hasattr(resource, "RLIMIT_NPROC"):
            set_soft(resource.RLIMIT_NPROC, 32)
        if os.geteuid() == 0:
            try:
                os.setgroups([])
                os.setgid(65534)
                os.setuid(65534)
            except OSError:
                pass
    return apply


def _worker_timeout(job: dict[str, Any], case_timeout: float) -> float:
    # A non-terminating candidate is stopped by the worker's per-case alarm.
    # Correct solutions get enough wall time for a large official test vector.
    count = len(job.get("tests") or []) if job.get("kind") == "livecodebench" else 1
    return max(10.0, min(600.0, case_timeout * max(1, count) + 5.0))


def _run_job(job: dict[str, Any], *, case_timeout: float) -> dict[str, Any]:
    job = {**job, "case_timeout_seconds": case_timeout}
    worker = Path(__file__).with_name("worker.py")
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    with tempfile.TemporaryDirectory(prefix="zevo-code-eval-") as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, str(worker)],
                input=json.dumps(job, ensure_ascii=False),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmp,
                env=env,
                timeout=_worker_timeout(job, case_timeout),
                preexec_fn=_limits(case_timeout, len(job.get("tests") or []) or 1),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "reason": "timeout"}
        if proc.returncode != 0:
            return {
                "passed": False,
                "reason": "runtime_error",
                "detail": (proc.stderr or proc.stdout or "worker exited")[-500:],
            }
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {
                "passed": False,
                "reason": "harness_error",
                "detail": (proc.stderr or proc.stdout or "invalid worker result")[-500:],
            }
        return result if isinstance(result, dict) else {
            "passed": False, "reason": "harness_error",
        }


def _case_timeout_seconds() -> float:
    try:
        return max(0.1, float(os.environ.get("ZEVO_CODE_CASE_TIMEOUT_SECONDS", "6")))
    except ValueError:
        return 6.0


def score_code_benchmark(
    *,
    adapter: str,
    predictions: Path,
    scoring_set: Path,
    prediction_column: str,
) -> dict[str, Any]:
    """Return a standard metrics object for one normalized code benchmark."""
    if adapter not in _SUPPORTED:
        raise ValueError(f"unsupported code execution adapter: {adapter!r}")
    _columns, rows = _read_records(scoring_set, label="code scoring set")
    with predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))
    if len(rows) != len(prediction_rows):
        raise ValueError("code predictions and scoring set have different row counts")
    if prediction_rows and prediction_column not in prediction_rows[0]:
        raise ValueError(f"prediction column {prediction_column!r} is absent")

    reasons: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    timeout = _case_timeout_seconds()
    passed = 0
    for index, (row, prediction) in enumerate(zip(rows, prediction_rows)):
        code = extract_python_code(prediction.get(prediction_column, ""))
        if not code:
            result = {"passed": False, "reason": "missing_prediction"}
        else:
            try:
                result = _run_job(_job(adapter, row, code), case_timeout=timeout)
            except (OSError, ValueError, TypeError) as exc:
                result = {
                    "passed": False,
                    "reason": "harness_error",
                    "detail": str(exc)[:500],
                }
        ok = result.get("passed") is True
        passed += int(ok)
        reason = "passed" if ok else str(result.get("reason") or "wrong_answer")
        reasons[reason] += 1
        if not ok and len(details) < 50:
            details.append({
                "row": index,
                "reason": reason,
                "detail": str(result.get("detail") or "")[:500],
            })

    total = len(rows)
    score = passed / total if total else 0.0
    return {
        "status": "succeeded",
        "score": score,
        "metric_used": "pass_at_1",
        "pass_at_1": score,
        "adapter": adapter,
        "passed": passed,
        "failed": total - passed,
        "total": total,
        "outcomes": dict(sorted(reasons.items())),
        "failures": details,
    }
