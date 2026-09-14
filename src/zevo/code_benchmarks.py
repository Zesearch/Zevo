"""Declarative bindings from public coding datasets to Zevo's code evaluator.

The Run pipeline only carries the adapter name resolved here.  Dataset-specific
field handling stays behind the shared code-execution metric instead of leaking
into Task forms, inference, or ticket orchestration.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.parse import urlparse
from uuid import uuid4


CodeExecutionAdapter: TypeAlias = Literal[
    "", "humaneval_plus", "mbpp_plus", "livecodebench", "code_contests",
]

_ADAPTERS: dict[str, CodeExecutionAdapter] = {
    "evalplus/humanevalplus": "humaneval_plus",
    "evalplus/mbppplus": "mbpp_plus",
    "livecodebench/code_generation_lite": "livecodebench",
    "sam-paech/livecodebench-code_generation_lite": "livecodebench",
    "deepmind/code_contests": "code_contests",
}

_ANSWER_PREFIX = "zevo-code-answer:v1:"
_UTF8_CHUNK_CHARS = 1024 * 1024
_ANSWER_FIELDS: dict[CodeExecutionAdapter, tuple[str, ...]] = {
    "livecodebench": ("private_test_cases",),
    "code_contests": (
        "private_tests", "generated_tests", "solutions",
        "incorrect_solutions",
    ),
}


def _hub_id(reference: str) -> str:
    value = (reference or "").strip().rstrip("/")
    if value.startswith("hf://datasets/"):
        value = value[len("hf://datasets/"):]
    elif value.startswith("hf://"):
        value = value[len("hf://"):]
    elif value.startswith(("https://", "http://")):
        parsed = urlparse(value)
        if parsed.netloc.casefold() not in {
            "huggingface.co", "www.huggingface.co",
        }:
            return ""
        parts = [part for part in parsed.path.split("/") if part]
        if parts[:1] == ["datasets"]:
            parts = parts[1:]
        value = "/".join(parts[:2])
    return value.casefold()


def code_execution_adapter_for(reference: str) -> CodeExecutionAdapter:
    """Return the registered adapter for a Hugging Face dataset reference."""
    return _ADAPTERS.get(_hub_id(reference), "")


def supports_code_execution(reference: str) -> bool:
    return bool(code_execution_adapter_for(reference))


def externalize_code_answers(
    adapter: CodeExecutionAdapter, row: dict[str, object],
) -> dict[str, object]:
    """Move exceptionally large hidden tests out of a tabular scoring row.

    LiveCodeBench can put hundreds of megabytes of compressed private cases in
    one CSV cell.  A content-addressed sidecar keeps the ordinary questions and
    row identity streamable while the token remains an answer field and is
    therefore removed before Inference.
    """
    fields = _ANSWER_FIELDS.get(adapter, ())
    projected = dict(row)
    for field in fields:
        raw = projected.get(field)
        if raw is None or raw == "":
            continue
        if isinstance(raw, str):
            if raw.startswith(_ANSWER_PREFIX):
                continue
            serialized = raw
        else:
            serialized = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
        projected[field] = _store_code_answer(serialized)
    return projected


def _store_code_answer(raw: str) -> str:
    """Write one hidden code-answer payload once and return its opaque token."""
    # Do not create a second, full-size UTF-8 copy. Some LiveCodeBench cells
    # are hundreds of megabytes; hashing and TextIOWrapper encoding the whole
    # value at once was enough to push the web task over its memory limit.
    def chunks() -> Iterable[bytes]:
        for start in range(0, len(raw), _UTF8_CHUNK_CHARS):
            yield raw[start:start + _UTF8_CHUNK_CHARS].encode("utf-8")

    return _store_code_answer_chunks(chunks)


def store_code_answer_buffer(raw: object) -> str:
    """Externalize an existing UTF-8 buffer without constructing a huge str.

    PyArrow exposes StringScalar data as a zero-copy buffer. LiveCodeBench's
    private cases use that route so conversion never holds both Arrow storage
    and a hundreds-of-megabytes Python string.
    """
    view = memoryview(raw)

    def chunks() -> Iterable[memoryview]:
        for start in range(0, len(view), _UTF8_CHUNK_CHARS):
            yield view[start:start + _UTF8_CHUNK_CHARS]

    return _store_code_answer_chunks(chunks)


def _store_code_answer_chunks(
    chunks: Callable[[], Iterable[bytes | memoryview]],
) -> str:
    """Hash and compress a repeatable byte stream without joining it."""

    hasher = hashlib.sha256()
    for chunk in chunks():
        hasher.update(chunk)
    digest = hasher.hexdigest()
    from zevo.paths import holdout_root

    root = Path(holdout_root()) / "code-execution-answers"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    target = root / f"{digest}.txt.gz"
    if not target.is_file():
        temporary = root / f".{digest}.{uuid4().hex}.tmp"
        try:
            with gzip.open(temporary, "wb") as handle:
                for chunk in chunks():
                    handle.write(chunk)
            temporary.chmod(0o600)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    target.chmod(0o600)
    return _ANSWER_PREFIX + digest


def resolve_code_answer(value: object) -> object:
    """Resolve an engine-owned content-addressed answer token when present."""
    if not isinstance(value, str) or not value.startswith(_ANSWER_PREFIX):
        return value
    digest = value[len(_ANSWER_PREFIX):]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid code-answer sidecar token")
    from zevo.paths import holdout_root

    path = Path(holdout_root()) / "code-execution-answers" / f"{digest}.txt.gz"
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise ValueError(f"code-answer sidecar is unavailable: {digest}") from exc
