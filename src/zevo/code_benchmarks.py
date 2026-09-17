"""Declarative bindings from public coding datasets to Zevo's code evaluator.

The Run pipeline only carries the adapter name resolved here.  Dataset-specific
field handling stays behind the shared code-execution metric instead of leaking
into Task forms, inference, or ticket orchestration.
"""
from __future__ import annotations

import gzip
import hashlib
import csv
import json
import os
import shutil
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.parse import urlparse
from uuid import uuid4


CodeExecutionAdapter: TypeAlias = Literal[
    "", "humaneval_plus", "mbpp_plus", "livecodebench", "code_contests",
]
CodeAnswerScope: TypeAlias = Literal["test", "validation"]

_ADAPTERS: dict[str, CodeExecutionAdapter] = {
    "evalplus/humanevalplus": "humaneval_plus",
    "evalplus/mbppplus": "mbpp_plus",
    "livecodebench/code_generation_lite": "livecodebench",
    "sam-paech/livecodebench-code_generation_lite": "livecodebench",
    "deepmind/code_contests": "code_contests",
}

_ANSWER_PREFIX = "zevo-code-answer:v1:"
_VALIDATION_ANSWER_PREFIX = "zevo-code-answer:validation:v1:"
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
    *, scope: CodeAnswerScope = "test",
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
            if _token_parts(raw) is not None:
                projected[field] = _rehome_token(raw, scope=scope)
                continue
            serialized = raw
        else:
            serialized = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
        projected[field] = _store_code_answer(serialized, scope=scope)
    return projected


def _answer_root(scope: CodeAnswerScope) -> Path:
    from zevo.paths import holdout_root, validation_code_answers_root

    if scope == "test":
        return Path(holdout_root()) / "code-execution-answers"
    if scope == "validation":
        return Path(validation_code_answers_root())
    raise ValueError(f"unknown code-answer scope: {scope!r}")


def _answer_prefix(scope: CodeAnswerScope) -> str:
    if scope == "test":
        return _ANSWER_PREFIX
    if scope == "validation":
        return _VALIDATION_ANSWER_PREFIX
    raise ValueError(f"unknown code-answer scope: {scope!r}")


def _token_parts(value: str) -> tuple[CodeAnswerScope, str] | None:
    for scope, prefix in (
        ("test", _ANSWER_PREFIX),
        ("validation", _VALIDATION_ANSWER_PREFIX),
    ):
        if value.startswith(prefix):
            digest = value[len(prefix):]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid code-answer sidecar token")
            return scope, digest
    return None


def _ensure_answer_root(scope: CodeAnswerScope) -> Path:
    root = _answer_root(scope)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    return root


def _rehome_token(value: str, *, scope: CodeAnswerScope) -> str:
    parts = _token_parts(value)
    if parts is None:
        return value
    source_scope, digest = parts
    if source_scope == scope:
        return value
    source = _answer_root(source_scope) / f"{digest}.txt.gz"
    root = _ensure_answer_root(scope)
    target = root / source.name
    if not target.is_file():
        if not source.is_file():
            raise ValueError(f"code-answer sidecar is unavailable: {digest}")
        temporary = root / f".{digest}.{uuid4().hex}.tmp"
        try:
            shutil.copy2(source, temporary)
            temporary.chmod(0o600)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return _answer_prefix(scope) + digest


def _store_code_answer(raw: str, *, scope: CodeAnswerScope = "test") -> str:
    """Write one hidden code-answer payload once and return its opaque token."""
    # Do not create a second, full-size UTF-8 copy. Some LiveCodeBench cells
    # are hundreds of megabytes; hashing and TextIOWrapper encoding the whole
    # value at once was enough to push the web task over its memory limit.
    def chunks() -> Iterable[bytes]:
        for start in range(0, len(raw), _UTF8_CHUNK_CHARS):
            yield raw[start:start + _UTF8_CHUNK_CHARS].encode("utf-8")

    return _store_code_answer_chunks(chunks, scope=scope)


def store_code_answer_buffer(
    raw: object, *, scope: CodeAnswerScope = "test",
) -> str:
    """Externalize an existing UTF-8 buffer without constructing a huge str.

    PyArrow exposes StringScalar data as a zero-copy buffer. LiveCodeBench's
    private cases use that route so conversion never holds both Arrow storage
    and a hundreds-of-megabytes Python string.
    """
    view = memoryview(raw)

    def chunks() -> Iterable[memoryview]:
        for start in range(0, len(view), _UTF8_CHUNK_CHARS):
            yield view[start:start + _UTF8_CHUNK_CHARS]

    return _store_code_answer_chunks(chunks, scope=scope)


def _store_code_answer_chunks(
    chunks: Callable[[], Iterable[bytes | memoryview]],
    *, scope: CodeAnswerScope = "test",
) -> str:
    """Hash and compress a repeatable byte stream without joining it."""

    hasher = hashlib.sha256()
    for chunk in chunks():
        hasher.update(chunk)
    digest = hasher.hexdigest()
    root = _ensure_answer_root(scope)
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
    return _answer_prefix(scope) + digest


def resolve_code_answer(value: object) -> object:
    """Resolve an engine-owned content-addressed answer token when present."""
    if not isinstance(value, str):
        return value
    parts = _token_parts(value)
    if parts is None:
        return value
    scope, digest = parts
    path = _answer_root(scope) / f"{digest}.txt.gz"
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise ValueError(f"code-answer sidecar is unavailable: {digest}") from exc


def rehome_code_answers_csv(
    adapter: CodeExecutionAdapter, path: str | Path, *, scope: CodeAnswerScope,
) -> int:
    """Make a cached coding CSV's sidecars readable in its scoring lane.

    Cache entries are shared by Test and Validation. A CSV cached for one lane
    can be reused by the other without redownloading large benchmark shards,
    but only after copying its compressed answer sidecars into that lane's
    separately mounted root and replacing their scoped tokens in the copy.
    """
    fields = _ANSWER_FIELDS.get(adapter, ())
    if not fields:
        return 0
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    source = Path(path)
    temporary = source.with_name(f".{source.name}.{uuid4().hex}.tmp")
    changed = 0
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as reader_handle, \
                temporary.open("w", encoding="utf-8", newline="") as writer_handle:
            reader = csv.DictReader(reader_handle)
            if not reader.fieldnames:
                raise ValueError(f"cached coding dataset has no CSV header: {source}")
            writer = csv.DictWriter(writer_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                for field in fields:
                    old = row.get(field) or ""
                    if old:
                        new = _rehome_token(old, scope=scope)
                        if new != old:
                            row[field] = new
                            changed += 1
                writer.writerow(row)
        if changed:
            os.replace(temporary, source)
        return changed
    finally:
        temporary.unlink(missing_ok=True)
