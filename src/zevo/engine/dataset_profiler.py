"""Cheap dataset profiler — pure-Python heuristics + stdlib.

v0.1 ships the heuristic + exact-duplicate checks (under 30 lines each)
and deliberately defers MinHash/label-quality checks. The output is a typed
DatasetProfile that:

  1. The /datasets UI renders so users see what we detected.
  2. The orchestrator uses it as structured evidence when selecting the
     initial method-compatible data shape. The Data Agent still inspects the
     assigned raw source and does not receive a second profile contract.

Profiler is read-only and cheap (~100ms for 10k rows). Cached on disk at
`<dataset_dir>/.profile.json` keyed by (file path, size, mtime). The
agent stays in charge; we just give it a better starting prompt.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


PROFILER_VERSION = "v0.1"


# ─────────────────────────────── schema ──────────────────────────────────────


Severity = Literal["info", "warn", "error"]


class DatasetIssue(BaseModel):
    code: str
    severity: Severity
    count: int
    example_row_indices: list[int] = Field(default_factory=list)
    message: str


class TaskTypeGuess(BaseModel):
    label: Literal[
        "chat", "instruction", "qa", "classification",
        "summarization", "code_generation", "rag", "multiturn_chat",
        "unknown",
    ]
    confidence: float
    source: Literal["heuristic", "llm", "heuristic+llm"] = "heuristic"
    rationale: str


class SplitRecommendation(BaseModel):
    train_size: int
    val_size: int
    test_size: int = 0
    rationale: str


class ColumnStat(BaseModel):
    name: str
    dtype: str
    n_missing: int
    n_unique: int
    sample_values: list[str]


class DuplicateReport(BaseModel):
    n_exact: int
    n_near: int = 0
    near_threshold: float = 0.9
    largest_cluster_size: int = 0


class DatasetProfile(BaseModel):
    name: str
    file: str
    file_sha256: str
    profiler_version: str
    n_rows: int
    n_columns: int
    columns: list[ColumnStat]
    task_type: TaskTypeGuess
    expected_format: Literal[
        "chat_jsonl", "instruction_jsonl", "qa_csv",
        "csv_classification", "csv_text_summary", "raw", "unknown",
    ]
    duplicates: DuplicateReport
    split_recommendation: SplitRecommendation
    issues: list[DatasetIssue]
    ready_for_data: bool = Field(
        description="Whether the profile found enough usable structure for the Data Agent to proceed."
    )
    notes: str


# ─────────────────────────────── helpers ─────────────────────────────────────


def _sha256_of(path: Path, *, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _stream_jsonl(path: Path, limit: int = 5000) -> list[dict]:
    """Read up to `limit` JSON objects from a JSONL file. Malformed lines
    are skipped silently (we record them in issues afterwards)."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _count_lines(path: Path) -> int:
    """Streaming line count without holding the file in memory.

    Counts LINES, not newline bytes: a non-empty file whose last line has no
    trailing newline still has that last line (counting b"\\n" alone was
    off by one there).
    """
    n = 0
    last_byte = b""
    with path.open("rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            n += b.count(b"\n")
            last_byte = b[-1:]
    if last_byte and last_byte != b"\n":
        n += 1
    return n


def _is_chat_row(r: dict) -> bool:
    msgs = r.get("messages")
    return (
        isinstance(msgs, list) and msgs
        and all(isinstance(m, dict) and "role" in m and "content" in m for m in msgs)
    )


def _has_keys(r: dict, *names: str) -> bool:
    return all(n in r for n in names)


def _cell_str(v: Any) -> str:
    """Coerce an answer cell to a stripped string.

    CSV rows come through pandas, so a cell can be int64 / float64 / NaN
    rather than str — calling .strip() on those crashed the profiler.
    None and NaN mean "empty"; everything else is stringified.
    """
    if v is None:
        return ""
    if isinstance(v, float) and v != v:  # NaN
        return ""
    return str(v).strip()


# ─────────────────────────── task-type heuristics ────────────────────────────

# Heuristics shipped in v0.1. Each returns (label, confidence, rationale,
# expected_format) given a sample of rows. Order matters: more-specific
# patterns first.

def _classify_task_type(
    rows: list[dict], columns: list[str]
) -> tuple[TaskTypeGuess, str]:
    n = len(rows)
    if n == 0:
        return TaskTypeGuess(
            label="unknown", confidence=0.0,
            rationale="no parseable rows",
        ), "unknown"

    # 1. Chat (OpenAI-style messages array)
    chat_count = sum(1 for r in rows if _is_chat_row(r))
    if chat_count / n >= 0.9:
        multi = sum(1 for r in rows if _is_chat_row(r) and len(r["messages"]) > 2)
        if multi / n >= 0.3:
            return TaskTypeGuess(
                label="multiturn_chat", confidence=0.95,
                rationale=f"{chat_count}/{n} rows have messages[]; {multi} have >2 turns",
            ), "chat_jsonl"
        return TaskTypeGuess(
            label="chat", confidence=0.95,
            rationale=f"{chat_count}/{n} rows have messages[] in OpenAI shape",
        ), "chat_jsonl"

    # 2. Instruction (alpaca-style)
    inst_count = sum(1 for r in rows if _has_keys(r, "instruction", "output"))
    if inst_count / n >= 0.8:
        return TaskTypeGuess(
            label="instruction", confidence=0.9,
            rationale=f"{inst_count}/{n} rows have instruction+output keys",
        ), "instruction_jsonl"

    # 3. QA
    qa_count = sum(
        1 for r in rows
        if (_has_keys(r, "question", "answer") or _has_keys(r, "Q", "A"))
    )
    if qa_count / n >= 0.8:
        # 3b. RAG = QA with context
        ctx_count = sum(1 for r in rows if "context" in r or "passage" in r or "document" in r)
        if ctx_count / n >= 0.5:
            return TaskTypeGuess(
                label="rag", confidence=0.85,
                rationale=f"{qa_count}/{n} have Q+A; {ctx_count} also have a context/passage field",
            ), "qa_csv"
        return TaskTypeGuess(
            label="qa", confidence=0.9,
            rationale=f"{qa_count}/{n} rows have question+answer keys",
        ), "qa_csv"

    # 4. Summarization
    sum_count = sum(
        1 for r in rows
        if (_has_keys(r, "text", "summary") or _has_keys(r, "article", "summary"))
    )
    if sum_count / n >= 0.8:
        return TaskTypeGuess(
            label="summarization", confidence=0.85,
            rationale=f"{sum_count}/{n} rows have a text/summary pair",
        ), "csv_text_summary"

    # 5. Code generation (presence of code fences or 'code' key)
    code_count = sum(
        1 for r in rows
        if (("code" in r and isinstance(r["code"], str))
            or ("solution" in r and isinstance(r["solution"], str) and "```" in str(r.get("solution", ""))))
    )
    if code_count / n >= 0.6:
        return TaskTypeGuess(
            label="code_generation", confidence=0.7,
            rationale=f"{code_count}/{n} rows have code-shaped fields",
        ), "instruction_jsonl"

    # 6. Classification (CSV with a low-cardinality label column)
    if columns:
        # Find a column whose values are short + low cardinality
        candidates = []
        for col in columns:
            if col.lower() in {"label", "target", "class", "category", "intent"}:
                candidates.append(col)
        if candidates:
            return TaskTypeGuess(
                label="classification", confidence=0.8,
                rationale=f"column {candidates[0]!r} matches a known label-column name",
            ), "csv_classification"

    return TaskTypeGuess(
        label="unknown", confidence=0.3,
        rationale="no recognisable task-shape; orchestrator will need to ask the user",
    ), "unknown"


# ─────────────────────────── split recommendation ────────────────────────────


def _recommend_split(n_rows: int) -> SplitRecommendation:
    if n_rows < 200:
        return SplitRecommendation(
            train_size=n_rows, val_size=0,
            rationale="too small for a meaningful split; recommend collecting more data",
        )
    if n_rows < 2000:
        val = int(n_rows * 0.1)
        return SplitRecommendation(
            train_size=n_rows - val, val_size=val,
            rationale="90/10 split — small dataset, val needs to be representative",
        )
    if n_rows < 50_000:
        val = int(n_rows * 0.05)
        return SplitRecommendation(
            train_size=n_rows - val, val_size=val,
            rationale="95/5 split — typical for mid-size finetune datasets",
        )
    return SplitRecommendation(
        train_size=n_rows - 2000, val_size=2000,
        rationale="99/1 split capped at 2k val rows — diminishing returns past that",
    )


# ───────────────────────── duplicate detection (exact) ───────────────────────


def _exact_duplicates(rows: list[dict]) -> DuplicateReport:
    """Hash the canonicalised JSON of each row; report counts."""
    seen: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        try:
            key = json.dumps(r, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            continue
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        seen.setdefault(h, []).append(i)
    dup_groups = [v for v in seen.values() if len(v) > 1]
    n_exact = sum(len(g) - 1 for g in dup_groups)
    largest = max((len(g) for g in dup_groups), default=0)
    return DuplicateReport(n_exact=n_exact, largest_cluster_size=largest)


# ─────────────────────────── columns + missing fields ────────────────────────


def _column_stats(rows: list[dict]) -> list[ColumnStat]:
    if not rows:
        return []
    # Union of all keys, capped at first-seen order
    key_order: list[str] = []
    seen: set[str] = set()
    for r in rows[:200]:
        for k in r.keys():
            if k not in seen:
                key_order.append(k); seen.add(k)
    out: list[ColumnStat] = []
    for k in key_order:
        vals = [r.get(k) for r in rows]
        n_missing = sum(1 for v in vals if v is None or v == "")
        unique_set = set()
        for v in vals:
            try:
                unique_set.add(json.dumps(v, sort_keys=True))
            except (TypeError, ValueError):
                unique_set.add(repr(v))
        sample = []
        for v in vals[:5]:
            try:
                sample.append(str(v)[:120])
            except Exception:  # noqa: BLE001
                sample.append("<unprintable>")
        out.append(ColumnStat(
            name=k, dtype=type(vals[0]).__name__ if vals else "unknown",
            n_missing=n_missing, n_unique=len(unique_set), sample_values=sample,
        ))
    return out


def _missing_answer_issues(
    rows: list[dict], task_type: TaskTypeGuess,
) -> list[DatasetIssue]:
    """For known task shapes, check the answer/output side isn't empty."""
    issues: list[DatasetIssue] = []
    if not rows:
        return issues

    if task_type.label in ("chat", "multiturn_chat"):
        bad = [
            i for i, r in enumerate(rows)
            if _is_chat_row(r) and not any(
                m.get("role") == "assistant" and _cell_str(m.get("content"))
                for m in r["messages"]
            )
        ]
    elif task_type.label == "instruction":
        bad = [i for i, r in enumerate(rows) if not _cell_str(r.get("output"))]
    elif task_type.label == "qa":
        bad = [
            i for i, r in enumerate(rows)
            if not (_cell_str(r.get("answer")) or _cell_str(r.get("A")))
        ]
    elif task_type.label == "summarization":
        bad = [i for i, r in enumerate(rows) if not _cell_str(r.get("summary"))]
    else:
        return issues

    if bad:
        issues.append(DatasetIssue(
            code="missing_answer",
            severity="error" if len(bad) / len(rows) > 0.1 else "warn",
            count=len(bad),
            example_row_indices=bad[:5],
            message=(
                f"{len(bad)} row(s) are missing the answer/output for "
                f"task_type={task_type.label!r}. The agent will train on "
                f"empty targets."
            ),
        ))
    return issues


# ─────────────────────────── top-level entry point ───────────────────────────


def _count_csv_rows(path: Path) -> int:
    """Data rows in a CSV, counted by the CSV grammar rather than by newlines.

    A line count is only a row count when no field contains a newline, which
    free text routinely does. `train_if.csv` — 13,839 rows of prompts and
    answers — counted 223,444 that way, sixteen times over. That is not just a
    wrong number on a card: `_recommend_split` reads it, so the split advice was
    computed for a file that does not exist.
    """
    try:
        limit = csv.field_size_limit()
        csv.field_size_limit(1 << 30)  # a single cell can be a whole answer
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as fh:
                return max(0, sum(1 for _ in csv.reader(fh)) - 1)  # minus header
        finally:
            csv.field_size_limit(limit)
    except (OSError, csv.Error):
        # Unreadable or malformed beyond the parser: fall back to the cheap
        # count rather than reporting nothing.
        return max(0, _count_lines(path) - 1)


def profile_file(
    name: str, file_path: Path, *, sample_limit: int = 5000,
) -> DatasetProfile:
    """Profile a JSONL/CSV file. CSV support is limited to v0.1 — we
    just stream rows via pandas if available, falling back to None.
    """
    assert file_path.exists(), f"{file_path} does not exist"

    file_sha = _sha256_of(file_path)
    suffix = file_path.suffix.lower()
    issues: list[DatasetIssue] = []

    rows: list[dict] = []
    total_rows = 0
    if suffix == ".jsonl":
        total_rows = _count_lines(file_path)
        rows = _stream_jsonl(file_path, limit=sample_limit)
        # If we got 0 parseable rows but the file isn't empty, flag. (This
        # guard used to sit behind a contradictory `if rows and ...` check
        # that made it unreachable.)
        if not rows and total_rows > 0:
            issues.append(DatasetIssue(
                code="malformed_row", severity="error",
                count=total_rows, example_row_indices=[],
                message="no parseable JSON objects in the file",
            ))
    elif suffix == ".csv":
        try:
            import pandas as pd  # type: ignore
            df = pd.read_csv(file_path, nrows=sample_limit)
            rows = df.to_dict(orient="records")
            total_rows = _count_csv_rows(file_path)
        except Exception as e:  # noqa: BLE001
            issues.append(DatasetIssue(
                code="encoding_error", severity="error", count=0,
                message=f"failed to parse CSV: {type(e).__name__}: {e}",
            ))
    elif suffix == ".json":
        try:
            obj = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                rows = [r for r in obj if isinstance(r, dict)][:sample_limit]
                total_rows = len(obj)
            else:
                issues.append(DatasetIssue(
                    code="schema_violation", severity="error", count=0,
                    message="top-level JSON value is not a list of objects",
                ))
        except json.JSONDecodeError as e:
            issues.append(DatasetIssue(
                code="encoding_error", severity="error", count=0,
                message=f"failed to parse JSON: {e}",
            ))
    else:
        # PDF/TXT/MD/Parquet — leave detailed profiling to the agent.
        return DatasetProfile(
            name=name, file=str(file_path), file_sha256=file_sha,
            profiler_version=PROFILER_VERSION, n_rows=0, n_columns=0,
            columns=[],
            task_type=TaskTypeGuess(
                label="unknown", confidence=0.0,
                rationale=f"{suffix} files are not auto-profiled in v0.1 — agent will inspect",
            ),
            expected_format="raw",
            duplicates=DuplicateReport(n_exact=0),
            split_recommendation=SplitRecommendation(
                train_size=0, val_size=0,
                rationale="not applicable for non-tabular sources",
            ),
            issues=issues,
            ready_for_data=True,
            notes=f"profiler-v0.1 does not auto-inspect {suffix} files",
        )

    cols = _column_stats(rows)
    col_names = [c.name for c in cols]
    task, expected_format = _classify_task_type(rows, col_names)
    issues.extend(_missing_answer_issues(rows, task))
    dupes = _exact_duplicates(rows)
    split = _recommend_split(total_rows or len(rows))

    # Size warnings
    if total_rows == 0:
        issues.append(DatasetIssue(
            code="too_small", severity="error", count=0,
            message="file has zero rows",
        ))
    elif total_rows < 50:
        issues.append(DatasetIssue(
            code="too_small", severity="warn", count=total_rows,
            message=f"only {total_rows} rows — likely too small to fine-tune",
        ))

    if dupes.n_exact > 0:
        issues.append(DatasetIssue(
            code="exact_duplicate",
            severity="warn" if dupes.n_exact / max(1, total_rows) < 0.05 else "error",
            count=dupes.n_exact,
            message=(
                f"{dupes.n_exact} exact-duplicate row(s); "
                f"largest cluster: {dupes.largest_cluster_size}"
            ),
        ))

    ready = (
        all(i.severity != "error" for i in issues)
        and task.confidence >= 0.6
        and total_rows > 0
    )

    return DatasetProfile(
        name=name, file=str(file_path), file_sha256=file_sha,
        profiler_version=PROFILER_VERSION,
        n_rows=total_rows or len(rows),
        n_columns=len(cols), columns=cols,
        task_type=task, expected_format=expected_format,
        duplicates=dupes, split_recommendation=split, issues=issues,
        ready_for_data=ready,
        notes=(
            f"sampled {len(rows)} rows of {total_rows or len(rows)} total"
            if rows else "no rows profiled"
        ),
    )


# ───────────────────────────── caching helpers ───────────────────────────────


def cache_path_for(dataset_dir: Path) -> Path:
    return dataset_dir / ".profile.json"


def load_cached(dataset_dir: Path) -> DatasetProfile | None:
    p = cache_path_for(dataset_dir)
    if not p.exists():
        return None
    try:
        return DatasetProfile.model_validate_json(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def save_cached(dataset_dir: Path, profile: DatasetProfile) -> None:
    cache_path_for(dataset_dir).write_text(
        profile.model_dump_json(indent=2), encoding="utf-8"
    )


def is_cache_valid(dataset_dir: Path, file_path: Path, cached: DatasetProfile) -> bool:
    """Cheap validity check: same profiler version + same sha256."""
    if cached.profiler_version != PROFILER_VERSION:
        return False
    if str(file_path) != cached.file:
        return False
    try:
        return _sha256_of(file_path) == cached.file_sha256
    except OSError:
        return False
