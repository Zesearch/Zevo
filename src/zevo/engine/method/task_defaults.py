"""Resolve a low-input dataset Task into an explicit scoring contract.

The dashboard's ordinary path asks for a name, objective and Test set.  These
helpers inspect a bounded part of a local table, infer conventional answer/id
fields, create the mechanical sample-submission CSV, and choose conservative
built-in defaults.  Ambiguous schemas fail with one actionable request for the
Advanced form instead of silently grading the wrong column.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from zevo.contracts.task_protocol import (
    TaskInferenceProtocol,
    default_task_inference_protocol,
)
from zevo.holdout_storage import resolve_asset
from zevo.paths import uploads_root


_ANSWER_NAMES = (
    "answer", "answers", "gold", "gold_answer", "ground_truth", "label",
    "target", "reference", "expected", "solution",
)
_ID_NAMES = ("id", "row_id", "example_id", "index")


@dataclass(frozen=True)
class ResolvedDatasetTaskDefaults:
    answer_fields: list[str]
    sample_submission: str
    metric: str
    metric_direction: str
    inference_protocol: TaskInferenceProtocol


def _first_record(path: Path) -> tuple[list[str], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            row = next(reader, None)
        return columns, dict(row or {})
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Test JSONL records must be objects")
                return list(value), value
        return [], {}
    if suffix == ".json":
        # ijson keeps a large array bounded to its first object.  Some small
        # fixtures are a single object rather than an array; handle that shape
        # without turning it into a one-column string representation.
        with path.open("r", encoding="utf-8") as handle:
            first = handle.read(256).lstrip()[:1]
        if first == "[":
            import ijson

            with path.open("rb") as handle:
                value = next(ijson.items(handle, "item"), None)
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
        if value is None:
            return [], {}
        if not isinstance(value, dict):
            raise ValueError("Test JSON records must be objects")
        return list(value), value
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        columns = list(parquet.schema_arrow.names)
        if not parquet.num_row_groups:
            return columns, {}
        rows = parquet.read_row_group(0).slice(0, 1).to_pylist()
        return columns, dict(rows[0]) if rows else {}
    raise ValueError(
        "automatic Task setup supports CSV, JSON, JSONL and Parquet Test sets"
    )


def _answer_fields(columns: Sequence[str]) -> list[str]:
    by_lower = {column.lower(): column for column in columns}
    for wanted in _ANSWER_NAMES:
        if wanted in by_lower:
            return [by_lower[wanted]]
    suffix_matches = [
        column for column in columns
        if column.lower().endswith(("_answer", "_label", "_target"))
    ]
    if len(suffix_matches) == 1:
        return suffix_matches
    raise ValueError(
        "Zevo could not identify the Test answer column safely. Open Advanced "
        "and select the answer field; conventional names such as answer, gold, "
        "label, target, reference, or solution are detected automatically."
    )


def _identity_columns(columns: Sequence[str], answers: Sequence[str]) -> list[str]:
    answer_set = set(answers)
    by_lower = {column.lower(): column for column in columns if column not in answer_set}
    return [by_lower[name] for name in _ID_NAMES if name in by_lower][:1]


def _write_sample_submission(
    *, task_name: str, test_set: str, columns: Sequence[str], first: dict[str, Any],
    answers: Sequence[str],
) -> str:
    identity = _identity_columns(columns, answers)
    digest = hashlib.sha256(
        json.dumps(
            [task_name, test_set, list(columns), list(answers)],
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:16]
    directory = Path(uploads_root()) / "_derived" / digest
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sample_submission.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*identity, "prediction"])
        writer.writeheader()
        writer.writerow({
            **{field: "" if first.get(field) is None else first.get(field) for field in identity},
            "prediction": "",
        })
    return str(path)


def _default_metric(protocol: TaskInferenceProtocol, answers: Sequence[str]) -> str:
    if protocol.task_type in {"classification", "multiple_choice"}:
        return "accuracy"
    if protocol.task_type == "generation" and any(
        name.lower() in {"reference", "references"} for name in answers
    ):
        return "token_f1"
    return "exact_match"


def resolve_dataset_task_defaults(
    *, task_name: str, objective: str, test_set: str,
    answer_fields: Sequence[str] = (), sample_submission: str = "",
    metric: str = "", metric_direction: str = "",
    inference_protocol: TaskInferenceProtocol | None = None,
) -> ResolvedDatasetTaskDefaults:
    """Fill omitted ordinary-Task fields without asking the user for a form."""
    test_value = test_set.strip()
    if not test_value:
        raise ValueError("Test set is required")
    resolved = Path(resolve_asset(test_value))
    needs_inspection = not answer_fields or not sample_submission.strip()
    columns: list[str] = []
    first: dict[str, Any] = {}
    if needs_inspection:
        if not resolved.is_file():
            raise ValueError(
                "Zevo cannot inspect this remote or unavailable Test set during "
                "automatic setup. Open Advanced and provide its answer field and "
                "sample submission, or choose a local file from Files."
            )
        columns, first = _first_record(resolved)
        if not columns or not first:
            raise ValueError("Test set must contain a header and at least one record")

    answers = [str(field).strip() for field in answer_fields if str(field).strip()]
    if not answers:
        answers = _answer_fields(columns)
    if columns:
        missing = [field for field in answers if field not in columns]
        if missing:
            raise ValueError(f"Test answer fields are absent from the Test set: {missing}")

    protocol = inference_protocol or default_task_inference_protocol(objective, metric)
    chosen_metric = metric.strip() or _default_metric(protocol, answers)
    sample = sample_submission.strip() or _write_sample_submission(
        task_name=task_name,
        test_set=test_value,
        columns=columns,
        first=first,
        answers=answers,
    )
    return ResolvedDatasetTaskDefaults(
        answer_fields=answers,
        sample_submission=sample,
        metric=chosen_metric,
        metric_direction=metric_direction.strip() or "max",
        inference_protocol=protocol,
    )
