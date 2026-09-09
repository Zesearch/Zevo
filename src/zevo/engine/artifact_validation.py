"""Deterministic validation for Data and Inference tabular artifacts.

Specialists receive exact command templates backed by this module.  Keeping
the checks here prevents an LLM-authored shell snippet from inventing a
working-directory-relative input path or a stricter data-quality rule than the
experiment contract actually requires.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from zevo.contracts.data import InferenceDataProfile


@dataclass(frozen=True)
class DataArtifactReport:
    training_rows: int
    validation_rows: int
    question_rows: int


@dataclass(frozen=True)
class PredictionArtifactReport:
    rows: int
    columns: tuple[str, ...]
    prediction_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class SystemScoringArtifacts:
    """Validation artifacts materialized by the engine, never by Data."""

    questions_path: str
    profile_path: str
    validation_dataset_path: str
    sample_submission_path: str


@dataclass(frozen=True)
class SanitizedTrainingArtifact:
    path: str
    removed_scoring_duplicates: int


def _absolute_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")
    return path


def _normalise_cell(value: Any) -> Any:
    # pandas represents an empty parquet cell as NaN.  Treat it like the empty
    # value it encodes, without treating an ordinary empty string as invalid.
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): _normalise_cell(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_cell(item) for item in value]
    return value


def _read_records(value: str | Path, *, label: str) -> tuple[list[str], list[dict[str, Any]]]:
    path = _absolute_file(value, label=label)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    elif suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{label} has invalid JSON on physical line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{label} line {line_number} must be a JSON object"
                    )
                rows.append(row)
        columns = _ordered_union(rows)
    elif suffix == ".json":
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} has invalid JSON: {exc}") from exc
        if not isinstance(loaded, list) or not all(
            isinstance(row, dict) for row in loaded
        ):
            raise ValueError(f"{label} must be a JSON array of objects")
        rows = list(loaded)
        columns = _ordered_union(rows)
    elif suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ValueError("parquet validation requires pandas/pyarrow") from exc
        frame = pd.read_parquet(path)
        columns = [str(column) for column in frame.columns]
        rows = [
            {str(key): _normalise_cell(item) for key, item in row.items()}
            for row in frame.to_dict(orient="records")
        ]
    else:
        raise ValueError(
            f"{label} has unsupported tabular suffix {suffix!r}; "
            "expected .csv, .json, .jsonl, .ndjson, or .parquet"
        )
    if not columns:
        raise ValueError(f"{label} has no columns")
    if not rows:
        raise ValueError(f"{label} has no records")
    return columns, rows


def _ordered_union(rows: Sequence[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _write_records(
    path: Path, *, columns: Sequence[str], rows: Sequence[dict[str, Any]],
) -> None:
    """Write records without changing their order or nested JSON values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
        return
    if suffix in {".jsonl", ".ndjson"}:
        path.write_text(
            "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return
    if suffix == ".json":
        path.write_text(
            json.dumps(list(rows), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        return
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ValueError("parquet materialization requires pandas/pyarrow") from exc
        pd.DataFrame(list(rows), columns=list(columns)).to_parquet(path, index=False)
        return
    raise ValueError(
        f"questions-only output has unsupported tabular suffix {suffix!r}"
    )


def _field_type(rows: Sequence[dict[str, Any]], column: str) -> str:
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "string"
    return "null"


def _task_shape(columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> str:
    lowered = {column.lower() for column in columns}
    if "messages" in lowered or "conversation" in lowered or "conversations" in lowered:
        return "multi-turn conversation"
    if "instruction" in lowered or "prompt" in lowered or "question" in lowered:
        return "instruction response"
    if rows and any(isinstance(value, (dict, list)) for value in rows[0].values()):
        return "structured input"
    return "tabular input"


def materialize_system_scoring_artifacts(
    *,
    scoring_source: str,
    answer_fields: Sequence[str],
    sample_submission: str,
    out_dir: str,
) -> SystemScoringArtifacts:
    """Create the answer-free Validation view after Data has finished.

    This engine-owned transform is deliberately independent of training-data
    selection. Data never receives the source path, answers, submission shape,
    or the resulting profile.
    """
    source = _absolute_file(scoring_source, label="Validation scoring source")
    columns, rows = _read_records(source, label="Validation scoring source")
    answers = list(answer_fields)
    if not answers or any(not field for field in answers):
        raise ValueError("Validation answer_fields must contain non-empty names")
    missing = [field for field in answers if field not in columns]
    if missing:
        raise ValueError(f"Validation answer fields are absent: {missing}")
    question_columns = [column for column in columns if column not in answers]
    if not question_columns:
        raise ValueError("Validation has no input fields after answer removal")
    question_rows = [
        {column: row.get(column) for column in question_columns}
        for row in rows
    ]

    suffix = source.suffix.lower()
    if suffix not in {".csv", ".json", ".jsonl", ".ndjson", ".parquet", ".pq"}:
        suffix = ".jsonl"
    questions = Path(out_dir) / f"validation_questions{suffix}"
    _write_records(questions, columns=question_columns, rows=question_rows)

    submission_columns = _csv_columns(
        sample_submission, label="Validation sample submission",
    )
    id_names = {"id", "row_id", "example_id", "index"}
    semantic_names = {
        "messages", "conversation", "conversations", "instruction", "prompt",
        "question", "input", "text", "context",
    }
    semantic_inputs = [
        column for column in question_columns if column.lower() in semantic_names
    ]
    input_fields = semantic_inputs or [
        column for column in question_columns if column.lower() not in id_names
    ] or list(question_columns)
    id_column = next(
        (column for column in question_columns if column.lower() in id_names), "",
    )
    stable_ids = bool(id_column) and all(
        row.get(id_column) not in (None, "") for row in question_rows
    ) and len({str(row.get(id_column)) for row in question_rows}) == len(question_rows)
    prediction_columns = [
        column for column in submission_columns if column not in question_columns
    ]
    profile = InferenceDataProfile(
        n_rows=len(question_rows),
        task_shape=_task_shape(question_columns, question_rows),
        record_fields={
            column: _field_type(question_rows, column) for column in question_columns
        },
        input_fields=input_fields,
        answer_fields_removed=answers,
        submission_columns=submission_columns,
        prediction_encoding=(
            "CSV prediction columns: " + ", ".join(prediction_columns)
            if prediction_columns
            else "CSV columns defined by the sample submission"
        ),
        stable_ids_present=stable_ids,
    )
    profile_path = Path(out_dir) / "inference_data_profile.json"
    profile_path.write_text(profile.model_dump_json(indent=2) + "\n", encoding="utf-8")

    _validate_questions_only(
        scoring_source=str(source),
        questions=str(questions),
        answer_fields=answers,
    )
    return SystemScoringArtifacts(
        questions_path=str(questions.resolve()),
        profile_path=str(profile_path.resolve()),
        validation_dataset_path=str(source.resolve()),
        sample_submission_path=str(
            _absolute_file(sample_submission, label="Validation sample submission").resolve()
        ),
    )


def validate_training_data_artifact(training_dataset: str) -> int:
    """Validate only the artifact Data is authorized to create."""
    _, rows = _read_records(training_dataset, label="training dataset")
    return len(rows)


_SEMANTIC_METADATA_KEYS = {
    "id", "row_id", "example_id", "index", "source", "split", "category",
    "num_turns", "role",
}


def _semantic_text(value: Any, *, key: str = "") -> list[str]:
    if key.lower() in _SEMANTIC_METADATA_KEYS:
        return []
    if isinstance(value, dict):
        parts: list[str] = []
        for child_key, child in value.items():
            parts.extend(_semantic_text(child, key=str(child_key)))
        return parts
    if isinstance(value, list):
        parts = []
        for child in value:
            parts.extend(_semantic_text(child))
        return parts
    if isinstance(value, str) and value.strip():
        return [" ".join(value.split()).casefold()]
    return []


def _semantic_fingerprint(record: dict[str, Any]) -> str:
    parts: list[str]
    messages = record.get("messages")
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if isinstance(message, dict):
                parts.extend(_semantic_text(message.get("content")))
            else:
                parts.extend(_semantic_text(message))
    elif isinstance(record.get("instruction"), dict) and isinstance(
        record.get("response"), dict,
    ):
        # Capybara-style multi-turn records store all user turns and all
        # assistant turns in separate keyed objects, while canonical SFT rows
        # interleave them as messages. Compare their semantic turn order.
        instructions = record["instruction"]
        responses = record["response"]
        parts = []
        ordered_turns = list(instructions)
        ordered_turns.extend(key for key in responses if key not in instructions)
        for turn in ordered_turns:
            parts.extend(_semantic_text(instructions.get(turn)))
            parts.extend(_semantic_text(responses.get(turn)))
    else:
        parts = _semantic_text(record)
    text = "\n".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def semantic_record_fingerprints(source: str) -> set[str]:
    """Return opaque exact semantic identities for one tabular population."""
    _, rows = _read_records(source, label="scoring source")
    return {
        fingerprint for row in rows
        if (fingerprint := _semantic_fingerprint(row))
    }


def sanitize_training_against_scoring(
    *,
    training_dataset: str,
    validation_source: str,
    blocked_fingerprints: Sequence[str] = (),
    out_dir: str,
) -> SanitizedTrainingArtifact:
    """Remove exact Validation/Test duplicates without exposing rows to Data."""
    training_path = _absolute_file(training_dataset, label="training dataset")
    training_columns, training_rows = _read_records(
        training_path, label="training dataset",
    )
    blocked = semantic_record_fingerprints(validation_source)
    blocked.update(str(value) for value in blocked_fingerprints if value)
    retained = [
        row for row in training_rows
        if _semantic_fingerprint(row) not in blocked
    ]
    removed = len(training_rows) - len(retained)
    if removed == 0:
        return SanitizedTrainingArtifact(str(training_path.resolve()), 0)
    if not retained:
        raise ValueError(
            "all selected Training rows duplicate a hidden scoring population"
        )
    suffix = training_path.suffix.lower()
    output = Path(out_dir) / f"dataset.validation_blind{suffix}"
    _write_records(output, columns=training_columns, rows=retained)
    validate_training_data_artifact(str(output))
    return SanitizedTrainingArtifact(str(output.resolve()), removed)


def _validate_questions_only(
    *, scoring_source: str, questions: str, answer_fields: Sequence[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    source_columns, source_rows = _read_records(
        scoring_source, label="scoring source",
    )
    question_columns, question_rows = _read_records(
        questions, label="questions-only data",
    )
    answers = list(answer_fields)
    if not answers or any(not field for field in answers):
        raise ValueError("answer_fields must contain one or more non-empty names")
    missing = [field for field in answers if field not in source_columns]
    if missing:
        raise ValueError(f"answer fields are absent from scoring source: {missing}")
    expected_columns = [column for column in source_columns if column not in answers]
    if question_columns != expected_columns:
        raise ValueError(
            "questions-only columns/order differ after exact answer removal: "
            f"expected {expected_columns}, got {question_columns}"
        )
    if len(question_rows) != len(source_rows):
        raise ValueError(
            "questions-only row count differs from scoring source: "
            f"expected {len(source_rows)}, got {len(question_rows)}"
        )
    for index, (source, question) in enumerate(zip(source_rows, question_rows)):
        expected = {
            key: _normalise_cell(value)
            for key, value in source.items()
            if key not in answers
        }
        actual = {key: _normalise_cell(value) for key, value in question.items()}
        if actual != expected:
            raise ValueError(
                "questions-only data changed a non-answer value or row order "
                f"at zero-based row {index}"
            )
    return question_columns, question_rows


def _csv_columns(value: str | Path, *, label: str) -> list[str]:
    path = _absolute_file(value, label=label)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        columns = list(next(csv.reader(handle), []))
    if not columns:
        raise ValueError(f"{label} has no CSV header")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{label} contains duplicate columns")
    return columns


def validate_data_artifacts(
    *,
    operation: str,
    scoring_source: str,
    questions: str,
    sample_submission: str,
    answer_fields: Sequence[str],
    training_dataset: str = "",
    validation_dataset: str = "",
    profile_path: str = "",
) -> DataArtifactReport:
    """Validate universal Data invariants without inventing quality policy.

    Empty strings inside source-faithful training records are permitted.  This
    validator checks parseability, population identity, exact answer removal,
    and the closed Inference profile; method-specific semantics remain owned by
    the selected Data Skill and Train contract.
    """
    if operation not in {"prepare_run_data", "prepare_holdout_data"}:
        raise ValueError(f"unsupported Data operation: {operation!r}")
    question_columns, question_rows = _validate_questions_only(
        scoring_source=scoring_source,
        questions=questions,
        answer_fields=answer_fields,
    )
    submission_columns = _csv_columns(
        sample_submission, label="sample submission",
    )
    if operation == "prepare_holdout_data":
        if training_dataset or validation_dataset or profile_path:
            raise ValueError(
                "held-out Data validation must not receive trainable/profile artifacts"
            )
        return DataArtifactReport(0, 0, len(question_rows))

    _, training_rows = _read_records(
        training_dataset, label="training dataset",
    )
    _, validation_rows = _read_records(
        validation_dataset, label="validation dataset",
    )
    if len(validation_rows) != len(question_rows):
        raise ValueError(
            "trainer Validation row count differs from the full scoring "
            f"population: expected {len(question_rows)}, got {len(validation_rows)}"
        )
    profile_file = _absolute_file(profile_path, label="inference data profile")
    try:
        profile = InferenceDataProfile.model_validate_json(
            profile_file.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"inference data profile is invalid: {exc}") from exc
    if profile.n_rows != len(question_rows):
        raise ValueError(
            "profile n_rows differs from questions-only data: "
            f"expected {len(question_rows)}, got {profile.n_rows}"
        )
    if list(profile.record_fields) != question_columns:
        raise ValueError(
            "profile record_fields differ from questions-only columns/order: "
            f"expected {question_columns}, got {list(profile.record_fields)}"
        )
    if profile.answer_fields_removed != list(answer_fields):
        raise ValueError(
            "profile answer_fields_removed differs from the work order: "
            f"expected {list(answer_fields)}, got {profile.answer_fields_removed}"
        )
    if profile.submission_columns != submission_columns:
        raise ValueError(
            "profile submission_columns differ from sample submission: "
            f"expected {submission_columns}, got {profile.submission_columns}"
        )

    # The canonical normalized outputs retain `id` when the source provides it.
    # Check disjointness only when both artifacts explicitly carry that field;
    # never guess a task-specific identity key.
    if all("id" in row for row in training_rows + validation_rows):
        training_ids = {str(row["id"]) for row in training_rows}
        validation_ids = {str(row["id"]) for row in validation_rows}
        overlap = training_ids & validation_ids
        if overlap:
            raise ValueError(
                "training and validation datasets overlap on id; first overlap: "
                f"{sorted(overlap)[0]!r}"
            )
    return DataArtifactReport(
        len(training_rows), len(validation_rows), len(question_rows),
    )


def validate_prediction_artifacts(
    *, predictions: str, questions: str, sample_submission: str,
) -> PredictionArtifactReport:
    """Validate measured-output shape using only assigned absolute inputs."""
    question_columns, question_rows = _read_records(
        questions, label="questions-only data",
    )
    expected_columns = _csv_columns(
        sample_submission, label="sample submission",
    )
    prediction_path = _absolute_file(predictions, label="predictions")
    with prediction_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_columns = list(reader.fieldnames or [])
        prediction_rows = [dict(row) for row in reader]
    if actual_columns != expected_columns:
        raise ValueError(
            "prediction columns/order differ from sample submission: "
            f"expected {expected_columns}, got {actual_columns}"
        )
    if not prediction_rows:
        raise ValueError("predictions has no records")
    if len(prediction_rows) != len(question_rows):
        raise ValueError(
            "prediction row count differs from questions-only data: "
            f"expected {len(question_rows)}, got {len(prediction_rows)}"
        )

    # Any output column also present in the questions is an engine-observable
    # identity/order column.  Compare all such columns instead of guessing that
    # a particular task calls its stable identifier `id`.
    shared_columns = [
        column for column in expected_columns if column in question_columns
    ]
    for index, (question, prediction) in enumerate(
        zip(question_rows, prediction_rows)
    ):
        for column in shared_columns:
            expected = "" if question.get(column) is None else str(question.get(column))
            if prediction.get(column, "") != expected:
                raise ValueError(
                    f"prediction {column!r} differs from questions/order at "
                    f"zero-based row {index}"
                )
    value_columns = tuple(
        column for column in actual_columns if column not in shared_columns
    )
    if not value_columns:
        raise ValueError(
            "sample submission has no prediction column outside question identity fields"
        )
    return PredictionArtifactReport(
        len(prediction_rows), tuple(actual_columns), value_columns,
    )


def validate_scoring_prediction_artifacts(
    *,
    predictions: str,
    scoring_set: str,
    sample_submission: str,
    answer_fields: Sequence[str],
) -> PredictionArtifactReport:
    """Re-check the sample-submission boundary immediately before scoring.

    Inference already validates against the questions-only view. Evaluation
    repeats the invariant against the full scoring population so neither a
    stale artifact binding nor a custom evaluator can silently accept shifted
    columns, row counts, or record order.
    """
    scoring_columns, scoring_rows = _read_records(
        scoring_set, label="scoring set",
    )
    expected_columns = _csv_columns(
        sample_submission, label="sample submission",
    )
    prediction_path = _absolute_file(predictions, label="predictions")
    with prediction_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_columns = list(reader.fieldnames or [])
        prediction_rows = [dict(row) for row in reader]
    if actual_columns != expected_columns:
        raise ValueError(
            "prediction columns/order differ from sample submission: "
            f"expected {expected_columns}, got {actual_columns}"
        )
    if len(prediction_rows) != len(scoring_rows):
        raise ValueError(
            "prediction row count differs from scoring set: "
            f"expected {len(scoring_rows)}, got {len(prediction_rows)}"
        )

    answers = set(answer_fields)
    identity_columns = [
        column for column in expected_columns
        if column in scoring_columns and column not in answers
    ]
    prediction_columns = [
        column for column in expected_columns if column not in identity_columns
    ]
    if not prediction_columns:
        raise ValueError(
            "sample submission has no prediction column outside scoring identity fields"
        )
    for index, (gold, prediction) in enumerate(zip(scoring_rows, prediction_rows)):
        for column in identity_columns:
            expected = "" if gold.get(column) is None else str(gold.get(column))
            if prediction.get(column, "") != expected:
                raise ValueError(
                    f"prediction {column!r} differs from scoring set/order at "
                    f"zero-based row {index}"
                )
    return PredictionArtifactReport(
        len(prediction_rows), tuple(actual_columns), tuple(prediction_columns),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Zevo Data or Inference artifacts",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("validate-data")
    data.add_argument(
        "--operation",
        required=True,
        choices=["prepare_run_data", "prepare_holdout_data"],
    )
    data.add_argument("--scoring-source", required=True)
    data.add_argument("--questions", required=True)
    data.add_argument("--sample-submission", required=True)
    data.add_argument("--answer-field", action="append", required=True)
    data.add_argument("--training-dataset", default="")
    data.add_argument("--validation-dataset", default="")
    data.add_argument("--profile", default="")

    training = subparsers.add_parser("validate-training-data")
    training.add_argument("--training-dataset", required=True)

    inference = subparsers.add_parser("validate-predictions")
    inference.add_argument("--predictions", required=True)
    inference.add_argument("--questions", required=True)
    inference.add_argument("--sample-submission", required=True)
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-data":
            report = validate_data_artifacts(
                operation=args.operation,
                scoring_source=args.scoring_source,
                questions=args.questions,
                sample_submission=args.sample_submission,
                answer_fields=args.answer_field,
                training_dataset=args.training_dataset,
                validation_dataset=args.validation_dataset,
                profile_path=args.profile,
            )
            print(
                "VALID DataArtifacts "
                f"training_rows={report.training_rows} "
                f"validation_rows={report.validation_rows} "
                f"question_rows={report.question_rows}"
            )
        elif args.command == "validate-training-data":
            rows = validate_training_data_artifact(args.training_dataset)
            print(f"VALID TrainingData rows={rows}")
        else:
            report = validate_prediction_artifacts(
                predictions=args.predictions,
                questions=args.questions,
                sample_submission=args.sample_submission,
            )
            print(
                "VALID PredictionArtifacts "
                f"rows={report.rows} columns={list(report.columns)}"
            )
    except (OSError, ValueError) as exc:
        print(f"INVALID {args.command}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(_main())
