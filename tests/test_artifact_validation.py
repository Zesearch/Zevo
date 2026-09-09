from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from zevo.engine.artifact_validation import (
    materialize_system_scoring_artifacts,
    sanitize_training_against_scoring,
    validate_data_artifacts,
    validate_prediction_artifacts,
)


def _json(path: Path, rows: list[dict]) -> str:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return str(path)


def _jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    return str(path)


def _csv(path: Path, columns: list[str], rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def test_data_validation_allows_source_faithful_empty_training_content(
    tmp_path: Path,
) -> None:
    training = _jsonl(tmp_path / "dataset.jsonl", [
        {
            "id": "train-1",
            "messages": [
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": ""},
            ],
        }
    ])
    validation = _jsonl(tmp_path / "validation_dataset.jsonl", [
        {
            "id": "val-1",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        }
    ])
    source = _json(tmp_path / "validation.json", [
        {"id": "val-1", "instruction": {"1": "question"}, "response": {"1": "answer"}}
    ])
    questions = _json(tmp_path / "validation_questions.json", [
        {"id": "val-1", "instruction": {"1": "question"}}
    ])
    submission = _csv(
        tmp_path / "sample_submission.csv",
        ["id", "prediction"],
        [{"id": "val-1", "prediction": ""}],
    )
    profile = tmp_path / "inference_data_profile.json"
    profile.write_text(json.dumps({
        "schema_version": 1,
        "source": "validation_questions_only",
        "n_rows": 1,
        "task_shape": "single turn",
        "record_fields": {"id": "string", "instruction": "object"},
        "input_fields": ["instruction"],
        "answer_fields_removed": ["response"],
        "submission_format": "csv",
        "submission_columns": ["id", "prediction"],
        "prediction_encoding": "string",
        "row_order_preserved": True,
        "stable_ids_present": True,
        "contains_answer_values": False,
        "contains_evaluation_logic": False,
    }), encoding="utf-8")

    report = validate_data_artifacts(
        operation="prepare_run_data",
        scoring_source=source,
        questions=questions,
        sample_submission=submission,
        answer_fields=["response"],
        training_dataset=training,
        validation_dataset=validation,
        profile_path=str(profile),
    )

    assert report.training_rows == 1
    assert report.validation_rows == 1
    assert report.question_rows == 1


def test_data_validation_rejects_changed_non_answer_value(tmp_path: Path) -> None:
    source = _json(tmp_path / "validation.json", [
        {"id": "val-1", "question": "original", "answer": "A"}
    ])
    questions = _json(tmp_path / "questions.json", [
        {"id": "val-1", "question": "changed"}
    ])
    submission = _csv(
        tmp_path / "sample.csv", ["id", "prediction"],
        [{"id": "val-1", "prediction": ""}],
    )

    with pytest.raises(ValueError, match="changed a non-answer value"):
        validate_data_artifacts(
            operation="prepare_holdout_data",
            scoring_source=source,
            questions=questions,
            sample_submission=submission,
            answer_fields=["answer"],
        )


def test_engine_materializes_validation_without_data_agent_logic(tmp_path: Path) -> None:
    source = _json(tmp_path / "validation.json", [
        {
            "id": "val-1",
            "instruction": {"1": "question"},
            "response": {"1": "answer"},
        },
        {
            "id": "val-2",
            "instruction": {"1": "another"},
            "response": {"1": "second"},
        },
    ])
    submission = _csv(
        tmp_path / "sample.csv", ["id", "prediction"],
        [{"id": "example", "prediction": ""}],
    )

    prepared = materialize_system_scoring_artifacts(
        scoring_source=source,
        answer_fields=["response"],
        sample_submission=submission,
        out_dir=str(tmp_path / "system"),
    )

    questions = json.loads(Path(prepared.questions_path).read_text())
    assert questions == [
        {"id": "val-1", "instruction": {"1": "question"}},
        {"id": "val-2", "instruction": {"1": "another"}},
    ]
    profile = json.loads(Path(prepared.profile_path).read_text())
    assert profile["answer_fields_removed"] == ["response"]
    assert profile["input_fields"] == ["instruction"]
    assert profile["contains_answer_values"] is False
    assert prepared.validation_dataset_path == str(Path(source).resolve())


def test_engine_removes_cross_schema_validation_duplicates_after_data(
    tmp_path: Path,
) -> None:
    training = _jsonl(tmp_path / "dataset.jsonl", [
        {"messages": [
            {"role": "user", "content": "same question"},
            {"role": "assistant", "content": "same answer"},
        ]},
        {"messages": [
            {"role": "user", "content": "training only"},
            {"role": "assistant", "content": "kept"},
        ]},
    ])
    validation = _json(tmp_path / "validation.json", [
        {
            "id": "v1",
            "instruction": {"1": "same question"},
            "response": {"1": "same answer"},
            "num_turns": 1,
        },
    ])

    sanitized = sanitize_training_against_scoring(
        training_dataset=training,
        validation_source=validation,
        out_dir=str(tmp_path / "system"),
    )

    assert sanitized.removed_scoring_duplicates == 1
    rows = [
        json.loads(line) for line in Path(sanitized.path).read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["messages"][0]["content"] == "training only"


def test_prediction_validation_uses_exact_paths_and_template_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    elsewhere = tmp_path / "elsewhere"
    inputs.mkdir()
    outputs.mkdir()
    elsewhere.mkdir()
    questions = _json(inputs / "questions.json", [
        {"id": "q-1", "prompt": "one"},
        {"id": "q-2", "prompt": "two"},
    ])
    submission = _csv(
        inputs / "sample.csv", ["id", "prediction"],
        [{"id": "example", "prediction": ""}],
    )
    predictions = _csv(
        outputs / "predictions.csv", ["id", "prediction"],
        [
            {"id": "q-1", "prediction": "A"},
            {"id": "q-2", "prediction": ""},
        ],
    )
    monkeypatch.chdir(elsewhere)

    report = validate_prediction_artifacts(
        predictions=predictions,
        questions=questions,
        sample_submission=submission,
    )

    assert report.rows == 2
    assert report.columns == ("id", "prediction")
