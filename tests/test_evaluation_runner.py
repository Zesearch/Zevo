from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

from zevo.contracts.evaluation import EvaluationTaskInput
from zevo.engine.agent.drivers.evaluation_runner import EvaluationRunnerDriver
from zevo.evaluator_storage import file_sha256


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(inp: EvaluationTaskInput, work_dir: Path):
    events: list[dict] = []
    result = asyncio.run(EvaluationRunnerDriver().run_agent(
        blueprint=None,
        input_payload=inp,
        workspace_dir=str(work_dir),
        event_sink=events.append,
    ))
    return result, events


def test_custom_evaluator_runs_through_bash_without_llm(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    gold = tmp_path / "gold.csv"
    _write_csv(predictions, [{"id": "1", "prediction": "A"}])
    _write_csv(gold, [{"id": "1", "answer": "A"}])
    evaluator = tmp_path / "eval.py"
    evaluator.write_text(
        "import json, sys\n"
        "json.dump({'score': 0.75, 'token_f1': 0.75}, open(sys.argv[3], 'w'))\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "work"
    result, events = _run(EvaluationTaskInput(
        ticket_id="eval-test-001",
        predictions_path=str(predictions),
        scoring_set=str(gold),
        sample_submission=str(predictions),
        evaluation_script=str(evaluator),
        evaluator_sha256=file_sha256(evaluator),
        answer_fields=["answer"],
        metric="token_f1",
    ), work_dir)

    assert result.driver == "evaluation_runner"
    assert result.model == ""
    assert result.output.status == "succeeded"
    assert json.loads(Path(result.output.metrics_path).read_text())["score"] == 0.75
    assert (work_dir / "evaluate.sh").read_text(encoding="utf-8").startswith(
        "#!/usr/bin/env bash\nset -euo pipefail\nexec python3 "
    )
    assert json.loads((work_dir / "metrics.json").read_text())["score"] == 0.75
    assert not any(e.get("type") == "turn_completed" for e in events)


def test_builtin_evaluator_is_deterministic_and_writes_metrics(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    gold = tmp_path / "gold.csv"
    _write_csv(predictions, [
        {"id": "1", "prediction": "A"},
        {"id": "2", "prediction": "B"},
    ])
    _write_csv(gold, [
        {"id": "1", "answer": "A"},
        {"id": "2", "answer": "C"},
    ])
    result, _ = _run(EvaluationTaskInput(
        ticket_id="eval-test-002",
        predictions_path=str(predictions),
        scoring_set=str(gold),
        sample_submission=str(predictions),
        evaluation_script="",
        answer_fields=["answer"],
        metric="accuracy",
        evaluation_config={"prediction_column": "prediction"},
    ), tmp_path / "work")

    assert result.output.status == "succeeded"
    assert json.loads(Path(result.output.metrics_path).read_text())["score"] == 0.5


def test_custom_evaluator_must_write_finite_numeric_score(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    gold = tmp_path / "gold.csv"
    _write_csv(predictions, [{"prediction": "A"}])
    _write_csv(gold, [{"answer": "A"}])
    evaluator = tmp_path / "eval.py"
    evaluator.write_text(
        "import json, sys\njson.dump({'accuracy': 1.0}, open(sys.argv[3], 'w'))\n",
        encoding="utf-8",
    )
    result, _ = _run(EvaluationTaskInput(
        ticket_id="eval-test-003",
        predictions_path=str(predictions),
        scoring_set=str(gold),
        sample_submission=str(predictions),
        evaluation_script=str(evaluator),
        evaluator_sha256=file_sha256(evaluator),
        answer_fields=["answer"],
        metric="accuracy",
    ), tmp_path / "work")

    assert result.output.status == "failed"
    assert "numeric top-level `score`" in result.output.error_message


def test_custom_evaluator_rejects_bytes_changed_after_freezing(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    gold = tmp_path / "gold.csv"
    _write_csv(predictions, [{"prediction": "A"}])
    _write_csv(gold, [{"answer": "A"}])
    evaluator = tmp_path / "eval.py"
    evaluator.write_text(
        "import json, sys\njson.dump({'score': 1.0}, open(sys.argv[3], 'w'))\n",
        encoding="utf-8",
    )
    frozen_digest = file_sha256(evaluator)
    evaluator.write_text(
        "import json, sys\njson.dump({'score': 0.0}, open(sys.argv[3], 'w'))\n",
        encoding="utf-8",
    )

    result, _ = _run(EvaluationTaskInput(
        ticket_id="eval-test-004",
        predictions_path=str(predictions),
        scoring_set=str(gold),
        sample_submission=str(predictions),
        evaluation_script=str(evaluator),
        evaluator_sha256=frozen_digest,
        answer_fields=["answer"],
        metric="accuracy",
    ), tmp_path / "work")

    assert result.output.status == "failed"
    assert "evaluator_sha256" in result.output.error_message


def test_custom_evaluator_never_runs_when_predictions_shift_from_sample(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.csv"
    gold = tmp_path / "gold.csv"
    sample = tmp_path / "sample.csv"
    marker = tmp_path / "evaluator-ran"
    _write_csv(predictions, [{"id": "1", "prediction": "A"}])
    _write_csv(gold, [{"id": "1", "answer": "A"}])
    _write_csv(sample, [{"id": "", "response": ""}])
    evaluator = tmp_path / "eval.py"
    evaluator.write_text(
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(marker)!r}).write_text('ran')\n"
        "json.dump({'score': 1.0}, open(sys.argv[3], 'w'))\n",
        encoding="utf-8",
    )

    result, _ = _run(EvaluationTaskInput(
        ticket_id="eval-shift-001",
        predictions_path=str(predictions),
        scoring_set=str(gold),
        sample_submission=str(sample),
        evaluation_script=str(evaluator),
        evaluator_sha256=file_sha256(evaluator),
        answer_fields=["answer"],
        metric="benchmark_average",
    ), tmp_path / "work-shift")

    assert result.output.status == "failed"
    assert "columns/order differ from sample submission" in result.output.error_message
    assert not marker.exists(), "schema validation must happen before custom code"


def test_builtin_prediction_column_is_derived_from_sample_submission(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.csv"
    gold = tmp_path / "gold.csv"
    sample = tmp_path / "sample.csv"
    _write_csv(predictions, [{"id": "1", "response": "red fox"}])
    _write_csv(gold, [{"id": "1", "answer": "the red fox"}])
    _write_csv(sample, [{"id": "", "response": ""}])

    result, _ = _run(EvaluationTaskInput(
        ticket_id="eval-sample-binding-001",
        predictions_path=str(predictions),
        scoring_set=str(gold),
        sample_submission=str(sample),
        evaluation_script="",
        answer_fields=["answer"],
        metric="token_f1",
    ), tmp_path / "work-sample-binding")

    assert result.output.status == "succeeded"
    metrics = json.loads(Path(result.output.metrics_path).read_text())
    assert metrics["score"] == 1.0
