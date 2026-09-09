from __future__ import annotations

from zevo.engine.observe.score_monitor import extract_score, materialize_metrics_artifact
from zevo.contracts.evaluation import EvaluationResult, EvaluationTaskInput
from zevo.engine.run.runner import _extract_summary_artifact_meta


def test_materialize_metrics_artifact_rejects_raw_json_as_a_path(tmp_path) -> None:
    path, metrics = materialize_metrics_artifact(
        '{"score": 72, "n_correct": 72, "n_total": 100}',
        str(tmp_path),
    )

    assert path == '{"score": 72, "n_correct": 72, "n_total": 100}'
    assert metrics == {}
    assert not (tmp_path / "metrics.json").exists()


def test_materialize_metrics_artifact_passes_existing_path_through(tmp_path) -> None:
    existing = tmp_path / "custom_metrics.json"
    existing.write_text('{"score": -12.5}', encoding="utf-8")

    path, metrics = materialize_metrics_artifact(str(existing), str(tmp_path))

    assert path == str(existing)
    assert metrics == {"score": -12.5}


def test_extract_score_prefers_metrics_then_valid_fallback() -> None:
    assert extract_score({"score": "12.75"}, fallback=0.1) == 12.75
    assert extract_score({"score": -4.0}, fallback=0.1) == -4.0
    assert extract_score({}, fallback=0.33) == 0.33
    assert extract_score({"score": True}, fallback=-1.0) == -1.0
    assert extract_score({"score": float("nan")}, fallback=-1.0) == -1.0
    assert extract_score({}, fallback=float("inf")) is None


def _input(ticket_id: str) -> EvaluationTaskInput:
    return EvaluationTaskInput(
        ticket_id=ticket_id,
        predictions_path="/tmp/predictions.csv",
        scoring_set="/tmp/scoring.csv",
        sample_submission="/tmp/sample_submission.csv",
        evaluation_script="",
        metric="accuracy",
    )


def test_evaluation_summary_uses_file_as_sole_score_authority(tmp_path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"score": -4.2}', encoding="utf-8")
    result = EvaluationResult(
        status="succeeded", ticket_id="eval-001", metrics_path=str(metrics),
        error_message="", notes="",
    )

    status, summary, artifact, meta = _extract_summary_artifact_meta(
        result, inp=_input("eval-001"), agent_id="evaluation", work_dir=str(tmp_path),
    )

    assert status == "succeeded"
    assert summary == "score=-4.2000"
    assert artifact == str(metrics)
    assert meta["score"] == -4.2


def test_evaluation_summary_rejects_missing_file_headline(tmp_path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"f1": 0.8}', encoding="utf-8")
    result = EvaluationResult(
        status="succeeded", ticket_id="eval-002", metrics_path=str(metrics),
        error_message="", notes="",
    )

    status, summary, _, _ = _extract_summary_artifact_meta(
        result, inp=_input("eval-002"), agent_id="evaluation", work_dir=str(tmp_path),
    )

    assert status == "failed"
    assert "no finite numeric `score`" in summary
