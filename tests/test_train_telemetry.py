from __future__ import annotations

import importlib.util
import json
import re
from types import SimpleNamespace

from zevo.engine.agent.loader import REPO_ROOT
from zevo.engine.observe.live_markers import LiveMarkerReader
from zevo.engine.observe.markers import parse_line
from zevo.engine.run.runner import _training_diagnostics_from_events


def _load_helper():
    path = REPO_ROOT / "playbook" / "runners" / "train_telemetry.py"
    spec = importlib.util.spec_from_file_location("zevo_test_train_telemetry", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_callback_forwards_all_numeric_trainer_logs(capsys) -> None:
    module = _load_helper()
    assert module.TELEMETRY_INTERVAL_STEPS == 20
    callback = module.ZevoTrainerTelemetryCallback("train-run-001")
    state = SimpleNamespace(
        is_world_process_zero=True,
        global_step=20,
        max_steps=200,
    )
    control = object()
    assert callback.on_log(
        None,
        state,
        control,
        logs={
            "loss": 1.25,
            "learning_rate": 2e-5,
            "grad_norm": 0.75,
            "epoch": 0.5,
            "label": "ignored",
            "overflow": float("inf"),
            "flag": True,
        },
    ) is control
    line = capsys.readouterr().out.strip()
    lines = line.splitlines()
    assert len(lines) == 2
    attempt = parse_line(lines[0])
    assert attempt is not None and attempt[0] == "attempt"
    attempt_id = attempt[1]["attempt_id"]
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        attempt_id,
    )
    parsed = parse_line(lines[1])
    assert parsed is not None
    kind, payload = parsed
    assert kind == "progress"
    assert payload["owner"] == "train-run-001"
    assert payload["attempt_id"] == attempt_id
    assert payload["step"] == 20 and payload["total"] == 200
    assert payload["loss"] == 1.25
    assert payload["learning_rate"] == 2e-5
    assert payload["grad_norm"] == 0.75
    assert payload["epoch"] == 0.5
    assert "label" not in payload and "overflow" not in payload and "flag" not in payload


def test_callback_preserves_evaluation_only_rows(capsys) -> None:
    module = _load_helper()
    callback = module.ZevoTrainerTelemetryCallback("train-run-001")
    callback.on_log(
        None,
        SimpleNamespace(is_world_process_zero=True, global_step=40, max_steps=200),
        object(),
        logs={"eval_loss": 0.8, "eval_token_f1": 0.6},
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert parse_line(lines[0])[0] == "attempt"  # type: ignore[index]
    body = json.loads(lines[1].split(":", 2)[2])
    assert body["eval_loss"] == 0.8
    assert body["eval_token_f1"] == 0.6


def test_callback_announces_one_attempt_and_reuses_it(capsys) -> None:
    module = _load_helper()
    callback = module.ZevoTrainerTelemetryCallback("train-run-001")
    state = SimpleNamespace(is_world_process_zero=True, global_step=0, max_steps=200)
    control = object()

    assert callback.on_train_begin(None, state, control) is control
    state.global_step = 20
    callback.on_log(None, state, control, logs={"loss": 1.0})
    state.global_step = 40
    callback.on_log(None, state, control, logs={"loss": 0.9})

    parsed = [parse_line(line) for line in capsys.readouterr().out.splitlines()]
    assert [item[0] for item in parsed if item is not None] == [
        "attempt", "progress", "progress",
    ]
    attempt_id = parsed[0][1]["attempt_id"]  # type: ignore[index]
    assert parsed[1][1]["attempt_id"] == attempt_id  # type: ignore[index]
    assert parsed[2][1]["attempt_id"] == attempt_id  # type: ignore[index]


def test_each_callback_instance_has_a_fresh_attempt_id() -> None:
    module = _load_helper()
    first = module.ZevoTrainerTelemetryCallback("train-run-001")
    second = module.ZevoTrainerTelemetryCallback("train-run-001")
    assert first.attempt_id != second.attempt_id


def test_runner_summarizes_only_the_newest_training_attempt() -> None:
    events = [
        {"event_type": "progress", "attempt_id": "failed", "current_step": 20,
         "total_steps": 100, "loss": 9.0, "extras": {}, "ts": "2026-01-01T00:00:01"},
        {"event_type": "progress", "attempt_id": "good", "current_step": 20,
         "total_steps": 100, "loss": 1.2, "extras": {}, "ts": "2026-01-01T00:01:01"},
        {"event_type": "progress", "attempt_id": "good", "current_step": 20,
         "total_steps": 100, "loss": 0.9, "extras": {"eval_loss": 0.9},
         "ts": "2026-01-01T00:01:02"},
        {"event_type": "progress", "attempt_id": "good", "current_step": 40,
         "total_steps": 100, "loss": 0.8, "extras": {}, "ts": "2026-01-01T00:01:03"},
        {"event_type": "progress", "attempt_id": "good", "current_step": 40,
         "total_steps": 100, "loss": 1.0, "extras": {"eval_loss": 1.0},
         "ts": "2026-01-01T00:01:04"},
        # Collect replays the same remote marker under a later heartbeat/phase.
        # It remains one point in the attempt, not a second curve segment.
        {"event_type": "progress", "attempt_id": "good", "current_step": 40,
         "total_steps": 100, "loss": 0.8, "extras": {"loss": 0.8},
         "phase": "complete", "ts": "2026-01-01T00:01:03"},
        # Trainer's terminal `train_loss` is an aggregate, not the step-40 loss.
        {"event_type": "progress", "attempt_id": "good", "current_step": 40,
         "total_steps": 100, "loss": 0.7,
         "extras": {"train_loss": 0.7, "train_runtime": 10.0},
         "ts": "2026-01-01T00:01:05"},
    ]
    result = _training_diagnostics_from_events(events)
    assert result.attempt_id == "good"
    assert result.training_loss.first == 1.2
    assert result.training_loss.final == 0.8
    assert result.training_loss.points == 2
    assert result.validation_loss.minimum == 0.9
    assert "validation_loss_rose_after_minimum" in result.observations


def test_live_reader_keeps_restarted_processes_separate(tmp_path) -> None:
    first = "550e8400-e29b-41d4-a716-446655440000"
    second = "550e8400-e29b-41d4-a716-446655440001"
    (tmp_path / "train.log").write_text(
        "\n".join([
            f"__ATTEMPT__:train-run-001:{first}@1723723199.0",
            "__PHASE__:train-run-001:training@1723723200.0",
            f'__PROGRESS__:train-run-001:{{"attempt_id":"{first}","step":20,"loss":1.2}}',
            f"__ATTEMPT__:train-run-001:{second}@1723723299.0",
            "__PHASE__:train-run-001:training@1723723300.0",
            f'__PROGRESS__:train-run-001:{{"attempt_id":"{second}","step":20,"loss":0.9}}',
        ]) + "\n"
    )
    events = LiveMarkerReader(tmp_path, "train-run-001").poll(finish=True)
    progress = [payload for kind, payload in events if kind == "progress"]
    phases = [payload for kind, payload in events if kind == "phase"]
    assert [payload["attempt_id"] for payload in progress] == [first, second]
    assert [payload["attempt_id"] for payload in phases] == [first, second]
