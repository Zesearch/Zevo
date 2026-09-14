from __future__ import annotations

import os
import time
from uuid import uuid4

import pytest
from pydantic import ValidationError


def test_setup_progress_is_scoped_to_the_bound_launch(tmp_path, monkeypatch) -> None:
    from zevo.engine.run import setup_progress

    monkeypatch.setenv("ZEVO_RUN_SETUP_DIR", str(tmp_path))
    setup_progress._PROGRESS.clear()
    first = str(uuid4())
    second = str(uuid4())

    token = setup_progress.bind(first)
    try:
        setup_progress.update(
            phase="benchmarks",
            completed=3,
            total=17,
            label="MATH-500",
        )
        setup_progress.update(
            second,
            phase="checking",
            label="Another launch",
        )
    finally:
        setup_progress.reset(token)

    assert setup_progress.read(first) == {
        "status": "active",
        "phase": "benchmarks",
        "completed": 3,
        "total": 17,
        "label": "MATH-500",
    }
    assert setup_progress.read(second)["label"] == "Another launch"


@pytest.mark.asyncio
async def test_setup_progress_endpoint_waits_then_returns_real_state(tmp_path, monkeypatch) -> None:
    from zevo.api.routers.shared.runs import get_run_setup_progress
    from zevo.engine.run import setup_progress

    monkeypatch.setenv("ZEVO_RUN_SETUP_DIR", str(tmp_path))
    setup_progress._PROGRESS.clear()
    setup_id = uuid4()
    waiting = await get_run_setup_progress(setup_id)
    assert waiting.status == "waiting"

    setup_progress.update(
        str(setup_id),
        phase="validation",
        completed=5,
        total=17,
        label="MATH-500",
    )
    current = await get_run_setup_progress(setup_id)
    assert current.status == "active"
    assert current.phase == "validation"
    assert current.completed == 5
    assert current.total == 17
    assert current.label == "MATH-500"


def test_setup_progress_survives_another_worker_and_reports_a_dead_one(
    tmp_path, monkeypatch,
) -> None:
    from zevo.engine.run import setup_progress

    monkeypatch.setenv("ZEVO_RUN_SETUP_DIR", str(tmp_path))
    setup_id = str(uuid4())
    setup_progress.update(
        setup_id, phase="benchmarks", completed=16, total=17, label="LiveCodeBench v3",
    )
    setup_progress._PROGRESS.clear()  # simulate a different Uvicorn worker
    assert setup_progress.read(setup_id)["label"] == "LiveCodeBench v3"

    progress_path = tmp_path / f"{setup_id}.json"
    old = progress_path.stat().st_mtime - setup_progress._FAILED_AFTER_SECONDS - 1
    os.utime(progress_path, (old, old))
    failed = setup_progress.read(setup_id)
    assert failed["status"] == "failed"
    assert failed["phase"] == "failed"


@pytest.mark.asyncio
async def test_setup_heartbeat_survives_a_blocked_request_event_loop(
    tmp_path, monkeypatch,
) -> None:
    from zevo.engine.run import setup_progress

    monkeypatch.setenv("ZEVO_RUN_SETUP_DIR", str(tmp_path))
    setup_progress._PROGRESS.clear()
    setup_id = str(uuid4())
    setup_progress.update(
        setup_id, phase="validation", completed=2, total=7,
        label="Coding · CodeContests",
    )
    progress_path = tmp_path / f"{setup_id}.json"
    old = time.time() - setup_progress._FAILED_AFTER_SECONDS - 10
    os.utime(progress_path, (old, old))

    async with setup_progress.heartbeat(setup_id, interval=0.01):
        # Code-answer projection and filesystem caching can synchronously hold
        # this event loop. The liveness clock must still move while it is held.
        time.sleep(0.08)
        assert progress_path.stat().st_mtime > old


def test_create_run_request_accepts_only_a_uuid_setup_id() -> None:
    from zevo.api.routers.shared.runs import CreateRunRequest

    setup_id = uuid4()
    parsed = CreateRunRequest(task_name="task", run_name="run", setup_id=setup_id)
    assert parsed.setup_id == setup_id
    with pytest.raises(ValidationError):
        CreateRunRequest(task_name="task", run_name="run", setup_id="not-a-uuid")
