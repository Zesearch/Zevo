from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError


def test_setup_progress_is_scoped_to_the_bound_launch() -> None:
    from zevo.engine.run import setup_progress

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
async def test_setup_progress_endpoint_waits_then_returns_real_state() -> None:
    from zevo.api.routers.shared.runs import get_run_setup_progress
    from zevo.engine.run import setup_progress

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


def test_create_run_request_accepts_only_a_uuid_setup_id() -> None:
    from zevo.api.routers.shared.runs import CreateRunRequest

    setup_id = uuid4()
    parsed = CreateRunRequest(task_name="task", run_name="run", setup_id=setup_id)
    assert parsed.setup_id == setup_id
    with pytest.raises(ValidationError):
        CreateRunRequest(task_name="task", run_name="run", setup_id="not-a-uuid")
