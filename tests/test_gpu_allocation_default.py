"""A Run keeps its GPUs for the whole run unless the launcher says otherwise.

Smoke run a676d8ac under the previous per_stage default re-provisioned a cloud
GPU three times in one iteration, re-prepared its data on the new host, and
took 157 minutes against 78 for the same task on one retained card.
"""
from __future__ import annotations

from zevo.api.routers.shared.runs import CreateRunRequest
from zevo.api.routers.ui.preflight import PreflightBody
from zevo.engine.run.gpu_lifecycle import needs_stage_release
from types import SimpleNamespace


def _auto(**over) -> dict:
    body = dict(mode="auto", task_name="t", run_name="r", gpu_provider="cloud",
                user_request={"task_objective": "o"})
    body.update(over)
    return body


def test_launch_defaults_to_one_allocation_for_the_whole_run() -> None:
    assert CreateRunRequest.model_validate(_auto()).gpu_allocation_mode == "per_run"
    assert PreflightBody.model_validate(
        {"mode": "auto", "user_request": {"task_objective": "o"}}
    ).gpu_allocation_mode == "per_run"


def test_per_stage_remains_an_explicit_choice() -> None:
    body = CreateRunRequest.model_validate(_auto(gpu_allocation_mode="per_stage"))
    assert body.gpu_allocation_mode == "per_stage"


def test_the_default_mode_never_releases_between_stages() -> None:
    run = SimpleNamespace(gpu_allocation_mode="per_run", gpu_provider="cloud")
    for agent in ("train", "inference"):
        for lane in ("optimization", "held_out_test"):
            ticket = SimpleNamespace(status="succeeded", agent_id=agent, lane=lane)
            assert needs_stage_release(run, ticket) is False

