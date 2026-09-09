"""Graceful cancel keeps the champion checkpoint before the GPU goes away.

Live failure this covers: run f834a204 was cancelled after five iterations on a
rented H100 and every LoRA adapter went with the box, because cancel destroyed
the instance milliseconds after killing the trainer and the registry stage had
never run. Now cancel records a weights policy, the run stays open while the
daemon copies the champion checkpoint off the box (or pushes it to the Hub),
and only then is the run closed and the box reaped.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import zevo.db as zevo_db
import zevo.engine.cost.budget as budget_mod
import zevo.engine.run.cancel_rescue as rescue
import zevo.engine.run.remote_jobs as remote_jobs
from zevo.api.routers.shared.runs import cancel_run
from zevo.contracts.cancel import CancelWeightsPolicy
from zevo.db.models import Base, RegistryModel, Run, Ticket, WorkProduct
from zevo.engine.run.scheduler.reconciler import (
    _close_finished_runs,
    _watchdog_halt_over_budget_runs,
)


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


_RUN = dict(
    task_name="t", task_objective="o", agent_objective="ao",
    metric="accuracy", metric_direction="max",
    validation_metric="accuracy", validation_metric_direction="max",
    status="running",
)


def _checkpoint_dir(tmp_path: Path, name: str) -> Path:
    d = tmp_path / "remote" / name
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"weights")
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    return d


async def _seed(db, tmp_path: Path, *, history, checkpoints, **run_over) -> Run:
    """One run with a succeeded train ticket + final checkpoint per iteration."""
    run = Run(id="r1", history=history, **{**_RUN, **run_over})
    db.add(run)
    db.add(Ticket(id="orch-1", run_id="r1", agent_id="orchestrator", status="cancelled",
                  payload={}, summary="", error_message=""))
    for iteration, path in checkpoints:
        tid = f"train-{iteration}"
        db.add(Ticket(id=tid, run_id="r1", agent_id="train", status="succeeded",
                      iteration=iteration, payload={}, summary="", error_message=""))
        db.add(WorkProduct(ticket_id=tid, role="checkpoint", path=str(path),
                           meta={"checkpoint_kind": "final", "base_model": "base/m",
                                 "training_method": "lora_sft"}))
    await db.commit()
    return run


def _hist(*scores: float) -> list[dict]:
    return [{"iteration": i + 1, "score": s, "source": "evaluation"} for i, s in enumerate(scores)]


def test_policy_contract_rejects_bad_inputs() -> None:
    with pytest.raises(ValidationError):
        CancelWeightsPolicy(weights="hf")
    with pytest.raises(ValidationError):
        CancelWeightsPolicy(weights="hf", hf_repo_id="no-slash")
    with pytest.raises(ValidationError):
        CancelWeightsPolicy(local_dir="relative/dir")
    assert CancelWeightsPolicy().weights == "download"
    assert CancelWeightsPolicy(weights="hf", hf_repo_id="me/model").hf_private is True


@pytest.mark.asyncio
async def test_champion_checkpoint_is_selected_over_latest(tmp_path) -> None:
    Session = await _session()
    async with Session() as db:
        run = await _seed(
            db, tmp_path, history=_hist(0.52, 0.47),
            checkpoints=[(1, _checkpoint_dir(tmp_path, "it1")), (2, _checkpoint_dir(tmp_path, "it2"))],
        )
        cand = await rescue.select_rescue_checkpoint(db, run)
        assert cand is not None and cand.ticket.id == "train-1" and cand.path.endswith("it1")


@pytest.mark.asyncio
async def test_latest_checkpoint_when_no_scored_iteration(tmp_path) -> None:
    Session = await _session()
    async with Session() as db:
        run = await _seed(
            db, tmp_path, history=[],
            checkpoints=[(1, _checkpoint_dir(tmp_path, "it1")), (2, _checkpoint_dir(tmp_path, "it2"))],
        )
        cand = await rescue.select_rescue_checkpoint(db, run)
        assert cand is not None and cand.ticket.id == "train-2"


@pytest.mark.asyncio
async def test_download_copies_registers_and_is_idempotent(tmp_path) -> None:
    Session = await _session()
    dest = tmp_path / "keep"
    async with Session() as db:
        run = await _seed(
            db, tmp_path, history=_hist(0.5),
            checkpoints=[(1, _checkpoint_dir(tmp_path, "it1"))],
            cancel_policy=CancelWeightsPolicy(local_dir=str(dest)).model_dump(),
        )
        outcome = await rescue.perform_rescue(db, run)
        model_dir = dest / "it1"
        assert outcome["model_path"] == str(model_dir)
        assert (model_dir / "adapter_model.safetensors").read_bytes() == b"weights"
        assert (model_dir / rescue.RESCUE_MARKER).exists()

        row = await db.get(RegistryModel, outcome["registry_version_tag"])
        assert row is not None and row.model_path == str(model_dir)
        assert row.base_model == "base/m" and row.training_method == "lora_sft"
        assert row.eval["validation_score"] == 0.5 and row.eval["rescued_on_cancel"] is True
        assert run.registry_version_tag == row.version_tag

        marker_before = (model_dir / rescue.RESCUE_MARKER).read_text()
        again = await rescue.perform_rescue(db, run)
        assert again["model_path"] == str(model_dir)
        assert (model_dir / rescue.RESCUE_MARKER).read_text() == marker_before


@pytest.mark.asyncio
async def test_hf_pushes_from_the_local_copy(tmp_path, monkeypatch) -> None:
    Session = await _session()
    pushed: list[tuple] = []
    monkeypatch.setattr(rescue, "hf_token", lambda: "hf_x")

    def fake_push(model_dir, *, repo_id, private, token):
        pushed.append((Path(model_dir).name, repo_id, private, token))
        return f"https://huggingface.co/{repo_id}"
    monkeypatch.setattr(rescue, "push_to_hf", fake_push)

    async with Session() as db:
        run = await _seed(
            db, tmp_path, history=_hist(0.5),
            checkpoints=[(1, _checkpoint_dir(tmp_path, "it1"))],
            cancel_policy=CancelWeightsPolicy(
                weights="hf", hf_repo_id="me/legal-lora", hf_private=False,
                local_dir=str(tmp_path / "keep"),
            ).model_dump(),
        )
        outcome = await rescue.perform_rescue(db, run)
        assert outcome["hf_url"] == "https://huggingface.co/me/legal-lora"
        assert pushed == [("it1", "me/legal-lora", False, "hf_x")]


@pytest.mark.asyncio
async def test_discard_and_nothing_to_keep(tmp_path) -> None:
    Session = await _session()
    async with Session() as db:
        run = await _seed(db, tmp_path, history=[], checkpoints=[],
                          cancel_policy={"weights": "discard"})
        assert await rescue.perform_rescue(db, run) == {"weights": "discard"}
        run.cancel_policy = {"weights": "download", "local_dir": str(tmp_path / "k")}
        outcome = await rescue.perform_rescue(db, run)
        assert outcome["note"] == "no succeeded train checkpoint to keep"
        assert (await db.execute(select(RegistryModel))).scalars().all() == []


@pytest.mark.asyncio
async def test_reconciler_leaves_cancel_requested_runs_open(tmp_path, monkeypatch) -> None:
    Session = await _session()

    class Snap:
        over_budget = True
        over_time_limit = False
        spent_usd = 99.0
        elapsed_runtime_hours = 0.0

    async def snap(session, run_id):
        return Snap()
    monkeypatch.setattr(budget_mod, "snapshot_for_run", snap)

    async with Session() as db:
        await _seed(db, tmp_path, history=[], checkpoints=[],
                    max_cost_usd=10.0, cancel_requested_at=datetime.now(timezone.utc))
        assert await _watchdog_halt_over_budget_runs(db) == 0
        assert await _close_finished_runs(db) == {"success": 0, "degraded": 0, "failed": 0, "halted": 0}
        run = await db.get(Run, "r1")
        assert run.status == "running"


@pytest.mark.asyncio
async def test_rescue_task_closes_the_run(tmp_path, monkeypatch) -> None:
    Session = await _session()
    monkeypatch.setattr(zevo_db, "get_session_factory", lambda: Session)
    dest = tmp_path / "keep"
    async with Session() as db:
        await _seed(
            db, tmp_path, history=_hist(0.5),
            checkpoints=[(1, _checkpoint_dir(tmp_path, "it1"))],
            cancel_requested_at=datetime.now(timezone.utc),
            cancel_policy=CancelWeightsPolicy(local_dir=str(dest)).model_dump(),
        )
        assert await rescue.complete_requested_cancels(db) == 1
        assert await rescue.complete_requested_cancels(db) == 0, "one task per run"
        await rescue._ACTIVE["r1"]
    async with Session() as db:
        run = await db.get(Run, "r1")
        assert run.status == "cancelled" and run.finished_at is not None
        assert run.cancel_outcome["model_path"] == str(dest / "it1")
        assert run.halted_reason == f"cancelled by user; weights kept at {dest / 'it1'}"


@pytest.mark.asyncio
async def test_rescue_failure_still_closes_the_run(tmp_path, monkeypatch) -> None:
    Session = await _session()
    monkeypatch.setattr(zevo_db, "get_session_factory", lambda: Session)

    async def boom(session, run):
        raise RuntimeError("scp exited 255: connection refused")
    monkeypatch.setattr(rescue, "perform_rescue", boom)

    async with Session() as db:
        await _seed(db, tmp_path, history=[], checkpoints=[],
                    cancel_requested_at=datetime.now(timezone.utc),
                    cancel_policy={"weights": "download"})
        await rescue.complete_requested_cancels(db)
        await rescue._ACTIVE["r1"]
    async with Session() as db:
        run = await db.get(Run, "r1")
        assert run.status == "cancelled"
        assert run.cancel_outcome["error"].startswith("scp exited 255")


@pytest.mark.asyncio
async def test_cancel_api_records_policy_and_keeps_run_open(tmp_path, monkeypatch) -> None:
    Session = await _session()

    async def no_remote(db, tickets):
        return 0
    monkeypatch.setattr(remote_jobs, "cancel_run_remote_jobs", no_remote)

    async with Session() as db:
        await _seed(db, tmp_path, history=[], checkpoints=[])
        db.add(Ticket(id="train-live", run_id="r1", agent_id="train", status="running",
                      payload={}, summary="", error_message=""))
        await db.commit()

        out = await cancel_run("r1", CancelWeightsPolicy(local_dir=str(tmp_path / "keep")), db)
        assert out["status"] == "cancelling" and out["tickets_cancelled"] == ["train-live"]
        run = await db.get(Run, "r1")
        assert run.status == "running" and run.cancel_requested_at is not None
        assert run.cancel_policy["weights"] == "download"
        assert (await db.get(Ticket, "train-live")).status == "cancelled"

        with pytest.raises(HTTPException) as err:
            await cancel_run("r1", None, db)
        assert err.value.status_code == 409


@pytest.mark.asyncio
async def test_cancel_api_refuses_before_killing_anything(tmp_path, monkeypatch) -> None:
    Session = await _session()
    monkeypatch.setattr(rescue, "hf_token", lambda: "")
    async with Session() as db:
        await _seed(db, tmp_path, history=[], checkpoints=[])
        db.add(Ticket(id="train-live", run_id="r1", agent_id="train", status="running",
                      payload={}, summary="", error_message=""))
        await db.commit()
        with pytest.raises(HTTPException) as err:
            await cancel_run("r1", CancelWeightsPolicy(
                weights="hf", hf_repo_id="me/m", local_dir=str(tmp_path / "k")), db)
        assert err.value.status_code == 400 and "HF_TOKEN" in err.value.detail
        assert (await db.get(Ticket, "train-live")).status == "running"
        assert (await db.get(Run, "r1")).cancel_requested_at is None


@pytest.mark.asyncio
async def test_cancel_api_discard_closes_immediately(tmp_path, monkeypatch) -> None:
    Session = await _session()

    async def no_remote(db, tickets):
        return 0
    monkeypatch.setattr(remote_jobs, "cancel_run_remote_jobs", no_remote)
    async with Session() as db:
        await _seed(db, tmp_path, history=[], checkpoints=[])
        out = await cancel_run("r1", CancelWeightsPolicy(weights="discard"), db)
        assert out["status"] == "cancelled"
        run = await db.get(Run, "r1")
        assert run.status == "cancelled" and run.cancel_outcome["weights"] == "discard"
