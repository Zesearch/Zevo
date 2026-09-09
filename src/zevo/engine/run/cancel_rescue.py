"""Graceful cancel: rescue the champion checkpoint before the GPU goes away.

`POST /runs/{id}/cancel` kills the run's work and records a
`CancelWeightsPolicy` on the Run; the run stays `running` so no reconciler
pass reaps its box. The daemon's `complete_requested_cancels` pass then, in
one background task per run:

  1. picks the checkpoint worth keeping: the validation champion's final
     checkpoint, else the newest succeeded train checkpoint;
  2. copies it off the box into `<work_dir_root>/<run>/models/<M-tag>/` (or
     the operator's folder) and mirrors it into `registry_models`, so it shows
     up in Saved models like a registered champion would;
  3. pushes it to the Hugging Face Hub when asked;
  4. flips the Run to `cancelled`. `_cleanup_terminal_resources` tears the box
     down on the next tick, exactly as for any other terminal run.

Every step converges on the same end state, so a daemon restart mid-rescue
resumes rather than duplicates: a finished copy leaves a marker file, the
registry upsert is keyed by the run's one tag, and the Hub upload is a commit
against the same repo.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.contracts.cancel import CancelWeightsPolicy
from zevo.contracts.model_registry import model_tag_for_run
from zevo.db.models import RegistryModel, Run, Ticket, WorkProduct
from zevo.engine.remote_transfer import build_download_command
from zevo.engine.run.remote_jobs import _device_info_for_ticket

log = logging.getLogger(__name__)

RESCUE_MARKER = ".zevo-rescued"


@dataclass
class RescueCandidate:
    ticket: Ticket
    path: str
    remote: bool
    meta: dict[str, Any]


def _champion_iteration(run: Run) -> int | None:
    lower_is_better = (run.validation_metric_direction or "max") == "min"
    best: tuple[float, int] | None = None
    for entry in run.history or []:
        if entry.get("source") == "baseline":
            continue
        score = entry.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        iteration = int(entry.get("iteration", -1))
        if best is None or (score < best[0] if lower_is_better else score > best[0]):
            best = (float(score), iteration)
    return None if best is None else best[1]


def _validation_score(run: Run, iteration: int) -> float | None:
    for entry in run.history or []:
        if int(entry.get("iteration", -1)) == iteration and entry.get("source") != "baseline":
            score = entry.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                return float(score)
    return None


async def select_rescue_checkpoint(session: AsyncSession, run: Run) -> RescueCandidate | None:
    rows = (await session.execute(
        select(WorkProduct, Ticket)
        .join(Ticket, WorkProduct.ticket_id == Ticket.id)
        .where(
            Ticket.run_id == run.id,
            Ticket.agent_id == "train",
            Ticket.status == "succeeded",
            WorkProduct.role == "checkpoint",
        )
        .order_by(Ticket.created_at.desc(), WorkProduct.created_at)
    )).all()
    if not rows:
        return None

    def pick(candidates: list) -> RescueCandidate:
        final = [r for r in candidates if (r[0].meta or {}).get("checkpoint_kind", "final") == "final"]
        wp, ticket = (final or candidates)[0]
        meta = dict(wp.meta or {})
        return RescueCandidate(ticket=ticket, path=wp.path, remote=meta.get("location") == "remote", meta=meta)

    champion = _champion_iteration(run)
    if champion is not None:
        on_champion = [r for r in rows if int(r[1].iteration or 0) == champion]
        if on_champion:
            return pick(on_champion)
    return pick(rows)


def default_rescue_dir(run: Run) -> Path:
    from zevo.api.config import settings
    return Path(settings.work_dir_root) / run.id / "models" / model_tag_for_run(run.id)


def prepare_rescue_dir(run: Run, policy: CancelWeightsPolicy) -> Path:
    """Create the folder the checkpoint will land in, and prove it is writable.

    Called from the cancel request so a bad folder is a 400 before anything
    is killed, not an error discovered after the trainer is gone.
    """
    dest = Path(policy.local_dir) if policy.local_dir else default_rescue_dir(run)
    dest.mkdir(parents=True, exist_ok=True)
    probe = dest / ".zevo-write-probe"
    probe.write_text("", encoding="utf-8")
    probe.unlink()
    return dest


def _run_transfer(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"scp exited {completed.returncode}: {(completed.stderr or '').strip()[:400]}"
        )


async def _copy_off_box(session: AsyncSession, candidate: RescueCandidate, dest: Path) -> Path:
    """Copy the checkpoint folder into `dest/<name>/`; returns that folder."""
    model_dir = dest / PurePosixPath(candidate.path).name
    if (model_dir / RESCUE_MARKER).exists():
        return model_dir
    dest.mkdir(parents=True, exist_ok=True)
    if candidate.remote:
        info = await _device_info_for_ticket(session, candidate.ticket)
        if info is None:
            raise RuntimeError(
                f"no device_info route for train ticket {candidate.ticket.id}; "
                "cannot reach the box the checkpoint is on"
            )
        command = build_download_command(
            info.ssh, remote_paths=[candidate.path], local_dir=str(dest), recursive=True,
        )
        await asyncio.to_thread(_run_transfer, command)
    else:
        source = Path(candidate.path)
        if not source.exists():
            raise RuntimeError(f"checkpoint path no longer exists: {source}")
        if source.resolve() != model_dir.resolve():
            await asyncio.to_thread(shutil.copytree, source, model_dir, dirs_exist_ok=True)
    if not model_dir.exists():
        raise RuntimeError(f"copy finished but {model_dir} is missing")
    (model_dir / RESCUE_MARKER).write_text(
        datetime.now(timezone.utc).isoformat(), encoding="utf-8",
    )
    return model_dir


async def _mirror_registry(
    session: AsyncSession, run: Run, candidate: RescueCandidate, model_dir: Path,
) -> str:
    tag = model_tag_for_run(run.id)
    row = await session.get(RegistryModel, tag)
    if row is None:
        row = RegistryModel(version_tag=tag, run_id=run.id)
        session.add(row)
    iteration = int(candidate.ticket.iteration or 0)
    row.iteration = iteration
    row.base_model = str(candidate.meta.get("base_model") or "")
    row.training_method = str(candidate.meta.get("training_method") or "")
    row.model_path = str(model_dir)
    row.task_objective = run.task_objective or ""
    row.metric = run.validation_metric or run.metric
    row.metric_direction = run.validation_metric_direction or run.metric_direction
    row.eval = {
        "validation_score": _validation_score(run, iteration),
        "rescued_on_cancel": True,
        "source_ticket_id": candidate.ticket.id,
    }
    run.registry_version_tag = tag
    return tag


def hf_token() -> str:
    """The Settings page writes HF_TOKEN to .env; a value saved after boot is
    only there, so read the file first and fall back to the process env."""
    from zevo.api.routers.ui.settings import ENV_PATH, _read_env
    return _read_env(ENV_PATH).get("HF_TOKEN", "") or os.environ.get("HF_TOKEN", "")


def push_to_hf(model_dir: Path, *, repo_id: str, private: bool, token: str) -> str:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir), repo_id=repo_id,
        commit_message="Zevo: checkpoint rescued on run cancel",
        ignore_patterns=[RESCUE_MARKER],
    )
    return f"https://huggingface.co/{repo_id}"


async def _registered_model_dir(session: AsyncSession, run: Run) -> Path | None:
    """The registry stage already pulled this run's champion off the box."""
    if not (run.registry_version_tag or "").strip():
        return None
    row = await session.get(RegistryModel, run.registry_version_tag)
    if row is None or not row.model_path:
        return None
    path = Path(row.model_path)
    return path if path.exists() else None


async def perform_rescue(session: AsyncSession, run: Run) -> dict[str, Any]:
    """Carry out the run's cancel policy. Returns the outcome to record."""
    policy = CancelWeightsPolicy.model_validate(run.cancel_policy or {})
    outcome: dict[str, Any] = {"weights": policy.weights}
    if policy.weights == "discard":
        return outcome

    model_dir = await _registered_model_dir(session, run)
    if model_dir is not None:
        outcome["note"] = "champion was already registered; nothing to copy"
    else:
        candidate = await select_rescue_checkpoint(session, run)
        if candidate is None:
            outcome["note"] = "no succeeded train checkpoint to keep"
            return outcome
        dest = Path(policy.local_dir) if policy.local_dir else default_rescue_dir(run)
        model_dir = await _copy_off_box(session, candidate, dest)
        outcome["registry_version_tag"] = await _mirror_registry(session, run, candidate, model_dir)
        outcome["source_ticket_id"] = candidate.ticket.id
        await session.commit()
    outcome["model_path"] = str(model_dir)

    if policy.weights == "hf":
        token = hf_token()
        if not token:
            raise RuntimeError("HF_TOKEN is not set in Settings")
        outcome["hf_url"] = await asyncio.to_thread(
            push_to_hf, model_dir,
            repo_id=policy.hf_repo_id, private=policy.hf_private, token=token,
        )
    return outcome


_ACTIVE: dict[str, asyncio.Task[None]] = {}


async def complete_requested_cancels(session: AsyncSession) -> int:
    """Start a rescue task for every run whose cancel is waiting on one.

    A run already being worked on is skipped, so the 5s reconcile tick cannot
    start a second copy of a transfer that takes minutes. Returns how many
    tasks this pass started.
    """
    runs = (await session.execute(
        select(Run).where(
            Run.cancel_requested_at.is_not(None),
            Run.status.in_(["planning", "running"]),
        )
    )).scalars().all()
    started = 0
    for run in runs:
        task = _ACTIVE.get(run.id)
        if task is not None and not task.done():
            continue
        _ACTIVE[run.id] = asyncio.create_task(_finish_cancel(run.id))
        started += 1
    return started


async def _finish_cancel(run_id: str) -> None:
    from zevo.db import get_session_factory
    Session = get_session_factory()
    try:
        async with Session() as session:
            run = await session.get(Run, run_id)
            if run is None:
                return
            try:
                outcome = await perform_rescue(session, run)
            except Exception as exc:  # the run must still close; the error is the outcome
                log.exception("[cancel-rescue] run %s: rescue failed", run_id[:8])
                await session.rollback()
                run = await session.get(Run, run_id)
                outcome = {"weights": (run.cancel_policy or {}).get("weights", ""),
                           "error": str(exc)[:500]}
            now = datetime.now(timezone.utc)
            outcome["finished_at"] = now.isoformat()
            run.cancel_outcome = outcome
            run.status = "cancelled"
            run.halted_reason = "cancelled by user; " + _describe(outcome)
            run.finished_at = now
            await session.commit()
            log.info("[cancel-rescue] run %s closed: %s", run_id[:8], _describe(outcome))
    finally:
        _ACTIVE.pop(run_id, None)


def _describe(outcome: dict[str, Any]) -> str:
    if outcome.get("error"):
        return f"weights rescue failed: {outcome['error']}"
    if outcome.get("hf_url"):
        return f"weights pushed to {outcome['hf_url']}"
    if outcome.get("model_path"):
        return f"weights kept at {outcome['model_path']}"
    if outcome.get("note"):
        return outcome["note"]
    return "weights discarded"
