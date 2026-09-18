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

Failed preservation retains the source with an actionable failure; it never
silently authorizes destruction. Retries and individual subprocesses are bounded.
A daemon restart mid-rescue
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
import sys
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
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


async def _run_transfer(command: list[str]) -> None:
    """Cancellation kills and reaps the transfer, rather than orphaning a thread."""
    proc = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        _stdout, stderr = await proc.communicate()
        if proc.returncode:
            raise RuntimeError(f"transfer exited {proc.returncode}: {stderr.decode(errors='replace')[:400]}")
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


async def _copy_off_box(session: AsyncSession, candidate: RescueCandidate, dest: Path) -> Path:
    """Copy the checkpoint folder into `dest/<name>/`; returns that folder."""
    remote_path = PurePosixPath(candidate.path)
    if not remote_path.is_absolute() or ".." in remote_path.parts or not remote_path.name:
        raise RuntimeError("checkpoint path must be a canonical absolute path")
    model_dir = dest / remote_path.name
    if model_dir.is_symlink() or dest.resolve() not in model_dir.resolve().parents:
        raise RuntimeError("checkpoint destination escapes the rescue folder")
    identity = json.dumps({"ticket_id": candidate.ticket.id, "path": candidate.path}, sort_keys=True)
    if (model_dir / RESCUE_MARKER).exists():
        if (model_dir / RESCUE_MARKER).read_text(encoding="utf-8") == identity:
            return model_dir
        raise RuntimeError("rescue destination belongs to a different checkpoint; choose another folder")
    dest.mkdir(parents=True, exist_ok=True)
    owner = model_dir / ".zevo-rescue-owner"
    source_is_destination = not candidate.remote and Path(candidate.path).resolve() == model_dir.resolve()
    if model_dir.exists() and not source_is_destination:
        if not owner.is_file() or owner.read_text(encoding="utf-8") != identity:
            raise RuntimeError("rescue destination is not owned by this checkpoint; choose another folder")
    model_dir.mkdir(parents=True, exist_ok=True)
    owner.write_text(identity, encoding="utf-8")
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
        await _run_transfer(command)
    else:
        source = Path(candidate.path)
        if not source.exists():
            raise RuntimeError(f"checkpoint path no longer exists: {source}")
        if source.resolve() != model_dir.resolve():
            await _run_transfer([sys.executable, "-c", "import shutil,sys; shutil.copytree(sys.argv[1],sys.argv[2],dirs_exist_ok=True)", str(source), str(model_dir)])
    if not model_dir.exists():
        raise RuntimeError(f"copy finished but {model_dir} is missing")
    (model_dir / RESCUE_MARKER).write_text(
        identity, encoding="utf-8",
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


async def push_to_hf(model_dir: Path, *, repo_id: str, private: bool, token: str) -> str:
    # Credentials go through stdin, never process arguments or logs.
    script = (
        "import json,sys; from huggingface_hub import HfApi; p=json.load(sys.stdin); "
        "api=HfApi(token=p['token']); api.create_repo(p['repo'],private=p['private'],exist_ok=True); "
        "api.upload_folder(folder_path=p['path'],repo_id=p['repo'],ignore_patterns=['.zevo-*'])"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        await proc.communicate(json.dumps({"token": token, "repo": repo_id, "private": private, "path": str(model_dir)}).encode())
        if proc.returncode:
            raise RuntimeError(f"Hugging Face upload exited {proc.returncode}; local checkpoint retained")
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
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
        outcome["hf_url"] = await push_to_hf(
            model_dir,
            repo_id=policy.hf_repo_id, private=policy.hf_private, token=token,
        )
    return outcome


_ACTIVE: dict[str, asyncio.Task[None]] = {}


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def rescue_request_outcome(policy: CancelWeightsPolicy, now: datetime) -> dict[str, Any]:
    return {"status": "pending", "attempts": 0,
            "deadline_at": (now + timedelta(seconds=policy.rescue_timeout_seconds)).isoformat(),
            "source_retained": True, "compute_may_accrue": True}


async def complete_requested_cancels(session: AsyncSession) -> int:
    """Claim bounded attempts durably, including recovery after daemon restarts."""
    runs = (await session.execute(select(Run).where(
        Run.cancel_requested_at.is_not(None), Run.status.in_(["planning", "running"]),
    ).with_for_update(skip_locked=True))).scalars().all()
    started = 0
    now = datetime.now(timezone.utc)
    claimed = []
    for run in runs:
        if run.id in _ACTIVE and not _ACTIVE[run.id].done():
            continue
        policy = CancelWeightsPolicy.model_validate(run.cancel_policy or {})
        outcome = {**rescue_request_outcome(policy, _aware(run.cancel_requested_at)), **dict(run.cancel_outcome or {})}
        if outcome.get("status") == "preservation_failed":
            continue
        retry_at = outcome.get("retry_at")
        if retry_at and _aware(datetime.fromisoformat(retry_at)) > now:
            continue
        lease = outcome.get("lease_until")
        if lease and _aware(datetime.fromisoformat(lease)) > now:
            continue  # another scheduler process owns the transfer
        outcome.update(status="rescuing", claim_token=str(uuid.uuid4()),
                       lease_until=(now + timedelta(seconds=policy.attempt_timeout_seconds + 10)).isoformat())
        run.cancel_outcome = outcome
        claimed.append(run.id)
    await session.commit()
    for run_id in claimed:
        _ACTIVE[run_id] = asyncio.create_task(_finish_cancel(run_id))
        started += 1
    return started


async def _rescue_with_abort(Session, session, run, timeout: float):
    """Abort an in-flight copy when another API process requests discard/retry."""
    token = (run.cancel_outcome or {}).get("claim_token")
    task = asyncio.create_task(perform_rescue(session, run))
    deadline = asyncio.get_running_loop().time() + timeout
    try:
        while not task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("checkpoint rescue attempt exceeded its deadline")
            await asyncio.wait({task}, timeout=min(1.0, remaining))
            if task.done():
                break
            async with Session() as check:
                current = await check.get(Run, run.id)
                if current is None or current.status not in {"planning", "running"} or (current.cancel_outcome or {}).get("claim_token") != token:
                    raise asyncio.CancelledError()
        return await task
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def _finish_cancel(run_id: str) -> None:
    from zevo.db import get_session_factory
    Session = get_session_factory()
    try:
        async with Session() as session:
            run = await session.get(Run, run_id)
            if run is None:
                return
            policy = CancelWeightsPolicy.model_validate(run.cancel_policy or {})
            prior = dict(run.cancel_outcome or {})
            token = prior.get("claim_token")
            now = datetime.now(timezone.utc)
            deadline = _aware(datetime.fromisoformat(prior["deadline_at"]))
            attempts = int(prior.get("attempts") or 0)
            error = ""
            try:
                if now >= deadline or attempts >= policy.max_attempts:
                    raise TimeoutError("checkpoint rescue retry window exhausted")
                outcome = await _rescue_with_abort(
                    Session, session, run,
                    min(policy.attempt_timeout_seconds, (deadline - now).total_seconds()),
                )
            except Exception as exc:
                error = str(exc) or type(exc).__name__
                await session.rollback()
                outcome = {"weights": policy.weights, "error": error[:500]}
            # Never overwrite a newer discard/retry from another process.
            await session.refresh(run)
            if run.status not in {"planning", "running"} or (run.cancel_outcome or {}).get("claim_token") != token:
                return
            attempts += 1
            now = datetime.now(timezone.utc)
            local = await _registered_model_dir(session, run)
            exhausted = now >= deadline or attempts >= policy.max_attempts
            if error and local is None and (not exhausted or not policy.discard_on_failure):
                run.cancel_outcome = {
                    **outcome, "status": "preservation_failed" if exhausted else "retry_pending",
                    "attempts": attempts, "deadline_at": deadline.isoformat(),
                    "retry_at": (now + timedelta(seconds=30)).isoformat(),
                    "retryable": True, "source_retained": True, "compute_may_accrue": True,
                }
                run.halted_reason = "Checkpoint preservation failed; source retained and compute may still accrue. Retry or explicitly discard."
            else:
                if local is not None:
                    outcome["model_path"] = str(local)
                outcome.update(status="preservation_failed" if error else "completed",
                               attempts=attempts, finished_at=now.isoformat(),
                               source_retained=local is not None,
                               compute_may_accrue=True, retryable=bool(error))
                if error and policy.discard_on_failure and local is None:
                    outcome["discarded_after_failure"] = True
                run.cancel_outcome = outcome
                run.status = str((run.lifecycle or {}).get("rescue_terminal_status") or "cancelled")
                run.halted_reason = "Run stopped; " + _describe(outcome)
                run.finished_at = now
            await session.commit()
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
