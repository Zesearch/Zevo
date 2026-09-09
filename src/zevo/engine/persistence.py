"""DB-write helpers for Run rows and the Registry manifest-to-database mirror.

System B owns per-ticket execution (heartbeat-driven runner under
`zevo.engine.run.runner` + drivers under `zevo.engine.agent.drivers`). This module
keeps just the two cross-system bridges still needed by the CLI,
backend routers, and wakeup daemon:

  - `create_run`: insert a Run row from a UserRequest.
  - `upsert_registry_from_manifest`: mirror one Registry Ticket's selected
    champion snapshot into the `registry_models` table.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.contracts.model_registry import RegistryEntry, model_tag_for_run
from zevo.db import RegistryModel, Run


def _host_path(p: str) -> str:
    """Normalize an in-container path to the HOST path the user sees, so the
    registry_models mirror never stores /tmp/... or /app/... . The registry
    agent already writes host paths; this is defense against a stray container
    path. Idempotent on paths that are already host-relative."""
    if not p:
        return p
    # Keep this bridge on the same one-root mapping used by the API. In the
    # standard deployment /app/data/runs/... is data/runs/... on the host.
    # Older special cases produced runs/... or workspace/runs/..., neither of
    # which points at the current bind mount.
    from zevo.api.artifacts import host_path
    return host_path(p)


def _registered_at(value: object) -> datetime | None:
    """Parse the Registry agent's committed timestamp without inventing one.

    Mirroring an unchanged stable ``M-<run8>`` entry must not make the model
    look newly registered. A winning replacement writes a new timestamp to
    YAML; a losing challenger leaves the YAML entry and its time untouched.
    """
    if isinstance(value, datetime):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


async def create_run(
    session: AsyncSession,
    *,
    task_name: str,
    task_objective: str,
    agent_objective: str,
    metric: str,
    metric_direction: str,
    validation_metric: str,
    validation_metric_direction: str,
    run_name: str = "",
    commit: bool = True,
) -> Run:
    """Insert a Run row from a UserRequest.

    ``commit=False`` flushes (assigning the id) WITHOUT committing, so the
    caller can complete the rest of run creation — split settlement, the
    supervisor Ticket — inside the SAME transaction and commit once at the end.
    That keeps creation atomic: if a later step raises (e.g. a recoverable 400
    while deriving the validation split), the session rolls back and no orphan
    ``planning`` Run is left behind.
    """
    run = Run(
        task_name=task_name,
        # Optional: callers that do not name the execution (the CLI, the API)
        # leave it blank and every list falls back to the run id.
        run_name=run_name,
        task_objective=task_objective,
        agent_objective=agent_objective,
        metric=metric,
        metric_direction=metric_direction,
        validation_metric=validation_metric,
        validation_metric_direction=validation_metric_direction,
        status="planning",
    )
    session.add(run)
    if commit:
        await session.commit()
    else:
        await session.flush()
    await session.refresh(run)
    return run


async def upsert_registry_from_manifest(
    session: AsyncSession,
    *,
    registry_manifest_path: str,
    run_id: str = "",
) -> int:
    """Mirror one Registry Ticket's committed champion into registry_models.

    The manifest is a Ticket-local snapshot, never a global append-only file.
    It must contain exactly the stable entry owned by ``run_id``. Returns one
    for an insert and zero for a replacement of that Run's existing champion.
    """
    import yaml

    p = Path(registry_manifest_path)
    if not p.exists():
        return 0
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    models = data.get("models", {}) or {}
    if not isinstance(models, dict):
        raise ValueError("Registry manifest models must be a mapping")
    if not run_id:
        raise ValueError("registry mirror requires the producing run_id")
    expected_tag = model_tag_for_run(run_id)
    if set(models) != {expected_tag}:
        raise ValueError(
            "Registry manifest must contain exactly the producing Run's stable "
            f"entry {expected_tag!r}"
        )

    if await session.get(Run, run_id) is None:
        raise ValueError(f"registry mirror run does not exist: {run_id}")

    entry = RegistryEntry.model_validate(models[expected_tag])
    if entry.run_id != run_id:
        raise ValueError(
            f"Registry manifest run_id differs: expected {run_id!r}, got {entry.run_id!r}"
        )
    realized = entry.model_dump(mode="json")
    existing = await session.get(RegistryModel, expected_tag)
    inserted = existing is None
    if existing is None:
        existing = RegistryModel(version_tag=expected_tag, run_id=run_id)
        session.add(existing)
    elif existing.run_id != run_id:
        # `M-<run8>` is compact, so defend its primary key against a prefix
        # collision. Missing ownership is invalid too; there is no legacy mode.
        raise ValueError(
            f"registry tag {expected_tag!r} belongs to run {existing.run_id!r}, "
            f"not {run_id!r}"
        )
    existing.base_model = entry.base_model
    existing.iteration = entry.iteration
    existing.training_method = entry.training_method
    existing.dataset_source = _host_path(entry.dataset_source)
    existing.model_path = _host_path(entry.model_path)
    existing.task_objective = entry.task_objective
    existing.metric = entry.metric
    existing.metric_direction = entry.metric_direction
    existing.eval = realized["eval"]
    committed_at = _registered_at(entry.registered_at)
    if committed_at is not None:
        existing.registered_at = committed_at
    await session.commit()
    return int(inserted)
