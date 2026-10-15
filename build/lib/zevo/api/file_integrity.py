"""Atomic publication for mutable Files-catalogue uploads."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, BinaryIO

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.db import Run, Ticket, WorkProduct
from zevo.engine.run.input_snapshots import (
    file_set_snapshot,
    record_evaluation_identity,
    snapshot_local_input,
)
from zevo.paths import files_root, holdout_root


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def publish_upload(source: BinaryIO, destination: Path, *, max_bytes: int) -> int:
    """Atomically publish complete bytes, replacing the live File if needed.

    The staging file is fsynced before ``replace``. An interrupted upload never
    damages the current File, while a successful upload becomes the new live
    version in one filesystem operation. Existing Runs read their own copies.
    """
    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".upload-", delete=False) as out:
            staged = Path(out.name)
            written = 0
            while chunk := source.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(413, "file exceeds the 200MB upload limit")
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise HTTPException(409, "The destination is not a regular file.")
        if (destination.is_file() and destination.stat().st_size == written
                and _digest(destination) == _digest(staged)):
            return written
        os.replace(staged, destination)
        staged = None
        return written
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def _catalogue_reference(value: str, name: str) -> tuple[Path, bool] | None:
    """Return a logical Files member and whether its bytes are private."""
    raw = Path(str(value or "").strip())
    if not str(value or "").strip():
        return None
    roots = (
        (Path(files_root()), False),
        (Path("/app/data/files"), False),
        (Path("data/files"), False),
        (Path(holdout_root()) / "files", True),
        (Path("/run/zevo-holdout/files"), True),
    )
    for root, private in roots:
        try:
            relative = raw.relative_to(root)
        except ValueError:
            continue
        if not relative.parts or relative.parts[0] != name:
            return None
        # Use the configured public spelling as the logical key. The private
        # snapshot helper resolves it to the protected mirror when required.
        return Path(files_root()) / relative, private
    return None


def _rewrite_catalogue_references(
    value: Any, *, run_id: str, name: str,
) -> tuple[Any, bool]:
    if isinstance(value, str):
        reference = _catalogue_reference(value, name)
        if reference is None:
            return value, False
        logical, private = reference
        copied = snapshot_local_input(run_id, str(logical), private=private)
        return (copied, copied != str(logical))
    if isinstance(value, dict):
        changed = False
        out = {}
        for key, child in value.items():
            out[key], child_changed = _rewrite_catalogue_references(
                child, run_id=run_id, name=name,
            )
            changed = changed or child_changed
        return out, changed
    if isinstance(value, list):
        changed = False
        out = []
        for child in value:
            rewritten, child_changed = _rewrite_catalogue_references(
                child, run_id=run_id, name=name,
            )
            out.append(rewritten)
            changed = changed or child_changed
        return out, changed
    return value, False


def _remote_ids(name: str) -> set[str]:
    source = Path(files_root()) / name / "source.json"
    try:
        payload = json.loads(source.read_text())
    except (OSError, TypeError, ValueError):
        return set()
    return {
        str(item.get("id") or "")
        for item in (payload.get("remote") or [])
        if isinstance(item, dict) and item.get("id")
    }


def _contains_remote_reference(value: Any, identifiers: set[str]) -> bool:
    if not identifiers:
        return False
    if isinstance(value, str):
        return value.strip() in identifiers
    if isinstance(value, dict):
        return any(_contains_remote_reference(child, identifiers) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_remote_reference(child, identifiers) for child in value)
    return False


async def preserve_legacy_run_inputs(db: AsyncSession, name: str) -> None:
    """Move pre-snapshot Run references off one live File before mutation.

    Files were immutable before Run snapshots were introduced, so the current
    bytes are the same bytes a legacy Run launched with. The first mutation
    copies them into each affected Run and rewrites its stored paths. Later
    edits to the same File skip those Runs once its name is recorded in
    ``preserved_file_sets``; other referenced Files are preserved independently.
    """
    # Serialize the one-time conversion with concurrent File edits. The first
    # request commits the snapshot before touching live bytes; later requests
    # then observe ``preserved_file_sets`` and leave that snapshot untouched.
    runs = (await db.execute(select(Run).with_for_update())).scalars().all()
    legacy = [
        run for run in runs
        if not (run.input_snapshot or {}).get("local_inputs_copied")
        and name not in set((run.input_snapshot or {}).get("preserved_file_sets") or [])
    ]
    if not legacy:
        return
    run_ids = [run.id for run in legacy]
    tickets = (await db.execute(
        select(Ticket).where(Ticket.run_id.in_(run_ids))
    )).scalars().all()
    ticket_by_id = {ticket.id: ticket for ticket in tickets}
    products = (await db.execute(
        select(WorkProduct).where(WorkProduct.ticket_id.in_(list(ticket_by_id)))
    )).scalars().all() if ticket_by_id else []
    tickets_by_run: dict[str, list[Ticket]] = {}
    for ticket in tickets:
        tickets_by_run.setdefault(ticket.run_id, []).append(ticket)
    products_by_run: dict[str, list[WorkProduct]] = {}
    for product in products:
        ticket = ticket_by_id.get(product.ticket_id)
        if ticket is not None:
            products_by_run.setdefault(ticket.run_id, []).append(product)

    file_snapshot = file_set_snapshot(name)
    remote_ids = _remote_ids(name)
    touched = False
    for run in legacy:
        changed = False
        original_holdout = dict(run.holdout or {})
        original_request: dict[str, Any] = {}
        for ticket in tickets_by_run.get(run.id, []):
            if ticket.id == run.supervisor_ticket_id and isinstance(ticket.payload, dict):
                request = ticket.payload.get("user_request")
                if isinstance(request, dict):
                    original_request = dict(request)
                    break
        referenced = any(
            _contains_remote_reference(getattr(run, field), remote_ids)
            for field in ("holdout", "decision_pins", "customizations", "model_lineages")
        )
        referenced = referenced or any(
            _contains_remote_reference(getattr(ticket, field), remote_ids)
            for ticket in tickets_by_run.get(run.id, [])
            for field in ("payload", "inputs", "customization")
        )

        for field in ("holdout", "decision_pins", "customizations", "model_lineages"):
            rewritten, field_changed = _rewrite_catalogue_references(
                getattr(run, field), run_id=run.id, name=name,
            )
            if field_changed:
                setattr(run, field, rewritten)
                changed = True
        for ticket in tickets_by_run.get(run.id, []):
            for field in ("payload", "inputs", "customization"):
                rewritten, field_changed = _rewrite_catalogue_references(
                    getattr(ticket, field), run_id=run.id, name=name,
                )
                if field_changed:
                    setattr(ticket, field, rewritten)
                    changed = True
        for product in products_by_run.get(run.id, []):
            rewritten_path, path_changed = _rewrite_catalogue_references(
                product.path, run_id=run.id, name=name,
            )
            rewritten_meta, meta_changed = _rewrite_catalogue_references(
                product.meta, run_id=run.id, name=name,
            )
            if path_changed:
                product.path = rewritten_path
            if meta_changed:
                product.meta = rewritten_meta
            changed = changed or path_changed or meta_changed
            referenced = referenced or _contains_remote_reference(
                (product.path, product.meta), remote_ids,
            )
        if not changed and not referenced:
            continue
        snapshot = dict(run.input_snapshot or {})
        snapshot["version"] = 1
        snapshot.setdefault("local_inputs_copied", False)
        snapshot["preserved_file_sets"] = sorted({
            *list(snapshot.get("preserved_file_sets") or []), name,
        })
        snapshot.setdefault("task", None)
        known_files = {
            str(item.get("name") or ""): dict(item)
            for item in (snapshot.get("files") or []) if isinstance(item, dict)
        }
        if file_snapshot:
            known_files[name] = file_snapshot
        snapshot["files"] = [known_files[key] for key in sorted(known_files) if key]
        snapshot.setdefault("sources", {
            "dataset": str(original_request.get("dataset") or ""),
            "test_sets": list(original_holdout.get("test_sets") or []),
            "validation_sets": list(original_holdout.get("validation_sets") or []),
        })
        snapshot.setdefault("evaluation_sha256", "")
        snapshot["legacy_preserved"] = True
        run.input_snapshot = record_evaluation_identity(snapshot, original_holdout)
        touched = True
    if touched:
        await db.commit()
