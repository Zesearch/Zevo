"""Immutable upload publication and reference-aware catalogue removal."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, BinaryIO

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.db import Run, Task, TaskSetting, Ticket, WorkProduct
from zevo.paths import files_root, holdout_root


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def publish_upload(source: BinaryIO, destination: Path, *, max_bytes: int) -> int:
    """Publish complete bytes once, without replacing an existing version.

    A hard link is an atomic no-clobber publish on the same filesystem. An
    identical retry is idempotent; different content requires a new filename
    or file set. Failures only remove our staging file, never prior data.
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
        try:
            os.link(staged, destination)
        except FileExistsError:
            # Never follow a caller/worker-created symlink during publication.
            if (destination.is_symlink() or not destination.is_file()
                    or destination.stat().st_size != written
                    or _digest(destination) != _digest(staged)):
                raise HTTPException(409, "This filename already contains a different version. Upload with a new filename or file-set name; existing versions are immutable.")
        return written
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _file_key(value: str) -> Path | None:
    """Unify public/private and host/container spellings before comparing."""
    raw = Path(value.strip())
    roots = (Path(files_root()), Path(holdout_root()) / "files", Path("/app/data/files"), Path("/run/zevo-holdout/files"), Path("data/files"))
    for root in roots:
        try:
            return raw.resolve().relative_to(root.resolve())
        except (ValueError, OSError):
            continue
    return None


async def assert_unreferenced(db: AsyncSession, paths: list[Path]) -> None:
    """Fail closed while a saved task, setting, run or artifact uses an asset.

    Historical Runs count too: removing their input changes reproducibility.
    A file-set reference covers every member, not just a named leaf. Deleting
    the owning records or creating a new file-set version is an explicit step.
    """
    targets = [key for path in paths if (key := _file_key(str(path))) is not None]
    if not targets:
        raise HTTPException(409, "Cannot prove this file is safe to remove")
    for model in (Task, TaskSetting, Run, Ticket, WorkProduct):
        # Read plain stored values, never relationships or external assets.
        rows = (await db.execute(select(model))).scalars().all()
        for row in rows:
            values = [getattr(row, column.key) for column in model.__table__.columns]
            for value in _strings(values):
                key = _file_key(value)
                if key is not None and any(
                    key == target or key.is_relative_to(target) or target.is_relative_to(key)
                    for target in targets
                ):
                    raise HTTPException(409, "This file is referenced by a saved task, setting, run or artifact. Keep this version or remove its references before deleting it.")
