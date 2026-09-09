"""Freeze and verify custom evaluators for either scoring lane.

Separately supplied Validation and held-out Test may own independent evaluator
contracts; Test-derived Validation reuses Test's contract. Zevo copies every
uploaded Python scorer into a content-addressed directory, records its SHA-256,
and verifies those exact bytes before the owning lane executes it.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from zevo.holdout_storage import resolve_asset
from zevo.paths import DATA_ROOT, evaluators_root


_CONTAINER_DATA_ROOT = Path("/app/data")


def _local_path(value: str | Path) -> Path:
    raw = Path(str(value or "").strip())
    resolved = Path(resolve_asset(raw))
    if resolved.is_file():
        return resolved
    try:
        relative = raw.relative_to(_CONTAINER_DATA_ROOT)
    except ValueError:
        relative = None
    if relative is not None:
        host = DATA_ROOT / relative
        if host.is_file():
            return host
    if not raw.is_absolute():
        candidate = Path(__file__).resolve().parents[2] / raw
        if candidate.is_file():
            return candidate
    return resolved


def file_sha256(path: str | Path) -> str:
    source = _local_path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_evaluator(path: str | Path) -> tuple[str, str]:
    """Return ``(managed_path, sha256)`` for a valid custom Python scorer."""
    source = _local_path(path)
    if not source.is_file():
        raise ValueError(f"evaluation_script is not a readable file: {path}")
    if source.suffix.lower() != ".py":
        raise ValueError("custom evaluation_script must be a Python (.py) file")
    try:
        compile(source.read_text(encoding="utf-8"), str(source), "exec")
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError(f"custom evaluation_script is not valid Python: {exc}") from exc

    digest = file_sha256(source)
    destination = Path(evaluators_root()) / digest[:2] / f"{digest}.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        temporary = destination.with_suffix(".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    return str(destination), digest


def verify_evaluator(path: str | Path, expected_sha256: str) -> None:
    """Reject mutation or replacement of a frozen evaluator."""
    if not expected_sha256:
        raise ValueError("custom evaluator is missing evaluator_sha256")
    try:
        actual = file_sha256(path)
    except OSError as exc:
        raise ValueError(f"cannot read custom evaluation_script: {exc}") from exc
    if actual != expected_sha256:
        raise ValueError(
            "custom evaluation_script bytes do not match evaluator_sha256; "
            "create a new Task for a different evaluator"
        )
