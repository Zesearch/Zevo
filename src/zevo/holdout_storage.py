"""Physical storage boundary for held-out Test assets.

Task and Run records keep their familiar logical paths under ``data/files`` or
``data/uploads`` so the UI does not need a second path language.  The bytes,
however, are mirrored beneath :func:`zevo.paths.holdout_root`.  Only the
backend and the ``held_out_test`` scheduler mount that root.

This module is intentionally small and side-effect free except for
``protect_asset``.  Readers call ``resolve_asset``; writers call
``protect_asset`` once when a Test contract is accepted.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from zevo.paths import files_root, holdout_root, uploads_root


_CONTAINER_DATA_ROOT = Path("/app/data")


def _managed_source(path: str | Path) -> tuple[str, Path, Path] | None:
    """Return ``(kind, configured_root, relative_path)`` for managed data.

    Persisted paths may have been written inside a container and later read by
    a host-side command, so ``/app/data/...`` is translated to the configured
    roots before existence checks.
    """
    raw = Path(str(path).strip())
    candidates = (
        ("files", Path(files_root())),
        ("uploads", Path(uploads_root())),
    )
    for kind, root in candidates:
        try:
            return kind, root, raw.relative_to(root)
        except ValueError:
            pass

    # CLI users commonly pass repo-relative paths while the API persists
    # container-absolute ones. Both name the same managed asset.
    if not raw.is_absolute():
        parts = raw.parts
        if len(parts) >= 3 and parts[:2] == ("data", "files"):
            return "files", Path(files_root()), Path(*parts[2:])
        if len(parts) >= 3 and parts[:2] == ("data", "uploads"):
            return "uploads", Path(uploads_root()), Path(*parts[2:])

    try:
        rel = raw.relative_to(_CONTAINER_DATA_ROOT)
    except ValueError:
        return None
    if not rel.parts:
        return None
    if rel.parts[0] == "files":
        return "files", Path(files_root()), Path(*rel.parts[1:])
    if rel.parts[0] == "uploads":
        return "uploads", Path(uploads_root()), Path(*rel.parts[1:])
    return None


def private_mirror(path: str | Path) -> Path | None:
    """The private mirror for one logical managed path, if it has one."""
    managed = _managed_source(path)
    if managed is None:
        return None
    kind, _root, rel = managed
    return Path(holdout_root()) / kind / rel


def resolve_asset(path: str | Path) -> str:
    """Resolve a logical Test path to bytes visible to the held-out engine.

    A private copy wins even if a stale public copy exists.  Remote identifiers
    and unmanaged paths are returned unchanged.
    """
    value = str(path or "").strip()
    if not value:
        return ""
    raw = Path(value)
    private_root = Path(holdout_root())
    try:
        if raw.resolve().is_relative_to(private_root.resolve()) and raw.is_file():
            return str(raw)
    except (OSError, ValueError):
        pass
    mirror = private_mirror(value)
    if mirror is not None and mirror.is_file():
        return str(mirror)
    return value


def protect_asset(path: str | Path) -> str:
    """Move a managed local Test asset behind the private storage boundary.

    The returned value remains the original logical path for managed Files and
    Uploads assets.  That keeps saved Tasks stable while making the bytes
    absent from the optimization scheduler's mount.  Calling this repeatedly
    is idempotent.
    """
    value = str(path or "").strip()
    if not value:
        return ""
    mirror = private_mirror(value)
    if mirror is None:
        # Hub ids/URLs are materialized by the held-out Data lane.  Refuse to
        # relocate arbitrary code/repo paths here: doing so could delete source
        # files a user meant to share for another purpose.
        return value

    managed = _managed_source(value)
    assert managed is not None
    _kind, root, rel = managed
    source = root / rel
    if mirror.is_file() and not source.is_file():
        return value
    if not source.is_file():
        return value

    mirror.parent.mkdir(parents=True, exist_ok=True)
    tmp = mirror.with_name(
        f".{mirror.name}.{hashlib.sha256(str(source).encode()).hexdigest()[:10]}.tmp"
    )
    shutil.copy2(source, tmp)
    tmp.replace(mirror)
    source.unlink()

    # UUID attachment directories and test/ folders should disappear once
    # empty; stop at the managed root and never remove a file-set directory.
    for parent in source.parents:
        if parent == root or not parent.is_relative_to(root):
            break
        if any(parent.iterdir()):
            break
        if managed[0] == "files" and parent.parent == root:
            break
        parent.rmdir()
    return value


def protect_assets(*paths: str | Path) -> tuple[str, ...]:
    """Protect several Test assets in field order."""
    return tuple(protect_asset(path) for path in paths)
