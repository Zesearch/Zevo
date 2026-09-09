"""Where Zevo's data lives — one place that knows the layout.

Everything the system writes at run time sits under a single root:

    data/files/      what the user uploaded — the Files page
    data/uploads/    attachments posted through the UI
    data/evaluators/ content-addressed custom evaluators
    data/runs/       per-run artifacts, one directory per run

Held-out Test assets are the deliberate exception.  They live beneath
``.zevo-private/holdout`` (or ``ZEVO_HOLDOUT_ROOT``) so the ordinary
optimization scheduler can run without that directory mounted at all.

and the root keeps its name across the container boundary: the host's `./data`
is `/app/data`, so a path means the same thing on both sides and turning one
into the other is `s|^/app/||`. It used to take three different rules, because
`./runs` arrived as `/tmp/zevo_run` and `./workspace/datasets` as neither of the
names the UI or the API used for it.

The roots are DERIVED from the repo root rather than written out as `/app/...`.
In the container the repo IS `/app`, so the two agree; on a developer's checkout
the same expression still points somewhere that exists, which a literal
`/app/data/runs` does not — tests that create a work dir failed on exactly that.

Env overrides stay for deployments that keep data elsewhere; they win when set.
"""
from __future__ import annotations

import os
from pathlib import Path


# `src/zevo/paths.py` -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = REPO_ROOT / "data"


def _override(env: str, default: Path) -> str:
    value = (os.environ.get(env) or "").strip()
    return value.rstrip("/") if value else str(default)


def work_dir_root() -> str:
    """Per-run artifacts. `ZEVO_WORK_DIR` overrides."""
    return _override("ZEVO_WORK_DIR", DATA_ROOT / "runs")


def files_root() -> str:
    """The user-uploaded files catalogue. `ZEVO_FILES_DIR` overrides."""
    return _override("ZEVO_FILES_DIR", DATA_ROOT / "files")


def uploads_root() -> str:
    """Attachments posted through the UI. `ZEVO_UPLOAD_ROOT` overrides."""
    return _override("ZEVO_UPLOAD_ROOT", DATA_ROOT / "uploads")


def evaluators_root() -> str:
    """Content-addressed custom evaluators shared by both scoring lanes."""
    return _override("ZEVO_EVALUATORS_ROOT", DATA_ROOT / "evaluators")


def holdout_root() -> str:
    """Engine-private Test assets. ``ZEVO_HOLDOUT_ROOT`` overrides.

    This root must be mounted only into the backend and the dedicated
    held-out scheduler.  Optimization agents must not be able to traverse it.
    """
    return _override("ZEVO_HOLDOUT_ROOT", REPO_ROOT / ".zevo-private" / "holdout")
