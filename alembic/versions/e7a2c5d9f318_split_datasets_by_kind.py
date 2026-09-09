"""one dataset is one kind: split every bundle into <name>-train and <name>-eval

A bundle held both halves of a task: `train.csv` beside `test.csv`,
`test_public.csv`, `eval.py` and `sample_submission.csv`. They are not the same
kind of thing. The scoring four ARE the task's definition and must not drift;
the training file is what you swap when you try something else. Keeping them in
one directory meant a dataset could not say what it was for, and "show me my
training data" had to be answered by reading filenames.

So each bundle becomes two: `<name>-train` and `<name>-eval`, each declaring its
`kind` in source.json. The task rows are repointed — `dataset` at the training
half, the four scoring paths at the evaluation half.

Runs already made keep the paths they recorded, and those paths no longer
resolve; a finished run's input-file preview is the only thing that reads them.

The file moves live here rather than in the repo because workspace/ is
gitignored — same reason e5f6a8b9c1d2 copied the bundles in to begin with.

Revision ID: e7a2c5d9f318
Revises: d1f4b6c8e250
Create Date: 2026-08-09
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import sqlalchemy as sa
from alembic import op


revision = "e7a2c5d9f318"
down_revision = "d1f4b6c8e250"
branch_labels = None
depends_on = None


_ROOT = Path(__file__).resolve().parents[2]
_DATASETS = Path(os.environ.get("DATASETS_DIR", str(_ROOT / "workspace" / "datasets")))

# The scoring apparatus, by the names the bundles use. Everything else in a
# bundle is training data.
_EVAL_FILES = {"test.csv", "test_public.csv", "eval.py", "sample_submission.csv"}
_META = {"source.json", ".profile.json"}
# The task columns that point at each half.
_EVAL_COLUMNS = ("test_set", "test_set_public", "evaluation_script", "sample_submission")


def _write_source(d: Path, *, note: str, kind: str, remote: list) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "source.json").write_text(
        json.dumps({"note": note, "kind": kind, "remote": remote}, indent=2) + "\n"
    )


def upgrade() -> None:
    if not _DATASETS.is_dir():
        # A fresh checkout with no catalogue yet — the paths in the task rows
        # still have to move, so keep going.
        _DATASETS.mkdir(parents=True, exist_ok=True)

    moved: dict[str, tuple[str, str]] = {}   # old name -> (train name, eval name)
    for d in sorted(p for p in _DATASETS.iterdir() if p.is_dir()):
        # Already split, or already declares what it is: leave it alone.
        if d.name.endswith(("-train", "-eval")):
            continue
        src = {}
        f = d / "source.json"
        if f.is_file():
            try:
                src = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                src = {}
        if (src.get("kind") or "").strip():
            continue

        files = [p for p in d.iterdir() if p.is_file() and p.name not in _META]
        evals = [p for p in files if p.name in _EVAL_FILES]
        trains = [p for p in files if p.name not in _EVAL_FILES]
        note = str(src.get("note") or "")
        remote = list(src.get("remote") or [])
        # Remote files are training data: nothing is scored against a hub id.
        train_dir, eval_dir = _DATASETS / f"{d.name}-train", _DATASETS / f"{d.name}-eval"

        if trains or remote:
            _write_source(train_dir, note=note, kind="training", remote=remote)
            for p in trains:
                shutil.move(str(p), str(train_dir / p.name))
        if evals:
            _write_source(eval_dir, note=note, kind="evaluation", remote=[])
            for p in evals:
                shutil.move(str(p), str(eval_dir / p.name))
        if trains or remote or evals:
            moved[d.name] = (train_dir.name, eval_dir.name)
            # Whatever is left is bookkeeping (a stale profile cache for a file
            # that has moved), so the directory goes.
            shutil.rmtree(d, ignore_errors=True)

    if not moved:
        return

    # Repoint the tasks: the training column at the training half, the four
    # scoring columns at the evaluation half.
    conn = op.get_bind()
    for old, (train_name, eval_name) in moved.items():
        old_prefix = f"{_DATASETS}/{old}/"
        conn.execute(
            sa.text(
                "UPDATE tasks SET dataset = replace(dataset, :old, :new) "
                "WHERE dataset LIKE :like"
            ),
            {"old": old_prefix, "new": f"{_DATASETS}/{train_name}/", "like": f"{old_prefix}%"},
        )
        for col in _EVAL_COLUMNS:
            conn.execute(
                sa.text(
                    f"UPDATE tasks SET {col} = replace({col}, :old, :new) "
                    f"WHERE {col} LIKE :like"
                ),
                {"old": old_prefix, "new": f"{_DATASETS}/{eval_name}/", "like": f"{old_prefix}%"},
            )
        # Settings carry a training path of their own.
        conn.execute(
            sa.text(
                "UPDATE task_settings SET dataset = replace(dataset, :old, :new) "
                "WHERE dataset LIKE :like"
            ),
            {"old": old_prefix, "new": f"{_DATASETS}/{train_name}/", "like": f"{old_prefix}%"},
        )


def downgrade() -> None:
    """Put each pair back into one directory and repoint the tasks at it."""
    conn = op.get_bind()
    if not _DATASETS.is_dir():
        return
    for d in sorted(p for p in _DATASETS.iterdir() if p.is_dir()):
        if not d.name.endswith("-train"):
            continue
        base = d.name[: -len("-train")]
        whole = _DATASETS / base
        src = {}
        f = d / "source.json"
        if f.is_file():
            try:
                src = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                src = {}
        _write_source(whole, note=str(src.get("note") or ""), kind="",
                      remote=list(src.get("remote") or []))
        for half in (d, _DATASETS / f"{base}-eval"):
            if not half.is_dir():
                continue
            for p in half.iterdir():
                if p.is_file() and p.name not in _META:
                    shutil.move(str(p), str(whole / p.name))
            shutil.rmtree(half, ignore_errors=True)
        prefix = f"{_DATASETS}/{base}"
        for col in ("dataset",) + _EVAL_COLUMNS:
            conn.execute(
                sa.text(
                    f"UPDATE tasks SET {col} = replace(replace({col}, "
                    f":train, :whole), :eval, :whole) WHERE {col} LIKE :like"
                ),
                {"train": f"{prefix}-train/", "eval": f"{prefix}-eval/",
                 "whole": f"{prefix}/", "like": f"{prefix}-%"},
            )
        conn.execute(
            sa.text(
                "UPDATE task_settings SET dataset = replace(dataset, :train, :whole) "
                "WHERE dataset LIKE :like"
            ),
            {"train": f"{prefix}-train/", "whole": f"{prefix}/", "like": f"{prefix}-train/%"},
        )
