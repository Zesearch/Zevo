"""put each domain's files back in one folder: `<name>-train` + `<name>-eval` -> `<name>`

Splitting a bundle by kind (e7a2c5d9f318) rested on a dataset being one thing
or the other: data a model learns from, or the fixed apparatus it is scored by.
That distinction is real, but it belongs to the FILE and to the place the file
is USED — a task names its test set, a setting names its training and
validation data — not to the folder holding them. Declared on the folder as
well, the two could disagree, and only one of them was what a run actually
read.

So the halves come back together. Nothing is lost in the merge: the two
directories of a pair never held a file of the same name (`train.csv` on one
side, `eval.py` / `sample_submission.csv` / `test.csv` on the other), and their
`source.json` notes were written as one sentence and copied to both.

Revision ID: b7e4c02a9d15
Revises: a2f8c31d5e04
Create Date: 2026-08-11
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import sqlalchemy as sa
from alembic import op


revision = "b7e4c02a9d15"
down_revision = "a2f8c31d5e04"
branch_labels = None
depends_on = None


_ROOT = Path(__file__).resolve().parents[2]
_DATASETS = Path(os.environ.get("DATASETS_DIR", str(_ROOT / "workspace" / "datasets")))
_META = "source.json"
# Which half a file belongs to, for the downgrade. Same list the split used.
_EVAL_FILES = {"test.csv", "eval.py", "sample_submission.csv"}

# Every column and payload that can hold one of these paths. `runs.holdout`
# carries the test-set paths the harness measures against, so a merge that
# skipped it would leave in-flight runs pointing at a directory that is gone.
_TEXT_COLUMNS = [
    ("tasks", "test_set"), ("tasks", "evaluation_script"),
    ("tasks", "sample_submission"), ("tasks", "dataset"),
    ("task_settings", "dataset"), ("task_settings", "validation_set"),
]
_JSON_COLUMNS = [("tickets", "payload"), ("runs", "holdout")]


def _bases() -> list[str]:
    if not _DATASETS.is_dir():
        return []
    names = {p.name for p in _DATASETS.iterdir() if p.is_dir()}
    return sorted({
        n.rsplit("-", 1)[0] for n in names
        if n.endswith(("-train", "-eval")) and n.rsplit("-", 1)[0]
    })


def _repoint(conn, old: str, new: str) -> None:
    for table, col in _TEXT_COLUMNS:
        conn.execute(
            sa.text(f"UPDATE {table} SET {col} = replace({col}, :old, :new) "
                    f"WHERE {col} LIKE :like"),
            {"old": old, "new": new, "like": f"%{old}%"},
        )
    for table, col in _JSON_COLUMNS:
        conn.execute(
            sa.text(f"UPDATE {table} SET {col} = replace({col}::text, :old, :new)::jsonb "
                    f"WHERE {col}::text LIKE :like"),
            {"old": old, "new": new, "like": f"%{old}%"},
        )


def upgrade() -> None:
    conn = op.get_bind()
    for base in _bases():
        merged = _DATASETS / base
        merged.mkdir(parents=True, exist_ok=True)
        note, remote = "", []
        for half in ("train", "eval"):
            src = _DATASETS / f"{base}-{half}"
            if not src.is_dir():
                continue
            for f in sorted(src.iterdir()):
                if not f.is_file():
                    continue
                if f.name == _META:
                    try:
                        blob = json.loads(f.read_text())
                    except (json.JSONDecodeError, OSError):
                        blob = {}
                    note = note or str(blob.get("note") or "")
                    for r in blob.get("remote") or []:
                        key = (r.get("id"), r.get("split") or "")
                        if key not in {(x.get("id"), x.get("split") or "") for x in remote}:
                            remote.append(r)
                    continue
                target = merged / f.name
                # Never clobber: the pairs do not collide today, but a
                # hand-edited bundle might, and losing a file to a rename is
                # not something the user would find out about.
                if target.exists():
                    target = merged / f"{f.stem}__{half}{f.suffix}"
                shutil.move(str(f), str(target))
            _repoint(conn, f"/{base}-{half}/", f"/{base}/")
            shutil.rmtree(src, ignore_errors=True)
        # `kind` is gone from DatasetSource; writing it back would resurrect a
        # field nothing reads.
        (merged / _META).write_text(
            json.dumps({"note": note, "remote": remote}, indent=2) + "\n"
        )


def downgrade() -> None:
    conn = op.get_bind()
    if not _DATASETS.is_dir():
        return
    for merged in sorted(p for p in _DATASETS.iterdir() if p.is_dir()):
        files = [f for f in merged.iterdir() if f.is_file() and f.name != _META]
        if not any(f.name in _EVAL_FILES for f in files):
            continue  # not a merged bundle
        try:
            blob = json.loads((merged / _META).read_text())
        except (json.JSONDecodeError, OSError):
            blob = {}
        for half, want_eval in (("train", False), ("eval", True)):
            side = _DATASETS / f"{merged.name}-{half}"
            side.mkdir(parents=True, exist_ok=True)
            for f in files:
                if (f.name in _EVAL_FILES) is want_eval:
                    shutil.move(str(f), str(side / f.name))
            (side / _META).write_text(json.dumps({
                "note": blob.get("note", ""),
                "kind": "evaluation" if want_eval else "training",
                # Remote entries are training data: nothing is scored on a hub id.
                "remote": [] if want_eval else (blob.get("remote") or []),
            }, indent=2) + "\n")
            _repoint(conn, f"/{merged.name}/", f"/{merged.name}-{half}/")
        shutil.rmtree(merged, ignore_errors=True)
