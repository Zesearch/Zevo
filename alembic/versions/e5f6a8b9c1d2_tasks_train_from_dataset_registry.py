"""Register the shipped bundles as datasets, and point tasks at them.

The five training sets the product ships were only ever files under user_data/;
nothing listed them, named them, or profiled them. This promotes each bundle
into the dataset catalogue (workspace/datasets/<name>/) and repoints the task
rows b2c3d4e5f6a8 seeded, so a task says WHICH dataset it trains on rather than
where a file happens to sit.

The file copy lives here rather than in the repo because workspace/ is
gitignored — a fresh checkout has an empty catalogue, and a migration that only
rewrote the paths would leave every task pointing at a file that does not exist.

Test sets and eval scripts are copied in too (a dataset is the whole bundle),
but the tasks' `test_set` / `evaluation_script` still point at user_data: those
are the scoring contract, and repointing them is a runtime change, not a
cataloguing one.

Revision ID: e5f6a8b9c1d2
Revises: d4e5f6a8b9c1
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from alembic import op
import sqlalchemy as sa


revision = "e5f6a8b9c1d2"
down_revision = "d4e5f6a8b9c1"
branch_labels = None
depends_on = None


_ROOT = Path(__file__).resolve().parents[2]
_USER_DATA = Path(os.environ.get("USER_DATA_DIR", str(_ROOT / "user_data")))
_DATASETS = Path(os.environ.get("DATASETS_DIR", str(_ROOT / "workspace" / "datasets")))

_HF = "https://huggingface.co/datasets/"

# bundle -> (catalogue name, what it is, remote files it does not store).
# Provenance matches the build script that originally produced these files
# (since removed from ops/scripts).
_BUNDLES: dict[str, tuple[str, str, list[dict]]] = {
    "med": ("medqa-usmle",
            "USMLE-style medical multiple choice. The main medical benchmark here.", []),
    "science": ("arc-challenge",
                "Grade-school science multiple choice (ARC-Challenge).", []),
    "math": ("gsm8k",
             "Grade-school math word problems, answered with a boxed number.", []),
    "code": ("mbpp",
             "Python programming problems, scored by running their unit tests.", []),
    "tiny": ("medqa-tiny",
             "A 10-row slice of MedQA for end-to-end smoke tests. Not scored.", []),
    "ifeval": ("ifeval",
               "Free-form instruction following, scored by the official IFEval checkers. "
               "Trains on a HuggingFace set, not a local file.",
               [{"role": "train", "kind": "huggingface", "id": "trl-lib/Capybara",
                 "url": _HF + "trl-lib/Capybara"}]),
}


def _provision() -> None:
    """Copy each bundle into the catalogue. Never overwrites an existing file:
    a user who has already edited their copy keeps it."""
    for bundle, (name, note, remote) in _BUNDLES.items():
        src = _USER_DATA / bundle
        if not src.is_dir():
            continue
        dst = _DATASETS / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if f.is_file() and not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
        meta = dst / "source.json"
        if not meta.exists():
            meta.write_text(json.dumps({"note": note, "remote": remote}, indent=2) + "\n")


def upgrade() -> None:
    _provision()
    conn = op.get_bind()
    for bundle, (name, _note, _remote) in _BUNDLES.items():
        conn.execute(
            sa.text("UPDATE tasks SET dataset = :new WHERE dataset = :old"),
            {"old": f"/app/user_data/{bundle}/train.csv",
             "new": f"/app/workspace/datasets/{name}/train.csv"},
        )


def downgrade() -> None:
    # Paths go back; the copied files stay. Deleting a catalogue the user may
    # have added to is not something a schema downgrade should do.
    conn = op.get_bind()
    for bundle, (name, _note, _remote) in _BUNDLES.items():
        conn.execute(
            sa.text("UPDATE tasks SET dataset = :old WHERE dataset = :new"),
            {"old": f"/app/user_data/{bundle}/train.csv",
             "new": f"/app/workspace/datasets/{name}/train.csv"},
        )
