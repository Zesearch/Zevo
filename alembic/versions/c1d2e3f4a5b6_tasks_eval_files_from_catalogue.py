"""Point the shipped tasks' test files at the catalogue too.

e5f6a8b9c1d2 moved only the training set, on the reasoning that test sets and
eval scripts are the scoring contract rather than training material. That left
every shipped task half in the catalogue and half in user_data — and the
launch dialog, which offers a dataset picker for each file, could not match the
user_data paths against anything, so it showed them as raw text.

The catalogue already holds byte-identical copies (e5f6a8b9c1d2 copies the whole
bundle, not just train.csv), and the eval scripts take their paths as argv —
they resolve nothing relative to their own location — so moving the reference is
safe. One place for data now.

Revision ID: c1d2e3f4a5b6
Revises: e5f6a8b9c1d2
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "c1d2e3f4a5b6"
down_revision = "e5f6a8b9c1d2"
branch_labels = None
depends_on = None


# bundle -> catalogue name, the same mapping e5f6a8b9c1d2 provisioned.
_BUNDLES = {
    "med": "medqa-usmle",
    "science": "arc-challenge",
    "math": "gsm8k",
    "code": "mbpp",
    "tiny": "medqa-tiny",
    "ifeval": "ifeval",
}
_COLUMNS = ("test_set", "evaluation_script", "sample_submission")
_FILES = ("test.csv", "test_public.csv", "eval.py", "sample_submission.csv")


def _move(reverse: bool = False) -> None:
    conn = op.get_bind()
    for bundle, name in _BUNDLES.items():
        for f in _FILES:
            old = f"/app/user_data/{bundle}/{f}"
            new = f"/app/workspace/datasets/{name}/{f}"
            src, dst = (new, old) if reverse else (old, new)
            for col in _COLUMNS:
                conn.execute(
                    sa.text(f"UPDATE tasks SET {col} = :dst WHERE {col} = :src"),
                    {"src": src, "dst": dst},
                )


def upgrade() -> None:
    _move()


def downgrade() -> None:
    _move(reverse=True)
