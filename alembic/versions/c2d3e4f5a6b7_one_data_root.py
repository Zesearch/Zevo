"""point every stored path at the single data/ root

The three places runtime data lived — `./workspace/datasets`, `./workspace/uploads`
and `./runs` — are now `./data/files`, `./data/uploads` and `./data/runs`, and
each keeps its name inside the container (`/app/data/...`). Two prefixes moved:

    /app/workspace/datasets/...  ->  /app/data/files/...
    /app/workspace/...           ->  /app/data/...        (uploads)
    /tmp/zevo_run/...            ->  /app/data/runs/...

Stored paths have to move with them. `tasks` and `task_settings` are LIVE — every
new run reads them to find its training and scoring files — so leaving those
behind would break the next run, not just the record of old ones. The rest is
history that the UI still reads: work_products drives the Artifacts panel's
download links, heartbeat_runs.stdout_path the transcript, and tickets.payload
what a stage was asked to do.

Text columns are rewritten with `replace`; the JSONB ones are cast to text,
rewritten and cast back, which is safe here because the prefixes cannot occur
inside a key name.

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2026-08-14

"""
from typing import Sequence, Union

from alembic import op


revision: str = 'c2d3e4f5a6b7'
down_revision: Union[str, None] = 'b1c2d3e4f5a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# (old, new). Order matters: the datasets prefix is more specific than the
# workspace one and has to be rewritten first, or `/app/workspace/datasets`
# would become `/app/data/datasets`.
_MOVES = [
    ("/app/workspace/datasets/", "/app/data/files/"),
    ("/app/workspace/", "/app/data/"),
    ("/tmp/zevo_run/", "/app/data/runs/"),
]
# Same specificity rule on the way back: `/app/data/files/` must be rewritten
# before the generic `/app/data/`, or the generic rule turns it into
# `/app/workspace/files/` and the specific one never matches. Sorting by
# prefix length (longest first) is that rule stated once, instead of relying
# on a hand-reversed list order.
_REVERSE = sorted(((new, old) for old, new in _MOVES), key=lambda p: -len(p[0]))

# table -> text columns
_TEXT = {
    "tasks": ["dataset", "test_set", "evaluation_script", "sample_submission"],
    "task_settings": ["dataset", "validation_set", "validation_eval_script",
                      "validation_sample_submission"],
    "work_products": ["path"],
    "heartbeat_runs": ["stdout_path", "stderr_path"],
    "registry_models": ["dataset", "model_path"],
    "comments": ["body"],
}
# table -> jsonb columns
_JSON = {
    "runs": ["holdout"],
    "tickets": ["payload", "refs"],
}


def _apply(moves) -> None:
    for table, cols in _TEXT.items():
        for col in cols:
            for old, new in moves:
                op.execute(
                    f"UPDATE {table} SET {col} = replace({col}, '{old}', '{new}') "
                    f"WHERE {col} LIKE '%{old}%'"
                )
    for table, cols in _JSON.items():
        for col in cols:
            for old, new in moves:
                op.execute(
                    f"UPDATE {table} SET {col} = "
                    f"replace({col}::text, '{old}', '{new}')::jsonb "
                    f"WHERE {col}::text LIKE '%{old}%'"
                )


def upgrade() -> None:
    _apply(_MOVES)


def downgrade() -> None:
    _apply(_REVERSE)
