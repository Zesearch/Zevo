"""Auto run mode: agent-derived scoring contract.

Two changes to `runs`:

* widen `ck_runs_mode` to accept 'auto' -- an objective-only launch whose
  scoring contract (metric, direction, held-out Test set, Validation split) is
  derived by the Data agent's `scope_problem` Ticket and settled by the engine
  afterwards, instead of being supplied by the user at creation;
* add `scoring_settled` -- true for every existing row and for every mode that
  settles at creation; an `auto` Run is created false and flips to true once
  settlement writes the contract onto the Run.

The check-constraint rewrite is Postgres only, matching c3f1a2b40d56 (sqlite
builds tables from the models, which already carry the widened constraint).

Revision ID: f1a2b3c4d5e7
Revises: e1a2b3c4d5e6
Create Date: 2026-09-05
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "f1a2b3c4d5e7"
down_revision = "e1a2b3c4d5e6"
branch_labels = None
depends_on = None

_MODES_NEW = "('full_pipeline', 'customized_pipeline', 'single_stage', 'auto')"
_MODES_OLD = "('full_pipeline', 'customized_pipeline', 'single_stage')"


def _set_mode_check(values: str) -> None:
    op.execute("ALTER TABLE runs DROP CONSTRAINT IF EXISTS ck_runs_mode")
    op.create_check_constraint("ck_runs_mode", "runs", f"mode IN {values}")


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.add_column(sa.Column(
            "scoring_settled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ))
    if op.get_bind().dialect.name == "postgresql":
        _set_mode_check(_MODES_NEW)


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        # Rows created in auto mode cannot satisfy the narrower constraint.
        op.execute("DELETE FROM runs WHERE mode = 'auto'")
        _set_mode_check(_MODES_OLD)
    with op.batch_alter_table("runs") as batch:
        batch.drop_column("scoring_settled")
