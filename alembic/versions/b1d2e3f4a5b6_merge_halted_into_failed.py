"""merge the retired halted Run status into failed

Revision ID: b1d2e3f4a5b6
Revises: a0c1e2f3b4d5
"""
from __future__ import annotations

from alembic import op


revision = "b1d2e3f4a5b6"
down_revision = "a0c1e2f3b4d5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_runs_status", "runs", type_="check")
    op.execute("UPDATE runs SET status = 'failed' WHERE status = 'halted'")
    op.execute(
        """
        UPDATE runs
        SET lifecycle = jsonb_set(
            COALESCE(lifecycle, '{}'::jsonb),
            '{rescue_terminal_status}',
            '"failed"'::jsonb,
            true
        )
        WHERE lifecycle ->> 'rescue_terminal_status' = 'halted'
        """
    )
    op.create_check_constraint(
        "ck_runs_status",
        "runs",
        "status IN ('planning', 'running', 'success', 'degraded', 'failed', 'cancelled')",
    )


def downgrade() -> None:
    # The old distinction cannot be reconstructed after the values are merged.
    op.drop_constraint("ck_runs_status", "runs", type_="check")
    op.create_check_constraint(
        "ck_runs_status",
        "runs",
        "status IN ('planning', 'running', 'success', 'degraded', 'failed', 'halted', 'cancelled')",
    )
