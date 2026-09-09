"""run_stop_policy

Adds runs.min_delta_per_iter + runs.regression_tolerance for the
Zevo model-improvement loop policy (#7).

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-06-03 03:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("min_delta_per_iter", sa.Float(), nullable=False, server_default="0.0"),
    )
    op.add_column(
        "runs",
        sa.Column("regression_tolerance", sa.Float(), nullable=False, server_default="0.0"),
    )


def downgrade() -> None:
    op.drop_column("runs", "regression_tolerance")
    op.drop_column("runs", "min_delta_per_iter")
