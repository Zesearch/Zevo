"""Keep a historical Setting name on each Run.

Revision ID: f9b0c1d2e3a4
Revises: e8f9a0b1c2d3
"""

import sqlalchemy as sa
from alembic import op


revision = "f9b0c1d2e3a4"
down_revision = "e8f9a0b1c2d3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("setting_name", sa.String(length=32), nullable=False, server_default=""),
    )
    op.execute(
        """
        UPDATE runs
        SET setting_name = task_settings.name
        FROM task_settings
        WHERE runs.setting_id = task_settings.id
        """
    )


def downgrade() -> None:
    op.drop_column("runs", "setting_name")
