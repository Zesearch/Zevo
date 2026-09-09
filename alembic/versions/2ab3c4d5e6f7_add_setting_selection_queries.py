"""Add natural-language model and method selection guidance.

Revision ID: 2ab3c4d5e6f7
Revises: 19f84ac2d7e1
Create Date: 2026-08-29
"""
from alembic import op
import sqlalchemy as sa


revision = "2ab3c4d5e6f7"
down_revision = "19f84ac2d7e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "task_settings",
        sa.Column("model_query", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "task_settings",
        sa.Column("method_query", sa.Text(), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("task_settings", "method_query")
    op.drop_column("task_settings", "model_query")
