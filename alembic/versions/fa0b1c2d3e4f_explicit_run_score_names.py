"""Give Run summary scores explicit validation/test semantics.

Revision ID: fa0b1c2d3e4f
Revises: f9b0c1d2e3a4
"""

import sqlalchemy as sa
from alembic import op


revision = "fa0b1c2d3e4f"
down_revision = "f9b0c1d2e3a4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("runs", "headline_score", new_column_name="validation_score")
    op.alter_column("runs", "best_score", new_column_name="best_validation_score")
    op.alter_column("runs", "best_test_score", new_column_name="champion_test_score")
    op.drop_column("runs", "test_score")


def downgrade() -> None:
    op.add_column("runs", sa.Column("test_score", sa.Float(), nullable=True))
    op.execute("UPDATE runs SET test_score = champion_test_score")
    op.alter_column("runs", "champion_test_score", new_column_name="best_test_score")
    op.alter_column("runs", "best_validation_score", new_column_name="best_score")
    op.alter_column("runs", "validation_score", new_column_name="headline_score")
