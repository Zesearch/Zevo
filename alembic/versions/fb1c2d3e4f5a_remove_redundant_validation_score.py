"""Remove the duplicate Run validation headline.

Revision ID: fb1c2d3e4f5a
Revises: fa0b1c2d3e4f
"""

import sqlalchemy as sa
from alembic import op


revision = "fb1c2d3e4f5a"
down_revision = "fa0b1c2d3e4f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("runs", "validation_score")


def downgrade() -> None:
    op.add_column("runs", sa.Column("validation_score", sa.Float(), nullable=True))
    op.execute("UPDATE runs SET validation_score = best_validation_score")
