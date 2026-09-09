"""registry_run_id

Adds registry_models.run_id (nullable FK to runs.id) so the lineage
endpoint can join registry → run → all_work_products without
heuristic path-matching.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-06-03 02:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "registry_models",
        sa.Column("run_id", sa.String(36), nullable=True),
    )
    op.create_foreign_key(
        "fk_registry_models_run_id",
        "registry_models", "runs",
        ["run_id"], ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_registry_models_run_id", "registry_models", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_registry_models_run_id", table_name="registry_models")
    op.drop_constraint("fk_registry_models_run_id", "registry_models", type_="foreignkey")
    op.drop_column("registry_models", "run_id")
