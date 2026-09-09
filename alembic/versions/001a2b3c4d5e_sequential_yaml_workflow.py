"""replace intent/rule state with specialist YAML artifacts

Revision ID: 001a2b3c4d5e
Revises: a16b7c8d9e0f
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "001a2b3c4d5e"
down_revision = "a16b7c8d9e0f"
branch_labels = None
depends_on = None


_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    # This is a replacement, not a compatibility layer: durable Specialist
    # YAML artifacts supersede both mutable Run-level stores.
    with op.batch_alter_table("runs") as batch:
        batch.drop_column("iteration_intents")
        batch.drop_column("active_rules")


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.add_column(
            sa.Column(
                "active_rules", _JSON, nullable=False,
                server_default='{"version": 0, "rules": {}}',
            )
        )
        batch.add_column(
            sa.Column("iteration_intents", _JSON, nullable=False, server_default="{}")
        )
