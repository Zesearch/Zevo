"""Persist bounded Run finalization and recovery state."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = "e1a2b3c4d5f6"
down_revision = "e0f1a2b3c4d5"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("runs", sa.Column("lifecycle", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False, server_default="{}"))

def downgrade():
    op.drop_column("runs", "lifecycle")
