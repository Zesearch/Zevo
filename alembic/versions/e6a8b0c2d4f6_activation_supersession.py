"""Record instruction supersession on the activation."""
from alembic import op
import sqlalchemy as sa

revision = "e6a8b0c2d4f6"
down_revision = "d3f4a5b6c7d8"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("heartbeat_runs", sa.Column("superseded_by_instruction_id", sa.String(36), nullable=False, server_default=""))


def downgrade():
    op.drop_column("heartbeat_runs", "superseded_by_instruction_id")
