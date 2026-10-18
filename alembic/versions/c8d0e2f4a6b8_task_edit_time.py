"""Record Task edit times without inventing timestamps for historical rows."""
from alembic import op
import sqlalchemy as sa

revision = "c8d0e2f4a6b8"
down_revision = "f7b9c1d3e5a7"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("tasks", sa.Column("definition_updated_at", sa.DateTime(timezone=True), nullable=True))


def downgrade():
    op.drop_column("tasks", "definition_updated_at")
