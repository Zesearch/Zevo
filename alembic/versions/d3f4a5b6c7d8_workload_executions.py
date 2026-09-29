"""Register workload telemetry independently of Agent activations."""
from alembic import op
import sqlalchemy as sa

revision = "d3f4a5b6c7d8"
down_revision = "c2e3f4a5b6c7"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "workload_executions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("ticket_id", sa.String(64), sa.ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("heartbeat_id", sa.String(36), sa.ForeignKey("heartbeat_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("runtime_key", sa.String(256), nullable=False),
        sa.Column("log_path", sa.Text(), nullable=False),
        sa.Column("purpose", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_workload_executions_ticket_id", "workload_executions", ["ticket_id"])
    op.create_index("ix_workload_executions_heartbeat_id", "workload_executions", ["heartbeat_id"])


def downgrade():
    op.drop_table("workload_executions")
