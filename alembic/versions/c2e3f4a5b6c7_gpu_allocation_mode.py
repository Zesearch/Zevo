"""Add run-owned GPU allocation lifetime."""
from alembic import op
import sqlalchemy as sa
revision = "c2e3f4a5b6c7"
down_revision = "b1d2e3f4a5b6"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("runs", sa.Column("gpu_allocation_mode", sa.String(16), nullable=False, server_default="per_stage"))
    # Preserve the existing direct-host lease semantics for historical runs.
    op.execute("UPDATE runs SET gpu_allocation_mode='per_run' WHERE gpu_provider IN ('instance', 'cloud')")
    op.create_check_constraint("ck_runs_gpu_allocation_mode", "runs", "gpu_allocation_mode IN ('per_stage', 'per_run')")

def downgrade():
    op.drop_constraint("ck_runs_gpu_allocation_mode", "runs", type_="check")
    op.drop_column("runs", "gpu_allocation_mode")
