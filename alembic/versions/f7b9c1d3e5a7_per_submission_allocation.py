"""Allow Slurm allocations scoped to one workload submission."""
from alembic import op
revision = "f7b9c1d3e5a7"
down_revision = "e6a8b0c2d4f6"
branch_labels = None
depends_on = None

def upgrade():
    op.drop_constraint("ck_runs_gpu_allocation_mode", "runs", type_="check")
    op.create_check_constraint("ck_runs_gpu_allocation_mode", "runs", "gpu_allocation_mode IN ('per_stage', 'per_run', 'per_submission')")

def downgrade():
    op.execute("UPDATE runs SET gpu_allocation_mode='per_stage' WHERE gpu_allocation_mode='per_submission'")
    op.drop_constraint("ck_runs_gpu_allocation_mode", "runs", type_="check")
    op.create_check_constraint("ck_runs_gpu_allocation_mode", "runs", "gpu_allocation_mode IN ('per_stage', 'per_run')")
