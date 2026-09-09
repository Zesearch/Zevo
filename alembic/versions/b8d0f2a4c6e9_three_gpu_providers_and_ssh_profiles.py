"""make SSH profiles belong to cluster or instance providers

Revision ID: b8d0f2a4c6e9
Revises: a7c9e1f3b5d8
"""

import sqlalchemy as sa
from alembic import op


revision = "b8d0f2a4c6e9"
down_revision = "a7c9e1f3b5d8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("ssh_hosts") as batch:
        batch.add_column(sa.Column(
            "category", sa.String(length=16), nullable=False,
            server_default="instance",
        ))
        batch.add_column(sa.Column(
            "key_path", sa.Text(), nullable=False, server_default="",
        ))
        batch.add_column(sa.Column(
            "remote_dir", sa.Text(), nullable=False, server_default="",
        ))
        batch.add_column(sa.Column(
            "env_setup", sa.Text(), nullable=False, server_default="",
        ))
        batch.create_check_constraint(
            "ck_ssh_hosts_category",
            "category IN ('cluster', 'instance')",
        )

    # Historical rows may use the two categories this migration replaces.
    # They are provenance only; map them to the SSH-backed Instance category
    # before installing the narrower invariant.
    op.execute(
        "UPDATE runs SET gpu_provider = 'instance' "
        "WHERE gpu_provider IN ('ssh', 'local')"
    )
    with op.batch_alter_table("runs") as batch:
        batch.drop_constraint("ck_runs_gpu_provider", type_="check")
        batch.create_check_constraint(
            "ck_runs_gpu_provider",
            "gpu_provider IN ('cluster', 'cloud', 'instance')",
        )


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.drop_constraint("ck_runs_gpu_provider", type_="check")
        batch.create_check_constraint(
            "ck_runs_gpu_provider",
            "gpu_provider IN ('instance', 'cluster', 'cloud', 'ssh', 'local')",
        )

    with op.batch_alter_table("ssh_hosts") as batch:
        batch.drop_constraint("ck_ssh_hosts_category", type_="check")
        batch.drop_column("env_setup")
        batch.drop_column("remote_dir")
        batch.drop_column("key_path")
        batch.drop_column("category")
