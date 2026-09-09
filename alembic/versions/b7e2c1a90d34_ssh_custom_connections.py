"""custom SSH connections: ssh_hosts table + runs.ssh_host_id + gpu_provider='ssh'

Adds the "custom SSH connections" feature (single-operator): register named,
verified SSH-reachable GPU boxes and target one per run via gpu_provider='ssh'.

- ssh_hosts: registered boxes (no tenant/user scoping). private_key is stored
  encrypted at rest when ZEVO_SECRET_ENC_KEY is set, plaintext otherwise.
- runs.ssh_host_id: which SshHost a run targets (empty for other providers).
- widen ck_runs_gpu_provider to allow 'ssh'.

Idempotent.

Revision ID: b7e2c1a90d34
Revises: ae9f0a1b2c3d
Create Date: 2026-08-29
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import JSONB


revision = "b7e2c1a90d34"
down_revision = "ae9f0a1b2c3d"
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    tables = set(insp.get_table_names())

    if "ssh_hosts" not in tables:
        op.create_table(
            "ssh_hosts",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("label", sa.String(length=128), nullable=False, server_default=""),
            sa.Column("host", sa.String(length=255), nullable=False),
            sa.Column("port", sa.Integer(), nullable=False, server_default="22"),
            sa.Column("username", sa.String(length=64), nullable=False),
            sa.Column("private_key", sa.Text(), nullable=False, server_default=""),
            sa.Column("status", sa.String(length=16), nullable=False, server_default="unverified"),
            sa.Column("gpu_info", _JSON, nullable=True),
            sa.Column("last_error", sa.Text(), nullable=False, server_default=""),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=True),
        )

    run_cols = {c["name"] for c in insp.get_columns("runs")}
    if "ssh_host_id" not in run_cols:
        op.add_column(
            "runs",
            sa.Column("ssh_host_id", sa.String(length=36), nullable=False, server_default=""),
        )

    # Widen the gpu_provider CHECK to allow 'ssh'. Postgres only — sqlite (tests)
    # builds tables from the models, which already carry the widened constraint.
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE runs DROP CONSTRAINT IF EXISTS ck_runs_gpu_provider")
        op.create_check_constraint(
            "ck_runs_gpu_provider",
            "runs",
            "gpu_provider IN ('instance', 'cluster', 'cloud', 'ssh')",
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE runs DROP CONSTRAINT IF EXISTS ck_runs_gpu_provider")
        op.create_check_constraint(
            "ck_runs_gpu_provider",
            "runs",
            "gpu_provider IN ('instance', 'cluster', 'cloud')",
        )
    with op.batch_alter_table("runs") as batch:
        batch.drop_column("ssh_host_id")
    op.drop_table("ssh_hosts")
