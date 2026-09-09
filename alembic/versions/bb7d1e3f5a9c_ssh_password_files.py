"""allow SSH connections to authenticate with a password file

Revision ID: bb7d1e3f5a9c
Revises: b8d0f2a4c6e9
"""

import sqlalchemy as sa
from alembic import op


revision = "bb7d1e3f5a9c"
down_revision = "b8d0f2a4c6e9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("ssh_hosts") as batch:
        batch.add_column(sa.Column(
            "password_path", sa.Text(), nullable=False, server_default="",
        ))
        batch.create_check_constraint(
            "ck_ssh_hosts_verified_credential",
            "status <> 'verified' OR "
            "((key_path <> '' AND password_path = '') OR "
            "(key_path = '' AND password_path <> ''))",
        )


def downgrade() -> None:
    with op.batch_alter_table("ssh_hosts") as batch:
        batch.drop_constraint("ck_ssh_hosts_verified_credential", type_="check")
        batch.drop_column("password_path")
