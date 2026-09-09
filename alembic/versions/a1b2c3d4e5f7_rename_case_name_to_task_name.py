"""rename runs.case_name -> runs.task_name

The column has always held the name of the task a run executes -- a predefined
package name, or whatever the user called their custom task. "case" was
fixture-shop vocabulary that leaked into the schema and the API; every surface
now says "task".

Revision ID: a1b2c3d4e5f7
Revises: f6b7c8d9e0a1
Create Date: 2026-07-31

"""
from alembic import op


revision = "a1b2c3d4e5f7"
down_revision = "f6b7c8d9e0a1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A rename keeps the data: existing runs carry their name across untouched.
    op.alter_column("runs", "case_name", new_column_name="task_name")


def downgrade() -> None:
    op.alter_column("runs", "task_name", new_column_name="case_name")
