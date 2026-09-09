"""name held-out scoring fields symmetrically with validation fields

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-08-14

"""
from typing import Sequence, Union

from alembic import op


revision: str = "d3e4f5a6b7c8"
down_revision: Union[str, None] = "c2d3e4f5a6b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "tasks", "sample_submission", new_column_name="test_sample_submission"
    )
    op.alter_column(
        "tasks", "evaluation_script", new_column_name="test_eval_script"
    )

    # Run holdout state is JSONB and survives task edits. Rename its keys too,
    # while preserving any row that already contains the canonical key.
    op.execute("""
        UPDATE runs
        SET holdout = (holdout - 'sample_submission') ||
            jsonb_build_object(
                'test_sample_submission',
                COALESCE(holdout->'test_sample_submission', holdout->'sample_submission')
            )
        WHERE holdout ? 'sample_submission'
    """)
    op.execute("""
        UPDATE runs
        SET holdout = (holdout - 'evaluation_script') ||
            jsonb_build_object(
                'test_eval_script',
                COALESCE(holdout->'test_eval_script', holdout->'evaluation_script')
            )
        WHERE holdout ? 'evaluation_script'
    """)


def downgrade() -> None:
    op.execute("""
        UPDATE runs
        SET holdout = (holdout - 'test_sample_submission') ||
            jsonb_build_object('sample_submission', holdout->'test_sample_submission')
        WHERE holdout ? 'test_sample_submission'
    """)
    op.execute("""
        UPDATE runs
        SET holdout = (holdout - 'test_eval_script') ||
            jsonb_build_object('evaluation_script', holdout->'test_eval_script')
        WHERE holdout ? 'test_eval_script'
    """)
    op.alter_column(
        "tasks", "test_eval_script", new_column_name="evaluation_script"
    )
    op.alter_column(
        "tasks", "test_sample_submission", new_column_name="sample_submission"
    )
