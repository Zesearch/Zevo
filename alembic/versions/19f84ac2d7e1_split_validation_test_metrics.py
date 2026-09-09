"""Give Validation and Test independent metric contracts.

Revision ID: 19f84ac2d7e1
Revises: 0a9b8c7d6e5f
Create Date: 2026-08-29
"""
from alembic import op
import sqlalchemy as sa


revision = "19f84ac2d7e1"
down_revision = "0a9b8c7d6e5f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for column in (
        sa.Column(
            "validation_metric_type", sa.String(length=16), nullable=False,
            server_default="builtin",
        ),
        sa.Column(
            "validation_metric", sa.String(length=64), nullable=False,
            server_default="token_f1",
        ),
        sa.Column(
            "validation_metric_direction", sa.String(length=3), nullable=False,
            server_default="max",
        ),
        sa.Column(
            "validation_evaluation_script", sa.Text(), nullable=False,
            server_default="",
        ),
        sa.Column(
            "validation_evaluator_sha256", sa.String(length=64), nullable=False,
            server_default="",
        ),
    ):
        op.add_column("task_settings", column)
    op.create_check_constraint(
        "ck_task_settings_validation_metric_type", "task_settings",
        "validation_metric_type IN ('builtin', 'custom')",
    )
    op.create_check_constraint(
        "ck_task_settings_validation_metric_direction", "task_settings",
        "validation_metric_direction IN ('max', 'min')",
    )
    # Existing settings preserve their former behavior explicitly: before this
    # revision both lanes necessarily used the Task contract.
    op.execute("""
        UPDATE task_settings AS s
        SET validation_metric_type = t.metric_type,
            validation_metric = t.metric,
            validation_metric_direction = t.metric_direction,
            validation_evaluation_script = t.evaluation_script,
            validation_evaluator_sha256 = t.evaluator_sha256
        FROM tasks AS t
        WHERE t.name = s.task_name
    """)

    op.add_column(
        "runs",
        sa.Column(
            "validation_metric", sa.String(length=64), nullable=False,
            server_default="token_f1",
        ),
    )
    op.add_column(
        "runs",
        sa.Column(
            "validation_metric_direction", sa.String(length=3), nullable=False,
            server_default="max",
        ),
    )
    op.create_check_constraint(
        "ck_runs_validation_metric_direction", "runs",
        "validation_metric_direction IN ('max', 'min')",
    )
    op.execute("""
        UPDATE runs
        SET validation_metric = metric,
            validation_metric_direction = metric_direction
    """)

    # The previous shape stored one implementation in Run.holdout. Copy it to
    # both explicit lanes once, then remove the ambiguous generic keys.
    op.execute("""
        UPDATE runs
        SET holdout =
            (holdout - 'metric_type' - 'evaluation_script' - 'evaluator_sha256')
            || jsonb_build_object(
                'test_metric_type', COALESCE(holdout->>'metric_type', 'builtin'),
                'test_evaluation_script', COALESCE(holdout->>'evaluation_script', ''),
                'test_evaluator_sha256', COALESCE(holdout->>'evaluator_sha256', ''),
                'validation_metric_type', COALESCE(holdout->>'metric_type', 'builtin'),
                'validation_evaluation_script', COALESCE(holdout->>'evaluation_script', ''),
                'validation_evaluator_sha256', COALESCE(holdout->>'evaluator_sha256', '')
            )
    """)

    # Supervisor requests keep generic metric fields as the Task/Test contract;
    # the new explicit fields carry the Validation contract.
    op.execute("""
        UPDATE tickets
        SET payload = jsonb_set(
            payload,
            '{user_request}',
            (payload->'user_request') || jsonb_build_object(
                'validation_metric_type', COALESCE(
                    payload#>>'{user_request,metric_type}', 'builtin'
                ),
                'validation_metric', COALESCE(
                    payload#>>'{user_request,metric}', 'token_f1'
                ),
                'validation_metric_direction', COALESCE(
                    payload#>>'{user_request,metric_direction}', 'max'
                ),
                'validation_evaluation_script', COALESCE(
                    payload#>>'{user_request,evaluation_script}', ''
                ),
                'validation_evaluator_sha256', COALESCE(
                    payload#>>'{user_request,evaluator_sha256}', ''
                )
            )
        )
        WHERE jsonb_typeof(payload->'user_request') = 'object'
    """)


def downgrade() -> None:
    op.execute("""
        UPDATE runs
        SET holdout =
            (holdout
                - 'test_metric_type'
                - 'test_evaluation_script'
                - 'test_evaluator_sha256'
                - 'validation_metric_type'
                - 'validation_evaluation_script'
                - 'validation_evaluator_sha256')
            || jsonb_build_object(
                'metric_type', COALESCE(holdout->>'test_metric_type', 'builtin'),
                'evaluation_script', COALESCE(holdout->>'test_evaluation_script', ''),
                'evaluator_sha256', COALESCE(holdout->>'test_evaluator_sha256', '')
            )
    """)
    op.execute("""
        UPDATE tickets
        SET payload = jsonb_set(
            payload,
            '{user_request}',
            (payload->'user_request')
                - 'validation_metric_type'
                - 'validation_metric'
                - 'validation_metric_direction'
                - 'validation_evaluation_script'
                - 'validation_evaluator_sha256'
        )
        WHERE jsonb_typeof(payload->'user_request') = 'object'
    """)
    op.drop_constraint(
        "ck_runs_validation_metric_direction", "runs", type_="check"
    )
    op.drop_column("runs", "validation_metric_direction")
    op.drop_column("runs", "validation_metric")
    op.drop_constraint(
        "ck_task_settings_validation_metric_direction", "task_settings",
        type_="check",
    )
    op.drop_constraint(
        "ck_task_settings_validation_metric_type", "task_settings",
        type_="check",
    )
    for name in (
        "validation_evaluator_sha256",
        "validation_evaluation_script",
        "validation_metric_direction",
        "validation_metric",
        "validation_metric_type",
    ):
        op.drop_column("task_settings", name)
