"""Use generic scores and the canonical generation_backend name.

Revision ID: d7e8f9a0b1c2
Revises: c6a1e9f2b470
"""

from alembic import op
import sqlalchemy as sa


revision = "d7e8f9a0b1c2"
down_revision = "c6a1e9f2b470"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("metric", sa.String(64), nullable=True))
    op.add_column("runs", sa.Column("metric", sa.String(64), nullable=True))
    op.add_column("registry_models", sa.Column("metric", sa.String(64), nullable=True))
    op.add_column("registry_models", sa.Column("metric_direction", sa.String(3), nullable=True))
    op.execute("UPDATE tasks SET metric = 'accuracy'")
    op.execute("UPDATE runs SET metric = 'accuracy'")
    op.execute("UPDATE registry_models SET metric = 'accuracy'")
    op.execute("UPDATE registry_models SET metric_direction = 'max'")
    op.alter_column("tasks", "metric", nullable=False)
    op.alter_column("runs", "metric", nullable=False)
    op.alter_column("registry_models", "metric", nullable=False)
    op.alter_column("registry_models", "metric_direction", nullable=False)

    op.alter_column("runs", "framework", new_column_name="generation_backend")
    op.alter_column("runs", "headline_accuracy", new_column_name="headline_score")
    op.alter_column("runs", "best_accuracy", new_column_name="best_score")
    op.alter_column("runs", "test_accuracy", new_column_name="test_score")
    op.alter_column("runs", "best_test_accuracy", new_column_name="best_test_score")
    op.alter_column("score_events", "accuracy", new_column_name="score")

    for column in ("headline_score", "best_score", "test_score", "best_test_score"):
        op.execute(f"UPDATE runs SET {column} = NULL WHERE {column} = -1")
        op.alter_column("runs", column, nullable=True, server_default=None)

    # History is an engine-owned JSON array. Move its two numeric facts to the
    # canonical keys so current code has exactly one shape to read.
    op.execute(
        """
        UPDATE runs
        SET history = COALESCE((
          SELECT json_agg(
            (elem::jsonb - 'accuracy' - 'test_accuracy')
            || CASE WHEN elem::jsonb ? 'accuracy'
                    THEN jsonb_build_object('score', elem::jsonb -> 'accuracy')
                    ELSE '{}'::jsonb END
            || CASE WHEN elem::jsonb ? 'test_accuracy'
                    THEN jsonb_build_object('test_score', elem::jsonb -> 'test_accuracy')
                    ELSE '{}'::jsonb END
          )
          FROM json_array_elements(COALESCE(runs.history, '[]'::json)) AS elem
        ), '[]'::json)
        """
    )
    op.execute(
        """
        UPDATE registry_models
        SET eval = (COALESCE(eval, '{}'::jsonb) - 'accuracy' - 'headline_accuracy')
          || CASE
               WHEN COALESCE(eval, '{}'::jsonb) ? 'score' THEN '{}'::jsonb
               WHEN COALESCE(eval, '{}'::jsonb) ? 'headline_accuracy'
                 THEN jsonb_build_object('score', eval -> 'headline_accuracy')
               WHEN COALESCE(eval, '{}'::jsonb) ? 'accuracy'
                 THEN jsonb_build_object('score', eval -> 'accuracy')
               ELSE '{}'::jsonb
             END
        """
    )

    op.create_check_constraint(
        "ck_runs_mode", "runs",
        "mode IN ('full_pipeline', 'customized_pipeline', 'single_stage')",
    )
    op.create_check_constraint(
        "ck_runs_status", "runs",
        "status IN ('planning', 'running', 'success', 'degraded', 'failed', 'halted', 'cancelled')",
    )
    op.create_check_constraint(
        "ck_tickets_status", "tickets",
        "status IN ('queued', 'running', 'awaiting_input', 'succeeded', 'degraded', 'failed', 'skipped', 'cancelled')",
    )
    op.create_check_constraint(
        "ck_tickets_input_format", "tickets", "input_format IN ('typed', 'freeform')",
    )
    op.create_check_constraint(
        "ck_tickets_lane", "tickets", "lane IN ('optimization', 'held_out_test')",
    )
    op.create_check_constraint(
        "ck_runs_generation_backend", "runs",
        "generation_backend IN ('hf', 'vllm')",
    )
    op.create_check_constraint(
        "ck_runs_gpu_provider", "runs",
        "gpu_provider IN ('instance', 'cluster', 'cloud')",
    )
    op.create_check_constraint(
        "ck_score_events_split", "score_events",
        "split IN ('validation', 'test')",
    )
    op.create_check_constraint(
        "ck_score_events_source", "score_events",
        "source IN ('baseline', 'trained', 'validation')",
    )
    op.create_check_constraint(
        "ck_registry_models_metric_direction", "registry_models",
        "metric_direction IN ('max', 'min')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_registry_models_metric_direction", "registry_models", type_="check")
    op.drop_constraint("ck_score_events_source", "score_events", type_="check")
    op.drop_constraint("ck_score_events_split", "score_events", type_="check")
    op.drop_constraint("ck_runs_gpu_provider", "runs", type_="check")
    op.drop_constraint("ck_runs_generation_backend", "runs", type_="check")
    op.drop_constraint("ck_tickets_lane", "tickets", type_="check")
    op.drop_constraint("ck_tickets_input_format", "tickets", type_="check")
    op.drop_constraint("ck_tickets_status", "tickets", type_="check")
    op.drop_constraint("ck_runs_status", "runs", type_="check")
    op.drop_constraint("ck_runs_mode", "runs", type_="check")
    op.alter_column("score_events", "score", new_column_name="accuracy")
    op.alter_column("runs", "best_test_score", new_column_name="best_test_accuracy")
    op.alter_column("runs", "test_score", new_column_name="test_accuracy")
    op.alter_column("runs", "best_score", new_column_name="best_accuracy")
    op.alter_column("runs", "headline_score", new_column_name="headline_accuracy")
    op.alter_column("runs", "generation_backend", new_column_name="framework")
    op.drop_column("registry_models", "metric")
    op.drop_column("registry_models", "metric_direction")
    op.drop_column("runs", "metric")
    op.drop_column("tasks", "metric")
