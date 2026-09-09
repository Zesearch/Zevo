"""Name the Capybara metric after what its scorer computes.

Revision ID: e8f9a0b1c2d3
Revises: d7e8f9a0b1c2
"""

from alembic import op


revision = "e8f9a0b1c2d3"
down_revision = "d7e8f9a0b1c2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE tasks SET metric = 'token_f1' WHERE name = 'capybara'")
    op.execute("UPDATE runs SET metric = 'token_f1' WHERE task_name = 'capybara'")
    op.execute(
        """
        UPDATE score_events
        SET metric_name = 'token_f1'
        WHERE run_id IN (SELECT id FROM runs WHERE task_name = 'capybara')
        """
    )
    op.execute(
        """
        UPDATE registry_models
        SET metric = 'token_f1',
            eval = (COALESCE(eval, '{}'::jsonb) - 'accuracy' - 'conversation_token_f1')
              || jsonb_build_object(
                   'token_f1',
                   COALESCE(
                     eval -> 'token_f1',
                     eval -> 'conversation_token_f1',
                     eval -> 'accuracy',
                     eval -> 'score'
                   )
                 )
        WHERE run_id IN (SELECT id FROM runs WHERE task_name = 'capybara')
        """
    )
    op.execute(
        """
        UPDATE tickets
        SET payload = CASE
          WHEN payload ? 'metric'
            THEN jsonb_set(payload, '{metric}', '"token_f1"'::jsonb)
          ELSE payload
        END
        WHERE run_id IN (SELECT id FROM runs WHERE task_name = 'capybara')
        """
    )
    op.execute(
        """
        UPDATE tickets
        SET payload = jsonb_set(payload, '{user_request,metric}', '"token_f1"'::jsonb)
        WHERE run_id IN (SELECT id FROM runs WHERE task_name = 'capybara')
          AND payload #> '{user_request,metric}' IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("UPDATE tasks SET metric = 'accuracy' WHERE name = 'capybara'")
    op.execute("UPDATE runs SET metric = 'accuracy' WHERE task_name = 'capybara'")
    op.execute(
        """
        UPDATE score_events
        SET metric_name = 'accuracy'
        WHERE run_id IN (SELECT id FROM runs WHERE task_name = 'capybara')
        """
    )
    op.execute(
        """
        UPDATE registry_models
        SET metric = 'accuracy',
            eval = (COALESCE(eval, '{}'::jsonb) - 'token_f1')
              || jsonb_build_object('accuracy', COALESCE(eval -> 'token_f1', eval -> 'score'))
        WHERE run_id IN (SELECT id FROM runs WHERE task_name = 'capybara')
        """
    )
    op.execute(
        """
        UPDATE tickets
        SET payload = CASE
          WHEN payload ? 'metric'
            THEN jsonb_set(payload, '{metric}', '"accuracy"'::jsonb)
          ELSE payload
        END
        WHERE run_id IN (SELECT id FROM runs WHERE task_name = 'capybara')
        """
    )
    op.execute(
        """
        UPDATE tickets
        SET payload = jsonb_set(payload, '{user_request,metric}', '"accuracy"'::jsonb)
        WHERE run_id IN (SELECT id FROM runs WHERE task_name = 'capybara')
          AND payload #> '{user_request,metric}' IS NOT NULL
        """
    )
