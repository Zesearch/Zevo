"""Replace lane-specific metric keys in stored Run state.

Revision ID: 0a9b8c7d6e5f
Revises: b0c1d2e3f4a5
Create Date: 2026-08-29
"""
from alembic import op


revision = "0a9b8c7d6e5f"
down_revision = "b0c1d2e3f4a5"
branch_labels = None
depends_on = None


_BUILTIN_SQL = (
    "'accuracy', 'exact_match', 'f1', 'token_f1', 'bleu', 'rouge_l'"
)


def upgrade() -> None:
    # Run.holdout is the engine-owned complete scoring specification. Replace
    # the two lane implementations with one Task implementation and remove the
    # old binding flag in the same write; runtime code has no compatibility
    # fallback for any of these keys.
    op.execute(f"""
        UPDATE runs
        SET holdout =
            (holdout
                - 'test_eval_script'
                - 'validation_eval_script'
                - 'validation_needs_own_scorer')
            || jsonb_build_object(
                'metric_type', CASE
                    WHEN lower(metric) IN ({_BUILTIN_SQL}) THEN 'builtin'
                    ELSE 'custom'
                END,
                'evaluation_script', CASE
                    WHEN lower(metric) IN ({_BUILTIN_SQL}) THEN ''
                    ELSE COALESCE(
                        holdout->>'evaluation_script',
                        holdout->>'test_eval_script',
                        holdout->>'validation_eval_script',
                        ''
                    )
                END,
                'evaluator_sha256', COALESCE(holdout->>'evaluator_sha256', ''),
                'validation_needs_metric_binding', COALESCE(
                    holdout->'validation_needs_metric_binding',
                    holdout->'validation_needs_own_scorer',
                    'false'::jsonb
                )
            )
        WHERE holdout ? 'test_eval_script'
           OR holdout ? 'validation_eval_script'
           OR holdout ? 'validation_needs_own_scorer'
    """)

    # Supervisor payloads are the Agent-visible request and intentionally do
    # not contain evaluator code. Stamp the new discriminator and remove both
    # obsolete request fields instead of teaching UserRequest to accept them.
    op.execute(f"""
        UPDATE tickets
        SET payload = jsonb_set(
            payload,
            '{{user_request}}',
            ((payload->'user_request')
                - 'test_eval_script'
                - 'validation_eval_script')
            || jsonb_build_object(
                'metric_type', CASE
                    WHEN lower(payload#>>'{{user_request,metric}}')
                         IN ({_BUILTIN_SQL}) THEN 'builtin'
                    ELSE 'custom'
                END,
                'evaluation_script', '',
                'evaluator_sha256', ''
            )
        )
        WHERE jsonb_typeof(payload->'user_request') = 'object'
          AND (
              (payload->'user_request') ? 'test_eval_script'
              OR (payload->'user_request') ? 'validation_eval_script'
          )
    """)


def downgrade() -> None:
    # Downgrade restores the former shape, not discarded lane-specific code.
    # Both legacy fields receive the single implementation because divergence
    # is no longer representable after this migration.
    op.execute("""
        UPDATE tickets
        SET payload = jsonb_set(
            payload,
            '{user_request}',
            ((payload->'user_request')
                - 'metric_type'
                - 'evaluation_script'
                - 'evaluator_sha256')
            || jsonb_build_object(
                'test_eval_script', '',
                'validation_eval_script', ''
            )
        )
        WHERE jsonb_typeof(payload->'user_request') = 'object'
    """)
    op.execute("""
        UPDATE runs
        SET holdout =
            (holdout
                - 'metric_type'
                - 'evaluation_script'
                - 'evaluator_sha256'
                - 'validation_needs_metric_binding')
            || jsonb_build_object(
                'test_eval_script', COALESCE(holdout->>'evaluation_script', ''),
                'validation_eval_script', COALESCE(holdout->>'evaluation_script', ''),
                'validation_needs_own_scorer', COALESCE(
                    holdout->'validation_needs_metric_binding', 'false'::jsonb
                )
            )
    """)
