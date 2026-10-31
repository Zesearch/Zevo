"""Scoring capability is exposed independently of experiment ordering."""
from __future__ import annotations

import pytest

from zevo.contracts.orchestrator import (
    VERIFIABLE_REWARD_METRICS,
    OrchestratePayload,
    verifiable_reward_available,
)


# ─────────────────────── the scoring-contract signal ────────────────────────


@pytest.mark.parametrize("metric", sorted(VERIFIABLE_REWARD_METRICS))
def test_builtin_correctness_metrics_are_verifiable(metric: str) -> None:
    assert verifiable_reward_available(metric, "builtin") is True


@pytest.mark.parametrize("metric", ["f1", "token_f1", "bleu", "rouge_l"])
def test_overlap_metrics_are_not_verifiable(metric: str) -> None:
    # Graded/overlap metrics score partial similarity, not correctness: a +1/0
    # verifier would misrepresent them.
    assert verifiable_reward_available(metric, "builtin") is False


def test_custom_evaluator_is_never_verifiable() -> None:
    # A custom evaluator is opaque, so accuracy-by-name under it proves nothing.
    assert verifiable_reward_available("accuracy", "custom") is False


def test_signal_is_case_and_whitespace_insensitive() -> None:
    assert verifiable_reward_available("  Accuracy ", " BuiltIn ") is True


def _payload(metric: str, metric_type: str) -> OrchestratePayload:
    return OrchestratePayload(
        task_objective="obj",
        agent_objective="obj",
        user_request={
            "task_objective": "obj",
            "metric": metric,
            "metric_direction": "max",
            "metric_type": metric_type,
            "training_method": "",
            "dataset": "",
            "base_model": "",
            "test_set": "t.csv",
            "test_sample_submission": "s.csv",
            "constraints": [],
        },
        trigger={"type": "ticket_completed"},
        budget={"max_cost_usd": 0.0, "spent_usd": 0.0, "remaining_usd": 0.0},
        runtime={
            "gpu_provider": "instance",
            "num_gpus": 1,
            "generation_backend": "vllm",
        },
    )


def test_supervisor_context_surfaces_the_signal_true() -> None:
    payload = _payload("accuracy", "builtin")
    assert payload.verifiable_reward_available is True
    # It must ride along in the serialized context the supervisor actually sees.
    assert payload.model_dump()["verifiable_reward_available"] is True


def test_supervisor_context_surfaces_the_signal_false() -> None:
    payload = _payload("rouge_l", "builtin")
    assert payload.verifiable_reward_available is False
    assert payload.model_dump()["verifiable_reward_available"] is False
