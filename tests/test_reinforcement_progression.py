"""Pin the verifiable-reward method-progression policy and its signal.

The orchestrator installs GRPO/RFT but historically never reached them: the SFT
branch read as "not yet exhausted" and reinforcement as a last resort. These
tests pin the fix — a scoring-contract signal that a verifiable +1/0 correctness
reward exists, and a pure advisory policy that promotes SFT -> RFT -> GRPO once a
reasonable SFT baseline is in place, while respecting user method pins and
preserving prior behavior when no verifiable reward exists.

Pure-function tests — no DB.
"""
from __future__ import annotations

import pytest

from zevo.contracts.orchestrator import (
    VERIFIABLE_REWARD_METRICS,
    OrchestratePayload,
    verifiable_reward_available,
)
from zevo.engine.method.loop_policy import ProgressionDecision, reinforcement_progression


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


# ─────────────────────── the progression policy ─────────────────────────────


def _decide(**kw) -> ProgressionDecision:
    base = dict(
        verifiable_reward_available=True,
        method_pinned=False,
        active_method="lora_sft",
        sft_iterations_completed=1,
    )
    base.update(kw)
    return reinforcement_progression(**base)


def test_promotes_rft_after_a_reasonable_sft_baseline() -> None:
    d = _decide(active_method="lora_sft", sft_iterations_completed=1)
    assert d.promote is True
    assert d.next_method == "rft"


def test_full_sft_baseline_also_promotes_rft() -> None:
    d = _decide(active_method="full_sft", sft_iterations_completed=2)
    assert d.promote is True
    assert d.next_method == "rft"


def test_rft_bridge_promotes_grpo_next() -> None:
    d = _decide(active_method="rft")
    assert d.promote is True
    assert d.next_method == "grpo"


def test_grpo_is_terminal_no_further_promotion() -> None:
    d = _decide(active_method="grpo")
    assert d.promote is False
    assert d.next_method == ""


def test_no_promotion_before_an_sft_baseline_exists() -> None:
    # Requirement 4: promote only *after* a reasonable SFT baseline.
    d = _decide(active_method="lora_sft", sft_iterations_completed=0)
    assert d.promote is False
    assert d.next_method == ""


def test_custom_min_baseline_threshold_is_respected() -> None:
    assert _decide(sft_iterations_completed=1, min_sft_iterations=2).promote is False
    assert _decide(sft_iterations_completed=2, min_sft_iterations=2).promote is True


def test_no_verifiable_reward_preserves_prior_behavior() -> None:
    # Requirement 4: absent the reward, reinforcement stays gated behind ordinary
    # branch exhaustion — this policy declines to promote.
    d = _decide(verifiable_reward_available=False)
    assert d.promote is False
    assert d.next_method == ""


def test_user_method_pin_removes_the_transition_level() -> None:
    # Requirement 3: a hard pin removes the Method-transition level; the policy
    # must NOT override it, even with a verifiable reward and a baseline.
    d = _decide(method_pinned=True, active_method="lora_sft", sft_iterations_completed=3)
    assert d.promote is False
    assert d.next_method == ""
    assert "pin" in d.reason.lower()


def test_a_non_sft_non_rft_active_method_is_not_promoted() -> None:
    # e.g. a preference method: not on the SFT -> RFT -> GRPO ladder.
    d = _decide(active_method="dpo")
    assert d.promote is False
    assert d.next_method == ""


def test_active_method_is_case_insensitive() -> None:
    d = _decide(active_method="LoRA_SFT", sft_iterations_completed=1)
    assert d.promote is True
    assert d.next_method == "rft"
