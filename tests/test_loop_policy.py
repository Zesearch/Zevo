"""Pin the Zevo model-improvement loop stop-decider.

Each test exercises ONE trigger in isolation so a regression points
at exactly which decision broke. Tests are pure-function — no DB.
"""
from __future__ import annotations

import pytest

from zevo.engine.method.loop_policy import decide


# ─────────────────────────── single triggers ────────────────────────────────


def test_target_hit_stops_immediately() -> None:
    d = decide(
        scores=[0.50, 0.70],
        iterations_completed=2, iteration_budget=10,
        stop_threshold=0.65,
    )
    assert d.should_stop is True
    assert d.trigger == "target_hit"
    assert d.recommended_action == "mark_done"


def test_min_target_is_reached_at_or_below_threshold() -> None:
    d = decide(
        scores=[0.30, 0.12],
        iterations_completed=2, iteration_budget=10,
        stop_threshold=0.15,
        metric_direction="min",
    )
    assert d.should_stop is True
    assert d.trigger == "target_hit"


def test_min_regression_is_an_increase_from_the_prior_low() -> None:
    d = decide(
        scores=[0.30, 0.10, 0.24],
        iterations_completed=3, iteration_budget=10,
        stop_threshold=0.05,
        metric_direction="min",
        regression_tolerance=0.10,
    )
    assert d.should_stop is True
    assert d.trigger == "regressed"


def test_budget_exhausted_with_no_target_marks_done() -> None:
    d = decide(
        scores=[0.20, 0.35, 0.40],
        iterations_completed=3, iteration_budget=3,
        stop_threshold=None,
    )
    assert d.should_stop is True
    assert d.trigger == "budget_exhausted"
    # No explicit target → "as high as you can get" → done is correct
    assert d.recommended_action == "mark_done"


def test_budget_exhausted_with_unmet_target_marks_done() -> None:
    """Out of iterations short of the target is a RESULT, not a failure.

    The orchestrator's own stop table says `mark_done` here ("Out of budget"),
    and this module is advisory to that table — a policy that told a run which
    trained and registered three models to mark itself failed contradicted the
    instructions the agent actually follows.
    """
    d = decide(
        scores=[0.20, 0.35, 0.40],
        iterations_completed=3, iteration_budget=3,
        stop_threshold=0.90,
    )
    assert d.should_stop is True
    assert d.trigger == "budget_exhausted"
    assert d.recommended_action == "mark_done"


def test_budget_exhausted_with_no_scores_marks_failed() -> None:
    """Nothing was ever measured — there is no best model to hand over."""
    d = decide(scores=[], iterations_completed=3, iteration_budget=3,
               stop_threshold=None)
    assert d.should_stop is True
    assert d.trigger == "budget_exhausted"
    assert d.recommended_action == "mark_failed"


def test_over_budget_with_score_keeps_usable_result() -> None:
    """The cap stops spending; it does not turn a valid result into a failure."""
    d = decide(
        scores=[0.95],
        iterations_completed=1, iteration_budget=10,
        stop_threshold=0.5,
        over_budget=True,
    )
    assert d.should_stop is True
    assert d.trigger == "cost_exhausted"
    assert d.recommended_action == "mark_done"


def test_over_budget_without_score_marks_failed() -> None:
    d = decide(
        scores=[], iterations_completed=0, iteration_budget=10,
        stop_threshold=0.5, over_budget=True,
    )
    assert d.should_stop is True
    assert d.trigger == "cost_exhausted"
    assert d.recommended_action == "mark_failed"


def test_time_limit_with_score_keeps_usable_result() -> None:
    d = decide(
        scores=[0.61], iterations_completed=1, iteration_budget=0,
        stop_threshold=None, over_time_limit=True,
    )
    assert d.should_stop is True
    assert d.trigger == "time_exhausted"
    assert d.recommended_action == "mark_done"


def test_time_limit_without_score_marks_failed() -> None:
    d = decide(
        scores=[], iterations_completed=0, iteration_budget=0,
        stop_threshold=None, over_time_limit=True,
    )
    assert d.should_stop is True
    assert d.trigger == "time_exhausted"
    assert d.recommended_action == "mark_failed"


def test_regression_triggers_when_drop_exceeds_tolerance() -> None:
    d = decide(
        scores=[0.30, 0.55, 0.40],
        iterations_completed=3, iteration_budget=10,
        stop_threshold=0.80,
        regression_tolerance=0.10,
    )
    # prior best = 0.55, latest = 0.40, drop = 0.15 > 0.10 tol
    assert d.should_stop is True
    assert d.trigger == "regressed"
    # `mark_done`, keeping the 0.55 model. A round that scored worse is what
    # exploration looks like — the orchestrator's table answers a regression
    # with "try a different direction", never with failing the run — and the
    # user-set `regression_tolerance` says when to stop looking, not when to
    # declare the run broken.
    assert d.recommended_action == "mark_done"


def test_regression_does_NOT_fire_within_tolerance() -> None:
    d = decide(
        scores=[0.30, 0.55, 0.50],
        iterations_completed=3, iteration_budget=10,
        stop_threshold=0.80,
        regression_tolerance=0.10,
    )
    # drop 0.05 < 0.10 tol → keep going
    assert d.should_stop is False
    assert d.trigger == "none"


def test_plateau_fires_after_two_small_moves() -> None:
    d = decide(
        scores=[0.30, 0.50, 0.502, 0.501],
        iterations_completed=4, iteration_budget=10,
        stop_threshold=0.80,
        min_delta_per_iter=0.01,
    )
    # last two deltas: 0.001, 0.002 both < 0.01 min → plateau
    assert d.should_stop is True
    assert d.trigger == "plateau"
    assert d.recommended_action == "mark_done"


def test_plateau_does_NOT_fire_with_one_small_move() -> None:
    """A single small delta after a big one is NOT a plateau — could be
    a temporary stall."""
    d = decide(
        scores=[0.30, 0.50, 0.55, 0.555],
        iterations_completed=4, iteration_budget=10,
        stop_threshold=0.80,
        min_delta_per_iter=0.01,
    )
    assert d.should_stop is False


# ─────────────────────────── precedence ─────────────────────────────────────


def test_trigger_precedence_target_then_budget_then_regression() -> None:
    """If multiple triggers would fire, target_hit wins (it's the
    happiest outcome), then budget, then regression, then plateau."""
    # Both target hit AND budget exhausted — target_hit wins
    d = decide(
        scores=[0.95], iterations_completed=10, iteration_budget=10,
        stop_threshold=0.90,
    )
    assert d.trigger == "target_hit"


# ───────────────────────────── edge cases ───────────────────────────────────


def test_no_scores_yet_doesnt_stop() -> None:
    d = decide(
        scores=[], iterations_completed=0, iteration_budget=10,
        stop_threshold=0.5,
    )
    assert d.should_stop is False
    assert d.trigger == "none"


def test_disabled_policy_never_fires_regression_or_plateau() -> None:
    """Default thresholds (0.0) = trigger off."""
    d = decide(
        scores=[0.30, 0.55, 0.40, 0.55, 0.55, 0.55],
        iterations_completed=6, iteration_budget=100,
        stop_threshold=0.90,
        min_delta_per_iter=0.0,           # disabled
        regression_tolerance=0.0,         # disabled
    )
    # Big regression (0.55 → 0.40) AND plateau (0.55 × 3) — neither
    # should fire with thresholds disabled.
    assert d.should_stop is False
    assert d.trigger == "none"


def test_zero_is_a_real_target() -> None:
    """Zero is valid; only None disables threshold stopping."""
    d = decide(
        scores=[0.10],
        iterations_completed=1, iteration_budget=5,
        stop_threshold=0.0,
    )
    assert d.trigger == "target_hit"
