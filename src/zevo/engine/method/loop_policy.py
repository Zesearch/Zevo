"""Zevo model-improvement loop policy — when to stop iterating.

**Advisory, not the decision.** The orchestrator stops when its own Stop policy says
to (playbook/agents/orchestrator/platform.md, "Stop policy"), and
nothing in the harness enforces this module's verdict — it exists so
`GET /runs/{id}/should-stop` can answer "would the policy stop here?" without
re-deriving the rules.

That makes agreement with the prose a requirement, not a nicety. Two verdicts
here used to contradict it outright: crossing an explicit user regression
tolerance recommended `mark_failed` instead of retaining the best usable model,
and exhausting `iteration_budget` without reaching `stop_threshold` also
recommended `mark_failed`, where the documented policy is `mark_done` with the
best model so far. Both now match the documented policy.

Stop triggers (any one fires → stop):
  - **target_hit**:        latest score reaches run.stop_threshold in the
                           declared direction (≥ for max, ≤ for min)
  - **budget_exhausted**:  iterations_completed ≥ run.iteration_budget
  - **cost_exhausted**:    budget snapshot says over_budget
  - **time_exhausted**:    runtime snapshot says the Run reached its wall-clock cap
  - **regressed**:         latest score worsened by more than the explicit
                           user-set run.regression_tolerance vs prior best
  - **plateau**:           2+ iterations in a row moved the score by
                           less than run.min_delta_per_iter

**Plateau is opt-in and off by default** (`min_delta_per_iter = 0.0`), and
deliberately has no default value. Any number picked here would be picked
without knowing how many rows the run validates on, and the two cannot be
separated: on a 150-row set a 0.5-point threshold is under one row, so it
fires on a coin flip. Whether the curve has flattened is left to the
orchestrator, which is told `validation_rows` and asked to judge — see
"Stop policy" in playbook/agents/orchestrator/platform.md. Setting the knob
is the USER overriding that judgement with their own threshold, not the system
supplying one.

If none fire, another experiment is permitted; the Orchestrator may still stop
based on expected benefit and cost. Returns the trigger code + a
human-readable reason the orchestrator includes in its mark_done/
mark_failed summary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zevo.engine.method.score_direction import MetricDirection, best_score, improvement


StopTrigger = Literal[
    "target_hit", "budget_exhausted", "cost_exhausted", "time_exhausted",
    "regressed", "plateau", "none",
]


@dataclass
class StopDecision:
    should_stop: bool
    trigger: StopTrigger
    reason: str
    recommended_action: Literal["mark_done", "mark_failed", "continue"]


def decide(
    *,
    scores: list[float],            # ordered list of trained validation scores
    iterations_completed: int,
    iteration_budget: int,
    stop_threshold: float | None,
    metric_direction: MetricDirection = "max",
    min_delta_per_iter: float = 0.0,
    regression_tolerance: float = 0.0,
    over_budget: bool = False,
    over_time_limit: bool = False,
) -> StopDecision:
    """Compute the stop decision from current loop state.

    Pure function — no DB, no IO. Easy to test, easy for the orchestrator
    to call via /runs/{id}/should-stop.
    """
    if over_time_limit:
        best = best_score(scores, metric_direction)
        return StopDecision(
            should_stop=True, trigger="time_exhausted",
            reason="run reached its wall-clock limit; refusing to start more work",
            recommended_action="mark_done" if best is not None else "mark_failed",
        )

    if over_budget:
        best = best_score(scores, metric_direction)
        return StopDecision(
            should_stop=True, trigger="cost_exhausted",
            reason="run is over its $ cap; refusing to spawn more tickets",
            recommended_action="mark_done" if best is not None else "mark_failed",
        )

    latest = scores[-1] if scores else None

    target_hit = latest is not None and stop_threshold is not None and (
        latest >= stop_threshold
        if metric_direction == "max"
        else latest <= stop_threshold
    )
    if target_hit:
        return StopDecision(
            should_stop=True, trigger="target_hit",
            reason=f"stop threshold {stop_threshold:.4f} hit (latest={latest:.4f})",
            recommended_action="mark_done",
        )

    if iteration_budget > 0 and iterations_completed >= iteration_budget:
        best = best_score(scores, metric_direction)
        # `mark_done` whether or not the target was reached: the run produced
        # and registered models, and "we ran out of iterations short of the
        # target" is a result, not a failure. `mark_failed` is for a run that
        # cannot continue.
        recommended = "mark_done" if best is not None else "mark_failed"
        best_text = f"{best:.4f}" if best is not None else "none"
        return StopDecision(
            should_stop=True, trigger="budget_exhausted",
            reason=(
                f"iteration_budget exhausted ({iterations_completed}/{iteration_budget}); "
                f"best={best_text}"
                + (f" threshold={stop_threshold:.4f} not hit" if stop_threshold is not None else "")
            ),
            recommended_action=recommended,
        )

    # Regression: latest dropped vs prior best by more than tolerance
    if len(scores) >= 2 and regression_tolerance > 0:
        prior_best = best_score(scores[:-1], metric_direction)
        regression = -improvement(latest, prior_best, metric_direction)
        if regression > regression_tolerance:
            return StopDecision(
                should_stop=True, trigger="regressed",
                reason=(
                    f"score regressed: prior best {prior_best:.4f}, "
                    f"latest {latest:.4f} (regression {regression:.4f} > "
                    f"tolerance {regression_tolerance:.4f})"
                ),
                # `mark_done`, keeping the best model — not `mark_failed`. The
                # run worked; a direction stopped paying. Only a run that cannot
                # continue is a failure.
                recommended_action="mark_done",
            )

    # Plateau: 2+ consecutive moves smaller than min_delta_per_iter
    if min_delta_per_iter > 0 and len(scores) >= 3:
        d1 = abs(scores[-1] - scores[-2])
        d2 = abs(scores[-2] - scores[-3])
        if d1 < min_delta_per_iter and d2 < min_delta_per_iter:
            return StopDecision(
                should_stop=True, trigger="plateau",
                reason=(
                    f"plateau: last two deltas {d1:.4f}, {d2:.4f} both "
                    f"below min_delta_per_iter {min_delta_per_iter:.4f}"
                ),
                recommended_action="mark_done",
            )

    return StopDecision(
        should_stop=False, trigger="none",
        reason="no configured stop trigger fired; the Orchestrator may continue or stop on expected value",
        recommended_action="continue",
    )
