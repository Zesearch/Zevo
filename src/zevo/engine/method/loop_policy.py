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

If none fire → keep iterating. Returns the trigger code + a
human-readable reason the orchestrator includes in its mark_done/
mark_failed summary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zevo.contracts.training_methods import REINFORCEMENT_PROGRESSION, SFT_METHODS
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
        reason="loop should continue: no stop trigger fired",
        recommended_action="continue",
    )


# ───────────────────────── method-progression policy ─────────────────────────
#
# **Advisory, not the decision** — the same doctrine as `decide` above. The
# orchestrator (playbook/agents/orchestrator/{goal,platform}.md) owns the actual
# Method-transition choice and must still record the `branch_transition` audit.
# This pure function exists so that "should the search progress from SFT to a
# verifiable-reward method now?" has one place agreed with the prose, rather
# than being re-derived in an LLM heartbeat.
#
# The problem it removes: GRPO/RFT are installed but never reached because the
# search treats the SFT branch as "not yet exhausted" and reinforcement as a
# last resort. When a verifiable correctness reward exists, the SOTA move is
# SFT(+distilled) -> RFT -> GRPO, promoted deliberately *after a reasonable SFT
# baseline* — not gated behind full SFT-branch exhaustion.


@dataclass
class ProgressionDecision:
    promote: bool
    # The single next method to promote ("rft" then "grpo"), or "" when nothing
    # is promoted. Never a set: the progression is ordered.
    next_method: str
    reason: str


def reinforcement_progression(
    *,
    verifiable_reward_available: bool,
    method_pinned: bool,
    active_method: str,
    sft_iterations_completed: int,
    min_sft_iterations: int = 1,
) -> ProgressionDecision:
    """Whether to promote a verifiable-reward method as the next Method lever.

    Pure function — no DB, no IO. Inputs the orchestrator already holds:

      - `verifiable_reward_available`: the scoring-contract signal surfaced on
        the supervisor context (a built-in correctness metric implies a +1/0
        reward). See `zevo.contracts.orchestrator.verifiable_reward_available`.
      - `method_pinned`: the user hard-pinned `training_method` via
        `decision_pins`. A pin removes the Method-transition level entirely, so
        this policy NEVER overrides it (requirement 3).
      - `active_method`: the current training-method family.
      - `sft_iterations_completed`: how many SFT-family candidates this Run has
        already measured on Validation — the evidence of a "reasonable baseline".
      - `min_sft_iterations`: how many SFT candidates make a baseline reasonable
        (default 1: at least one measured SFT model).

    Promotion is principled, not forced (requirement 4): it fires only when a
    verifiable reward exists AND an SFT baseline is established, and when absent
    the function preserves the existing SFT-branch-exhaustion behavior by
    declining to promote.
    """
    active = (active_method or "").strip().lower()

    # (3) A user method pin freezes the method family: the transition level is
    # removed and this policy must not override it.
    if method_pinned:
        return ProgressionDecision(
            False, "",
            "training_method is user-pinned; the Method-transition level is "
            "removed and reinforcement progression must not override it",
        )

    # (4) No verifiable reward -> preserve existing behavior: reinforcement stays
    # gated behind ordinary SFT-branch exhaustion, not promoted here.
    if not verifiable_reward_available:
        return ProgressionDecision(
            False, "",
            "no verifiable correctness reward for this task; reinforcement "
            "methods are not promoted (ordinary branch-exhaustion order applies)",
        )

    # Already on the reinforcement ladder: advance RFT -> GRPO, stop at GRPO.
    if active == "grpo":
        return ProgressionDecision(
            False, "",
            "already on GRPO, the terminal verifiable-reward lever",
        )
    if active == "rft":
        return ProgressionDecision(
            True, "grpo",
            "verifiable reward available and the RFT bridge is established; "
            "promote GRPO as the next lever (SFT -> RFT -> GRPO)",
        )

    # On an SFT baseline: promote RFT once the baseline is reasonable, without
    # waiting for the SFT branch to be fully exhausted.
    if active in SFT_METHODS:
        if sft_iterations_completed >= max(1, min_sft_iterations):
            return ProgressionDecision(
                True, REINFORCEMENT_PROGRESSION[0],
                "verifiable reward available and a reasonable SFT baseline is "
                f"established ({sft_iterations_completed} SFT iteration(s)); "
                "promote RFT as the next Method lever on the SFT -> RFT -> GRPO "
                "progression, rather than waiting for full SFT-branch exhaustion",
            )
        return ProgressionDecision(
            False, "",
            "verifiable reward available but no reasonable SFT baseline yet "
            f"({sft_iterations_completed}/{max(1, min_sft_iterations)} SFT "
            "iterations); establish an SFT baseline before promoting RFT/GRPO",
        )

    # Some other active method (e.g. a preference method): not on the
    # SFT -> RFT -> GRPO ladder, so this policy does not promote from it.
    return ProgressionDecision(
        False, "",
        f"active method {active!r} is not an SFT baseline; the verifiable-reward "
        "progression is not applicable from here",
    )
