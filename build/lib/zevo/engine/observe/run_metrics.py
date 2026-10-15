"""Derived numbers a run's history yields, shared by the routers that report them.

The baseline/best split lives here rather than in one router because two of them
need it — /runs for the Console's improvement figures and /leaderboard for the
per-task comparison — and two copies of this rule would drift.

`was_measured_on_heldout` is here for the same reason and is the more load
bearing of the two: it decides which runs are allowed to report a test score at
all, and the Tasks page and the leaderboard both have to answer that the same
way or one task shows two different "best" numbers.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select

from zevo.db.models import Run, ScoreEvent, Ticket
from zevo.engine.method.score_direction import (
    MetricDirection,
    best_score,
    improvement as score_improvement,
    is_better,
)


def upsert_history_fact(
    history: list[Any] | None,
    *,
    iteration: int,
    source: str,
    score: float,
    base_model: str = "",
    training_method: str = "",
    method_ids: list[str] | None = None,
    test_score: float | None = None,
    training_diagnostics: dict[str, Any] | None = None,
    generation_termination: dict[str, Any] | None = None,
) -> list[Any]:
    """Upsert the engine-owned facts for one iteration.

    Scores are produced by Evaluation and already recorded as ScoreEvents.
    Evaluation owns only these factual fields; the Orchestrator owns all four
    narrative fields and must fill them through the Journal PATCH contract.
    """
    rows = [dict(e) if isinstance(e, dict) else e for e in (history or [])]
    # Journal has exactly four narrative fields. `notes` was the former
    # catch-all and must not survive beside them as a fifth, ambiguous place to
    # explain an iteration.
    for row in rows:
        if isinstance(row, dict):
            row.pop("notes", None)
    key = (int(iteration), str(source))
    index = next((
        i for i, entry in enumerate(rows)
        if isinstance(entry, dict)
        and (int(entry.get("iteration", -1)), str(entry.get("source", ""))) == key
    ), -1)
    existing = dict(rows[index]) if index >= 0 else {}

    entry = {
        "iteration": int(iteration),
        "source": str(source),
        "method_ids": list(method_ids or existing.get("method_ids") or []),
        "training_method": training_method or str(existing.get("training_method") or ""),
        "base_model": base_model or str(existing.get("base_model") or ""),
        "score": float(score),
        "action": str(existing.get("action") or ""),
        "result": str(existing.get("result") or ""),
        "analysis": str(existing.get("analysis") or ""),
        "next": str(existing.get("next") or ""),
    }
    diagnostics = training_diagnostics or existing.get("training_diagnostics")
    if isinstance(diagnostics, dict) and diagnostics:
        entry["training_diagnostics"] = dict(diagnostics)
    termination = generation_termination or existing.get("generation_termination")
    if isinstance(termination, dict) and termination:
        entry["generation_termination"] = dict(termination)
    # The held-out value is engine-owned and never supplied by the supervisor.
    old_test = existing.get("test_score")
    if test_score is not None:
        entry["test_score"] = float(test_score)
    elif isinstance(old_test, (int, float)) and not isinstance(old_test, bool):
        entry["test_score"] = float(old_test)

    if index >= 0:
        rows[index] = entry
    else:
        rows.append(entry)
    return rows


JOURNAL_NARRATIVE_FIELDS = ("action", "result", "analysis", "next")


def validation_best_score(
    history: list[Any] | None, direction: MetricDirection = "max",
) -> float | None:
    """Best Validation score, including the untuned baseline probe."""
    return best_score(
        (
            float(entry["score"])
            for entry in (history or [])
            if isinstance(entry, dict)
            and isinstance(entry.get("score"), (int, float))
            and not isinstance(entry.get("score"), bool)
        ),
        direction,
    )


def incomplete_journal_entries(history: list[Any] | None) -> list[tuple[int, str]]:
    """Return factual iteration rows whose Orchestrator narrative is incomplete."""
    missing: list[tuple[int, str]] = []
    for entry in history or []:
        if not isinstance(entry, dict):
            continue
        source = str(entry.get("source") or "")
        if source not in ("baseline", "trained"):
            continue
        if any(not str(entry.get(field) or "").strip() for field in JOURNAL_NARRATIVE_FIELDS):
            missing.append((int(entry.get("iteration") or 0), source))
    return missing


def baseline_and_best(
    history: list[Any] | None, direction: MetricDirection = "max"
) -> tuple[float | None, float | None]:
    """Return the validation champion and its own model-lineage baseline.

    Entries carry `source: "baseline"` for the untuned probe; everything else is
    a training iteration. Missing / non-numeric scores are skipped; ``None``
    is the only absence marker because any finite number can be a valid score.
    """
    baselines: dict[str, float] = {}
    best_trained: float | None = None
    champion_base = ""
    for h in history or []:
        if not isinstance(h, dict):
            continue
        value = h.get("score")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        value = float(value)
        if h.get("source") == "baseline":
            baselines[str(h.get("base_model") or "")] = value
        elif is_better(value, best_trained, direction):
            best_trained = value
            champion_base = str(h.get("base_model") or "")
    baseline = baselines.get(champion_base) if best_trained is not None else None
    if best_trained is None and len(baselines) == 1:
        baseline = next(iter(baselines.values()))
    return baseline, best_trained


def improvement(
    history: list[Any] | None, direction: MetricDirection = "max"
) -> float | None:
    """Direction-normalized gain on the supplied metric's own scale.

    None when the run carries no baseline probe or never produced a trained
    score: without both ends there is no improvement to report, and any
    numeric sentinel collides with a real value — a genuine score change may
    itself be negative one.
    """
    baseline, best_trained = baseline_and_best(history, direction)
    if baseline is None or best_trained is None:
        return None
    return score_improvement(best_trained, baseline, direction)


def baseline_and_best_test(
    history: list[Any] | None, direction: MetricDirection = "max"
) -> tuple[float | None, float | None]:
    """The held-out pair: (champion lineage's baseline, champion's test).

    Not a copy of `baseline_and_best` with a different key. The "best" here is
    NOT the maximum test score — taking that would be choosing an iteration by
    its held-out result, which is selecting on the set the run is judged by, one
    level up from the loop doing it. It is the test score of whichever iteration
    won on VALIDATION, which is the same rule `Run.champion_test_score` follows.

    ``None`` for either end when the corresponding measurement did not land.
    """
    baseline_tests: dict[str, float] = {}
    best_val: float | None = None
    best_test: float | None = None
    champion_base = ""
    for h in history or []:
        if not isinstance(h, dict):
            continue
        raw_test = h.get("test_score")
        test = float(raw_test) if isinstance(raw_test, (int, float)) and not isinstance(raw_test, bool) else None
        if h.get("source") == "baseline":
            if test is not None:
                baseline_tests[str(h.get("base_model") or "")] = test
            continue
        val = h.get("score")
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        if is_better(float(val), best_val, direction):
            best_val, best_test = float(val), test
            champion_base = str(h.get("base_model") or "")
    baseline = baseline_tests.get(champion_base) if best_val is not None else None
    if best_val is None and len(baseline_tests) == 1:
        baseline = next(iter(baseline_tests.values()))
    return baseline, best_test


def improvement_test(
    history: list[Any] | None,
    validation_direction: MetricDirection = "max",
    test_direction: MetricDirection | None = None,
) -> float | None:
    """Points gained on the HELD-OUT set, baseline to champion. None if unknown."""
    baseline, best_test = baseline_and_best_test(history, validation_direction)
    if baseline is None or best_test is None:
        return None
    return score_improvement(
        best_test, baseline, test_direction or validation_direction,
    )


def was_measured_on_heldout():
    """Predicate: this run has a REAL held-out measurement.

    Both pieces are required: a test ScoreEvent proves a number landed, and a
    `held_out_test` Ticket proves it came through the isolated engine lane.
    Shared here so Tasks and Leaderboard use the same eligibility rule.
    """
    return (
        select(ScoreEvent.id)
        .where(ScoreEvent.run_id == Run.id, ScoreEvent.split == "test")
        .exists()
        & select(Ticket.id)
        .where(Ticket.run_id == Run.id, Ticket.lane == "held_out_test")
        .exists()
    )
