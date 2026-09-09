"""Leaderboard rows carry one champion's two held-out metrics side by side."""

from zevo.api.routers.ui.leaderboard import _rank_cells
from zevo.engine.observe.run_metrics import baseline_and_best, baseline_and_best_test


def _cell(
    run_id: str,
    *,
    task: str = "task",
    setting: tuple[str, ...] = ("s1",),
    score: float,
    improvement: float | None,
    direction: str = "max",
) -> dict:
    return {
        "run_id": run_id,
        "task_name": task,
        "setting_key": list(setting),
        "champion_test_score": score,
        "baseline_test_score": None,
        "improvement": improvement,
        "metric_direction": direction,
    }


def test_test_score_and_improvement_receive_independent_ranks() -> None:
    cells = [
        _cell("best-final", score=0.90, improvement=0.10),
        _cell("best-change", score=0.80, improvement=0.30),
    ]

    ranked = {cell["run_id"]: cell for cell in _rank_cells(cells, by="base")}

    assert ranked["best-final"]["champion_test_score_rank"] == 1
    assert ranked["best-final"]["improvement_rank"] == 2
    assert ranked["best-change"]["champion_test_score_rank"] == 2
    assert ranked["best-change"]["improvement_rank"] == 1


def test_min_metric_reverses_only_test_score_rank() -> None:
    cells = [
        _cell("low", score=0.20, improvement=0.40, direction="min"),
        _cell("high", score=0.50, improvement=0.10, direction="min"),
    ]

    ranked = {cell["run_id"]: cell for cell in _rank_cells(cells, by="base")}

    assert ranked["low"]["champion_test_score_rank"] == 1
    assert ranked["low"]["improvement_rank"] == 1
    assert ranked["high"]["champion_test_score_rank"] == 2
    assert ranked["high"]["improvement_rank"] == 2


def test_harness_ranks_are_scoped_to_task_and_setting() -> None:
    cells = [
        _cell("s1-low", setting=("s1",), score=0.40, improvement=0.10),
        _cell("s1-high", setting=("s1",), score=0.60, improvement=0.20),
        _cell("s2-only", setting=("s2",), score=0.10, improvement=-0.20),
    ]

    ranked = {cell["run_id"]: cell for cell in _rank_cells(cells, by="harness")}

    assert ranked["s1-high"]["champion_test_score_rank"] == 1
    assert ranked["s1-low"]["champion_test_score_rank"] == 2
    assert ranked["s2-only"]["champion_test_score_rank"] == 1
    assert ranked["s2-only"]["improvement_rank"] == 1


def test_ties_are_dense_and_missing_improvement_is_unranked() -> None:
    cells = [
        _cell("a", score=0.70, improvement=0.20),
        _cell("b", score=0.70, improvement=0.20),
        _cell("c", score=0.60, improvement=None),
    ]

    ranked = {cell["run_id"]: cell for cell in _rank_cells(cells, by="base")}

    assert ranked["a"]["champion_test_score_rank"] == 1
    assert ranked["b"]["champion_test_score_rank"] == 1
    assert ranked["c"]["champion_test_score_rank"] == 2
    assert ranked["a"]["improvement_rank"] == 1
    assert ranked["b"]["improvement_rank"] == 1
    assert ranked["c"]["improvement_rank"] is None
def test_champion_is_compared_with_its_own_model_lineage_baseline() -> None:
    history = [
        {"iteration": 0, "source": "baseline", "base_model": "model-a", "score": 0.60, "test_score": 0.58},
        {"iteration": 1, "source": "trained", "base_model": "model-a", "score": 0.70, "test_score": 0.68},
        {"iteration": 2, "source": "baseline", "base_model": "model-b", "score": 0.72, "test_score": 0.71},
        {"iteration": 2, "source": "trained", "base_model": "model-b", "score": 0.80, "test_score": 0.79},
    ]
    assert baseline_and_best(history) == (0.72, 0.80)
    assert baseline_and_best_test(history) == (0.71, 0.79)
