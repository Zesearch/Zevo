"""The final Registry keeps the best fine-tuned model however small its gain.

Policy from the product owner (2026-10-02): hold the fine-tuned model even if
the improvement is minuscule. Selection is among fine-tuned candidates only,
by a strict comparison with no minimum margin.
"""
from __future__ import annotations

import inspect

import pytest

from zevo.api.routers.shared import tickets
from zevo.engine.method.score_direction import champion_index


def test_a_one_row_gain_over_the_other_candidates_wins() -> None:
    # 200 Validation rows: 0.115 is 23 correct, 0.120 is 24 correct.
    assert champion_index([0.115, 0.120, 0.110], "max") == 1


def test_a_minuscule_margin_is_enough() -> None:
    assert champion_index([0.1150000, 0.1150001], "max") == 1
    assert champion_index([0.3000001, 0.3000000], "min") == 1


def test_the_only_fine_tuned_model_is_kept_whatever_it_scored() -> None:
    # Whether it beat the base model is reported, not used as a gate.
    assert champion_index([0.01], "max") == 0
    assert champion_index([9.7], "min") == 0


def test_ties_keep_the_earliest_iteration() -> None:
    assert champion_index([0.2, 0.2, 0.2], "max") == 0
    assert champion_index([0.1, 0.2, 0.2], "max") == 1


def test_direction_is_respected() -> None:
    assert champion_index([0.9, 0.4, 0.6], "min") == 1
    assert champion_index([0.9, 0.4, 0.6], "max") == 0


def test_nothing_measured_means_no_champion() -> None:
    assert champion_index([], "max") is None


def test_registry_creation_uses_the_shared_policy_and_no_margin() -> None:
    source = inspect.getsource(tickets)
    assert "champion_index(" in source
    for forbidden in ("min_delta", "min_improvement", "regression_tolerance"):
        assert forbidden not in source.split("champion_index(")[1][:600]
