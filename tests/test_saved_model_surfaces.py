"""Saved-model UI surfaces never confuse remote provenance with data loss."""

from types import SimpleNamespace

from zevo.api.routers.shared.runs import _artifact_availability
from zevo.api.routers.ui.leaderboard import _runs_with_saved_models


def test_harness_board_excludes_runs_without_a_saved_model() -> None:
    runs = [SimpleNamespace(id="saved"), SimpleNamespace(id="measured-only")]
    assert [run.id for run in _runs_with_saved_models(runs, {"saved": "M-saved"})] == [
        "saved"
    ]


def test_remote_checkpoint_is_not_reported_as_missing() -> None:
    assert _artifact_availability(
        role="checkpoint",
        meta={"location": "remote"},
        ticket_iteration=1,
        original_exists=False,
        retained_model=None,
        retained_exists=False,
    ) == "remote"


def test_selected_remote_checkpoint_reports_the_saved_model() -> None:
    retained = SimpleNamespace(iteration=2)
    assert _artifact_availability(
        role="checkpoint",
        meta={"location": "remote"},
        ticket_iteration=2,
        original_exists=False,
        retained_model=retained,
        retained_exists=True,
    ) == "saved_model"
    assert _artifact_availability(
        role="checkpoint",
        meta={"location": "remote"},
        ticket_iteration=1,
        original_exists=False,
        retained_model=retained,
        retained_exists=True,
    ) == "remote"


def test_only_an_absent_local_artifact_is_missing() -> None:
    assert _artifact_availability(
        role="script",
        meta={},
        ticket_iteration=0,
        original_exists=False,
        retained_model=None,
        retained_exists=False,
    ) == "missing"
