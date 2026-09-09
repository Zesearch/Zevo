"""Model cards report the same paired held-out champion outcomes as the UI."""

from datetime import datetime, timezone

from zevo.api.routers.ui.models import kept_models
from zevo.db import RegistryModel
from zevo.engine.observe.model_card import render_model_card


def test_model_card_pairs_final_test_score_with_improvement() -> None:
    card = render_model_card(
        registry={
            "registered_at": "2026-08-16T12:00:00Z",
            "eval": {"score": 0.91},
            "base_model": "org/base",
            "training_method": "sft",
        },
        run={
            "metric": "accuracy",
            "champion_test_score": 0.80,
            "baseline_test_score": 0.60,
            "improvement": 0.20,
            "best_validation_score": 0.91,
            "status": "success",
        },
    )

    assert "| Final held-out test accuracy | 0.8 |" in card
    assert "| Improvement over held-out baseline | +0.2 |" in card
    assert "| Held-out baseline accuracy | 0.6 |" in card
    assert "Champion held-out" not in card


def test_models_list_excludes_registry_rows_whose_artifact_is_missing(tmp_path) -> None:
    existing = tmp_path / "M-existing"
    existing.mkdir()
    common = {
        "iteration": 1,
        "base_model": "org/base",
        "training_method": "sft",
        "metric": "accuracy",
        "metric_direction": "max",
        "registered_at": datetime.now(timezone.utc),
    }
    rows = [
        RegistryModel(version_tag="M-existing", model_path=str(existing), **common),
        RegistryModel(version_tag="M-missing", model_path=str(tmp_path / "missing"), **common),
    ]

    assert [row.version_tag for row in kept_models(rows)] == ["M-existing"]
