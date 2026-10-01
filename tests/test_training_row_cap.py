"""A user's Training row cap is a typed field, not prose.

Smoke run b37b423c asked for "at most 500 training rows" in `data_query`. The
Data agent honoured it, the engine's iteration-0 rule ("prepare the entire
eligible split") rejected the subset, and a repair activation re-did the whole
preparation. The cap now lives in `max_training_rows`: the engine stamps it on
the Data work order, it is the only thing that permits an iteration-0 subset,
and prose limits are documented as not applied.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.shared.runs import CreateRunRequest, create_run
from zevo.contracts.data import DataRecipe, DataTaskInput
from zevo.contracts.orchestrator import AutoUserRequest, UserRequest
from zevo.db.models import Base, Run
from zevo.engine.run.runner import _max_training_rows, check_training_row_selection


def _recipe(**overrides) -> DataRecipe:
    body = dict(
        dataset_name="trivia", source_identity="query:sha256:" + "0" * 64,
        source_fingerprint="0" * 64, training_method="sft", method_format="prompt_completion",
        method_ids=["reformat_jsonl"], audit_steps=["mapped rows"],
    )
    body.update(overrides)
    return DataRecipe(**body)


def test_both_request_shapes_accept_the_cap() -> None:
    auto = AutoUserRequest(task_objective="o", max_training_rows=500)
    assert auto.max_training_rows == 500
    full = UserRequest(
        task_objective="o", metric="accuracy", metric_direction="max",
        metric_type="builtin", training_method="", dataset="", base_model="",
        test_set="/t.csv", test_answer_fields=["a"], test_sample_submission="/s.csv",
        constraints=[], max_training_rows=500,
    )
    assert full.max_training_rows == 500
    with pytest.raises(ValidationError):
        AutoUserRequest(task_objective="o", max_training_rows=-1)


def test_iteration0_without_a_cap_still_forbids_every_selection() -> None:
    check_training_row_selection(_recipe(), iteration=0, max_training_rows=0, n_rows_out=10_000)
    for field, value in (("subset", "train[:500]"), ("filters", ["len>3"]),
                         ("sampling", {"n": 5}), ("weighting", {"a": 1.0})):
        with pytest.raises(ValueError, match="initial Data selected"):
            check_training_row_selection(
                _recipe(**{field: value}), iteration=0, max_training_rows=0, n_rows_out=500,
            )


def test_iteration0_under_a_cap_permits_only_a_subset() -> None:
    check_training_row_selection(
        _recipe(subset="seed=0 sample 500"), iteration=0, max_training_rows=500, n_rows_out=500,
    )
    with pytest.raises(ValueError, match="filters"):
        check_training_row_selection(
            _recipe(subset="train[:500]", filters=["len>3"]),
            iteration=0, max_training_rows=500, n_rows_out=500,
        )


def test_the_cap_binds_every_iteration() -> None:
    with pytest.raises(ValueError, match="exceeds the Run's max_training_rows=500"):
        check_training_row_selection(
            _recipe(subset="train[:600]"), iteration=0, max_training_rows=500, n_rows_out=600,
        )
    with pytest.raises(ValueError, match="exceeds"):
        check_training_row_selection(_recipe(), iteration=2, max_training_rows=500, n_rows_out=501)
    check_training_row_selection(_recipe(), iteration=2, max_training_rows=500, n_rows_out=499)


def test_scope_problem_never_carries_the_cap() -> None:
    with pytest.raises(ValidationError, match="scope_problem"):
        DataTaskInput(
            ticket_id="scope-1", operation="scope_problem", run_id="r",
            task_objective="o", scoping_result_schema={"type": "object"},
            scoping_result_validation_command="python -m x", work_dir="/w",
            data_recipe_schema={}, data_recipe_validation_command="x",
            artifacts_validation_command="x", max_training_rows=500,
        )


def test_run_pin_reader_tolerates_absent_or_bad_values() -> None:
    assert _max_training_rows(Run(decision_pins={})) == 0
    assert _max_training_rows(Run(decision_pins={"max_training_rows": "500"})) == 500
    assert _max_training_rows(Run(decision_pins={"max_training_rows": "many"})) == 0
    assert _max_training_rows(Run(decision_pins=None)) == 0


@pytest.mark.asyncio
async def test_auto_launch_stores_the_cap_as_a_decision_pin(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ZEVO_WORK_DIR", str(tmp_path / "runs"))
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        response = await create_run(CreateRunRequest.model_validate(dict(
            mode="auto", task_name="cap", run_name="cap-1", gpu_provider="instance",
            user_request={"task_objective": "o", "max_training_rows": 500},
        )), db)
        run = await db.get(Run, response.run_id)
        assert run.decision_pins["max_training_rows"] == 500
        uncapped = await create_run(CreateRunRequest.model_validate(dict(
            mode="auto", task_name="cap", run_name="cap-2", gpu_provider="instance",
            user_request={"task_objective": "o"},
        )), db)
        assert "max_training_rows" not in (await db.get(Run, uncapped.run_id)).decision_pins
    await engine.dispose()
