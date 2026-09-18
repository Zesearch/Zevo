from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Base, ExecutionEvent, Run, ScoreEvent, Ticket, WorkProduct


def _member(name: str, metric: str = "accuracy") -> dict:
    return {
        "name": name,
        "test_set": f"/private/{name}.csv",
        "inference_query": "Solve {question} and return only the final answer.",
        "sample_submission": f"/private/{name}-submission.csv",
        "metric": metric,
        "answer_fields": ["answer"],
        "metric_direction": "max",
    }


def test_task_body_is_exactly_a_named_test_suite() -> None:
    from zevo.api.routers.ui.tasks import TaskBody

    body = TaskBody(
        name="reasoning-suite",
        task_objective="Improve reasoning quality.",
        test_sets=[_member("math"), _member("qa", "exact_match")],
    )
    assert [item.name for item in body.test_sets] == ["math", "qa"]

    with pytest.raises(ValidationError, match="unique"):
        TaskBody(
            name="duplicates",
            task_objective="Improve quality.",
            test_sets=[_member("math"), _member("MATH")],
        )
    missing_submission = {**_member("math"), "sample_submission": ""}
    with pytest.raises(ValidationError, match="sample submission"):
        TaskBody(
            name="missing-submission",
            task_objective="Improve quality.",
            test_sets=[missing_submission],
        )


def test_scoring_source_rows_are_display_metadata_not_a_cap() -> None:
    from zevo.contracts.orchestrator import TaskTestSet

    item = TaskTestSet(
        **_member("math"), source_rows=1_319, max_rows=500,
    )
    assert item.source_rows == 1_319
    assert item.max_rows == 500


def test_test_set_supports_custom_metric_script_and_direction() -> None:
    from zevo.api.routers.ui.tasks import TaskBody

    custom = {
        **_member("code"),
        "metric_type": "custom",
        "metric": "pass@1",
        "metric_direction": "min",
        "evaluation_script": "/uploads/pass_at_one.py",
    }
    body = TaskBody(
        name="custom-suite",
        task_objective="Measure code generation.",
        test_sets=[custom],
    )
    assert body.test_sets[0].metric_type == "custom"
    assert body.test_sets[0].metric_direction == "min"
    assert body.test_sets[0].evaluation_script == "/uploads/pass_at_one.py"

    with pytest.raises(ValidationError, match="require an evaluation_script"):
        TaskBody(
            name="missing-script",
            task_objective="Measure code generation.",
            test_sets=[{**custom, "evaluation_script": ""}],
        )
    with pytest.raises(ValidationError, match="must not carry a custom evaluator"):
        TaskBody(
            name="builtin-with-script",
            task_objective="Measure code generation.",
            test_sets=[{
                **_member("code"),
                "evaluation_script": "/uploads/unexpected.py",
            }],
        )
    with pytest.raises(ValidationError, match="share a metric direction"):
        TaskBody(
            name="mixed-directions",
            task_objective="Measure two datasets.",
            test_sets=[_member("one"), {**_member("two"), "metric_direction": "min"}],
        )


def test_validation_suite_uses_member_contracts_not_scalar_suite_average() -> None:
    from zevo.contracts.orchestrator import (
        TaskTestSet, UserRequest, scoring_asset_errors,
    )

    test = TaskTestSet(**_member("test"))
    validation = TaskTestSet(**_member("validation"))
    request = UserRequest(
        task_objective="Improve quality.",
        test_sets=[test],
        validation_sets=[validation],
        metric="accuracy",
        metric_direction="max",
        validation_metric_type="builtin",
        validation_metric="suite_average",
        validation_metric_direction="max",
        training_method="",
        dataset="",
        base_model="owner/model",
        test_set=test.test_set,
        test_answer_fields=test.answer_fields,
        test_sample_submission=test.sample_submission,
        constraints=[],
    )
    assert scoring_asset_errors(request) == []


def test_inference_query_renders_fields_without_treating_json_as_a_placeholder() -> None:
    from zevo.contracts.prompting import render_inference_query

    rendered = render_inference_query(
        'Solve {question}. Return JSON like {"answer": "..."}.',
        {"question": "2 + 2"},
    )
    assert rendered == 'Solve 2 + 2. Return JSON like {"answer": "..."}.'


def test_inference_query_preserves_latex_braces_as_literal_text() -> None:
    from zevo.contracts.prompting import render_inference_query

    rendered = render_inference_query(
        r"Solve carefully and finish with exactly one \boxed{answer}.",
        {"question": "2 + 2"},
    )
    assert rendered == (
        "Solve carefully and finish with exactly one \\boxed{answer}.\n\n2 + 2"
    )


def test_inference_query_substitutes_real_fields_and_preserves_other_braces() -> None:
    from zevo.contracts.prompting import render_inference_query

    rendered = render_inference_query(
        r"Solve {question} and finish with \boxed{answer}.",
        {"question": "2 + 2"},
    )
    assert rendered == r"Solve 2 + 2 and finish with \boxed{answer}."


@pytest.mark.asyncio
async def test_remote_test_member_uses_its_explicit_split_without_catalogue(
    tmp_path, monkeypatch,
) -> None:
    """A Task's scoring contract is portable across deployments.

    In particular, AIME publishes only ``default/train``. A deployment whose
    Files catalogue is absent or stored behind a tenant directory must still
    use the split explicitly saved on the Task instead of asking the generic
    resolver to guess a validation-like split.
    """
    import zevo.engine.remote_datasets as remote
    from zevo.contracts.orchestrator import TaskTestSet, UserRequest
    from zevo.engine.run.split_settlement import settle_splits

    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "private"))
    fetched: list[dict] = []

    async def fake_materialize(*, hub_id, split, config, out_dir, limit=0, answer_scope="test"):
        fetched.append({
            "hub_id": hub_id, "split": split, "config": config, "limit": limit,
            "answer_scope": answer_scope,
        })
        path = tmp_path / "aime.csv"
        path.write_text(
            "id,question,answer\n" + "".join(
                f"{index},problem-{index},{index % 10}\n" for index in range(120)
            ),
            encoding="utf-8",
        )
        return str(path), ["id", "question", "answer"], 120, "fetched AIME"

    def catalogue_must_not_be_needed(*_args, **_kwargs):
        raise AssertionError("explicit split/config must not require catalogue lookup")

    monkeypatch.setattr(remote, "materialize", fake_materialize)
    monkeypatch.setattr(remote, "lookup", catalogue_must_not_be_needed)

    sample = tmp_path / "aime-submission.csv"
    sample.write_text("id,prediction\nexample,<answer>\n", encoding="utf-8")
    validation = tmp_path / "validation.csv"
    validation.write_text(
        "id,question,answer\n" + "".join(
            f"{index},validation-{index},{index % 10}\n" for index in range(200)
        ),
        encoding="utf-8",
    )
    validation_sample = tmp_path / "validation-submission.csv"
    validation_sample.write_text(
        "id,prediction\nexample,<answer>\n", encoding="utf-8",
    )

    test = TaskTestSet(**{
        **_member("AIME 2024"),
        "test_set": "allenai/aime-2022-2025",
        "split": "train",
        "config": "default",
        "sample_submission": str(sample),
    })
    validation_member = TaskTestSet(**{
        **_member("validation"),
        "test_set": str(validation),
        "sample_submission": str(validation_sample),
    })
    request = UserRequest(
        task_objective="Improve mathematical reasoning.",
        test_sets=[test],
        validation_sets=[validation_member],
        metric=test.metric,
        metric_direction="max",
        training_method="",
        dataset="",
        base_model="owner/model",
        test_set=test.test_set,
        test_answer_fields=list(test.answer_fields),
        test_sample_submission=test.sample_submission,
        constraints=[],
    )
    run = Run(
        id="explicit-aime-split", task_name="suite", status="running",
        metric="accuracy", metric_direction="max",
        started_at=datetime.now(timezone.utc),
    )

    _agent_request, holdout, _note = await settle_splits(
        run, request, work_dir_root=str(tmp_path / "work"),
    )

    assert fetched == [{
        "hub_id": "allenai/aime-2022-2025",
        "split": "train",
        "config": "default",
        "limit": 0,
        "answer_scope": "test",
    }]
    assert holdout["test_sets"][0]["name"] == "AIME 2024"


@pytest.mark.asyncio
async def test_validation_is_derived_per_eligible_suite_member(
    tmp_path, monkeypatch,
) -> None:
    from zevo.contracts.orchestrator import TaskTestSet, UserRequest
    from zevo.engine.run.split_settlement import settle_splits

    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "private"))

    def write_test(name: str, rows: int) -> tuple[str, str]:
        test = tmp_path / f"{name}.csv"
        test.write_text(
            "id,question,answer\n" + "".join(
                f"{index},{name}-{index},{index % 4}\n" for index in range(rows)
            ),
            encoding="utf-8",
        )
        sample = tmp_path / f"{name}-submission.csv"
        sample.write_text("id,prediction\nexample,<answer>\n", encoding="utf-8")
        return str(test), str(sample)

    math, math_sample = write_test("math", 1_000)
    qa, qa_sample = write_test("qa", 2_000)
    livecodebench, livecodebench_sample = write_test("livecodebench", 1_000)
    aime, aime_sample = write_test("aime", 30)
    suite = [
        TaskTestSet(**{
            **_member("math"), "test_set": math,
            "sample_submission": math_sample,
        }),
        TaskTestSet(**{
            **_member("qa", "exact_match"), "test_set": qa,
            "sample_submission": qa_sample,
        }),
        TaskTestSet(**{
            **_member("livecodebench"), "test_set": livecodebench,
            "sample_submission": livecodebench_sample,
        }),
        TaskTestSet(**{
            **_member("aime"), "test_set": aime,
            "sample_submission": aime_sample,
        }),
    ]
    monkeypatch.setattr(
        "zevo.engine.run.split_settlement.code_execution_adapter_for",
        lambda reference: (
            "livecodebench" if "livecodebench" in reference else ""
        ),
    )
    request = UserRequest(
        task_objective="Improve reasoning across the evaluation suite.",
        test_sets=suite,
        metric=suite[0].metric,
        metric_direction="max",
        training_method="",
        dataset="",
        base_model="owner/model",
        test_set=math,
        test_answer_fields=["answer"],
        test_sample_submission=math_sample,
        constraints=[],
    )
    run = Run(
        id="validation-suite-run", task_name="suite", status="running",
        metric="suite_average", metric_direction="max",
        started_at=datetime.now(timezone.utc),
    )

    agent_request, holdout, _note = await settle_splits(
        run, request, work_dir_root=str(tmp_path / "work"),
    )

    assert [item["name"] for item in holdout["validation_sets"]] == [
        "math", "qa", "livecodebench",
    ]
    assert [item["n_rows"] for item in holdout["validation_sets"]] == [
        200, 400, 200,
    ]
    assert (
        holdout["validation_sets"][2]["code_execution_adapter"]
        == "livecodebench"
    )
    assert holdout["validation_final_test_only"] == [{
        "name": "aime",
        "reason": (
            "20% of the Test set is only 6 row(s); derived Validation requires "
                "at least 200 while final Test keeps at least 40. Keep this benchmark "
            "final-test-only or supply an independent Validation set."
        ),
    }]
    final_sizes = {}
    for item in holdout["test_sets"]:
        final_sizes[item["name"]] = sum(
            1 for _ in open(item["test_set"], encoding="utf-8")
        ) - 1
    assert final_sizes == {
        "math": 800, "qa": 1_600, "livecodebench": 800, "aime": 30,
    }
    assert agent_request.metric == "suite_average"
    assert agent_request.validation_metric == "suite_average"
    assert holdout["validation_aggregation"] == "unweighted_mean"


@pytest.mark.asyncio
async def test_independent_validation_suite_keeps_all_test_rows(
    tmp_path, monkeypatch,
) -> None:
    from zevo.contracts.orchestrator import TaskTestSet, UserRequest
    from zevo.engine.run.split_settlement import settle_splits

    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "private"))

    def write_set(name: str, rows: int) -> tuple[str, str]:
        dataset = tmp_path / f"{name}.csv"
        dataset.write_text(
            "id,question,answer\n" + "".join(
                f"{index},{name}-{index},{index % 4}\n" for index in range(rows)
            ),
            encoding="utf-8",
        )
        sample = tmp_path / f"{name}-submission.csv"
        sample.write_text("id,prediction\nexample,<answer>\n", encoding="utf-8")
        return str(dataset), str(sample)

    math, math_sample = write_set("test-math", 500)
    qa, qa_sample = write_set("test-qa", 500)
    gsm, gsm_sample = write_set("validation-gsm", 200)
    arc, arc_sample = write_set("validation-arc", 240)
    monkeypatch.chdir(tmp_path)
    tests = [
        TaskTestSet(**{
            **_member("math"), "test_set": math,
            "sample_submission": math_sample,
        }),
        TaskTestSet(**{
            **_member("qa", "exact_match"), "test_set": qa,
            "sample_submission": qa_sample,
        }),
    ]
    validation = [
        TaskTestSet(**{
            **_member("GSM8K"), "test_set": Path(gsm).name,
            "sample_submission": gsm_sample,
        }),
        TaskTestSet(**{
            **_member("ARC Challenge", "exact_match"), "test_set": arc,
            "sample_submission": arc_sample,
        }),
    ]
    request = UserRequest(
        task_objective="Improve reasoning across the evaluation suite.",
        test_sets=tests,
        validation_sets=validation,
        metric=tests[0].metric,
        metric_direction="max",
        training_method="",
        dataset="",
        base_model="owner/model",
        test_set=math,
        test_answer_fields=["answer"],
        test_sample_submission=math_sample,
        constraints=[],
    )
    run = Run(
        id="independent-validation-suite-run", task_name="suite",
        status="running", metric="suite_average", metric_direction="max",
        started_at=datetime.now(timezone.utc),
    )

    agent_request, holdout, _note = await settle_splits(
        run, request, work_dir_root=str(tmp_path / "work"),
    )

    assert [item["name"] for item in holdout["validation_sets"]] == [
        "GSM8K", "ARC Challenge",
    ]
    assert holdout["validation_rows"] == 440
    assert holdout["validation_source"] == "independent_validation_suite"
    assert holdout["validation_suite_policy"] == "supplied"
    assert holdout["validation_sets"][0]["validation_set"] == gsm
    assert holdout["validation_final_test_only"] == []
    assert {
        item["name"]: sum(
            1 for _ in open(item["test_set"], encoding="utf-8")
        ) - 1
        for item in holdout["test_sets"]
    } == {"math": 500, "qa": 500}
    assert agent_request.validation_sets == []
    assert agent_request.validation_metric == "suite_average"


@pytest.mark.asyncio
async def test_run_setup_normalizes_relative_validation_and_private_test_paths(
    tmp_path, monkeypatch,
) -> None:
    from zevo.contracts.orchestrator import TaskTestSet, UserRequest
    from zevo.engine.run.split_settlement import settle_splits
    from zevo.holdout_storage import private_mirror, protect_asset

    files = tmp_path / "data" / "files"
    test_dir = files / "test"
    validation_dir = files / "validation"
    test_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    (test_dir / "rows.csv").write_text(
        "id,question,answer\n1,test-question,yes\n", encoding="utf-8",
    )
    (test_dir / "submission.csv").write_text(
        "id,prediction\n1,yes\n", encoding="utf-8",
    )
    (validation_dir / "rows.csv").write_text(
        "id,question,answer\n" + "".join(
            f"{index},validation-question-{index},yes\n"
            for index in range(200)
        ),
        encoding="utf-8",
    )
    (validation_dir / "submission.csv").write_text(
        "id,prediction\n1,yes\n", encoding="utf-8",
    )
    monkeypatch.setenv("ZEVO_FILES_DIR", str(files))
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "private"))
    monkeypatch.chdir(tmp_path)
    test_path = "data/files/test/rows.csv"
    test_sample = "data/files/test/submission.csv"
    protect_asset(test_path)
    protect_asset(test_sample)

    test = TaskTestSet(**{
        **_member("test"), "test_set": test_path,
        "sample_submission": test_sample,
    })
    validation = TaskTestSet(**{
        **_member("validation"),
        "test_set": "data/files/validation/rows.csv",
        "sample_submission": "data/files/validation/submission.csv",
    })
    request = UserRequest(
        task_objective="Improve quality.", test_sets=[test],
        validation_sets=[validation], metric="accuracy",
        metric_direction="max", training_method="", dataset="",
        base_model="owner/model", test_set=test_path,
        test_answer_fields=["answer"],
        test_sample_submission=test_sample, constraints=[],
    )
    run = Run(
        id="relative-scoring-paths", task_name="suite", status="running",
        metric="accuracy", metric_direction="max",
        started_at=datetime.now(timezone.utc),
    )

    _agent_request, holdout, _note = await settle_splits(
        run, request, work_dir_root=str(tmp_path / "work"),
    )

    assert holdout["test_sets"][0]["test_set"] == str(private_mirror(test_path))
    assert holdout["test_sets"][0]["sample_submission"] == str(
        private_mirror(test_sample)
    )
    assert holdout["validation_sets"][0]["sample_submission"] == str(
        validation_dir / "submission.csv"
    )
    assert holdout["validation_sample_submission"] == str(
        validation_dir / "submission.csv"
    )


@pytest.mark.asyncio
async def test_run_setup_rejects_missing_validation_submission_before_data(
    tmp_path, monkeypatch,
) -> None:
    from zevo.contracts.orchestrator import TaskTestSet, UserRequest
    from zevo.engine.run.split_settlement import SplitSettlementError, settle_splits

    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "private"))
    test_set = tmp_path / "test.csv"
    test_set.write_text("id,question,answer\n1,test,yes\n", encoding="utf-8")
    test_sample = tmp_path / "test-submission.csv"
    test_sample.write_text("id,prediction\n1,yes\n", encoding="utf-8")
    validation_set = tmp_path / "validation.csv"
    validation_set.write_text(
        "id,question,answer\n" + "".join(
            f"{index},validation-{index},yes\n" for index in range(200)
        ),
        encoding="utf-8",
    )
    test = TaskTestSet(**{
        **_member("test"), "test_set": str(test_set),
        "sample_submission": str(test_sample),
    })
    validation = TaskTestSet(**{
        **_member("validation"), "test_set": str(validation_set),
        "sample_submission": str(tmp_path / "missing-submission.csv"),
    })
    request = UserRequest(
        task_objective="Improve quality.", test_sets=[test],
        validation_sets=[validation], metric="accuracy",
        metric_direction="max", training_method="", dataset="",
        base_model="owner/model", test_set=str(test_set),
        test_answer_fields=["answer"],
        test_sample_submission=str(test_sample), constraints=[],
    )
    run = Run(
        id="missing-validation-sample", task_name="suite", status="running",
        metric="accuracy", metric_direction="max",
        started_at=datetime.now(timezone.utc),
    )

    with pytest.raises(
        SplitSettlementError, match="Validation set 'validation' sample submission is invalid",
    ):
        await settle_splits(run, request, work_dir_root=str(tmp_path / "work"))


@pytest.mark.asyncio
async def test_validation_suite_publishes_one_unweighted_average(tmp_path) -> None:
    from zevo.engine.run.runner import _record_validation_component

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    validation_suite = [
        {
            "name": "math", "validation_set": "/work/math.csv",
            "metric": "accuracy", "metric_direction": "max",
        },
        {
            "name": "qa", "validation_set": "/work/qa.csv",
            "metric": "exact_match", "metric_direction": "max",
        },
    ]
    async with Session() as db:
        run = Run(
            id="validation-aggregate-run", task_name="suite", status="running",
            metric="suite_average", metric_direction="max",
            validation_metric="suite_average", validation_metric_direction="max",
            holdout={
                "validation_sets": validation_suite,
                "validation_suite_results": {},
                "validation_suite_recorded": [],
            },
            started_at=datetime.now(timezone.utc),
        )
        infers = [
            Ticket(
                id="infer-math", run_id=run.id, agent_id="inference",
                status="succeeded", lane="optimization", iteration=0,
                payload={"model_source": "base_model", "base_model": "owner/model"},
            ),
            Ticket(
                id="infer-qa", run_id=run.id, agent_id="inference",
                status="succeeded", lane="optimization", iteration=0,
                payload={
                    "model_source": "base_model", "base_model": "owner/model",
                    "test_set_name": "validation:qa",
                },
            ),
        ]
        evaluations = [
            Ticket(
                id="eval-math", run_id=run.id, agent_id="evaluation",
                status="succeeded", lane="optimization", iteration=0,
                payload={"metric": "accuracy"},
                inputs={"predictions": {
                    "source_ticket_id": "infer-math", "artifact_role": "predictions",
                    "work_product_id": "", "path": "",
                }},
            ),
            Ticket(
                id="eval-qa", run_id=run.id, agent_id="evaluation",
                status="succeeded", lane="optimization", iteration=0,
                payload={"metric": "exact_match", "test_set_name": "validation:qa"},
                inputs={"predictions": {
                    "source_ticket_id": "infer-qa", "artifact_role": "predictions",
                    "work_product_id": "", "path": "",
                }},
            ),
        ]
        metric_paths = [tmp_path / "math-metrics.json", tmp_path / "qa-metrics.json"]
        for path in metric_paths:
            path.write_text('{"score": 0.0}\n', encoding="utf-8")
        products = [
            WorkProduct(
                ticket_id=evaluation.id,
                role="metrics",
                path=str(path),
                meta={"score": 0.0},
            )
            for evaluation, path in zip(evaluations, metric_paths, strict=True)
        ]
        db.add_all([run, *infers, *evaluations, *products])
        await db.commit()

        complete, _source = await _record_validation_component(
            db, run, evaluations[0], 0.4,
        )
        assert complete is False
        assert (await db.execute(select(func.count()).select_from(ScoreEvent))).scalar_one() == 0

        complete, _source = await _record_validation_component(
            db, run, evaluations[1], 0.8,
        )
        assert complete is True
        event = (await db.execute(select(ScoreEvent))).scalar_one()
        assert event.metric_name == "suite_average"
        assert event.score == pytest.approx(0.6)
        assert event.extras["validation_sets"] == {
            "math": {
                "score": 0.4, "metric": "accuracy",
                "metric_direction": "max",
            },
            "qa": {
                "score": 0.8, "metric": "exact_match",
                "metric_direction": "max",
            },
        }
        await db.refresh(run)
        assert run.history[0]["validation_scores"] == {"math": 0.4, "qa": 0.8}
        assert run.history[0]["validation_metric_directions"] == {
            "math": "max", "qa": "max",
        }
        for path in metric_paths:
            stored = json.loads(path.read_text(encoding="utf-8"))
            assert stored["score"] == pytest.approx(0.6)
            assert stored["metric"] == "suite_average"
            assert set(stored["validation_sets"]) == {"math", "qa"}

    await engine.dispose()


@pytest.mark.asyncio
async def test_validation_member_selects_its_own_baseline_inference_query() -> None:
    from zevo.engine.run.runner import (
        _build_validation_suite_members,
        _spawn_validation_suite_evals,
    )

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    validation_suite = [
        {
            "name": "math", "public": "/work/math-public.csv",
            "validation_set": "/work/math.csv", "metric": "accuracy",
            "answer_fields": ["answer"],
            "sample_submission": "/work/math-submission.csv",
            "inference_data_profile": "/work/math-profile.json",
            "inference_query": "Solve {question}.",
        },
        {
            "name": "qa", "public": "/work/qa-public.csv",
            "validation_set": "/work/qa.csv", "metric": "exact_match",
            "answer_fields": ["answer"],
            "sample_submission": "/work/qa-submission.csv",
            "inference_data_profile": "/work/qa-profile.json",
            "inference_query": "Answer {question} with one short phrase.",
        },
    ]
    async with Session() as db:
        run = Run(
            id="validation-query-run", task_name="suite", status="running",
            metric="suite_average", metric_direction="max",
            holdout={"validation_sets": validation_suite},
            started_at=datetime.now(timezone.utc),
        )
        primary = Ticket(
            id="primary-infer", run_id=run.id, agent_id="inference",
            status="succeeded", lane="optimization", iteration=0,
            payload={
                "operation": "run_inference", "model_source": "base_model",
                "base_model": "owner/model", "scoring_set": "/work/math-public.csv",
                "sample_submission": "/work/math-submission.csv",
                "configuration_suggestions": {}, "configuration_pins": {},
            },
            inputs={"device_info": {
                "source_ticket_id": "infra-001", "artifact_role": "device_info",
                "work_product_id": "", "path": "",
            }},
        )
        db.add_all([run, primary])
        await db.commit()

        members = await _build_validation_suite_members(
            ticket=primary,
            payload=dict(primary.payload or {}),
            work_dir="/work/run/primary-infer",
            run=run,
            session=db,
        )
        assert len(members) == 1
        qa = members[0]
        assert qa.test_set_name == "validation:qa"
        assert qa.configuration_pins["inference_config"][
            "inference_query"
        ] == "Answer {question} with one short phrase."
        assert qa.configuration_mode == "select"
        assert qa.inference_data_profile_path == "/work/qa-profile.json"

        prediction = WorkProduct(
            ticket_id=primary.id,
            role="predictions",
            path="/work/run/primary-infer/suite/001/predictions.csv",
            meta={"suite_member_name": "qa"},
        )
        db.add(prediction)
        await db.commit()
        created = await _spawn_validation_suite_evals(
            db, run, primary, enqueue=False,
        )
        assert len(created) == 1
        evaluation = created[0]
        assert evaluation.agent_id == "evaluation"
        assert evaluation.payload["test_set_name"] == "validation:qa"
        assert evaluation.inputs["predictions"] == {
            "source_ticket_id": primary.id,
            "artifact_role": "predictions",
            "work_product_id": prediction.id,
            "path": prediction.path,
        }

    await engine.dispose()


@pytest.mark.asyncio
async def test_run_overview_reports_current_benchmark_and_suite_progress() -> None:
    from zevo.api.routers.shared.runs import _benchmark_progress

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        run = Run(
            id="progress-run", task_name="suite", status="running",
            metric="suite_average", metric_direction="max",
            holdout={
                "validation_sets": [
                    {"name": "math"}, {"name": "qa"},
                ],
                "test_sets": [
                    {"name": "math"}, {"name": "qa"},
                ],
            },
            started_at=datetime.now(timezone.utc),
        )
        primary = Ticket(
            id="infer-primary", run_id=run.id, agent_id="inference",
            status="succeeded", lane="optimization", iteration=0,
            payload={
                "model_source": "base_model", "base_model": "owner/model",
                "test_set_name": "validation:math",
            },
        )
        primary_eval = Ticket(
            id="eval-primary", run_id=run.id, agent_id="evaluation",
            status="succeeded", lane="optimization", iteration=0,
            inputs={"predictions": {"source_ticket_id": primary.id}},
        )
        qa = Ticket(
            id="infer-qa", run_id=run.id, agent_id="inference",
            status="running", lane="optimization", iteration=0,
            payload={
                "model_source": "base_model", "base_model": "owner/model",
                "test_set_name": "validation:qa",
            },
        )
        db.add_all([run, primary, primary_eval, qa])
        await db.commit()

        progress = _benchmark_progress(
            run, [primary, primary_eval, qa], reveal_holdout=True,
        )
        assert progress["validation"] == {
            "completed": 1,
            "inference_completed": 1,
            "total": 2,
            "failed": 0,
            "iteration": 0,
            "model_source": "base_model",
            "current": [{
                "suite": "validation", "name": "qa",
                "stage": "inference", "status": "running", "iteration": 0,
            }],
        }
        assert progress["test"]["total"] == 2
        assert progress["test"]["completed"] == 0
        assert progress["current"][0]["name"] == "qa"

        primary.status = "running"
        primary.summary = "Inference suite · qa (2/2)"
        progress = _benchmark_progress(
            run, [primary], reveal_holdout=True,
        )
        assert progress["validation"]["current"] == [{
            "suite": "validation", "name": "qa",
            "stage": "inference", "status": "running", "iteration": 0,
        }]

    await engine.dispose()


def test_combined_suite_progress_counts_completed_members_not_tickets() -> None:
    from zevo.api.routers.shared.runs import _benchmark_progress

    run = Run(
        id="combined-progress", task_name="suite", status="running",
        holdout={"validation_sets": [
            {"name": "math"}, {"name": "qa"}, {"name": "code"},
        ]},
    )
    inference = Ticket(
        id="combined-infer", run_id=run.id, agent_id="inference",
        status="running", lane="optimization", iteration=0,
        payload={"model_source": "base_model", "base_model": "owner/model"},
        summary="Inference suite · code (3/3)",
        created_at=datetime.now(timezone.utc),
    )
    finished = [
        ExecutionEvent(
            ticket_id=inference.id, heartbeat_id="submit", attempt_id="one",
            event_type="progress", phase="complete", current_step=10,
            total_steps=10, extras={"benchmark_name": name},
        ) for name in ("math", "qa")
    ]
    progress = _benchmark_progress(
        run, [inference], reveal_holdout=False, completion_events=finished,
    )
    assert progress["validation"]["inference_completed"] == 2
    assert progress["validation"]["completed"] == 0
    assert progress["validation"]["current"][0]["name"] == "code"

    evaluation_in_progress = Ticket(
        id="combined-eval", run_id=run.id, agent_id="evaluation",
        status="running", lane="optimization", iteration=0,
        inputs={"predictions": {"source_ticket_id": inference.id}},
        summary="Evaluation suite · code (3/3)",
        created_at=datetime.now(timezone.utc),
    )
    scored = [
        ExecutionEvent(
            ticket_id=evaluation_in_progress.id, heartbeat_id="score",
            attempt_id="one", event_type="progress", phase="benchmark_complete",
            current_step=1, total_steps=1, extras={"benchmark_name": name},
        ) for name in ("math", "qa")
    ]
    progress = _benchmark_progress(
        run, [inference, evaluation_in_progress], reveal_holdout=False,
        completion_events=[*finished, *scored],
    )
    assert progress["validation"]["completed"] == 2
    assert progress["validation"]["current"][0]["name"] == "code"

    # Finishing the last model-judge row is not the same as validating the
    # benchmark's metrics file, so it must not advance the suite counter.
    judge_rows_done = ExecutionEvent(
        ticket_id=evaluation_in_progress.id, heartbeat_id="score",
        attempt_id="one", event_type="progress", phase="scoring",
        current_step=50, total_steps=50,
        extras={"phase": "model_judge", "benchmark_name": "code"},
    )
    progress = _benchmark_progress(
        run, [inference, evaluation_in_progress], reveal_holdout=False,
        completion_events=[*finished, *scored, judge_rows_done],
    )
    assert progress["validation"]["completed"] == 2

    inference.status = "succeeded"
    evaluation = evaluation_in_progress
    evaluation.status = "succeeded"
    progress = _benchmark_progress(
        run, [inference, evaluation], reveal_holdout=False,
        completion_events=finished,
    )
    assert progress["validation"]["inference_completed"] == 3
    assert progress["validation"]["completed"] == 3

    heldout = Ticket(
        id="combined-heldout", run_id=run.id, agent_id="inference",
        status="succeeded", lane="held_out_test", iteration=0,
        payload={
            "model_source": "base_model", "base_model": "owner/model",
            "test_set_name": "math",
        },
        created_at=datetime.now(timezone.utc),
    )
    run.holdout = {
        **run.holdout,
        "test_sets": [{"name": name} for name in ("math", "qa", "code")],
    }
    progress = _benchmark_progress(run, [heldout], reveal_holdout=True)
    assert progress["test"]["inference_completed"] == 3


@pytest.mark.asyncio
async def test_heldout_suite_uses_one_inference_with_member_queries() -> None:
    from zevo.engine.run.runner import (
        _build_evaluation_suite_members,
        _build_holdout_suite_members,
        _spawn_holdout_infer,
        _spawn_holdout_suite_evals,
    )

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    suite = [
        {**_member("math"), "public": "/work/math-public.csv",
         "inference_data_profile": "/work/math-profile.json"},
        {**_member("qa", "exact_match"), "public": "/work/qa-public.csv",
         "inference_data_profile": "/work/qa-profile.json"},
    ]
    async with Session() as db:
        run = Run(
            id="suite-run", task_name="suite", status="running",
            metric="suite_average", metric_direction="max",
            holdout={"test_sets": suite},
            started_at=datetime.now(timezone.utc),
        )
        source = Ticket(
            id="validation-infer", run_id=run.id, agent_id="inference",
            status="succeeded", lane="optimization", iteration=0,
            payload={
                "operation": "run_inference", "model_source": "base_model",
                "base_model": "owner/model", "scoring_set": "/work/validation.csv",
                "sample_submission": "/work/validation-submission.csv",
                "configuration_suggestions": {}, "configuration_pins": {},
            },
            inputs={
                "device_info": {
                    "source_ticket_id": "infra-001", "artifact_role": "device_info",
                    "work_product_id": "", "path": "",
                },
            },
        )
        data_tickets = [
            Ticket(
                id=f"data-{name}", run_id=run.id, agent_id="data",
                status="succeeded", lane="held_out_test", iteration=0,
                payload={"test_set_name": name},
            )
            for name in ("math", "qa")
        ]
        db.add_all([run, source, *data_tickets])
        await db.commit()
        ticket = await _spawn_holdout_infer(
            db, run, source, enqueue=False,
        )
        assert ticket is not None
        members = await _build_holdout_suite_members(
            ticket=ticket,
            payload=dict(ticket.payload or {}),
            work_dir="/work/holdout",
            run=run,
            session=db,
        )
        assert len(members) == 1
        assert members[0].name == "qa"
        assert members[0].configuration_pins["inference_config"][
            "inference_query"
        ] == next(
            item["inference_query"] for item in suite if item["name"] == "qa"
        )
        predictions = [
            WorkProduct(
                ticket_id=ticket.id,
                role="predictions",
                path=f"/work/holdout/{name}/predictions.csv",
                meta={"suite_member_name": name},
            )
            for name in ("math", "qa")
        ]
        db.add_all(predictions)
        await db.commit()
        evaluations = await _spawn_holdout_suite_evals(
            db, run, ticket, enqueue=False,
        )
        assert [item.payload["test_set_name"] for item in evaluations] == ["math"]
        scoring_members = await _build_evaluation_suite_members(
            evaluations[0], dict(evaluations[0].payload or {}),
            dict(evaluations[0].inputs or {}), run, db,
        )
        assert [member.name for member in scoring_members] == ["qa"]
        assert [
            item.inputs["predictions"]["work_product_id"]
            for item in evaluations
        ] == [predictions[0].id]
        assert scoring_members[0].predictions_path == predictions[1].path

    async with Session() as db:
        tickets = (await db.execute(
            select(Ticket).where(
                Ticket.run_id == "suite-run",
                Ticket.lane == "held_out_test",
                Ticket.agent_id == "inference",
            ).order_by(Ticket.id)
        )).scalars().all()
        assert len(tickets) == 1
        assert tickets[0].payload["test_set_name"] == "math"
        assert tickets[0].payload["scoring_set"] == "/work/math-public.csv"
        assert tickets[0].payload["configuration_pins"]["inference_config"][
            "inference_query"
        ] == next(
            item["inference_query"] for item in suite if item["name"] == "math"
        )

    await engine.dispose()


@pytest.mark.asyncio
async def test_heldout_suite_creates_one_data_ticket() -> None:
    from zevo.engine.run.runner import _build_data_input, _spawn_holdout_data

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    suite = [_member("math"), _member("qa", "exact_match")]
    async with Session() as db:
        run = Run(
            id="one-data-run", task_name="suite", status="running",
            metric="suite_average", metric_direction="max",
            holdout={"test_sets": suite},
            started_at=datetime.now(timezone.utc),
        )
        source = Ticket(
            id="source-infer", run_id=run.id, agent_id="inference",
            status="succeeded", lane="optimization", iteration=0,
            payload={},
        )
        db.add_all([run, source])
        await db.commit()
        ticket = await _spawn_holdout_data(db, run, source, enqueue=False)
        assert ticket is not None
        inp = _build_data_input(
            ticket, dict(ticket.payload or {}), {}, "/work/data", run,
        )
        assert inp.test_set_name == "math"
        assert [member.name for member in inp.suite_members] == ["qa"]
        rows = (await db.execute(select(Ticket).where(
            Ticket.run_id == run.id, Ticket.lane == "held_out_test",
            Ticket.agent_id == "data",
        ))).scalars().all()
        assert len(rows) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_heldout_suite_publishes_one_unweighted_average() -> None:
    from zevo.engine.run.runner import _record_holdout_score

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    suite = [
        {**_member("math"), "public": "/work/math-public.csv",
         "inference_data_profile": ""},
        {**_member("qa", "exact_match"), "public": "/work/qa-public.csv",
         "inference_data_profile": ""},
    ]
    async with Session() as db:
        run = Run(
            id="aggregate-run", task_name="suite", status="running",
            metric="suite_average", metric_direction="max",
            validation_metric="accuracy", validation_metric_direction="max",
            holdout={"test_sets": suite, "suite_results": {}, "suite_recorded": []},
            history=[{"iteration": 0, "source": "baseline", "score": 0.5}],
            started_at=datetime.now(timezone.utc),
        )
        inference = Ticket(
            id="infer-suite", run_id=run.id, agent_id="inference",
            status="succeeded", lane="held_out_test", iteration=0,
            payload={"model_source": "base_model", "base_model": "owner/model"},
        )
        evaluation = Ticket(
            id="eval-suite", run_id=run.id, agent_id="evaluation",
            status="succeeded", lane="held_out_test", iteration=0,
            payload={"test_set_name": "math", "metric": "accuracy"},
            inputs={
                "predictions": {
                    "source_ticket_id": inference.id,
                    "artifact_role": "predictions", "work_product_id": "", "path": "",
                },
            },
        )
        db.add_all([run, inference, evaluation])
        await db.commit()

        await _record_holdout_score(
            db, run, evaluation, 0.4, member_name="math",
        )
        count = (await db.execute(
            select(func.count()).select_from(ScoreEvent)
        )).scalar_one()
        assert count == 0

        await _record_holdout_score(
            db, run, evaluation, 0.8, member_name="qa",
        )
        event = (await db.execute(select(ScoreEvent))).scalar_one()
        assert event.metric_name == "suite_average"
        assert event.score == pytest.approx(0.6)
        assert event.extras == {
            "aggregation": "unweighted_mean",
            "test_sets": {
                "math": {
                    "score": 0.4, "metric": "accuracy", "metric_direction": "max",
                },
                "qa": {
                    "score": 0.8, "metric": "exact_match", "metric_direction": "max",
                },
            },
        }
        await db.refresh(run)
        assert run.history[0]["test_score"] == pytest.approx(0.6)
        assert run.history[0]["test_scores"] == {"math": 0.4, "qa": 0.8}
        assert run.history[0]["test_metrics"] == {
            "math": "accuracy", "qa": "exact_match",
        }
        assert run.history[0]["test_metric_directions"] == {
            "math": "max", "qa": "max",
        }

    await engine.dispose()
