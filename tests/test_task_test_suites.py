from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Base, Run, ScoreEvent, Ticket, WorkProduct


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
            **_member("GSM8K"), "test_set": gsm,
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
            "math": {"score": 0.4, "metric": "accuracy"},
            "qa": {"score": 0.8, "metric": "exact_match"},
        }
        await db.refresh(run)
        assert run.history[0]["validation_scores"] == {"math": 0.4, "qa": 0.8}
        for path in metric_paths:
            stored = json.loads(path.read_text(encoding="utf-8"))
            assert stored["score"] == pytest.approx(0.6)
            assert stored["metric"] == "suite_average"
            assert set(stored["validation_sets"]) == {"math", "qa"}

    await engine.dispose()


@pytest.mark.asyncio
async def test_validation_member_selects_its_own_baseline_inference_query() -> None:
    from zevo.engine.run.runner import _spawn_validation_infer

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    validation_suite = [
        {
            "name": "math", "public": "/work/math-public.csv",
            "sample_submission": "/work/math-submission.csv",
            "inference_data_profile": "/work/math-profile.json",
            "inference_query": "Solve {question}.",
        },
        {
            "name": "qa", "public": "/work/qa-public.csv",
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

        created = await _spawn_validation_infer(db, run, primary, enqueue=False)
        assert len(created) == 1
        qa = created[0]
        assert qa.payload["test_set_name"] == "validation:qa"
        assert qa.payload["configuration_pins"]["inference_config"][
            "inference_query"
        ] == "Answer {question} with one short phrase."
        assert "inference_config" not in qa.inputs
        assert qa.inputs["inference_data_profile"] == {
            "artifact_role": "inference_data_profile",
            "path": "/work/qa-profile.json",
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

    await engine.dispose()


@pytest.mark.asyncio
async def test_heldout_suite_runs_each_inference_query_independently() -> None:
    from zevo.engine.run.runner import _spawn_holdout_infer

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
        await _spawn_holdout_infer(db, run, source, enqueue=False)

    async with Session() as db:
        tickets = (await db.execute(
            select(Ticket).where(
                Ticket.run_id == "suite-run",
                Ticket.lane == "held_out_test",
                Ticket.agent_id == "inference",
            ).order_by(Ticket.id)
        )).scalars().all()
        assert len(tickets) == 2
        by_name = {ticket.payload["test_set_name"]: ticket for ticket in tickets}
        assert by_name["math"].payload["scoring_set"] == "/work/math-public.csv"
        assert by_name["qa"].payload["scoring_set"] == "/work/qa-public.csv"
        for name, ticket in by_name.items():
            assert ticket.payload["configuration_pins"]["inference_config"][
                "inference_query"
            ] == next(item["inference_query"] for item in suite if item["name"] == name)

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
        infers = [
            Ticket(
                id=f"infer-{name}", run_id=run.id, agent_id="inference",
                status="succeeded", lane="held_out_test", iteration=0,
                payload={
                    "model_source": "base_model", "base_model": "owner/model",
                    "test_set_name": name,
                },
            )
            for name in ("math", "qa")
        ]
        evals = [
            Ticket(
                id=f"eval-{name}", run_id=run.id, agent_id="evaluation",
                status="succeeded", lane="held_out_test", iteration=0,
                payload={"test_set_name": name, "metric": metric},
                inputs={
                    "predictions": {
                        "source_ticket_id": f"infer-{name}",
                        "artifact_role": "predictions", "work_product_id": "", "path": "",
                    },
                },
            )
            for name, metric in (("math", "accuracy"), ("qa", "exact_match"))
        ]
        db.add_all([run, *infers, *evals])
        await db.commit()

        await _record_holdout_score(db, run, evals[0], 0.4)
        count = (await db.execute(
            select(func.count()).select_from(ScoreEvent)
        )).scalar_one()
        assert count == 0

        await _record_holdout_score(db, run, evals[1], 0.8)
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
