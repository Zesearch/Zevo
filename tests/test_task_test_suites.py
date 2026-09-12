from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Base, Run, ScoreEvent, Ticket


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


def test_inference_query_renders_fields_without_treating_json_as_a_placeholder() -> None:
    from zevo.contracts.prompting import render_inference_query

    rendered = render_inference_query(
        'Solve {question}. Return JSON like {"answer": "..."}.',
        {"question": "2 + 2"},
    )
    assert rendered == 'Solve 2 + 2. Return JSON like {"answer": "..."}.'


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
