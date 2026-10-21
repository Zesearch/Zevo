from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.contracts.configuration import load_inference_config
from zevo.contracts.inference import GenerationDiagnostics
from zevo.db.models import Base, Run, ScoreEvent, Ticket, WorkProduct
from zevo.engine.run import remeasurement as rm
import yaml
from zevo.contracts.configuration import InferenceRunConfig, InferencePromptExample, PromptMessageExample
from zevo.contracts.prompting import PromptContract, InferenceContract


def _write_inference_config(tmp_path: Path) -> Path:
    config = InferenceRunConfig(
        base_model="Qwen/Qwen3-0.6B-Base",
        prompt=PromptContract(
            prompt_framing="chat",
            model_reasoning_type="non_thinking",
            system_prompt="",
        ),
        measurement=InferenceContract(
            inference_config={
                "input_fields": ["instruction"],
                "answer_column": "prediction",
            },
        ),
        generation_backend="vllm",
        implementation_config={"trust_remote_code": False},
        tokenizer_source="Qwen/Qwen3-0.6B-Base",
        chat_template_source="tokenizer.chat_template",
        chat_template_hash="a" * 64,
        template_kwargs={},
        special_token_ids={"eos_token_id": 2},
        stop_token_ids=[2],
        prompt_example=InferencePromptExample(
            input_values={"instruction": "<INPUT:instruction>"},
            messages=[
                PromptMessageExample(
                    role="system", content="You are a helpful assistant."
                ),
                PromptMessageExample(role="user", content="<INPUT:instruction>"),
            ],
            rendered_prompt=(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<INPUT:instruction><|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
        ),
    )
    path = tmp_path / "inference_config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    return path



def diagnostics(n=20, limited=0, extra_turns=0):
    records = [{"request_index": i, "row_index": i, "turn": 1,
                "finish_reason": "length" if i < limited else "stop",
                "generated_tokens": 256} for i in range(n)]
    for _ in range(extra_turns):
        records.append({"request_index": len(records), "row_index": 0, "turn": 2,
                        "finish_reason": "length", "generated_tokens": 256})
    return GenerationDiagnostics.model_validate({"records": records})


def test_question_threshold_does_not_count_multiturn_requests():
    assert rm.question_truncation(diagnostics(limited=1, extra_turns=30))["fraction"] == .05
    assert rm.question_truncation(diagnostics(limited=2))["fraction"] > rm.THRESHOLD


def config_file(tmp_path):
    path = _write_inference_config(tmp_path)
    config = load_inference_config(path)
    config.implementation_config["llm_kwargs"] = {"max_model_len": 4096}
    path.write_text(config.model_dump_json())
    return path


def test_only_token_allowance_changes_and_pin_is_respected(tmp_path):
    config = load_inference_config(config_file(tmp_path))
    revised = rm.enlarged_config(config, {})
    old = config.model_dump()
    expected = deepcopy(old)
    expected["measurement"]["max_new_tokens"] *= 2
    assert revised.model_dump() == expected
    assert config.model_dump() == old
    with pytest.raises(ValueError, match="user-pinned"):
        rm.enlarged_config(config, {"decoding_config": {"max_new_tokens": 256}})
    config.implementation_config["llm_kwargs"]["max_model_len"] = 256
    with pytest.raises(ValueError, match="context headroom"):
        rm.enlarged_config(config, {})


@pytest_asyncio.fixture
async def scenario(tmp_path, monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as c:
        await c.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async def no_queue(*args, **kwargs):
        pass
    monkeypatch.setattr(rm, "_enqueue", no_queue)
    async with Session() as session:
        model = "Qwen/Qwen3-0.6B-Base"
        suite = [{"name": name, "validation_set": f"/{name}.csv", "test_set": f"/{name}.csv",
                  "metric": "accuracy", "metric_direction": "max", "answer_fields": ["answer"],
                  "sample_submission": f"/{name}-sample.csv"} for name in ("math", "qa")]
        run = Run(id="run", status="running", task_name="test", metric="accuracy",
                  metric_direction="max", validation_metric="accuracy", validation_metric_direction="max",
                  history=[{"iteration": i, "source": "baseline" if i == 0 else "trained",
                            "base_model": model, "score": .3 + .1 * i} for i in range(2)],
                  holdout={"validation_sets": suite}, lifecycle={})
        session.add(run)
        await session.flush()
        tickets = []
        config = config_file(tmp_path)
        for i in range(3):
            ticket = Ticket(id=f"original-{i}", run_id=run.id, agent_id="inference",
                            status="succeeded", lane="optimization", iteration=i,
                            payload={"model_source": "base_model" if i == 0 else "checkpoint",
                                     "base_model": model, "scoring_set": "/math.csv"}, inputs={})
            session.add(ticket)
            await session.flush()
            for name in ("math", "qa"):
                path = tmp_path / f"{i}-{name}.json"
                path.write_text(diagnostics(limited=2 if i == 2 and name == "math" else 0).model_dump_json())
                session.add(WorkProduct(ticket_id=ticket.id, role="generation_diagnostics", path=str(path), meta={"suite_member_name": name}))
                session.add(WorkProduct(ticket_id=ticket.id, role="inference_config", path=str(config), meta={"suite_member_name": name}))
            tickets.append(ticket)
            if i < 2:
                session.add(ScoreEvent(run_id=run.id, iteration=i, source="baseline" if i == 0 else "trained",
                                       split="validation", score=.3 + i * .1, metric_name="accuracy"))
        await session.commit()
        yield session, run, tickets, tmp_path
    await engine.dispose()


async def finish_inference(session, ticket, tmp_path, limited=0):
    ticket.status = "succeeded"
    for name in ("math", "qa"):
        path = tmp_path / f"{ticket.id}-{name}.json"
        path.write_text(diagnostics(limited=limited if name == "math" else 0).model_dump_json())
        session.add(WorkProduct(ticket_id=ticket.id, role="generation_diagnostics", path=str(path), meta={"suite_member_name": name}))
        session.add(WorkProduct(ticket_id=ticket.id, role="predictions", path=str(path), meta={"suite_member_name": name}))
    await session.commit()


async def finish_evaluation(session, ticket, tmp_path, scores=(.7, .9)):
    ticket.status = "succeeded"
    for name, score in zip(("math", "qa"), scores):
        path = tmp_path / f"{ticket.id}-{name}.json"
        path.write_text(json.dumps({"score": score}))
        session.add(WorkProduct(ticket_id=ticket.id, role="metrics", path=str(path), meta={"suite_member_name": name, "score": score}))
    await session.commit()


@pytest.mark.asyncio
async def test_complete_cohort_publishes_atomically_and_recovery_is_idempotent(scenario, monkeypatch):
    session, run, originals, root = scenario
    async def handoff(session, run, batch):
        batch["handoff_pending"] = False
        rm._store(run, batch)
        await session.commit()
    monkeypatch.setattr(rm, "_handoff", handoff)
    assert await rm.intercept_completion(session, originals[-1])
    batch = deepcopy(rm.pending(run))
    assert len(batch["cohort"]) == 3
    math = load_inference_config(batch["configs"]["math"])
    qa = load_inference_config(batch["configs"]["qa"])
    assert math.measurement.max_new_tokens == 512
    assert qa.measurement.max_new_tokens == 256
    assert not rm.configuration_override(run, originals[0], "math")
    history = deepcopy(run.history)
    for index, item in enumerate(batch["cohort"]):
        await rm.advance(session, run.id)
        inference = await session.get(Ticket, item["inference_id"])
        assert rm.configuration_override(run, inference, "math") == batch["configs"]["math"]
        await finish_inference(session, inference, root)
        await rm.intercept_completion(session, inference)
        active = rm.pending(run)
        evaluation = await session.get(Ticket, active["cohort"][index]["evaluation_id"])
        assert run.history == history
        assert len((await session.scalars(select(ScoreEvent))).all()) == 2
        await finish_evaluation(session, evaluation, root, scores=(.5 + index * .1, .7))
        await rm.intercept_completion(session, evaluation)
    assert rm.pending(run) is None
    assert rm.batches(run)[0]["status"] == "published"
    scores = list((await session.scalars(select(ScoreEvent).order_by(ScoreEvent.iteration))).all())
    assert [s.score for s in scores] == pytest.approx([.6, .65, .7])
    assert len(rm.batches(run)[0]["previous_score_events"]) == 2
    assert all(s.extras["measurement_revision"] == batch["id"] for s in scores)
    assert run.best_validation_score == pytest.approx(.7)
    assert rm.configuration_override(run, originals[0], "math") == batch["configs"]["math"]
    await rm.advance(session, run.id)
    assert await rm.intercept_completion(session, originals[-1])
    assert len((await session.scalars(select(ScoreEvent))).all()) == 3


@pytest.mark.asyncio
async def test_failure_keeps_all_old_scores_and_contracts(scenario):
    session, run, originals, root = scenario
    old = deepcopy(run.history)
    await rm.intercept_completion(session, originals[-1])
    batch = rm.pending(run)
    ticket = await session.get(Ticket, batch["cohort"][0]["inference_id"])
    ticket.status = "failed"
    await session.commit()
    await rm.intercept_completion(session, ticket)
    assert run.status == "degraded"
    assert run.history == old
    await session.refresh(originals[0])
    assert not rm.configuration_override(run, originals[0], "math")
    assert len((await session.scalars(select(ScoreEvent))).all()) == 2


@pytest.mark.asyncio
async def test_persistent_truncation_replays_whole_cohort_then_stops(scenario):
    session, run, originals, root = scenario
    await rm.intercept_completion(session, originals[-1])
    for attempt in range(2):
        await rm.advance(session, run.id)
        batch = deepcopy(rm.pending(run))
        ticket = await session.get(Ticket, batch["cohort"][0]["inference_id"])
        await finish_inference(session, ticket, root, limited=5)
        await rm.intercept_completion(session, ticket)
        if attempt == 0:
            new = rm.pending(run)
            assert [c["source_id"] for c in new["cohort"]] == [t.id for t in originals]
            assert load_inference_config(new["configs"]["math"]).measurement.max_new_tokens == 1024
    assert run.status == "degraded"
    assert len((await session.scalars(select(ScoreEvent))).all()) == 2


@pytest.mark.asyncio
async def test_cancelled_run_does_not_launch_more_work(scenario):
    session, run, originals, root = scenario
    await rm.intercept_completion(session, originals[-1])
    run.status = "cancelled"
    await session.commit()
    before = len((await session.scalars(select(Ticket))).all())
    await rm.advance(session, run.id)
    assert len((await session.scalars(select(Ticket))).all()) == before


@pytest.mark.asyncio
async def test_publication_failure_rolls_back_earlier_checkpoint_updates(scenario, monkeypatch):
    session, run, originals, root = scenario
    from zevo.engine.run import runner
    record = runner._record_validation_score
    calls = 0
    async def failing_record(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected publication failure")
        return await record(*args, **kwargs)
    monkeypatch.setattr(runner, "_record_validation_score", failing_record)
    old = deepcopy(run.history)
    await rm.intercept_completion(session, originals[-1])
    cohort = deepcopy(rm.pending(run)["cohort"])
    for index, item in enumerate(cohort):
        inference = await session.get(Ticket, item["inference_id"])
        await finish_inference(session, inference, root)
        await rm.advance(session, run.id)
        evaluation = await session.get(Ticket, rm.pending(run)["cohort"][index]["evaluation_id"])
        await finish_evaluation(session, evaluation, root)
        await rm.advance(session, run.id)
    assert run.status == "degraded"
    assert run.history == old
    assert sorted(s.score for s in (await session.scalars(select(ScoreEvent))).all()) == [.3, .4]


@pytest.mark.asyncio
async def test_test_lane_replaces_test_scores_without_exposing_them_to_agents(scenario, monkeypatch):
    session, run, originals, root = scenario
    from zevo.api.routers.shared.runs import _summary
    from zevo.engine.run.runner import _agent_history
    holdout = deepcopy(run.holdout)
    holdout["test_sets"] = holdout.pop("validation_sets")
    run.holdout = holdout
    for ticket in originals:
        ticket.lane = "held_out_test"
        ticket.payload = {**ticket.payload, "test_set_name": "math"}
    await session.commit()
    async def handoff(session, run, batch):
        batch["handoff_pending"] = False
        rm._store(run, batch)
        await session.commit()
    monkeypatch.setattr(rm, "_handoff", handoff)
    old_validation = [r["score"] for r in run.history]
    await rm.intercept_completion(session, originals[-1])
    cohort = deepcopy(rm.pending(run)["cohort"])
    for index, item in enumerate(cohort):
        inference = await session.get(Ticket, item["inference_id"])
        await finish_inference(session, inference, root)
        await rm.advance(session, run.id)
        evaluation = await session.get(Ticket, rm.pending(run)["cohort"][index]["evaluation_id"])
        await finish_evaluation(session, evaluation, root)
        await rm.advance(session, run.id)
    assert rm.batches(run)[0]["status"] == "published"
    assert [r["score"] for r in run.history] == old_validation
    assert len((await session.scalars(select(ScoreEvent).where(ScoreEvent.split == "test"))).all()) == 3
    assert all("test_score" not in row for row in _agent_history(run.history))
    assert rm.KEY not in _summary(run, holdout=False).lifecycle
    assert rm.KEY in _summary(run, holdout=True).lifecycle


@pytest.mark.asyncio
async def test_input_builder_uses_revision_for_baseline_and_checkpoint(scenario, monkeypatch):
    session, run, originals, root = scenario
    from zevo.engine.run import runner
    from zevo.contracts.infrastructure import SlurmStageJobContract
    async def no_members(**kwargs):
        return []
    async def no_job(**kwargs):
        return SlurmStageJobContract()
    monkeypatch.setattr(runner, "_build_validation_suite_members", no_members)
    monkeypatch.setattr(runner, "_build_holdout_suite_members", no_members)
    monkeypatch.setattr(runner, "_slurm_stage_job_contract", no_job)
    await rm.intercept_completion(session, originals[-1])
    batch = rm.pending(run)
    baseline = await session.get(Ticket, batch["cohort"][0]["inference_id"])
    kwargs = dict(payload=baseline.payload, work_dir=str(root), run=run, session=session, generation_backend="vllm",
                  inputs={"device_info": {"path": "/device.json"}})
    inp = await runner._build_inference_input(ticket=baseline, **kwargs)
    assert inp.configuration_mode == "reuse"
    assert inp.inference_config_path == batch["configs"]["math"]


@pytest.mark.asyncio
async def test_reconciler_resumes_without_a_completion_callback(scenario):
    session, run, originals, root = scenario
    await rm.intercept_completion(session, originals[-1])
    batch = deepcopy(rm.pending(run))
    inference = await session.get(Ticket, batch["cohort"][0]["inference_id"])
    await finish_inference(session, inference, root)
    run_id = run.id
    # A different DB session stands in for a restarted engine process.
    Session = async_sessionmaker(session.bind, expire_on_commit=False)
    async with Session() as restarted:
        assert await rm.reconcile(restarted) == 1
        recovered = await restarted.get(Run, run_id)
        evaluation_id = rm.pending(recovered)["cohort"][0]["evaluation_id"]
        assert (await restarted.get(Ticket, evaluation_id)).status == "queued"
        assert await rm.reconcile(restarted) == 1
        assert rm.pending(recovered)["cohort"][0]["evaluation_id"] == evaluation_id


@pytest.mark.asyncio
async def test_staged_scores_cannot_change_headline_and_old_evaluations_are_superseded(scenario):
    session, run, originals, root = scenario
    from zevo.contracts.evaluation import EvaluationResult
    from zevo.engine.run.runner import _apply_run_score_monitor
    # Single-benchmark monitoring normally updates the headline immediately.
    run.holdout = {"validation_sets": run.holdout["validation_sets"][:1]}
    run.best_validation_score = .4
    evaluation = Ticket(id="staged-eval", agent_id="evaluation", lane="optimization",
                        inputs={"predictions": {"source_ticket_id": "original-2"}})
    run.lifecycle = {rm.KEY: [{"id": "r", "status": "running", "lane": "optimization",
        "base_model": originals[-1].payload["base_model"], "configs": {},
        "cohort": [{"source_id": "original-2", "inference_id": "staged-infer", "evaluation_id": evaluation.id}]}]}
    result = EvaluationResult(ticket_id=evaluation.id, status="succeeded", metrics_path="/metrics.json", error_message="", notes="")
    _apply_run_score_monitor(run, evaluation, result, {"score": .99})
    assert run.best_validation_score == .4
    state = deepcopy(run.lifecycle)
    state[rm.KEY][0]["status"] = "published"
    run.lifecycle = state
    evaluation.id = "old-eval"
    _apply_run_score_monitor(run, evaluation, result, {"score": .99})
    assert run.best_validation_score == .4
