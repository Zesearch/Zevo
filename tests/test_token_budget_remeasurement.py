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
    assert rm.enlarged_config(config, {}).measurement.max_new_tokens == 256
    config.implementation_config["llm_kwargs"] = {}
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
    run = await session.get(Run, ticket.run_id)
    tmp_path = Path(rm.attempt_work_dir(run, ticket.id, str(tmp_path / ticket.id)))
    tmp_path.mkdir(parents=True, exist_ok=True)
    retained = rm.replay_item(run, ticket.id).get("retained", {}) if rm.active_attempt(run, ticket.id) else {}
    for name in ("math", "qa"):
        if name in retained:
            continue
        path = tmp_path / f"{ticket.id}-{name}.json"
        path.write_text(diagnostics(limited=limited if name == "math" else 0).model_dump_json())
        session.add(WorkProduct(ticket_id=ticket.id, role="generation_diagnostics", path=str(path), meta={"suite_member_name": name}))
        session.add(WorkProduct(ticket_id=ticket.id, role="predictions", path=str(path), meta={
            "suite_member_name": name, "configuration": load_inference_config(
                rm.configuration_override(run, ticket, name)).model_dump(mode="json")}))
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
    assert {c['inference_id'] for c in batch['cohort']} == {t.id for t in originals}
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
        await finish_inference(session, inference, root, limited=10 if index else 0)
        await rm.intercept_completion(session, inference)
        assert len(rm.batches(run)) == 1  # Historical truncation cannot reopen expansion.
        active = rm.pending(run)
        evaluation = await session.get(Ticket, active["cohort"][index]["evaluation_id"])
        assert run.history == history
        assert len((await session.scalars(select(ScoreEvent))).all()) == 2
        await finish_evaluation(session, evaluation, root, scores=(.5 + inference.iteration * .1, .7))
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
    assert len((await session.scalars(select(Ticket).where(Ticket.agent_id == 'inference'))).all()) == 3


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
@pytest.mark.parametrize("stalls", [False, True])
async def test_current_reaches_ceiling_then_history_is_scored_without_expansion(scenario, monkeypatch, stalls):
    session, run, originals, root = scenario
    async def handoff(session, run, batch):
        batch["handoff_pending"] = False
        rm._store(run, batch)
        await session.commit()
    monkeypatch.setattr(rm, "_handoff", handoff)
    await rm.intercept_completion(session, originals[-1])
    caps = (512, 1024) if stalls else (512, 1024, 2048, 4096)
    for attempt, cap in enumerate(caps):
        await rm.advance(session, run.id)
        batch = deepcopy(rm.pending(run))
        assert load_inference_config(batch['configs']['math']).measurement.max_new_tokens == cap
        assert batch['cohort'][0]['source_id'] == originals[-1].id
        for old in batch['cohort'][1:]:
            assert (await session.get(Ticket, old['inference_id'])).status == 'succeeded'
            assert not old.get('inference_attempt')
        current = await session.get(Ticket, batch['cohort'][0]['inference_id'])
        await finish_inference(session, current, root, limited=5 if stalls else 9-attempt)
        await rm.advance(session, run.id)
    batch = deepcopy(rm.pending(run))
    assert batch is not None, rm.batches(run)[-1].get('error')
    assert batch['budget_finalized']
    assert 'math' in batch['stalled_members' if stalls else 'ceiling_evidence']
    assert len(batch['budget_progress']) == len(caps)
    assert len({p['revision'] for p in batch['budget_progress']}) == len(caps)
    assert len({p['inference_id'] for p in batch['budget_progress']}) == 1
    assert len((await session.scalars(select(Ticket).where(Ticket.agent_id == 'inference'))).all()) == 3
    products = await rm._products(session, originals[-1].id, 'generation_diagnostics')
    # Original and every changed-member attempt remain readable; unchanged qa is reused.
    assert len({p.path for p in products}) == len(caps) + 3
    assert all(Path(p.path).is_file() for p in products)
    # Recovery callbacks cannot count one inference as multiple attempts.
    await rm.advance(session, run.id)
    assert len(rm.pending(run)['budget_progress']) == len(caps)
    for index, item in enumerate(batch['cohort']):
        await rm.advance(session, run.id)
        if index:
            inference = await session.get(Ticket, item['inference_id'])
            # History must not reopen expansion, even with substantial truncation.
            await finish_inference(session, inference, root, limited=10)
            await rm.advance(session, run.id)
        evaluation = await session.get(Ticket, rm.pending(run)['cohort'][index]['evaluation_id'])
        await finish_evaluation(session, evaluation, root)
        await rm.advance(session, run.id)
    assert run.status == 'running'
    assert rm.batches(run)[-1]['status'] == 'published'
    assert len((await session.scalars(select(ScoreEvent))).all()) == 3


@pytest.mark.asyncio
async def test_current_already_at_ceiling_continues_normal_evaluation(scenario):
    session, run, originals, root = scenario
    path = Path((await rm._products(session, originals[0].id, 'inference_config'))[0].path)
    config = load_inference_config(path)
    config.measurement.max_new_tokens = 4096
    path.write_text(config.model_dump_json())
    assert not await rm.intercept_completion(session, originals[-1])
    assert run.status == 'running'
    assert not rm.batches(run)


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
    baseline = await rm._new_inference(session, run, batch, batch["cohort"][1])
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
async def test_existing_evaluation_is_requeued_without_reusing_old_metrics(scenario):
    session, run, originals, root = scenario
    evaluation = Ticket(id='existing-eval', run_id=run.id, agent_id='evaluation',
                        lane='optimization', iteration=2, status='succeeded',
                        payload={'test_set_name': 'validation:math'},
                        inputs={'predictions': {'source_ticket_id': originals[-1].id, 'path': '/old.csv'}})
    session.add(evaluation)
    await session.flush()
    await finish_evaluation(session, evaluation, root, scores=(.99, .99))
    old_products = {p.id for p in await rm._products(session, evaluation.id, 'metrics')}
    await rm.intercept_completion(session, originals[-1])
    await finish_inference(session, originals[-1], root)
    await rm.advance(session, run.id)
    item = rm.pending(run)['cohort'][0]
    assert item['evaluation_id'] == evaluation.id
    assert evaluation.status == 'queued'
    assert evaluation.inputs['predictions']['path'] != '/old.csv'
    assert old_products <= set(item['evaluation_attempt']['previous_product_ids'])
    assert await rm._products(session, evaluation.id, 'metrics', run) == []
    assert await rm._products(session, evaluation.id, 'metrics')  # Audit is retained.
    # A stale success with no new metrics must never publish the old .99 score.
    evaluation.status = 'succeeded'
    await session.commit()
    with pytest.raises(ValueError, match='complete evaluation suite'):
        await rm._scores(session, evaluation, rm._suite(run, 'optimization'), run)


@pytest.mark.asyncio
async def test_evaluation_suite_binds_only_current_attempt_predictions(scenario):
    from zevo.engine.run.runner import _build_evaluation_suite_members
    session, run, originals, root = scenario
    source = originals[-1]
    session.add(WorkProduct(ticket_id=source.id, role='predictions', path='/stale-qa.csv',
                            meta={'suite_member_name': 'qa'}))
    await session.commit()
    await rm.intercept_completion(session, source)
    await finish_inference(session, source, root)
    await rm.advance(session, run.id)
    evaluation = await session.get(Ticket, rm.pending(run)['cohort'][0]['evaluation_id'])
    members = await _build_evaluation_suite_members(evaluation, evaluation.payload,
                                                    evaluation.inputs, run, session)
    assert len(members) == 1
    assert members[0].name == 'qa'
    assert 'measurement-attempts' in members[0].predictions_path
    assert members[0].predictions_path != '/stale-qa.csv'


def test_legacy_ticket_provenance_does_not_cycle_after_reuse():
    run = Run(lifecycle={rm.KEY: [
        {'cohort': [{'source_id': 'original', 'inference_id': 'remeasure-old'}]},
        {'reuse_tickets': True, 'cohort': [{'source_id': 'remeasure-old', 'inference_id': 'original'}]},
    ]})
    assert rm._original_ticket_id(run, 'remeasure-old') == 'original'
    assert rm._original_ticket_id(run, 'original') == 'original'


@pytest.mark.asyncio
async def test_reopened_inference_cannot_complete_with_previous_diagnostics(scenario):
    session, run, originals, root = scenario
    old_history = deepcopy(run.history)
    await rm.intercept_completion(session, originals[-1])
    assert originals[-1].status == 'queued'
    assert await rm._products(session, originals[-1].id, 'generation_diagnostics', run) == []
    originals[-1].status = 'succeeded'
    await session.commit()
    await rm.advance(session, run.id)
    assert run.status == 'degraded'
    assert run.history == old_history
    assert 'complete question-level diagnostics' in rm.batches(run)[-1]['error']


@pytest.mark.asyncio
async def test_later_iteration_reuses_previously_published_model_tickets(scenario, monkeypatch):
    session, run, originals, root = scenario
    async def handoff(session, run, batch):
        batch['handoff_pending'] = False
        rm._store(run, batch)
        await session.commit()
    monkeypatch.setattr(rm, '_handoff', handoff)

    async def complete():
        cohort = deepcopy(rm.pending(run)['cohort'])
        for item in cohort:
            await rm.advance(session, run.id)
            inference = await session.get(Ticket, item['inference_id'])
            await finish_inference(session, inference, root)
            await rm.advance(session, run.id)
            assert rm.pending(run), rm.batches(run)[-1].get('error')
            active = next(c for c in rm.pending(run)['cohort'] if c['inference_id'] == inference.id)
            evaluation = await session.get(Ticket, active['evaluation_id'])
            await finish_evaluation(session, evaluation, root)
            await rm.advance(session, run.id)

    await rm.intercept_completion(session, originals[-1])
    await complete()
    published = deepcopy(rm.batches(run)[-1])
    new = Ticket(id='original-3', run_id=run.id, agent_id='inference', status='succeeded',
                 lane='optimization', iteration=3,
                 payload={**deepcopy(originals[-1].payload), 'test_set_name': ''}, inputs={})
    session.add(new)
    await session.flush()
    for name, path in published['configs'].items():
        session.add(WorkProduct(ticket_id=new.id, role='inference_config', path=path, meta={'suite_member_name': name}))
    await finish_inference(session, new, root, limited=4)
    await rm.intercept_completion(session, new)
    assert {c['inference_id'] for c in rm.pending(run)['cohort']} == {t.id for t in originals} | {new.id}
    await complete()
    assert rm.batches(run)[-1]['status'] == 'published'
    assert not rm.is_unpublished_replay(run, originals[0])
    assert len((await session.scalars(select(Ticket).where(Ticket.agent_id == 'inference'))).all()) == 4
    assert len((await session.scalars(select(Ticket).where(Ticket.agent_id == 'evaluation'))).all()) == 4


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
    evaluation.id = "old-eval"
    _apply_run_score_monitor(run, evaluation, result, {"score": .99})
    assert run.best_validation_score == .4  # Delayed prior evaluation before its new attempt.
    state = deepcopy(run.lifecycle)
    state[rm.KEY][0]["status"] = "published"
    run.lifecycle = state
    evaluation.id = "old-eval"
    _apply_run_score_monitor(run, evaluation, result, {"score": .99})
    assert run.best_validation_score == .4


async def add_looping_outputs(session, source, root, member='math'):
    import csv
    products = await rm._products(session, source.id, 'generation_diagnostics')
    product = next(p for p in products if p.meta['suite_member_name'] == member)
    Path(product.path).write_text(diagnostics(limited=10).model_dump_json())
    path = root / f'{source.id}-{member}-pred.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'complete_prediction'])
        w.writeheader()
        w.writerows({'id': i, 'complete_prediction': 'These exact words keep repeating without solving the problem. ' * 100 if i < 10 else 'done'} for i in range(20))
    session.add(WorkProduct(ticket_id=source.id, role='predictions', path=str(path), meta={'suite_member_name': member}))
    await session.commit()


@pytest.mark.asyncio
async def test_repetitive_current_checkpoint_does_not_expand(scenario):
    session, run, originals, root = scenario
    await add_looping_outputs(session, originals[-1], root)
    evidence = await rm._evidence(session, run, originals[-1])
    assert evidence['math']['repetition']['dominated']
    assert not await rm.intercept_completion(session, originals[-1])
    assert not rm.batches(run)
    assert run.status == 'running'


@pytest.mark.asyncio
async def test_iteration_two_retains_repetitive_baseline_but_replays_iteration_one(scenario):
    session, run, originals, root = scenario
    await add_looping_outputs(session, originals[0], root)
    run.holdout = {**run.holdout, 'validation_suite_results': {
        f"baseline|0|{originals[0].payload['base_model']}": {
            'math': {'score': .2, 'metric': 'accuracy', 'metric_direction': 'max'},
            'qa': {'score': .4, 'metric': 'accuracy', 'metric_direction': 'max'},
        }}}
    await session.commit()
    assert await rm.intercept_completion(session, originals[-1])
    batch = rm.pending(run)
    current, baseline, iteration_one = batch['cohort']
    assert baseline['retained_only']
    assert baseline['retained']['math']['evidence']['repetition']['dominated']
    assert not iteration_one['retained_only']
    assert not current['retained_only']
    assert (await session.get(Ticket, baseline['inference_id'])).status == 'succeeded'
    assert (await session.get(Ticket, iteration_one['inference_id'])).status == 'succeeded'
    assert not baseline.get('inference_attempt')
    assert not iteration_one.get('inference_attempt')
    assert await session.get(Ticket, current['inference_id']) is not None


@pytest.mark.asyncio
async def test_partial_repetition_exemption_preserves_score_and_selects_other_primary(scenario, monkeypatch):
    session, run, originals, root = scenario
    await add_looping_outputs(session, originals[0], root)
    # Both benchmarks expand; only baseline math has repetition evidence.
    qa_diag = next(p for p in await rm._products(session, originals[-1].id, 'generation_diagnostics') if p.meta['suite_member_name']=='qa')
    Path(qa_diag.path).write_text(diagnostics(limited=3).model_dump_json())
    run.holdout = {**run.holdout, 'validation_suite_results': {
        f"baseline|0|{originals[0].payload['base_model']}": {
            'math': {'score': .2, 'metric': 'accuracy', 'metric_direction': 'max'},
            'qa': {'score': .4, 'metric': 'accuracy', 'metric_direction': 'max'},
        }}, 'validation_sets': [{**m, 'public': m['validation_set'], 'inference_data_profile': '/profile.json'} for m in run.holdout['validation_sets']]}
    await session.commit()
    async def handoff(session, run, batch):
        batch['handoff_pending'] = False
        rm._store(run, batch)
        await session.commit()
    monkeypatch.setattr(rm, '_handoff', handoff)
    await rm.intercept_completion(session, originals[-1])
    batch = deepcopy(rm.pending(run))
    baseline = batch['cohort'][1]
    assert baseline['active_members'] == ['qa']
    replay = await rm._new_inference(session, run, batch, baseline)
    assert replay.payload['test_set_name'] == 'validation:qa'
    from zevo.engine.run.scheduler.bindings import resolve_input_bindings
    resolved = await resolve_input_bindings(session, run_id=run.id, inputs=replay.inputs)
    assert resolved['inference_data_profile']['artifact_role'] == 'inference_data_profile'
    assert resolved['inference_data_profile']['path'] == '/profile.json'
    assert rm.configuration_override(run, replay, 'math') == baseline['retained']['math']['config_path']
    assert load_inference_config(rm.configuration_override(run, replay, 'qa')).measurement.max_new_tokens == 512
    for i, item in enumerate(batch['cohort']):
        await rm.advance(session, run.id)
        replay = await session.get(Ticket, item['inference_id'])
        replay.status = 'succeeded'
        for name in item['active_members']:
            path = root / f'{replay.id}-{name}.json'
            path.write_text(diagnostics().model_dump_json())
            for role in ['predictions', 'generation_diagnostics']:
                session.add(WorkProduct(ticket_id=replay.id, role=role, path=str(path), meta={'suite_member_name': name,
                    'configuration': load_inference_config(batch['configs'][name]).model_dump(mode='json')}))
        await session.commit()
        await rm.intercept_completion(session, replay)
        evaluation = await session.get(Ticket, rm.pending(run)['cohort'][i]['evaluation_id'])
        evaluation.status = 'succeeded'
        for name in item['active_members']:
            session.add(WorkProduct(ticket_id=evaluation.id, role='metrics', path=str(root / f'{evaluation.id}-{name}-metrics.json'), meta={'suite_member_name': name, 'score': .8}))
        await session.commit()
        await rm.intercept_completion(session, evaluation)
    assert rm.batches(run)[0]['status'] == 'published', rm.batches(run)[0].get('error')
    scores = (await session.scalars(select(ScoreEvent).order_by(ScoreEvent.iteration))).all()
    assert [s.score for s in scores] == pytest.approx([.5,.8,.8])
    assert 'math' in scores[0].extras['repetition_exemptions']


@pytest.mark.asyncio
async def test_replay_input_builders_exclude_retained_primary(scenario):
    from zevo.engine.run.runner import _build_validation_suite_members, _build_evaluation_suite_members
    session, run, originals, root = scenario
    await add_looping_outputs(session, originals[0], root)
    for p in await rm._products(session, originals[-1].id, 'generation_diagnostics'):
        Path(p.path).write_text(diagnostics(limited=4).model_dump_json())
    run.holdout = {**run.holdout, 'validation_suite_results': {
        f"baseline|0|{originals[0].payload['base_model']}": {
            'math': {'score': .2}, 'qa': {'score': .4}}},
        'validation_sets': [{**m, 'public': m['validation_set'], 'inference_data_profile': '/profile.json'} for m in run.holdout['validation_sets']]}
    await session.commit()
    await rm.intercept_completion(session, originals[-1])
    item = rm.pending(run)['cohort'][1]
    inference = await rm._new_inference(session, run, rm.pending(run), item)
    members = await _build_validation_suite_members(ticket=inference, payload=inference.payload, work_dir=str(root), run=run, session=session)
    assert not members  # qa is primary; retained math must not be submitted again.
    evaluation = Ticket(id='test-eval', run_id=run.id, agent_id='evaluation', lane='optimization',
                        payload={'test_set_name': 'validation:qa'})
    batch = deepcopy(rm.pending(run));batch['cohort'][1]['evaluation_id'] = evaluation.id
    rm._store(run, batch)
    members = await _build_evaluation_suite_members(evaluation, evaluation.payload,
        {'predictions': {'source_ticket_id': inference.id}}, run, session)
    assert not members  # Retained math must not be rescored either.


@pytest.mark.asyncio
async def test_previous_iteration_can_be_exempt_too(scenario):
    session, run, originals, root = scenario
    await add_looping_outputs(session, originals[1], root)
    run.holdout = {**run.holdout, 'validation_suite_results': {
        f"trained|1|{originals[1].payload['base_model']}": {
            'math': {'score': .2}, 'qa': {'score': .6}}}}
    await session.commit()
    await rm.intercept_completion(session, originals[2])
    batch = rm.pending(run)
    assert not batch['cohort'][0]['retained_only']
    assert not batch['cohort'][1]['retained_only']
    assert batch['cohort'][2]['retained_only']


@pytest.mark.asyncio
async def test_fully_exempt_baseline_score_survives_publish(scenario, monkeypatch):
    session, run, originals, root = scenario
    await add_looping_outputs(session, originals[0], root)
    run.holdout = {**run.holdout, 'validation_suite_results': {
        f"baseline|0|{originals[0].payload['base_model']}": {
            'math': {'score': .2}, 'qa': {'score': .4}}}}
    await session.commit()
    async def handoff(session, run, batch):
        batch['handoff_pending'] = False
        rm._store(run, batch)
        await session.commit()
    monkeypatch.setattr(rm, '_handoff', handoff)
    await rm.intercept_completion(session, originals[2])
    batch = deepcopy(rm.pending(run))
    for index, item in enumerate(batch['cohort']):
        if item.get('retained_only'):
            continue
        await rm.advance(session, run.id)
        inference = await session.get(Ticket, item['inference_id'])
        await finish_inference(session, inference, root)
        await rm.intercept_completion(session, inference)
        evaluation = await session.get(Ticket, rm.pending(run)['cohort'][index]['evaluation_id'])
        await finish_evaluation(session, evaluation, root)
        await rm.intercept_completion(session, evaluation)
    assert rm.batches(run)[0]['status'] == 'published'
    scores = (await session.scalars(select(ScoreEvent).order_by(ScoreEvent.iteration))).all()
    assert [s.score for s in scores] == pytest.approx([.3, .8, .8])
    assert 'measurement_revision' not in (scores[0].extras or {})


def test_budget_stagnation_is_per_member_and_decrease_resets_streak(tmp_path):
    path = str(config_file(tmp_path))
    batch = {'configs': {'math': path, 'qa': path},
             'evidence': {'math': {'fraction': .3}, 'qa': {'fraction': .3}},
             'expanded_members': ['math', 'qa']}
    rm._record_budget_progress(batch, 'one', {'math': {'fraction': .3}, 'qa': {'fraction': .4}})
    assert batch['no_improvement_streaks'] == {'math': 1, 'qa': 1}
    batch['evidence'] = {'math': {'fraction': .3}, 'qa': {'fraction': .4}}
    rm._record_budget_progress(batch, 'two', {'math': {'fraction': .2}, 'qa': {'fraction': .4}})
    assert batch['no_improvement_streaks'] == {'math': 0, 'qa': 2}
    assert set(batch['stalled_members']) == {'qa'}
    batch['expanded_members'] = ['math']
    batch['evidence'] = {'math': {'fraction': .2}, 'qa': {'fraction': .4}}
    rm._record_budget_progress(batch, 'three', {'math': {'fraction': .2}, 'qa': {'fraction': .4}})
    assert batch['no_improvement_streaks'] == {'math': 1, 'qa': 2}


@pytest.mark.parametrize("previous,current,improved", [
    (.20, .199, False), (.20, .181, False), (.20, .18, True),
    (.30, .27, True), (.20, .15, True), (.20, .21, False),
    (0.0, 0.0, False),
])
def test_budget_progress_requires_meaningful_relative_drop(tmp_path, previous, current, improved):
    path = str(config_file(tmp_path))
    batch = {'configs': {'math': path}, 'evidence': {'math': {'fraction': previous}},
             'expanded_members': ['math'], 'no_improvement_streaks': {'math': 1}}
    rm._record_budget_progress(batch, 'one', {'math': {'fraction': current}})
    assert batch['no_improvement_streaks']['math'] == (0 if improved else 2)
    assert ('math' in batch.get('stalled_members', {})) == (not improved)
    assert batch['budget_progress'][0]['meaningful_improvement'] is improved


def test_stall_waits_until_the_budget_reaches_the_floor_another_lane_settled(tmp_path):
    # Test lane started at 32 tokens and doubled twice with a flat ~44% truncation
    # curve while Validation had already settled the same model at 512.
    path = str(config_file(tmp_path))
    batch = {'configs': {'test': path}, 'evidence': {'test': {'fraction': .4378}},
             'expanded_members': ['test'], 'no_improvement_streaks': {'test': 1}}
    rm._record_budget_progress(batch, 'one', {'test': {'fraction': .4360}}, floor=1024)
    assert batch['no_improvement_streaks']['test'] == 2
    assert 'stalled_members' not in batch
    assert batch['budget_progress'][0]['below_settled_floor'] is True
    assert batch['budget_progress'][0]['settled_budget_floor'] == 1024
    # The same flat curve at or above the floor is a real stall.
    batch = {'configs': {'test': path}, 'evidence': {'test': {'fraction': .4378}},
             'expanded_members': ['test'], 'no_improvement_streaks': {'test': 1}}
    cap = load_inference_config(path).measurement.max_new_tokens
    rm._record_budget_progress(batch, 'two', {'test': {'fraction': .4360}}, floor=cap)
    assert 'test' in batch['stalled_members']
    assert batch['budget_progress'][0]['below_settled_floor'] is False


def test_settled_budget_floor_reads_published_batches_of_the_same_model_only():
    run = Run(id='r', task_name='t', run_name='r', status='running', lifecycle={rm.KEY: [
        {'id': 'a', 'status': 'published', 'lane': 'optimization', 'base_model': 'm',
         'budget_progress': [{'max_new_tokens': 256}, {'max_new_tokens': 512}]},
        {'id': 'b', 'status': 'superseded', 'lane': 'optimization', 'base_model': 'm',
         'budget_progress': [{'max_new_tokens': 4096}]},
        {'id': 'c', 'status': 'published', 'lane': 'held_out_test', 'base_model': 'other',
         'budget_progress': [{'max_new_tokens': 2048}]},
    ]})
    assert rm.settled_budget_floor(run, {'base_model': 'm'}) == 512
    assert rm.settled_budget_floor(run, {'base_model': 'other'}) == 2048
    assert rm.settled_budget_floor(run, {'base_model': 'none'}) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["optimization", "held_out_test"])
async def test_checkpoint_replay_uses_staged_suite_configs_before_baseline_lookup(scenario, lane):
    from zevo.engine.run.runner import _build_validation_suite_members, _build_holdout_suite_members
    session, run, originals, root = scenario
    suite = [{**m, 'public': m['validation_set'], 'inference_data_profile': '/profile.json'}
             for m in run.holdout['validation_sets']]
    run.holdout = {**run.holdout, 'validation_sets': suite, 'test_sets': suite}
    if lane == 'held_out_test':
        for ticket in originals:
            ticket.lane = lane
            ticket.payload = {**ticket.payload, 'test_set_name': 'math'}
    await session.commit()
    assert await rm.intercept_completion(session, originals[-1])
    batch = rm.pending(run)
    item = batch['cohort'][0]
    replay = await rm._new_inference(session, run, batch, item)
    builder = _build_holdout_suite_members if lane == 'held_out_test' else _build_validation_suite_members
    members = await builder(ticket=replay, payload=replay.payload, work_dir=str(root), run=run, session=session)
    assert [m.name for m in members] == ['qa']
    assert members[0].inference_config_path == batch['configs']['qa']
    assert members[0].configuration_mode == 'reuse'


async def add_reusable_suite(session, run, originals, root, lane, scored):
    import csv
    suite = [{**m, 'public': m['validation_set'], 'inference_data_profile': '/profile.json'}
             for m in run.holdout['validation_sets']]
    results = {}
    for ticket in originals:
        ticket.lane = lane
        ticket.payload = {**ticket.payload, 'test_set_name': 'math' if lane == 'held_out_test' else ''}
        for name in ('math', 'qa'):
            path = root / f'{ticket.id}-{name}-predictions.csv'
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'complete_prediction'])
                writer.writeheader()
                writer.writerows({'id': i, 'complete_prediction': 'answer'} for i in range(20))
            session.add(WorkProduct(ticket_id=ticket.id, role='predictions', path=str(path),
                                    meta={'suite_member_name': name,
                                          'configuration': load_inference_config(root / 'inference_config.yaml').model_dump(mode='json')}))
        if scored:
            source = 'baseline' if ticket.iteration == 0 else 'trained'
            results[f"{source}|{ticket.iteration}|{ticket.payload['base_model']}"] = {
                'math': {'score': .2, 'evaluation_ticket_id': 'old-math-eval',
                         'prediction_work_product_id': 'old-math-pred',
                         'test_contract': {'legacy': 'math'},
                         'inference_identity': {'measurement': {'max_new_tokens': 256}}},
                'qa': {'score': .4, 'evaluation_ticket_id': 'old-qa-eval',
                       'prediction_work_product_id': 'old-qa-pred',
                       'test_contract': {'legacy': 'qa'},
                       'inference_identity': {'measurement': {'max_new_tokens': 256}}}}
    prefix = 'suite' if lane == 'held_out_test' else 'validation_suite'
    run.holdout = {**run.holdout, 'validation_sets': suite, 'test_sets': suite,
                   f'{prefix}_results': results}
    await session.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize('lane', ['optimization', 'held_out_test'])
@pytest.mark.parametrize('scored', [False, True])
@pytest.mark.parametrize('changed', ['math', 'qa'])
async def test_unchanged_member_reuses_predictions_and_only_missing_scores(scenario, monkeypatch, lane, scored, changed):
    from zevo.engine.run.runner import _build_validation_suite_members, _build_holdout_suite_members, _build_evaluation_suite_members
    session, run, originals, root = scenario
    await add_reusable_suite(session, run, originals, root, lane, scored)
    reused = 'qa' if changed == 'math' else 'math'
    for product in await rm._products(session, originals[-1].id, 'generation_diagnostics'):
        Path(product.path).write_text(diagnostics(limited=2 if product.meta['suite_member_name'] == changed else 0).model_dump_json())
    async def handoff(session, run, batch):
        batch['handoff_pending'] = False
        rm._store(run, batch)
        await session.commit()
    monkeypatch.setattr(rm, '_handoff', handoff)
    assert await rm.intercept_completion(session, originals[-1])
    batch = deepcopy(rm.pending(run))
    assert batch['affected'] == [changed]
    for index, item in enumerate(batch['cohort']):
        assert item['active_members'] == [changed]  # Current, baseline AND historical checkpoint.
        assert item['retained'][reused]['reason'] == 'unchanged_configuration'
        assert ('score' in item['retained'][reused]) == scored
        await rm.advance(session, run.id)
        replay = await session.get(Ticket, item['inference_id'])
        builder = _build_holdout_suite_members if lane == 'held_out_test' else _build_validation_suite_members
        assert not await builder(ticket=replay, payload=replay.payload, work_dir=str(root), run=run, session=session)
        # Only math is generated; qa remains the exact previous artifact.
        replay.status = 'succeeded'
        path = root / f'{replay.id}-revised-math.json'
        path.write_text(diagnostics().model_dump_json())
        for role in ['predictions', 'generation_diagnostics']:
            session.add(WorkProduct(ticket_id=replay.id, role=role, path=str(path), meta={'suite_member_name': changed,
                'configuration': load_inference_config(batch['configs'][changed]).model_dump(mode='json')}))
        await session.commit()
        products = await rm._products(session, replay.id, 'predictions', run)
        assert next(p for p in products if p.meta['suite_member_name'] == reused).path == item['retained'][reused]['assets']['predictions']['path']
        await rm.intercept_completion(session, replay)
        evaluation = await session.get(Ticket, rm.pending(run)['cohort'][index]['evaluation_id'])
        members = await _build_evaluation_suite_members(evaluation, evaluation.payload, evaluation.inputs, run, session)
        assert [m.name for m in members] == ([] if scored else ['qa'])
        evaluation.status = 'succeeded'
        for name in rm.evaluation_members(batch, item):
            session.add(WorkProduct(ticket_id=evaluation.id, role='metrics', path=str(root / f'{evaluation.id}-{name}-metrics.json'),
                                    meta={'suite_member_name': name, 'score': .8 if name == changed else (.4 if name == 'qa' else .2)}))
        await session.commit()
        await rm.intercept_completion(session, evaluation)
    assert rm.batches(run)[0]['status'] == 'published', rm.batches(run)[0].get('error')
    events = (await session.scalars(select(ScoreEvent).where(ScoreEvent.split == ('test' if lane == 'held_out_test' else 'validation')))).all()
    assert len(events) == 3
    assert [e.score for e in events] == pytest.approx([.6 if changed == 'math' else .5] * 3)
    assert all(e.extras['reused_members'] == {reused: 'unchanged_configuration'} for e in events)
    if lane == 'held_out_test':
        for result in run.holdout['suite_results'].values():
            assert result[changed]['inference_identity']['measurement']['max_new_tokens'] == 512
            assert result[changed]['prediction_work_product_id']
            if scored:
                assert result[reused]['evaluation_ticket_id'] == f'old-{reused}-eval'
                assert result[reused]['prediction_work_product_id'] == f'old-{reused}-pred'
                assert result[reused]['test_contract'] == {'legacy': reused}
                assert result[reused]['inference_identity']['measurement']['max_new_tokens'] == 256



@pytest.mark.asyncio
async def test_unchanged_config_without_complete_artifacts_is_not_reused(scenario):
    session, run, originals, root = scenario
    await add_reusable_suite(session, run, originals, root, 'optimization', False)
    (root / f'{originals[0].id}-qa-predictions.csv').unlink()
    await rm.intercept_completion(session, originals[-1])
    batch = rm.pending(run)
    assert batch['cohort'][0]['active_members'] == ['math']
    assert batch['cohort'][1]['active_members'] == ['math', 'qa']


@pytest.mark.asyncio
async def test_unchanged_predictions_without_scores_need_evaluation_not_gpu(scenario):
    session, run, originals, root = scenario
    await add_reusable_suite(session, run, originals, root, 'optimization', False)
    config = str(config_file(root))
    item = {'source_id': originals[-1].id, 'inference_id': originals[-1].id}
    batch = {'id': 'score-only', 'lane': 'optimization', 'status': 'running', 'base_model': originals[-1].payload['base_model'],
             'reuse_tickets': True, 'configs': {'math': config, 'qa': config}, 'cohort': [item]}
    await rm._plan_retention(session, run, batch, [])
    assert item['active_members'] == [] and not item['retained_only']
    rm._store(run, batch)
    replay = await rm._new_inference(session, run, batch, item)
    assert replay.status == 'succeeded'
    assert rm.evaluation_members(batch, item) == ['math', 'qa']
    assert len(await rm._products(session, replay.id, 'predictions', run)) == 2


@pytest.mark.asyncio
async def test_retry_reuses_settled_member_predictions_without_stale_score(scenario):
    session, run, originals, root = scenario
    await add_reusable_suite(session, run, originals, root, 'optimization', True)
    for product in await rm._products(session, originals[-1].id, 'generation_diagnostics'):
        Path(product.path).write_text(diagnostics(limited=2).model_dump_json())
    await rm.intercept_completion(session, originals[-1])
    batch = deepcopy(rm.pending(run))
    current = originals[-1]
    await finish_inference(session, current, root, limited=2)
    for name, path in batch['configs'].items():
        session.add(WorkProduct(ticket_id=current.id, role='inference_config', path=path, meta={'suite_member_name': name}))
    await session.commit()
    evidence = await rm._evidence(session, run, current)
    batch['status'] = 'superseded'
    rm._store(run, batch)
    await rm._retry(session, run, batch, evidence, ['math'])
    new = rm.pending(run)
    assert new['cohort'][0]['active_members'] == ['math']
    assert new['cohort'][0]['retained']['qa']['config_path'] == batch['configs']['qa']
    assert 'score' not in new['cohort'][0]['retained']['qa']
    # Baseline has never run either new budget, so both still need replay.
    assert new['cohort'][1]['active_members'] == ['math', 'qa']


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['running', 'paused', 'published'])
async def test_cohort_blocks_next_iteration_between_successful_tickets(scenario, status):
    from zevo.engine.run.runner import _holdout_settled
    session, run, originals, root = scenario
    batch = {'id': 'barrier', 'status': status, 'lane': 'held_out_test', 'cohort': []}
    rm._store(run, batch)
    await session.commit()
    assert all(t.status == 'succeeded' for t in originals)
    assert await _holdout_settled(session, run.id) is (status == 'published')
    if status == 'paused':
        await rm.advance(session, run.id)
        assert rm.pending(run)['status'] == 'paused'
        assert run.status == 'running'


@pytest.mark.asyncio
async def test_legacy_primary_retention_ignores_database_config_key_order(scenario):
    session, run, originals, root = scenario
    await add_reusable_suite(session, run, originals, root, 'held_out_test', True)
    for role in ['generation_diagnostics', 'inference_config']:
        for product in await rm._products(session, originals[0].id, role):
            if product.meta['suite_member_name'] == 'math':
                product.meta = {'suite_primary': True}
    await session.commit()
    path = str(config_file(root))
    revised = root / 'larger-qa.json'
    revised.write_text(rm.enlarged_config(load_inference_config(path), {}).model_dump_json())
    item = {'source_id': originals[0].id, 'inference_id': originals[0].id}
    # PostgreSQL JSONB does not retain suite order in this map.
    batch = {'id': 'order', 'lane': 'held_out_test', 'base_model': originals[0].payload['base_model'],
             'configs': {'qa': str(revised), 'math': path}, 'cohort': [item]}
    await rm._plan_retention(session, run, batch, ['qa'])
    assert item['active_members'] == ['qa']
    assert item['retained']['math']['score']['score'] == .2


@pytest.mark.asyncio
@pytest.mark.parametrize('lane', ['optimization', 'held_out_test'])
async def test_replay_primary_diagnostics_follow_ticket_not_original_suite_order(scenario, lane):
    session, run, originals, root = scenario
    await add_reusable_suite(session, run, originals, root, lane, False)
    ticket = originals[-1]
    ticket.payload = {**ticket.payload, 'test_set_name': 'qa' if lane == 'held_out_test' else 'validation:qa'}
    for product in await rm._products(session, ticket.id, 'generation_diagnostics'):
        if product.meta['suite_member_name'] == 'qa':
            product.meta = {'suite_primary': True}
            Path(product.path).write_text(diagnostics(limited=3).model_dump_json())
    await session.commit()
    evidence = await rm._evidence(session, run, ticket)
    assert evidence['qa']['truncated_questions'] == 3
    assert evidence['math']['truncated_questions'] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('cap', [1024, 2048])
async def test_holdout_score_uses_bound_prediction_not_old_or_newest_attempt(scenario, cap):
    from zevo.engine.run.runner import _record_holdout_score
    session, run, originals, root = scenario
    await add_reusable_suite(session, run, originals, root, 'held_out_test', False)
    inference = originals[-1]
    config = load_inference_config(config_file(root))
    config.measurement.max_new_tokens = cap
    bound = WorkProduct(ticket_id=inference.id, role='predictions', path=str(root / 'bound.csv'),
                        meta={'suite_member_name': 'math', 'configuration': config.model_dump(mode='json')})
    session.add(bound)
    await session.flush()
    config.measurement.max_new_tokens = cap * 2
    session.add(WorkProduct(ticket_id=inference.id, role='predictions', path=str(root / 'later.csv'),
                           meta={'suite_member_name': 'math', 'configuration': config.model_dump(mode='json')}))
    evaluation = Ticket(id='bound-eval', run_id=run.id, agent_id='evaluation', lane='held_out_test',
                        status='succeeded', iteration=inference.iteration, payload={'test_set_name': 'math'},
                        inputs={'predictions': {'source_ticket_id': inference.id, 'work_product_id': bound.id}})
    session.add(evaluation)
    await session.commit()
    await _record_holdout_score(session, run, evaluation, .9, member_name='math')
    result = run.holdout['suite_results'][f"trained|2|{inference.payload['base_model']}"]['math']
    assert result['inference_identity']['measurement']['max_new_tokens'] == cap
    assert result['prediction_work_product_id'] == bound.id


@pytest.mark.asyncio
async def test_secondary_evaluation_uses_frozen_prediction_and_rejects_wrong_binding(scenario):
    from zevo.engine.run.runner import _build_evaluation_suite_members
    session, run, originals, root = scenario
    await rm.intercept_completion(session, originals[-1])
    await finish_inference(session, originals[-1], root)
    await rm.advance(session, run.id)
    batch = deepcopy(rm.pending(run))
    item = batch['cohort'][0]
    evaluation = await session.get(Ticket, item['evaluation_id'])
    bound = await session.get(WorkProduct, item['evaluation_predictions']['qa'])
    session.add(WorkProduct(ticket_id=originals[-1].id, role='predictions', path=str(root / 'later-qa.csv'),
                           meta={'suite_member_name': 'qa'}))
    await session.commit()
    members = await _build_evaluation_suite_members(evaluation, evaluation.payload, evaluation.inputs, run, session)
    assert members[0].predictions_path == bound.path
    assert (await rm.evaluation_prediction(session, run, evaluation, 'qa')).id == bound.id
    item['evaluation_predictions']['qa'] = item['evaluation_predictions']['math']
    rm._store(run, batch)
    with pytest.raises(ValueError, match='binding is invalid'):
        await rm.evaluation_prediction(session, run, evaluation, 'qa')


@pytest.mark.asyncio
@pytest.mark.parametrize('damage', ['missing_configuration', 'wrong_configuration', 'missing_prediction'])
async def test_invalid_replay_provenance_keeps_entire_old_cohort(scenario, monkeypatch, damage):
    session, run, originals, root = scenario
    handoffs = []
    async def handoff(*args):
        handoffs.append(True)
    monkeypatch.setattr(rm, '_handoff', handoff)
    original_history = deepcopy(run.history)
    original_scores = [(e.id, e.score) for e in (await session.scalars(select(ScoreEvent))).all()]
    await rm.intercept_completion(session, originals[-1])
    cohort = deepcopy(rm.pending(run)['cohort'])
    for index, item in enumerate(cohort):
        inference = await session.get(Ticket, item['inference_id'])
        await finish_inference(session, inference, root)
        await rm.advance(session, run.id)
        active = rm.pending(run)['cohort'][index]
        evaluation = await session.get(Ticket, active['evaluation_id'])
        await finish_evaluation(session, evaluation, root)
        if index == len(cohort) - 1:
            product = await session.get(WorkProduct, active['evaluation_predictions']['math'])
            if damage == 'missing_prediction':
                Path(product.path).unlink()
            else:
                meta = deepcopy(product.meta)
                if damage == 'missing_configuration':
                    meta.pop('configuration')
                else:
                    meta['configuration']['measurement']['max_new_tokens'] = 256
                product.meta = meta
            await session.commit()
        await rm.advance(session, run.id)
    assert run.status == 'degraded'
    assert rm.batches(run)[-1]['status'] == 'failed'
    assert run.history == original_history
    assert [(e.id, e.score) for e in (await session.scalars(select(ScoreEvent))).all()] == original_scores
    assert not handoffs
