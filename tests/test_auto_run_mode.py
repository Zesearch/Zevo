"""Auto run mode: agent-derived scoring with standard training ownership.

Pinned here:
  - the request shape: `mode="auto"` derives scoring but accepts the ordinary
    optional data/model/method pins and hints; the
    pipeline modes still require a complete UserRequest;
  - creation: an auto Run needs no scoring contract, settles NO split at
    creation, creates no supervisor, and emits exactly one queued
    `scope_problem` Data Ticket (+ wakeup) in the same commit;
  - full_pipeline is unchanged: it still rejects a missing scoring contract and
    settles at creation;
  - the ScopingResult contract's guardrails (benchmark provenance, mandatory
    decontamination for synthesis, no fabricated benchmark rows, Validation
    mirrors Test);
  - the Data payload/input/result contracts for `scope_problem`;
  - post-scoping settlement writes the contract onto the Run, moves the
    held-out behind the private boundary, creates the supervisor with a
    `run_created` trigger and wakes the Orchestrator; a failed scoping Ticket
    fails the Run; the Orchestrator is never woken while unsettled.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.shared.runs import CreateRunRequest, _summary, create_run
from zevo.contracts.data import DataResult, DataTaskInput
from zevo.contracts.orchestrator import AutoUserRequest, UserRequest
from zevo.contracts.scoping import (
    BenchmarkProvenance,
    DecontaminationEvidence,
    ScopingResult,
    SynthesisProvenance,
    load_scoping_result,
)
from zevo.contracts.tickets import DataPayload, RunMode, validate_stored_payload
from zevo.db.models import AgentWakeupRequest, Base, Run, Task, Ticket
from zevo.engine.agent.drivers.stub import _stub_data, _stub_scoping
from zevo.engine.run.runner import (
    _build_data_input,
    _essential_artifact_fields,
    _maybe_wake_supervisor,
)
from zevo.engine.run.scoping import (
    complete_scoping,
    is_scoping_ticket,
    new_scoping_ticket,
    scoping_ticket_id,
)


# ─────────────────────────────── fixtures ────────────────────────────────────


async def _session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def _full_request(**overrides) -> dict:
    body = dict(
        task_objective="Improve bar-exam answering.",
        metric="accuracy", metric_direction="max", metric_type="builtin",
        training_method="", dataset="", base_model="",
        test_set="/data/test.csv", test_answer_fields=["answer"],
        test_sample_submission="/data/sample.csv",
        constraints=[],
    )
    body.update(overrides)
    return body


def _auto_body(**overrides) -> dict:
    body = dict(
        mode="auto", task_name="bar-exam-auto", run_name="first-auto",
        user_request={"task_objective": "Improve bar-exam answering."},
        gpu_provider="instance",
    )
    body.update(overrides)
    return body


def _benchmark_result(tmp_path: Path, **overrides) -> ScopingResult:
    test_set = tmp_path / "bench.csv"
    test_set.write_text("id,q,answer\n1,x,A\n", encoding="utf-8")
    sample = tmp_path / "sample.csv"
    sample.write_text("id,prediction\n1,A\n", encoding="utf-8")
    body = dict(
        metric="accuracy", metric_direction="max",
        eval_source="public_benchmark",
        test_set_path=str(test_set), test_answer_fields=["answer"],
        test_sample_submission_path=str(sample), test_rows=1200,
        rationale="MMLU professional_law measures legal MCQ accuracy.",
        benchmark=BenchmarkProvenance(
            hub_id="cais/mmlu", config="professional_law", split="test", rows=1200,
        ),
    )
    body.update(overrides)
    return ScopingResult(**body)


def _synth_blocks(rows: int = 1000) -> dict:
    return dict(
        synthesis=SynthesisProvenance(
            teacher_model="meta/Llama-3-70B-Instruct",
            generation_params={"temperature": 0.7, "self_consistency_k": 3},
            seed_sources=["state bar syllabus topics"],
            rows_generated=1500, rows_verified=1100, rows_kept=rows,
        ),
        decontamination=DecontaminationEvidence(
            method="exact + 13-gram overlap",
            compared_against=["reglab/barexam_qa", "suggested training corpus"],
            overlap_removed=100,
        ),
    )


# ──────────────────────────── request shape ──────────────────────────────────


def test_run_mode_vocabulary_includes_auto() -> None:
    assert "auto" in RunMode.__args__  # type: ignore[attr-defined]


def test_auto_request_needs_only_the_objective() -> None:
    body = CreateRunRequest.model_validate(_auto_body())
    assert body.mode == "auto"
    assert isinstance(body.user_request, AutoUserRequest)
    assert body.user_request.task_objective == "Improve bar-exam answering."
    assert body.user_request.test_query == ""
    assert body.user_request.data_query == ""


def test_auto_request_accepts_training_ownership_and_forbids_scoring_fields() -> None:
    body = CreateRunRequest.model_validate(_auto_body(user_request={
        "task_objective": "o", "test_query": "use a diverse legal MCQ benchmark",
        "dataset": "org/train", "dataset_split": "train",
        "data_query": "prefer bar-exam QA corpora", "base_model": "org/model",
        "model_query": "small instruct model", "training_method": "lora_sft",
        "method_query": "start with SFT",
        "constraints": ["no external APIs"],
    }))
    assert body.user_request.dataset == "org/train"
    assert body.user_request.base_model == "org/model"
    assert body.user_request.training_method == "lora_sft"
    assert body.user_request.test_query == "use a diverse legal MCQ benchmark"
    assert body.user_request.model_query == "small instruct model"
    assert body.user_request.constraints == ["no external APIs"]
    # A complete scoring contract is not an Auto request.
    with pytest.raises(ValidationError):
        CreateRunRequest.model_validate(_auto_body(user_request=_full_request()))
    # Nor is a partial one: metric alone matches neither shape.
    with pytest.raises(ValidationError):
        CreateRunRequest.model_validate(
            _auto_body(user_request={"task_objective": "o", "metric": "accuracy"})
        )
    with pytest.raises(ValidationError, match="requires user_request"):
        CreateRunRequest.model_validate(_auto_body(user_request=None))


def test_auto_mode_rejects_settings_and_customizations() -> None:
    with pytest.raises(ValidationError, match="Setting"):
        CreateRunRequest.model_validate(_auto_body(save_setting=True))
    with pytest.raises(ValidationError, match="customizations"):
        CreateRunRequest.model_validate(_auto_body(customizations={}))


def test_pipeline_modes_still_require_a_complete_user_request() -> None:
    # Byte-for-byte: a complete request still parses as a UserRequest ...
    body = CreateRunRequest.model_validate({
        "task_name": "t", "run_name": "r", "user_request": _full_request(),
    })
    assert body.mode == "full_pipeline"
    assert isinstance(body.user_request, UserRequest)
    # ... and an objective-only body is still rejected, for both pipeline modes.
    for mode in ("full_pipeline", "customized_pipeline"):
        with pytest.raises(ValidationError, match="complete user_request"):
            CreateRunRequest.model_validate({
                "mode": mode, "task_name": "t", "run_name": "r",
                "user_request": {"task_objective": "o"},
            })
    with pytest.raises(ValidationError):
        CreateRunRequest.model_validate({
            "task_name": "t", "run_name": "r",
            "user_request": _full_request(metric=""),
        })


# ────────────────────────────── creation ─────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_create_run_skips_scoring_and_settlement_and_emits_scoping(
    tmp_path, monkeypatch,
) -> None:
    from zevo.api.routers.shared import runs as runs_router

    monkeypatch.setenv("ZEVO_WORK_DIR", str(tmp_path / "runs"))

    async def must_not_settle(run, request):  # pragma: no cover - the assertion
        raise AssertionError("auto mode must not settle splits at creation")

    monkeypatch.setattr(runs_router, "_settle_splits", must_not_settle)
    engine, Session = await _session_factory()
    async with Session() as db:
        response = await create_run(CreateRunRequest.model_validate(_auto_body(
            user_request={
                "task_objective": "Improve bar-exam answering.",
                "test_query": "prefer a public, diverse legal MCQ benchmark",
                "dataset": "org/bar-train", "data_query": "bar exam QA",
                "base_model": "org/base-model", "model_query": "a 7B instruct model",
                "training_method": "lora_sft", "method_query": "SFT first",
            },
            max_cost_usd=25.0, iteration_budget=3,
        )), db)
        run = await db.get(Run, response.run_id)
        assert run is not None
        assert run.mode == "auto"
        assert run.status == "running"
        assert run.scoring_settled is False
        assert run.metric == "" and run.validation_metric == ""
        assert run.holdout == {}
        assert run.supervisor_ticket_id == ""
        assert run.decision_pins == {
            "dataset": "org/bar-train",
            "data_query": "bar exam QA",
            "base_model": "org/base-model",
            "model_query": "a 7B instruct model",
            "training_method": "lora_sft",
            "method_query": "SFT first",
        }
        assert run.max_cost_usd == 25.0 and run.iteration_budget == 3

        tickets = (await db.execute(select(Ticket).where(Ticket.run_id == run.id))).scalars().all()
        assert [t.id for t in tickets] == [scoping_ticket_id(run)]
        scope = tickets[0]
        assert scope.agent_id == "data" and scope.lane == "optimization"
        assert scope.status == "queued" and scope.iteration == 0
        assert scope.payload["operation"] == "scope_problem"
        assert scope.payload["task_objective"] == "Improve bar-exam answering."
        assert scope.payload["test_query"] == "prefer a public, diverse legal MCQ benchmark"
        assert scope.payload["data_query"] == ""
        assert "model_query" not in scope.payload
        assert is_scoping_ticket(scope)

        wakeups = (await db.execute(select(AgentWakeupRequest))).scalars().all()
        assert [(w.agent_id, w.ticket_id, w.status) for w in wakeups] == [
            ("data", scope.id, "queued"),
        ]

        summary = _summary(run)
        assert summary.mode == "auto"
        assert summary.scoring_settled is False
        assert summary.eval_source == ""
    await engine.dispose()


@pytest.mark.asyncio
async def test_auto_create_run_refuses_a_predefined_task(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ZEVO_WORK_DIR", str(tmp_path / "runs"))
    engine, Session = await _session_factory()
    async with Session() as db:
        db.add(Task(
            name="bar-exam-auto", task_objective="predefined", metric="accuracy",
            metric_direction="max", test_set="/t.csv", test_answer_fields=["a"],
            test_sample_submission="/s.csv",
        ))
        await db.commit()
        with pytest.raises(HTTPException) as exc:
            await create_run(CreateRunRequest.model_validate(_auto_body()), db)
        assert exc.value.status_code == 400
        assert "predefined" in str(exc.value.detail)
        assert (await db.execute(select(Run))).scalars().all() == []
    await engine.dispose()


@pytest.mark.asyncio
async def test_full_pipeline_still_requires_scoring_and_settles_at_creation(
    tmp_path, monkeypatch,
) -> None:
    from zevo.api.routers.shared import runs as runs_router

    monkeypatch.setenv("ZEVO_WORK_DIR", str(tmp_path / "runs"))
    calls: list[str] = []

    async def settled(run, request):
        calls.append(run.id)
        return request, {"test_set": request.test_set, "validation_set": "/v.csv"}, "settled"

    monkeypatch.setattr(runs_router, "_settle_splits", settled)
    engine, Session = await _session_factory()
    async with Session() as db:
        # Missing scoring assets are still a 400 before anything is persisted.
        with pytest.raises(HTTPException) as exc:
            await create_run(CreateRunRequest.model_validate({
                "task_name": "custom", "run_name": "r",
                "user_request": _full_request(test_set=""),
                "gpu_provider": "instance",
            }), db)
        assert exc.value.status_code == 400
        assert "test_set is required" in str(exc.value.detail)
        assert calls == []

        response = await create_run(CreateRunRequest.model_validate({
            "task_name": "custom", "run_name": "r", "user_request": _full_request(),
            "gpu_provider": "instance",
        }), db)
        run = await db.get(Run, response.run_id)
        assert calls == [run.id]  # settled exactly once, at creation
        assert run.mode == "full_pipeline"
        assert run.scoring_settled is True
        assert run.metric == "accuracy"
        assert run.holdout["validation_set"] == "/v.csv"
        assert run.supervisor_ticket_id == f"orchestrate-{run.id[:8]}-001"
        sup = await db.get(Ticket, run.supervisor_ticket_id)
        assert sup is not None and sup.agent_id == "orchestrator"
        assert sup.payload["trigger"]["type"] == "run_created"
        # (The stubbed settlement above returns the request as-is; blanking of
        # the Test lane is pinned by the real-settlement test below.)
        wakeups = (await db.execute(select(AgentWakeupRequest))).scalars().all()
        assert [(w.agent_id, w.ticket_id) for w in wakeups] == [("orchestrator", sup.id)]
    await engine.dispose()


# ──────────────────────────── ScopingResult ──────────────────────────────────


def test_benchmark_result_requires_provenance_and_exact_row_count(tmp_path) -> None:
    result = _benchmark_result(tmp_path)
    assert result.eval_source == "public_benchmark"
    assert result.effective_validation_contract() == ("builtin", "accuracy", "max", "")
    with pytest.raises(ValidationError, match="benchmark provenance"):
        _benchmark_result(tmp_path, benchmark=None)
    # Rows in the file must be exactly the rows materialized: no filling in.
    with pytest.raises(ValidationError, match="never fabricated"):
        _benchmark_result(tmp_path, test_rows=1300)
    with pytest.raises(ValidationError, match="no synthesis provenance"):
        _benchmark_result(tmp_path, **_synth_blocks())


def test_synthesized_result_requires_teacher_provenance_and_decontamination(tmp_path) -> None:
    ok = _benchmark_result(
        tmp_path, eval_source="synthesized", benchmark=None, test_rows=1000,
        **_synth_blocks(),
    )
    assert ok.synthesis is not None and ok.decontamination is not None
    assert ok.provenance_summary()["decontamination"]["overlap_removed"] == 100
    blocks = _synth_blocks()
    with pytest.raises(ValidationError, match="synthesis provenance"):
        _benchmark_result(
            tmp_path, eval_source="synthesized", benchmark=None, test_rows=1000,
            decontamination=blocks["decontamination"],
        )
    with pytest.raises(ValidationError, match="decontamination evidence"):
        _benchmark_result(
            tmp_path, eval_source="synthesized", benchmark=None, test_rows=1000,
            synthesis=blocks["synthesis"],
        )
    with pytest.raises(ValidationError):
        DecontaminationEvidence(checked=False, method="exact", compared_against=["x"])
    with pytest.raises(ValidationError, match="rows_kept must equal test_rows"):
        _benchmark_result(
            tmp_path, eval_source="synthesized", benchmark=None, test_rows=999,
            **_synth_blocks(),
        )


def test_scoping_metric_contract_is_checked_like_a_user_request(tmp_path) -> None:
    with pytest.raises(ValidationError, match="unknown built-in Test metric"):
        _benchmark_result(tmp_path, metric="vibes")
    with pytest.raises(ValidationError, match="custom Test metrics require"):
        _benchmark_result(tmp_path, metric="legal_score", metric_type="custom")
    with pytest.raises(ValidationError, match="must not carry a custom evaluator"):
        _benchmark_result(tmp_path, evaluation_script="/x/eval.py")
    # Validation mirrors Test: an explicit identical contract is fine, a
    # different one is not (there is no separate Validation population).
    same = _benchmark_result(
        tmp_path, validation_metric="accuracy", validation_metric_direction="max",
        validation_metric_type="builtin",
    )
    assert same.effective_validation_contract()[1] == "accuracy"
    with pytest.raises(ValidationError, match="must mirror Test"):
        _benchmark_result(
            tmp_path, validation_metric="token_f1", validation_metric_direction="max",
            validation_metric_type="builtin",
        )
    with pytest.raises(ValidationError, match="given together"):
        _benchmark_result(tmp_path, validation_metric="accuracy")
    with pytest.raises(ValidationError, match="absolute"):
        _benchmark_result(tmp_path, test_set_path="relative/test.csv")


def test_scoping_result_round_trips_through_its_file_validator(tmp_path) -> None:
    result = _benchmark_result(tmp_path)
    path = tmp_path / "scoping_result.json"
    path.write_text(result.model_dump_json(), encoding="utf-8")
    assert load_scoping_result(path) == result
    from zevo.contracts.scoping import _main
    assert _main(["validate", str(path)]) == 0
    path.write_text(json.dumps({"metric": "accuracy"}), encoding="utf-8")
    assert _main(["validate", str(path)]) == 1


# ──────────────────────── Data contracts for scope_problem ───────────────────


def test_scope_problem_payload_carries_only_evaluation_scoping_inputs() -> None:
    payload = validate_stored_payload(agent_id="data", input_format="typed", payload={
        "operation": "scope_problem", "task_objective": "o",
        "test_query": "prefer public benchmark coverage",
        "constraints": ["no external APIs"],
    })
    assert payload["operation"] == "scope_problem"
    assert payload["test_query"] == "prefer public benchmark coverage"
    assert payload["metric"] == "" and payload["scoring_set"] == ""
    with pytest.raises(ValidationError, match="requires task_objective"):
        DataPayload(operation="scope_problem")
    with pytest.raises(ValidationError, match="derives the scoring contract itself"):
        DataPayload(operation="scope_problem", task_objective="o", metric="accuracy")
    with pytest.raises(ValidationError, match="derives the scoring contract itself"):
        DataPayload(operation="scope_problem", task_objective="o", training_method="lora_sft")
    with pytest.raises(ValidationError, match="derives the scoring contract itself"):
        DataPayload(operation="scope_problem", task_objective="o", data_query="training hint")
    # Existing operations are untouched: they still require a metric and may
    # not carry the scoping fields.
    with pytest.raises(ValidationError, match="requires metric"):
        DataPayload(operation="prepare_holdout_data", scoring_set="/t.csv", answer_fields=["a"])
    with pytest.raises(ValidationError, match="scoping fields"):
        DataPayload(
            operation="prepare_holdout_data", scoring_set="/t.csv",
            answer_fields=["a"], metric="accuracy", task_objective="o",
        )
    # The Orchestrator's request payload cannot author a scoping Ticket.
    from zevo.contracts.tickets import PipelineDataRequestPayload
    with pytest.raises(ValidationError):
        PipelineDataRequestPayload(operation="scope_problem", training_method="lora_sft")


def test_scope_problem_data_result_produces_only_the_scoping_file() -> None:
    base = dict(ticket_id="t", error_message="", notes="n")
    ok = DataResult(status="succeeded", operation="scope_problem",
                    scoping_result_path="/w/scoping_result.json", **base)
    assert _essential_artifact_fields(ok) == ["scoping_result_path"]
    with pytest.raises(ValidationError, match="requires scoping_result_path"):
        DataResult(status="succeeded", operation="scope_problem", **base)
    with pytest.raises(ValidationError, match="produces only scoping_result_path"):
        DataResult(status="succeeded", operation="scope_problem",
                   scoping_result_path="/w/s.json", training_dataset_path="/w/d.jsonl", **base)
    with pytest.raises(ValidationError, match="produces only scoping_result_path"):
        DataResult(status="succeeded", operation="scope_problem",
                   scoping_result_path="/w/s.json", synthesis_generated_rows=5,
                   synthesis_teacher_model="t", decontamination_checked=True, **base)
    with pytest.raises(ValidationError, match="valid only for scope_problem"):
        DataResult(status="succeeded", operation="prepare_holdout_data",
                   scoring_public_path="/p", scoping_result_path="/w/s.json", **base)
    # Failure never needs artifacts.
    DataResult(status="failed", operation="scope_problem", ticket_id="t",
               error_message="no suitable benchmark", notes="n")


def test_runner_builds_a_scoping_work_order_without_stamping_a_metric() -> None:
    run = Run(id="auto-run-1", metric="", mode="auto", scoring_settled=False)
    ticket = new_scoping_ticket(run, AutoUserRequest(
        task_objective="Improve bar-exam answering.", base_model="org/model",
        model_query="7B",
        test_query="prefer a public legal benchmark",
        constraints=["no external APIs"],
    ))
    inp = _build_data_input(
        ticket, ticket.payload, {}, "/w", run,
        {"task_objective": "x", "agent_objective": "pin org/model"},
    )
    assert isinstance(inp, DataTaskInput)
    assert inp.operation == "scope_problem"
    assert inp.task_objective == "Improve bar-exam answering."
    assert inp.test_query == "prefer a public legal benchmark"
    assert inp.constraints == ["no external APIs"]
    assert inp.run_context == {}
    assert inp.metric == "" and inp.scoring_set == "" and inp.training_method == ""
    assert inp.scoping_result_schema["title"] == "ScopingResult"
    assert "zevo.contracts.scoping validate" in inp.scoping_result_validation_command
    # The same work order on the held-out lane is refused.
    ticket.lane = "held_out_test"
    with pytest.raises(ValueError, match="optimization lane only"):
        _build_data_input(ticket, ticket.payload, {}, "/w", run, {})
    # Private held-out preparation still demands a metric.
    with pytest.raises(ValidationError, match="requires metric"):
        DataTaskInput(
            ticket_id="t", operation="prepare_holdout_data", run_id="r",
            scoring_set="/t.csv", answer_fields=["a"], data_recipe_schema={},
            data_recipe_validation_command="x", artifacts_validation_command="x",
            work_dir="/w",
        )


# ─────────────────────── post-scoping settlement ─────────────────────────────


async def _auto_run_with_scoping_ticket(db, tmp_path, monkeypatch) -> tuple[Run, Ticket]:
    monkeypatch.setenv("ZEVO_WORK_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "private"))
    response = await create_run(CreateRunRequest.model_validate(_auto_body(
        user_request={
            "task_objective": "Improve bar-exam answering.",
            "data_query": "bar exam QA",
            "base_model": "org/base-model",
            "training_method": "lora_sft",
        },
    )), db)
    run = await db.get(Run, response.run_id)
    scope = await db.get(Ticket, scoping_ticket_id(run))
    return run, scope


@pytest.mark.asyncio
async def test_successful_scoping_settles_the_contract_and_wakes_the_orchestrator(
    tmp_path, monkeypatch,
) -> None:
    engine, Session = await _session_factory()
    async with Session() as db:
        run, scope = await _auto_run_with_scoping_ticket(db, tmp_path, monkeypatch)
        work_dir = tmp_path / "runs" / run.id / scope.id
        output = _stub_data({**scope.payload, "ticket_id": scope.id}, work_dir)
        assert isinstance(output, DataResult) and output.operation == "scope_problem"
        scoping = load_scoping_result(output.scoping_result_path)
        original_test_set = Path(scoping.test_set_path)
        assert original_test_set.is_file()

        scope.status = "succeeded"
        await db.commit()
        await complete_scoping(db, scope, output)
        await db.refresh(run)

        # The contract is on the Run, exactly as creation would have written it.
        assert run.scoring_settled is True
        assert run.status == "running"
        assert (run.metric, run.metric_direction) == ("accuracy", "max")
        assert (run.validation_metric, run.validation_metric_direction) == ("accuracy", "max")
        holdout = run.holdout
        assert holdout["eval_source"] == "public_benchmark"
        assert holdout["scoping_ticket_id"] == scope.id
        assert holdout["scoping"]["benchmark"]["hub_id"] == "stub/benchmark"
        assert holdout["validation_source"] == "test_split"
        assert holdout["validation_rows"] == 200
        assert holdout["test_answer_fields"] == ["answer"]
        assert Path(holdout["validation_set"]).is_file()
        # The held-out lives behind the private boundary; the scoping copy is gone.
        private = (tmp_path / "private").resolve()
        assert Path(holdout["test_set"]).resolve().is_relative_to(private)
        assert Path(holdout["test_sample_submission"]).resolve().is_relative_to(private)
        assert not original_test_set.exists()
        assert run.decision_pins["data_query"] == "bar exam QA"
        assert run.decision_pins["base_model"] == "org/base-model"
        assert run.decision_pins["training_method"] == "lora_sft"

        # The supervisor exists with a run_created trigger and no scoring assets.
        assert run.supervisor_ticket_id == f"orchestrate-{run.id[:8]}-001"
        sup = await db.get(Ticket, run.supervisor_ticket_id)
        assert sup is not None and sup.status == "queued"
        payload = sup.payload
        assert payload["mode"] == "auto"
        assert payload["trigger"]["type"] == "run_created"
        assert payload["trigger"]["ticket_id"] == ""
        req = payload["user_request"]
        assert req["metric"] == "accuracy" and req["metric_type"] == "builtin"
        assert req["test_set"] == "" and req["test_answer_fields"] == []
        assert req["validation_set"] == ""
        assert req["validation_answer_fields"] == []
        assert req["validation_sample_submission"] == ""
        assert req["data_query"] == "bar exam QA"
        assert req["dataset"] == ""
        assert req["base_model"] == "org/base-model"
        assert req["training_method"] == "lora_sft"
        assert payload["validation_rows"] == 200

        wakeups = (await db.execute(
            select(AgentWakeupRequest).order_by(AgentWakeupRequest.created_at)
        )).scalars().all()
        assert [(w.agent_id, w.ticket_id) for w in wakeups][-1] == ("orchestrator", sup.id)

        summary = _summary(run)
        assert summary.scoring_settled is True and summary.eval_source == "public_benchmark"

        # Settling twice is a no-op: the contract is immutable afterwards.
        await complete_scoping(db, scope, output)
        assert len((await db.execute(select(Ticket).where(
            Ticket.agent_id == "orchestrator",
        ))).scalars().all()) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_scoping_with_too_few_rows_fails_the_run_with_the_reason(
    tmp_path, monkeypatch,
) -> None:
    engine, Session = await _session_factory()
    async with Session() as db:
        run, scope = await _auto_run_with_scoping_ticket(db, tmp_path, monkeypatch)
        work_dir = tmp_path / "runs" / run.id / scope.id
        output = _stub_scoping({**scope.payload, "ticket_id": scope.id}, work_dir, rows=50)
        scope.status = "succeeded"
        await db.commit()
        await complete_scoping(db, scope, output)
        await db.refresh(run)
        assert run.status == "failed"
        assert run.scoring_settled is False
        assert "derived Validation requires at least" in run.halted_reason
        assert run.supervisor_ticket_id == ""
        assert (await db.execute(select(AgentWakeupRequest).where(
            AgentWakeupRequest.agent_id == "orchestrator",
        ))).scalars().all() == []
    await engine.dispose()


@pytest.mark.asyncio
async def test_failed_scoping_ticket_fails_the_run_and_never_wakes_the_orchestrator(
    tmp_path, monkeypatch,
) -> None:
    engine, Session = await _session_factory()
    async with Session() as db:
        run, scope = await _auto_run_with_scoping_ticket(db, tmp_path, monkeypatch)
        # A ticket still repairing is left alone.
        scope.status = "repairing"
        await db.commit()
        await complete_scoping(db, scope, None)
        await db.refresh(run)
        assert run.status == "running"

        scope.status = "failed"
        scope.error_message = "no suitable public benchmark and synthesis unverifiable"
        await db.commit()
        await complete_scoping(db, scope, None)
        await db.refresh(run)
        assert run.status == "failed"
        assert "no suitable public benchmark" in run.halted_reason
        assert run.finished_at is not None

        # And an unsettled auto Run never gets a supervisor wake from anywhere.
        run.status = "running"
        run.supervisor_ticket_id = "orchestrate-would-be"
        await db.commit()
        await _maybe_wake_supervisor(db, scope)
        assert (await db.execute(select(Ticket).where(
            Ticket.agent_id == "orchestrator",
        ))).scalars().all() == []
        assert (await db.execute(select(AgentWakeupRequest).where(
            AgentWakeupRequest.agent_id == "orchestrator",
        ))).scalars().all() == []
    await engine.dispose()


# ─────────────────────────────── migration ───────────────────────────────────


def test_auto_mode_migration_chains_off_the_previous_head() -> None:
    versions = Path("alembic/versions")
    revisions: dict[str, tuple[str, ...]] = {}
    for file in versions.glob("*.py"):
        text = file.read_text(encoding="utf-8")
        rev = re.search(r'^revision(?:\s*:[^=]+)?\s*=\s*["\']([^"\']+)["\']', text, re.M)
        down = re.search(r"^down_revision(?:\s*:[^=]+)?\s*=\s*(.+)$", text, re.M)
        assert rev and down, file
        revisions[rev.group(1)] = tuple(re.findall(r'["\']([0-9a-f]+)["\']', down.group(1)))
    referenced = {d for downs in revisions.values() for d in downs}
    heads = sorted(set(revisions) - referenced)
    assert len(heads) == 1, heads
    assert revisions["f1a2b3c4d5e7"] == ("e1a2b3c4d5e6",)
    assert "scoring_settled" in (versions / "f1a2b3c4d5e7_auto_run_mode.py").read_text()
