from __future__ import annotations

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.shared.runs import (
    CreateRunRequest,
    _resolve_run_compute,
    _summary,
    create_run,
)
from zevo.contracts.orchestrator import UserRequest
from zevo.db.models import Agent, Base, Run, SshHost, Task, TaskSetting, Ticket


def _request() -> UserRequest:
    return UserRequest(metric="accuracy",
        task_objective="test",
        metric_direction="max",
        validation_metric_type="builtin",
        validation_metric="token_f1",
        validation_metric_direction="max",
        training_method="",
        dataset="",
        data_query="",
        base_model="",
        test_set="/data/test.csv",
        test_answer_fields=["answer"],
        test_sample_submission="/data/sample.csv",
        metric_type="builtin",
        evaluation_script="",
        constraints=[],
    )


def test_selection_queries_are_empty_guidance_not_decision_pins() -> None:
    request = _request()
    assert request.data_query == ""
    assert request.model_query == ""
    assert request.method_query == ""

    guided = request.model_copy(update={
        "data_query": "prefer well-curated sources",
        "model_query": "prefer a model that fits the available hardware",
        "method_query": "prefer a staged optimization plan",
    })
    assert guided.dataset == ""
    assert guided.base_model == ""
    assert guided.training_method == ""


@pytest.mark.asyncio
async def test_blank_gpu_provider_requires_a_concrete_default(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("VASTAI_API_KEY=configured-but-not-selected\n", encoding="utf-8")
    monkeypatch.setenv("ZEVO_ENV_FILE", str(env_file))

    with pytest.raises(HTTPException) as exc_info:
        await _resolve_run_compute(None, CreateRunRequest(task_name="t", run_name="r"))
    assert getattr(exc_info.value, "status_code", None) == 422
    assert "GPU backend is required" in str(getattr(exc_info.value, "detail", ""))

@pytest.mark.asyncio
async def test_credentials_do_not_become_default_without_an_explicit_choice(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "VASTAI_API_KEY=configured\n"
        "LAMBDA_API_KEY=secret_configured\n"
        "ZEVO_CLUSTER_SSH_HOST=login.example.edu\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ZEVO_ENV_FILE", str(env_file))
    with pytest.raises(HTTPException) as exc_info:
        await _resolve_run_compute(None, CreateRunRequest(task_name="t", run_name="r"))
    assert getattr(exc_info.value, "status_code", None) == 422

    env_file.write_text(
        env_file.read_text(encoding="utf-8") + "ZEVO_DEFAULT_COMPUTE=cloud:lambda\n",
        encoding="utf-8",
    )
    target = await _resolve_run_compute(None, CreateRunRequest(task_name="t", run_name="r"))
    assert target.provider == "cloud"
    assert target.cloud_backend == "lambda"


def test_preflight_requires_a_provider_when_no_default_was_resolved(monkeypatch) -> None:
    from zevo.api.routers.ui.preflight import PreflightBody, _check_infra

    monkeypatch.setenv("VASTAI_API_KEY", "configured-but-not-selected")
    items = []
    _check_infra(None, items)
    assert [item.code for item in items] == [
        "gpu_provider_required", "gpu_count", "generation_backend",
    ]

    body = PreflightBody(
        user_request=_request(), gpu_provider="cluster", num_gpus=2,
        generation_backend="hf",
    )
    assert body.gpu_provider == "cluster"
    assert body.num_gpus == 2
    assert body.generation_backend == "hf"


def test_preflight_checks_each_provider_against_its_own_remote_root(monkeypatch) -> None:
    from zevo.api.routers.ui.preflight import _check_infra

    monkeypatch.setenv("VASTAI_API_KEY", "configured")
    monkeypatch.delenv("ZEVO_CLUSTER_REMOTE_DIR", raising=False)
    cloud_items = []
    _check_infra("cloud", cloud_items)
    assert "cloud_ready" in [item.code for item in cloud_items]
    assert "cluster_remote_dir" not in [item.code for item in cloud_items]

    monkeypatch.setenv("ZEVO_CLUSTER_SSH_HOST", "cluster.example")
    monkeypatch.setenv("ZEVO_CLUSTER_REMOTE_DIR", "/scratch/zevo")
    monkeypatch.delenv("ZEVO_INSTANCE_REMOTE_DIR", raising=False)
    cluster_items = []
    _check_infra("cluster", cluster_items)
    assert "cluster_ready" in [item.code for item in cluster_items]
    assert "instance_remote_dir" not in [item.code for item in cluster_items]

    monkeypatch.setenv("ZEVO_INSTANCE_SSH_HOST", "instance.example")
    instance_items = []
    _check_infra("instance", instance_items)
    assert "instance_remote_dir" in [item.code for item in instance_items]


@pytest.mark.asyncio
async def test_claude_auth_status_accepts_api_key_fallback(monkeypatch) -> None:
    from zevo.api.routers.ui.auth_status import get_auth_status
    from zevo.api.routers.ui.settings import ALLOWED_KEYS

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test-value-long-enough")
    status = await get_auth_status()
    claude = next(row for row in status.drivers if row.driver == "claude_cli")
    assert claude.ready is True
    assert claude.has_api_key is True
    assert claude.effective == "api_key"
    assert {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"} <= ALLOWED_KEYS
    assert "HF_TOKEN" in ALLOWED_KEYS


@pytest.mark.asyncio
async def test_auth_status_treats_unreadable_root_credentials_as_absent(
    monkeypatch,
) -> None:
    from pathlib import Path

    from zevo.api.routers.ui.auth_status import get_auth_status

    original_stat = Path.stat

    def guarded_stat(path: Path, *args, **kwargs):
        if str(path).startswith("/root/"):
            raise PermissionError(str(path))
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    status = await get_auth_status()

    claude = next(row for row in status.drivers if row.driver == "claude_cli")
    codex = next(row for row in status.drivers if row.driver == "codex_cli")
    assert claude.ready is True and claude.effective == "api_key"
    assert codex.ready is True and codex.effective == "api_key"


def test_settings_reports_generic_instance_without_exposing_connection_values(monkeypatch) -> None:
    from zevo.api.routers.ui.settings import _environment_ssh_statuses

    monkeypatch.setenv("ZEVO_INSTANCE_SSH_HOST", "gpu.example.edu")
    monkeypatch.setenv("ZEVO_INSTANCE_SSH_USER", "private-user")
    monkeypatch.setenv("ZEVO_INSTANCE_SSH_KEY", "/private/key")
    monkeypatch.setenv("ZEVO_INSTANCE_REMOTE_DIR", "/private/workspace")
    monkeypatch.setenv("ZEVO_INSTANCE_ENV_SETUP", "conda activate private-env")
    monkeypatch.delenv("ZEVO_CLUSTER_SSH_HOST", raising=False)
    monkeypatch.delenv("ZEVO_CLUSTER_REMOTE_DIR", raising=False)

    rows = _environment_ssh_statuses()
    instance = next(row for row in rows if row.id == "instance")
    assert instance.label == "Instance"
    assert instance.configured is True
    assert instance.status == "unverified"
    serialized = " ".join(row.model_dump_json() for row in rows)
    assert "private-user" not in serialized
    assert "/private/key" not in serialized
    assert "/private/workspace" not in serialized

    monkeypatch.setenv("ZEVO_INSTANCE_SSH_NAME", "My research GPU")
    named = next(row for row in _environment_ssh_statuses() if row.id == "instance")
    assert named.label == "My research GPU"


def test_preflight_requires_explicit_hugging_face_base_model_ids() -> None:
    from zevo.api.routers.ui.preflight import _check_base_model

    items = []
    _check_base_model(_request().model_copy(update={"base_model": "qwen"}), items)
    assert [(item.severity, item.code) for item in items] == [
        ("blocker", "base_model_not_huggingface"),
    ]

    items = []
    _check_base_model(
        _request().model_copy(update={"base_model": "Qwen/Qwen3-0.6B-Base"}),
        items,
    )
    assert [(item.severity, item.code) for item in items] == [
        ("info", "base_model_known"),
    ]

    items = []
    _check_base_model(
        _request().model_copy(update={"base_model": "allenai/OLMo-2-0425-1B"}),
        items,
    )
    assert [(item.severity, item.code) for item in items] == [
        ("info", "base_model_known"),
    ]


@pytest.mark.asyncio
async def test_preflight_provider_check_ignores_deterministic_runner(monkeypatch) -> None:
    from zevo.api.routers.ui import preflight as preflight_module
    from zevo.api.routers.ui.preflight import _check_provider_creds
    from zevo.engine.agent.loader import list_agent_ids

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        for agent_id in list_agent_ids():
            db.add(Agent(
                id=agent_id,
                name=agent_id,
                title=agent_id,
                reports_to="orchestrator",
                identity_path=f"playbook/agents/{agent_id}/identity.md",
            ))
        db.add(Agent(
            id="evaluation",
            name="Evaluation Runner",
            title="Quality Evaluator",
            reports_to="system",
            identity_path="playbook/runners/evaluation.md",
            default_driver="evaluation_runner",
            default_model="no-llm",
        ))
        await db.commit()

        monkeypatch.setattr(
            preflight_module, "_driver_ready", lambda _driver: (True, "test"),
        )
        items = []
        await _check_provider_creds(items, db)
        assert [item.code for item in items] == ["provider_creds_ok"]
        assert "evaluation" not in items[0].message.lower()

    await engine.dispose()


@pytest.mark.asyncio
async def test_task_evaluation_contract_is_immutable_after_a_run_exists() -> None:
    from fastapi import HTTPException

    from zevo.api.routers.ui.tasks import TaskPatch, update_task

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        db.add(Task(
            name="stable-task",
            task_objective="stable objective",
            test_set="/test.jsonl",
            test_answer_fields=["answer"],
            test_sample_submission="/sample.jsonl",
            metric_type="builtin",
            evaluation_script="",
            metric="accuracy",
            metric_direction="max",
        ))
        db.add(Run(metric="accuracy", id="stable-run", task_name="stable-task"))
        await db.commit()

        with pytest.raises(HTTPException) as exc:
            await update_task(
                "stable-task", TaskPatch(metric="token_f1"), db=db,
            )
        assert exc.value.status_code == 409
        assert "evaluation contract is immutable" in str(exc.value.detail)

    await engine.dispose()

def test_task_contract_contains_no_hidden_setting_or_runtime_defaults() -> None:
    from zevo.api.routers.ui.tasks import TaskBody
    from pydantic import ValidationError

    fields = set(TaskBody.model_fields)
    assert fields == {
        "name", "task_objective", "test_set", "test_answer_fields",
        "test_sample_submission", "metric_type", "evaluation_script",
        "metric", "metric_direction",
    }
    with pytest.raises(ValidationError):
        TaskBody(name="missing-target")


def test_setting_contract_has_no_target_override() -> None:
    from zevo.api.routers.ui.tasks import SettingBody

    assert "metric_direction" not in SettingBody.model_fields
    with pytest.raises(ValidationError):
        SettingBody(metric_direction="max")
    with pytest.raises(ValidationError, match="model_reasoning_type"):
        SettingBody(model_reasoning_type="thinking")


def test_user_request_accepts_structured_experiment_preferences() -> None:
    request = _request().model_copy(update={
        "training_method": "dpo",
        "prompt_framing": "chat:Qwen/Qwen3-0.6B",
        "system_prompt": "Answer directly.",
        "loss_objective_config": {"beta": 0.1, "loss_type": "sigmoid"},
        "inference_config": {"input_fields": ["question"]},
        "decoding_config": {"temperature": 0.0, "seed": 7},
    })
    # model_copy does not revalidate; exercise the API-shaped boundary.
    request = UserRequest.model_validate(request.model_dump())
    assert request.loss_objective_config["beta"] == 0.1
    assert request.decoding_config == {"temperature": 0.0, "seed": 7}

    with pytest.raises(ValidationError, match="model_reasoning_type"):
        UserRequest.model_validate({
            **_request().model_dump(),
            "model_reasoning_type": "thinking",
        })


@pytest.mark.asyncio
async def test_predefined_run_uses_system_defaults_without_saving_a_setting(tmp_path, monkeypatch) -> None:
    from zevo.api.routers.shared import runs as runs_router

    monkeypatch.setenv("ZEVO_WORK_DIR", str(tmp_path / "runs"))
    default_host_id = "default-empire-ai"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"ZEVO_DEFAULT_COMPUTE=connection:{default_host_id}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ZEVO_ENV_FILE", str(env_file))

    async def settled(run, request):
        return request, {"test_set": request.test_set}, "already settled"

    monkeypatch.setattr(runs_router, "_settle_splits", settled)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async with Session() as db:
        db.add(Agent(
            id="orchestrator", name="Orchestrator", title="Supervisor",
            identity_path="playbook/agents/orchestrator/identity.md",
        ))
        db.add(SshHost(
            id=default_host_id,
            label="Empire AI Beta",
            host="beta.example.edu",
            port=22,
            username="researcher",
            category="cluster",
            key_path="/tmp/test-key",
            remote_dir="/projects/researcher/zevo",
            env_setup="source /projects/researcher/env.sh",
            status="verified",
        ))
        db.add(Task(metric="accuracy",
            name="budgeted", task_objective="Improve the model.",
            metric_direction="max",
            test_set="/data/test.csv", test_answer_fields=["answer"],
            test_sample_submission="/data/sample.csv",
        ))
        await db.commit()

        response = await create_run(CreateRunRequest(
            task_name="budgeted", run_name="budgeted-defaults",
        ), db)
        run = await db.get(Run, response.run_id)
        settings = (await db.execute(
            select(TaskSetting).where(TaskSetting.task_name == "budgeted")
        )).scalars().all()

        assert run is not None
        assert run.iteration_budget == 0
        assert run.max_cost_usd == 0
        assert run.max_runtime_hours == 0
        assert run.stop_threshold is None
        assert run.generation_backend == "vllm"
        assert run.gpu_provider == "cluster"
        assert run.ssh_host_id == default_host_id
        assert run.num_gpus == 0
        assert run.task_objective == "Improve the model."
        assert run.agent_objective == (
            "Improve the model. NO training set is provided — acquire and curate "
            "the training data yourself. Choose the training method(s) yourself. "
            "Pick the base model yourself."
        )
        sup = await db.get(Ticket, run.supervisor_ticket_id)
        assert sup is not None
        assert sup.payload["task_objective"] == "Improve the model."
        assert sup.payload["user_request"]["task_objective"] == "Improve the model."
        assert sup.payload["agent_objective"] == run.agent_objective
        assert settings == []

    await engine.dispose()


@pytest.mark.asyncio
async def test_validation_carve_preserves_source_pin_and_initializes_governance(
    tmp_path, monkeypatch,
) -> None:
    from zevo.api.routers.shared import runs as runs_router

    original = "/uploads/train.csv"
    resolved = str(tmp_path / "train-minus-validation.csv")
    monkeypatch.setenv("ZEVO_WORK_DIR", str(tmp_path / "runs"))

    async def settled(run, request):
        return (
            request.model_copy(update={"dataset": resolved}),
            {
                "test_set": request.test_set,
                "validation_set": str(tmp_path / "validation.csv"),
            },
            "validation carved from training",
        )

    monkeypatch.setattr(runs_router, "_settle_splits", settled)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async with Session() as db:
        db.add(Agent(
            id="orchestrator", name="Orchestrator", title="Supervisor",
            identity_path="playbook/agents/orchestrator/identity.md",
        ))
        db.add(Task(
            name="carved", task_objective="Improve the model.", metric="accuracy",
            metric_direction="max", test_set="/data/test.csv",
            test_answer_fields=["answer"], test_sample_submission="/data/sample.csv",
        ))
        await db.commit()

        request = _request().model_copy(update={
            "task_objective": "Improve the model.",
            "dataset": original,
            "base_model": "Qwen/Qwen3-0.6B-Base",
            "training_method": "lora_sft",
        })
        response = await create_run(CreateRunRequest(
            task_name="carved", run_name="carved-run", user_request=request,
            gpu_provider="instance",
        ), db)
        run = await db.get(Run, response.run_id)
        assert run.decision_pins["dataset_source"] == original
        assert run.decision_pins["dataset"] == resolved
        assert "active_rules" not in Run.__table__.columns
        assert "iteration_intents" not in Run.__table__.columns

    await engine.dispose()


@pytest.mark.asyncio
async def test_predefined_task_test_metric_can_be_overridden_for_one_run(tmp_path, monkeypatch) -> None:
    """A Task supplies defaults without locking one Run's Test scorer."""
    monkeypatch.setenv("ZEVO_WORK_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ZEVO_EVALUATORS_ROOT", str(tmp_path / "evaluators"))
    evaluator = tmp_path / "benchmark_average.py"
    evaluator.write_text("# valid custom evaluator\n", encoding="utf-8")
    test_set = tmp_path / "test.csv"
    test_sample = tmp_path / "test_sample.csv"
    validation_set = tmp_path / "validation.csv"
    validation_sample = tmp_path / "validation_sample.csv"
    test_set.write_text("question,answer\nq,a\n", encoding="utf-8")
    test_sample.write_text("prediction\na\n", encoding="utf-8")
    validation_set.write_text("question,answer\nv,a\n", encoding="utf-8")
    validation_sample.write_text("prediction\na\n", encoding="utf-8")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async with Session() as db:
        db.add(Agent(
            id="orchestrator", name="Orchestrator", title="Supervisor",
            identity_path="playbook/agents/orchestrator/identity.md",
        ))
        db.add(Task(metric="accuracy",
            name="lower-is-better",
            task_objective="Reduce the benchmark error.",
            metric_direction="min",
            test_set="/data/test.csv", test_answer_fields=["answer"],
            test_sample_submission="/data/sample.csv",
        ))
        await db.commit()

        request = _request().model_copy(update={
            "metric_type": "custom",
            "metric": "benchmark_average",
            "metric_direction": "max",
            "evaluation_script": str(evaluator),
            "evaluator_sha256": "",
            "test_set": str(test_set),
            "test_sample_submission": str(test_sample),
            "validation_set": str(validation_set),
            "validation_answer_fields": ["answer"],
            "validation_sample_submission": str(validation_sample),
        })
        response = await create_run(CreateRunRequest(
            task_name="lower-is-better",
            run_name="test-metric-override",
            user_request=request,
            gpu_provider="instance",
        ), db)
        run = await db.get(Run, response.run_id)
        assert run is not None
        assert run.metric == "benchmark_average"
        assert run.metric_direction == "max"
        assert run.holdout["test_metric_type"] == "custom"
        assert run.holdout["test_evaluation_script"] != str(evaluator)
        assert run.holdout["test_evaluator_sha256"]

        # The reusable Task remains the default for later launches.
        task = await db.get(Task, "lower-is-better")
        assert task is not None
        assert task.metric_type == "builtin"
        assert task.metric == "accuracy"
        assert task.metric_direction == "min"

    await engine.dispose()


@pytest.mark.asyncio
async def test_saving_first_custom_setting_creates_task_and_setting(tmp_path, monkeypatch) -> None:
    """The Launch form's Save choice persists both reusable objects."""
    from zevo.api.routers.shared import runs as runs_router

    monkeypatch.setenv("ZEVO_WORK_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ZEVO_EVALUATORS_ROOT", str(tmp_path / "evaluators"))
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text("# valid custom evaluator\n", encoding="utf-8")

    async def settled(run, request):
        return request, {"test_set": request.test_set}, "already settled"

    monkeypatch.setattr(runs_router, "_settle_splits", settled)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    request = _request().model_copy(update={
        "task_objective": "Answer the benchmark questions.",
        "metric_direction": "min",
        "test_set": "/data/test.csv",
        "test_answer_fields": ["answer"],
        "test_sample_submission": "/data/sample.csv",
        "metric_type": "custom",
        "evaluation_script": str(evaluator),
    })
    async with Session() as db:
        db.add(Agent(
            id="orchestrator", name="Orchestrator", title="Supervisor",
            identity_path="playbook/agents/orchestrator/identity.md",
        ))
        await db.commit()

        response = await create_run(CreateRunRequest(
            task_name="new-benchmark",
            run_name="first-attempt",
            user_request=request,
            gpu_provider="instance",
            save_setting=True,
            setting_name="baseline",
        ), db)

        task = await db.get(Task, "new-benchmark")
        settings = (await db.execute(
            select(TaskSetting).where(TaskSetting.task_name == "new-benchmark")
        )).scalars().all()
        assert task is not None
        assert task.task_objective == "Answer the benchmark questions."
        assert task.evaluation_script.endswith(".py")
        assert len(task.evaluator_sha256) == 64
        assert task.metric_direction == "min"
        run = await db.get(Run, response.run_id)
        assert run is not None
        assert run.metric_direction == "min"
        assert run.task_objective == "Answer the benchmark questions."
        assert run.agent_objective.startswith("Answer the benchmark questions. ")
        assert "NO training set is provided" in run.agent_objective
        assert run.setting_id == settings[0].id
        assert run.setting_name == "baseline"
        assert len(settings) == 1
        assert settings[0].name == "baseline"
        assert "metric_direction" not in {c.name for c in TaskSetting.__table__.columns}

    await engine.dispose()


@pytest.mark.asyncio
async def test_run_request_reports_the_persisted_runtime() -> None:
    from zevo.api.routers.shared.runs import get_run_request

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async with Session() as db:
        db.add(Run(metric="accuracy",
            id="runtime-request", task_name="", task_objective="inspect runtime",
            agent_objective="inspect runtime",
            status="running", gpu_provider="cluster", num_gpus=4,
            generation_backend="hf", holdout={}, history=[],
        ))
        await db.commit()

        response = await get_run_request(
            "runtime-request", Request({"type": "http", "headers": []}), db,
        )
        assert response.gpu_provider == "cluster"
        assert response.num_gpus == 4
        assert response.generation_backend == "hf"

    await engine.dispose()


def test_run_runtime_fields_have_one_canonical_location() -> None:
    body = CreateRunRequest.model_validate({
        "task_name": "runtime",
        "run_name": "runtime-contract",
        "user_request": _request().model_dump(),
        "generation_backend": "hf",
        "num_gpus": 2,
        "gpu_provider": "cluster",
        "max_runtime_hours": 10,
    })
    assert body.generation_backend == "hf"
    assert body.num_gpus == 2
    assert body.gpu_provider == "cluster"
    assert body.max_runtime_hours == 10
    dumped_request = body.user_request.model_dump()
    assert {
        "generation_backend", "num_gpus", "gpu_provider", "max_runtime_hours",
    }.isdisjoint(dumped_request)

    # A deadline is accepted only on this Run envelope, never on the reusable
    # Setting contract.
    from zevo.api.routers.ui.tasks import SettingBody
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        SettingBody.model_validate({"name": "s1", "max_runtime_hours": 10})

    for nested_runtime_key, value in (
        ("generation_backend", "hf"),
        ("num_gpus", 2),
        ("gpu_provider", "cluster"),
    ):
        with pytest.raises(ValidationError):
            UserRequest.model_validate({
                **_request().model_dump(),
                nested_runtime_key: value,
            })


def test_stop_threshold_is_optional_and_uses_the_metric_scale() -> None:
    from zevo.api.routers.ui.tasks import SettingBody

    assert CreateRunRequest(task_name="t", run_name="r").stop_threshold is None
    assert CreateRunRequest(
        task_name="t", run_name="r", stop_threshold=0.0,
    ).stop_threshold == 0.0
    assert CreateRunRequest(
        task_name="t", run_name="r", stop_threshold=1.0,
    ).stop_threshold == 1.0
    assert SettingBody(name="s", stop_threshold=0.5).stop_threshold == 0.5
    assert CreateRunRequest(
        task_name="t", run_name="r", stop_threshold=-12.5,
    ).stop_threshold == -12.5
    assert SettingBody(name="s", stop_threshold=1800).stop_threshold == 1800


def test_full_pipeline_rejects_user_owned_experiment_contract_details() -> None:
    from pydantic import ValidationError

    customized_request = UserRequest.model_validate({
        **_request().model_dump(),
        "training_method": "dpo",
        "prompt_framing": "chat",
        "system_prompt": "Answer concisely.",
        "loss_objective_config": {"beta": 0.2},
        "inference_config": {"input_fields": ["question"]},
        "decoding_config": {"max_new_tokens": 128, "temperature": 0.0, "seed": 7},
    })

    customized = CreateRunRequest(
        task_name="t",
        run_name="r",
        mode="customized_pipeline",
        user_request=customized_request,
    )
    assert customized.user_request is not None
    assert customized.user_request.prompt_framing == "chat"
    assert customized.user_request.decoding_config["seed"] == 7

    with pytest.raises(ValidationError, match="Full Pipeline does not accept"):
        CreateRunRequest(
            task_name="t",
            run_name="r",
            mode="full_pipeline",
            user_request=customized_request,
        )

    autonomous = CreateRunRequest(
        task_name="t",
        run_name="r",
        mode="full_pipeline",
        user_request=_request(),
    )
    assert autonomous.user_request is not None
    assert autonomous.user_request.prompt_framing == ""
    assert autonomous.user_request.decoding_config == {}


def test_test_setup_is_required_by_the_run_contract() -> None:
    from zevo.contracts.orchestrator import scoring_asset_errors

    incomplete = _request().model_copy(update={
        "test_set": "", "test_answer_fields": [], "test_sample_submission": "",
    })
    assert scoring_asset_errors(incomplete) == [
        "test_set is required",
        "test_answer_fields are required",
        "test_sample_submission is required",
    ]


def test_test_derived_validation_inherits_the_complete_test_metric_contract() -> None:
    from zevo.contracts.orchestrator import (
        inherit_test_validation_contract,
        scoring_asset_errors,
    )

    contradictory = _request().model_copy(update={
        "metric_type": "builtin",
        "metric": "accuracy",
        "metric_direction": "min",
        "validation_metric_type": "custom",
        "validation_metric": "unrelated_score",
        "validation_metric_direction": "max",
        "validation_evaluation_script": "/wrong/evaluator.py",
        "validation_answer_fields": ["wrong_answer"],
        "validation_sample_submission": "/wrong/sample.csv",
    })

    resolved = inherit_test_validation_contract(contradictory)

    assert resolved.validation_metric_type == "builtin"
    assert resolved.validation_metric == "accuracy"
    assert resolved.validation_metric_direction == "min"
    assert resolved.validation_evaluation_script == ""
    assert resolved.validation_answer_fields == []
    assert resolved.validation_sample_submission == ""
    assert scoring_asset_errors(contradictory) == []

    independent = contradictory.model_copy(update={"validation_set": "/data/val.csv"})
    assert inherit_test_validation_contract(independent) is independent


def test_test_derived_validation_accepts_blank_metric_inputs_from_launch_ui() -> None:
    from zevo.contracts.orchestrator import (
        inherit_test_validation_contract,
        scoring_asset_errors,
    )

    payload = _request().model_dump()
    for field in (
        "validation_metric_type",
        "validation_metric",
        "validation_metric_direction",
    ):
        payload.pop(field)

    draft = UserRequest.model_validate(payload)

    assert draft.validation_metric_type == ""
    assert draft.validation_metric == ""
    assert draft.validation_metric_direction == ""
    assert scoring_asset_errors(draft) == []

    resolved = inherit_test_validation_contract(draft)
    assert resolved.validation_metric_type == draft.metric_type
    assert resolved.validation_metric == draft.metric
    assert resolved.validation_metric_direction == draft.metric_direction


def test_independent_validation_still_requires_its_metric_contract() -> None:
    from zevo.contracts.orchestrator import scoring_asset_errors

    independent = _request().model_copy(update={
        "validation_set": "/data/val.csv",
        "validation_answer_fields": ["answer"],
        "validation_sample_submission": "/data/val-sample.csv",
        "validation_metric_type": "",
        "validation_metric": "",
        "validation_metric_direction": "",
    })

    assert scoring_asset_errors(independent) == [
        "validation_set requires validation_metric_type",
        "validation_set requires validation_metric_direction",
    ]


def test_task_metric_contract_names_are_canonical_and_old_names_are_rejected() -> None:
    req = _request().model_copy(update={
        "test_sample_submission": "/data/test-sample.csv",
        "metric_type": "custom",
        "evaluation_script": "/data/evaluate.py",
    })
    assert req.test_sample_submission == "/data/test-sample.csv"
    assert req.evaluation_script == "/data/evaluate.py"

    from pydantic import ValidationError
    for old_key, value in (
        ("sample_submission", "/data/old-sample.csv"),
        ("test_eval_script", "/data/old-eval.py"),
        ("validation_eval_script", "/data/old-val-eval.py"),
    ):
        with pytest.raises(ValidationError):
            UserRequest.model_validate({**_request().model_dump(), old_key: value})


def test_a_real_zero_test_score_is_not_rendered_as_missing() -> None:
    run = Run(metric="accuracy", validation_metric="token_f1",
        validation_metric_direction="max",
        id="r-zero", task_name="zero", run_name="zero-score",
        task_objective="measure zero", agent_objective="measure zero",
        mode="full_pipeline", metric_direction="max", customizations={},
        status="success", summary="",
        registry_version_tag="", halted_reason="", history=[],
        holdout={}, num_gpus=1,
        iteration_budget=1, iterations_completed=0,
        stop_threshold=0.0, best_validation_score=0.0, max_cost_usd=0.0,
        champion_test_score=0.0,
    )
    summary = _summary(run, holdout=True)
    assert summary.champion_test_score == 0.0
    assert summary.best_validation_score == 0.0


def test_run_summary_derives_best_validation_from_validation_history() -> None:
    run = Run(
        metric="accuracy", validation_metric="token_f1",
        validation_metric_direction="max",
        id="r-score-lanes", task_name="score-lanes", run_name="score-lanes",
        task_objective="keep score lanes separate",
        agent_objective="keep score lanes separate",
        mode="full_pipeline", metric_direction="max", customizations={},
        status="running", summary="", registry_version_tag="",
        halted_reason="",
        history=[
            {"iteration": 0, "source": "baseline", "score": 0.1},
            {"iteration": 1, "source": "trained", "score": 0.4},
            {"iteration": 2, "source": "trained", "score": 0.3},
        ],
        holdout={}, num_gpus=1, iteration_budget=3,
        iterations_completed=2, stop_threshold=None,
        # Simulate the old bug: a held-out Test score contaminated this column.
        best_validation_score=0.9, max_cost_usd=0.0,
    )

    summary = _summary(run)

    assert summary.best_validation_score == 0.4


def test_a_setting_naming_a_validation_set_must_name_its_shape() -> None:
    """The API enforces what the form enforces, or the form is only advice.

    A setting with a validation set and no answer fields sends the harness to
    the TEST side for them — different columns and a different binding — and
    the data agent fails one ticket in, after an hour
    and a GPU rental. That run exists; it is data-021. The dialog started
    refusing to save it, but `POST /tasks/{n}/settings` still accepted it, so
    anything that was not the dialog could still create one.
    """
    import pytest
    from fastapi import HTTPException
    from zevo.api.routers.ui.tasks import SettingBody, _validate_named_validation_assets

    complete = SettingBody(
        name="s1", validation_set="/data/val.csv",
        validation_answer_fields=["response"],
        validation_sample_submission="/data/val_sample.csv",
    )
    _validate_named_validation_assets(complete)  # does not raise

    # No validation set at all is the carve path; Data prepares its binding.
    _validate_named_validation_assets(SettingBody(name="s2"))

    for missing in ("validation_answer_fields", "validation_sample_submission"):
        blank = complete.model_copy(update={
            missing: [] if missing.endswith("fields") else ""})
        with pytest.raises(HTTPException) as e:
            _validate_named_validation_assets(blank)
        assert e.value.status_code == 400
        assert missing in e.value.detail


def test_task_and_agent_objectives_need_no_string_reverse_parsing() -> None:
    from zevo.api.routers.ui.tasks import compose_objective, setting_clauses

    problem = "Turn the base model into an instruction-following model"
    for dataset, model, method in (
        ("/d/train.json", "Qwen/Qwen3-0.6B-Base", "full_sft"),   # L1, all pinned
        ("", "", ""),                                            # L4, none pinned
        ("/d/train.json", "", "dpo"),                            # a mix
    ):
        full = compose_objective(problem, dataset=dataset, base_model=model,
                                 training_method=method)
        assert full == " ".join([
            problem,
            *setting_clauses(dataset=dataset, base_model=model, training_method=method),
        ])


def test_a_failed_ticket_stops_a_run_reading_as_a_clean_success() -> None:
    """A registered model must not outvote a failure.

    The reconciler closed any run that produced a registered model as
    `success`, whatever else had failed — and the thing most likely to settle
    last is the multi-hop held-out branch, even though it starts alongside
    Validation, because it still needs the GPU. Run 28845478 lost its final test measurement to
    an expired allocation, scored its last iteration on validation alone, and
    reported a green success.

    It is not a write-off either: seven iterations ran and a model came out.
    `degraded` is that distinction.
    """
    from zevo.contracts.tickets import TERMINAL_RUN_STATUSES
    from zevo.engine.run.scheduler.reconciler import decide_run_status

    # Reconciliation never invents a clean success; only an explicit
    # Orchestrator terminal PATCH may produce it.
    assert decide_run_status(has_registered_model=True, failed_ticket_ids=[]) == "degraded"
    assert decide_run_status(
        has_registered_model=True,
        failed_ticket_ids=["holdout-infer-28845478-008"]) == "degraded"
    assert decide_run_status(has_registered_model=False, failed_ticket_ids=["x"]) == "failed"
    assert decide_run_status(has_registered_model=False, failed_ticket_ids=[]) == "halted"

    # A degraded run is OVER. Leaving it out of the terminal set means every
    # daemon keeps picking it up and the UI counts it as still going.
    assert "degraded" in TERMINAL_RUN_STATUSES


def test_cli_defaults_use_the_canonical_user_request_keys() -> None:
    from zevo.cli.zevo import _UR_DEFAULTS

    assert "task_objective" in _UR_DEFAULTS
    assert "objective" not in _UR_DEFAULTS
    assert "gpu_provider" not in _UR_DEFAULTS
    assert "metric_direction" not in _UR_DEFAULTS
    UserRequest.model_validate({
        **_UR_DEFAULTS,
        "task_objective": "improve it",
        "metric": "accuracy",
        "metric_direction": "max",
    })


def test_worker_generation_backend_defaults_match_the_run_default() -> None:
    from zevo.contracts.inference import InferenceTaskInput
    from zevo.contracts.train import TrainTaskInput

    assert TrainTaskInput.model_fields["generation_backend"].default == "vllm"
    assert InferenceTaskInput.model_fields["generation_backend"].default == "vllm"


def test_journal_request_accepts_only_orchestrator_narrative_fields() -> None:
    from pydantic import ValidationError

    from zevo.api.routers.shared.runs import PatchRunRequest

    for extra in ("method", "training_method", "base_model", "score", "accuracy", "notes"):
        with pytest.raises(ValidationError):
            PatchRunRequest.model_validate({
                "history_entry": {
                    "iteration": 1,
                    "source": "trained",
                    "action": "Applied full SFT.",
                    "result": "Validation increased.",
                    "analysis": "The controlled training change explains the gain.",
                    "next": "Inspect errors before the next iteration.",
                    extra: "not orchestrator-owned",
                },
            })


def test_run_journal_requires_all_four_nonempty_parts() -> None:
    from pydantic import ValidationError

    from zevo.api.routers.shared.runs import PatchRunRequest

    with pytest.raises(ValidationError):
        PatchRunRequest.model_validate({
            "history_entry": {
                "iteration": 0,
                "source": "baseline",
                "action": "Measure baseline",
                "result": "VAL=0.2",
                "analysis": "",
                "next": "Train",
            },
        })


def test_run_patch_rejects_engine_owned_facts() -> None:
    from pydantic import ValidationError

    from zevo.api.routers.shared.runs import PatchRunRequest

    for field, value in (
        ("headline_score", 0.5),
        ("best_score", 0.5),
        ("iterations_completed", 1),
        ("registry_version_tag", "M-run"),
    ):
        with pytest.raises(ValidationError):
            PatchRunRequest.model_validate({field: value})


@pytest.mark.asyncio
async def test_supervisor_can_explicitly_finish_its_run() -> None:
    from datetime import datetime, timezone

    from zevo.api.routers.shared.runs import PatchRunRequest, patch_run

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    complete = {
        "iteration": 1, "source": "trained", "score": 0.5,
        "action": "trained", "result": "improved",
        "analysis": "validated", "next": "stop",
    }
    async with Session() as db:
        db.add(Run(metric="accuracy",
            id="explicit-success", task_name="task", status="running",
            supervisor_ticket_id="orchestrate-explicit-001",
            registry_version_tag="M-explicit", iterations_completed=1,
            history=[complete], started_at=datetime.now(timezone.utc),
        ))
        db.add(Ticket(
            id="orchestrate-explicit-001", run_id="explicit-success",
            agent_id="orchestrator", status="running", payload={},
        ))
        db.add(Ticket(
            id="registry-explicit-001", run_id="explicit-success",
            agent_id="registry", status="succeeded", payload={}, iteration=1,
        ))
        await db.commit()

        result = await patch_run(
            "explicit-success",
            PatchRunRequest(status="success", summary="Champion retained."),
            db,
        )
        assert result.status == "success"
        assert result.summary == "Champion retained."

    await engine.dispose()


@pytest.mark.asyncio
async def test_journal_patch_changes_only_narrative_on_an_existing_fact() -> None:
    from datetime import datetime, timezone

    from fastapi import HTTPException

    from zevo.api.routers.shared.runs import PatchRunRequest, patch_run

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    factual = {
        "iteration": 1,
        "source": "trained",
        "method_ids": ["inline_transform"],
        "training_method": "full_sft",
        "base_model": "base/model",
        "score": 0.42,
        "test_score": 0.39,
        "action": "",
        "result": "",
        "analysis": "",
        "next": "",
    }
    async with Session() as db:
        db.add(Run(metric="accuracy",
            id="journal-run",
            task_name="task",
            status="running",
            started_at=datetime.now(timezone.utc),
            history=[factual],
        ))
        await db.commit()
        await patch_run("journal-run", PatchRunRequest(history_entry={
            "iteration": 1,
            "source": "trained",
            "action": "Applied full SFT.",
            "result": "Validation reached 0.42.",
            "analysis": "The controlled change produced the gain.",
            "next": "Inspect errors.",
        }), db)
        run = await db.get(Run, "journal-run")
        assert run is not None
        await db.refresh(run)
        row = run.history[0]
        assert row["base_model"] == "base/model"
        assert row["score"] == 0.42
        assert row["test_score"] == 0.39
        assert row["action"] == "Applied full SFT."
        with pytest.raises(HTTPException) as exc:
            await patch_run("journal-run", PatchRunRequest(history_entry={
                    "iteration": 2,
                    "source": "trained",
                    "action": "a", "result": "Validation was measured at 0.5.",
                    "analysis": "why", "next": "n",
                }), db)
        assert exc.value.status_code == 409

    await engine.dispose()
