"""The run tunes on validation and is judged on test — prove they stay apart.

Three things have to hold or the arrangement is decorative:

  1. the carve produces a validation set something can actually score, takes
     its rows OUT of wherever they came from, and produces the same split every
     time;
  2. the orchestrator is handed the validation paths and no test path at all,
     and its history never carries a held-out number;
  3. a held-out ticket does not wake the supervisor, and the score API reveals
     the test series only to the dashboard proxy or after completion.

Each of these was a decision someone could undo without noticing — a payload
field copied through, a filter dropped from a query — and none of them fails
loudly. The run would keep working and quietly report a fitted number.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from starlette.requests import Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Agent, AgentWakeupRequest, Base, Run, ScoreEvent, Ticket, WorkProduct
from zevo.engine.method.validation_split import SplitError, carve, pick_source, resolve


# ───────────────────────────── fixtures ──────────────────────────────────────


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_instance_device(path: Path, *, run_id: str) -> None:
    path.write_text(json.dumps({
        "schema_version": 1,
        "run_id": run_id,
        "ticket_id": f"infra-{run_id}-001",
        "purpose": "train",
        "provider": "instance",
        "cloud_backend": "",
        "instance_id": "fixed-host",
        "auto_release": False,
        "host": "gpu.example",
        "ssh": {
            "host": "gpu.example", "port": 22, "user": "u",
            "key_path": "/tmp/test-key",
        },
        "instance": {
            "env_setup": "", "workdir": f"/work/{run_id}",
            "hf_cache": "/work/hf", "visible_devices": "0",
        },
        "gpu": {
            "has_gpu": True, "gpu_count": 1, "gpu_name": "A100",
            "vram_gb": 80, "vram_mb": 81920,
            "devices": [{"index": 0, "name": "A100", "vram_mb": 81920}],
        },
        "cuda": {"driver_version": "570", "cuda_version": "12"},
        "cost": {"dph_total": 0},
        "resource_plan": {
            "purpose": "train",
            "required_working_set_gib": 12,
            "host_memory_components_gib": {"model_and_runtime": 12},
            "host_memory_formula_gib": 32,
            "site_min_ram_gib": 0,
            "num_gpus": 1, "min_vram_gb": 40, "min_ram_gb": 32,
            "min_cpus": 8, "time_limit_hours": 0, "disk_gb": 0,
            "rationale": "fixed test host",
        },
        "probe_source": "remote_nvidia-smi",
        "probed_at": "2026-09-02T00:00:00Z",
    }), encoding="utf-8")


@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    """A task laid out the way `data/files/<name>/` is."""
    d = tmp_path / "task"
    train = [{"question": f"q{i}", "answer": f"long a{i}", "gold": str(i)} for i in range(400)]
    test = [{"question": f"t{i}", "answer": f"long a{i}", "gold": str(i)} for i in range(1_000)]
    _write_csv(d / "train.csv", ["question", "answer", "gold"], train)
    _write_csv(d / "test.csv", ["question", "answer", "gold"], test)
    _write_csv(
        d / "sample_submission.csv", ["id", "prediction"],
        [{"id": "example-id", "prediction": "<model output>"}],
    )
    return d


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        for aid in ("data", "infrastructure", "train", "inference",
                    "evaluation", "registry", "orchestrator"):
            s.add(Agent(
                id=aid, name=aid, title=aid, reports_to="orchestrator",
                default_driver="stub", default_model="stub-model",
                output_schema="", tools=[], identity_path="",
            ))
        await s.commit()
    yield Session
    await engine.dispose()


# ─────────────────────────────── the carve ───────────────────────────────────


def test_carve_takes_validation_rows_out_of_test_only(task_dir: Path, tmp_path: Path) -> None:
    """Validation leaves Test while the complete Training population is untouched."""
    out = carve(
        dataset=str(task_dir / "train.csv"),
        test_set=str(task_dir / "test.csv"),
        test_answer_fields=["answer", "gold"],
        out_dir=str(tmp_path / "splits"),
    )
    assert out.source == "test"
    val = list(csv.DictReader(open(out.validation_set)))
    rest = list(csv.DictReader(open(out.test_set)))
    train = list(csv.DictReader(open(out.dataset)))
    assert len(val) == 200
    assert len(val) + len(rest) == 1_000
    assert len(train) == 400
    assert not {r["question"] for r in val} & {r["question"] for r in rest}


def test_prediction_preview_joins_questions_by_id(tmp_path: Path) -> None:
    from zevo.api.routers.shared.runs import (
        _prediction_with_questions_preview,
        _prediction_with_questions_rows,
        _preview_columns,
    )

    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        '{"id":"a","question":"Why?"}\n'
        '{"id":"b","question":"How?"}\n',
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.csv"
    _write_csv(
        predictions, ["id", "answer"],
        [{"id": "a", "answer": "Because"}, {"id": "b", "answer": "Carefully"}],
    )

    preview = _prediction_with_questions_preview(predictions, questions)
    assert "Question + prediction preview" in preview
    assert '"question": "Why?"' in preview
    assert '"prediction": "Because"' in preview
    rows = _prediction_with_questions_rows(predictions, questions)
    assert _preview_columns(rows) == ["id", "question", "prediction"]
    assert rows[1] == {"id": "b", "question": "How?", "prediction": "Carefully"}
    second_page = _prediction_with_questions_rows(
        predictions, questions, limit=1, offset=1,
    )
    assert second_page == [
        {"id": "b", "question": "How?", "prediction": "Carefully"},
    ]

    truth = tmp_path / "truth.csv"
    _write_csv(
        truth, ["id", "question", "answer"],
        [{"id": "a", "question": "Why?", "answer": "Indeed"},
         {"id": "b", "question": "How?", "answer": "Carefully"}],
    )
    rows_with_truth = _prediction_with_questions_rows(
        predictions, questions, ground_truth=truth, answer_fields=["answer"],
    )
    assert _preview_columns(rows_with_truth) == [
        "id", "question", "ground_truth", "prediction",
    ]
    assert rows_with_truth[0]["ground_truth"] == "Indeed"
    assert rows_with_truth[0]["prediction"] == "Because"


def test_carve_keeps_the_columns_eval_scores_on(task_dir: Path, tmp_path: Path) -> None:
    """The carve keeps the ground truth: the questions-only copy is the data
    agent's job, built later by dropping exactly the columns reported here."""
    out = carve(
        dataset=str(task_dir / "train.csv"),
        test_set=str(task_dir / "test.csv"),
        test_answer_fields=["answer", "gold"],
        out_dir=str(tmp_path / "splits"),
    )
    # The FULL set, ground truth intact — the questions-only copy is the data
    # agent's job now, built by dropping exactly these columns.
    assert csv.DictReader(open(out.validation_set)).fieldnames == ["question", "answer", "gold"]
    assert out.validation_answer_fields == ["answer", "gold"]


def test_carve_is_reproducible(task_dir: Path, tmp_path: Path) -> None:
    """Same inputs, same split — two runs of one task have to be comparable."""
    kw = dict(
        dataset=str(task_dir / "train.csv"),
        test_set=str(task_dir / "test.csv"),
        test_answer_fields=["answer", "gold"],
    )
    a = carve(out_dir=str(tmp_path / "a"), **kw)
    b = carve(out_dir=str(tmp_path / "b"), **kw)
    assert Path(a.validation_set).read_text() == Path(b.validation_set).read_text()


def test_training_shape_does_not_change_test_derived_validation(
    task_dir: Path, tmp_path: Path,
) -> None:
    """A different Training schema cannot redirect Validation away from Test."""
    prompts = tmp_path / "prompts.csv"
    _write_csv(prompts, ["prompt", "completion"],
               [{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(50)])
    kind, _ = pick_source(dataset=str(prompts), test_set=str(task_dir / "test.csv"))
    assert kind == "test"

    out = carve(
        dataset=str(prompts),
        test_set=str(task_dir / "test.csv"),
        test_answer_fields=["answer", "gold"],
        out_dir=str(tmp_path / "splits"),
    )
    assert out.source == "test"
    assert not out.needs_metric_binding
    assert out.validation_answer_fields == ["answer", "gold"]
    val = list(csv.DictReader(open(out.validation_set)))
    rest = list(csv.DictReader(open(out.test_set)))
    assert len(val) == 200 and len(val) + len(rest) == 1_000
    assert len(list(csv.DictReader(open(prompts)))) == 50


def test_a_carve_matching_the_task_binding_needs_no_mapping(
    task_dir: Path, tmp_path: Path,
) -> None:
    """The common case must not pay for the rare one: when the training file
    carries the Task answer fields, the carved source already satisfies the
    frozen metric binding."""
    out = carve(
        dataset=str(task_dir / "train.csv"),
        test_set=str(task_dir / "test.csv"),
        test_answer_fields=["answer", "gold"],
        out_dir=str(tmp_path / "splits"),
    )
    assert not out.needs_metric_binding
    assert out.validation_answer_fields == ["answer", "gold"]


def test_a_run_does_not_need_materialized_training_to_split_test(
    task_dir: Path, tmp_path: Path,
) -> None:
    """Missing Training is valid because the split is defined only by Test."""
    assert pick_source(dataset="", test_set=str(task_dir / "test.csv"))[0] == "test"
    out = carve(
        dataset="", test_set=str(task_dir / "test.csv"),
        test_answer_fields=["answer", "gold"], out_dir=str(tmp_path / "splits"),
    )
    assert out.n_validation == 200 and out.n_remaining == 800
    assert len(list(csv.DictReader(open(task_dir / "test.csv")))) == 1_000


def test_preflight_accepts_a_single_submission_example_for_large_test(
    task_dir: Path,
) -> None:
    """The template defines output shape; it is not a 1,000-row prediction."""
    from zevo.api.routers.ui.preflight import _check_test_set
    from zevo.contracts.orchestrator import UserRequest

    request = UserRequest(
        task_objective="o", metric="token_f1", metric_direction="max",
        validation_metric_type="builtin", validation_metric="token_f1",
        validation_metric_direction="max", training_method="", dataset="",
        base_model="", test_set=str(task_dir / "test.csv"),
        test_answer_fields=["answer"],
        test_sample_submission=str(task_dir / "sample_submission.csv"),
        constraints=[],
    )
    items = []

    _check_test_set(request, items)

    assert [(item.severity, item.code) for item in items] == [
        ("info", "validation_split_ready")
    ]


def test_resolve_prefers_what_the_user_supplied(task_dir: Path, tmp_path: Path) -> None:
    _write_csv(task_dir / "mine.csv", ["question", "gold"], [{"question": "q", "gold": "1"}])
    out = resolve(
        validation_set=str(task_dir / "mine.csv"), validation_answer_fields=["gold"],
        dataset=str(task_dir / "train.csv"), test_set=str(task_dir / "test.csv"),
        test_answer_fields=["gold"],
        out_dir=str(tmp_path / "splits"),
    )
    assert out.source == "user"
    assert out.validation_set == str(task_dir / "mine.csv")
    assert out.dataset == "", "nothing was carved, so nothing should be repointed"


def test_resolve_derives_from_test_instead_of_adopting_a_sibling(
    task_dir: Path, tmp_path: Path,
) -> None:
    """An unnamed Validation is a deterministic Test split, not a sibling guess."""
    _write_csv(task_dir / "validation.csv", ["question", "gold"], [{"question": "q", "gold": "1"}])
    _write_csv(task_dir / "validation_public.csv", ["question"], [{"question": "q"}])
    out = resolve(
        validation_set="", validation_answer_fields=[],
        dataset=str(task_dir / "train.csv"), test_set=str(task_dir / "test.csv"),
        test_answer_fields=["gold"],
        out_dir=str(tmp_path / "splits"),
    )
    assert out.source == "test"
    assert out.validation_set != str(task_dir / "validation.csv")
    assert out.validation_answer_fields == ["gold"]


def test_resolve_refuses_a_validation_set_that_is_not_a_file(
    task_dir: Path, tmp_path: Path,
) -> None:
    """Naming a hub id here used to fall through to the carve: the run tuned on
    a set the user did not pick, and nothing anywhere said so."""
    with pytest.raises(SplitError, match="not a file"):
        resolve(
            validation_set="trl-lib/Capybara", validation_answer_fields=["gold"],
            dataset=str(task_dir / "train.csv"), test_set=str(task_dir / "test.csv"),
            test_answer_fields=["gold"],
            out_dir=str(tmp_path / "splits"),
        )


def test_carve_rejects_missing_test_answer_fields(tmp_path: Path) -> None:
    """A derived Validation binding must exactly match the Test contract."""
    train = tmp_path / "train.csv"
    _write_csv(train, ["question", "gold"], [{"question": f"q{i}", "gold": str(i)} for i in range(40)])
    test = tmp_path / "test.csv"
    _write_csv(test, ["question", "gold"], [{"question": f"t{i}", "gold": "1"} for i in range(1_000)])
    with pytest.raises(SplitError, match="rationale"):
        carve(
            dataset=str(train), test_set=str(test),
            test_answer_fields=["gold", "rationale"],
            declared_columns=["gold", "rationale"], out_dir=str(tmp_path / "o"),
        )


# ─────────────────── what the orchestrator is handed ─────────────────────────


@pytest.mark.asyncio
async def test_run_creation_blanks_the_test_paths(
    task_dir: Path, tmp_path: Path, monkeypatch,
) -> None:
    """The isolation is structural: no test path in the payload means no ticket
    can be aimed at the test set, whatever the agent decides it wants."""
    from zevo.api.routers.shared.runs import _settle_splits
    from zevo.contracts.orchestrator import UserRequest
    import zevo.api.routers.shared.runs as runs_mod
    import zevo.paths as paths_mod

    monkeypatch.setattr(
        runs_mod.settings.__class__, "work_dir_root",
        property(lambda self: str(tmp_path / "work")),
    )
    monkeypatch.setattr(paths_mod, "holdout_root", lambda: tmp_path / "private")

    run = Run(metric="accuracy", id="r-splits", task_name="t", started_at=datetime.now(timezone.utc))
    request = UserRequest(metric="accuracy",
        task_objective="o", metric_direction="max",
        validation_metric_type="builtin", validation_metric="token_f1",
        validation_metric_direction="max",
        training_method="", dataset=str(task_dir / "train.csv"),
        data_query="", base_model="m",
        test_set=str(task_dir / "test.csv"),
        test_answer_fields=["answer", "gold"],
        test_sample_submission=str(task_dir / "sample_submission.csv"),
        metric_type="builtin", evaluation_script="",
        constraints=[],
    )
    agent_request, holdout, _ = await _settle_splits(run, request)

    assert agent_request.test_set == ""
    assert agent_request.test_answer_fields == []
    assert agent_request.validation_set == ""
    assert agent_request.validation_answer_fields == []
    assert agent_request.validation_sample_submission == ""
    assert agent_request.dataset == str(task_dir / "train.csv")
    dumped = agent_request.model_dump()
    assert not any("test.csv" in str(v) for v in dumped.values()), dumped

    # The harness keeps them, because it is the one doing the measuring.
    assert holdout["test_set"] != str(task_dir / "test.csv")
    assert len(list(csv.DictReader(open(holdout["test_set"])))) == 800
    assert holdout["test_answer_fields"] == ["answer", "gold"]
    assert holdout["validation_source"] == "test_split"
    assert holdout["validation_metric_type"] == request.metric_type
    assert agent_request.metric == request.metric
    assert agent_request.metric_direction == request.metric_direction
    assert holdout["validation_policy"] == "supplied"
    assert holdout["validation_fraction"] == 0.0
    assert Path(holdout["validation_set"]).is_file()
    assert len(list(csv.DictReader(open(holdout["validation_set"])))) == 200
    validation_template = list(csv.DictReader(open(
        holdout["validation_sample_submission"]
    )))
    test_template = list(csv.DictReader(open(holdout["test_sample_submission"])))
    assert validation_template == test_template == [
        {"id": "example-id", "prediction": "<model output>"}
    ]


def test_agent_history_drops_the_held_out_column() -> None:
    """The loop reads its own journal to plan the next iteration. Leaving the
    test score in it would let it select on the test set through the back."""
    from zevo.engine.run.runner import _agent_history

    out = _agent_history([
        {"iteration": 1, "score": 0.5, "test_score": 0.41, "action": "lr up"},
        {"iteration": 2, "score": 0.6, "action": "more epochs"},
    ])
    assert all("test_score" not in e for e in out)
    assert out[0]["score"] == 0.5, "the validation score must survive"


# ────────────────────── the held-out lane stays quiet ────────────────────────


@pytest.mark.asyncio
async def test_validation_ticket_is_created_before_its_private_mirror(
    session_factory,
) -> None:
    """Creating Validation Evaluation must not start held-out Test early."""
    from zevo.api.routers.shared.tickets import CreateTicketBody, create_ticket

    async with session_factory() as s:
        s.add(Run(metric="accuracy",
            id="r-order", task_name="t", status="running",
            holdout={
                "test_set": "/task/test.csv", "test_public": "/work/test.csv",
                "validation_set": "/task/val.csv",
                "validation_answer_fields": ["gold"],
                "validation_sample_submission": "/task/val_sample.csv",
                "validation_metric_type": "builtin",
            },
                history=[], started_at=datetime.now(timezone.utc),
            ))
        s.add(Ticket(
            id="infer-r-order-001", run_id="r-order", agent_id="inference",
            status="succeeded", lane="optimization", iteration=0,
            payload={"model_source": "base_model", "base_model": "base/model",
                     "scoring_set": "/work/val.csv",
                     "sample_submission": "/task/val_sample.csv",
                     "prompt_framing": "chat", "inference_config": {},
                     "max_new_tokens": 256, "temperature": 0.0,
                     "top_p": 1.0, "top_k": 0,
                     "repetition_penalty": 1.0, "seed": 0},
            inputs={},
        ))
        s.add(WorkProduct(
            ticket_id="infer-r-order-001", role="predictions",
            path="/work/predictions.csv", meta={},
        ))
        await s.commit()

        created = await create_ticket(CreateTicketBody(
            agent_id="evaluation", run_id="r-order", iteration=0,
            payload={},
            inputs={"predictions": {
                "source_ticket_id": "infer-r-order-001",
                "artifact_role": "predictions",
            }},
        ), db=s)
        assert created.payload["scoring_set"] == "/task/val.csv"
        assert created.payload["answer_fields"] == ["gold"]
        assert created.payload["sample_submission"] == "/task/val_sample.csv"

        rows = (await s.execute(
            select(Ticket).where(Ticket.run_id == "r-order").order_by(Ticket.created_at)
        )).scalars().all()
        measured = [t for t in rows if t.agent_id == "evaluation" or t.lane == "held_out_test"]
        assert [(t.agent_id, t.lane) for t in measured] == [
            ("evaluation", "optimization"),
        ]
        assert measured[0].iteration == 0
        wakes = (await s.execute(
            select(AgentWakeupRequest)
            .where(AgentWakeupRequest.status == "queued")
            .order_by(AgentWakeupRequest.created_at)
        )).scalars().all()
        assert [(w.agent_id, w.ticket_id) for w in wakes] == [
            ("evaluation", measured[0].id),
        ]


@pytest.mark.asyncio
async def test_holdout_paths_resolve_only_when_the_ticket_executes(
    session_factory, tmp_path: Path, monkeypatch,
) -> None:
    """Ticket creation keeps logical paths; the held-out runner resolves bytes.

    The optimization scheduler is allowed to create the held-out Data ticket
    but is intentionally unable to see the private mount. Persisting a physical
    path there either fails to resolve or leaks the storage boundary into the
    work order. Runtime resolution is the only point shared by all held-out
    Data, Inference, and Evaluation tickets.
    """
    from zevo.engine.run.runner import (
        _resolve_heldout_payload_assets,
        _spawn_holdout_data,
    )
    from zevo.holdout_storage import private_mirror, protect_assets

    files = tmp_path / "data" / "files"
    private = tmp_path / "private"
    test_set = files / "capybara" / "test" / "test.json"
    sample = files / "capybara" / "test" / "sample.csv"
    test_set.parent.mkdir(parents=True)
    test_set.write_text('[{"id":"test-1","response":"secret"}]')
    sample.write_text("id,prediction\ntest-1,\n")
    monkeypatch.setenv("ZEVO_FILES_DIR", str(files))
    monkeypatch.setenv("ZEVO_UPLOAD_ROOT", str(tmp_path / "data" / "uploads"))
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(private))

    logical_test, logical_sample = protect_assets(str(test_set), str(sample))
    assert not test_set.exists()
    assert not sample.exists()

    async with session_factory() as s:
        run = Run(
            id="r-runtime-holdout", task_name="capybara", status="running",
            metric="token_f1", metric_direction="max",
            holdout={
                "test_set": logical_test,
                "test_answer_fields": ["response"],
                "test_sample_submission": logical_sample,
                "test_metric_type": "builtin",
                "test_evaluation_script": "",
                "test_evaluator_sha256": "",
                "test_public": "",
            },
            history=[], started_at=datetime.now(timezone.utc),
        )
        source = Ticket(
            id="infer-r-runtime-001", run_id=run.id, agent_id="inference",
            status="succeeded", lane="optimization", iteration=0,
            payload={}, inputs={},
        )
        s.add_all([run, source])
        await s.commit()

        created = await _spawn_holdout_data(
            s, run, source, enqueue=False,
        )
        assert created is not None
        assert created.payload["scoring_set"] == logical_test
        assert created.payload["sample_submission"] == logical_sample

        runtime = _resolve_heldout_payload_assets(created, created.payload)
        assert runtime["scoring_set"] == str(private_mirror(logical_test))
        assert runtime["sample_submission"] == str(private_mirror(logical_sample))
        # Runtime resolution must not write capability-bearing paths to DB.
        assert created.payload["scoring_set"] == logical_test
        assert created.payload["sample_submission"] == logical_sample


@pytest.mark.asyncio
async def test_incomplete_journal_blocks_the_next_pipeline_transition(
    session_factory,
) -> None:
    from fastapi import HTTPException
    from zevo.api.routers.shared.tickets import CreateTicketBody, create_ticket

    async with session_factory() as s:
        s.add(Run(metric="accuracy", validation_metric="token_f1",
            validation_metric_direction="max",
            id="r-journal-gate", task_name="t", status="running",
            history=[{
                "iteration": 0, "source": "baseline", "score": 0.2,
                "action": "", "result": "", "analysis": "", "next": "",
            }],
            started_at=datetime.now(timezone.utc),
        ))
        await s.commit()
        with pytest.raises(HTTPException) as exc:
            await create_ticket(CreateTicketBody(
                agent_id="evaluation", run_id="r-journal-gate", iteration=1,
                payload={}, inputs={},
            ), db=s)
        assert exc.value.status_code == 409
        assert exc.value.detail["error"] == "iteration Journal is incomplete"


@pytest.mark.asyncio
async def test_held_out_ticket_does_not_wake_the_supervisor(session_factory) -> None:
    """`_maybe_wake_supervisor` would put "held-out eval finished" into the
    orchestrator's `last_event` and re-run it. The gate is one `if`."""
    from zevo.db.models import AgentWakeupRequest
    from zevo.engine.run.runner import _maybe_wake_supervisor

    Session = session_factory
    async with Session() as s:
        s.add(Run(metric="accuracy",
            id="r-quiet", task_name="t", task_objective="Improve t.", status="running",
            agent_objective="Improve t.",
            supervisor_ticket_id="orchestrate-r-quiet-001",
            started_at=datetime.now(timezone.utc),
        ))
        s.add(Ticket(
            id="holdout-eval-r-quiet-001", run_id="r-quiet", agent_id="evaluation",
            status="succeeded", lane="held_out_test", payload={}
        ))
        await s.commit()

    # The runner gates on lane before calling this; assert the Ticket is in the
    # held-out-test lane and that an optimization completion does the waking, so a
    # future refactor cannot quietly route both through the same path.
    async with Session() as s:
        tk = (await s.execute(
            select(Ticket).where(Ticket.id == "holdout-eval-r-quiet-001")
        )).scalar_one()
        assert tk.lane == "held_out_test"

        tk.lane = "optimization"
        await s.commit()
        await _maybe_wake_supervisor(s, tk)
        woken = (await s.execute(select(AgentWakeupRequest))).scalars().all()
        assert [w.agent_id for w in woken] == ["orchestrator"]


@pytest.mark.asyncio
async def test_the_lane_prepares_mirrors_and_files_the_result(session_factory) -> None:
    """After Validation Evaluation exists, walk the private measurement lane."""
    from zevo.engine.run.runner import _advance_measurements, _spawn_holdout_infer
    from zevo.contracts.data import DataResult

    Session = session_factory
    async with Session() as s:
        s.add(Run(metric="accuracy",
            id="r-lane", task_name="t", status="running",
            supervisor_ticket_id="orchestrate-r-lane-001",
            iterations_completed=0,
            best_validation_score=0.99,
            holdout={
                "test_set": "/task/test.csv",
                "test_answer_fields": ["answer", "gold"],
                "validation_set": "/task/validation.csv",
                "validation_answer_fields": ["answer", "gold"],
                "validation_public": "/work/validation_public.csv",
                "test_public": "",
                "test_sample_submission": "/task/sample_submission.csv",
                "test_metric_type": "custom",
                "test_evaluation_script": "/task/eval.py",
                "test_evaluator_sha256": "",
            },
                history=[{
                    "iteration": 1, "source": "trained", "base_model": "m",
                    "score": 0.62,
                }],
            started_at=datetime.now(timezone.utc),
        ))
        s.add(Ticket(
            id="infer-001", run_id="r-lane", agent_id="inference",
                status="succeeded", iteration=1, payload={
                    "operation": "run_inference",
                    "model_source": "checkpoint", "base_model": "m",
                    "scoring_set": "/task/validation.csv",
                    "sample_submission": "/task/sample_submission.csv",
                    "configuration_suggestions": {}, "configuration_pins": {},
            }, inputs={
                "checkpoint": {"source_ticket_id": "train-001", "artifact_role": "checkpoint", "work_product_id": "", "path": ""},
                "device_info": {"source_ticket_id": "infra-001", "artifact_role": "device_info", "work_product_id": "", "path": ""},
            },
        ))
        await s.commit()

    # Validation Evaluation creation attaches the private branch to its exact
    # inference. Nothing is public yet, so the branch first raises its own Data
    # ticket to strip answers from the held-out set.
    async with Session() as s:
        tk = (await s.execute(select(Ticket).where(Ticket.id == "infer-001"))).scalar_one()
        run = (await s.execute(select(Run).where(Run.id == "r-lane"))).scalar_one()
        await _spawn_holdout_infer(s, run, tk)
    async with Session() as s:
        prep = (await s.execute(
            select(Ticket).where(Ticket.agent_id == "data", Ticket.lane == "held_out_test")
        )).scalar_one()
        assert prep.payload["dataset"] == "", "the copy job is not a training job"
        assert (await s.execute(
            select(func.count()).select_from(Ticket).where(
                Ticket.agent_id == "inference", Ticket.lane == "held_out_test")
        )).scalar_one() == 0, "no mirror until there is something to infer on"
        prep.status = "succeeded"
        await s.commit()

    # Hop 2: the copy lands, is frozen on the run, and the mirror goes out.
    async with Session() as s:
        prep = (await s.execute(
            select(Ticket).where(Ticket.agent_id == "data", Ticket.lane == "held_out_test")
        )).scalar_one()
        await _advance_measurements(s, prep, -1.0, DataResult(
            status="succeeded", ticket_id=prep.id,
            operation="prepare_holdout_data",
            scoring_public_path="/work/test_public.csv",
            n_rows_in=0, n_rows_out=0, error_message="", notes="",
        ))
    async with Session() as s:
        run = (await s.execute(select(Run).where(Run.id == "r-lane"))).scalar_one()
        assert run.holdout["test_public"] == "/work/test_public.csv"
        mirror = (await s.execute(
            select(Ticket).where(Ticket.agent_id == "inference", Ticket.lane == "held_out_test")
        )).scalar_one()
        # Same model and exact Inference YAML; only the set differs.
        assert mirror.payload["scoring_set"] == "/work/test_public.csv"
        assert mirror.inputs["checkpoint"]["source_ticket_id"] == "train-001"
        assert mirror.inputs["inference_config"] == {
            "artifact_role": "inference_config",
            "source_ticket_id": "infer-001",
        }
        assert mirror.iteration == 1
        mirror.status = "succeeded"
        await s.commit()

    # Attaching the same Evaluation/inference pair twice must not buy test twice.
    async with Session() as s:
        tk = (await s.execute(select(Ticket).where(Ticket.id == "infer-001"))).scalar_one()
        run = (await s.execute(select(Run).where(Run.id == "r-lane"))).scalar_one()
        await _spawn_holdout_infer(s, run, tk)
    async with Session() as s:
        assert (await s.execute(
            select(func.count()).select_from(Ticket).where(
                Ticket.agent_id == "inference", Ticket.lane == "held_out_test")
        )).scalar_one() == 1

    # Hop 3: the mirror is scored by the user's own eval.py.
    async with Session() as s:
        mirror = (await s.execute(
            select(Ticket).where(Ticket.agent_id == "inference", Ticket.lane == "held_out_test")
        )).scalar_one()
        await _advance_measurements(s, mirror, -1.0)
    async with Session() as s:
        ev = (await s.execute(
            select(Ticket).where(Ticket.agent_id == "evaluation", Ticket.lane == "held_out_test")
        )).scalar_one()
        assert ev.payload["scoring_set"] == "/task/test.csv"
        assert ev.payload["evaluation_script"] == "/task/eval.py"
        assert ev.inputs["predictions"]["source_ticket_id"] == mirror.id
        ev.status = "succeeded"
        await s.commit()

    # Hop 4: the number is filed against the iteration it belongs to.
    async with Session() as s:
        ev = (await s.execute(
            select(Ticket).where(Ticket.agent_id == "evaluation", Ticket.lane == "held_out_test")
        )).scalar_one()
        await _advance_measurements(s, ev, 0.55)
    async with Session() as s:
        run = (await s.execute(select(Run).where(Run.id == "r-lane"))).scalar_one()
        rows = (await s.execute(select(ScoreEvent))).scalars().all()
        assert [(r.split, r.score) for r in rows] == [("test", 0.55)]
        # The champion is picked on validation; the reported number is what THAT
        # iteration scored held out, not the best test score seen.
        assert run.champion_test_score == 0.55
        assert run.best_validation_score == 0.62
        assert run.history[0]["test_score"] == 0.55


def test_held_out_evaluation_cannot_update_validation_headline() -> None:
    from zevo.contracts.evaluation import EvaluationResult
    from zevo.engine.run.runner import _apply_run_score_monitor

    run = Run(
        id="r-score-isolation", task_name="t", metric="accuracy",
        best_validation_score=0.4, started_at=datetime.now(timezone.utc),
    )
    held_out = Ticket(
        id="holdout-eval-r-score-isolation-001", run_id=run.id,
        agent_id="evaluation", lane="held_out_test", status="running",
    )
    result = EvaluationResult(
        status="succeeded", ticket_id=held_out.id, metrics_path="/tmp/metrics.json",
        error_message="", notes="",
    )

    _apply_run_score_monitor(run, held_out, result, {"score": 0.9})

    assert run.best_validation_score == 0.4


@pytest.mark.asyncio
async def test_scores_api_reveals_test_series_only_to_ui_or_after_terminal(
    session_factory, tmp_path: Path, monkeypatch,
) -> None:
    """Agents stay blind while the trusted dashboard can render live state."""
    from zevo.api.routers.shared.runs import list_run_scores
    from zevo.api.ui_access import _ui_access_token

    token_file = tmp_path / "ui-token"
    token_file.write_text("dashboard-only", encoding="utf-8")
    monkeypatch.setenv("ZEVO_UI_TOKEN_FILE", str(token_file))
    _ui_access_token.cache_clear()
    agent_request = Request({"type": "http", "headers": []})
    ui_request = Request({
        "type": "http",
        "headers": [(b"x-zevo-ui-access", b"dashboard-only")],
    })

    Session = session_factory
    async with Session() as s:
        s.add(Run(metric="accuracy", id="r-scores", task_name="t", status="running",
                  champion_test_score=0.71, started_at=datetime.now(timezone.utc)))
        s.add(ScoreEvent(run_id="r-scores", iteration=1, split="validation",
                         source="trained", score=0.80, metric_name="accuracy"))
        s.add(ScoreEvent(run_id="r-scores", iteration=1, split="test",
                         source="trained", score=0.71, metric_name="accuracy"))
        await s.commit()

    async with Session() as s:
        agent_view = await list_run_scores("r-scores", agent_request, db=s)
        assert [e.score for e in agent_view.events] == [0.80]
        assert agent_view.test_events == []
        assert agent_view.champion_test_score is None
        assert not any(e.score == 0.71 for e in agent_view.events)

        ui_view = await list_run_scores("r-scores", ui_request, db=s)
        assert [e.score for e in ui_view.test_events] == [0.71]
        assert ui_view.champion_test_score == 0.71

        run = (await s.execute(select(Run).where(Run.id == "r-scores"))).scalar_one()
        run.status = "success"
        await s.commit()
        user_view = await list_run_scores("r-scores", agent_request, db=s)
        assert [e.score for e in user_view.test_events] == [0.71]
        assert user_view.champion_test_score == 0.71
    _ui_access_token.cache_clear()


# ───────────────── which hub split the run actually pulls ────────────────────


def test_hub_id_detection_does_not_claim_local_paths(tmp_path: Path) -> None:
    """Mistaking a file for a hub id sends the data agent to the network for
    something already on disk."""
    from zevo.engine.remote_datasets import looks_like_hub_id

    assert looks_like_hub_id("trl-lib/Capybara")
    assert not looks_like_hub_id("/app/workspace/datasets/medqa-usmle-train/train.csv")
    assert not looks_like_hub_id("data/train.csv")
    assert not looks_like_hub_id("https://example.com/a/b")
    assert not looks_like_hub_id("")


def test_declared_split_reaches_the_data_ticket(tmp_path: Path, monkeypatch) -> None:
    """The catalogue split is resolved before the Data Ticket is stored."""
    import zevo.engine.remote_datasets as rd
    from zevo.api.routers.shared.tickets import _resolve_data_source_selection

    catalogue = tmp_path / "datasets"
    (catalogue / "capy-train").mkdir(parents=True)
    (catalogue / "capy-train" / "source.json").write_text(json.dumps({
        "note": "", "kind": "training",
        "remote": [
            {"role": "train", "kind": "huggingface", "id": "trl-lib/Capybara",
             "url": "", "split": "train[:2000]", "config": "default"},
            {"role": "validation", "kind": "huggingface", "id": "trl-lib/Capybara",
             "url": "", "split": "validation", "config": "default"},
        ],
    }))
    monkeypatch.setattr(rd, "_DATASETS_DIR", catalogue)

    built = _resolve_data_source_selection({
        "dataset": "trl-lib/Capybara", "dataset_split": "", "dataset_config": "",
    })
    # The TRAIN entry wins: this is the training dataset being built, and the
    # repo's validation split is a different row under the same id.
    assert built["dataset_split"] == "train[:2000]"
    assert built["dataset_config"] == "default"


def test_an_undeclared_hub_id_resolves_to_train_before_data_runs(tmp_path: Path, monkeypatch) -> None:
    """A bare hub id has the canonical `train` split; Data does not choose it."""
    import zevo.engine.remote_datasets as rd
    from zevo.api.routers.shared.tickets import _resolve_data_source_selection

    monkeypatch.setattr(rd, "_DATASETS_DIR", tmp_path / "nothing-here")
    resolved = _resolve_data_source_selection({
        "dataset": "someone/unlisted", "dataset_split": "", "dataset_config": "",
    })
    assert resolved["dataset_split"] == "train"
    assert resolved["dataset_config"] == ""


@pytest.mark.asyncio
async def test_a_hub_validation_set_is_fetched_before_the_run_starts(
    task_dir: Path, tmp_path: Path, monkeypatch,
) -> None:
    """A hub id is not a file, and the split is settled before any agent exists
    to download one — so run creation fetches it itself and the rest of the
    system goes on dealing in paths."""
    import zevo.api.routers.shared.runs as runs_mod
    import zevo.engine.remote_datasets as rd
    from zevo.api.routers.shared.runs import _settle_splits
    from zevo.contracts.orchestrator import UserRequest

    fetched: dict = {}

    async def fake_materialize(*, hub_id, split, config, out_dir, limit=0):
        fetched.update(hub_id=hub_id, split=split, config=config, limit=limit)
        path = Path(out_dir) / "validation.csv"
        _write_csv(path, ["question", "gold"],
                   [{"question": f"h{i}", "gold": str(i)} for i in range(30)])
        return str(path), ["question", "gold"], 30, f"fetched 30 rows from {hub_id} ({split})"

    monkeypatch.setattr(rd, "materialize", fake_materialize)
    monkeypatch.setattr(
        runs_mod.settings.__class__, "work_dir_root",
        property(lambda self: str(tmp_path / "work")),
    )

    run = Run(metric="accuracy", id="r-hub", task_name="t", started_at=datetime.now(timezone.utc))
    request = UserRequest(metric="accuracy",
        task_objective="o", metric_direction="max",
        validation_metric_type="builtin", validation_metric="token_f1",
        validation_metric_direction="max",
        training_method="", dataset=str(task_dir / "train.csv"),
        data_query="", base_model="m",
        test_set=str(task_dir / "test.csv"), test_answer_fields=["answer", "gold"],
        validation_set="openai/gsm8k", validation_split="test", validation_config="main",
        # Declared, because a fetched set does not have to share the test set's
        # columns and this one does not: it carries `gold` and no `answer`.
        # Left blank it would inherit both and be refused, which is the point of
        # the check — see the field-existence test above.
        validation_answer_fields=["gold"],
        test_sample_submission="", metric_type="builtin", evaluation_script="",
        constraints=[],
    )
    agent_request, holdout, _ = await _settle_splits(run, request)

    assert fetched["hub_id"] == "openai/gsm8k"
    assert fetched["split"] == "test", "the declared split, not a guessed one"
    assert fetched["config"] == "main"
    assert fetched["limit"] == 0, "the whole split, not a prefix of it"
    # Only the engine keeps the materialized path; the loop is blind to it.
    assert agent_request.validation_set == ""
    assert holdout["validation_set"].endswith("validation.csv")
    assert Path(holdout["validation_set"]).is_file()
    assert holdout["validation_source"] == "huggingface"
    assert "fetched 30 rows" in holdout["note"]
    # Nothing was carved, so the training data is untouched.
    assert agent_request.dataset == str(task_dir / "train.csv")


@pytest.mark.asyncio
async def test_blank_validation_splits_test_without_fetching_hub_training(
    task_dir: Path, tmp_path: Path, monkeypatch,
) -> None:
    """Run creation derives Validation from Test and leaves Hub Training to Data."""
    import zevo.api.routers.shared.runs as runs_mod
    import zevo.engine.remote_datasets as rd
    import zevo.paths as paths_mod
    from zevo.api.routers.shared.runs import _settle_splits
    from zevo.contracts.orchestrator import UserRequest

    fetched: dict = {}

    async def fake_materialize(*, hub_id, split, config, out_dir, limit=0):
        fetched.update(hub_id=hub_id, split=split, config=config, limit=limit)
        path = Path(out_dir) / "validation.csv"
        _write_csv(
            path,
            ["question", "answer"],
            [
                {"question": f"train-{i}", "answer": str(i)}
                for i in range(20)
            ],
        )
        return str(path), ["question", "answer"], 20, "fetched Hub training rows"

    monkeypatch.setattr(rd, "materialize", fake_materialize)
    monkeypatch.setattr(
        runs_mod.settings.__class__, "work_dir_root",
        property(lambda self: str(tmp_path / "work")),
    )
    monkeypatch.setattr(paths_mod, "holdout_root", lambda: tmp_path / "private")

    run = Run(
        metric="accuracy", id="r-hub-train", task_name="t",
        started_at=datetime.now(timezone.utc),
    )
    request = UserRequest(
        metric="accuracy", task_objective="o", metric_direction="max",
        validation_metric_type="builtin", validation_metric="token_f1",
        validation_metric_direction="max",
        training_method="", dataset="owner/training-repo",
        dataset_split="train", dataset_config="default", data_query="",
        base_model="m", test_set=str(task_dir / "test.csv"),
        test_answer_fields=["answer"], validation_set="",
        test_sample_submission=str(task_dir / "sample_submission.csv"),
        metric_type="builtin", evaluation_script="", constraints=[],
    )

    agent_request, holdout, note = await _settle_splits(run, request)

    assert fetched == {}, "Data, not Run creation, owns Training acquisition"
    assert holdout["validation_source"] == "test_split"
    assert holdout["validation_policy"] == "supplied"
    assert holdout["test_set"] != str(task_dir / "test.csv")
    assert agent_request.dataset == "owner/training-repo"
    assert agent_request.dataset_split == "train"
    assert agent_request.dataset_config == "default"
    assert agent_request.validation_set == ""
    assert Path(holdout["validation_set"]).is_file()
    assert "carved 200 validation rows" in note


@pytest.mark.asyncio
async def test_a_hub_validation_set_that_cannot_be_fetched_stops_the_run(
    task_dir: Path, tmp_path: Path, monkeypatch,
) -> None:
    """Falling through to a carve here would tune the run on a set the user did
    not pick, and say nothing about it."""
    import zevo.api.routers.shared.runs as runs_mod
    import zevo.engine.remote_datasets as rd
    from zevo.api.routers.shared.runs import _settle_splits
    from fastapi import HTTPException
    from zevo.contracts.orchestrator import UserRequest

    async def boom(**_kw):
        raise rd.MaterializeError("'x/y' has no 'validation' split. It has: default/train")

    monkeypatch.setattr(rd, "materialize", boom)
    monkeypatch.setattr(
        runs_mod.settings.__class__, "work_dir_root",
        property(lambda self: str(tmp_path / "work")),
    )

    run = Run(metric="accuracy", id="r-hub-bad", task_name="t", started_at=datetime.now(timezone.utc))
    request = UserRequest(metric="accuracy",
        task_objective="o", metric_direction="max",
        validation_metric_type="builtin", validation_metric="token_f1",
        validation_metric_direction="max",
        training_method="", dataset=str(task_dir / "train.csv"),
        data_query="", base_model="m",
        test_set=str(task_dir / "test.csv"), test_answer_fields=["gold"],
        validation_set="x/y", validation_split="validation",
        test_sample_submission="", metric_type="builtin", evaluation_script="", constraints=[],
    )
    with pytest.raises(HTTPException) as e:
        await _settle_splits(run, request)
    assert e.value.status_code == 400
    assert "no 'validation' split" in str(e.value.detail)


@pytest.mark.asyncio
async def test_an_ambiguous_config_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    """A repo can hold many datasets. `cais/mmlu` has 59 subject configs, each
    with a `validation` split — taking the first would have validated the run on
    `abstract_algebra`: a real split, a plausible number, the wrong dataset."""
    from zevo.engine.remote_datasets import MaterializeError, _resolve_split

    class FakeClient:
        async def get(self, url, params=None):
            class R:
                @staticmethod
                def raise_for_status() -> None: ...
                @staticmethod
                def json() -> dict:
                    return {"splits": [
                        {"config": c, "split": s}
                        for c in ("abstract_algebra", "anatomy", "astronomy")
                        for s in ("test", "validation", "dev")
                    ]}
            return R()

    client = FakeClient()
    with pytest.raises(MaterializeError, match="name which one in `config`"):
        await _resolve_split(client, "cais/mmlu", "validation", "")

    # Named, so there is nothing to guess.
    assert await _resolve_split(client, "cais/mmlu", "validation", "anatomy") \
        == ("anatomy", "validation")

    # One config with the split is unambiguous even without naming it.
    class SingleConfig(FakeClient):
        async def get(self, url, params=None):
            class R:
                @staticmethod
                def raise_for_status() -> None: ...
                @staticmethod
                def json() -> dict:
                    return {"splits": [{"config": "default", "split": s}
                                       for s in ("train", "validation")]}
            return R()

    assert await _resolve_split(SingleConfig(), "someone/simple", "", "") \
        == ("default", "validation")


def test_the_carved_fraction_holds_at_every_size(tmp_path: Path) -> None:
    """Every eligible Test gives exactly 20%; smaller Tests require Validation."""
    from zevo.engine.method.validation_split import (
        FRACTION, MIN_VALIDATION_ROWS, _n_validation,
    )

    for total in (1_000, 20_000, 200_000):
        assert _n_validation(total) == round(total * FRACTION)
    assert _n_validation(1_000) == MIN_VALIDATION_ROWS
    for total in (1, 2, 999):
        with pytest.raises(SplitError, match="Validation|validation"):
            _n_validation(total)


def test_the_baseline_phase_is_iteration_zero() -> None:
    """The baseline probe IS iteration 0, so the phase that produces it cannot
    also be called iteration 1. `iterations_completed` counts Validation-
    measured trained candidates and is 0 both before the baseline and during
    the first training loop, so it cannot tell them apart on its own."""
    from zevo.engine.run.runner import _current_iteration

    class R:
        def __init__(self, done, history):
            self.iterations_completed, self.history = done, history

    assert _current_iteration(R(0, [])) == 0, "nothing measured yet"
    assert _current_iteration(R(0, [{"iteration": 1, "score": 0.4}])) == 0, \
        "a training entry without a baseline still leaves the baseline undone"
    baseline = [{"iteration": 0, "source": "baseline", "score": 0.3}]
    assert _current_iteration(R(0, baseline)) == 1, "recording it advances to 1"
    assert _current_iteration(R(2, baseline)) == 3


def test_the_complete_stored_data_payload_is_the_only_runner_authority(tmp_path: Path, monkeypatch) -> None:
    """Runner executes stored Training facts and exposes no Validation facts."""
    import zevo.engine.remote_datasets as rd
    from zevo.engine.run.runner import _build_data_input

    catalogue = tmp_path / "datasets"
    (catalogue / "capy-train").mkdir(parents=True)
    (catalogue / "capy-train" / "source.json").write_text(json.dumps({
        "note": "", "kind": "training",
        "remote": [{"role": "train", "kind": "huggingface", "id": "trl-lib/Capybara",
                    "url": "", "split": "train", "config": "default"}],
    }))
    monkeypatch.setattr(rd, "_DATASETS_DIR", catalogue)

    run = Run(metric="accuracy", id="r", task_name="t", started_at=datetime.now(timezone.utc),
              holdout={
                  "dataset_split": "wrong", "dataset_config": "wrong",
                  "validation_set": "/validation.jsonl",
                  "validation_answer_fields": ["answer"],
              })
    ticket = Ticket(id="data-001", run_id="r", agent_id="data", payload={})
    payload = {"operation": "prepare_run_data",
               "dataset_source": "trl-lib/Capybara",
               "dataset": "trl-lib/Capybara", "dataset_split": "train[:500]",
               "dataset_config": "other",
                   "data_query": "", "training_method": "full_sft",
                   "data_intent_signature": "a" * 64}

    built = _build_data_input(ticket, payload, {}, str(tmp_path), run)
    assert built.dataset_split == "train[:500]"
    assert built.dataset_config == "other"
    assert built.expected_source_identity == "trl-lib/Capybara"
    assert built.scoring_set == ""
    assert built.answer_fields == []
    assert built.evaluation_script == ""
    assert built.sample_submission == ""
    assert "--scoring-source" not in built.artifacts_validation_command
    assert "validate-training-data" in built.artifacts_validation_command


def test_local_data_input_supplies_the_exact_source_fingerprint(tmp_path: Path) -> None:
    """A local Data Agent copies a digest; it never chooses its serialization."""
    from zevo.contracts.configuration import file_sha256
    from zevo.engine.run.runner import _build_data_input

    source = tmp_path / "source with spaces.json"
    source.write_text('[{"instruction":"q","response":"a"}]', encoding="utf-8")
    run = Run(
        metric="accuracy", id="r", task_name="t",
        started_at=datetime.now(timezone.utc), holdout={},
    )
    ticket = Ticket(id="data-001", run_id="r", agent_id="data", payload={})
    payload = {
        "operation": "prepare_run_data",
        "dataset_source": str(source),
        "dataset": str(source),
        "data_query": "",
        "training_method": "full_sft",
        "data_intent_signature": "a" * 64,
    }

    built = _build_data_input(ticket, payload, {}, str(tmp_path), run)
    assert built.expected_source_identity == str(source)
    assert built.expected_source_fingerprint == file_sha256(source)
    assert "--source-file" in built.data_recipe_validation_command
    assert "--source-identity" in built.data_recipe_validation_command
    assert str(source) in built.data_recipe_validation_command
    fingerprint_schema = built.data_recipe_schema["properties"]["source_fingerprint"]
    assert fingerprint_schema["pattern"] == r"^[0-9a-f]{64}$"


def test_pipeline_data_payload_is_stamped_without_the_validation_contract() -> None:
    from zevo.api.routers.shared.tickets import _stamp_pipeline_payload

    run = Run(
        metric="benchmark_average", validation_metric="token_f1",
        validation_metric_direction="max",
        id="r", task_name="t", mode="full_pipeline",
        started_at=datetime.now(timezone.utc),
        holdout={
            "validation_set": "/task/validation.json",
            "validation_answer_fields": ["response"],
            "validation_metric_type": "custom",
            "validation_evaluation_script": "/task/val_eval.py",
            "validation_sample_submission": "/task/val_sample.csv",
        },
    )
    stamped = _stamp_pipeline_payload(
        run=run,
        agent_id="data",
        payload={
            "operation": "prepare_run_data",
            "dataset": "/task/train.json",
            "training_method": "full_sft",
        },
    )
    for field in (
        "scoring_set", "answer_fields", "evaluation_script", "metric",
        "sample_submission",
    ):
        assert field not in stamped


def test_pipeline_data_payload_does_not_receive_unsettled_validation() -> None:
    from zevo.api.routers.shared.tickets import _stamp_pipeline_payload

    run = Run(
        metric="benchmark_average", validation_metric="token_f1",
        validation_metric_direction="max",
        id="r", task_name="t", mode="full_pipeline",
        started_at=datetime.now(timezone.utc),
        holdout={
            "validation_set": "",
            "validation_answer_fields": [],
            "test_answer_fields": ["response"],
            "validation_source": "",
            "validation_policy": "supplied",
            "validation_fraction": 0.0,
            "validation_needs_metric_binding": False,
        },
    )
    stamped = _stamp_pipeline_payload(
        run=run,
        agent_id="data",
        payload={
            "operation": "prepare_run_data",
            "dataset": "",
            "data_query": "acquire suitable public instruction data",
            "training_method": "full_sft",
        },
    )
    assert "scoring_set" not in stamped
    assert "answer_fields" not in stamped


@pytest.mark.asyncio
async def test_prepared_data_preserves_the_test_derived_validation_contract(
    session_factory, tmp_path: Path,
) -> None:
    from zevo.contracts.data import DataResult, InferenceDataProfile
    from zevo.engine.run.runner import _record_prepared_scoring_data

    training = tmp_path / "dataset.jsonl"
    validation_dataset = tmp_path / "validation_dataset.jsonl"
    training.write_text(
        "".join(json.dumps({"messages": [i]}) + "\n" for i in range(17)),
        encoding="utf-8",
    )
    validation_dataset.write_text(
        "".join(json.dumps({"messages": [i]}) + "\n" for i in range(3)),
        encoding="utf-8",
    )
    validation_source = tmp_path / "validation_source.csv"
    validation_public = tmp_path / "validation_questions.csv"
    _write_csv(
        validation_source,
        ["id", "instruction", "answer"],
        [
            {"id": str(i), "instruction": f"q{i}", "answer": f"a{i}"}
            for i in range(3)
        ],
    )
    _write_csv(
        validation_public,
        ["id", "instruction"],
        [{"id": str(i), "instruction": f"q{i}"} for i in range(3)],
    )
    sample = tmp_path / "sample.csv"
    _write_csv(sample, ["id", "prediction"], [])
    evaluator = tmp_path / "eval.py"
    evaluator.write_text("# scorer\n", encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(InferenceDataProfile(
        n_rows=3,
        task_shape="single-turn text input",
        record_fields={"id": "string", "instruction": "string"},
        input_fields=["instruction"],
        answer_fields_removed=["answer"],
        submission_columns=["id", "prediction"],
        prediction_encoding="one prediction string per row",
        stable_ids_present=True,
    ).model_dump_json(), encoding="utf-8")

    async with session_factory() as session:
        run = Run(
            metric="accuracy", id="r-freeze-derived", task_name="t",
            status="running", mode="full_pipeline",
            started_at=datetime.now(timezone.utc),
            supervisor_ticket_id="orchestrate-r-freeze-001",
            decision_pins={"data_query": "acquire matching data"},
            holdout={
                "test_set": "/task/test.csv",
                "validation_set": str(validation_source),
                "validation_answer_fields": ["answer"],
                "validation_source": "test_split",
                "validation_policy": "supplied",
                "validation_fraction": 0.0,
                "validation_rows": 3,
                "validation_sample_submission": str(sample),
                "validation_public": "",
                "validation_needs_metric_binding": True,
            },
        )
        data_ticket = Ticket(
            id="data-r-freeze-001", run_id=run.id, agent_id="data",
            lane="optimization", iteration=0, status="succeeded", payload={},
        )
        supervisor = Ticket(
            id=run.supervisor_ticket_id, run_id=run.id, agent_id="orchestrator",
            lane="optimization", iteration=0, status="running",
            payload={"user_request": {
                "dataset": "", "dataset_split": "", "dataset_config": "",
            }},
        )
        session.add_all([run, data_ticket, supervisor])
        await session.commit()

        prepared_result = DataResult(
            status="succeeded", ticket_id=data_ticket.id,
            error_message="", notes="prepared derived Validation",
            operation="prepare_run_data",
            training_dataset_path=str(training),
            data_recipe_path=str(tmp_path / "recipe.json"),
            n_rows_in=20, n_rows_out=17,
        ).model_copy(update={
            "validation_source_path": str(validation_source),
            "validation_answer_fields": ["answer"],
            "validation_dataset_path": str(validation_source),
            "scoring_public_path": str(validation_public),
            "inference_data_profile_path": str(profile_path),
            "sample_submission_path": str(sample),
        })
        await _record_prepared_scoring_data(
            session, run, data_ticket, prepared_result,
        )

        assert run.holdout["validation_set"] == str(validation_source)
        assert run.holdout["validation_source"] == "test_split"
        assert run.holdout["validation_policy"] == "supplied"
        assert run.holdout["validation_rows"] == 3
        assert run.holdout["test_set"] == "/task/test.csv"
        assert run.decision_pins == {"data_query": "acquire matching data"}
        assert (supervisor.payload["user_request"])["dataset"] == ""
        from zevo.api.routers.shared.tickets import _stamp_pipeline_payload
        revision = _stamp_pipeline_payload(
            run=run,
            agent_id="data",
            payload={
                "operation": "prepare_run_data",
                "dataset": str(training),
                "data_query": "acquire matching data",
                "training_method": "full_sft",
                "recipe_intent": {"direction": "filter low-quality rows"},
            },
        )
        assert revision["dataset"] == str(training)
        assert revision["validation_policy"] == "supplied"
        assert "scoring_set" not in revision


@pytest.mark.parametrize(
    ("agent_id", "payload"),
    [
        ("infrastructure", {"operation": "provision", "purpose": "inference"}),
        ("registry", {}),
        ("orchestrator", {"operation": "advance"}),
    ],
)
def test_pipeline_payload_stamping_does_not_treat_other_agents_as_evaluation(
    agent_id: str, payload: dict[str, str],
) -> None:
    """Only Evaluation owns scoring fields; unrelated Agent schemas stay isolated."""
    from zevo.api.routers.shared.tickets import _stamp_pipeline_payload

    run = Run(
        metric="benchmark_average", validation_metric="token_f1",
        validation_metric_direction="max",
        id="r", task_name="t", mode="full_pipeline",
        started_at=datetime.now(timezone.utc),
        holdout={
            "validation_set": "/task/validation.json",
            "validation_answer_fields": ["response"],
            "validation_metric_type": "custom",
            "validation_evaluation_script": "/task/val_eval.py",
            "validation_sample_submission": "/task/val_sample.csv",
        },
    )

    assert _stamp_pipeline_payload(
        run=run, agent_id=agent_id, payload=payload,
    ) == payload


def test_pipeline_evaluation_payload_is_stamped_with_scoring_contract() -> None:
    from zevo.api.routers.shared.tickets import _stamp_pipeline_payload

    run = Run(
        metric="benchmark_average", validation_metric="token_f1",
        validation_metric_direction="max",
        id="r", task_name="t", mode="full_pipeline",
        started_at=datetime.now(timezone.utc),
        holdout={
            "validation_set": "/task/validation.json",
            "validation_answer_fields": ["response"],
            "validation_metric_type": "custom",
            "validation_evaluation_script": "/task/val_eval.py",
            "validation_sample_submission": "/task/val_sample.csv",
        },
    )

    assert _stamp_pipeline_payload(
        run=run, agent_id="evaluation", payload={},
    ) == {
        "scoring_set": "/task/validation.json",
        "evaluation_script": "/task/val_eval.py",
        "evaluator_sha256": "",
        "answer_fields": ["response"],
        "sample_submission": "/task/val_sample.csv",
        "metric": "token_f1",
    }


@pytest.mark.asyncio
async def test_pipeline_ticket_uses_only_run_owned_customization(session_factory) -> None:
    from fastapi import HTTPException
    from zevo.api.routers.shared.tickets import CreateTicketBody, create_ticket
    from zevo.contracts.customizations import AgentCustomization

    async with session_factory() as s:
        s.add(Run(
            id="r-customization", task_name="t", status="running",
            mode="customized_pipeline", metric="accuracy",
            holdout={
                "validation_set": "/task/validation.jsonl",
                "validation_answer_fields": ["answer"],
            },
            customizations={
                "agents": {
                    "data": {
                        "instructions": "Keep all valid rows",
                        "input_paths": [], "output_dir": "",
                        "parameters": {"target_size": 123},
                        "enforcement": "strict",
                    },
                },
                "note": "",
            },
            history=[], started_at=datetime.now(timezone.utc),
        ))
        await s.commit()
        created = await create_ticket(CreateTicketBody(
            agent_id="data", run_id="r-customization", iteration=0,
            payload={
                "operation": "prepare_run_data", "dataset": "/task/train.jsonl",
                "training_method": "full_sft",
            },
        ), db=s)
        assert created.payload["configuration_pins"] == {"target_size": 123}
        assert created.customization["instructions"] == "Keep all valid rows"
        assert created.customization["parameters"] == {}
        stored = (await s.execute(
            select(Ticket).where(Ticket.id == created.id)
        )).scalar_one()
        stored.status = "failed"
        await s.commit()

    async with session_factory() as s:
        with pytest.raises(HTTPException, match="declared once"):
            await create_ticket(CreateTicketBody(
                agent_id="data", run_id="r-customization", iteration=0,
                payload={
                    "operation": "prepare_run_data", "dataset": "/task/train.jsonl",
                    "training_method": "full_sft",
                },
                customization=AgentCustomization(
                    parameters={"target_size": 456}, enforcement="strict",
                ),
            ), db=s)

def test_each_lane_uses_its_own_metric_evaluator_and_sample(tmp_path: Path) -> None:
    """Validation and Test remain reproducible without sharing semantics."""
    from zevo.engine.run.runner import _build_evaluation_input

    run = Run(
        metric="benchmark_average", metric_direction="max",
        validation_metric="token_f1", validation_metric_direction="max",
        id="r", task_name="t", started_at=datetime.now(timezone.utc), holdout={
        "test_set": "/task/test.csv",
        "test_answer_fields": ["test_gold"],
        "validation_set": "/splits/validation.csv",
        "validation_answer_fields": ["validation_gold"],
        "test_evaluation_script": "/task/test_eval.py",
        "test_sample_submission": "/task/test_sample.csv",
        "validation_evaluation_script": "/task/val_eval.py",
        "validation_sample_submission": "/task/val_sample.csv",
    })
    inputs = {"predictions": {"path": "/tmp/predictions.csv"}}

    loop = Ticket(id="eval-001", run_id="r", agent_id="evaluation", payload={}, lane="optimization")
    held = Ticket(id="holdout-eval-001", run_id="r", agent_id="evaluation",
                  payload={}, lane="held_out_test")

    validation_payload = {
        "metric": "token_f1",
        "scoring_set": "/splits/validation.csv",
        "evaluation_script": "/task/val_eval.py",
        "sample_submission": "/task/val_sample.csv",
        "answer_fields": ["validation_gold"],
    }
    test_payload = {
        "metric": "benchmark_average",
        "scoring_set": "/task/test.csv",
        "evaluation_script": "/task/test_eval.py",
        "sample_submission": "/task/test_sample.csv",
        "answer_fields": ["test_gold"],
    }
    assert _build_evaluation_input(loop, validation_payload, inputs, "/w", run).evaluation_script \
        == "/task/val_eval.py"
    assert _build_evaluation_input(held, test_payload, inputs, "/w", run).evaluation_script \
        == "/task/test_eval.py"
    assert _build_evaluation_input(loop, validation_payload, inputs, "/w", run).metric \
        == "token_f1"
    assert _build_evaluation_input(held, test_payload, inputs, "/w", run).metric \
        == "benchmark_average"
    assert _build_evaluation_input(loop, validation_payload, inputs, "/w", run).answer_fields \
        == ["validation_gold"]
    assert _build_evaluation_input(held, test_payload, inputs, "/w", run).answer_fields \
        == ["test_gold"]


def test_optimization_data_is_blind_while_heldout_data_gets_its_contract() -> None:
    """Optimization Data sees no scoring assets; private Test stripping does."""
    from zevo.engine.run.runner import _build_data_input, _validation_files

    task_files = {
        "test_set": "/task/test.csv", "test_answer_fields": ["gold"],
        "test_evaluation_script": "/task/test_eval.py",
        "test_sample_submission": "/task/sample_submission.csv",
        "validation_set": "/splits/validation.csv",
        "validation_answer_fields": ["gold"],
        "validation_evaluation_script": "/task/val_eval.py",
        "validation_sample_submission": "/task/val_sample.csv",
    }
    loop = Ticket(id="data-001", run_id="r", agent_id="data", payload={}, lane="optimization")
    held = Ticket(id="holdout-data-001", run_id="r", agent_id="data", payload={}, lane="held_out_test")

    # Same shape as the test set: the task's own pair is right.
    run = Run(metric="benchmark_average", validation_metric="token_f1",
              validation_metric_direction="max",
              id="r", task_name="t", started_at=datetime.now(timezone.utc),
              holdout=dict(task_files))
    assert _validation_files(run.holdout) == ("/task/val_eval.py", "/task/val_sample.csv")

    # A different shape remains engine-owned and is not exposed to Data.
    run.holdout = {**task_files, "validation_needs_metric_binding": True,
                   "validation_answer_fields": ["response"]}
    run.holdout["validation_sample_submission"] = ""
    assert _validation_files(run.holdout) == ("/task/val_eval.py", "")
    data_payload = {
        "operation": "prepare_run_data",
            "dataset": "/training.jsonl", "training_method": "full_sft",
            "data_intent_signature": "a" * 64,
    }
    built = _build_data_input(loop, data_payload, {}, "/work", run)
    assert built.scoring_set == ""
    assert built.answer_fields == []
    assert built.evaluation_script == "" and built.sample_submission == ""

    # The held-out lane is never in this position: the test set arrives with all
    # four of its files and none of them is the agent's to invent.
    held_built = _build_data_input(held, {
        "operation": "prepare_holdout_data",
        "scoring_set": "/task/test.csv", "answer_fields": ["gold"],
        "metric": "benchmark_average",
        "evaluation_script": "/task/test_eval.py",
        "sample_submission": "/task/sample_submission.csv",
    }, {}, "/work", run)
    assert held_built.scoring_set == "/task/test.csv"
    assert held_built.evaluation_script == "/task/test_eval.py"
    assert held_built.sample_submission == "/task/sample_submission.csv"


def test_test_answer_fields_define_the_derived_validation_binding(
    tmp_path: Path,
) -> None:
    """Derived Validation inherits Test's fields, even if stale Validation fields arrive."""
    _write_csv(tmp_path / "test.csv", ["question", "gold"],
               [{"question": f"t{i}", "gold": str(i)} for i in range(1_000)])
    _write_csv(tmp_path / "train.csv", ["prompt", "completion"],
               [{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(40)])

    out = resolve(
        validation_set="", validation_answer_fields=["completion"],
        dataset=str(tmp_path / "train.csv"), test_set=str(tmp_path / "test.csv"),
        test_answer_fields=["gold"],
        out_dir=str(tmp_path / "splits"),
    )
    assert out.validation_answer_fields == ["gold"]
    assert not out.needs_metric_binding


def test_a_json_dataset_is_carved_as_json(tmp_path: Path) -> None:
    """The carve must hand back the format it was given.

    A conversation dataset keeps every turn of one exchange under one key, so
    its records are nested. Read as CSV, a JSON array parses as one column named
    `[`; written back as CSV, the nested value becomes a quoted string and the
    task's own scorer no longer recognizes its input. Neither failure raises.
    """
    src = tmp_path / "train.json"
    src.write_text(json.dumps([
        {"id": f"t{i}", "instruction": {"1": f"q{i}"}, "response": {"1": f"a{i}", "2": f"b{i}"}}
        for i in range(40)
    ]), encoding="utf-8")
    test = tmp_path / "test.json"
    test.write_text(json.dumps([
        {
            "id": f"e{i}", "instruction": {"1": f"q{i}"},
            "response": {"1": f"a{i}", "2": f"b{i}"},
        }
        for i in range(1_000)
    ]), encoding="utf-8")

    out = carve(
        dataset=str(src), test_set=str(test), test_answer_fields=["response"],
        declared_columns=["response"],
        out_dir=str(tmp_path / "splits"),
    )
    assert out.validation_set.endswith(".json")
    assert out.test_set.endswith(".json")

    val = json.loads(Path(out.validation_set).read_text())
    rest = json.loads(Path(out.test_set).read_text())
    assert len(val) == 200 and len(val) + len(rest) == 1_000
    # The nested object is still an object, not a string of one.
    assert isinstance(val[0]["response"], dict)
    assert set(val[0]["response"]) == {"1", "2"}
    assert not {r["id"] for r in val} & {r["id"] for r in rest}
    # `response` is present, so the task's scorer reads this shape fine.
    assert out.validation_answer_fields == ["response"]
    assert not out.needs_metric_binding


@pytest.mark.asyncio
async def test_a_round_is_over_only_when_both_lanes_have_reported(session_factory) -> None:
    """The next iteration waits for the test measurement, not just validation.

    Starting the next round on the validation score alone leaves the held-out
    lane measuring a model the run has already moved past: iteration 3's test
    point arrives while iteration 4 trains, or never, if the run ends first.
    The two series then cannot be read against each other, which is the only
    reason both exist.

    A lane that FAILED must not hold the loop forever, though — a gap in one
    series beats a run that cannot advance.
    """
    from zevo.engine.run.runner import _holdout_settled

    async with session_factory() as s:
        s.add(Run(metric="accuracy", id="r", task_name="t", started_at=datetime.now(timezone.utc)))
        await s.commit()

        assert await _holdout_settled(s, "r"), "no lane in flight is settled"

        s.add(Ticket(id="holdout-infer-1", run_id="r", agent_id="inference",
                     status="running", lane="held_out_test",
                     payload={}))
        await s.commit()
        assert not await _holdout_settled(s, "r")

        # Terminal states all count as settled, including the unhappy ones.
        for status in ("succeeded", "failed", "cancelled", "degraded"):
            t = await s.get(Ticket, "holdout-infer-1")
            t.status = status
            await s.commit()
            assert await _holdout_settled(s, "r"), f"{status} should not block"

        # A ticket of the LOOP's own never blocks: the barrier is about the
        # mirror lane, and the loop cannot wait on itself.
        s.add(Ticket(id="infer-1", run_id="r", agent_id="inference", status="running", lane="optimization",
                     payload={}))
        await s.commit()
        assert await _holdout_settled(s, "r")


@pytest.mark.asyncio
async def test_the_deferred_wake_is_handed_back_by_whichever_mirror_ticket_settles(
    session_factory,
) -> None:
    """The wake is released by the lane STOPPING, not by the lane succeeding.

    Releasing only on the mirror's final evaluation assumed the lane always
    reaches one. It does not: the copy job that strips the test set can fail,
    and then no eval is ever raised, no release ever fires, and the run sits on
    a wake nothing will ever issue. Any mirror completion that leaves the lane
    settled has to hand it back.

    And a lane that settles with nothing waiting must issue NOTHING — during
    iteration 0 the loop's own evaluation has usually woken the supervisor
    already, and a second wake would replan the round.
    """
    from zevo.engine.run.runner import _release_deferred_wake

    async with session_factory() as s:
        # A supervisor the wake can actually reach: without one
        # `_maybe_wake_supervisor` returns early and the assertions below would
        # pass whatever the release did.
        s.add(Run(metric="accuracy", id="r", task_name="t", status="running",
                  supervisor_ticket_id="orchestrate-r-001",
                  started_at=datetime.now(timezone.utc)))
        s.add(Ticket(id="orchestrate-r-001", run_id="r", agent_id="orchestrator",
                     status="succeeded", lane="optimization",
                     payload={}))
        s.add(Ticket(id="eval-1", run_id="r", agent_id="evaluation", status="succeeded", lane="optimization", payload={}))
        await s.commit()

        # Nothing deferred: no supervisor ticket is created.
        await _release_deferred_wake(s, "r")
        assert (await s.get(Ticket, "eval-1")).supervisor_wake_deferred is False
        assert (await s.get(Ticket, "orchestrate-r-001")).status == "succeeded", (
            "a lane that settled with nothing waiting re-armed the supervisor"
        )

        # Now one IS deferred, and the mirror's copy job failed — the case that
        # never reaches an eval.
        ev = await s.get(Ticket, "eval-1")
        ev.supervisor_wake_deferred = True
        s.add(Ticket(id="holdout-data-1", run_id="r", agent_id="data", status="failed", lane="held_out_test", payload={}))
        await s.commit()

        await _release_deferred_wake(s, "r")
        sup = await s.get(Ticket, "orchestrate-r-001")
        await s.refresh(sup)
        assert sup.status == "queued", "the held wake was never handed back"
        assert (sup.payload or {}).get("trigger", {}).get("status") == "succeeded", (
            "the supervisor must be told its OWN child finished, never the mirror's"
        )
        assert (await s.get(Ticket, "eval-1")).supervisor_wake_deferred is False, (
            "the flag must clear, or the next mirror completion releases it again"
        )


def test_every_holdout_key_the_runner_reads_is_one_the_run_actually_writes() -> None:
    """`run.holdout` is a JSON blob, so a misspelled key is not an error.

    It reads as absent. `holdout.get("test_set_public")` returned "" forever
    because the key is called `test_public`, and nothing anywhere complained:
    the held-out inference payload silently fell through to a default, and the
    contamination guard silently stopped watching half the test lane. Both bugs
    were invisible in a passing test suite and in a running system.

    So compare the two sides directly. This is a spelling check, not a
    behaviour check, which is exactly what a schemaless dict needs.
    """
    import re
    from pathlib import Path

    # Split settlement (the literal that seeds the dict) lives engine-side so an
    # auto Run can settle after creation; the auto path adds its provenance keys.
    settlement = Path("src/zevo/engine/run/split_settlement.py").read_text()
    written = set(re.findall(
        r'^\s+"(\w+)":',
        # The literal that seeds the dict, up to its closing brace.
        re.search(r"holdout = \{(.*?)\n    \}", settlement, re.S).group(1),
        re.M,
    ))
    for writer in (
        "src/zevo/api/routers/shared/runs.py",
        "src/zevo/engine/run/scoping.py",
    ):
        written |= set(re.findall(r'holdout\["(\w+)"\]\s*=', Path(writer).read_text()))
    # Written by the runner once Data has produced the once-per-Run package.
    written |= {"validation_public", "test_public", "validation_dataset"}

    runner = Path("src/zevo/engine/run/runner.py").read_text()
    read = set(re.findall(r'holdout\.get\("(\w+)"', runner))
    read |= set(re.findall(r'\(getattr\(run, "holdout", None\) or \{\}\)\.get\("(\w+)"', runner))
    # Keys named in a tuple the guard iterates, e.g. `for k in ("a", "b")`.
    for grp in re.findall(r'for k in \(([^)]*)\)', runner):
        read |= set(re.findall(r'"(\w+)"', grp))

    stray = sorted(read - written)
    assert not stray, (
        f"runner reads holdout keys nothing writes: {stray}. "
        f"Written: {sorted(written)}"
    )


def test_a_named_validation_set_is_checked_against_its_declared_answer_fields(
    task_dir: Path, tmp_path: Path,
) -> None:
    """Refuse at run creation what would otherwise fail an hour in.

    `validation_answer_fields` falls back to `test_answer_fields`, which is
    right when the two sets share a shape and silently wrong when they do not.
    A conversation-shaped validation set inheriting IFEval's
    `instruction_id_list` gets the data agent a file with no such column, and
    it fails — correctly, but after infra has provisioned and a GPU is
    billing. That run exists; it is data-021.

    The set is settled before any agent starts, so the check costs one file
    read here and saves the whole provisioning round trip.
    """
    val = tmp_path / "val.csv"
    _write_csv(val, ["question", "response"],
               [{"question": f"q{i}", "response": f"a{i}"} for i in range(20)])

    # Declared fields that ARE there: fine.
    out = resolve(
        validation_set=str(val), validation_answer_fields=["response"],
        dataset=str(task_dir / "train.csv"), test_set=str(task_dir / "test.csv"),
        test_answer_fields=["answer", "gold"], out_dir=str(tmp_path / "a"),
    )
    assert out.source == "user" and out.validation_answer_fields == ["response"]

    # Declaring none is refused; Test fields never fill the Validation contract.
    with pytest.raises(SplitError) as e:
        resolve(
            validation_set=str(val), validation_answer_fields=[],
            dataset=str(task_dir / "train.csv"), test_set=str(task_dir / "test.csv"),
            test_answer_fields=["answer", "gold"], out_dir=str(tmp_path / "b"),
        )
    assert "requires its own validation_answer_fields" in str(e.value)
    assert "not a fallback" in str(e.value)

    # A typo in the declared fields is the same failure and gets the same answer.
    with pytest.raises(SplitError):
        resolve(
            validation_set=str(val), validation_answer_fields=["respones"],
            dataset=str(task_dir / "train.csv"), test_set=str(task_dir / "test.csv"),
            test_answer_fields=[], out_dir=str(tmp_path / "c"),
        )


def test_a_csv_row_count_is_rows_not_lines(tmp_path: Path) -> None:
    """Free text contains newlines; a line count is then not a row count.

    `train_if.csv` — 13,839 prompt/answer pairs — profiled as 223,444 rows,
    sixteen times over, because every newline inside a quoted answer counted as
    a record. The wrong number reached the dataset card AND `_recommend_split`,
    which sized a train/val split for a file that does not exist.
    """
    from zevo.engine.dataset_profiler import _count_csv_rows

    p = tmp_path / "multiline.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt", "response"])
        for i in range(5):
            w.writerow([f"q{i}", f"line one\nline two\nline three of answer {i}"])

    assert sum(1 for _ in p.open(encoding="utf-8")) == 16, "5 rows spanning 16 lines"
    assert _count_csv_rows(p) == 5

    # A cell larger than csv's default field limit must not raise.
    big = tmp_path / "big.csv"
    with big.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["a"]); w.writerow(["x" * 200_000])
    assert _count_csv_rows(big) == 1


def test_the_dataset_card_profiles_the_training_file(tmp_path: Path) -> None:
    """Being training data beats being a preferred FORMAT.

    The picker used to loop suffix-first (`.jsonl`, `.csv`, `.json`) and take
    the training file within each. A bundle whose training file is
    `train/train.json` therefore got profiled as whatever `.csv` sat beside it,
    because `.json` is last — so the card described a file the run does not
    read. And among two training files, the one actually called `train` is the
    training file; a `train_if.csv` variant beside it is not.
    """
    from zevo.api.routers.ui.files import _pick_primary_data_file

    d = tmp_path / "bundle"
    for rel in ("train/train.json", "train/train_if.csv",
                "test/test.json", "validation/val.json",
                "test/test_sample_submission.csv"):
        f = d / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("[]" if f.suffix == ".json" else "a,b\n1,2\n")
    assert _pick_primary_data_file(d) == d / "train" / "train.json"

    # With no training file at all, format preference decides — and an empty
    # sample_submission must not win on alphabetical order alone.
    bare = tmp_path / "bare"
    (bare / "a").mkdir(parents=True)
    (bare / "a" / "sample_submission.csv").write_text("a,b\n")
    (bare / "a" / "rows.jsonl").write_text('{"a": 1}\n')
    assert _pick_primary_data_file(bare) == bare / "a" / "rows.jsonl"

    assert _pick_primary_data_file(tmp_path / "nope") is None


@pytest.mark.asyncio
async def test_the_test_score_reaches_the_journal_whichever_half_lands_first(
    session_factory,
) -> None:
    """Validation materialization joins a held-out score that landed first."""
    from zevo.engine.run.runner import _record_validation_score

    async with session_factory() as s:
        run = Run(metric="accuracy", id="r", task_name="t", started_at=datetime.now(timezone.utc),
                  history=[], champion_test_score=None)
        s.add(run)
        s.add(Ticket(
            id="infer-r-001", run_id="r", agent_id="inference",
            lane="optimization", status="succeeded", iteration=1,
            payload={"base_model": "base/model"}, inputs={},
        ))
        evaluation = Ticket(
            id="eval-r-001", run_id="r", agent_id="evaluation",
            lane="optimization", status="succeeded", iteration=1, payload={},
            inputs={"predictions": {"source_ticket_id": "infer-r-001"}},
        )
        s.add(evaluation)
        s.add(ScoreEvent(run_id="r", iteration=1, split="test", source="trained",
                         score=0.2100, metric_name="accuracy"))
        await s.commit()

        await _record_validation_score(s, run, evaluation, 0.3000)
        await s.refresh(run)
        assert run.history == [{
            "iteration": 1,
            "source": "trained",
            "method_ids": [],
            "training_method": "",
            "base_model": "base/model",
            "score": 0.3000,
            "action": "",
            "result": "",
            "analysis": "",
            "next": "",
            "test_score": 0.2100,
        }]
        assert run.champion_test_score == 0.2100


def test_evaluation_owned_history_fact_preserves_decision_prose_and_test_score() -> None:
    from zevo.engine.observe.run_metrics import upsert_history_fact

    prior = [{
        "iteration": 0,
        "source": "baseline",
        "score": -1.0,
        "test_score": 0.0182,
        "action": "Measure the untouched model.",
        "analysis": "The baseline is weak.",
    }]
    got = upsert_history_fact(
        prior,
        iteration=0,
        source="baseline",
        score=0.024,
        base_model="Qwen/Qwen3-0.6B-Base",
    )
    assert len(got) == 1
    assert got[0]["score"] == 0.024
    assert got[0]["test_score"] == 0.0182
    assert got[0]["analysis"] == "The baseline is weak."
    assert got[0]["base_model"] == "Qwen/Qwen3-0.6B-Base"


def test_evaluation_owned_history_fact_preserves_generation_termination() -> None:
    from zevo.engine.observe.run_metrics import upsert_history_fact

    termination = {
        "total_requests": 200,
        "stopped_requests": 175,
        "length_limited_requests": 25,
        "finish_reason_counts": {"stop": 175, "length": 25},
        "stop_reason_counts": {"100265": 175, "null": 25},
        "max_generated_tokens": 1024,
    }
    got = upsert_history_fact(
        [], iteration=1, source="trained", score=0.2,
        generation_termination=termination,
    )
    assert got[0]["generation_termination"] == termination

    refreshed = upsert_history_fact(
        got, iteration=1, source="trained", score=0.3,
    )
    assert refreshed[0]["generation_termination"] == termination

def test_the_trainer_evals_on_the_run_s_validation_set_and_never_the_test_set() -> None:
    """`val_loss` has to be a loss on the VALIDATION set, or it is a lie.

    The engine attaches raw Validation only after Data returns; Train then
    renders a temporary eval view with the declared answer fields.

    The dangerous half is the held-out lane: it runs the SAME data agent over
    the TEST set. A training-shaped copy of that would put the test set one
    config line away from being trained on, so the freeze must never take one
    from a held-out ticket.
    """
    from zevo.contracts.data import DataResult
    from zevo.contracts.train import TrainTaskInput

    assert "validation_dataset_path" in DataResult.model_fields
    assert "validation_dataset_path" in TrainTaskInput.model_fields

    src = Path("src/zevo/engine/run/runner.py").read_text()
    record = src[src.index("async def _record_prepared_scoring_data"):]
    record = record[:record.index("\nasync def ", 10)]
    # The engine-bound path is taken inside the validation-lane branch.
    guard = record.index("if not is_held_out_test:")
    assert record.index('("validation_dataset", "validation_dataset_path")') > guard, (
        "a train-visible copy of the HELD-OUT set must never be frozen"
    )
    assert "test_train_shaped" not in src, "there is no such thing, by design"


@pytest.mark.asyncio
async def test_the_train_ticket_is_handed_the_raw_validation_set(
    session_factory,
    tmp_path: Path,
) -> None:
    """The path is stamped by the harness, not carried by the orchestrator.

    Which file the trainer may compute a loss on is a fact about the run's
    splits. The orchestrator plans what to try; it does not choose which set
    measures the result.
    """
    from zevo.engine.run.runner import _build_train_input

    async with session_factory() as s:
        run = Run(
            metric="accuracy", id="r", task_name="t",
            started_at=datetime.now(timezone.utc),
            holdout={"validation_answer_fields": ["response"]},
        )
        s.add(run)
        s.add(Ticket(id="train-1", run_id="r", agent_id="train", status="queued",
                     iteration=1, payload={}))
        await s.commit()
        tk = await s.get(Ticket, "train-1")

        payload = {
            "base_model": "Qwen/Qwen3-0.6B",
            "model_source": "base_model",
            "parent_selection_rationale": "Initial adaptation starts from Baseline.",
            "training_method_pin": "lora_sft",
            "data_signature": "d" * 64,
        }
        inference_config = tmp_path / "inference_config.yaml"
        inference_config.write_text("schema_version: 1\n", encoding="utf-8")
        device_info = tmp_path / "device_info.json"
        _write_instance_device(device_info, run_id=run.id)
        inputs = {
            "training_dataset": {"path": "/w/dataset.jsonl"},
            "validation_dataset": {"path": "/w/validation_dataset.jsonl"},
            "inference_config": {"path": str(inference_config)},
            "device_info": {"path": str(device_info)},
        }
        built = await _build_train_input(
            tk, payload, inputs, work_dir="/w", generation_backend="vllm",
            run=run, session=s,
        )
        assert built.validation_dataset_path == "/w/validation_dataset.jsonl"
        assert built.validation_answer_fields == ["response"]
        assert built.dataset_path == "/w/dataset.jsonl", "training data is untouched"


@pytest.mark.asyncio
async def test_train_receives_public_wandb_tracking_without_persisting_the_key(
    session_factory, tmp_path: Path, monkeypatch,
) -> None:
    from zevo.engine.run.runner import _build_train_input

    monkeypatch.setenv("WANDB_API_KEY", "secret-token-that-must-not-be-stored")
    monkeypatch.setenv("WANDB_ENTITY", "zesearch")
    monkeypatch.setenv("WANDB_PROJECT", "zevo-public")
    inference_config = tmp_path / "inference_config.yaml"
    inference_config.write_text("schema_version: 1\n", encoding="utf-8")
    async with session_factory() as session:
        run = Run(
            metric="accuracy", id="wandb-run-123", task_name="t",
            holdout={"validation_answer_fields": ["response"]},
        )
        ticket = Ticket(
            id="train-wandb-001", run_id=run.id, agent_id="train",
            iteration=1, status="queued", payload={},
        )
        session.add_all([run, ticket])
        await session.commit()
        device_info = tmp_path / "device.json"
        _write_instance_device(device_info, run_id=run.id)
        built = await _build_train_input(
            ticket,
            {
                "base_model": "Qwen/Qwen3-0.6B",
                "model_source": "base_model",
                "parent_selection_rationale": "Initial branch.",
                "data_signature": "d" * 64,
            },
            {
                "training_dataset": {"path": "/w/train.jsonl"},
                "validation_dataset": {"path": "/w/validation.jsonl"},
                "inference_config": {"path": str(inference_config)},
                "device_info": {"path": str(device_info)},
            },
            work_dir="/w", generation_backend="vllm", run=run, session=session,
        )
    execution = built.execution_contract
    assert execution.tracking_provider == "weights_and_biases"
    assert execution.tracking_url.endswith("/runs/wandb-ru-train-wandb-001")
    assert execution.secret_environment_names == ["WANDB_API_KEY"]
    assert "WANDB_API_KEY" not in execution.required_environment
    assert "secret-token-that-must-not-be-stored" not in built.model_dump_json()



def test_each_run_data_version_produces_only_a_training_package() -> None:
    """Data returns signed Training artifacts; scoring artifacts are rejected."""
    from pydantic import ValidationError
    from zevo.contracts.data import DataRecipe, DataResult

    recipe = DataRecipe(
        dataset_name="train.jsonl",
        source_identity="/source/train.jsonl",
        source_fingerprint="a" * 64,
        training_method="full_sft",
        method_format="messages",
        method_ids=["inline_transform"],
        audit_steps=["normalized the JSONL records"],
    )

    complete = DataResult(
        status="succeeded",
        ticket_id="data-1",
        operation="prepare_run_data",
        training_dataset_path="/w/train.jsonl",
        prepare_script_path="/w/prepare_data.py",
        data_recipe_path="/w/data_recipe.json",
        error_message="",
        notes="prepared",
    )
    assert complete.operation == "prepare_run_data"

    with pytest.raises(ValidationError, match="data_recipe_path"):
        DataResult(
            status="succeeded",
            ticket_id="data-2",
            operation="prepare_run_data",
            training_dataset_path="/w/train.jsonl",
            error_message="",
            notes="incomplete",
        )

    with pytest.raises(ValidationError, match="engine-owned Validation"):
        DataResult(
            status="succeeded",
            ticket_id="data-3",
            operation="prepare_run_data",
            training_dataset_path="/w/train.jsonl",
            data_recipe_path="/w/data_recipe.json",
            scoring_public_path="/w/validation_public.jsonl",
            error_message="",
            notes="invalid scoring access",
        )

    with pytest.raises(ValidationError, match="must not produce trainable"):
        DataResult(
            status="succeeded",
            ticket_id="holdout-data-1",
            operation="prepare_holdout_data",
            training_dataset_path="/w/leak.jsonl",
            scoring_public_path="/w/test_public.jsonl",
            error_message="",
            notes="invalid held-out output",
        )


def test_the_boards_rank_on_the_held_out_series() -> None:
    """A board is a comparison, so it has to be on the set nobody tuned against.

    `baseline_and_best` reads the VALIDATION series — the set each run
    optimized — so a dashboard built on it ranks runs by how well each fitted
    its own yardstick. The held-out pair is the honest one.

    And the "best" end is NOT the maximum test score. Picking the iteration with
    the highest held-out result is selecting on the set the run is judged by,
    one level up from the loop doing it. It is the test score of whichever
    iteration won on VALIDATION.
    """
    from zevo.engine.observe.run_metrics import (
        baseline_and_best, baseline_and_best_test, improvement_test,
    )

    history = [
        {"iteration": 0, "source": "baseline", "score": 0.1671, "test_score": 0.1311},
        {"iteration": 6, "score": 0.4707, "test_score": 0.2696},   # val champion
        {"iteration": 7, "score": 0.4578, "test_score": 0.2515},
    ]
    assert baseline_and_best(history) == (0.1671, 0.4707)
    assert baseline_and_best_test(history) == (0.1311, 0.2696)
    assert improvement_test(history) == pytest.approx(0.1385)

    # THE rule: a round that scored higher on test but lower on validation does
    # not become the champion.
    tempting = history + [{"iteration": 8, "score": 0.40, "test_score": 0.99}]
    assert baseline_and_best_test(tempting) == (0.1311, 0.2696)

    # A run older than the held-out lane has no pair, and must not report 0.
    old = [{"iteration": 0, "source": "baseline", "score": 0.23},
           {"iteration": 1, "score": 0.41}]
    assert baseline_and_best_test(old) == (None, None)
    # None, not a numeric sentinel: any negative number may be a real regression.
    assert improvement_test(old) is None

    # And a negative regression must remain numeric, distinct from unknown.
    regressed = [
        {"iteration": 0, "source": "baseline", "score": 0.30, "test_score": 0.30},
        {"iteration": 1, "score": 0.35, "test_score": 0.29},
    ]
    assert improvement_test(regressed) == pytest.approx(-0.01)

    # Champion measured, baseline not: still no gain to report.
    half = [{"iteration": 0, "source": "baseline", "score": 0.2},
            {"iteration": 1, "score": 0.4, "test_score": 0.3}]
    assert baseline_and_best_test(half) == (None, 0.3)
    assert improvement_test(half) is None


def test_min_direction_uses_the_lowest_validation_champion() -> None:
    from zevo.engine.observe.run_metrics import (
        baseline_and_best, baseline_and_best_test, improvement, improvement_test,
    )

    history = [
        {"iteration": 0, "source": "baseline", "score": 0.60, "test_score": 0.55},
        {"iteration": 1, "source": "trained", "score": 0.40, "test_score": 0.38},
        {"iteration": 2, "source": "trained", "score": 0.20, "test_score": 0.25},
        {"iteration": 3, "source": "trained", "score": 0.30, "test_score": 0.10},
    ]
    assert baseline_and_best(history, "min") == (0.60, 0.20)
    assert baseline_and_best_test(history, "min") == (0.55, 0.25)
    assert improvement(history, "min") == pytest.approx(0.40)
    assert improvement_test(history, "min") == pytest.approx(0.30)
