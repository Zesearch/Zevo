from __future__ import annotations

from pathlib import Path
import shutil

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.file_integrity import preserve_legacy_run_inputs
from zevo.contracts.orchestrator import TaskTestSet, UserRequest
from zevo.db.models import Base, Run, Task, Ticket
from zevo.engine.run.input_snapshots import (
    detect_input_changes,
    launch_input_snapshot,
    snapshot_user_request,
)


def _request(files, test, sample) -> UserRequest:
    member = TaskTestSet(
        name="quality",
        test_set=str(test),
        inference_query="Answer {question}",
        sample_submission=str(sample),
        metric="accuracy",
        answer_fields=["answer"],
    )
    return UserRequest(
        task_objective="Answer accurately",
        test_sets=[member],
        test_set=str(test),
        test_answer_fields=["answer"],
        test_sample_submission=str(sample),
        metric="accuracy",
        metric_direction="max",
        dataset=str(files / "bundle" / "train.csv"),
        base_model="owner/model",
        training_method="",
        constraints=[],
    )


def test_run_copies_inputs_and_reports_live_task_file_changes(tmp_path, monkeypatch):
    files = tmp_path / "files"
    holdout = tmp_path / "private"
    work = tmp_path / "runs"
    public = files / "bundle"
    private = holdout / "files" / "bundle"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    train = public / "train.csv"
    test = public / "test.csv"  # logical path; protected bytes live privately
    sample = public / "sample.csv"
    train.write_text("prompt,response\nhello,world\n")
    (private / "test.csv").write_text("question,answer\none,1\n")
    (private / "sample.csv").write_text("prediction\n1\n")

    monkeypatch.setenv("ZEVO_FILES_DIR", str(files))
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(holdout))
    monkeypatch.setenv("ZEVO_WORK_DIR", str(work))

    request = _request(files, test, sample)
    task = Task(
        name="mutable",
        task_objective="Answer accurately",
        test_sets=[item.model_dump(mode="json") for item in request.test_sets],
        test_set=str(test),
        test_answer_fields=["answer"],
        test_sample_submission=str(sample),
        metric_type="builtin",
        evaluation_script="",
        evaluator_sha256="",
        metric="accuracy",
        metric_direction="max",
    )
    launch = launch_input_snapshot(task, request)
    copied = snapshot_user_request("run-1", request)
    copied_train = copied.dataset
    copied_test = copied.test_sets[0].test_set
    assert copied_train != str(train)
    assert copied_test != str(test)
    assert Path(copied_train).read_text() == train.read_text()
    assert Path(copied_test).read_text() == (private / "test.csv").read_text()

    run = Run(
        id="run-1", task_name="mutable", metric="accuracy",
        input_snapshot=launch,
    )
    assert detect_input_changes(run, task) == []

    train.write_text("prompt,response\nchanged,value\n")
    changes = detect_input_changes(run, task)
    assert changes == [{"kind": "file", "name": "bundle", "status": "modified"}]
    assert Path(copied_train).read_text() == "prompt,response\nhello,world\n"

    task.task_objective = "A revised objective"
    changes = detect_input_changes(run, task)
    assert {item["kind"] for item in changes} == {"task", "file"}

    shutil.rmtree(public)
    shutil.rmtree(private)
    changes = detect_input_changes(run, None)
    assert {tuple(sorted(item.items())) for item in changes} == {
        tuple(sorted({"kind": "task", "name": "mutable", "status": "deleted"}.items())),
        tuple(sorted({"kind": "file", "name": "bundle", "status": "deleted"}.items())),
    }


@pytest.mark.asyncio
async def test_first_file_mutation_moves_legacy_run_references_to_run_snapshot(
    tmp_path, monkeypatch,
):
    files = tmp_path / "files"
    holdout = tmp_path / "private"
    work = tmp_path / "runs"
    public = files / "bundle"
    private = holdout / "files" / "bundle"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    train = public / "train.csv"
    test = private / "test.csv"
    train.write_text("prompt,response\nold,answer\n")
    test.write_text("question,answer\nold,1\n")
    monkeypatch.setenv("ZEVO_FILES_DIR", str(files))
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(holdout))
    monkeypatch.setenv("ZEVO_WORK_DIR", str(work))

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        run = Run(
            id="legacy", task_name="old-task", metric="accuracy",
            supervisor_ticket_id="orchestrate-legacy-001",
            decision_pins={"dataset": str(train)},
            holdout={"test_set": str(test)},
        )
        supervisor = Ticket(
            id="orchestrate-legacy-001", run_id="legacy",
            agent_id="orchestrator",
            payload={"user_request": {"dataset": str(train)}},
        )
        db.add_all([run, supervisor])
        await db.commit()

        await preserve_legacy_run_inputs(db, "bundle")
        await db.refresh(run)
        await db.refresh(supervisor)
        copied_train = Path(run.decision_pins["dataset"])
        copied_test = Path(run.holdout["test_set"])
        assert copied_train.read_text() == "prompt,response\nold,answer\n"
        assert copied_test.read_text() == "question,answer\nold,1\n"
        assert supervisor.payload["user_request"]["dataset"] == str(copied_train)
        assert run.input_snapshot["local_inputs_copied"] is False
        assert run.input_snapshot["preserved_file_sets"] == ["bundle"]
        assert run.input_snapshot["files"][0]["name"] == "bundle"

        train.write_text("prompt,response\nnew,answer\n")
        test.write_text("question,answer\nnew,2\n")
        await preserve_legacy_run_inputs(db, "bundle")
        assert copied_train.read_text() == "prompt,response\nold,answer\n"
        assert copied_test.read_text() == "question,answer\nold,1\n"

    await engine.dispose()
