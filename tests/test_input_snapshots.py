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
    input_change_details,
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
    details = input_change_details(run, task)
    objective = details[0]["fields"][0]
    assert objective["field"] == "task_objective"
    assert objective["before"] == "Answer accurately"
    assert objective["after"] == "A revised objective"
    assert objective["before_lines"] == [{"text": "Answer accurately", "changed": True}]
    # Same-size/same-schema dataset edits still appear via their content hash.
    assert any(row["field"] == "public/train.csv / sha256" for row in details[1]["fields"])
    assert not any("private/test.csv" in row["field"] for row in details[1]["fields"])

    train.write_text("prompt,response,extra\nchanged,value,1\n")
    fields = input_change_details(run, task)[1]["fields"]
    columns = next(row for row in fields if row["field"].endswith(" / columns"))
    assert columns["before"] == ["prompt", "response"]
    assert columns["after"] == ["prompt", "response", "extra"]

    shutil.rmtree(public)
    shutil.rmtree(private)
    changes = detect_input_changes(run, None)
    assert {tuple(sorted(item.items())) for item in changes} == {
        tuple(sorted({"kind": "task", "name": "mutable", "status": "deleted"}.items())),
        tuple(sorted({"kind": "file", "name": "bundle", "status": "deleted"}.items())),
    }
    deleted = input_change_details(run, None)
    assert deleted[0]["status"] == "deleted"
    assert all(row["after"] is None for row in deleted[0]["fields"])
    assert all(row["after"] is None for row in deleted[1]["fields"])


def test_legacy_comparison_only_uses_hash_verified_definition():
    from types import SimpleNamespace
    from zevo.engine.method.evaluation_identity import digest_json

    original = {"task_objective": "Original", "test_sets": [{"name": "quality", "inference_query": "Answer"}]}
    run = SimpleNamespace(task_name="task", task_objective="Original", holdout={}, input_snapshot={
        "version": 1, "task": {"name": "task", "sha256": digest_json(original)},
        "sources": {"test_sets": original["test_sets"]},
    })
    task = SimpleNamespace(name="task", task_objective="Updated", test_sets=original["test_sets"])
    assert input_change_details(run, task)[0]["fields"][0]["before"] == "Original"
    run.task_objective = "Unverifiable"
    detail = input_change_details(run, task)[0]
    assert detail["fields"] == []
    assert "unavailable" in detail["note"]


def test_file_details_do_not_snapshot_a_shared_tenant_root(tmp_path, monkeypatch):
    from zevo.engine.run.input_snapshots import catalogue_names, file_set_snapshot

    files = tmp_path / "files"
    for tenant in ("alice", "bob"):
        folder = files / "_t" / tenant / "bundle"
        folder.mkdir(parents=True)
        (folder / "notes.txt").write_text(tenant)
    monkeypatch.setenv("ZEVO_FILES_DIR", str(files))
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "private"))
    assert catalogue_names([str(files / "_t/alice/bundle/notes.txt")]) == ["_t/alice/bundle"]
    assert "entries" not in file_set_snapshot("_t")
    saved = file_set_snapshot("_t/alice/bundle")
    assert [entry.get("text") for entry in saved["entries"]] == ["alice"]


@pytest.mark.asyncio
async def test_comparison_rejects_agent_requests_before_reading_private_inputs(monkeypatch):
    from fastapi import HTTPException
    from starlette.requests import Request
    from zevo.api.routers.shared.runs import get_run_input_changes
    from zevo.api import ui_access

    monkeypatch.setattr(ui_access, "_ui_access_token", lambda: "dashboard-token")
    for headers in [[], [(b"x-zevo-ui-access", b"dashboard-token"), (b"x-zevo-worker", b"1")]]:
        with pytest.raises(HTTPException) as error:
            await get_run_input_changes("run", Request({"type": "http", "headers": headers}), None)
        assert error.value.status_code == 403


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


@pytest.mark.asyncio
async def test_comparison_inherits_run_detail_access_checks(monkeypatch):
    from fastapi import FastAPI, HTTPException
    from httpx import ASGITransport, AsyncClient
    from zevo.api.routers.shared import runs
    from zevo.api import ui_access

    monkeypatch.setattr(ui_access, "_ui_access_token", lambda: "dashboard-token")
    app = FastAPI()
    app.include_router(runs.router)

    async def deny_run():
        raise HTTPException(404, "run not visible")

    async def unused_db():
        yield None

    app.dependency_overrides[runs.get_run] = deny_run
    app.dependency_overrides[runs.get_db] = unused_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/runs/other-users-run/input-changes", headers={"x-zevo-ui-access": "dashboard-token"})
    assert response.status_code == 404
    assert response.json()["detail"] == "run not visible"
