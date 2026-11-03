from uuid import uuid4

import pytest
import typer
from fastapi import BackgroundTasks, HTTPException, Response

from zevo.api.routers.shared.runs import CreateRunRequest, create_run, submit_run
from zevo.api.routers.ui.preflight import _check_model_destination
from zevo.cli.zevo import _launch_model_policy
from zevo.contracts.model_save import ModelSavePolicy


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full_pipeline", "customized_pipeline", "auto"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_launch_without_destination_cannot_create_or_queue_work(mode, asynchronous):
    body = CreateRunRequest(task_name="example", run_name="example", mode=mode,
        user_request={"task_objective": "Learn legal reasoning"} if mode == "auto" else None,
        setup_id=uuid4() if asynchronous else None)
    background = BackgroundTasks()
    response = Response()
    # No database or caller: missing destination must fail before either is used.
    with pytest.raises(HTTPException, match="Final model destination is required") as error:
        await submit_run(body, response, background, db=None)
    assert error.value.status_code == 422
    assert background.tasks == []
    with pytest.raises(HTTPException) as error:
        await create_run(body, db=None)
    assert error.value.status_code == 422


def test_preflight_blocks_missing_destination_and_hf_token():
    items = []
    _check_model_destination(None, items, {})
    assert [(i.code, i.severity) for i in items] == [("model_destination_missing", "blocker")]
    items = []
    _check_model_destination(ModelSavePolicy(hf_repo_id="owner/model"), items, {})
    assert items[0].code == "model_destination_token_missing"
    for policy, keys in [(ModelSavePolicy(hf_repo_id="owner/model"), {"HF_TOKEN": "test"}),
                         (ModelSavePolicy(weights="remote", remote_dir="/models"), {})]:
        items = []
        _check_model_destination(policy, items, keys)
        assert items == []


@pytest.mark.parametrize("hf,remote", [("", ""), ("owner/model", "/models"), ("invalid", ""), ("", "/"), ("", "relative")])
def test_cli_rejects_missing_conflicting_or_invalid_destination(hf, remote):
    with pytest.raises(typer.BadParameter):
        _launch_model_policy(hf, remote)


def test_cli_preserves_destination_and_repository_visibility():
    assert _launch_model_policy(" owner/model ", "", False) == {
        "weights": "hf", "hf_repo_id": "owner/model", "hf_private": False, "remote_dir": ""}
    assert _launch_model_policy("", " /models ")["remote_dir"] == "/models"
