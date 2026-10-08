"""The same preflight route accepts real Auto and Standard envelopes without inventing a scorer."""
import asyncio
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from zevo.api.routers.ui import preflight as module


def test_auto_preflight_checks_selected_runtime_without_requiring_test_assets(monkeypatch):
    captured = {}
    def infra(provider, items, **kwargs):
        captured.update(provider=provider, **kwargs)
    monkeypatch.setattr(module, "_check_infra", infra)
    monkeypatch.setattr(module, "_check_provider_creds", AsyncMock())
    monkeypatch.setattr(module, "_check_orphaned_instances", AsyncMock())
    body = module.PreflightBody(mode="auto", user_request={"task_objective": "Improve short factual answers"}, gpu_provider="instance", ssh_host_id="selected-host", num_gpus=1)
    db = AsyncMock()
    db.get.return_value = SimpleNamespace(status="verified", category="instance")
    result = asyncio.run(module.preflight(body, db))
    assert captured["ssh_host_id"] == "selected-host"
    assert captured["provider"] == "instance"
    assert not result.blockers
    assert "auto_scoring_pending" in result.risks


def test_scoring_contract_cannot_accidentally_pass_as_standard_auto_draft():
    with pytest.raises(ValidationError, match="full scoring contract"):
        module.PreflightBody(user_request={"task_objective": "Improve short factual answers"})


def test_deleted_selected_host_blocks_auto_launch(monkeypatch):
    monkeypatch.setattr(module, "_check_infra", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_check_provider_creds", AsyncMock())
    monkeypatch.setattr(module, "_check_orphaned_instances", AsyncMock())
    db = AsyncMock()
    db.get.return_value = None
    body = module.PreflightBody(mode="auto", user_request={"task_objective": "Improve factual answers"}, gpu_provider="instance", ssh_host_id="deleted")
    result = asyncio.run(module.preflight(body, db))
    assert "ssh_host_missing" in result.blockers
    assert result.status == "blocked"


@pytest.mark.parametrize("provider", ["cluster", "cloud", "instance"])
@pytest.mark.parametrize("use_default", [False, True])
def test_per_submission_requires_slurm_at_preflight_and_launch(monkeypatch, provider, use_default):
    from fastapi import HTTPException
    from zevo.api.routers.shared import runs
    target = SimpleNamespace(provider=provider, cloud_backend="", ssh_host_id="")
    monkeypatch.setattr(module, "resolve_default_compute", AsyncMock(return_value=target))
    monkeypatch.setattr(runs, "resolve_default_compute", AsyncMock(return_value=target))
    monkeypatch.setattr(module, "_check_infra", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_check_provider_creds", AsyncMock())
    monkeypatch.setattr(module, "_check_orphaned_instances", AsyncMock())
    db = AsyncMock()
    fields = dict(mode="auto", user_request={"task_objective": "Improve factual answers"},
                  gpu_provider="" if use_default else provider, gpu_allocation_mode="per_submission")
    if use_default:
        fields.pop("gpu_provider")
    result = asyncio.run(module.preflight(module.PreflightBody(**fields), db))
    assert ("gpu_allocation_mode" in result.blockers) == (provider != "cluster")
    body = runs.CreateRunRequest(task_name="task", run_name="run", **fields)
    if provider == "cluster":
        assert asyncio.run(runs._resolve_run_compute(db, body)).provider == "cluster"
    else:
        with pytest.raises(HTTPException, match="Slurm"):
            asyncio.run(runs._resolve_run_compute(db, body))
