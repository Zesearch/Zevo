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
