from __future__ import annotations

import pytest
from fastapi import HTTPException

from zevo.api.routers.ui.agents import _require_public_agent


def test_evaluation_runner_has_no_agent_detail_surface() -> None:
    with pytest.raises(HTTPException) as exc:
        _require_public_agent("evaluation")
    assert exc.value.status_code == 404


@pytest.mark.parametrize(
    "agent_id",
    ["orchestrator", "infrastructure", "data", "inference", "train", "registry"],
)
def test_llm_agents_remain_public(agent_id: str) -> None:
    _require_public_agent(agent_id)
