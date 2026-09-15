from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from zevo.engine.agent.loader import AgentBlueprint
from zevo.engine.agent.drivers import claude_cli as claude_module
from zevo.engine.agent.drivers.claude_cli import (
    ClaudeCliDriver,
    ClaudeStreamStalled,
    DEFAULT_CLAUDE_IDLE_TIMEOUT_SECONDS,
    _await_cli_activity,
    _claude_idle_timeout_seconds,
)


@pytest.mark.asyncio
async def test_silent_cli_task_hits_the_idle_watchdog() -> None:
    loop = asyncio.get_running_loop()
    started = loop.time()
    task = asyncio.create_task(asyncio.sleep(60))
    try:
        with pytest.raises(ClaudeStreamStalled, match="no stdout/stderr bytes"):
            await _await_cli_activity(
                [task],
                last_activity=lambda: started,
                idle_timeout_seconds=0.04,
                agent_id="orchestrator",
            )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_recent_activity_allows_cli_task_to_finish() -> None:
    loop = asyncio.get_running_loop()
    last = loop.time()

    async def _active_work() -> None:
        nonlocal last
        for _ in range(4):
            await asyncio.sleep(0.01)
            last = loop.time()

    task = asyncio.create_task(_active_work())
    await _await_cli_activity(
        [task],
        last_activity=lambda: last,
        idle_timeout_seconds=0.025,
        agent_id="data",
    )


def test_idle_watchdog_defaults_and_can_be_disabled(monkeypatch) -> None:
    monkeypatch.delenv("ZEVO_CLAUDE_IDLE_TIMEOUT_SECONDS", raising=False)
    assert _claude_idle_timeout_seconds() == DEFAULT_CLAUDE_IDLE_TIMEOUT_SECONDS

    monkeypatch.setenv("ZEVO_CLAUDE_IDLE_TIMEOUT_SECONDS", "0")
    assert _claude_idle_timeout_seconds() == 0


@pytest.mark.asyncio
async def test_driver_kills_a_real_silent_process_group(
    monkeypatch, tmp_path,
) -> None:
    fake_claude = tmp_path / "fake-claude"
    fake_claude.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    fake_claude.chmod(0o755)
    monkeypatch.setenv("ZEVO_CLAUDE_IDLE_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setattr(
        claude_module, "_resolve_auth", lambda env: ("api_key", "test"),
    )

    class _Payload(BaseModel):
        ticket_id: str = "ticket-1"

    blueprint = AgentBlueprint(
        id="watchdog-test",
        name="Watchdog Test",
        title="Watchdog Test",
        reports_to="",
        default_driver="claude_cli",
        default_model="",
        instructions="Wait silently.",
        output_schema=None,
        tools=[],
        identity_path="",
    )
    started = asyncio.get_running_loop().time()

    with pytest.raises(ClaudeStreamStalled):
        await ClaudeCliDriver(str(fake_claude)).run_agent(
            blueprint=blueprint,
            input_payload=_Payload(),
            workspace_dir=str(tmp_path / "work"),
        )

    assert asyncio.get_running_loop().time() - started < 2.0
