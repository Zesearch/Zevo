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
        with pytest.raises(ClaudeStreamStalled, match="no substantive stdout/stderr output"):
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
@pytest.mark.parametrize("heartbeat_only", [False, True])
async def test_driver_kills_a_real_silent_process_group(
    monkeypatch, tmp_path, heartbeat_only,
) -> None:
    fake_claude = tmp_path / "fake-claude"
    fake_claude.write_text(
        ("#!/usr/bin/env python3\nimport time\nwhile True:\n"
         " print('{\"type\":\"tool_progress\",\"heartbeat\":true}', flush=True)\n"
         " time.sleep(0.005)\n") if heartbeat_only else
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


def test_tool_heartbeats_do_not_extend_explicit_timeout():
    import json
    now = 0.0
    activity = claude_module._CliActivity(lambda: now, max_tool_seconds=60)
    activity.stdout((json.dumps({'type': 'assistant', 'message': {'content': [
        {'type': 'tool_use', 'id': 'long', 'name': 'Bash',
         'input': {'timeout': 120000}}]}}) + '\n').encode())
    now = 40
    activity.stdout(b'{"type":"tool_progress","elapsed_time_seconds":40}\n')
    assert activity.latest() == 40
    now = 80
    activity.stdout(b'{"type":"tool_progress","elapsed_time_seconds":80}\n')
    assert activity.latest() == 60  # capped allowance, not extended by keepalive


def test_fragmented_heartbeat_does_not_count_but_real_stream_does():
    now = 0.0
    activity = claude_module._CliActivity(lambda: now, 60)
    now = 10
    activity.stdout(b'{"type":"tool_')
    activity.stdout(b'progress","heartbeat":true}\n')
    assert activity.latest() == 0
    activity.stdout(b'{"type":"stream_event","event":')
    assert activity.latest() == 10
    now = 20
    activity.stderr(b'real stderr output')
    assert activity.latest() == 20


def test_tool_result_clears_quiet_allowance():
    import json
    now = 0.0
    activity = claude_module._CliActivity(lambda: now, 100)
    for obj in [
        {'type': 'assistant', 'message': {'content': [
            {'type': 'tool_use', 'id': 'x', 'name': 'Bash', 'input': {'timeout': 90000}}]}},
        {'type': 'user', 'message': {'content': [
            {'type': 'tool_result', 'tool_use_id': 'x', 'content': 'done'}]}},
    ]:
        activity.stdout((json.dumps(obj) + '\n').encode())
    now = 50
    assert activity.latest() == 0
