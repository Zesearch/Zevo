from __future__ import annotations

import asyncio
import signal

import pytest

from zevo.engine.run import process_registry


class _Process:
    pid = 12345
    returncode = None


def test_process_registered_during_gate_is_stopped_then_resumed(monkeypatch) -> None:
    key = "train-ticket"
    proc = _Process()
    seen: list[signal.Signals] = []
    monkeypatch.setattr(
        process_registry,
        "_signal_process_group",
        lambda _proc, sig: seen.append(sig) or True,
    )
    try:
        process_registry.pause(key)
        process_registry.register(key, proc)
        assert seen == [signal.SIGSTOP]
        assert process_registry.is_paused(key) is True

        process_registry.resume(key)
        assert seen == [signal.SIGSTOP, signal.SIGCONT]
        assert process_registry.is_paused(key) is False
    finally:
        process_registry.unregister(key, proc)
        process_registry.resume(key)


@pytest.mark.asyncio
async def test_in_process_tool_waits_for_instruction_decision() -> None:
    key = "data-ticket"
    process_registry.pause(key)
    try:
        waiter = asyncio.create_task(
            process_registry.wait_until_resumed(key, poll_seconds=0.001)
        )
        await asyncio.sleep(0.005)
        assert waiter.done() is False
        process_registry.resume(key)
        await asyncio.wait_for(waiter, timeout=0.1)
    finally:
        process_registry.resume(key)


@pytest.mark.asyncio
async def test_paused_time_does_not_consume_process_timeout() -> None:
    key = "evaluation-ticket"
    completion: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    process_registry.pause(key)
    try:
        waiter = asyncio.create_task(process_registry.wait_for_active_time(
            completion, key=key, timeout=0.01, poll_seconds=0.001,
        ))
        await asyncio.sleep(0.03)
        assert waiter.done() is False
        process_registry.resume(key)
        completion.set_result("finished")
        assert await asyncio.wait_for(waiter, timeout=0.1) == "finished"
    finally:
        process_registry.resume(key)
