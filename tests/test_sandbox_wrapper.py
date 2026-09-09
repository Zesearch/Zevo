"""Tests for the explicit per-agent OpenShell workspace lifecycle."""
from __future__ import annotations

import pytest

from zevo.engine.agent.drivers import sandbox


BASE = ["claude", "-p", "--output-format", "stream-json", "--add-dir", "/ws", "-"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(sandbox.os.environ):
        if k.startswith("ZEVO_OPENSHELL"):
            monkeypatch.delenv(k, raising=False)
    yield


def test_configure_env_noop_in_none_mode() -> None:
    env = {"FOO": "bar"}
    assert sandbox.configure_sandbox_env(env, mode="none") == {"FOO": "bar"}


def test_openshell_gateway_endpoint_and_insecure_default(monkeypatch) -> None:
    monkeypatch.setenv("ZEVO_OPENSHELL_GATEWAY", "https://host.docker.internal:17670")
    env: dict[str, str] = {}
    sandbox.configure_sandbox_env(env, mode="openshell")
    assert env["OPENSHELL_GATEWAY_ENDPOINT"] == "https://host.docker.internal:17670"
    # local gateway uses a self-signed cert → insecure defaults on
    assert env["OPENSHELL_GATEWAY_INSECURE"] == "1"


def test_openshell_gateway_insecure_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ZEVO_OPENSHELL_GATEWAY", "https://gw.example.com")
    monkeypatch.setenv("ZEVO_OPENSHELL_GATEWAY_INSECURE", "0")
    env: dict[str, str] = {}
    sandbox.configure_sandbox_env(env, mode="openshell")
    assert env["OPENSHELL_GATEWAY_ENDPOINT"] == "https://gw.example.com"
    assert "OPENSHELL_GATEWAY_INSECURE" not in env


# ─────────────────── Phase 2: workspace-I/O lifecycle (pure) ─────────────────

def test_remote_workdir_is_sandbox_root_plus_leaf() -> None:
    # upload nests the workspace under its basename inside /sandbox
    assert sandbox.remote_workdir("/runs/r1/data-001") == "/sandbox/data-001"
    assert sandbox.remote_workdir("/tmp/ws/train-007") == "/sandbox/train-007"


def test_sandbox_exec_argv_shape(monkeypatch) -> None:
    base = ["claude", "-p", "--output-format", "stream-json", "-"]
    out = sandbox.sandbox_exec_argv("zevo-data-data-001", base, "/runs/r1/data-001")
    assert out[:5] == ["openshell", "sandbox", "exec", "-n", "zevo-data-data-001"]
    assert "--no-tty" in out
    assert out[out.index("--workdir") + 1] == "/sandbox/data-001"
    sep = out.index("--")
    assert out[sep + 1:] == base       # agent argv preserved verbatim after --
