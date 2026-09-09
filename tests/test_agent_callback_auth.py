"""Agents learn the API auth header from the engine, not by guessing.

A hosted deployment gates the API behind X-Zevo-Service. The playbooks'
callbacks send `-H "$ZEVO_API_AUTH_HEADER"` when it is set; on an open install
it is unset and the expansion is empty.
"""
from __future__ import annotations

from types import SimpleNamespace

from zevo.engine.agent.drivers._agent_loop import build_tool_env


def _env(base: dict) -> dict:
    return build_tool_env(base_env=base, workspace_dir="/w/run-1/ticket-1",
                          input_payload=SimpleNamespace(ticket_id="t1"), agent_id="data")


def test_service_token_becomes_the_auth_header() -> None:
    env = _env({"ZEVO_SERVICE_TOKEN": "s3cret"})
    assert env["ZEVO_API_AUTH_HEADER"] == "X-Zevo-Service: s3cret"


def test_open_install_sets_no_header_and_an_explicit_one_is_kept() -> None:
    assert "ZEVO_API_AUTH_HEADER" not in _env({})
    env = _env({"ZEVO_SERVICE_TOKEN": "s3cret", "ZEVO_API_AUTH_HEADER": "Authorization: Bearer x"})
    assert env["ZEVO_API_AUTH_HEADER"] == "Authorization: Bearer x"
