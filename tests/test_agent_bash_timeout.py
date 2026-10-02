"""Agents get a Bash tool that can run a remote command to completion.

Claude Code's default 120 s Bash timeout turned every remote inference or
training command into background-and-poll: run 7 (a676d8ac) spent 10 of its
16 inference minutes in TaskOutput polls. The driver now sets the CLI's
BASH_DEFAULT_TIMEOUT_MS / BASH_MAX_TIMEOUT_MS so a foreground command is one
tool call.
"""
from __future__ import annotations

from zevo.engine.agent.drivers import claude_cli as cli


def test_defaults_cover_an_hour_and_the_train_contract() -> None:
    env = cli._bash_timeout_env({})
    assert env["BASH_DEFAULT_TIMEOUT_MS"] == str(3600 * 1000)
    assert int(env["BASH_MAX_TIMEOUT_MS"]) >= 14400 * 1000


def test_deployment_overrides_through_zevo_variables() -> None:
    env = cli._bash_timeout_env({
        cli.BASH_DEFAULT_TIMEOUT_ENV: "600", cli.BASH_MAX_TIMEOUT_ENV: "7200",
    })
    assert env == {"BASH_DEFAULT_TIMEOUT_MS": "600000", "BASH_MAX_TIMEOUT_MS": "7200000"}


def test_max_is_never_below_default_and_bad_values_fall_back() -> None:
    env = cli._bash_timeout_env({cli.BASH_DEFAULT_TIMEOUT_ENV: "7200", cli.BASH_MAX_TIMEOUT_ENV: "60"})
    assert env["BASH_MAX_TIMEOUT_MS"] == "7200000"
    env = cli._bash_timeout_env({cli.BASH_DEFAULT_TIMEOUT_ENV: "soon"})
    assert env["BASH_DEFAULT_TIMEOUT_MS"] == str(cli.DEFAULT_AGENT_BASH_TIMEOUT_SECONDS * 1000)


def test_explicit_claude_code_values_are_left_alone() -> None:
    env = cli._bash_timeout_env({"BASH_DEFAULT_TIMEOUT_MS": "90000"})
    assert "BASH_DEFAULT_TIMEOUT_MS" not in env
    assert "BASH_MAX_TIMEOUT_MS" in env
