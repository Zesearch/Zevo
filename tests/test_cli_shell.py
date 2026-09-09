from pathlib import Path

import pytest

from zevo.cli import shell


def test_parse_command_preserves_quoted_arguments() -> None:
    assert shell.parse_command('run create --run-name "demo run"') == [
        "run", "create", "--run-name", "demo run",
    ]


def test_parse_command_rejects_repeated_executable_prefix() -> None:
    with pytest.raises(ValueError, match="already inside Zevo"):
        shell.parse_command("zevo run list")


def test_host_runner_marks_forwarded_command_as_internal() -> None:
    runner = shell.CommandRunner(Path("/repo"), "backend")
    argv = runner._argv(["run", "list"])

    assert argv[-4:] == ["backend", "zevo", "run", "list"]
    assert "ZEVO_INTERACTIVE_COMMAND=1" in argv


def test_shell_runs_multiple_commands_without_exiting(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = iter(["run list", "task list", "exit"])
    calls: list[list[str]] = []

    class FakeRunner:
        def run(self, args: list[str]) -> int:
            calls.append(args)
            return 0

    monkeypatch.setattr("builtins.input", lambda _prompt: next(entered))
    monkeypatch.setattr(shell, "_print_landing", lambda: None)
    monkeypatch.setattr(shell, "_setup_readline", lambda _path: None)

    assert shell.interactive_shell(FakeRunner(), history_path=Path("unused")) == 0
    assert calls == [["run", "list"], ["task", "list"]]
