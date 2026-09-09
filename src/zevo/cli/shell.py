"""Interactive command shell used by the host shim and direct installations.

The public CLI has one entry point: run ``zevo`` first, then enter commands
without repeating the executable name. Command execution is still delegated
to the canonical Typer application, so this shell does not duplicate parsing
or business logic.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional, Sequence


# Match Rich's former ``bold yellow`` landing color. The 256-color gold used
# here previously was noticeably brighter on the macOS terminal palette.
_GOLD = "\033[1;33m"
_COMMAND_ACCENT = "\033[1;37m"
_DIM = "\033[2m"
_RESET = "\033[0m"

_BANNER = r"""
   ███████╗███████╗██╗   ██╗ ██████╗
   ╚══███╔╝██╔════╝██║   ██║██╔═══██╗
     ███╔╝ █████╗  ██║   ██║██║   ██║
    ███╔╝  ██╔══╝  ╚██╗ ██╔╝██║   ██║
   ███████╗███████╗ ╚████╔╝ ╚██████╔╝
   ╚══════╝╚══════╝  ╚═══╝   ╚═════╝"""

_COMMANDS: dict[tuple[str, ...], tuple[str, ...]] = {
    (): (
        "run", "agent", "ticket", "task", "file", "models", "model", "price",
        "wakeups", "heartbeats", "dashboard", "leaderboard", "hardware",
        "settings", "levels", "gpu-search", "gpu-rent", "gpu-destroy",
        "set-cloud-backend", "set-secret", "daemon", "server", "seed-agents",
        "ui", "help", "clear", "exit",
    ),
    ("run",): ("create", "list", "show", "watch", "artifact", "cancel", "rm"),
    ("agent",): (
        "list", "show", "status", "skills", "instructions", "drivers", "set",
        "ping", "task", "run", "invoke",
    ),
    ("ticket",): (
        "list", "show", "cancel", "rerun", "retry-status", "heartbeat", "message",
    ),
    ("task",): ("list", "show", "add", "edit", "rm", "setting"),
    ("task", "setting"): ("list", "show", "add", "edit", "rm"),
    ("file",): ("list", "show", "add", "add-remote", "note", "edit-remote", "rm", "profile"),
    ("model",): ("show", "card", "compare"),
    ("price",): ("models", "gpu"),
    ("wakeups",): ("list", "show"),
    ("heartbeats",): ("list", "tail", "cancel"),
    ("help",): (
        "run", "agent", "ticket", "task", "file", "models", "model", "price",
        "wakeups", "heartbeats", "dashboard", "leaderboard", "hardware",
        "settings", "levels", "gpu-search", "gpu-rent", "gpu-destroy",
        "set-cloud-backend", "set-secret", "daemon", "server", "seed-agents",
    ),
}


def parse_command(line: str) -> list[str]:
    """Parse one prompt line and reject the removed executable prefix."""
    args = shlex.split(line)
    if args and args[0] == "zevo":
        raise ValueError("You are already inside Zevo; enter the command without 'zevo'.")
    return args


def _print_landing() -> None:
    print(f"{_GOLD}{_BANNER}{_RESET}")
    print(f"   {_DIM}A self-improving system for evolving language models{_RESET}\n")
    inner_width = 50
    top_label = "─ Zevo Command Shell "
    message_before = " Enter a command. Use "
    message_after = " to see every command."
    message_width = len(message_before) + len("help") + len(message_after)
    print(
        f"{_GOLD}╭{top_label}"
        f"{'─' * (inner_width - len(top_label))}╮{_RESET}"
    )
    print(
        f"{_GOLD}│{_RESET}{message_before}{_COMMAND_ACCENT}help{_RESET}{message_after}"
        f"{' ' * (inner_width - message_width)}{_GOLD}│{_RESET}"
    )
    print(f"{_GOLD}╰{'─' * inner_width}╯{_RESET}\n")


def _print_help() -> None:
    print("Commands are entered without a 'zevo' prefix:\n")
    print("  run         Create, inspect, watch, cancel, or remove Runs")
    print("  task        Manage Tasks and their saved Settings")
    print("  file        Manage reusable input files")
    print("  agent       Inspect or configure Agents")
    print("  ticket      Inspect and repair work Tickets")
    print("  models      List saved models")
    print("  model       Show, print, or compare individual models")
    print("  price       Inspect model-token and GPU prices")
    print("  wakeups     Inspect the Agent wake-up queue")
    print("  heartbeats  Inspect live Agent execution")
    print("  dashboard   Show system-wide Run statistics")
    print("  leaderboard Compare saved models")
    print("  hardware    Inspect available compute")
    print("  settings    Inspect configuration and credentials")
    print("  levels      Show autonomy levels")
    print("  gpu-*       Search, rent, or destroy cloud GPUs")
    print("  set-*       Configure credentials or the cloud backend")
    print("  daemon      Run the scheduler heartbeat loop")
    print("  server      Run the API server")
    print("  seed-agents Seed the Agent catalogue")
    print("  ui [page]   Open the web interface")
    print("  clear       Clear this screen")
    print("  exit        Leave Zevo\n")
    print("Use 'help <command>' or '<command> --help' for command-specific options.")


def _setup_readline(history_path: Path) -> None:
    try:
        import atexit
        import readline
    except ImportError:
        return

    def complete(text: str, state: int) -> Optional[str]:
        before = readline.get_line_buffer()[: readline.get_begidx()]
        try:
            words = shlex.split(before)
        except ValueError:
            words = before.split()
        candidates = _COMMANDS.get(tuple(words), ())
        matches = [candidate for candidate in candidates if candidate.startswith(text)]
        return matches[state] + " " if state < len(matches) else None

    readline.set_completer(complete)
    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    try:
        readline.read_history_file(history_path)
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    def save_history() -> None:
        try:
            readline.write_history_file(history_path)
        except OSError:
            pass

    atexit.register(save_history)


class CommandRunner:
    def __init__(self, repo: Path, service: str, *, local: bool = False) -> None:
        self.repo = repo
        self.service = service
        self.local = local

    def _argv(self, args: Sequence[str], *, no_tty: bool = False) -> list[str]:
        if self.local:
            return [sys.executable, "-m", "zevo.cli.zevo", *args]
        argv = [
            "docker", "compose", "-f", str(self.repo / "docker-compose.yml"),
            "exec",
        ]
        if no_tty:
            argv.append("-T")
        argv.extend([
            "-e", "ZEVO_API_BASE=http://web:80",
            "-e", "ZEVO_INTERACTIVE_COMMAND=1",
            self.service, "zevo", *args,
        ])
        return argv

    def run(self, args: Sequence[str]) -> int:
        env = os.environ.copy()
        env["ZEVO_INTERACTIVE_COMMAND"] = "1"
        env["DOCKER_CLI_HINTS"] = "false"
        try:
            return subprocess.run(self._argv(args), env=env, check=False).returncode
        except KeyboardInterrupt:
            print(f"\n{_DIM}command interrupted; Zevo Shell is still open{_RESET}")
            return 130
        except FileNotFoundError as exc:
            print(f"command runner unavailable: {exc}", file=sys.stderr)
            return 127

    def help(self, args: Sequence[str]) -> int:
        env = os.environ.copy()
        env["ZEVO_INTERACTIVE_COMMAND"] = "1"
        env["DOCKER_CLI_HINTS"] = "false"
        env["NO_COLOR"] = "1"
        try:
            result = subprocess.run(
                self._argv([*args, "--help"], no_tty=True),
                env=env,
                check=False,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            print(f"command runner unavailable: {exc}", file=sys.stderr)
            return 127
        output = (result.stdout + result.stderr).replace("zevo ", "")
        print(output.rstrip())
        return result.returncode

    def open_ui(self, page: str = "") -> int:
        port = "5173"
        env_file = self.repo / ".env"
        if env_file.is_file():
            for raw_line in env_file.read_text(encoding="utf-8").splitlines():
                if raw_line.startswith("WEB_PORT="):
                    port = raw_line.partition("=")[2].strip() or port
        url = f"http://localhost:{port}/{page.lstrip('/')}"
        print(f"opening {url}")
        if not webbrowser.open(url):
            print(f"open it manually: {url}")
        return 0


def interactive_shell(runner: CommandRunner, *, history_path: Optional[Path] = None) -> int:
    _print_landing()
    _setup_readline(history_path or Path.home() / ".zevo_history")
    prompt = f"\001{_GOLD}\002zevo › \001{_RESET}\002"

    while True:
        try:
            line = input(prompt)
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print(f"\n{_DIM}use 'exit' or Ctrl-D to leave Zevo{_RESET}")
            continue

        try:
            args = parse_command(line)
        except ValueError as exc:
            print(f"{_GOLD}{exc}{_RESET}")
            continue
        if not args:
            continue

        command = args[0].lower()
        if command in {"exit", "quit"}:
            return 0
        if command == "clear":
            print("\033[2J\033[H", end="")
            continue
        if command == "help":
            if len(args) == 1:
                _print_help()
            else:
                runner.help(args[1:])
            continue
        if command in {"ui", "open"}:
            if len(args) > 2:
                print("usage: ui [page]")
            else:
                runner.open_ui(args[1] if len(args) == 2 else "")
            continue
        if "--help" in args or "-h" in args:
            runner.help([arg for arg in args if arg not in {"--help", "-h"}])
            continue
        runner.run(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--service", default=os.environ.get("ZEVO_SVC", "backend"))
    parser.add_argument("--local", action="store_true")
    options = parser.parse_args(argv)
    repo = (options.repo or Path(__file__).resolve().parents[3]).resolve()
    return interactive_shell(CommandRunner(repo, options.service, local=options.local))


if __name__ == "__main__":
    raise SystemExit(main())
