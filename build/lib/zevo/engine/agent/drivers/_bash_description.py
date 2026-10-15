"""Deterministic, non-mutating labels for shell tool calls.

Agent-authored descriptions are preferred.  This module exists because a model
may still omit an optional CLI-tool field, while the UI needs a stable Overview
line that is not a wall of shell.  It never rewrites or replaces the command in
the transcript; callers add the returned text as separate event metadata.
"""
from __future__ import annotations

import re
import shlex
from pathlib import PurePosixPath


_MAX_DESCRIPTION = 160
_SETUP_WORDS = {"set", "cd", "export", "source", ".", "unset", "umask", "shopt"}
_BASH_TOOL_NAMES = {"bash", "run_bash", "exec_command", "shell"}


def is_bash_tool(name: object) -> bool:
    return str(name or "").strip().lower() in _BASH_TOOL_NAMES


def _clean(value: object) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) > _MAX_DESCRIPTION:
        return text[: _MAX_DESCRIPTION - 1].rstrip() + "…"
    return text


def _script_label(command: str) -> str:
    lower = command.lower()
    if re.search(r"(?:^|[/\s])train\.py(?:\s|$)", lower):
        return "Run training on the assigned GPU"
    if re.search(r"(?:^|[/\s])predict\.py(?:\s|$)", lower):
        return "Run inference on the assigned GPU"
    if re.search(r"(?:^|[/\s])eval(?:uate)?\.py(?:\s|$)", lower):
        return "Run evaluation"
    if any(token in lower for token in (
        "prepare_data.py", "reformat.py", "dataset.py", "prepare_dataset.py",
    )):
        return "Prepare the training dataset"
    return ""


def _first_action(command: str) -> str:
    """Return the first non-setup shell segment without evaluating it."""
    for segment in re.split(r"(?:\n|;|&&|\|\|)+", command):
        segment = segment.strip()
        if not segment or segment.startswith(("#", "<<")):
            continue
        try:
            words = shlex.split(segment, comments=False, posix=True)
        except ValueError:
            words = segment.split()
        if not words:
            continue
        head = words[0]
        if head in _SETUP_WORDS or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", head):
            continue
        return segment
    return command.strip().splitlines()[0].strip() if command.strip() else ""


def _basename_operand(words: list[str]) -> str:
    for word in reversed(words[1:]):
        if word.startswith("-") or word in {"|", ">", ">>", "2>&1"}:
            continue
        clean = word.strip("'\"")
        if clean and not clean.startswith(("http://", "https://", "$")):
            return PurePosixPath(clean).name
    return ""


def describe_bash_call(command: object, supplied: object = "") -> str:
    """Return an outcome-oriented label without exposing raw shell details."""
    authored = _clean(supplied)
    if authored:
        return authored

    raw = str(command or "").strip()
    if not raw:
        return "Run a shell command"
    lower = raw.lower()

    # Codex commonly reports its execution wrapper as ``bash -lc <command>``.
    # The wrapped command is the action; the shell process is transport.
    try:
        wrapper_words = shlex.split(raw, comments=False, posix=True)
    except ValueError:
        wrapper_words = []
    if (
        len(wrapper_words) >= 3
        and PurePosixPath(wrapper_words[0]).name in {"bash", "sh", "zsh"}
        and wrapper_words[1] in {"-c", "-lc"}
    ):
        return describe_bash_call(" ".join(wrapper_words[2:]))

    # The common helper may be defined in the same multiline call. Require an
    # invocation or the concrete message endpoint, not merely ``post_message()``.
    if (
        re.search(r"(?m)^\s*post_message\s+(?!\()", raw)
        or re.search(r"/tickets/[^\s'\"]+/messages", lower)
    ):
        return "Post a ticket update"

    script = _script_label(raw)
    if "ssh " in lower or re.search(r"(^|\s)srun\s", lower):
        if "nvidia-smi" in lower:
            return "Check remote GPU availability"
        if script:
            return script
        if "<<" in raw:
            return "Run a script on the assigned remote host"
        return "Run a command on the assigned remote host"

    if "<<" in raw:
        action = _first_action(raw).lower()
        if re.search(r"\bpython(?:3)?\b", action):
            return script or "Run an inline Python script"
        return script or "Run an inline shell script"

    if script:
        return script

    action = _first_action(raw)
    try:
        words = shlex.split(action, comments=False, posix=True)
    except ValueError:
        words = action.split()
    if not words:
        return "Run a shell command"
    head = PurePosixPath(words[0]).name.lower()
    operand = _basename_operand(words)

    if head in {"python", "python3"}:
        if "-c" in words:
            return "Run a Python command"
        return f"Run {operand}" if operand else "Run a Python script"
    if head == "curl":
        if "/messages" in lower:
            return "Post a ticket update"
        if re.search(r"\s(?:-x\s+)?(?:post|put|patch|delete)\b", lower):
            return "Update Zevo API state"
        return "Read Zevo API state"
    if head in {"jq"}:
        return "Inspect structured output"
    if head in {"rg", "grep", "find", "ls"}:
        return "Inspect files and state"
    if head in {"cat", "head", "tail", "sed"}:
        return f"Inspect {operand}" if operand else "Inspect command output"
    if head in {"scp", "rsync"}:
        return "Transfer files"
    if head in {"cp", "mv"}:
        return f"Prepare {operand}" if operand else "Prepare local files"
    if head == "mkdir":
        return "Prepare working directories"
    if head == "chmod":
        return f"Make {operand} executable" if operand else "Prepare an executable script"
    if head in {"docker", "git", "pip", "pip3", "npm", "pnpm"}:
        subcommand = next((w for w in words[1:] if not w.startswith("-")), "")
        return _clean(f"Run {head} {subcommand}" if subcommand else f"Run {head}")
    return _clean(f"Run {head} command")


__all__ = ["describe_bash_call", "is_bash_tool"]
