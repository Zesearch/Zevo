"""Shared tool-use helper for the SDK/API drivers.

Codex CLI ships built-in shell / read / write tools. Anthropic and
Bedrock don't -- they only know what tools you declare. To preserve the
contract that an agent can do `curl`, `jq`, `python -c ...`,
etc. without declaring anything, the Anthropic and Bedrock drivers
provide three built-in tools at the SDK layer:

    run_bash(command: str) -> {stdout, stderr, exit_code}
    read_file(path: str) -> str | error
    write_file(path: str, content: str) -> {bytes_written}

This module owns:

  - Vendor-neutral tool *schema* dicts (each driver translates to its
    SDK's exact tool_use JSON format).
  - Implementations of the three tools as plain async functions.
  - emit_tool_event(...) -- produces the canonical tool_call / tool_result
    event pair the UI's LiveTranscript expects.

Both event types carry the same `item_id` so the UI can pair them.
"""
from __future__ import annotations

import asyncio
import os
import signal
import uuid
from typing import Any, Callable

from zevo.engine.agent.drivers._bash_description import describe_bash_call
from zevo.engine.run import process_registry


# ---------------------------------------------------------------------------
# Vendor-neutral tool zevo.contracts. Each driver re-maps these onto its SDK's
# exact tool-spec format.
# ---------------------------------------------------------------------------

BASH_TOOL = {
    "name": "run_bash",
    "description": (
        "Execute a bash command. Use this for `curl`, `jq`, `python -c ...`, "
        "`cat`, `grep`, `pip install`, etc. The command runs inside the "
        "ticket's WORK_DIR with TICKET_ID / AGENT_ID / RUN_ID / ZEVO_API_BASE "
        "exported. Returns stdout, stderr, and exit_code."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to run (single line or `bash -c '<multiline>'`).",
            },
            "description": {
                "type": "string",
                "description": (
                    "Required concise outcome-oriented label for the UI Overview "
                    "(about 3-8 words). Describe why this command is being run; "
                    "do not repeat shell syntax, paths, credentials, or secrets."
                ),
            },
            "timeout_sec": {
                "type": "integer",
                "description": "Optional timeout in seconds (default 14400 for Train, 120 otherwise; max ZEVO_BASH_MAX_SEC, default 14400).",
            },
        },
        "required": ["command", "description"],
    },
}

READ_FILE_TOOL = {
    "name": "read_file",
    "description": (
        "Read a UTF-8 text file from disk. Returns the file contents as a "
        "string. Use this to inspect generated scripts, training output, "
        "device_info.json, etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or work-dir-relative path."},
            "max_chars": {
                "type": "integer",
                "description": "Cap output length (default 8000). Useful to avoid huge files.",
            },
        },
        "required": ["path"],
    },
}

WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": (
        "Write a UTF-8 text file. Overwrites if it exists. Returns the number "
        "of bytes written. Use this to materialize generated `train.py`, "
        "`reformat.py`, `predict.py`, etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or work-dir-relative path."},
            "content": {"type": "string", "description": "Full file contents."},
        },
        "required": ["path", "content"],
    },
}

SKILL_TOOL = {
    "name": "Skill",
    "description": (
        "Load a method skill for this stage. The available skills (and when to "
        "use each) are in your prompt's 'Method Skills' section. Call "
        "action='list' to see them again, or action='load' with name='<skill>' "
        "(e.g. 'lora-sft') to load that skill's full instructions — then follow "
        "them and write the code. Follow the ownership in your typed input: Train "
        "honors `training_method_pin` and otherwise selects from compatible Skills; "
        "Data uses `training_method` only to choose record semantics and selects its "
        "own preparation Skills; Infrastructure loads the matching site Skill."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "load"],
                "description": "'list' the staged skills, or 'load' one by name.",
            },
            "name": {
                "type": "string",
                "description": "Skill to load (hyphen or underscore form). Required for action='load'.",
            },
        },
        "required": ["action"],
    },
}


# The three every agent always has. `Skill` is added on top only when
# the agent actually has skills staged, so a stage with none does not
# advertise a tool that can only answer "there are none".
BUILTIN_TOOLS: list[dict] = [BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def exec_bash(
    command: str,
    *,
    cwd: str,
    env: dict[str, str],
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Run `bash -c <command>` and return {stdout, stderr, exit_code}.

    Timeout is clamped to [1, ZEVO_BASH_MAX_SEC] (default max 14400s = 4h, so long
    remote training runs can block synchronously). On timeout the process is
    killed and exit_code is set to 124 (matches GNU timeout's convention).

    Two things make this cancellable, and both are needed:

    `start_new_session` puts the shell in a process group of its own, so the
    whole tree -- the shell, the ssh it opens, the srun behind that -- can be
    signalled at once. Without it the group is the daemon's own and killing it
    would take the daemon down with it, so the only safe option would be to kill
    the shell alone and orphan everything it started.

    Registering under the TICKET id is what makes cancel able to find this at
    all. The CLI drivers register their own subprocess under the heartbeat id,
    but an in-process driver has no such subprocess -- these shells are the only
    thing it spawns, and until they were tracked, cancelling a run left them
    running against a GPU the run no longer held.
    """
    _max = int(os.environ.get("ZEVO_BASH_MAX_SEC", "14400") or "14400")
    default_timeout = 14400 if str(env.get("AGENT_ID") or "") == "train" else 120
    timeout_sec = max(1, min(int(timeout_sec or default_timeout), _max))
    proc = await asyncio.create_subprocess_exec(
        "bash", "-c", command,
        cwd=cwd or None,
        env={**env},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    key = str(env.get("TICKET_ID") or "")
    process_registry.register(key, proc)
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        _kill_tree(proc)
        return {
            "stdout": "",
            "stderr": f"[run_bash] timed out after {timeout_sec}s",
            "exit_code": 124,
        }
    except asyncio.CancelledError:
        # The turn is being torn down; the shell must not outlive it.
        _kill_tree(proc)
        raise
    finally:
        process_registry.unregister(key, proc)
    return {
        "stdout": (stdout_b or b"").decode("utf-8", errors="replace"),
        "stderr": (stderr_b or b"").decode("utf-8", errors="replace"),
        "exit_code": int(proc.returncode or 0),
    }


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the shell and everything it started. Never raises."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


def exec_read(path: str, *, cwd: str, max_chars: int = 8000) -> dict[str, Any]:
    """Read a UTF-8 text file. Always returns a dict to keep the tool-result
    shape uniform; on error, sets `error` and leaves `content` empty."""
    max_chars = max(1, int(max_chars or 8000))
    abs_path = path if os.path.isabs(path) else os.path.join(cwd or "", path)
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            data = f.read(max_chars + 1)
    except FileNotFoundError:
        return {"path": abs_path, "content": "", "error": "file not found"}
    except OSError as e:
        return {"path": abs_path, "content": "", "error": str(e)}
    truncated = len(data) > max_chars
    if truncated:
        data = data[:max_chars]
    return {
        "path": abs_path,
        "content": data,
        "truncated": truncated,
        "error": "",
    }


def exec_write(path: str, content: str, *, cwd: str) -> dict[str, Any]:
    """Write a UTF-8 text file, creating parent dirs if needed."""
    abs_path = path if os.path.isabs(path) else os.path.join(cwd or "", path)
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            n = f.write(content)
    except OSError as e:
        return {"path": abs_path, "bytes_written": 0, "error": str(e)}
    return {"path": abs_path, "bytes_written": int(n), "error": ""}


# ---------------------------------------------------------------------------
# Event-shape helpers (the canonical tool_call / tool_result event format)
# ---------------------------------------------------------------------------

def _new_item_id() -> str:
    """A short unique id used to pair tool_call <-> tool_result events."""
    return f"item_{uuid.uuid4().hex[:12]}"


def emit_tool_call(
    event_sink: Callable[[dict], None] | None,
    *,
    item_id: str,
    tool: str,
    args: dict[str, Any],
) -> None:
    """Emit `{"type": "tool_call", "payload": {tool, command/args, item_id}}`.

    For `run_bash`, surface the command as a top-level `command` field so
    the UI renders it like the canonical shell tool_call. For other tools, pass
    the full args dict under `args`.
    """
    if event_sink is None:
        return
    payload: dict[str, Any] = {"tool": tool, "item_id": item_id}
    if tool == "run_bash":
        command = str(args.get("command", ""))
        supplied = str(args.get("description", "") or "").strip()
        payload["input"] = dict(args)
        payload["command"] = command
        payload["description"] = describe_bash_call(command, supplied)
        payload["description_source"] = "agent" if supplied else "fallback"
    else:
        payload["args"] = args
    try:
        event_sink({"type": "tool_call", "payload": payload})
    except Exception:
        pass


def emit_tool_result(
    event_sink: Callable[[dict], None] | None,
    *,
    item_id: str,
    tool: str,
    result: dict[str, Any],
) -> None:
    """Emit `{"type": "tool_result", "payload": {tool, output, exit_code, item_id}}`.

    Shape mirrors the canonical command_execution `item_completed` event so the
    UI's TranscriptEventRow doesn't need to special-case us.
    """
    if event_sink is None:
        return
    payload: dict[str, Any] = {"tool": tool, "item_id": item_id}
    if tool == "run_bash":
        payload["output"] = (result.get("stdout", "") or "") + (
            ("\n[stderr] " + result["stderr"]) if result.get("stderr") else ""
        )
        payload["exit_code"] = int(result.get("exit_code", 0) or 0)
    else:
        # read_file / write_file / future tools: pass the structured result
        # under `output` as JSON so the UI renders it sensibly.
        payload["output"] = result
        payload["exit_code"] = 1 if result.get("error") else 0
    try:
        event_sink({"type": "tool_result", "payload": payload})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cost meter helper
# ---------------------------------------------------------------------------

def emit_turn_completed(
    event_sink: Callable[[dict], None] | None,
    *,
    usage: dict[str, Any],
) -> None:
    """Emit `{"type": "turn_completed", "payload": {"usage": {...}}}` so the
    runner's `accumulate_usage` (runner.py:448-451) folds the tokens into
    HeartbeatRun.input_tokens etc.

    `usage` MUST already be in the harness's snake_case shape:
        {input_tokens, output_tokens, cached_input_tokens, reasoning_output_tokens}
    """
    if event_sink is None or not isinstance(usage, dict):
        return
    try:
        event_sink({"type": "turn_completed", "payload": {"usage": dict(usage)}})
    except Exception:
        pass


def emit_agent_message(
    event_sink: Callable[[dict], None] | None, message: str
) -> None:
    """Emit `{"type": "agent_message", "payload": {"message": str}}`."""
    if event_sink is None or not message:
        return
    try:
        event_sink({"type": "agent_message", "payload": {"message": message}})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shared env-var injection
# ---------------------------------------------------------------------------

def build_tool_env(
    *, base_env: dict[str, str] | None, workspace_dir: str, input_payload: Any,
    agent_id: str = "",
) -> dict[str, str]:
    """Compose the env dict passed to `exec_bash` so the agent's shell
    finds TICKET_ID / AGENT_ID / RUN_ID / ZEVO_API_BASE / WORK_DIR.

    AGENT_ID is taken from the explicit `agent_id` arg (the blueprint id) —
    the task-input schemas only carry `ticket_id`, NOT agent_id/run_id, so
    without this AGENT_ID would be empty and the agent's `curl
    "author":"$AGENT_ID"` messages post a blank author (which the message
    endpoint then mistakes for a human follow-up and re-wakes the ticket).
    """
    env = {**(base_env or os.environ)}
    env["WORK_DIR"] = workspace_dir
    env.setdefault("ZEVO_API_BASE", env.get("ZEVO_API_BASE", "http://backend:8000"))
    for attr, key in (
        ("ticket_id", "TICKET_ID"),
        ("agent_id", "AGENT_ID"),
        ("run_id", "RUN_ID"),
    ):
        v = getattr(input_payload, attr, "")
        if v:
            env[key] = str(v)
    # The blueprint id is the authoritative agent_id (input_payload lacks it).
    if agent_id:
        env["AGENT_ID"] = str(agent_id)
    # run_id isn't in the typed inputs either — derive it from the per-run work
    # dir (<root>/<run-id>/<ticket>), so $RUN_ID is set for the agent's shell
    # (used to build the per-run REMOTE work dir). Without this it was empty and
    # agents fell back to $TICKET_ID (e.g. remote dir <remote>/infra-002).
    if not env.get("RUN_ID"):
        from pathlib import Path as _P
        env["RUN_ID"] = _P(workspace_dir).parent.name
    # A hosted deployment gates the API behind a shared service token. The
    # playbooks' callbacks add `-H "$ZEVO_API_AUTH_HEADER"` when it is set, so
    # the header name lives here once instead of in every agent's guesswork
    # (a held-out Data agent tried Bearer/X-Service-Token/X-API-Key and gave
    # up). Unset on an open install, where the expansion is empty.
    token = env.get("ZEVO_SERVICE_TOKEN", "").strip()
    if token and not env.get("ZEVO_API_AUTH_HEADER"):
        env["ZEVO_API_AUTH_HEADER"] = f"X-Zevo-Service: {token}"
    return env


__all__ = [
    "BASH_TOOL", "READ_FILE_TOOL", "WRITE_FILE_TOOL", "SKILL_TOOL", "BUILTIN_TOOLS",
    "exec_bash", "exec_read", "exec_write",
    "emit_tool_call", "emit_tool_result", "emit_turn_completed", "emit_agent_message",
    "build_tool_env",
    "_new_item_id",
]
