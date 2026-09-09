"""Driver backed by the OpenAI Codex CLI (subprocess).

Uses the `codex` binary from `@openai/codex` in its non-interactive
`exec` mode. Both container images already install it and log it in at
boot; this is the piece that was missing.

Real invocation (codex-cli 0.14x):
    codex exec \
        --json                      # JSONL events to stdout
        --model <model>             # e.g. gpt-5-codex
        --cd <workspace>            # agent's working root
        --skip-git-repo-check       # a run's work dir is not a git repo
        --dangerously-bypass-approvals-and-sandbox  # non-interactive permission mode
        --output-schema <file>      # constrain the final JSON to our schema
        -                           # read the prompt from stdin

Auth precedence (resolved by the image's entrypoint, checked here):
  1. A Codex-managed ChatGPT session in $CODEX_HOME/auth.json. Compose
     seeds the complete file from a host that ran `codex login`; never
     extract its browser-session JWT into an environment variable.
  2. OPENAI_API_KEY -- optional per-call fallback when no cache exists.
  3. Neither -> raise, naming both supported paths.

`--output-schema` is the one real advantage over the Claude CLI here:
the typed Result contract goes to the model as an actual JSON Schema, so
the final message is constrained rather than merely asked for. We still
parse defensively, because the flag binds the LAST message and an agent
that stops early leaves nothing to parse.

Cancel: registers the subprocess in process_registry under BOTH the
heartbeat and the ticket, matching what the cancel paths look up.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from zevo.engine.run import process_registry
from zevo.engine.agent.loader import AgentBlueprint
from zevo.engine.agent.drivers._json_utils import extract_trailing_json
from zevo.engine.agent.drivers._subprocess_stream import iter_subprocess_lines
from zevo.engine.agent.drivers._bash_description import describe_bash_call
from zevo.engine.agent.drivers.base import DriverRunResult
from zevo.engine.observe.markers import parse_line as _parse_marker
from zevo.engine.observe.markers import scan_text as _scan_markers
from zevo.engine.agent.skills import stage_skills, unstage_skills


CODEX_BIN_ENV = "ZEVO_CODEX_BIN"
DEFAULT_CODEX_BIN = "codex"


def _warn(where: str, exc: BaseException) -> None:
    print(f"[codex_cli.{where}] WARN: {type(exc).__name__}: {exc}",
          file=sys.stderr, flush=True)


def _resolve_auth(env: dict[str, str]) -> tuple[str, str]:
    """(mode, detail) for the operator, so a run says which wallet it spent.

    Mirrors the order the images' entrypoints log in with, so what this
    reports is what the binary will actually use.
    """
    from zevo.engine.agent.drivers.codex_auth import codex_auth_mode

    home = env.get("CODEX_HOME") or str(Path(env.get("HOME", "/root")) / ".codex")
    cache = Path(home, "auth.json")
    mode = codex_auth_mode(cache)
    if mode == "chatgpt":
        return "chatgpt_plan", f"{home}/auth.json"
    if mode == "api_key":
        return "api_key", f"{home}/auth.json (per-call billing)"
    if mode == "invalid":
        return "invalid", f"{home}/auth.json"
    if env.get("OPENAI_API_KEY"):
        return "api_key", "OPENAI_API_KEY (per-call billing)"
    return "none", ""


def _normalise_codex_event(obj: dict) -> list[dict]:
    """Map one `codex exec --json` event onto the harness event contract.

    Codex nests the interesting part under `msg` with its own `type`
    vocabulary. Anything unrecognised is passed through as `raw` rather
    than dropped -- a transcript that silently omits events is worse
    than one with a line we did not style.
    """
    # Codex <=0.146 nested events under ``msg``. Codex 0.147 switched to
    # thread/turn/item events, with the useful payload under ``item``. Accept
    # both so upgrading the CLI does not turn every transcript into raw JSON.
    event_type = str(obj.get("type") or "")
    item = obj.get("item") if isinstance(obj.get("item"), dict) else None
    if event_type in ("item.started", "item.updated", "item.completed") and item:
        item_kind = str(item.get("type") or "")
        if item_kind == "agent_message":
            text = str(item.get("text") or "")
            return _agent_message_events(text)
        if item_kind == "reasoning":
            text = str(item.get("text") or "")
            return ([{"type": "reasoning", "payload": {"text": text}}]
                    if text else [])
        if item_kind == "command_execution":
            phase = {
                "item.started": "begin",
                "item.updated": "output",
                "item.completed": "end",
            }[event_type]
            command = str(item.get("command") or "")
            return [{"type": "tool_call", "payload": {
                "name": "run_bash",
                "phase": phase,
                "command": command,
                "description": describe_bash_call(command),
                "description_source": "fallback",
                "exit_code": item.get("exit_code"),
                "text": str(item.get("aggregated_output") or ""),
            }}]

    if event_type == "turn.completed":
        usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
        return [{"type": "turn_completed", "payload": {"usage": usage}}]

    msg = obj.get("msg") if isinstance(obj.get("msg"), dict) else obj
    kind = str(msg.get("type") or obj.get("type") or "")

    if kind in ("agent_message", "agent_message_delta"):
        text = str(msg.get("message") or msg.get("delta") or "")
        return _agent_message_events(text)

    if kind in ("agent_reasoning", "agent_reasoning_delta",
                "agent_reasoning_raw_content", "agent_reasoning_raw_content_delta"):
        text = str(msg.get("text") or msg.get("delta") or "")
        return [{"type": "reasoning", "payload": {"text": text}}] if text else []

    if kind in ("exec_command_begin", "exec_command_end", "exec_command_output_delta"):
        cmd = msg.get("command")
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        text = str(msg.get("chunk") or msg.get("stdout") or "")
        command = str(cmd or "")
        out = [{"type": "tool_call", "payload": {
            "name": "run_bash",
            "phase": "begin" if kind.endswith("begin") else
                     "end" if kind.endswith("end") else "output",
            "command": command,
            "description": describe_bash_call(command),
            "description_source": "fallback",
            "exit_code": msg.get("exit_code"),
            "text": text,
        }}]
        # `codex exec --json` puts command output inside JSONL events, so the
        # non-JSON marker path below never sees it. Scan the output deltas for
        # __ATTEMPT__/__PHASE__/__PROGRESS__/__CONFIG__ so markers stream live (mirrors how
        # claude_cli scans tool-output text). Deltas ONLY: exec_command_end
        # carries the aggregated stdout and would re-emit every marker.
        if kind == "exec_command_output_delta" and text:
            for mkind, data in _scan_markers(text):
                if mkind == "attempt":
                    out.append({"type": "attempt", "payload": data})
                elif mkind == "phase":
                    out.append({"type": "phase", "payload": data})
                elif mkind == "progress":
                    out.append({"type": "progress", "payload": data})
                elif mkind == "config":
                    out.append({"type": "config", "payload": data})
        return out

    if kind in ("patch_apply_begin", "patch_apply_end", "turn_diff"):
        return [{"type": "tool_call", "payload": {
            "name": "apply_patch", "phase": kind, "text": str(msg.get("unified_diff") or ""),
        }}]

    if kind == "mcp_tool_call_begin" or kind == "mcp_tool_call_end":
        return [{"type": "tool_call", "payload": {
            "name": str((msg.get("invocation") or {}).get("tool") or "mcp"),
            "phase": "begin" if kind.endswith("begin") else "end",
        }}]

    if kind in ("token_count", "task_started", "task_complete", "turn_complete"):
        return [{"type": kind, "payload": msg}]

    if kind == "error" or kind == "stream_error":
        return [{"type": "error", "payload": {"text": str(msg.get("message") or msg)}}]

    return [{"type": "raw", "payload": {"text": json.dumps(obj)[:2000]}}]


def _agent_message_events(text: str) -> list[dict]:
    """Map Codex assistant text without deciding whether it is final.

    Codex does not label an ``agent_message`` as progress versus final output.
    That distinction can only be made after the following stream event arrives,
    so :class:`_CodexTranscriptGate` owns it. Raw stdout remains untouched for
    final Result extraction and validation.
    """
    if not text.strip():
        return []
    # `message`, not `text`: the canonical agent_message payload key every
    # other driver emits and the frontend StepTimeline reads.
    return [{"type": "agent_message", "payload": {"message": text}}]


_CODEX_RESULT_ENVELOPE_KEYS = frozenset({
    "status", "ticket_id", "error_message",
})
_CODEX_TERMINAL_EVENT_TYPES = frozenset({
    "turn_completed", "turn_complete", "task_complete",
})


def _intermediate_agent_message_events(event: dict) -> list[dict]:
    """Turn one known-intermediate Codex message into transcript narration.

    With ``codex exec --output-schema``, the model may wrap progress narration
    in the final Result schema and therefore populate the required ``status``
    with ``failed`` even though no operation failed. Once a later stream event
    proves such a Result-shaped message was intermediate, omit it completely:
    its status is not authoritative and interpreting its prose would make the
    UI depend on model wording. Plain progress narration remains visible. Real
    command failures arrive as ``tool_call`` events and never pass through this
    filter.
    """
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    text = str(payload.get("message") or "")
    if not text.strip():
        return []

    try:
        candidate = json.loads(text)
    except json.JSONDecodeError:
        candidate = None
    if (
        isinstance(candidate, dict)
        and _CODEX_RESULT_ENVELOPE_KEYS.issubset(candidate)
    ):
        return []

    return [{"type": "agent_message", "payload": {"message": text}}]


class _CodexTranscriptGate:
    """Separate progress narration from the final structured Result.

    An assistant message is held until the next event resolves its role:

    * another message, reasoning, or tool activity means it was intermediate;
    * a terminal turn event means it was the final Result and it is omitted
      from the transcript (raw stdout still supplies it to ``_final_message``).

    This state belongs to one subprocess stream, not to a Ticket or Run.
    """

    def __init__(self) -> None:
        self._pending_agent_message: dict | None = None

    def feed(self, event: dict) -> list[dict]:
        event_type = str(event.get("type") or "")

        if event_type == "agent_message":
            visible = self._flush_intermediate()
            self._pending_agent_message = event
            return visible

        if event_type in _CODEX_TERMINAL_EVENT_TYPES:
            # The pending message immediately preceding turn completion is the
            # typed final Result. Do not duplicate it in the transcript.
            self._pending_agent_message = None
            return [event]

        visible = self._flush_intermediate()
        visible.append(event)
        return visible

    def finish(self) -> None:
        """Drop an unresolved last message; final parsing uses raw stdout."""
        self._pending_agent_message = None

    def _flush_intermediate(self) -> list[dict]:
        if self._pending_agent_message is None:
            return []
        pending = self._pending_agent_message
        self._pending_agent_message = None
        return _intermediate_agent_message_events(pending)


def _line_to_events(line: str) -> list[dict]:
    """One raw stdout line -> zero or more structured events."""
    s = line.rstrip("\n")
    if not s.strip():
        return []
    if s.lstrip().startswith("{"):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            return _normalise_codex_event(obj)

    # The agents print __ATTEMPT__ / __PHASE__ / __PROGRESS__ / __CONFIG__ markers on
    # stdout; those drive the live progress bar and matter more than the
    # model's prose.
    marker = _parse_marker(s)
    if marker is not None:
        kind, data = marker
        if kind == "attempt":
            return [{"type": "attempt", "payload": data}]
        if kind == "phase":
            return [{"type": "phase", "payload": data}]
        if kind == "progress":
            return [{"type": "progress", "payload": data}]
        if kind == "config":
            return [{"type": "config", "payload": data}]

    return [{"type": "raw", "payload": {"text": s}}]


def _final_message(raw_stdout: str) -> str:
    """The agent's last assistant message, from the JSONL stream."""
    for line in reversed(raw_stdout.splitlines()):
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        msg = obj.get("msg") if isinstance(obj.get("msg"), dict) else obj
        if msg.get("type") == "agent_message" and msg.get("message"):
            return str(msg["message"])
        item = obj.get("item") if isinstance(obj.get("item"), dict) else None
        if (
            obj.get("type") == "item.completed"
            and item
            and item.get("type") == "agent_message"
            and item.get("text")
        ):
            return str(item["text"])
    return raw_stdout


def _codex_output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return the Pydantic schema in Codex Structured Outputs strict form.

    Pydantic omits fields with defaults from an object's ``required`` array.
    The Responses API used by ``codex exec --output-schema`` instead requires
    every property to be listed, including fields whose semantic default is an
    empty string/list, and every object to reject additional properties. Apply
    that rule recursively, including objects in ``$defs``, while leaving the
    model's own cached schema untouched. Reject an open mapping locally: simply
    closing one would silently turn it into an object that can only be empty.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def visit(node: Any, path: str = "$") -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties)
                node["additionalProperties"] = False
            elif (
                node.get("type") == "object"
                and node.get("additionalProperties") is not False
            ):
                raise ValueError(
                    "Codex strict output schema cannot contain an open-ended "
                    f"object at {path}; replace dict[str, Any] with a typed model "
                    "or a list of typed key/value entries"
                )
            for key, value in node.items():
                visit(value, f"{path}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{path}/{index}")

    visit(schema)
    return schema


class CodexCliDriver:
    name = "codex_cli"

    def _resolve_bin(self) -> str:
        override = os.environ.get(CODEX_BIN_ENV, "").strip()
        if override:
            return override
        found = shutil.which(DEFAULT_CODEX_BIN)
        if not found:
            raise RuntimeError(
                "codex_cli driver requested but the `codex` binary is not on "
                f"PATH. Install @openai/codex, set {CODEX_BIN_ENV}, or pick "
                "another driver."
            )
        return found

    async def run_agent(
        self,
        *,
        blueprint: AgentBlueprint,
        input_payload: BaseModel,
        workspace_dir: str,
        stdout_sink: Callable[[str], None] | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        conversation: list[dict[str, Any]] | None = None,
        max_turns: int = 60,
        model_override: str = "",
        sandbox_mode: str = "",
    ) -> DriverRunResult:
        # Accepted for interface parity only -- this driver has no sandbox
        # runtime. Say so instead of silently ignoring a requested mode.
        if sandbox_mode and sandbox_mode != "none":
            raise ValueError("codex_cli does not support OpenShell; use claude_cli")
        codex_bin = self._resolve_bin()
        model = model_override or blueprint.default_model
        Path(workspace_dir).mkdir(parents=True, exist_ok=True)

        # Codex scans <cwd>/.agents/skills for repository-scoped Skills. This
        # differs from Claude Code's <cwd>/.claude/skills discovery root.
        stage_skills(blueprint.id, workspace_dir, layout="codex")
        try:

            from zevo.engine.agent.drivers._prompt import render_agent_prompt
            prompt = render_agent_prompt(
                blueprint, input_payload, conversation,
                strict_json=True, schema_inline=True,
            )
            prompt += (
                "\n\n## CODEX CLI EXECUTION DISCIPLINE\n\n"
                "Use tools directly while executing. Do not emit provisional "
                "Result objects, and do not narrate channel, schema, or response-"
                "format self-corrections. Emit exactly one structured Result: "
                "your final assistant message after all execution and validation."
            )

            env = {**os.environ}
            env.setdefault("HOME", "/root")
            env["WORK_DIR"] = workspace_dir
            env.setdefault("ZEVO_API_BASE", env.get("ZEVO_API_BASE", "http://backend:8000"))
            for attr, key in (("ticket_id", "TICKET_ID"),
                              ("agent_id", "AGENT_ID"),
                              ("run_id", "RUN_ID")):
                v = getattr(input_payload, attr, "")
                if v:
                    env[key] = str(v)
            env.setdefault("AGENT_ID", blueprint.id)
            if not env.get("RUN_ID"):
                env["RUN_ID"] = Path(workspace_dir).parent.name

            auth_mode, auth_detail = _resolve_auth(env)
            if auth_mode in ("none", "invalid"):
                prefix = (
                    "codex_cli found an invalid auth.json; remove it and log in again.\n"
                    if auth_mode == "invalid"
                    else "codex_cli requires one of:\n"
                )
                raise RuntimeError(
                    prefix
                    + "  (a) A complete Codex login cache at $CODEX_HOME/auth.json. "
                    "Run `codex login` on a trusted machine and mount or seed the "
                    "file; do not extract an individual token from it.\n"
                    "  (b) OPENAI_API_KEY in .env -- usage-billed fallback.\n"
                    f"Looked at CODEX_HOME={env.get('CODEX_HOME') or '$HOME/.codex'!r}."
                )

            # The typed Result contract, handed over as a real JSON Schema. The
            # file has to outlive the subprocess, so it is cleaned up in the
            # finally rather than by a context manager around the spawn.
            schema_path = ""
            if blueprint.output_schema is not None:
                fd, schema_path = tempfile.mkstemp(suffix=".schema.json", prefix="zevo-")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(_codex_output_schema(blueprint.output_schema), f)

            cmd = [
                codex_bin, "exec",
                "--json",
                "--cd", workspace_dir,
                # A run's work dir is a plain directory. Without this codex
                # refuses to start outside a git repo.
                "--skip-git-repo-check",
                # The container is the execution environment. This flag makes
                # the non-interactive driver usable; it does not claim a
                # per-Ticket filesystem boundary. The agent's whole job is to
                # write files and shell out to a GPU box.
                "--dangerously-bypass-approvals-and-sandbox",
            ]
            if model:
                cmd.extend(["--model", model])
            if schema_path:
                cmd.extend(["--output-schema", schema_path])
            cmd.append("-")  # prompt on stdin

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=workspace_dir,
                start_new_session=True,
            )

            # Both keys: the heartbeat is what the heartbeat-cancel endpoint
            # knows, the ticket is what the runner's flusher knows.
            hb_id = env.get("HEARTBEAT_ID", "")
            tk_id = env.get("TICKET_ID", "")
            for key in (hb_id, tk_id):
                if key:
                    process_registry.register(key, proc)

            if proc.stdin is None:
                raise RuntimeError("codex subprocess has no stdin")
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []
            transcript_gate = _CodexTranscriptGate()

            if event_sink is not None:
                try:
                    event_sink({"type": "codex_auth", "payload": {
                        "mode": auth_mode, "detail": auth_detail,
                        "model": model or "(default)",
                    }})
                except Exception as e:
                    _warn("event_sink.auth", e)

            async def _drain(stream: asyncio.StreamReader, is_stdout: bool) -> None:
                async for line in iter_subprocess_lines(stream):
                    text = line.decode("utf-8", errors="replace")
                    (stdout_chunks if is_stdout else stderr_chunks).append(text)
                    if is_stdout and stdout_sink is not None:
                        stdout_sink(text)
                    if event_sink is None:
                        continue
                    try:
                        if is_stdout:
                            for ev in _line_to_events(text):
                                for visible in transcript_gate.feed(ev):
                                    event_sink(visible)
                        else:
                            event_sink({"type": "stderr", "payload": {"text": text}})
                    except Exception as e:
                        _warn("event_sink", e)

            try:
                await asyncio.gather(_drain(proc.stdout, True), _drain(proc.stderr, False))
                exit_code = await proc.wait()
            finally:
                transcript_gate.finish()
                for key in (hb_id, tk_id):
                    if key:
                        process_registry.unregister(key, proc)
                if schema_path:
                    try:
                        os.unlink(schema_path)
                    except OSError:
                        pass
        finally:
            # Every exit path -- auth raise, spawn failure, cancel, success --
            # removes the staged <workspace>/.agents/skills copy.
            unstage_skills(workspace_dir, layout="codex")

        raw_stdout = "".join(stdout_chunks)
        raw_stderr = "".join(stderr_chunks)
        final_message = _final_message(raw_stdout)

        if exit_code != 0:
            if exit_code < 0 or exit_code == 143:
                raise RuntimeError(
                    f"codex exec cancelled (signal "
                    f"{-exit_code if exit_code < 0 else 'SIGTERM'}) "
                    f"for agent {blueprint.id}"
                )
            # stderr may contain a harmless warning before the JSON event
            # stream reports the real API failure. Include both, with stdout
            # last, so the actionable turn.failed message survives the tail.
            err_text = "\n".join(x for x in (raw_stderr, final_message) if x)
            err_tail = err_text[-1000:].strip()
            raise RuntimeError(
                f"codex exec failed (exit={exit_code}) for agent "
                f"{blueprint.id}: {err_tail}"
            )

        if blueprint.output_schema is None:
            class _Empty(BaseModel):
                pass
            return DriverRunResult(
                output=_Empty(), raw_stdout=raw_stdout, raw_stderr=raw_stderr,
                exit_code=exit_code, driver=self.name, model=model,
            )

        # --output-schema constrains the last message, but an agent that
        # stops early never sends one, so this still has to fail loudly
        # rather than assume.
        data = extract_trailing_json(
            final_message, min_keys=2, output_schema=blueprint.output_schema,
        )
        if data is None:
            raise ValueError(
                f"codex exec produced no parseable trailing JSON for agent "
                f"{blueprint.id}. Final message tail:\n{final_message[-800:]}"
            )
        return DriverRunResult(
            output=blueprint.output_schema.model_validate(data),
            raw_stdout=raw_stdout, raw_stderr=raw_stderr,
            exit_code=exit_code, driver=self.name, model=model,
        )
