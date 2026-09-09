"""Driver backed by the Anthropic Claude Code CLI (subprocess).

Uses the `claude` binary from `@anthropic-ai/claude-code` in headless /
print mode.

Real invocation (claude-code 2.x):
    claude \
        -p                            # print mode (non-interactive)
        --output-format stream-json   # JSONL events to stdout
        --verbose                     # required by stream-json
        --include-partial-messages    # stream thinking text as content_block deltas
        --forward-subagent-text       # surface nested subagent text too
        --add-dir <workspace>         # let it read/write under here
        --dangerously-skip-permissions   # non-interactive CLI permission mode
        --model <model>               # e.g. claude-sonnet-4-5
        -                                # read prompt from stdin

Auth precedence is a Zevo policy, resolved and enforced at spawn time:
  1. CLAUDE_CODE_OAUTH_TOKEN, then an interactive OAuth session.
  2. ANTHROPIC_AUTH_TOKEN.
  3. ANTHROPIC_API_KEY.
  4. Neither -> raise with a clear message pointing at the supported paths.

Lower-priority credentials are removed from the child environment so the
reported auth mode and the credential Claude actually uses cannot diverge.

Cancel: registers the subprocess in process_registry by HEARTBEAT_ID
so `POST /tickets/{id}/cancel` -> SIGTERM works.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from zevo.engine.agent.loader import AgentBlueprint
from zevo.engine.agent.drivers.base import DriverRunResult
from zevo.engine.agent.skills import stage_skills, unstage_skills
from zevo.engine.agent.drivers.sandbox import (
    configure_sandbox_env, open_sandbox, close_sandbox, sandbox_exec_argv,
)
from zevo.engine.agent.drivers._json_utils import extract_trailing_json
from zevo.engine.agent.drivers._bash_description import (
    describe_bash_call,
    is_bash_tool,
)
from zevo.engine.agent.drivers._subprocess_stream import iter_subprocess_lines
from zevo.engine.observe.markers import parse_line as _parse_marker
from zevo.engine.observe.markers import scan_text as _scan_markers


def _warn(where: str, exc: BaseException) -> None:
    print(f"[claude_cli.{where}] WARN: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
from zevo.engine.run import process_registry


CLAUDE_BIN_ENV = "ZEVO_CLAUDE_BIN"
DEFAULT_CLAUDE_BIN = "claude"


def _find_claude_bin() -> str:
    explicit = os.environ.get(CLAUDE_BIN_ENV, "").strip()
    if explicit:
        return explicit
    found = shutil.which(DEFAULT_CLAUDE_BIN)
    if not found:
        raise FileNotFoundError(
            f"`claude` not found on PATH and ${CLAUDE_BIN_ENV} not set. "
            "Install the Anthropic Claude Code CLI "
            "(`npm i -g @anthropic-ai/claude-code`) or set ZEVO_CLAUDE_BIN "
            "to its absolute path."
        )
    return found


def _resolve_auth(env: dict[str, str]) -> tuple[str, str]:
    """Select and enforce the auth mode for the spawned `claude`.

    Returns (mode, detail) -- mode is one of:
      'oauth_token'   long-lived CLAUDE_CODE_OAUTH_TOKEN env var,
                      issued by `claude setup-token` (Max / Pro plan)
      'oauth_session' interactive ~/.claude/.credentials.json session
                      from `claude /login` (Max / Pro plan)
      'auth_token'    ANTHROPIC_AUTH_TOKEN bearer token
      'api_key'       ANTHROPIC_API_KEY env var (per-call billing)
      'none'          no auth found

    `env` is the private child-process copy created by `run_agent`; sanitizing
    it never mutates `os.environ`.
    """
    if env.get("CLAUDE_CODE_OAUTH_TOKEN"):
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        env.pop("ANTHROPIC_API_KEY", None)
        return ("oauth_token", "CLAUDE_CODE_OAUTH_TOKEN env (subscription)")
    home = env.get("HOME", "/root")
    creds = Path(home) / ".claude" / ".credentials.json"
    if creds.exists():
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        env.pop("ANTHROPIC_API_KEY", None)
        return ("oauth_session", f"oauth session at {creds}")
    if env.get("ANTHROPIC_AUTH_TOKEN"):
        env.pop("ANTHROPIC_API_KEY", None)
        return ("auth_token", "ANTHROPIC_AUTH_TOKEN env var")
    if env.get("ANTHROPIC_API_KEY"):
        return ("api_key", "ANTHROPIC_API_KEY env var (per-call billing)")
    return ("none", "")


# tool_use_id -> tool name, remembered from the assistant block so the matching
# tool_result can be judged by WHICH tool produced it. Ids are unique per call,
# so sharing one map across concurrent heartbeats is safe; the cap is only to
# stop a long session growing it without bound.
_TOOL_BY_USE_ID: dict[str, str] = {}
_TOOL_MAP_CAP = 4096

# Results from these are the file the agent just wrote, echoed back -- not the
# output of anything that ran. Scanning them attributed a script's OWN
# `print("__PHASE__:<ticket>:saving_model@<time>")` lines to the agent as phases it had
# reached, so an inference stage reported `reformatting` and `saving_model`
# (a data phase and a train phase) simply because it wrote a script mentioning
# them. Bash output, where remote runs actually emit their markers, is
# unaffected.
_NON_EMITTING_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit", "Update"}


# --include-partial-messages makes the CLI stream extended-thinking text
# delta-by-delta as `stream_event` lines (Anthropic's raw content_block_*
# events) BEFORE the complete assistant message. We accumulate thinking_delta
# text per content-block index and flush one `reasoning` event when the block
# stops. _THINKING_EMITTED remembers flushed thinking text so the COMPLETE
# thinking block the CLI still delivers at message end is not emitted twice.
# Shared module state, like _TOOL_BY_USE_ID above: lines are parsed in order
# within a run, and the buffer/seen entries are short-lived (cleared on flush /
# on the matching complete block), keeping cross-heartbeat interference tiny.
_THINKING_BUFFER: dict[int, list[str]] = {}
_THINKING_EMITTED: set[str] = set()
_THINKING_EMITTED_CAP = 512


def _remember_thinking(text: str) -> None:
    """Record flushed thinking text so the later complete block dedupes."""
    if len(_THINKING_EMITTED) >= _THINKING_EMITTED_CAP:
        _THINKING_EMITTED.clear()
    _THINKING_EMITTED.add(text.strip())


def _normalise_stream_event(event: dict) -> list[dict]:
    """Handle one partial streaming event from --include-partial-messages.

    These are Anthropic's raw streaming events wrapped in a `stream_event`
    envelope. Only extended-thinking deltas produce a transcript event: we
    accumulate `thinking_delta` text per content-block index and flush a single
    `reasoning` event on `content_block_stop`. Text/tool deltas and every other
    partial shape are swallowed here — the COMPLETE assistant message that
    follows still carries the final text and tool_use blocks, so surfacing the
    partials too would duplicate agent_message / tool_call output.
    """
    et = event.get("type")
    out: list[dict] = []

    if et == "content_block_start":
        block = event.get("content_block") or {}
        if block.get("type") == "thinking":
            idx = int(event.get("index", 0) or 0)
            seed = block.get("thinking", "") or ""
            _THINKING_BUFFER[idx] = [seed] if seed else []
        return out

    if et == "content_block_delta":
        delta = event.get("delta") or {}
        if delta.get("type") == "thinking_delta":
            idx = int(event.get("index", 0) or 0)
            piece = delta.get("thinking", "") or ""
            if piece:
                _THINKING_BUFFER.setdefault(idx, []).append(piece)
        # signature_delta / text_delta / input_json_delta: nothing to surface.
        return out

    if et == "content_block_stop":
        idx = int(event.get("index", 0) or 0)
        parts = _THINKING_BUFFER.pop(idx, None)
        if parts is not None:
            text = "".join(parts)
            if text.strip():
                _remember_thinking(text)
                out.append({"type": "reasoning", "payload": {"text": text}})
        return out

    # message_start / message_delta / message_stop / anything else: the complete
    # assistant + result events drive the rest of the transcript, so swallow.
    return out


def _normalise_claude_event(obj: dict) -> list[dict]:
    """Map one Claude stream-json line to canonical events.

    Claude emits:
      {"type":"system","subtype":"init",...}        -> swallow
      {"type":"assistant","message":{content:[...]}} -> agent_message + tool_call
      {"type":"user","message":{content:[...tool_result...]}} -> tool_result
      {"type":"result","subtype":"success","result":"...",...,"usage":{...}}
                                                     -> turn_completed + agent_message (final)
      {"type":"error",...}                            -> raw / stderr-ish

    Returns a list because one assistant message often carries multiple
    content blocks (text + tool_use mixed).
    """
    t = obj.get("type")
    out: list[dict] = []

    if t == "stream_event":
        # Partial streaming envelope from --include-partial-messages. The real
        # Anthropic event is nested under "event"; some builds put it at the top
        # level, so fall back to obj itself.
        return _normalise_stream_event(obj.get("event") or obj)

    if t == "system":
        # init / status / config events -- not useful to the user, just
        # surface them as a small event so they appear in the
        # raw view if needed.
        return [{"type": f"claude_{t}_{obj.get('subtype','')}", "payload": obj}]

    if t == "assistant":
        msg = obj.get("message") or {}
        content = msg.get("content") or []
        for block in content:
            bt = block.get("type")
            if bt == "text":
                text = block.get("text", "")
                if text.strip():
                    out.append({"type": "agent_message", "payload": {"message": text}})
            elif bt == "tool_use":
                use_id = str(block.get("id", ""))
                tool_name = str(block.get("name", ""))
                tool_input = (
                    dict(block.get("input") or {})
                    if isinstance(block.get("input"), dict) else {}
                )
                if use_id:
                    if len(_TOOL_BY_USE_ID) >= _TOOL_MAP_CAP:
                        _TOOL_BY_USE_ID.clear()
                    _TOOL_BY_USE_ID[use_id] = tool_name
                payload = {
                    "tool": tool_name,
                    # Preserve the exact model-authored input. Description
                    # metadata is additive and never rewrites this record.
                    "input": tool_input,
                    "tool_use_id": block.get("id", ""),
                }
                if is_bash_tool(tool_name):
                    supplied = str(tool_input.get("description") or "").strip()
                    payload["description"] = describe_bash_call(
                        tool_input.get("command", ""), supplied,
                    )
                    payload["description_source"] = (
                        "agent" if supplied else "fallback"
                    )
                out.append({
                    "type": "tool_call",
                    "payload": payload,
                })
            elif bt == "thinking":
                text = block.get("thinking", "") or block.get("text", "")
                if text.strip():
                    # If partial thinking_delta streaming already flushed this
                    # block's reasoning (--include-partial-messages), don't emit
                    # it a second time. Discard the marker so the set stays small.
                    if text.strip() in _THINKING_EMITTED:
                        _THINKING_EMITTED.discard(text.strip())
                    else:
                        out.append({"type": "reasoning", "payload": {"text": text}})
        # Capture per-message usage if present (some Claude builds emit
        # usage incrementally on the assistant message instead of only
        # on the final result event).
        usage = msg.get("usage")
        if isinstance(usage, dict):
            out.append({"type": "turn_completed", "payload": {
                "usage": _normalise_usage(usage),
            }})
        return out

    if t == "user":
        # User-role messages in stream-json carry tool_result blocks
        # back from Claude (mirrors the API's user role).
        msg = obj.get("message") or {}
        content = msg.get("content") or []
        for block in content:
            if block.get("type") == "tool_result":
                content_val = block.get("content", "")
                if isinstance(content_val, list):
                    # content can be [{"type":"text","text":"..."}]
                    text_parts = []
                    for c in content_val:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text_parts.append(c.get("text", ""))
                    content_val = "\n".join(text_parts)
                out.append({
                    "type": "tool_result",
                    "payload": {
                        "tool_use_id": block.get("tool_use_id", ""),
                        "output": content_val,
                        "is_error": block.get("is_error", False),
                    },
                })
                # Remote training (SSH) emits __PHASE__/__PROGRESS__ lines INSIDE
                # the bash tool output, not on the agent's own stdout. Scan them
                # here so the loss/phase chart works for remote runs too.
                tool_name = _TOOL_BY_USE_ID.pop(block.get("tool_use_id", ""), "")
                if (
                    isinstance(content_val, str)
                    # Cheap prefilter before the regex scan. "__P" covers
                    # __PHASE__/__PROGRESS__; __CONFIG__/__ATTEMPT__ must be checked
                    # separately or a result whose only marker is a config
                    # line is never scanned.
                    and (
                        "__P" in content_val
                        or "__CONFIG__" in content_val
                        or "__ATTEMPT__" in content_val
                    )
                    and tool_name not in _NON_EMITTING_TOOLS
                ):
                    # Scan the WHOLE result, every marker, in order. A remote
                    # stage's log comes back with its newlines escaped, so it is
                    # one physical line carrying hundreds of markers; keeping one
                    # per line kept 2 of 31 here, and never a __PHASE__.
                    for kind, data in _scan_markers(content_val):
                        if kind == "attempt":
                            out.append({"type": "attempt", "payload": data})
                        elif kind == "phase":
                            out.append({"type": "phase", "payload": data})
                        elif kind == "progress":
                            out.append({"type": "progress", "payload": data})
                        elif kind == "config":
                            out.append({"type": "config", "payload": data})
        return out

    if t == "result":
        # Final event of a session. Carry the result text as one last
        # agent_message + a turn_completed with usage so the cost meter
        # works.
        out.append({"type": "turn_completed", "payload": {
            "usage": _normalise_usage(obj.get("usage") or {}),
        }})
        result_text = obj.get("result") or ""
        if isinstance(result_text, str) and result_text.strip():
            # We DON'T emit this as agent_message because the runner has
            # its own "extract trailing JSON" pass over raw_stdout; if
            # this final text contains the *Result Pydantic, that flow
            # picks it up. We do emit a lightweight task_complete event
            # so the UI knows the session ended.
            out.append({"type": "task_complete", "payload": {
                "subtype": obj.get("subtype", ""),
                "result_preview": result_text[:500],
            }})
        return out

    if t == "error":
        return [{"type": "stderr", "payload": {"text": obj.get("message") or json.dumps(obj)}}]

    # Unknown -- pass through so nothing disappears.
    return [{"type": f"claude_{t or 'unknown'}", "payload": obj}]


def _normalise_usage(u: dict) -> dict:
    """Convert Claude usage keys to the canonical keys our runner
    accumulates on. Claude uses cache_read/cache_creation; we fold them
    into cached_input_tokens for the harness's pricing.estimate_cost.
    """
    inp = int(u.get("input_tokens") or 0)
    out = int(u.get("output_tokens") or 0)
    cache_read = int(u.get("cache_read_input_tokens") or 0)
    cache_create = int(u.get("cache_creation_input_tokens") or 0)
    # Claude reports input_tokens EXCLUDING cache; our cost code
    # expects input_tokens INCLUDING cache (then subtracts cached at the
    # cached-rate), so add them.
    total_in = inp + cache_read + cache_create
    cached = cache_read + cache_create
    return {
        "input_tokens": total_in,
        "output_tokens": out,
        "cached_input_tokens": cached,
        "reasoning_output_tokens": 0,
    }


def _line_to_events(line: str) -> list[dict]:
    """Convert one raw stdout line to zero or more structured events."""
    s = line.rstrip("\n")
    if not s.strip():
        return []

    if s.lstrip().startswith("{"):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            return _normalise_claude_event(obj)

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


def _render_prompt(
    blueprint: AgentBlueprint,
    input_payload: BaseModel,
    conversation: list[dict] | None = None,
) -> str:
    """Delegates to the shared renderer. Claude needs schema_inline=True
    because the model drifts off strict JSON contracts without seeing
    the actual JSON Schema body (smoke-verified: was 9 ValidationErrors
    → 0 once we inlined). Strict JSON rules on."""
    from zevo.engine.agent.drivers._prompt import render_agent_prompt
    return render_agent_prompt(
        blueprint, input_payload, conversation,
        strict_json=True, schema_inline=True,
    )


class ClaudeCliDriver:
    name = "claude_cli"

    def __init__(self, claude_bin: str | None = None):
        self._claude_bin_override = claude_bin

    def _resolve_bin(self) -> str:
        if self._claude_bin_override:
            return self._claude_bin_override
        return _find_claude_bin()

    async def run_agent(
        self,
        *,
        blueprint: AgentBlueprint,
        input_payload: BaseModel,
        workspace_dir: str,
        stdout_sink: Callable[[str], None] | None = None,
        event_sink: Callable[[dict], None] | None = None,
        conversation: list[dict] | None = None,
        max_turns: int = 60,
        model_override: str = "",
        sandbox_mode: str = "",
    ) -> DriverRunResult:
        claude_bin = self._resolve_bin()
        # Claude Code uses its own model id format (claude-sonnet-4-6,
        # claude-opus-4-8, etc.). The agent's default_model may be a
        # non-Claude id; we pass through but warn in events if it doesn't
        # look like a Claude id.
        model = model_override or blueprint.default_model
        Path(workspace_dir).mkdir(parents=True, exist_ok=True)

        # Stage THIS agent's skills into <workspace>/.claude/skills/ so Claude
        # Code discovers them natively (it keys off the subprocess cwd, set to
        # workspace_dir below — `--add-dir` does NOT surface skills). Removed in
        # the finally after the run so it sees only its own stage's ZEVO skills.
        stage_skills(blueprint.id, workspace_dir)
        try:

            prompt = _render_prompt(blueprint, input_payload, conversation=conversation)

            # Auth resolution.
            env = {**os.environ}
            env.setdefault("HOME", "/root")
            # Claude Code refuses --dangerously-skip-permissions when
            # running as root for security reasons. Containers always run
            # as root, so we set IS_SANDBOX=1 to signal the binary that this
            # is an intentional containerized environment. This bypasses the
            # CLI's root check; it is not itself a filesystem security boundary.
            # (Per Claude Code's
            # own root-check bypass; see
            # https://github.com/anthropics/claude-code/issues, search for
            # IS_SANDBOX.)
            env.setdefault("IS_SANDBOX", "1")
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
            env.setdefault("AGENT_ID", blueprint.id)
            # run_id isn't in the typed inputs — derive it from the per-run work dir
            # (<root>/<run-id>/<ticket>) so $RUN_ID is set for the agent's shell.
            # Without this it was empty and agents fell back to $TICKET_ID, so the
            # remote work dir became <remote>/infra-005 instead of <remote>/<run-id>.
            # (Mirrors _agent_loop.build_tool_env, which the bedrock driver uses.)
            if not env.get("RUN_ID"):
                from pathlib import Path as _P
                env["RUN_ID"] = _P(workspace_dir).parent.name
            auth_mode, auth_detail = _resolve_auth(env)
            if auth_mode == "none":
                raise RuntimeError(
                    "claude_cli requires one of:\n"
                    "  (a) CLAUDE_CODE_OAUTH_TOKEN in .env  -- recommended for "
                    "Max/Pro; generate via `claude setup-token` on the host.\n"
                    "  (b) OAuth session at $HOME/.claude/.credentials.json "
                    "-- via `docker compose exec scheduler claude /login`.\n"
                    "  (c) ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY in .env "
                    "-- managed or usage-billed fallback.\n"
                    f"Looked at HOME={env.get('HOME')!r} for OAuth creds."
                )

            cmd = [
                claude_bin,
                "-p",
                "--output-format", "stream-json",
                "--verbose",
                # Stream extended-thinking text as partial content_block deltas
                # (parsed into `reasoning` events) instead of only opaque
                # thinking_tokens progress markers; forward subagent text too so
                # nested Task-tool reasoning/output reaches the transcript.
                "--include-partial-messages",
                "--forward-subagent-text",
                "--add-dir", workspace_dir,
                "--dangerously-skip-permissions",
            ]
            if model:
                cmd.extend(["--model", model])
            # Read prompt from stdin via '-'.
            cmd.append("-")

            # Optional per-agent sandbox (Feature 5: NVIDIA OpenShell). No-op in the
            # default `none` mode: open_sandbox returns None and cmd runs directly, as
            # before. In `openshell` mode we create a keepalive sandbox, UPLOAD this
            # workspace into it (skills already staged), and run the agent via
            # `sandbox exec` — stdin prompt + stream-json stdout still proxy through.
            # close_sandbox (in the finally below) downloads artifacts back + deletes.
            configure_sandbox_env(env, mode=sandbox_mode)
            sandbox_name = await open_sandbox(blueprint.id, workspace_dir, mode=sandbox_mode)
            if sandbox_name:
                cmd = sandbox_exec_argv(sandbox_name, cmd, workspace_dir)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                # cwd = workspace so Claude Code treats it as the project root and
                # discovers the staged <workspace>/.claude/skills/. Agents already
                # cd $WORK_DIR for their own shell work. (In openshell mode the
                # policy's include_workdir surfaces this dir inside the sandbox.)
                cwd=workspace_dir,
                start_new_session=True,
            )

            hb_id = env.get("HEARTBEAT_ID", "")
            if hb_id:
                process_registry.register(hb_id, proc)

            if proc.stdin is None:
                raise RuntimeError("claude subprocess has no stdin")
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            # Emit a synthetic event so the operator sees which auth mode
            # was used (helps when debugging "is this even using my Max plan?").
            if event_sink is not None:
                try:
                    event_sink({"type": "claude_auth", "payload": {
                        "mode": auth_mode, "detail": auth_detail, "model": model or "(default)",
                    }})
                except Exception as e:
                    _warn("event_sink.auth", e)

            async def _drain_stdout(stream: asyncio.StreamReader) -> None:
                async for line in iter_subprocess_lines(stream):
                    try:
                        text = line.decode("utf-8")
                    except UnicodeDecodeError as e:
                        text = line.decode("utf-8", errors="replace")
                        _warn(
                            "stdout.decode",
                            Exception(f"{e}; raw_bytes={line[:80]!r}"),
                        )
                    stdout_chunks.append(text)
                    if stdout_sink is not None:
                        stdout_sink(text)
                    if event_sink is not None:
                        for ev in _line_to_events(text):
                            try:
                                event_sink(ev)
                            except Exception as e:
                                _warn("event_sink.stdout", e)

            async def _drain_stderr(stream: asyncio.StreamReader) -> None:
                async for line in iter_subprocess_lines(stream):
                    try:
                        text = line.decode("utf-8")
                    except UnicodeDecodeError as e:
                        text = line.decode("utf-8", errors="replace")
                        _warn(
                            "stderr.decode",
                            Exception(f"{e}; raw_bytes={line[:80]!r}"),
                        )
                    stderr_chunks.append(text)
                    if event_sink is not None:
                        try:
                            event_sink({"type": "stderr", "payload": {"text": text}})
                        except Exception as e:
                            _warn("event_sink.stderr", e)

            try:
                await asyncio.gather(
                    _drain_stdout(proc.stdout),
                    _drain_stderr(proc.stderr),
                )
                exit_code = await proc.wait()
            finally:
                if hb_id:
                    process_registry.unregister(hb_id)
                # Sandbox teardown: download the workspace back (collecting artifacts
                # the agent wrote inside the sandbox) then delete it. No-op when not
                # sandboxed. Do this BEFORE unstage so skills are cleaned last.
                if sandbox_name:
                    await close_sandbox(sandbox_name, workspace_dir)
        finally:
            # Stage over: remove this agent's skills from the workspace,
            # on every exit path (including the auth raise above).
            unstage_skills(workspace_dir)

        raw_stdout = "".join(stdout_chunks)
        raw_stderr = "".join(stderr_chunks)

        # The final assistant text lives in the last {"type":"result"}
        # event's "result" field. Extract by parsing the JSONL stream.
        final_message = ""
        for line in reversed(raw_stdout.splitlines()):
            s = line.strip()
            if not s.startswith("{"):
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "result":
                final_message = str(obj.get("result") or "")
                break
        if not final_message:
            final_message = raw_stdout

        if exit_code != 0:
            if exit_code < 0 or exit_code == 143:
                raise RuntimeError(
                    f"claude exec cancelled (signal {-exit_code if exit_code < 0 else 'SIGTERM'}) "
                    f"for agent {blueprint.id}"
                )
            err_tail = (raw_stderr or final_message)[-500:].strip()
            raise RuntimeError(
                f"claude exec failed (exit={exit_code}) for agent {blueprint.id}: {err_tail}"
            )

        if blueprint.output_schema is None:
            class _Empty(BaseModel):
                pass
            return DriverRunResult(
                output=_Empty(),
                raw_stdout=raw_stdout, raw_stderr=raw_stderr,
                exit_code=exit_code, driver=self.name, model=model,
            )

        # min_keys=2 rejects bare {} placeholders Claude sometimes emits
        # mid-stream (e.g. empty tool-call input). Every typed Result
        # schema we ship has more than 2 required fields.
        data = extract_trailing_json(
            final_message, min_keys=2, output_schema=blueprint.output_schema,
        )
        if data is None:
            raise ValueError(
                f"claude exec produced no parseable trailing JSON line for agent "
                f"{blueprint.id}. Final message tail:\n{final_message[-800:]}"
            )
        output = blueprint.output_schema.model_validate(data)
        return DriverRunResult(
            output=output,
            raw_stdout=raw_stdout, raw_stderr=raw_stderr,
            exit_code=exit_code, driver=self.name, model=model,
        )
