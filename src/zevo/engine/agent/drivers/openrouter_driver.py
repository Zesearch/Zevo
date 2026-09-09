"""Driver backed by OpenRouter (HTTP, OpenAI-compatible chat completions).

One key fronts several hundred models from every major lab, which is the
point of it here: trying an agent on a different model is a string
change, not a new driver and a new set of credentials.

The wire format is OpenAI's `/chat/completions` with tool calling, so
this is the same in-process turn loop the bedrock driver runs -- same
shared tools from `_agent_loop`, same events, same typed-Result contract
-- with the request and response shapes swapped. Nothing is streamed:
one request per turn, tools dispatched, results appended, repeat.

Auth: `OPENROUTER_API_KEY`. Per-call billing against your OpenRouter
credit; there is no subscription mode.

Model ids are namespaced by provider, e.g. `qwen/qwen3.8-max` (the
default), `moonshotai/kimi-k3`, `google/gemini-3.6-flash`,
`deepseek/deepseek-v4-pro`.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable

from pydantic import BaseModel

from zevo.engine.agent.loader import AgentBlueprint
from zevo.engine.agent.drivers._agent_loop import (
    BUILTIN_TOOLS,
    SKILL_TOOL,
    _new_item_id,
    build_tool_env,
    emit_agent_message,
    emit_tool_call,
    emit_tool_result,
    emit_turn_completed,
    exec_bash,
    exec_read,
    exec_write,
)
from zevo.engine.agent.drivers._json_utils import extract_trailing_json
from zevo.engine.agent.drivers.base import DriverRunResult
from zevo.engine.observe.markers import scan_text as _scan_markers
from zevo.engine.agent.skills import (
    stage_skills, unstage_skills, list_staged_skills, load_staged_skill,
)


API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.8-max"
DEFAULT_MAX_TOKENS = 8192
# Long enough for a turn that thinks; the harness's own ticket timeout is
# the real ceiling.
REQUEST_TIMEOUT_S = 900


def _warn(where: str, exc: BaseException) -> None:
    print(f"[openrouter.{where}] WARN: {type(exc).__name__}: {exc}",
          file=sys.stderr, flush=True)


def _to_openai_tool(tool: dict) -> dict:
    """Harness tool spec -> OpenAI function-tool spec."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        },
    }


def _emit_markers_from_stdout(event_sink, stdout: str) -> None:
    """Surface Zevo attempt/phase/progress/config markers printed inside a tool's
    stdout, so the live progress bar and the loss chart work the same way
    they do under every other driver."""
    if event_sink is None or not stdout:
        return
    # Whole-string scan: a remote stage's log reaches us with its newlines
    # escaped, so splitting on lines finds one marker where there are dozens.
    for marker in _scan_markers(stdout):
        kind, data = marker
        try:
            if kind == "attempt":
                event_sink({"type": "attempt", "payload": data})
            elif kind == "phase":
                event_sink({"type": "phase", "payload": data})
            elif kind == "progress":
                event_sink({"type": "progress", "payload": data})
            elif kind == "config":
                event_sink({"type": "config", "payload": data})
        except Exception as e:
            _warn("event_sink.marker", e)


def _map_usage(raw: Any) -> dict[str, int]:
    """OpenAI-shaped usage -> the harness's snake_case cost fields.

    OpenRouter passes the upstream provider's numbers through, and the
    cached-token count hides in a nested `prompt_tokens_details` when the
    provider reports one at all.
    """
    if not isinstance(raw, dict):
        return {}
    details = raw.get("prompt_tokens_details") or {}
    out = {
        "input_tokens": int(raw.get("prompt_tokens") or 0),
        "output_tokens": int(raw.get("completion_tokens") or 0),
        "cached_input_tokens": int(details.get("cached_tokens") or 0),
        "reasoning_output_tokens": int(
            (raw.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        ),
    }
    return out if any(out.values()) else {}


def _render_user_text(
    input_payload: BaseModel,
    output_schema: type[BaseModel] | None,
    conversation: list[dict] | None,
) -> str:
    from zevo.engine.agent.drivers._prompt import render_user_message
    return render_user_message(
        input_payload, output_schema, conversation,
        strict_json=False, schema_inline=True,
    )


async def _post(payload: dict, *, api_key: str) -> dict:
    """One chat-completions call, off the event loop.

    httpx is already a dependency (the backend uses it), and the call is
    made in a thread rather than async so a driver that is otherwise
    synchronous in shape stays easy to follow.
    """
    import asyncio

    import httpx

    def _call() -> dict:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # OpenRouter attributes traffic by these; they are optional but
            # they are how a key's usage is legible on their dashboard.
            "HTTP-Referer": os.environ.get("ZEVO_PUBLIC_URL", "https://github.com/Zesearch"),
            "X-Title": "Zevo",
        }
        with httpx.Client(timeout=REQUEST_TIMEOUT_S) as c:
            r = c.post(API_URL, headers=headers, json=payload)
            if r.status_code >= 400:
                raise RuntimeError(
                    f"openrouter {r.status_code}: {r.text[:400]}"
                )
            return r.json()

    return await asyncio.to_thread(_call)


async def _dispatch_tool(
    name: str, args: dict, *, cwd: str, env: dict[str, str], staged_dir=None
) -> dict[str, Any]:
    if name == "Skill":
        action = str(args.get("action", "load")).lower()
        if action == "list":
            return {"skills": list_staged_skills(staged_dir)}
        body = load_staged_skill(staged_dir, str(args.get("name", "")))
        if not body:
            avail = [s["name"] for s in list_staged_skills(staged_dir)]
            return {"error": f"no skill {args.get('name')!r} staged. available: {avail}"}
        return {"skill": str(args.get("name", "")), "instructions": body}
    if name == "run_bash":
        return await exec_bash(
            command=str(args.get("command", "")), cwd=cwd, env=env,
            timeout_sec=(
                int(args["timeout_sec"]) if args.get("timeout_sec") is not None else None
            ),
        )
    if name == "read_file":
        return exec_read(path=str(args.get("path", "")), cwd=cwd,
                         max_chars=int(args.get("max_chars") or 8000))
    if name == "write_file":
        return exec_write(path=str(args.get("path", "")),
                          content=str(args.get("content", "")), cwd=cwd)
    return {"error": f"unknown tool {name!r}"}


class OpenRouterDriver:
    name = "openrouter"

    @staticmethod
    def _api_key() -> str:
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "openrouter driver requires OPENROUTER_API_KEY in .env. "
                "Create one at https://openrouter.ai/keys, then set it with "
                "`set-secret OPENROUTER_API_KEY` inside Zevo so it is never typed on "
                "a command line."
            )
        return key

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
        sandbox_mode: str = "",   # accepted for parity; in-process, no sandbox
    ) -> DriverRunResult:
        api_key = self._api_key()
        model_id = model_override or blueprint.default_model or DEFAULT_MODEL

        staged = stage_skills(blueprint.id, workspace_dir)
        tools = list(BUILTIN_TOOLS) + ([SKILL_TOOL] if staged else [])
        tool_specs = [_to_openai_tool(t) for t in tools]

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": blueprint.instructions},
            {"role": "user", "content": _render_user_text(
                input_payload, blueprint.output_schema, conversation
            )},
        ]

        tool_env = build_tool_env(
            base_env=None, workspace_dir=workspace_dir,
            input_payload=input_payload, agent_id=blueprint.id,
        )

        final_text = ""
        raw_chunks: list[str] = []
        turn = 0

        try:
            for turn in range(max_turns):
                resp = await _post({
                    "model": model_id,
                    "messages": messages,
                    "tools": tool_specs,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                }, api_key=api_key)

                usage = _map_usage(resp.get("usage"))
                if usage:
                    emit_turn_completed(event_sink, usage=usage)

                choices = resp.get("choices") or []
                if not choices:
                    raise RuntimeError(
                        f"openrouter returned no choices for {model_id}: "
                        f"{json.dumps(resp)[:400]}"
                    )
                msg = choices[0].get("message") or {}
                text = str(msg.get("content") or "").strip()
                tool_calls = list(msg.get("tool_calls") or [])

                if text:
                    emit_agent_message(event_sink, text)
                    raw_chunks.append(text)
                    final_text = text

                if not tool_calls:
                    break

                # The assistant turn goes back verbatim, tool_call ids and
                # all: the next request has to answer each id exactly.
                messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tool_calls,
                })

                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    tool_name = str(fn.get("name") or "")
                    call_id = str(tc.get("id") or _new_item_id())
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}

                    emit_tool_call(event_sink, item_id=call_id, tool=tool_name, args=args)
                    result = await _dispatch_tool(
                        tool_name, args, cwd=workspace_dir, env=tool_env,
                        staged_dir=staged,
                    )
                    emit_tool_result(event_sink, item_id=call_id, tool=tool_name,
                                     result=result)

                    if tool_name == "run_bash":
                        out = str(result.get("stdout") or result.get("output") or "")
                        _emit_markers_from_stdout(event_sink, out)
                        if stdout_sink is not None and out:
                            stdout_sink(out + "\n")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result)[:20000],
                    })
        finally:
            unstage_skills(workspace_dir)

        if stdout_sink is not None:
            stdout_sink(f"[openrouter] agent={blueprint.id} model={model_id} "
                        f"turns={turn + 1}\n")

        raw_stdout = "\n\n".join(raw_chunks)
        if blueprint.output_schema is None:
            class _Empty(BaseModel):
                pass
            return DriverRunResult(
                output=_Empty(), raw_stdout=raw_stdout, raw_stderr="",
                exit_code=0, driver=self.name, model=model_id,
            )

        data = extract_trailing_json(
            final_text, min_keys=2, output_schema=blueprint.output_schema,
        )
        if data is None:
            raise ValueError(
                f"openrouter produced no parseable trailing JSON for agent "
                f"{blueprint.id} on {model_id}. Final message tail:\n"
                f"{final_text[-800:]}"
            )
        return DriverRunResult(
            output=blueprint.output_schema.model_validate(data),
            raw_stdout=raw_stdout, raw_stderr="",
            exit_code=0, driver=self.name, model=model_id,
        )
