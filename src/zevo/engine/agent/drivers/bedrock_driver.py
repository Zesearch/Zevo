"""Driver backed by AWS Bedrock's Converse API.

Bedrock Converse is a vendor-neutral API across Claude / Llama / Mistral
/ Nova / Cohere / etc., so this one driver covers every model Bedrock
hosts. Same in-process turn loop as the openrouter driver (shared tools
from `_agent_loop`), but with Bedrock's tool-schema format and
inputTokens/outputTokens key mapping.

Auth (in order of preference):
  - `AWS_BEARER_TOKEN_BEDROCK` — a Bedrock API key (bearer token); boto3
    (botocore >= 1.39) uses it automatically. Simplest; no AWS access key needed.
  - standard boto3 chain: `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`,
    `~/.aws/credentials`, or an IAM instance role.
Region: `AWS_REGION` env, defaults to `us-east-1`.

Model IDs (open-weight models use the bare id; Claude 4.x+ needs the `us.`
inference-profile prefix). The priced set lives in `engine/cost/pricing.py`;
all of these support tool use via Converse:
  moonshotai.kimi-k2.5            (default — strong agentic / tool-use model)
  deepseek.v3.2
  minimax.minimax-m2.5
  zai.glm-5
  qwen.qwen3-coder-next
  us.anthropic.claude-sonnet-4-6
"""
from __future__ import annotations

import os
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
from zevo.engine.agent.drivers.base import DriverRunResult
from zevo.engine.agent.drivers._json_utils import extract_trailing_json as _extract_trailing_json
from zevo.engine.observe.markers import scan_text as _scan_markers
from zevo.engine.agent.skills import (
    stage_skills, unstage_skills, list_staged_skills, load_staged_skill,
)


# Bedrock has no native Skill runtime (that's a Claude Code feature), so we
# expose our own tool. It covers EVERY model routed through Bedrock — Kimi, GLM,
# Nova, Claude, … — since they all share this one in-process loop. Skills are
# staged into <workspace>/.claude/skills/ (same place claude_cli uses); this
# tool lists + loads them from there.


def _emit_markers_from_stdout(event_sink, stdout: str) -> None:
    """Parse Zevo attempt/phase/progress/config markers out of bash stdout and
    re-emit them as execution events — the same path the
    runner uses to build execution_events (loss charts + config panel). Without
    this, the training/inference scripts' markers never become phases on the
    bedrock driver (they only sit in the raw stdout)."""
    if event_sink is None or not stdout:
        return
    # Whole-string scan: a remote stage's log reaches us with its newlines
    # escaped, so splitting on lines finds one marker where there are dozens.
    for parsed in _scan_markers(stdout):
        kind, payload = parsed
        if kind == "attempt":
            event_sink({"type": "attempt", "payload": payload})
        elif kind == "phase":
            event_sink({"type": "phase", "payload": payload})
        elif kind == "progress":
            event_sink({"type": "progress", "payload": payload})
        elif kind == "config":
            event_sink({"type": "config", "payload": payload})


DEFAULT_MODEL = "moonshotai.kimi-k2.5"
DEFAULT_MAX_TOKENS = 4096


def _to_bedrock_tool(tool: dict) -> dict:
    """Translate a vendor-neutral tool dict to Bedrock Converse's
    `toolSpec` format.

    Converse format:
        {"toolSpec": {
            "name": ...,
            "description": ...,
            "inputSchema": {"json": {...JSONSchema...}}
        }}
    """
    return {
        "toolSpec": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "inputSchema": {"json": tool["input_schema"]},
        }
    }


def _render_user_text(
    input_payload: BaseModel,
    output_schema: type[BaseModel] | None,
    conversation: list[dict] | None,
) -> str:
    """Delegates to the shared renderer. Bedrock Converse takes plain
    text under content blocks — same shape as the SDK Anthropic call."""
    from zevo.engine.agent.drivers._prompt import render_user_message
    return render_user_message(
        input_payload, output_schema, conversation,
        strict_json=False, schema_inline=True,
    )


def _map_usage(raw_usage: Any) -> dict[str, int]:
    """Bedrock Converse usage -> harness snake_case dict.

    Converse returns: inputTokens, outputTokens, totalTokens
    Some models also report cacheReadInputTokens / cacheWriteInputTokens.
    """
    if not isinstance(raw_usage, dict):
        return {}
    def _get(*keys: str) -> int:
        for k in keys:
            v = raw_usage.get(k)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
        return 0
    return {
        "input_tokens": _get("inputTokens"),
        "output_tokens": _get("outputTokens"),
        "cached_input_tokens": _get("cacheReadInputTokens"),
        "reasoning_output_tokens": 0,
    }


class BedrockDriver:
    name = "bedrock"

    def __init__(self, region: str | None = None):
        self._region_override = region

    def _client(self):
        try:
            import boto3
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "`boto3` package is not installed. Add it to "
                "pyproject.toml dependencies and rebuild the backend image."
            ) from e
        region = (
            self._region_override
            or os.environ.get("AWS_REGION")
            or "us-east-1"
        )
        # boto3 picks up AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
        # ~/.aws/credentials / IAM role automatically; no key handling here.
        return boto3.client("bedrock-runtime", region_name=region)

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
        sandbox_mode: str = "",   # accepted for interface parity; in-process driver, no sandbox
    ) -> DriverRunResult:
        client = self._client()
        model_id = model_override or blueprint.default_model or DEFAULT_MODEL

        # Stage this agent's skills into <workspace>/.claude/skills/; expose the
        # `Skill` tool only when there are any. Removed in the finally below.
        staged = stage_skills(blueprint.id, workspace_dir)
        try:
            tools = list(BUILTIN_TOOLS) + ([SKILL_TOOL] if staged else [])
            tool_config = {"tools": [_to_bedrock_tool(t) for t in tools]}
            system_blocks = [{"text": blueprint.instructions}]

            # Converse messages are role + content (list of blocks).
            messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {"text": _render_user_text(
                            input_payload, blueprint.output_schema, conversation
                        )},
                    ],
                },
            ]

            tool_env = build_tool_env(
                base_env=None, workspace_dir=workspace_dir, input_payload=input_payload,
                agent_id=blueprint.id,
            )

            final_text = ""
            raw_stdout_chunks: list[str] = []

            for turn in range(max_turns):
                resp = await _converse_async(
                    client,
                    modelId=model_id,
                    system=system_blocks,
                    messages=messages,
                    toolConfig=tool_config,
                    inferenceConfig={"maxTokens": DEFAULT_MAX_TOKENS},
                )

                # Cost meter.
                usage = _map_usage(resp.get("usage"))
                if usage:
                    emit_turn_completed(event_sink, usage=usage)

                output = resp.get("output") or {}
                assistant_msg = output.get("message") or {}
                content_blocks = list(assistant_msg.get("content") or [])

                tool_uses: list[dict] = []
                text_pieces: list[str] = []
                for block in content_blocks:
                    if "text" in block:
                        text_pieces.append(block.get("text", "") or "")
                    elif "toolUse" in block:
                        tool_uses.append(block["toolUse"])

                text_this_turn = "".join(text_pieces).strip()
                if text_this_turn:
                    emit_agent_message(event_sink, text_this_turn)
                    raw_stdout_chunks.append(text_this_turn)
                    final_text = text_this_turn

                # Stop when the model says it's done.
                stop_reason = resp.get("stopReason", "")
                if not tool_uses or stop_reason == "end_turn":
                    break

                # Echo the assistant message back into the conversation so
                # Bedrock sees its own toolUse ids on the next turn.
                messages.append({"role": "assistant", "content": content_blocks})

                # Dispatch tools and build tool_result blocks.
                tool_result_blocks: list[dict[str, Any]] = []
                for tu in tool_uses:
                    tool_name = str(tu.get("name", ""))
                    tool_id = str(tu.get("toolUseId") or _new_item_id())
                    tool_input = tu.get("input") or {}
                    emit_tool_call(
                        event_sink, item_id=tool_id, tool=tool_name, args=tool_input
                    )
                    result = await _dispatch_tool(
                        tool_name, tool_input, cwd=workspace_dir, env=tool_env,
                        staged_dir=staged,
                    )
                    emit_tool_result(
                        event_sink, item_id=tool_id, tool=tool_name, result=result
                    )
                    # Surface the training/inference markers (__PROGRESS__/__CONFIG__/
                    # __PHASE__) that print inside the bash tool's stdout, as phase
                    # events → loss charts + the config (temperature, etc.) panel.
                    if tool_name == "run_bash":
                        tool_out = result.get("stdout") or result.get("output") or ""
                        _emit_markers_from_stdout(event_sink, str(tool_out))
                        # Also mirror the raw stdout into the heartbeat log so the
                        # training output is visible in the transcript.
                        if stdout_sink is not None and tool_out:
                            stdout_sink(str(tool_out) + "\n")
                    tool_result_blocks.append({
                        "toolResult": {
                            "toolUseId": tool_id,
                            "content": [{"text": _result_to_text(tool_name, result)}],
                            "status": (
                                "error"
                                if (
                                    result.get("error")
                                    or (
                                        tool_name == "run_bash"
                                        and result.get("exit_code") not in (None, 0)
                                    )
                                )
                                else "success"
                            ),
                        }
                    })
                messages.append({"role": "user", "content": tool_result_blocks})

        finally:
            # Skills staged for the turn loop must not outlive it -- remove the
            # staged copy on every exit path, not just clean completion.
            unstage_skills(workspace_dir)

        if stdout_sink is not None:
            stdout_sink(f"[bedrock] agent={blueprint.id} model={model_id} turns={turn + 1}\n")

        if blueprint.output_schema is None:
            class _Empty(BaseModel):
                pass
            return DriverRunResult(
                output=_Empty(),
                raw_stdout="\n\n".join(raw_stdout_chunks),
                raw_stderr="",
                exit_code=0,
                driver=self.name,
                model=model_id,
            )

        # min_keys=2 rejects bare {} / one-key placeholders, matching every
        # other driver's acceptance rule.
        data = _extract_trailing_json(
            final_text, min_keys=2, output_schema=blueprint.output_schema,
        )
        if data is None:
            raise ValueError(
                f"bedrock driver produced no parseable trailing JSON for "
                f"agent {blueprint.id}. Final message tail:\n{final_text[-800:]}"
            )
        output = blueprint.output_schema.model_validate(data)

        return DriverRunResult(
            output=output,
            raw_stdout="\n\n".join(raw_stdout_chunks),
            raw_stderr="",
            exit_code=0,
            driver=self.name,
            model=model_id,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _converse_async(client, **kwargs):
    """Run the (blocking) boto3 call in a thread so we don't stall the loop."""
    import asyncio
    return await asyncio.to_thread(client.converse, **kwargs)


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
            command=str(args.get("command", "")),
            cwd=cwd,
            env=env,
            timeout_sec=(
                int(args["timeout_sec"]) if args.get("timeout_sec") is not None else None
            ),
        )
    if name == "read_file":
        return exec_read(
            path=str(args.get("path", "")),
            cwd=cwd,
            max_chars=int(args.get("max_chars") or 8000),
        )
    if name == "write_file":
        return exec_write(
            path=str(args.get("path", "")),
            content=str(args.get("content", "")),
            cwd=cwd,
        )
    return {"error": f"unknown tool: {name!r}"}


def _result_to_text(tool: str, result: dict) -> str:
    import json
    if tool == "Skill" and result.get("instructions"):
        return str(result["instructions"])   # hand the model the raw skill text
    if tool == "run_bash":
        out = result.get("stdout", "") or ""
        err = result.get("stderr", "") or ""
        code = result.get("exit_code", 0)
        text = out
        if err:
            text += f"\n[stderr]\n{err}"
        text += f"\n[exit_code] {code}"
        return text
    return json.dumps(result, ensure_ascii=False)
