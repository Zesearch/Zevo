"""Shared prompt assembly for every driver.

Each driver (claude_cli / bedrock) used to copy-paste a near-identical
_render_prompt or _render_user_message, and the copies drifted (one inlined
the JSON Schema, the other didn't). This module is the single source of
truth so:

  - every driver produces equivalent prompts → the agent's
    behaviour shouldn't depend on which driver runs it
  - the JSON-strict block is one knob (`strict_json=True/False`)
    instead of three half-equivalent variants
  - future drivers (cerebras, deepseek, ...) get the same prompt for
    free

Each driver still owns its OWN transport (subprocess vs SDK), but the
TEXT they hand to the model comes from here.

Surface:
  render_agent_prompt(blueprint, input_payload, conversation,
                      strict_json=True, schema_inline=False) -> str
  render_user_message(input_payload, output_schema, conversation,
                      strict_json=True, schema_inline=False) -> str

The two functions differ only in their HEADER:
  - render_agent_prompt prepends the full instructions (for the CLI
    drivers that push prompt-as-stdin without a separate system msg)
  - render_user_message expects the instructions to be sent SEPARATELY
    as the system prompt (for the SDK drivers that have both fields)

They share the payload / conversation / strict-JSON blocks.
"""
from __future__ import annotations

import json

from pydantic import BaseModel

from zevo.engine.agent.loader import AgentBlueprint


# ─────────────────────────── shared blocks ──────────────────────────────────


def _payload_block(input_payload: BaseModel) -> str:
    """`## Ticket payload (<schema>)\n\n```json\n{...}\n````"""
    return (
        f"## Ticket payload ({type(input_payload).__name__})\n\n"
        f"```json\n{input_payload.model_dump_json(indent=2)}\n```"
    )


def _conversation_block(conversation: list[dict] | None) -> str:
    """Empty string if no conversation. Otherwise: separator + the
    standard CONVERSATION SO FAR explainer + each message as
    `**[author]** body`."""
    if not conversation:
        return ""
    lines: list[str] = [
        "",
        "---",
        "",
        "## CONVERSATION SO FAR",
        "",
        (
            "Below is the conversation that has happened on THIS ticket "
            "before this run. The MOST RECENT user message is the live "
            "instruction — treat it as the current request, even if it "
            "asks you to change or redo something you already did. Use "
            "earlier messages as context. Acknowledge what changed in "
            "your first action (a one-line POST to "
            "/tickets/$TICKET_ID/messages)."
        ),
        "",
    ]
    for c in conversation:
        author = str(c.get("author") or "?")
        body = (str(c.get("body") or "")).strip()
        if not body:
            continue
        lines.append(f"**[{author}]** {body}")
        lines.append("")
    return "\n".join(lines)


def _do_the_work_block() -> str:
    """The anti-fabrication / anti-half-done block. Applied to every
    driver because every agent benefits from being told NOT to stop
    after just planning. Particularly important for Claude (drifts toward
    verbose planning + early exit) and any future model that does the
    same."""
    return (
        "\n\n## CRITICAL — DO THE WHOLE TASK IN THIS ONE RESPONSE\n\n"
        "Execute the entire task NOW, in this single turn: actually run "
        "the shell commands, read/write the real files, and finish. Do "
        "NOT stop after merely announcing a plan or saying what you "
        "'will' do next. A response that ends with only a planning "
        "sentence (e.g. \"Setting up a plan...\", \"Let me start by...\") "
        "and no completed work + no final JSON is a HARD FAILURE — the "
        "harness receives nothing and your ticket fails. Keep going "
        "until the work is genuinely done and you have printed the JSON "
        "described below. Never end your turn early."
    )


# How much of the JSON Schema may go in the prompt. Generous, because what it
# guards against is a runaway schema, not a large one.
_SCHEMA_MAX_CHARS = 12000


def _strip_prose(node: object) -> object:
    """The schema without its `description` / `title` text.

    Those two carry most of a schema's bytes and none of its contract: the
    agent already has the same prose in its `platform.md`, at length. What it
    cannot get anywhere else is the field names, the types, and `required`.
    """
    if isinstance(node, dict):
        return {k: _strip_prose(v) for k, v in node.items()
                if k not in ("description", "title")}
    if isinstance(node, list):
        return [_strip_prose(v) for v in node]
    return node


def _schema_json(output_schema: type[BaseModel]) -> str:
    """The schema as JSON that is still JSON when it lands in the prompt.

    This used to be `json.dumps(...)[:4000]`, which is not a size limit but a
    corruption: a schema over the cap arrived cut mid-string, with no closing
    braces and — because Pydantic emits it last — no `required` array at all.
    DataResult (4344 chars) and TrainResult (4199) were both over it, so the two
    agents most likely to miss a required field were the two told which fields
    were required by a truncated document, right under a rule saying the
    validator rejects anything that does not match the schema exactly.

    Drop the prose first; that fits every schema we have (4.3KB -> 1.3KB). Only
    if something is still oversized do we cut, and then we say so in-band rather
    than letting the JSON just stop.
    """
    full = json.dumps(output_schema.model_json_schema(), indent=2)
    if len(full) <= _SCHEMA_MAX_CHARS:
        return full
    lean = json.dumps(_strip_prose(output_schema.model_json_schema()), indent=2)
    if len(lean) <= _SCHEMA_MAX_CHARS:
        return lean
    return lean[:_SCHEMA_MAX_CHARS] + "\n… TRUNCATED — see your platform.md for the full contract"


def _schema_hint_block(
    output_schema: type[BaseModel] | None,
    *,
    strict_json: bool,
    schema_inline: bool,
) -> str:
    """The REQUIRED FINAL OUTPUT instruction.

    `strict_json` (default True): emit the "single line, no fences, no
    trailing prose" rules. Set False for drivers/contexts where wrapping
    in a ```json fence is acceptable.

    `schema_inline`: include the JSON Schema body. Claude needs this
    to stay on-contract (was 9 ValidationErrors → 0 in our smoke).
    Codex usually doesn't, but the agent reads the same instructions so
    enabling it everywhere is safe.
    """
    if output_schema is None:
        return ""
    name = output_schema.__name__

    schema_block = ""
    if schema_inline:
        try:
            schema_json = _schema_json(output_schema)
            schema_block = f"\n\n```json\n{schema_json}\n```"
        except Exception:  # noqa: BLE001
            schema_block = ""

    if strict_json:
        rules = (
            f"\n\n## REQUIRED FINAL OUTPUT (strict)\n\n"
            f"When you are DONE with the task, your VERY LAST line of stdout "
            f"MUST be a single JSON OBJECT conforming to the "
            f"`{name}` Pydantic schema.\n\n"
            f"HARD RULES:\n"
            f"  - Output the JSON on ONE LINE. No surrounding markdown fences.\n"
            f"  - Print NOTHING after the JSON (no 'done', no '(no further output)').\n"
            f"  - Field names + types MUST match the schema exactly; the "
            f"harness's Pydantic validator rejects unknown keys, wrong types, "
            f"or missing required fields and fails the ticket.\n"
            f"  - Boolean fields → JSON true/false (not 'true' strings). "
            f"Integer fields → bare numbers (not '12').\n"
            f"  - If a field is optional and you have nothing to report, "
            f"omit it OR pass the schema's default; do NOT pass null unless "
            f"the schema explicitly allows it."
        )
    else:
        rules = (
            f"\n\n## REQUIRED FINAL OUTPUT\n\n"
            f"When you are DONE, your VERY LAST message must end with a "
            f"single JSON object conforming to the `{name}` Pydantic "
            f"schema. The harness parses the trailing JSON; if it's "
            f"missing or malformed the ticket fails. You may wrap it in "
            f"a ```json fence or leave it bare."
        )
    return rules + schema_block


# ─────────────────────────── public renderers ────────────────────────────────


def render_agent_prompt(
    blueprint: AgentBlueprint,
    input_payload: BaseModel,
    conversation: list[dict] | None = None,
    *,
    strict_json: bool = True,
    schema_inline: bool = False,
) -> str:
    """Used by the subprocess driver (claude_cli) that hands
    instructions + payload + schema as ONE stdin blob."""
    return (
        f"{blueprint.instructions.rstrip()}\n\n---\n\n"
        f"{_payload_block(input_payload)}"
        f"{_conversation_block(conversation)}"
        f"{_do_the_work_block()}"
        f"{_schema_hint_block(blueprint.output_schema, strict_json=strict_json, schema_inline=schema_inline)}"
    )


def render_user_message(
    input_payload: BaseModel,
    output_schema: type[BaseModel] | None,
    conversation: list[dict] | None = None,
    *,
    strict_json: bool = False,
    schema_inline: bool = False,
) -> str:
    """Used by SDK drivers (anthropic, bedrock) that send the instructions as
    the SEPARATE system prompt and only the payload / conversation /
    schema instruction as the user message.

    Note: SDK drivers default to `strict_json=False` because their
    transport DOES allow markdown fences cleanly (the response is JSON-
    decoded by extract_trailing_json regardless). Pass strict_json=True
    if you've seen the model drift.
    """
    parts: list[str] = [
        (
            f"Here is the {type(input_payload).__name__} for ticket "
            f"{getattr(input_payload, 'ticket_id', '')}.\n"
            f"Use the built-in `run_bash`, `read_file`, and `write_file` tools "
            f"to do your work, then return the structured output described in "
            f"your system instructions."
        ),
        "",
        _payload_block(input_payload),
    ]
    conv = _conversation_block(conversation)
    if conv:
        parts.append(conv)
    parts.append(_do_the_work_block())
    parts.append(
        _schema_hint_block(output_schema, strict_json=strict_json, schema_inline=schema_inline)
    )
    return "\n".join(p for p in parts if p)
