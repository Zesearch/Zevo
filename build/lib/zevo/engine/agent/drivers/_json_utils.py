"""Helpers shared by the CLI-based driver (claude_cli).

extract_trailing_json: pull the typed *Result Pydantic JSON out of an
agent's final message even when the model wrapped it in a ```json fence,
included prose before it, etc. The harness uses this to validate
against the per-agent output_schema.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError


_DECODER = json.JSONDecoder()


def extract_trailing_json(
    text: str,
    *,
    min_keys: int = 1,
    output_schema: type[BaseModel] | None = None,
) -> dict[str, Any] | None:
    """Return the terminal top-level JSON object from an agent message.

    Decode every ``{`` candidate, then order objects by where their closing
    brace occurs (rightmost first) and by span (outermost first). Starting at
    the rightmost opening brace is incorrect for nested output: it selects the
    last child mapping, such as ``DataResult.loss_contract``, instead of the
    complete Result that contains it.

    When ``output_schema`` is supplied, prefer the first candidate that
    validates as that exact Result. If none validates, return the structurally
    most likely terminal object so the caller's normal validation step retains
    the detailed Pydantic error rather than reporting "no JSON".

    `min_keys` defaults to 1 (rejects bare `{}`). Drivers can raise it
    when calling for a strict typed result — Claude in particular tends
    to emit empty `{}` placeholders mid-stream that would otherwise
    shadow the real terminal SynthResult JSON.
    """
    decoded: list[tuple[int, int, dict[str, Any]]] = []
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = _DECODER.raw_decode(text, idx=start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and len(obj) >= min_keys:
            decoded.append((end, end - start, obj))
    if not decoded:
        return None

    decoded.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if output_schema is not None:
        for _end, _span, obj in decoded:
            try:
                output_schema.model_validate(obj)
            except ValidationError:
                continue
            return obj
    return decoded[0][2]
