"""Incremental heartbeat accounting from provider-reported usage.

A provider session total replaces its interim turn totals; it is not another
turn. Repeated message snapshots update that message rather than billing it
again. Counters are absolute when persisted, so a database retry is idempotent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.db import HeartbeatRun
from zevo.engine.cost.pricing import estimate_cost

FIELDS = ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens")


def _counts(usage: dict[str, Any]) -> dict[str, int]:
    counts = {}
    for key in FIELDS:
        try:
            counts[key] = max(0, int(usage.get(key, 0) or 0))
        except (ValueError, TypeError, OverflowError):
            counts[key] = 0
    return counts


@dataclass
class LiveUsage:
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FIELDS, 0))
    messages: dict[str, dict[str, int]] = field(default_factory=dict)
    cumulative: bool = False

    def record(self, payload: dict[str, Any]) -> None:
        raw = payload.get("usage")
        if not isinstance(raw, dict):
            return
        current = _counts(raw)
        if payload.get("usage_kind") == "cumulative":
            # Some CLI versions omit the final usage object entirely. Do not
            # erase the known interim spend for an empty result event.
            if raw:
                self.counts = current
                self.cumulative = True
            return
        if self.cumulative:
            return  # delayed interim events cannot inflate a final total
        message_id = str(payload.get("usage_id") or "")
        previous = self.messages.get(message_id, {}) if message_id else {}
        for key in FIELDS:
            self.counts[key] += current[key] - previous.get(key, 0)
        if message_id:
            self.messages[message_id] = current

    def values(self, model: str) -> dict[str, int | float]:
        return {**self.counts, "estimated_cost_usd": estimate_cost(model, self.counts)}


async def persist_usage(
    session: AsyncSession, heartbeat_id: str, model: str, usage: LiveUsage,
) -> None:
    """Caller commits together with the transcript batch; never closes a run."""
    await session.execute(
        update(HeartbeatRun).where(HeartbeatRun.id == heartbeat_id)
        .values(**usage.values(model))
        .execution_options(synchronize_session=False)
    )
