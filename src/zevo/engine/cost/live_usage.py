"""Incremental heartbeat accounting from provider-reported usage.

A provider session total replaces its interim turn totals; it is not another
turn. Repeated message snapshots update that message rather than billing it
again. Counters are absolute when persisted, so a database retry is idempotent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.db import HeartbeatRun
from zevo.engine.cost.pricing import USAGE_FIELDS, estimate_cost

FIELDS = USAGE_FIELDS
# Cache TTL detail is retained in transcript payloads, without requiring a
# schema migration. Only these aggregate fields exist on HeartbeatRun.
PERSISTED_FIELDS = ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens")


def _valid_cost(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


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
    reported_cost_usd: float | None = None

    def record(self, payload: dict[str, Any]) -> None:
        raw = payload.get("usage")
        if not isinstance(raw, dict):
            return
        current = _counts(raw)
        if payload.get("usage_kind") == "cumulative":
            reported = _valid_cost(payload.get("reported_cost_usd"))
            if reported is None:
                models = payload.get("model_usage")
                if isinstance(models, dict) and models:
                    costs = [_valid_cost(v.get("costUSD")) if isinstance(v, dict) else None
                             for v in models.values()]
                    if all(cost is not None for cost in costs):
                        reported = _valid_cost(sum(cost for cost in costs if cost is not None))
            if reported is not None:
                self.reported_cost_usd = reported
            # Some CLI versions omit the final usage object entirely. Do not
            # erase the known interim spend for an empty result event.
            if raw:
                self.counts = current
                self.cumulative = True
            if reported is not None:
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
        cost = (self.reported_cost_usd if self.reported_cost_usd is not None
                else estimate_cost(model, self.counts))
        return {**{key: self.counts[key] for key in PERSISTED_FIELDS},
                "estimated_cost_usd": round(cost, 6)}


async def persist_usage(
    session: AsyncSession, heartbeat_id: str, model: str, usage: LiveUsage,
) -> None:
    """Caller commits together with the transcript batch; never closes a run."""
    await session.execute(
        update(HeartbeatRun).where(HeartbeatRun.id == heartbeat_id)
        .values(**usage.values(model))
        .execution_options(synchronize_session=False)
    )
