"""Append-only audit log.

Call `await audit(session, event_type, ..., before=..., after=...)`
from any mutating endpoint. Fire-and-forget — if the audit insert
fails (DB down, malformed JSON), we log to stderr and move on; never
let audit failure mask the original operation.

No PII scanning. No retention policy yet (manual delete by date when
the table grows). The point is operations history: who started run X,
who patched agent Y, who cleared a secret.
"""
from __future__ import annotations

import sys
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from zevo.db import AuditEvent


async def audit(
    session: AsyncSession,
    *,
    event_type: str,
    actor: str = "anonymous",
    target_type: str = "",
    target_id: str = "",
    summary: str = "",
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
) -> None:
    """Append one audit row. Never raises; logs failure to stderr."""
    try:
        session.add(AuditEvent(
            event_type=event_type, actor=actor,
            target_type=target_type, target_id=target_id,
            summary=summary[:1000],
            before=before or {}, after=after or {},
        ))
        await session.commit()
    except Exception as e:  # noqa: BLE001
        print(
            f"[audit] WARN: failed to record {event_type!r} for "
            f"{target_type}:{target_id}: {type(e).__name__}: {e}",
            file=sys.stderr, flush=True,
        )
