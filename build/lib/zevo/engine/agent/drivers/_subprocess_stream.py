"""Subprocess stream helpers shared by the CLI drivers.

``StreamReader.readline()`` is implemented with ``readuntil()`` and therefore
inherits asyncio's small separator limit. CLI JSONL events legitimately exceed
that limit when a shell tool returns a long line (for example minified JSON or
an accidentally-read binary). Read fixed-size chunks and frame newlines here so
one large event cannot crash the entire Ticket before the CLI can recover.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable


async def iter_subprocess_lines(
    stream: asyncio.StreamReader,
    *,
    chunk_size: int = 64 * 1024,
    on_chunk: Callable[[bytes], None] | None = None,
) -> AsyncIterator[bytes]:
    """Yield complete newline-delimited records without ``readline`` limits.

    The final unterminated record is yielded at EOF. Memory use is proportional
    to the largest single CLI event, which must already be materialized by the
    JSON parser; ordinary multi-line output is released one line at a time.
    """
    chunk_size = max(1, int(chunk_size))
    pending = bytearray()
    while True:
        chunk = await stream.read(chunk_size)
        if not chunk:
            break
        # Report raw-byte activity before waiting for a newline. Claude can
        # emit one very large JSONL event over several reads; a watchdog that
        # only observes completed lines would incorrectly call that silence.
        if on_chunk is not None:
            on_chunk(chunk)
        pending.extend(chunk)
        consumed = 0
        while True:
            newline = pending.find(b"\n", consumed)
            if newline < 0:
                break
            yield bytes(pending[consumed:newline + 1])
            consumed = newline + 1
        if consumed:
            del pending[:consumed]
    if pending:
        yield bytes(pending)
