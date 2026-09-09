from __future__ import annotations

import asyncio

import pytest

from zevo.engine.agent.drivers._subprocess_stream import iter_subprocess_lines


@pytest.mark.asyncio
async def test_long_unbroken_cli_event_does_not_hit_streamreader_limit() -> None:
    stream = asyncio.StreamReader(limit=32)
    # The incident was a 1.4 MiB /bin/bash accidentally read through `cat $0`.
    # Stay above that exact failure size, not merely above asyncio's 64 KiB
    # default.
    long_event = b"x" * (2 * 1024 * 1024)
    stream.feed_data(long_event + b"\nnext\n")
    stream.feed_eof()

    lines = [line async for line in iter_subprocess_lines(stream, chunk_size=4096)]

    assert lines == [long_event + b"\n", b"next\n"]


@pytest.mark.asyncio
async def test_final_unterminated_cli_event_is_preserved() -> None:
    stream = asyncio.StreamReader(limit=8)
    stream.feed_data(b"first\nlast-without-newline")
    stream.feed_eof()

    lines = [line async for line in iter_subprocess_lines(stream, chunk_size=3)]

    assert lines == [b"first\n", b"last-without-newline"]
