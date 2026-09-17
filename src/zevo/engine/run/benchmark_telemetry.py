"""Stable namespace for progress steps emitted by a combined benchmark job."""
from __future__ import annotations

from hashlib import blake2s


def benchmark_progress_phase(phase: str, benchmark_name: str) -> str:
    """Keep equal row-step numbers from different benchmarks distinct.

    ExecutionEvent's unique progress key includes phase but not JSON extras.
    The full human-readable benchmark name remains in extras; a short stable
    suffix makes the existing key safe without changing the database schema.
    """
    name = benchmark_name.strip()
    if not name:
        return phase
    digest = blake2s(name.encode("utf-8"), digest_size=6).hexdigest()
    return f"{(phase or 'inference')[:50]}:{digest}"
