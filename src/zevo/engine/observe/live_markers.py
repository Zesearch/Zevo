"""Incrementally read marker lines from logs that are still growing.

Remote stages normally pipe their output through ``tee`` into the Ticket work
directory. CLI drivers do not necessarily return that output until the remote
command exits, so the runner uses this reader to discover markers directly from
the local log while the command is active.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from zevo.engine.observe.markers import scan_text


_LINE_BREAK = re.compile(br"[\r\n]+")


@dataclass
class _FileState:
    offset: int = 0
    remainder: bytes = b""


@dataclass
class LiveMarkerReader:
    """Read each completed log segment once, including pre-existing content."""

    root: Path
    owner: str
    start_at_end: bool = False
    states: dict[Path, _FileState] = field(default_factory=dict)
    phases: dict[Path, str] = field(default_factory=dict)
    attempt_ids: dict[Path, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.start_at_end:
            return
        # A re-awakened Ticket can reuse a work directory containing an old
        # train.log. The live runner follows bytes produced by THIS activation;
        # the recovery CLI intentionally keeps the default backfill behaviour.
        for path in sorted(self.root.glob("*.log")):
            try:
                if path.is_file():
                    self.states[path] = _FileState(offset=path.stat().st_size)
            except OSError:
                continue

    def poll(self, *, finish: bool = False) -> list[tuple[str, dict[str, Any]]]:
        observed: list[tuple[str, dict[str, Any]]] = []
        for path in sorted(self.root.glob("*.log")):
            if not path.is_file():
                continue
            state = self.states.setdefault(path, _FileState())
            try:
                size = path.stat().st_size
                if size < state.offset:  # log was truncated or replaced
                    state.offset = 0
                    state.remainder = b""
                with path.open("rb") as fh:
                    fh.seek(state.offset)
                    chunk = fh.read()
                state.offset += len(chunk)
            except OSError:
                continue

            data = state.remainder + chunk
            parts = _LINE_BREAK.split(data)
            if finish or data.endswith((b"\r", b"\n")):
                complete, state.remainder = parts, b""
            else:
                complete, state.remainder = parts[:-1], parts[-1]

            for raw in complete:
                if b"__" not in raw:
                    continue
                for parsed in scan_text(raw.decode("utf-8", errors="replace")):
                    if parsed is None:
                        continue
                    kind, payload = parsed
                    payload = dict(payload)
                    marker_owner = str(payload.pop("owner", "") or "")
                    if marker_owner and marker_owner != self.owner:
                        continue
                    if kind == "attempt":
                        self.attempt_ids[path] = str(payload.get("attempt_id") or "")
                    elif kind == "phase":
                        self.phases[path] = str(payload.get("phase") or "")
                    elif kind == "progress" and self.phases.get(path):
                        payload.setdefault("phase", self.phases[path])
                    if kind in {"phase", "progress"} and self.attempt_ids.get(path):
                        payload.setdefault("attempt_id", self.attempt_ids[path])
                    observed.append((kind, payload))
        return observed


def _request_json(url: str, *, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urlopen(request, timeout=5) as response:  # noqa: S310 - internal operator URL
        return json.loads(response.read().decode("utf-8"))


def relay_log(root: Path, ticket_id: str, api_base: str, interval: float = 2.0) -> None:
    """Relay an already-active Ticket's local markers to the progress API."""
    reader = LiveMarkerReader(root=root, owner=ticket_id)
    progress_url = f"{api_base.rstrip('/')}/tickets/{ticket_id}/progress"
    ticket_url = f"{api_base.rstrip('/')}/tickets/{ticket_id}"
    pending: list[tuple[str, dict[str, Any]]] = []
    while True:
        try:
            pending.extend(reader.poll())
            while pending:
                kind, payload = pending[0]
                _request_json(progress_url, body={"kind": kind, **payload})
                pending.pop(0)
            status = str(_request_json(ticket_url).get("status") or "")
            if status not in {"todo", "blocked", "running", "awaiting_input"}:
                pending.extend(reader.poll(finish=True))
                while pending:
                    kind, payload = pending[0]
                    _request_json(progress_url, body={"kind": kind, **payload})
                    pending.pop(0)
                return
        except (OSError, URLError, json.JSONDecodeError):
            # Keep the first unsent marker queued and retry it before reading
            # further. The terminal runner sweep remains the final fallback.
            pass
        time.sleep(max(interval, 0.25))


def main() -> None:
    parser = argparse.ArgumentParser(description="Relay live Zevo log markers")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ticket", required=True)
    parser.add_argument("--api", default="http://backend:8000/api")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    relay_log(args.root, args.ticket, args.api, args.interval)


if __name__ == "__main__":
    main()
