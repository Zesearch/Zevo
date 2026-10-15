"""Parse Ticket-owned attempt/phase/progress/config markers from agent stdout.

Modeled on TuneLLM/agent/agent/job_handler.py. Scripts emit case-sensitive
__ATTEMPT__, __PHASE__, __PROGRESS__, and __CONFIG__ lines. Nothing may precede
the marker on its line; anything else is ignored.

Examples emitted by train.py / predict.py templates:
    __ATTEMPT__:train-abcd1234-001:550e8400-e29b-41d4-a716-446655440000@1723723199.0
    __PHASE__:train-abcd1234-001:loading_model@1723723200.0
    __PROGRESS__:train-abcd1234-001:{"step": 100, "loss": 1.234, "t": 1723723201.0}
    __PHASE__:train-abcd1234-001:saving_model@1723723300.0

parse_line() returns one of:
    ("attempt", {"attempt_id": uuid, "owner": ticket, "t": unix_seconds_or_None})
    ("phase", {"phase": name, "owner": ticket, "t": unix_seconds_or_None})
    ("progress", {**parsed_json, "owner": ticket})
    ("config", {**parsed_json, "owner": ticket})
    None
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Literal


PHASE_PREFIX = "__PHASE__:"
# A phase name is a short identifier -- every one the templates emit is
# snake_case ("loading_model", "saving_model"). Bounding it is what keeps a
# stray match from becoming a phase: drivers scan tool results for markers, and
# a tool result that echoes a script back contains the script's own
# `print("__PHASE__:...")` lines. Those arrive JSON-escaped as ONE line, so
# there is no newline to stop at and the "name" ran to the end of the file --
# thousands of characters, over the 64-char column, failing the whole batch
# insert and losing every real phase alongside it.
PHASE_MAX_LEN = 64
_PHASE_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
PROGRESS_PREFIX = "__PROGRESS__:"
CONFIG_PREFIX = "__CONFIG__:"
ATTEMPT_PREFIX = "__ATTEMPT__:"


ParsedMarker = (
    tuple[Literal["attempt"], dict[str, Any]]
    |
    tuple[Literal["phase"], dict[str, Any]]
    | tuple[Literal["progress"], dict[str, Any]]
    | tuple[Literal["config"], dict[str, Any]]
    | None
)


_OWNER_RE = re.compile(r"^([a-z][a-z0-9_-]*-\d+):")


def _split_owner(body: str) -> tuple[str, str]:
    """Peel the required `<ticket-id>:` prefix off a marker's payload.

    A marker read out of a file says nothing about who produced it: an agent
    that tails another stage's log, or cats a script, sees perfectly well-formed
    markers belonging to someone else, and they are indistinguishable from the
    ones its own program just printed. Stamping the emitting ticket into the
    marker is what makes them distinguishable. Unstamped markers are rejected.
    """
    m = _OWNER_RE.match(body)
    return (m.group(1), body[m.end():]) if m else ("", body)


# A marker may carry the moment it was PRINTED, as `@<unix-seconds>` after the
# phase name or as a `t` key in a progress/config payload.
#
# Without it the only time we have is when the harness first SAW the line, and
# for a stage that runs on another machine that is one instant for the whole
# log: the agent waits for the job, reads the file in one go, and every marker
# in it lands on the same timestamp. Ordering and spacing are then lost even
# though the run took minutes. The emitter knows the real time; let it say so.
_AT_RE = re.compile(r"@(\d+(?:\.\d+)?)$")
_ATTEMPT_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _split_time(name: str) -> tuple[str, float | None]:
    m = _AT_RE.search(name)
    if not m:
        return name, None
    try:
        return name[:m.start()], float(m.group(1))
    except ValueError:
        return name, None


def parse_line(line: str) -> ParsedMarker:
    """Try to parse a marker. Returns None on miss."""
    s = line.rstrip("\n")
    if s.startswith(ATTEMPT_PREFIX):
        owner, body = _split_owner(s[len(ATTEMPT_PREFIX):].strip())
        if not owner:
            return None
        attempt_id = body.split(maxsplit=1)[0] if body else ""
        attempt_id, at = _split_time(attempt_id)
        if not _ATTEMPT_ID_RE.fullmatch(attempt_id):
            return None
        return (
            "attempt",
            {"attempt_id": attempt_id, "owner": owner, "t": at},
        )
    if s.startswith(PHASE_PREFIX):
        # Stop at the first whitespace: a marker glued to trailing text (a tqdm
        # fragment, a quote-and-comma from echoed source) still yields its name.
        owner, body = _split_owner(s[len(PHASE_PREFIX):].strip())
        if not owner:
            return None
        name = body.split(maxsplit=1)[0] if body else ""
        name, at = _split_time(name)
        if not name or len(name) > PHASE_MAX_LEN or not set(name) <= _PHASE_OK:
            return None
        return ("phase", {"phase": name, "owner": owner, "t": at})
    if s.startswith(PROGRESS_PREFIX):
        owner, body = _split_owner(s[len(PROGRESS_PREFIX):].strip())
        if not owner:
            return None
        try:
            # raw_decode, not loads: tqdm writes its bar with \r and no newline,
            # so a marker arrives glued to whatever follows it on the line.
            data, _ = json.JSONDecoder().raw_decode(body)
            if isinstance(data, dict):
                return ("progress", {**data, "owner": owner})
        except json.JSONDecodeError:
            return None
    if s.startswith(CONFIG_PREFIX):
        owner, body = _split_owner(s[len(CONFIG_PREFIX):].strip())
        if not owner:
            return None
        try:
            data, _ = json.JSONDecoder().raw_decode(body)
            if isinstance(data, dict):
                return ("config", {**data, "owner": owner})
        except json.JSONDecodeError:
            return None
    return None


class MarkerStream:
    """Stateful stdout-line consumer.

    Tracks the most recently seen phase so PROGRESS lines can be tagged
    with it without scripts needing to repeat the phase name. Calls the
    supplied on_phase / on_progress callbacks for each parsed marker.
    """

    def __init__(
        self,
        *,
        on_attempt: Callable[[str], None] | None = None,
        on_phase: Callable[[str], None] | None = None,
        on_progress: Callable[[str, dict[str, Any]], None] | None = None,
        on_config: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.current_phase = ""
        self.current_attempt_id = ""
        self.on_attempt = on_attempt
        self.on_phase = on_phase
        self.on_progress = on_progress
        self.on_config = on_config

    def feed(self, line: str) -> None:
        parsed = parse_line(line)
        if parsed is None:
            return
        kind, payload = parsed
        if kind == "attempt":
            attempt_id = str(payload.get("attempt_id") or "")
            self.current_attempt_id = attempt_id
            if self.on_attempt is not None:
                self.on_attempt(attempt_id)
        elif kind == "phase":
            name = payload.get("phase", "") if isinstance(payload, dict) else str(payload)
            self.current_phase = name
            if self.on_phase is not None:
                self.on_phase(name)
        elif kind == "progress":
            if self.on_progress is not None:
                if self.current_attempt_id:
                    payload.setdefault("attempt_id", self.current_attempt_id)
                self.on_progress(self.current_phase, payload)  # type: ignore[arg-type]
        elif kind == "config":
            if self.on_config is not None:
                self.on_config(payload)  # type: ignore[arg-type]


_MARKER_RE = re.compile(r"__(?:ATTEMPT|PHASE|PROGRESS|CONFIG)__:")

# A payload ends at the first line break. Two kinds:
#   real     — an ordinary newline in the stream
#   literal  — the two characters \ and n, which is what a newline becomes when
#              a log is carried back through JSON-escaped tool output; a remote
#              stage's log arrives as one enormous physical line that way.
# The literal cut applies to __ATTEMPT__ and __PHASE__. A __PROGRESS__ / __CONFIG__ payload
# is itself JSON, and any string inside it that contains a newline carries that
# same two-character escape -- cutting there truncates the JSON mid-value. One
# `prompt_render` field (a rendered chat template, newlines and all) was enough
# to make every config marker unparseable, so the whole Inference settings panel
# vanished. JSON knows where it ends; let raw_decode find it.
_BREAKS_REAL = ("\n", "\r")
_BREAKS_ESCAPED = ("\\n", "\\r")


def scan_text(text: str) -> list[ParsedMarker]:
    """Every marker in `text`, in the order it appears.

    Callers used to look for one prefix at a time and keep a single hit per
    line. On a remote stage that loses almost everything: the log comes back
    with its newlines escaped, so hundreds of markers share one physical line,
    and taking one meant discarding the rest. Because progress was checked
    first, the discarded ones were always the __PHASE__ markers -- a training
    run recorded its steps but never the phase they belonged to, leaving every
    step filed under whatever phase happened to be current before the run.

    Each marker is bounded by the start of the next one, so a marker glued to
    the end of a tqdm fragment still parses.
    """
    out: list[ParsedMarker] = []
    hits = list(_MARKER_RE.finditer(text))
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        seg = text[m.start():end]
        breaks = _BREAKS_REAL
        # A payload that is JSON knows where it ends (`raw_decode` finds it), so
        # cutting it at a literal "\n" would truncate a value that legitimately
        # contains one. A phase name has no such boundary — it ends at the line —
        # and out of a JSON-escaped log the only line break present is the escaped
        # one, so it must stop there or swallow the whole file.
        if seg.startswith((ATTEMPT_PREFIX, PHASE_PREFIX)):
            breaks = _BREAKS_REAL + _BREAKS_ESCAPED
        cuts = [seg.find(b) for b in breaks if seg.find(b) != -1]
        if cuts:
            seg = seg[:min(cuts)]
        parsed = parse_line(seg)
        if parsed is not None:
            out.append(parsed)
    return out
