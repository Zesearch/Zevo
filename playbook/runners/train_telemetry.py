"""System-owned Hugging Face Trainer telemetry for Zevo Train stages.

Copy this file beside the generated training script and import the callback.
The helper intentionally has no experiment-policy logic: it only preserves
numeric values the Trainer already reports and stamps them with the Ticket.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from numbers import Real
from typing import Any

try:
    from transformers import TrainerCallback
except ImportError:  # lets contract/unit checks import without GPU dependencies
    class TrainerCallback:  # type: ignore[no-redef]
        pass


TELEMETRY_INTERVAL_STEPS = 20


def _finite_number(value: Any) -> int | float | None:
    """Return a JSON-safe finite scalar, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    converted = float(value)
    if not math.isfinite(converted):
        return None
    if isinstance(value, int):
        return int(value)
    return converted


class ZevoTrainerTelemetryCallback(TrainerCallback):
    """Forward Trainer metrics and identify each real training process.

    A Ticket heartbeat may repair an implementation failure and launch Trainer
    again.  Each callback instance therefore owns a fresh UUID.  The UUID is
    emitted before optimization starts and repeated on every progress marker,
    so persistence and the UI never splice a restarted run into the failed
    curve merely because both executions reported the same optimizer steps.
    """

    def __init__(self, ticket_id: str) -> None:
        if not ticket_id.strip():
            raise ValueError("ticket_id is required for Zevo telemetry")
        self.ticket_id = ticket_id
        self.attempt_id = str(uuid.uuid4())
        self._attempt_announced = False

    def _announce_attempt(self) -> None:
        if self._attempt_announced:
            return
        print(
            f"__ATTEMPT__:{self.ticket_id}:{self.attempt_id}@{time.time()}",
            flush=True,
        )
        self._attempt_announced = True

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        if getattr(state, "is_world_process_zero", True):
            self._announce_attempt()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not getattr(state, "is_world_process_zero", True) or not logs:
            return control
        # Some Trainer integrations call on_log without on_train_begin in a
        # smoke path. Announce lazily too; the guard keeps the marker singular.
        self._announce_attempt()
        payload: dict[str, str | int | float] = {
            "attempt_id": self.attempt_id,
            "step": int(getattr(state, "global_step", 0) or 0),
            "total": int(getattr(state, "max_steps", 0) or 0),
            "t": time.time(),
        }
        has_metric = False
        for key, raw_value in logs.items():
            value = _finite_number(raw_value)
            if value is not None:
                payload[str(key)] = value
                has_metric = True
        if has_metric:
            print(
                f"__PROGRESS__:{self.ticket_id}:"
                + json.dumps(payload, separators=(",", ":"), sort_keys=True),
                flush=True,
            )
        return control
