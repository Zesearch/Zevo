"""System-owned Hugging Face Trainer telemetry for Zevo Train stages.

Copy this file beside the generated training script and import the callback.
The helper intentionally has no experiment-policy logic: it only preserves
numeric values the Trainer already reports and stamps them with the Ticket.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
import tempfile
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


def validate_dataloader_runtime(workers: int, plan: dict | None = None) -> None:
    """Check the allocated CPU envelope and temporary filesystem before workers start."""
    if workers <= 0:
        return
    plan = plan or {}
    cpus = plan.get("cpus_per_rank")
    local_world = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    # Slurm can give each rank a separate affinity mask or one shared task mask.
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", str(affinity)))
    tasks_on_node = re.match(r"\d+", os.environ.get("SLURM_NTASKS_PER_NODE", "1"))
    tasks = max(1, int(tasks_on_node.group()) if tasks_on_node else 1)
    ranks_per_task = max(1, math.ceil(local_world / tasks))
    per_rank = min(affinity, max(1, allocated // ranks_per_task))
    if type(cpus) is not int or workers + 1 > cpus or cpus > per_rank:
        raise ValueError("DataLoader CPU plan exceeds the runtime per-rank allocation")
    node_memory = os.environ.get("SLURM_MEM_PER_NODE")
    if node_memory and float(plan.get("memory_budget_gib", 0)) > float(node_memory) / 1024 / local_world:
        raise ValueError("DataLoader memory plan exceeds the runtime per-rank allocation")
    temp = Path(tempfile.gettempdir()).resolve()
    mounts = []
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            mount = Path(left.split()[4].replace("\\040", " "))
            if temp == mount or mount in temp.parents:
                mounts.append((len(mount.parts), right.split()[0]))
    except (OSError, ValueError, IndexError) as exc:
        raise ValueError("Cannot verify DataLoader temporary filesystem") from exc
    filesystem = max(mounts, default=(0, ""))[1]
    if filesystem not in {"tmpfs", "ramfs", "ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "overlay"}:
        raise ValueError("DataLoader workers require a verified node-local TMPDIR")
    # Verify creation and cleanup on the same inherited directory as workers.
    with tempfile.TemporaryDirectory(prefix="zevo-loader-", dir=temp) as directory:
        Path(directory, "write-check").write_text("ok")


class ZevoTrainerTelemetryCallback(TrainerCallback):
    """Forward Trainer metrics and identify each real training process.

    A Ticket heartbeat may repair an implementation failure and launch Trainer
    again.  Each callback instance therefore owns a fresh UUID.  The UUID is
    emitted before optimization starts and repeated on every progress marker,
    so persistence and the UI never splice a restarted run into the failed
    curve merely because both executions reported the same optimizer steps.
    """

    def __init__(self, ticket_id: str, dataloader_worker_plan: dict | None = None) -> None:
        if not ticket_id.strip():
            raise ValueError("ticket_id is required for Zevo telemetry")
        self.dataloader_worker_plan = dataloader_worker_plan
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
        validate_dataloader_runtime(
            int(getattr(args, "dataloader_num_workers", 0)), self.dataloader_worker_plan,
        )
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
