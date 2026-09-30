"""System-owned checkpoint commit helpers for generated Train programs.

The experiment may choose a Trainer and distributed strategy, but it must not
invent the filesystem transaction that turns live weights into a durable Zevo
checkpoint.  These helpers keep slow rank-zero I/O outside the NCCL process
group, shard full models, validate their file graph, and publish one atomic
commit marker.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MARKER_NAME = ".zevo-checkpoint.json"
DEFAULT_MAX_SHARD_SIZE = "2GB"


@dataclass(frozen=True)
class RankZeroSaveContext:
    """Result of detaching ordinary DDP workers before slow checkpoint I/O."""

    is_writer: bool
    model: Any | None


def detach_ddp_workers(trainer: Any) -> RankZeroSaveContext:
    """Synchronize once, unwrap the model, then end NCCL before rank-zero I/O.

    Call this only for ordinary replicated DDP. FSDP and DeepSpeed need their
    framework's collective state-dict materialization first; once that has
    written a staging directory they can use :func:`commit_checkpoint_directory`.
    Non-zero ranks receive ``is_writer=False`` and should return normally from
    the generated Train program. No collective may follow this call.
    """
    accelerator = trainer.accelerator
    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(trainer.model)
    rank = int(os.environ.get("RANK", "0"))
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except (ImportError, RuntimeError):
        # A single-process execution or an already-destroyed group has no
        # distributed resource left to release.
        pass
    return RankZeroSaveContext(is_writer=rank == 0, model=model if rank == 0 else None)


def _checkpoint_files(root: Path) -> tuple[list[str], list[str]]:
    names = sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    )
    empty = [name for name in names if (root / name).stat().st_size <= 0]
    if empty:
        raise ValueError(f"checkpoint contains empty files: {empty}")
    weights = [
        name for name in names
        if name.endswith((".safetensors", ".bin"))
        and not name.endswith(("optimizer.bin", "scheduler.bin"))
    ]
    configuration = next(
        (name for name in ("config.json", "adapter_config.json") if name in names),
        "",
    )
    if not configuration or not weights:
        raise ValueError(
            "checkpoint requires config.json or adapter_config.json and "
            "non-empty model weights"
        )
    for index_name in (
        "model.safetensors.index.json", "pytorch_model.bin.index.json",
    ):
        index_path = root / index_name
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        referenced = sorted(set((index.get("weight_map") or {}).values()))
        missing = [name for name in referenced if not (root / name).is_file()]
        if missing:
            raise ValueError(f"checkpoint index references missing shards: {missing}")
    return names, weights


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def commit_checkpoint_directory(
    staging_dir: str | os.PathLike[str],
    final_dir: str | os.PathLike[str],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a staged checkpoint and atomically publish the directory."""
    staging = Path(staging_dir).resolve()
    final = Path(final_dir).resolve()
    if staging == final or staging.parent != final.parent:
        raise ValueError("checkpoint staging and final directories must be siblings")
    if final.is_dir() and (final / MARKER_NAME).is_file():
        _checkpoint_files(final)
        return json.loads((final / MARKER_NAME).read_text(encoding="utf-8"))
    if not staging.is_dir():
        raise ValueError(f"checkpoint staging directory does not exist: {staging}")
    names, weights = _checkpoint_files(staging)
    receipt = {
        "schema_version": 1,
        "committed_at_unix": int(time.time()),
        "checkpoint_path": str(final),
        "files": names,
        "file_sizes": {name: (staging / name).stat().st_size for name in names},
        "weight_files": weights,
        "weight_bytes": sum((staging / name).stat().st_size for name in weights),
        "metadata": dict(metadata or {}),
    }
    _atomic_json(staging / MARKER_NAME, receipt)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise ValueError(f"uncommitted checkpoint destination already exists: {final}")
    os.replace(staging, final)
    return receipt


def save_transformers_model_rank_zero(
    trainer: Any,
    tokenizer: Any,
    final_dir: str | os.PathLike[str],
    *,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    safe_serialization: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Commit a full Transformers model after an ordinary replicated DDP run.

    All ranks call the function. Non-zero ranks detach and return ``None``;
    rank zero writes bounded shards to a sibling staging directory and commits
    it atomically. Generated code must return normally when it receives None.
    """
    context = detach_ddp_workers(trainer)
    if not context.is_writer:
        return None
    final = Path(final_dir).resolve()
    staging = final.with_name(final.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    context.model.save_pretrained(
        staging,
        safe_serialization=safe_serialization,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(staging)
    return commit_checkpoint_directory(staging, final, metadata=metadata)


def record_training_plan(path, *, ticket_id, config_path, planned_steps, planned_epochs):
    """Freeze the resolved full-run target before the first optimizer step."""
    import hashlib
    import yaml
    config_bytes = Path(config_path).read_bytes()
    training = yaml.safe_load(config_bytes)["training"]
    if type(planned_steps) is not int or planned_steps <= 0:
        raise ValueError("planned_steps must be a positive resolved optimizer-step count")
    if planned_epochs != training["num_epochs"]:
        raise ValueError("resolved epochs differ from the configured training plan")
    policy = training.get("early_stopping")
    if policy is not None:
        policy = {**policy, "min_delta": policy.get("min_delta", 0.0)}
    plan = {
        "ticket_id": ticket_id,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "planned_steps": planned_steps,
        "planned_epochs": planned_epochs,
        "early_stopping": policy,
    }
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("training plan already exists with different targets")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, plan)
    return plan


def _validate_training_progress(plan, state):
    import math
    target = plan.get("planned_steps")
    step = state.get("global_step")
    if type(target) is not int or target <= 0 or type(step) is not int or step <= 0:
        raise ValueError("training completion requires positive planned and completed steps")
    if state.get("max_steps") != target:
        raise ValueError("runtime step target differs from the frozen training plan")
    if step >= target:
        return "plan_completed"
    policy = plan.get("early_stopping")
    if not policy:
        raise ValueError(f"training incomplete: {step}/{target} optimizer steps")
    best = None
    stale = 0
    last_step = -1
    for event in state.get("log_history", []):
        if policy["metric"] not in event:
            continue
        event_step = event.get("step", -1)
        if type(event_step) is not int or event_step <= last_step or event_step > step:
            raise ValueError("early-stopping evaluations must have increasing runtime steps")
        value = float(event[policy["metric"]])
        if not math.isfinite(value):
            raise ValueError("early-stopping metric must be finite")
        improvement = (best - value if policy["mode"] == "min" else value - best) if best is not None else None
        if best is None or improvement > policy["min_delta"]:
            best, stale = value, 0
        else:
            stale += 1
        last_step = event_step
    if stale >= policy["patience"] and last_step == step:
        return "early_stopping"
    raise ValueError(f"training incomplete: {step}/{target}; declared early stopping not reached")


def commit_training_checkpoint(staging_dir, final_dir, *, plan_path, trainer_state_path):
    """Publish a final model only after its original training target is satisfied."""
    plan = json.loads(Path(plan_path).read_text())
    state = json.loads(Path(trainer_state_path).read_text())
    reason = _validate_training_progress(plan, state)
    metric = (plan.get("early_stopping") or {}).get("metric")
    completion = {"plan": plan, "state": {
        "global_step": state["global_step"], "max_steps": state["max_steps"],
        "log_history": [{"step": row.get("step"), metric: row[metric]}
                        for row in state.get("log_history", []) if metric and metric in row],
    }, "reason": reason}
    result = commit_checkpoint_directory(staging_dir, final_dir, metadata={"training_completion": completion})
    if result.get("metadata", {}).get("training_completion") != completion:
        raise ValueError("published checkpoint belongs to a different training completion")
    return result


def verify_training_completion(checkpoint_dir, *, ticket_id, config_sha256, training):
    """Verify the actual published files and completion record, independently of Result prose."""
    root = Path(checkpoint_dir)
    receipt = json.loads((root / MARKER_NAME).read_text())
    _checkpoint_files(root)
    for name, size in receipt.get("file_sizes", {}).items():
        file = root / name
        if not file.is_file() or file.stat().st_size != size:
            raise ValueError(f"committed checkpoint file missing or changed: {name}")
    if not receipt.get("file_sizes"):
        raise ValueError("checkpoint lacks a verifiable file manifest")
    completion = receipt.get("metadata", {}).get("training_completion")
    if not completion:
        raise ValueError("checkpoint exists but has no training completion evidence")
    plan = completion["plan"]
    if plan.get("ticket_id") != ticket_id or plan.get("config_sha256") != config_sha256:
        raise ValueError("training completion belongs to another Ticket or configuration")
    if plan.get("planned_epochs") != training["num_epochs"] or plan.get("early_stopping") != training.get("early_stopping"):
        raise ValueError("training completion changed the configured termination policy")
    reason = _validate_training_progress(plan, completion["state"])
    if reason != completion.get("reason"):
        raise ValueError("training completion reason is inconsistent")
    return {"reason": reason, "completed_steps": completion["state"]["global_step"],
            "planned_steps": plan["planned_steps"]}
