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
