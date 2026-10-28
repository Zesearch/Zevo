#!/usr/bin/env python3
"""Run one inference suite in independent, GPU-isolated model replicas.

This system helper is deliberately model-agnostic. The agent's predict.py is
still the only code that renders prompts or generates predictions. We only
partition answer-free CSV rows, launch that script, and merge its artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path

import yaml

csv.field_size_limit(1 << 30)


@dataclass
class Part:
    member_index: int
    start: int
    end: int
    member: dict
    questions: str = ""
    output: str = ""
    diagnostics: str = ""
    unparseable: int = 0

    @property
    def size(self) -> int:
        return self.end - self.start


def _csv_rows(path: str) -> tuple[list[str], list[dict[str, str]]]:
    if Path(path).suffix.lower() != ".csv":
        raise ValueError(f"parallel inference requires prepared CSV questions: {path}")
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"questions have no CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _complete(member: dict, expected_rows: int) -> bool:
    try:
        fields, predictions = _csv_rows(member["output"])
        sample_fields, _ = _csv_rows(member["sample_submission"])
        with open(member["diagnostics"], encoding="utf-8") as handle:
            records = json.load(handle)["records"]
        return (
            fields == sample_fields
            and len(predictions) == expected_rows
            and len(records) >= expected_rows
            and [int(record.get("request_index", -1)) for record in records]
            == list(range(len(records)))
            and {int(record.get("row_index", -1)) for record in records}
            == set(range(expected_rows))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _atomic_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    target = path.with_name(path.name + ".parallel.tmp")
    _write_csv(target, fields, rows)
    os.replace(target, path)


def _atomic_json(path: Path, value: dict) -> None:
    target = path.with_name(path.name + ".parallel.tmp")
    _write_json(target, value)
    os.replace(target, path)


def _partitions(
    members: list[dict], worker_count: int, root: Path,
) -> tuple[list[Part], list[int], set[int]]:
    loaded = [_csv_rows(member["questions"]) for member in members]
    counts = [len(rows) for _, rows in loaded]
    total = sum(counts)
    target = max(1, math.ceil(total / max(1, worker_count)))
    parts: list[Part] = []
    complete: set[int] = set()
    for index, member in enumerate(members):
        fields, rows = loaded[index]
        if not rows:
            raise ValueError(f"inference benchmark has no question rows: {member['name']}")
        if _complete(member, len(rows)):
            print(f"[parallel] reuse complete benchmark {member['name']}", flush=True)
            complete.add(index)
            continue
        nshards = min(worker_count, max(1, math.ceil(len(rows) / target)))
        for shard in range(nshards):
            start = len(rows) * shard // nshards
            end = len(rows) * (shard + 1) // nshards
            if start == end:
                continue
            part = Part(index, start, end, member)
            part_dir = root / f"m{index:03d}-s{shard:02d}"
            part.questions = str(part_dir / "questions.csv")
            part.output = str(part_dir / "predictions.csv")
            part.diagnostics = str(part_dir / "diagnostics.json")
            _write_csv(Path(part.questions), fields, rows[start:end])
            parts.append(part)
    return parts, counts, complete


def _allocate(parts: list[Part], worker_count: int) -> list[list[Part]]:
    assigned: list[list[Part]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for part in sorted(parts, key=lambda item: (-item.size, item.member_index, item.start)):
        worker = min(range(worker_count), key=lambda item: (loads[item], item))
        assigned[worker].append(part)
        loads[worker] += part.size
    return assigned


def _cache_signature(model: str, config: dict) -> str | None:
    """Discover compatibility without importing torch or initializing CUDA."""
    import hashlib
    import importlib.metadata
    import platform
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap,driver_version',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
        if gpu.returncode or not gpu.stdout.strip():
            return None
    except (OSError, subprocess.TimeoutExpired):
        return None
    versions = {}
    for package in ('torch', 'vllm', 'triton', 'flashinfer-python', 'nvidia-cutlass-dsl', 'transformers'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'absent'
    identity = {'schema': 1, 'uid': os.getuid(), 'python': sys.version,
                'environment': sys.prefix, 'architecture': platform.machine(),
                'gpu': sorted(set(gpu.stdout.strip().splitlines())), 'packages': versions,
                'model': model, 'engine': _engine_identity(config)}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]


class _CacheSlots(dict):
    def __init__(self):
        super().__init__()
        self.fds = {}


@contextmanager
def _cache_slots(base: dict[str, str], model: str, config: dict, workers: int):
    """Hold exclusive cache leases until every worker has shut down.

    Contention or an unknown GPU/runtime falls back to job-private caches.
    There is no waiting on another run and no deletion of shared artifacts.
    """
    import fcntl
    root = base.get('ZEVO_INFERENCE_CACHE_ROOT')
    if not root and base.get('HF_HOME'):
        root = str(Path(base['HF_HOME']).expanduser().parent / 'inference-runtime-cache')
    slots = _CacheSlots()
    locks = []
    try:
        signature = _cache_signature(model, config) if root else None
        if root and signature:
            directory = Path(root).expanduser().resolve() / f'uid-{os.getuid()}' / signature
            try:
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                for worker in range(workers):
                    for index in range(32):
                        slot = directory / f'slot-{index:02d}'
                        slot.mkdir(exist_ok=True, mode=0o700)
                        lock = (slot / '.lease').open('a+')
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except OSError:
                            lock.close()
                            continue
                        locks.append(lock)
                        reused = (slot / '.initialized').exists()
                        print(f"[cache] worker {worker}: {'existing' if reused else 'new'} isolated cache slot {slot}", flush=True)
                        (slot / '.initialized').touch()
                        slots[worker] = slot
                        slots.fds[worker] = lock.fileno()
                        break
            except OSError as exc:
                print(f'[cache] persistent cache unavailable: {exc}; remaining workers use temporary caches', flush=True)
        if len(slots) < workers:
            print(f'[cache] {workers - len(slots)} worker(s) use temporary caches (no compatible free persistent slot)', flush=True)
        yield slots
    finally:
        for lock in locks:
            lock.close()


def _worker_environment(base: dict[str, str], root: Path, devices: list[str],
                        cache: Path | None = None) -> dict[str, str]:
    env = dict(base)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    # IPC and HOME remain private. Only compiled/runtime caches can persist.
    for key, suffix in (("TMPDIR", ""), ("TMP", ""), ("TEMP", ""), ("HOME", "home")):
        location = root / suffix
        location.mkdir(parents=True, exist_ok=True)
        env[key] = str(location)
    cache_root = cache or root
    for key, suffix in (
        ("XDG_CACHE_HOME", "home/.cache"),
        ("FLASHINFER_CACHE_DIR", "home/.cache/flashinfer"),
        ("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("CUDA_CACHE_PATH", "cuda"), ("VLLM_CACHE_ROOT", "vllm"),
        ("OUTLINES_CACHE_DIR", "outlines"),
    ):
        location = cache_root / suffix
        location.mkdir(parents=True, exist_ok=True)
        env[key] = str(location)
    if cache is not None:
        # Some FlashInfer versions use Path.home() instead of their env knob.
        (root / "home/.cache").symlink_to(cache / "home/.cache", target_is_directory=True)
    if len(env["TMPDIR"]) > 80:
        raise ValueError("worker TMPDIR is too long for vLLM IPC sockets")
    return env


def _emit(ticket: str, payload: dict) -> None:
    payload["t"] = int(time.time())
    print(f"__PROGRESS__:{ticket}:{json.dumps(payload, separators=(',', ':'))}", flush=True)


def _phase(ticket: str, name: str) -> None:
    print(f"__PHASE__:{ticket}:{name}@{int(time.time())}", flush=True)


def _merge(member: dict, parts: list[Part], expected: int) -> int:
    predictions: list[dict] = []
    records: list[dict] = []
    fields: list[str] | None = None
    cursor = 0
    for part in sorted(parts, key=lambda item: item.start):
        if part.start != cursor:
            raise ValueError(f"inference shards have a gap or overlap for {member['name']}")
        part_fields, rows = _csv_rows(part.output)
        if fields is None:
            fields = part_fields
        elif fields != part_fields:
            raise ValueError(f"inference shard columns differ for {member['name']}")
        with open(part.diagnostics, encoding="utf-8") as handle:
            shard_records = json.load(handle)["records"]
        if len(rows) != part.size or len(shard_records) < part.size:
            raise ValueError(f"inference shard row/diagnostics mismatch for {member['name']}")
        if [int(record.get("request_index", -1)) for record in shard_records] != list(
            range(len(shard_records))
        ):
            raise ValueError(f"inference shard request ids are not dense for {member['name']}")
        if {int(record.get("row_index", -1)) for record in shard_records} != set(
            range(part.size)
        ):
            raise ValueError(f"inference shard requests do not cover all rows for {member['name']}")
        predictions.extend(rows)
        request_offset = len(records)
        for offset, record in enumerate(shard_records):
            adjusted = dict(record)
            adjusted["row_index"] = part.start + int(record["row_index"])
            adjusted["request_index"] = request_offset + offset
            records.append(adjusted)
        cursor = part.end
    if cursor != expected or fields is None:
        raise ValueError(f"inference shards do not cover {member['name']}")
    _atomic_csv(Path(member["output"]), fields, predictions)
    _atomic_json(Path(member["diagnostics"]), {"schema_version": 1, "records": records})
    return len(records)


def _load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _engine_identity(config: dict) -> dict:
    # Prompt mappings and sampling parameters do not construct the model. All
    # engine/tokenizer settings must agree before sharing a loaded replica.
    return {key: config.get(key) for key in (
        "base_model", "generation_backend", "tokenizer_source",
        "chat_template_source", "chat_template_hash", "template_kwargs", "special_token_ids",
    )} | {"implementation_config": {
        key: value for key, value in config.get("implementation_config", {}).items()
        if key not in {"raw_output_column"}
    }}


def _calibration_helper():
    # The engine copies this exact standalone helper next to the runner.
    import importlib.util
    path = Path(__file__).with_name("zevo_length_calibration.py")
    spec = importlib.util.spec_from_file_location("zevo_resident_calibration", path)
    if spec is None or spec.loader is None:
        raise ValueError("system length calibration helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resident_worker(predict: str, model: str, manifest: str, summary: str,
                     ticket: str, gate: str, calibrate: bool) -> int:
    """Load once, calibrate assigned whole benchmarks, then generate shards."""
    import importlib.util
    members = json.loads(Path(manifest).read_text())["members"]
    spec = importlib.util.spec_from_file_location("zevo_predict_adapter", predict)
    if spec is None or spec.loader is None:
        raise ValueError("predict module is unavailable")
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    if not callable(getattr(adapter, "load_engine", None)) or not callable(getattr(adapter, "predict_member", None)):
        raise ValueError("resident predict.py requires load_engine and predict_member")
    engine = adapter.load_engine(model, _load_config(members[0]["config"]))
    helper = _calibration_helper() if calibrate else None
    try:
        if calibrate:
            for member in members:
                if not member.get("_calibration_owner"):
                    continue
                questions = member["_full_questions"]
                if _load_config(member["config"]).get("measurement", {}).get("inference_config", {}).get("scoring_mode") == "option_loglikelihood":
                    continue
                # Reuse accepted evidence on a retry, without rerunning samples.
                try:
                    helper.validate(member["config"], questions)
                    continue
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                fields, rows, _ = helper.sample_rows(questions)
                sample_root = Path(member["config"]).parent / "resident-calibration"
                sample_root.mkdir(parents=True, exist_ok=True)
                sample_path = sample_root / "questions.csv"
                _write_csv(sample_path, fields, rows)
                accepted = False
                for index, candidate in enumerate(member["calibration_configs"]):
                    cfg = _load_config(candidate)
                    # Only the selected owner's worker writes the final YAML.
                    Path(member["config"]).write_bytes(Path(candidate).read_bytes())
                    sample = {**member, "questions": str(sample_path),
                              "output": str(sample_root / f"predictions-{index}.csv"),
                              "diagnostics": str(sample_root / f"diagnostics-{index}.json")}
                    _phase(ticket, "length_calibration")
                    adapter.predict_member(engine, sample, cfg, progress=lambda step, total: None)
                    try:
                        helper.check(member["config"], questions, sample["diagnostics"], sample["output"])
                        accepted = True
                        break
                    except ValueError as exc:
                        # Malformed artifacts must not be mistaken for a short budget.
                        if not str(exc).startswith("length calibration failed:"):
                            raise
                        print(f"[calibration] {member['name']} candidate {index + 1}: {exc}", flush=True)
                if not accepted:
                    raise ValueError(f"length calibration exhausted for {member['name']}; inspect evidence before changing the plan")
        print("__ZEVO_RESIDENT_READY__", flush=True)
        # Parent validates every full benchmark before releasing any shard.
        while not Path(gate).exists():
            time.sleep(0.1)
        summaries = []
        for index, member in enumerate(members, 1):
            def progress(step, total):
                _emit(ticket, {"step": step, "total": total,
                               "benchmark_index": index, "benchmark_total": len(members)})
            result = adapter.predict_member(engine, member, _load_config(member["config"]), progress=progress)
            summaries.append(result)
        _atomic_json(Path(summary), {"members": summaries})
        return 0
    finally:
        close = getattr(adapter, "close_engine", None)
        if callable(close):
            close(engine)


def run(*, predict: str, model: str, suite: str, summary: str, ticket: str,
        allocated_gpus: int, gpus_per_worker: int, max_workers: int | None = None,
        resident: bool = False, calibrate: bool = False) -> int:
    with open(suite, encoding="utf-8") as handle:
        members = json.load(handle)["members"]
    if not members:
        raise ValueError("inference suite is empty")
    if calibrate and not resident:
        raise ValueError("in-process calibration requires --resident")
    if resident:
        identity = _engine_identity(_load_config(members[0]["config"]))
        for member in members:
            candidates = member.get("calibration_configs", []) if calibrate else []
            if calibrate and not 1 <= len(candidates) <= 4:
                raise ValueError("each calibration member requires 1–4 prevalidated candidate configs")
            for path in [member["config"], *candidates]:
                if _engine_identity(_load_config(path)) != identity:
                    raise ValueError("resident configs change engine initialization; prepare a compatible context/memory envelope or rebuild explicitly")
            if calibrate:
                baseline = _load_config(candidates[0])
                for path in candidates:
                    candidate = _load_config(path)
                    left, right = json.loads(json.dumps(baseline)), json.loads(json.dumps(candidate))
                    left.get("measurement", {}).pop("max_new_tokens", None)
                    right.get("measurement", {}).pop("max_new_tokens", None)
                    if left != right:
                        raise ValueError("resident calibration candidates may change only measurement.max_new_tokens")
    identities = [str(member.get("benchmark_id") or "").strip() for member in members]
    if any(not identity for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError("inference suite requires a distinct benchmark_id for every member")
    if (allocated_gpus < 1 or gpus_per_worker < 1
            or gpus_per_worker > allocated_gpus
            or (max_workers is not None and max_workers < 1)):
        raise ValueError("invalid GPU allocation or per-worker model-parallel size")
    for member in members:
        with open(member["config"], encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        tp = int(config["implementation_config"]["llm_kwargs"].get("tensor_parallel_size", 1))
        if tp != gpus_per_worker:
            raise ValueError(
                f"{member['name']}: YAML tensor_parallel_size={tp} differs "
                f"from gpus_per_worker={gpus_per_worker}"
            )
    mask = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if allocated_gpus > 1 and len(mask) < allocated_gpus:
        raise ValueError("parallel inference needs Slurm's complete CUDA_VISIBLE_DEVICES mask")
    if not mask:
        mask = [str(index) for index in range(allocated_gpus)]
    if len(mask) < allocated_gpus:
        raise ValueError("fewer visible GPUs than the registered Slurm allocation")
    possible = allocated_gpus // gpus_per_worker
    row_counts = [len(_csv_rows(member["questions"])[1]) for member in members]
    # The caller's resource plan includes model memory, host RAM and CPUs.
    # Partitioning limits concurrency further when there is less work than slots.
    workers = min(max_workers if max_workers is not None else possible,
                  possible, max(1, sum(row_counts)))
    print(
        f"[parallel] {len(members)} benchmarks, {sum(row_counts)} rows, "
        f"{workers} independent replica(s), {gpus_per_worker} GPU(s) each",
        flush=True,
    )
    _phase(ticket, "setup")
    with tempfile.TemporaryDirectory(prefix="zevo-inf-", dir=os.environ.get("TMPDIR")) as scratch, \
            _cache_slots(os.environ, model, _load_config(members[0]["config"]), workers) as cache_slots:
        root = Path(scratch)
        parts, counts, complete = _partitions(members, workers, root)
        already_done_rows = sum(counts[index] for index in complete)
        if not parts:
            print("[parallel] all benchmark artifacts are already complete", flush=True)
        assigned = _allocate(parts, workers)
        active = [(index, batch) for index, batch in enumerate(assigned) if batch]
        owners = {
            index: {
                worker for worker, batch in active
                if any(part.member_index == index for part in batch)
            }
            for index in range(len(members))
        }
        member_positions = {
            index: [
                (worker, local_index, part)
                for worker, batch in active
                for local_index, part in enumerate(batch)
                if part.member_index == index
            ]
            for index in range(len(members))
        }
        progress: dict[tuple[int, int], int] = {}
        committed: set[int] = set()
        request_counts: dict[int, int] = {}
        message_queue: queue.Queue[tuple[int, str | None]] = queue.Queue()
        processes: list[tuple[int, subprocess.Popen[str], list[Part]]] = []

        def commit_member(index: int) -> None:
            member = members[index]
            member_parts = [part for part in parts if part.member_index == index]
            if member_parts:
                request_count = _merge(member, member_parts, counts[index])
            else:
                with open(member["diagnostics"], encoding="utf-8") as handle:
                    request_count = len(json.load(handle)["records"])
            if not _complete(member, counts[index]):
                raise ValueError(
                    f"merged inference artifacts are incomplete for {member['name']}"
                )
            request_counts[index] = request_count
            committed.add(index)
            _emit(ticket, {
                "step": counts[index], "total": counts[index],
                "benchmark_id": member["benchmark_id"],
                "benchmark_name": member["name"], "benchmark_index": index + 1,
                "benchmark_total": len(members), "parallel_workers": workers,
                "suite_rows_completed": max(
                    sum(progress.values()) + already_done_rows,
                    sum(counts[item] for item in committed),
                ),
                "suite_rows_verified": sum(counts[item] for item in committed),
                "benchmark_complete": True,
                "suite_rows_total": sum(counts),
            })

        def commit_ready_members() -> None:
            # A worker can move to another benchmark without exiting. Commit
            # each member as soon as all of its shards are fully written and
            # validated, rather than waiting for that worker's entire batch.
            # Final generation progress may precede the artifact writes, so a
            # step==total report alone must never count as completion.
            for index, positions in member_positions.items():
                if index in committed or not positions:
                    continue
                if not all(
                    progress.get((worker, local_index), 0) >= part.size
                    for worker, local_index, part in positions
                ):
                    continue
                if all(
                    _complete({
                        "output": part.output,
                        "diagnostics": part.diagnostics,
                        "sample_submission": part.member["sample_submission"],
                    }, part.size)
                    for _, _, part in positions
                ):
                    commit_member(index)

        for index in sorted(complete):
            commit_member(index)
        _phase(ticket, "load_model")
        ready_workers: set[int] = set()
        gate = root / "generation-ready"
        helper = _calibration_helper() if calibrate else None
        try:
            for index, batch in active:
                worker_root = root / f"w{index}"
                env = _worker_environment(
                    os.environ, worker_root,
                    mask[index * gpus_per_worker:(index + 1) * gpus_per_worker],
                    cache=cache_slots.get(index),
                )
                worker_members = [
                    {**part.member, "questions": part.questions,
                     "output": part.output, "diagnostics": part.diagnostics,
                     "_full_questions": part.member["questions"],
                     "_calibration_owner": index == min(owners[part.member_index])}
                    for part in batch
                ]
                manifest = root / f"worker-{index}.json"
                worker_summary = root / f"worker-{index}-summary.json"
                _write_json(manifest, {"members": worker_members})
                command = [sys.executable, predict, "--model", model,
                           "--suite", str(manifest), "--summary", str(worker_summary),
                           "--ticket-id", ticket]
                if resident:
                    command = [sys.executable, str(Path(__file__).resolve()),
                               "--resident-worker", "--predict", str(Path(predict).resolve()),
                               "--model", model, "--suite", str(manifest),
                               "--summary", str(worker_summary), "--ticket-id", ticket,
                               "--generation-gate", str(gate)] + (["--calibrate"] if calibrate else [])
                proc = subprocess.Popen(
                    command, env=env, stdout=subprocess.PIPE, stderr=None,
                    text=True, bufsize=1,
                    # Keep the lease if the parent dies before this worker.
                    pass_fds=((cache_slots.fds[index],) if index in cache_slots.fds else ()),
                )
                processes.append((index, proc, batch))

                def read_stdout(worker: int, stream) -> None:
                    for line in stream:
                        message_queue.put((worker, line.rstrip("\n")))
                    message_queue.put((worker, None))

                threading.Thread(
                    target=read_stdout, args=(index, proc.stdout), daemon=True,
                ).start()
            readers_done: set[int] = set()
            successful_workers: set[int] = set()
            generation_started = False
            while len(readers_done) < len(active):
                try:
                    worker, line = message_queue.get(timeout=1)
                except queue.Empty:
                    commit_ready_members()
                    continue
                if line is None:
                    readers_done.add(worker)
                    finished = next(proc for index, proc, _ in processes if index == worker)
                    exit_code = finished.wait()
                    if exit_code != 0:
                        raise RuntimeError(
                            f"inference replica {worker} exited with code {exit_code}"
                        )
                    batch = assigned[worker]
                    with open(root / f"worker-{worker}-summary.json", encoding="utf-8") as handle:
                        worker_summaries = json.load(handle)["members"]
                    if len(worker_summaries) != len(batch):
                        raise ValueError(
                            f"worker {worker} returned incomplete suite summary"
                        )
                    for part, item in zip(batch, worker_summaries):
                        part.unparseable = int(item.get("n_unparseable") or 0)
                    successful_workers.add(worker)
                    commit_ready_members()
                    for member_index in range(len(members)):
                        if (member_index not in committed
                                and owners[member_index] <= successful_workers):
                            commit_member(member_index)
                    continue
                if resident and line == "__ZEVO_RESIDENT_READY__":
                    ready_workers.add(worker)
                    if len(ready_workers) == len(active):
                        if calibrate:
                            for member in members:
                                cfg = _load_config(member["config"])
                                if cfg.get("measurement", {}).get("inference_config", {}).get("scoring_mode") != "option_loglikelihood":
                                    helper.validate(member["config"], member["questions"])
                        gate.touch()
                    continue
                if not line.startswith("__PROGRESS__:"):
                    if line.startswith("__PHASE__:") and resident and "length_calibration@" in line:
                        _phase(ticket, "length_calibration")
                    elif not line.startswith("__PHASE__:"):
                        print(f"[worker {worker}] {line}", flush=True)
                    continue
                try:
                    data = json.loads(line.split(":", 2)[2])
                    local_index = int(data["benchmark_index"]) - 1
                    part = assigned[worker][local_index]
                    step = int(data["step"])
                    if step < 0 or step > part.size:
                        raise ValueError("worker progress exceeds its shard")
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise ValueError(f"invalid inference worker progress: {line}") from exc
                if not generation_started:
                    _phase(ticket, "generate")
                    generation_started = True
                progress[(worker, local_index)] = max(
                    progress.get((worker, local_index), 0), step,
                )
                member_index = part.member_index
                member_step = sum(
                    value for (worker_id, batch_index), value in progress.items()
                    if assigned[worker_id][batch_index].member_index == member_index
                )
                active_names = sorted({
                    assigned[worker_id][batch_index].member["name"]
                    for (worker_id, batch_index), value in progress.items()
                    if value < assigned[worker_id][batch_index].size
                })
                active_ids = sorted({
                    assigned[worker_id][batch_index].member["benchmark_id"]
                    for (worker_id, batch_index), value in progress.items()
                    if value < assigned[worker_id][batch_index].size
                })
                _emit(ticket, {
                    # A benchmark is complete only after its merged artifacts
                    # are durable and checked, not when a shard generated its
                    # final token. The API counts step==total as completed.
                    "step": min(member_step, counts[member_index] - 1),
                    "total": counts[member_index],
                    "benchmark_id": part.member["benchmark_id"],
                    "benchmark_name": part.member["name"],
                    "benchmark_index": member_index + 1,
                    "benchmark_total": len(members),
                    "active_benchmarks": active_names,
                    "active_benchmark_ids": active_ids,
                    "parallel_workers": workers,
                    "suite_rows_completed": sum(progress.values()) + already_done_rows,
                    "suite_rows_total": sum(counts),
                })
                commit_ready_members()
            if len(committed) != len(members):
                raise ValueError("inference workers finished without committing every benchmark")
            _phase(ticket, "write_outputs")
            summaries = [{
                "name": member["name"], "n_rows": counts[index],
                "n_requests": request_counts[index],
                "n_unparseable": sum(part.unparseable for part in parts
                                     if part.member_index == index),
            } for index, member in enumerate(members)]
            _atomic_json(Path(summary), {"members": summaries})
            _phase(ticket, "done")
            return 0
        finally:
            for _, proc, _ in processes:
                if proc.poll() is None:
                    proc.terminate()
            for _, proc, _ in processes:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--ticket-id", required=True)
    parser.add_argument("--allocated-gpus", type=int, default=1)
    parser.add_argument("--gpus-per-worker", type=int, default=1)
    parser.add_argument("--max-workers", type=int,
                        help="Replica limit from the CPU/RAM and GPU resource plan")
    parser.add_argument("--resident", action="store_true", help="Reuse loaded engines across calibration and generation")
    parser.add_argument("--calibrate", action="store_true", help="Gate generation on in-process length calibration")
    parser.add_argument("--resident-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--generation-gate", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.resident_worker:
            return _resident_worker(args.predict, args.model, args.suite, args.summary,
                                    args.ticket_id, args.generation_gate, args.calibrate)
        return run(
            predict=args.predict, model=args.model, suite=args.suite,
            summary=args.summary, ticket=args.ticket_id,
            allocated_gpus=args.allocated_gpus,
            gpus_per_worker=args.gpus_per_worker,
            max_workers=args.max_workers, resident=args.resident, calibrate=args.calibrate,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[parallel] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
