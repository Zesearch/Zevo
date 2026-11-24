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
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

# The deployed module and repository source expose the identical launch API.
import importlib.util
_runtime_path = Path(__file__).with_name("zevo_gpu_runtime.py")
if not _runtime_path.is_file():
    _runtime_path = Path(__file__).with_name("gpu_runtime.py")
_runtime_spec = importlib.util.spec_from_file_location("zevo_gpu_runtime", _runtime_path)
gpu_runtime = importlib.util.module_from_spec(_runtime_spec)
_runtime_spec.loader.exec_module(gpu_runtime)


_resume_path = Path(__file__).with_name("zevo_stage_resume.py")
if not _resume_path.is_file():
    _resume_path = Path(__file__).with_name("stage_resume.py")
_resume_spec = importlib.util.spec_from_file_location("zevo_stage_resume", _resume_path)
stage_resume = importlib.util.module_from_spec(_resume_spec)
_resume_spec.loader.exec_module(stage_resume)


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


def _identity_matches(member):
    try:
        proof = json.loads(Path(member["output"] + ".resume.json").read_text())
        return (proof["identity"] == member["_identity"]
                and proof["predictions"] == stage_resume.digest(Path(member["output"]).read_text())
                and proof["diagnostics"] == stage_resume.digest(Path(member["diagnostics"]).read_text()))
    except (OSError, ValueError, KeyError):
        return False


def _member_identity(member, model, predict):
    return stage_resume.digest({"model": model, "predict": Path(predict).read_text(),
                               "config": _load_config(member["config"]),
                               "questions": Path(member["questions"]).read_text(),
                               "schema": Path(member["sample_submission"]).read_text(),
                               "benchmark": member["benchmark_id"]})


def _partitions(
    members: list[dict], root: Path,
) -> tuple[list[Part], list[int], set[int]]:
    loaded = [_csv_rows(member["questions"]) for member in members]
    counts = [len(rows) for _, rows in loaded]
    parts: list[Part] = []
    complete: set[int] = set()
    for index, member in enumerate(members):
        fields, rows = loaded[index]
        if not rows:
            raise ValueError(f"inference benchmark has no question rows: {member['name']}")
        if _complete(member, len(rows)) and _identity_matches(member):
            print(f"[parallel] reuse complete benchmark {member['name']}", flush=True)
            complete.add(index)
            continue
        # Fixed boundaries survive a retry with a different replica count.
        for shard, start in enumerate(range(0, len(rows), 128)):
            end = min(start + 128, len(rows))
            part = Part(index, start, end, member)
            part_dir = root / f"m{index:03d}-s{shard:02d}"
            part.questions = str(part_dir / "questions.csv")
            part.output = str(part_dir / "predictions.csv")
            part.diagnostics = str(part_dir / "diagnostics.json")
            _write_csv(Path(part.questions), fields, rows[start:end])
            parts.append(part)
    return parts, counts, complete


def _worker_environment(base: dict[str, str], root: Path, devices: list[str]) -> dict[str, str]:
    env = dict(base)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    # IPC and HOME remain private. Only compiled/runtime caches can persist.
    for key, suffix in (("TMPDIR", ""), ("TMP", ""), ("TEMP", ""), ("HOME", "home")):
        location = root / suffix
        location.mkdir(parents=True, exist_ok=True)
        env[key] = str(location)
    cache_root = root
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
    if len(env["TMPDIR"]) > 80:
        raise ValueError("worker TMPDIR is too long for vLLM IPC sockets")
    return env


def _emit(ticket: str, payload: dict) -> None:
    if os.environ.get("ZEVO_INFERENCE_PROGRESS_SCOPE") == "calibration":
        payload = {**payload, "progress_scope": "calibration"}
    payload["t"] = int(time.time())
    print(f"__PROGRESS__:{ticket}:{json.dumps(payload, separators=(',', ':'))}", flush=True)


def _phase(ticket: str, name: str) -> None:
    if os.environ.get("ZEVO_INFERENCE_PROGRESS_SCOPE") == "calibration":
        name = "length_calibration"
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
        "base_model", "generation_backend", "tokenizer_source", "tokenizer_revision",
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


def _predict_resumable(adapter, engine, member, config, *, model, predict, ticket, progress):
    """Commit bounded prediction batches, including their complete diagnostics."""
    fields, rows = _csv_rows(member["questions"])
    root = os.environ.get("ZEVO_STAGE_RESUME_ROOT") or str(Path(member.get("_resume_root", member["output"])).parent / ".zevo-resume")
    identity = {"stage": "inference", "model": model, "config": config,
                "implementation": stage_resume.digest(Path(predict).read_text()),
                "benchmark": member["benchmark_id"], "row_offset": int(member.get("_row_offset", 0)),
                "questions": stage_resume.digest(rows),
                "schema": Path(member["sample_submission"]).read_text()}
    collected, diagnostics, n_unparseable = [], [], 0
    with tempfile.TemporaryDirectory(prefix="prediction-batch-") as scratch:
        batch_dir = Path(scratch)
        def generate(chunk, start):
            question_path = batch_dir / "questions.csv"
            _write_csv(question_path, fields, chunk)
            part = {**member, "questions": str(question_path),
                    "output": str(batch_dir / "predictions.csv"),
                    "diagnostics": str(batch_dir / "diagnostics.json"),
                    "_row_offset": int(member.get("_row_offset", 0)) + start}
            for key in ("output", "diagnostics"):
                Path(part[key]).unlink(missing_ok=True)
            if adapter is not None:
                result = adapter.predict_member(engine, part, config, progress=lambda step, total: None)
            else:
                manifest, summary = batch_dir / "suite.json", batch_dir / "summary.json"
                _write_json(manifest, {"members": [part]})
                summary.unlink(missing_ok=True)
                _phase(ticket, "load_model")
                with (batch_dir / "predict.log").open("w") as log:
                    proc = subprocess.Popen([sys.executable, predict, "--model", model, "--suite", str(manifest),
                                             "--summary", str(summary), "--ticket-id", ticket],
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    try:
                        for line in proc.stdout:
                            log.write(line)
                            # Batch-local progress cannot be mistaken for shard completion.
                            if not line.startswith(("__PROGRESS__:", "__ATTEMPT__:")):
                                print(line, end="", flush=True)
                        code = proc.wait()
                    finally:
                        if proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                if code:
                    with (batch_dir / "predict.log").open("rb") as log:
                        log.seek(max(0, log.seek(0, 2) - 4000))
                        detail = log.read().decode(errors="replace")
                    if code == stage_resume.YIELD_EXIT_CODE:
                        raise SystemExit(code)
                    raise RuntimeError(f"prediction batch failed ({code}): {detail}")
                result = json.loads(summary.read_text())["members"][0]
            if not _complete(part, len(chunk)):
                raise ValueError("prediction batch is incomplete; refusing to checkpoint")
            out_fields, predictions = _csv_rows(part["output"])
            records = json.loads(Path(part["diagnostics"]).read_text())["records"]
            return {"fields": out_fields, "rows": predictions, "records": records,
                    "n_unparseable": int(result.get("n_unparseable", 0))}
        for value in stage_resume.run_batches(rows, generate, root=root, ticket=ticket,
                                              identity=identity, phase_name="generating", report_progress=False):
            offset, request_offset = len(collected), len(diagnostics)
            collected.extend(value["rows"])
            diagnostics.extend({**record, "row_index": offset + int(record["row_index"]),
                                "request_index": request_offset + int(record["request_index"])}
                               for record in value["records"])
            n_unparseable += value["n_unparseable"]
            progress(len(collected), len(rows))
    _phase(ticket, "writing_artifacts")
    _atomic_csv(Path(member["output"]), value["fields"], collected)
    _atomic_json(Path(member["diagnostics"]), {"schema_version": 1, "records": diagnostics})
    return {"name": member["name"], "n_rows": len(collected),
            "n_requests": len(diagnostics), "n_unparseable": n_unparseable}


def _resident_worker(predict: str, model: str, manifest: str, summary: str,
                     ticket: str, gate: str, calibrate: bool, resident: bool = True) -> int:
    """Load once, calibrate assigned whole benchmarks, then generate shards."""
    import importlib.util
    manifest_data = json.loads(Path(manifest).read_text())
    members = manifest_data["members"]
    claims = Path(manifest_data["claims"])
    stage_resume.AllocationWindow().check(ticket, estimated_seconds=120)
    adapter, engine = None, None
    if resident:
        spec = importlib.util.spec_from_file_location("zevo_predict_adapter", predict)
        if spec is None or spec.loader is None:
            raise ValueError("predict module is unavailable")
        adapter = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = adapter
        started = time.monotonic()
        print("[startup] importing prediction adapter", flush=True)
        spec.loader.exec_module(adapter)
        print(f"[startup] adapter import took {time.monotonic() - started:.3f}s", flush=True)
        if not callable(getattr(adapter, "load_engine", None)) or not callable(getattr(adapter, "predict_member", None)):
            raise ValueError("resident predict.py requires load_engine and predict_member")
        started = time.monotonic()
        print("[startup] loading engine (tokenizer, architecture, weights, compile and warmup)", flush=True)
        engine = adapter.load_engine(model, _load_config(members[0]["config"]))
        print(f"[startup] engine ready after {time.monotonic() - started:.3f}s", flush=True)
    helper = _calibration_helper() if calibrate else None
    try:
        if calibrate:
            calibrated = set()
            for member in members:
                if not member.get("_calibration_owner") or member["benchmark_id"] in calibrated:
                    continue
                calibrated.add(member["benchmark_id"])
                questions = member["_full_questions"]
                if _load_config(member["config"]).get("measurement", {}).get("inference_config", {}).get("scoring_mode") == "option_loglikelihood":
                    continue
                # Reuse accepted evidence on a retry, without rerunning samples.
                try:
                    helper.validate(member["config"], questions)
                    continue
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                policy = helper.calibration_policy(member["config"])
                candidates = member["calibration_configs"]
                if len(candidates) > policy["max_configurations"]:
                    raise ValueError("calibration candidates exceed the declared resource budget")
                if any(helper.calibration_policy(candidate) != policy for candidate in candidates):
                    raise ValueError("calibration candidates must share one sampling/acceptance policy")
                fields, rows, _ = helper.sample_rows(questions, policy["sample_size"])
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
            # Atomic local-filesystem claims: one owner per batch. A failed worker
            # fails this allocation; durable batch receipts are replayed on retry.
            try:
                (claims / str(index)).mkdir()
            except FileExistsError:
                continue
            print("__ZEVO_PART_CLAIMED__:" + str(index), flush=True)
            def progress(step, total):
                _emit(ticket, {"step": step, "total": total,
                               "benchmark_index": index, "benchmark_total": len(members)})
            result = _predict_resumable(adapter, engine, member, _load_config(member["config"]),
                                        model=model, predict=predict, ticket=ticket, progress=progress)
            summaries.append({**result, "part_index": index})
            print("__ZEVO_PART_DONE__:" + json.dumps({"part_index": index, **result}), flush=True)
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
            if calibrate:
                helper = _calibration_helper()
                policy = helper.calibration_policy(member["config"])
                if not candidates or len(candidates) > policy["max_configurations"]:
                    raise ValueError("calibration candidates exceed the declared resource budget or are empty")
                if any(helper.calibration_policy(path) != policy for path in candidates):
                    raise ValueError("calibration candidates must share one sampling/acceptance policy")
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
    with tempfile.TemporaryDirectory(prefix="zevo-inf-", dir=os.environ.get("TMPDIR")) as scratch:
        root = Path(scratch)
        _phase(ticket, "prepare_runtime")
        # Only pre-stage a shared tokenizer when all workers agree on identity.
        first_identity = _engine_identity(_load_config(members[0]["config"]))
        shared_config = members[0]["config"] if all(
            _engine_identity(_load_config(member["config"])) == first_identity for member in members
        ) else ""
        runtime_env = gpu_runtime.prepare_environment(model=model, config_path=shared_config)
        for member in members:
            member["_identity"] = _member_identity(member, model, predict)
        parts, counts, complete = _partitions(members, root)
        already_done_rows = sum(counts[index] for index in complete)
        if not parts:
            print("[parallel] all benchmark artifacts are already complete", flush=True)
        # Each resident replica can consume any member: engine identity was
        # checked above; each task still carries its own frozen measurement YAML.
        active = [(index, parts) for index in range(workers)] if parts else []
        member_positions = {
            index: [(part_index, part) for part_index, part in enumerate(parts)
                    if part.member_index == index]
            for index in range(len(members))
        }
        claims = root / "claims"
        claims.mkdir()
        claimed_by: dict[int, int] = {}
        finished_parts: set[int] = set()
        progress: dict[int, int] = {}
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
            member = members[index]
            _atomic_json(Path(member["output"] + ".resume.json"), {
                "identity": _member_identity(member, model, predict),
                "predictions": stage_resume.digest(Path(member["output"]).read_text()),
                "diagnostics": stage_resume.digest(Path(member["diagnostics"]).read_text()),
            })
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
                    local_index in finished_parts
                    for local_index, part in positions
                ):
                    continue
                if all(
                    _complete({
                        "output": part.output,
                        "diagnostics": part.diagnostics,
                        "sample_submission": part.member["sample_submission"],
                    }, part.size)
                    for _, part in positions
                ):
                    commit_member(index)

        for index in sorted(complete):
            commit_member(index)
        _phase(ticket, "load_model")
        ready_workers: set[int] = set()
        startup_rank = 0
        gate = root / "generation-ready"
        helper = _calibration_helper() if calibrate else None
        try:
            for index, batch in active:
                worker_root = root / f"w{index}"
                env = _worker_environment(
                    runtime_env, worker_root,
                    mask[index * gpus_per_worker:(index + 1) * gpus_per_worker],
                )
                worker_members = [
                    {**part.member, "questions": part.questions,
                     "output": part.output, "diagnostics": part.diagnostics,
                     "_full_questions": part.member["questions"],
                     "_resume_root": summary, "_row_offset": part.start,
                     "_calibration_owner": index == part.member_index % workers}
                    for part in batch
                ]
                manifest = root / f"worker-{index}.json"
                worker_summary = root / f"worker-{index}-summary.json"
                _write_json(manifest, {"members": worker_members, "claims": str(claims)})
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--resident-worker", "--predict", str(Path(predict).resolve()),
                           "--model", model, "--suite", str(manifest),
                           "--summary", str(worker_summary), "--ticket-id", ticket,
                           "--generation-gate", str(gate)]
                if not resident:
                    command.append("--batch-cli")
                if calibrate:
                    command.append("--calibrate")
                proc = gpu_runtime.launch(
                    command, profile="inference", model=model, config_path=batch[0].member["config"],
                    env=env, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1,
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
                    if exit_code == stage_resume.YIELD_EXIT_CODE:
                        raise SystemExit(stage_resume.YIELD_EXIT_CODE)
                    if exit_code != 0:
                        raise RuntimeError(
                            f"inference replica {worker} exited with code {exit_code}"
                        )
                    with open(root / f"worker-{worker}-summary.json", encoding="utf-8") as handle:
                        worker_summaries = json.load(handle)["members"]
                    expected = {part for part, owner in claimed_by.items() if owner == worker}
                    reported = [int(item["part_index"]) - 1 for item in worker_summaries]
                    if len(reported) != len(set(reported)) or set(reported) != expected or not expected <= finished_parts:
                        raise ValueError(f"worker {worker} returned incomplete batch summary")
                    commit_ready_members()
                    continue
                if not generation_started:
                    milestones = (("non-default args:", "resolve_model"),
                                  ("Resolved architecture:", "initialize_engine"),
                                  ("Initializing a V1 LLM engine", "initialize_engine"),
                                  ("Loading model from scratch", "load_weights"),
                                  ("Dynamo bytecode transform", "compile_model"),
                                  ("Available KV cache memory", "warmup"),
                                  ("Graph capturing finished", "warmup"))
                    for rank, (text, phase) in enumerate(milestones, 1):
                        if text in line and rank > startup_rank:
                            startup_rank = rank
                            _phase(ticket, phase)
                if line == "__ZEVO_RESIDENT_READY__":
                    ready_workers.add(worker)
                    if len(ready_workers) == len(active):
                        if calibrate:
                            for member in members:
                                cfg = _load_config(member["config"])
                                if cfg.get("measurement", {}).get("inference_config", {}).get("scoring_mode") != "option_loglikelihood":
                                    helper.validate(member["config"], member["questions"])
                        gate.touch()
                    continue
                if line.startswith("__ZEVO_PART_CLAIMED__:"):
                    part_index = int(line.split(":", 1)[1]) - 1
                    if not 0 <= part_index < len(parts) or part_index in claimed_by:
                        raise ValueError("invalid or duplicate batch claim")
                    claimed_by[part_index] = worker
                    progress[part_index] = 0
                    continue
                if line.startswith("__ZEVO_PART_DONE__:"):
                    result = json.loads(line.split(":", 1)[1])
                    part_index = int(result["part_index"]) - 1
                    if claimed_by.get(part_index) != worker or part_index in finished_parts:
                        raise ValueError("invalid or duplicate batch completion")
                    part = parts[part_index]
                    if int(result["n_rows"]) != part.size or not _complete({
                            **part.member, "output": part.output, "diagnostics": part.diagnostics}, part.size):
                        raise ValueError("completed batch artifacts are incomplete")
                    part.unparseable = int(result.get("n_unparseable", 0))
                    finished_parts.add(part_index)
                    progress[part_index] = part.size
                    commit_ready_members()
                    continue
                if not line.startswith("__PROGRESS__:"):
                    if line.startswith("__PHASE__:"):
                        print(line, flush=True)
                    elif not line.startswith("__PHASE__:"):
                        print(f"[worker {worker}] {line}", flush=True)
                    continue
                try:
                    data = json.loads(line.split(":", 2)[2])
                    local_index = int(data["benchmark_index"]) - 1
                    part = parts[local_index]
                    if claimed_by.get(local_index) != worker:
                        raise ValueError("progress from a non-owning replica")
                    step = int(data["step"])
                    if step < 0 or step > part.size:
                        raise ValueError("worker progress exceeds its shard")
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise ValueError(f"invalid inference worker progress: {line}") from exc
                if not generation_started:
                    _phase(ticket, "generate")
                    generation_started = True
                progress[local_index] = max(
                    progress.get(local_index, 0), step,
                )
                member_index = part.member_index
                member_step = sum(
                    value for batch_index, value in progress.items()
                    if parts[batch_index].member_index == member_index
                )
                active_names = sorted({
                    parts[batch_index].member["name"]
                    for batch_index, value in progress.items()
                    if value < parts[batch_index].size
                })
                active_ids = sorted({
                    parts[batch_index].member["benchmark_id"]
                    for batch_index, value in progress.items()
                    if value < parts[batch_index].size
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
            if len(finished_parts) != len(parts) or len(committed) != len(members):
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
    parser.add_argument("--progress-scope", choices=("calibration", "measured"),
                        help="Use calibration for a standalone sample-only suite")
    parser.add_argument("--batch-cli", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resident-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--generation-gate", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.progress_scope:
        os.environ["ZEVO_INFERENCE_PROGRESS_SCOPE"] = args.progress_scope
    try:
        if args.resident_worker:
            return _resident_worker(args.predict, args.model, args.suite, args.summary,
                                    args.ticket_id, args.generation_gate, args.calibrate, not args.batch_cli)
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
