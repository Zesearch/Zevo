"""System-owned sequential training/Validation controller; no Test inputs.

The Train agent supplies argv commands, an initial boundary, and a full-plan
Trainer adapter. Each training subprocess must exit after saving resumable
state; only then does inference start, so training VRAM is released.
Subsequent boundaries and early stopping are decided by the Orchestrator.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

# The deployed module and repository source expose the identical launch API.
import importlib.util
_runtime_path = Path(__file__).with_name("zevo_gpu_runtime.py")
if not _runtime_path.is_file():
    _runtime_path = Path(__file__).with_name("gpu_runtime.py")
_runtime_spec = importlib.util.spec_from_file_location("zevo_gpu_runtime", _runtime_path)
gpu_runtime = importlib.util.module_from_spec(_runtime_spec)
_runtime_spec.loader.exec_module(gpu_runtime)



def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def verify_resume_checkpoint(path, *, step, planned_steps, world_size=1):
    root = Path(path)
    state = json.loads((root / "trainer_state.json").read_text())
    if state.get("global_step") != step or state.get("max_steps") != planned_steps:
        raise ValueError("segment changed the full training plan or saved the wrong step")
    # Recognize the checkpoint layouts saved/resumed by Transformers Trainer:
    # ordinary/DDP, DeepSpeed ZeRO and FSDP sharded state.
    deepspeed = bool(list(root.glob("global_step*/*optim_states.pt")))
    fsdp = bool(list(root.glob("optimizer*/**/*.distcp")))
    if deepspeed:
        if not (root / "latest").is_file() or not list(root.glob("global_step*/*model_states.pt")):
            raise ValueError("DeepSpeed checkpoint lacks its model/scheduler state or latest tag")
    else:
        if not fsdp and not (root / "optimizer.pt").is_file():
            raise ValueError("resumable checkpoint is missing optimizer.pt")
        if not (root / "scheduler.pt").is_file():
            raise ValueError("resumable checkpoint is missing scheduler.pt")
    rng_files = list(root.glob("rng_state*.pth"))
    if len(rng_files) < world_size:
        raise ValueError("resumable checkpoint is missing per-rank RNG state")
    sharded_weights = bool(list(root.glob("pytorch_model*/**/*.distcp")))
    if (not deepspeed and not sharded_weights
            and not list(root.glob("*.safetensors")) and not list(root.glob("pytorch_model*.bin"))):
        raise ValueError("resumable checkpoint is missing model weights")
    return state


def validate_measurement(measurement, contract, contract_sha256, step):
    if (measurement.get("split") != "validation"
            or measurement.get("contract_sha256") != contract_sha256
            or measurement.get("step") != step):
        raise ValueError("measurement must use this step's frozen Validation contract")
    metrics = measurement.get("metrics", {})
    if set(metrics) != {member["name"] for member in contract["members"]}:
        raise ValueError("measurement must include every frozen Validation member")
    values = []
    for member in contract["members"]:
        value = metrics[member["name"]]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Validation metrics must be finite numbers")
        values.append(value)
    # Matches the platform's suite_average over frozen members.
    return sum(values) / len(values)



def validate_decision(decision, request):
    if decision.get("request_id") != request["request_id"]:
        raise ValueError("decision belongs to another checkpoint request")
    if not str(decision.get("reason", "")).strip():
        raise ValueError("an adaptive decision requires a reason")
    action = decision.get("action")
    if action == "continue":
        step = decision.get("next_step")
        if type(step) is not int or not request["step"] < step <= request["planned_steps"]:
            raise ValueError("next_step must advance within the original plan")
        if len(request["measurements"]) >= 100 and step != request["planned_steps"]:
            raise ValueError("evaluation overhead limit reached; continue to the final step or stop")
    elif action == "stop":
        # One flat observation is not sufficient evidence for early stopping.
        if len(request["measurements"]) < 3:
            raise ValueError("early stopping requires at least three successful Validation checks")
    else:
        raise ValueError("action must be continue or stop")
    return decision


def emit_validation_report(ticket_id, ledger):
    marker = {"phase": "adaptive_validation_report", "step": ledger["measurements"][-1]["step"],
              "total": ledger["planned_steps"], "t": time.time(), "adaptive_report": ledger}
    print(f"__PROGRESS__:{ticket_id}:" + json.dumps(marker), flush=True)


def run_adaptive_training(*, ticket_id, contract_path, contract_sha256,
                          planned_steps, first_step, train_command,
                          work_dir, world_size=1):
    """Run full-state segments and ask the live agent after Validation checks.

    Train runs one full-state segment then exits this controller with status 0.
    The backend owns separate Inference and Evaluation tickets and queues a
    new Train activation only after the Orchestrator decision (or final score).
    The caller publishes report["selected_checkpoint"] only when this returns.
    """
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be positive")
    if type(planned_steps) is not int or not 0 < first_step <= planned_steps:
        raise ValueError("invalid initial evaluation boundary")
    raw = Path(contract_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != contract_sha256:
        raise ValueError("Validation contract hash mismatch")
    contract = json.loads(raw)
    if contract.get("split") != "validation" or not contract.get("members"):
        raise ValueError("only a nonempty Validation contract is supported")
    import math
    fraction = contract.get("first_validation_fraction")
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not 0 < fraction < 1 or type(first_step) is not int
            or first_step != max(1, math.ceil(planned_steps * fraction))):
        raise ValueError("first_step must match the Orchestrator's frozen first_validation_fraction")
    # Verify staged input bytes. Paths are relative so the entire frozen bundle
    # can be copied to a GPU host without rewriting the contract.
    for member in contract["members"]:
        for artifact in member["files"].values():
            file = Path(contract_path).parent / artifact["path"]
            if hashlib.sha256(file.read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError("frozen Validation artifact changed")
    for artifact in contract.get("helpers", {}).values():
        file = Path(contract_path).parent / artifact["path"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("frozen Validation helper changed")
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "adaptive-training.json"
    identity = {"ticket_id": ticket_id, "contract_sha256": contract_sha256,
                "planned_steps": planned_steps, "world_size": world_size}
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if any(ledger.get(k) != v for k, v in identity.items()):
            raise ValueError("existing adaptive ledger belongs to a different plan")
        if ledger.get("complete"):
            emit_validation_report(ticket_id, ledger)
            return ledger
    else:
        ledger = {**identity, "direction": contract["direction"], "measurements": [], "decisions": [],
                  "resume_checkpoint": "", "next_step": first_step}
        atomic_json(ledger_path, ledger)
    while True:
        step = ledger["next_step"]
        checkpoint = root / f"checkpoint-{step}"
        env = {**os.environ,
               "PYTHONPATH": str(Path(contract_path).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               "ZEVO_ADAPTIVE_TARGET_STEP": str(step),
               "ZEVO_ADAPTIVE_PLANNED_STEPS": str(planned_steps),
               "ZEVO_ADAPTIVE_RESUME_CHECKPOINT": ledger["resume_checkpoint"],
               "ZEVO_ADAPTIVE_CHECKPOINT": str(checkpoint),
               "ZEVO_ADAPTIVE_VALIDATION_CONTRACT": str(Path(contract_path).resolve())}
        # Reuse a completed segment after a controller/connection restart.
        ready = checkpoint / ".zevo-resumable.json"
        if not ready.exists():
            gpu_runtime.run(train_command, profile="train", model=ledger["resume_checkpoint"], env=env)
            verify_resume_checkpoint(checkpoint, step=step, planned_steps=planned_steps, world_size=world_size)
            sizes = {str(p.relative_to(checkpoint)): p.stat().st_size
                     for p in checkpoint.rglob("*") if p.is_file()}
            if not sizes or any(size <= 0 for size in sizes.values()):
                raise ValueError("checkpoint save contains empty files")
            atomic_json(ready, {"ticket_id": ticket_id, "step": step, "planned_steps": planned_steps, "file_sizes": sizes})
        manifest = json.loads(ready.read_text())
        if manifest.get("step") != step or manifest.get("planned_steps") != planned_steps:
            raise ValueError("checkpoint commit belongs to another training segment")
        for name, size in manifest["file_sizes"].items():
            file = checkpoint / name
            if not file.is_file() or file.stat().st_size != size:
                raise ValueError("resumable checkpoint is missing or changed")
        verify_resume_checkpoint(checkpoint, step=step, planned_steps=planned_steps, world_size=world_size)
        if not ledger["measurements"] or ledger["measurements"][-1]["step"] != step:
            metrics_path = root / f"validation-{step}.json"
            pending = ledger.get("pending_checkpoint")
            if not pending or pending["step"] != step:
                request_id = uuid.uuid4().hex
                pending = {**identity, "request_id": request_id, "step": step,
                           "checkpoint": str(checkpoint), "metrics_path": str(metrics_path),
                           "decision_path": str(root / f"decision-{request_id}.json"),
                           "direction": contract["direction"], "measurements": list(ledger["measurements"]),
                           "decisions": list(ledger["decisions"])}
                ledger["pending_checkpoint"] = pending
                metrics_path.unlink(missing_ok=True)
                atomic_json(ledger_path, ledger)
            if not metrics_path.exists():
                marker = {"phase": "adaptive_checkpoint", "step": step, "total": planned_steps,
                          "t": time.time(), "adaptive_checkpoint_request": pending}
                print(f"__PROGRESS__:{ticket_id}:" + json.dumps(marker), flush=True)
                # This is a successful segment handoff, not final Train completion.
                # Exit releases the allocation controller's workload slot.
                raise SystemExit(0)
            measurement = json.loads(metrics_path.read_text())
            if measurement.get("request_id") != pending["request_id"]:
                raise ValueError("evaluation response belongs to another checkpoint")
            score = validate_measurement(measurement, contract, contract_sha256, step)
            ledger["measurements"].append({
                "step": step, "score": score, "metrics": measurement["metrics"],
                "checkpoint": str(checkpoint),
            })
            ledger["resume_checkpoint"] = str(checkpoint)
            atomic_json(ledger_path, ledger)
        direction = contract["direction"]
        best = (max if direction == "max" else min)(
            ledger["measurements"], key=lambda row: row["score"])
        # Bound full-state storage to the selected best and the latest resume
        # point. Only our committed, older segment directories are eligible.
        import shutil
        keep = {Path(best["checkpoint"]), checkpoint}
        for row in ledger["measurements"]:
            old = Path(row["checkpoint"])
            marker = old / ".zevo-resumable.json"
            if old in keep or old.parent != root or old.is_symlink() or not marker.is_file():
                continue
            if old.name != f"checkpoint-{row['step']}":
                continue
            if json.loads(marker.read_text()).get("ticket_id") == ticket_id:
                shutil.rmtree(old)
        if step >= planned_steps:
            ledger.update(complete=True, reason="plan_completed",
                          selected_checkpoint=best["checkpoint"], selected_step=best["step"],
                          completed_steps=step)
            atomic_json(ledger_path, ledger)
            emit_validation_report(ticket_id, ledger)
            return ledger
        pending = {**ledger["pending_checkpoint"], "measurements": ledger["measurements"]}
        ledger["pending"] = pending
        atomic_json(ledger_path, ledger)
        if not Path(pending["decision_path"]).exists():
            # Defensive recovery: never train further or retain a GPU process
            # if a response was delivered but the decision is still pending.
            print(f"__PROGRESS__:{ticket_id}:" + json.dumps({
                "phase": "adaptive_checkpoint", "step": step, "total": planned_steps,
                "t": time.time(), "adaptive_checkpoint_request": ledger["pending_checkpoint"]}), flush=True)
            raise SystemExit(0)
        decision = validate_decision(json.loads(Path(pending["decision_path"]).read_text()), pending)
        ledger["decisions"].append(decision)
        if decision["action"] == "stop":
            ledger.update(complete=True, reason="adaptive_early_stopping",
                          selected_checkpoint=best["checkpoint"], selected_step=best["step"],
                          completed_steps=step)
            atomic_json(ledger_path, ledger)
            emit_validation_report(ticket_id, ledger)
            return ledger
        ledger["next_step"] = decision["next_step"]
        ledger.pop("pending", None)
        ledger.pop("pending_checkpoint", None)
        atomic_json(ledger_path, ledger)
        emit_validation_report(ticket_id, ledger)


def publish_adaptive_checkpoint(report_path, *, final_dir, plan_path, consolidated_dir=None):
    """Publish weights from the best measured step, retaining resumable originals."""
    import shutil
    from zevo_train_checkpoint import commit_training_checkpoint
    report = json.loads(Path(report_path).read_text())
    if not report.get("complete"):
        raise ValueError("adaptive training has not completed")
    source = Path(report["selected_checkpoint"])
    if consolidated_dir is not None:
        source = Path(consolidated_dir)
        provenance = json.loads((source / "adaptive-source.json").read_text())
        if provenance != {"checkpoint": report["selected_checkpoint"], "step": report["selected_step"]}:
            raise ValueError("consolidated model is not from the selected Validation checkpoint")
    final = Path(final_dir)
    staging = final.with_name(final.name + ".staging")
    if staging.exists():
        raise ValueError("publication staging already exists; verify before repairing")
    staging.mkdir(parents=True)
    # Full resumable state stays in checkpoint-N; the published model is weights only.
    for file in source.iterdir():
        if file.is_file() and (file.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja", ".tiktoken", ".vocab", ".bpe"}
                               or file.name.startswith("pytorch_model")):
            if file.name not in {"trainer_state.json", "adaptive-source.json", ".zevo-resumable.json"}:
                shutil.copy2(file, staging / file.name)
    return commit_training_checkpoint(
        staging, final, plan_path=plan_path,
        trainer_state_path=str(Path(report["resume_checkpoint"]) / "trainer_state.json"),
        adaptive_report_path=report_path,
    )
