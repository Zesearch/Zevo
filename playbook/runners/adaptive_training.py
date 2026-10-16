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
import subprocess
import time
import uuid


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



def score_validation_predictions(contract_path, *, predictions_dir, step, contract_sha256):
    """Run frozen scorers ourselves; the generated child supplies predictions only."""
    import runpy
    import sys
    base = Path(contract_path).parent
    contract = json.loads(Path(contract_path).read_text())
    metric_helper = contract["helpers"]["metrics"]
    scorer_path = base / metric_helper["path"]
    if hashlib.sha256(scorer_path.read_bytes()).hexdigest() != metric_helper["sha256"]:
        raise ValueError("built-in scorer changed")
    scorers = runpy.run_path(str(scorer_path))
    results = {}
    for index, member in enumerate(contract["members"]):
        files = member["files"]
        predictions = Path(predictions_dir) / f"{index:03d}" / "predictions.csv"
        labels = base / files["labels"]["path"]
        sample = base / files["sample_submission"]["path"]
        expected, _ = scorers["_load_table"](sample)
        actual, rows = scorers["_load_table"](predictions)
        gold_columns, gold = scorers["_load_table"](labels)
        if actual != expected or len(rows) != len(gold) or not rows:
            raise ValueError("predictions violate the frozen columns or row count")
        identity = [key for key in expected if key in gold_columns and key not in member["answer_fields"]]
        for prediction, answer in zip(rows, gold):
            if any(str(prediction.get(key, "")) != str(answer.get(key) if answer.get(key) is not None else "") for key in identity):
                raise ValueError("predictions violate the frozen row identity/order")
        output = predictions.with_name("metrics.json")
        output.unlink(missing_ok=True)
        script = files.get("evaluation_script")
        if script:
            subprocess.run([sys.executable, str(base / script["path"]),
                            str(predictions), str(labels), str(output)], check=True)
            metrics = json.loads(output.read_text())
        else:
            config = member.get("evaluation_config", {})
            prediction_columns = [key for key in expected if key not in identity]
            answer_fields = member["answer_fields"]
            prediction_column = config.get("prediction_column") or (prediction_columns[0] if len(prediction_columns) == 1 else "")
            answer_column = config.get("answer_column") or (answer_fields[0] if len(answer_fields) == 1 else "")
            if prediction_column not in prediction_columns or answer_column not in answer_fields:
                raise ValueError("ambiguous or invalid built-in evaluation columns")
            metrics = scorers["score_default"](
                predictions, labels, metric=member["metric"],
                prediction_col_hint=prediction_column, gold_col_hint=answer_column,
                strict=config.get("strict", False), f1_average=config.get("f1_average", "micro"))
            if metrics.get("status") != "succeeded":
                raise ValueError("built-in Validation scoring failed")
            atomic_json(output, metrics)
        score = metrics.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("frozen scorer returned no finite score")
        results[member["name"]] = score
    return {"split": "validation", "step": step, "contract_sha256": contract_sha256, "metrics": results}


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
                          evaluate_command, work_dir, decision_timeout=1800, world_size=1):
    """Run full-state segments and ask the live agent after Validation checks.

    Both commands are argv lists (no shell). Train reads ZEVO_* variables and
    writes checkpoint-<step> with ordinary Trainer state; evaluation writes
    member predictions under ZEVO_ADAPTIVE_PREDICTIONS_DIR. The caller publishes report["selected_checkpoint"]
    with the normal checkpoint transaction helper after this returns.
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
            subprocess.run(train_command, env=env, check=True)
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
            env["ZEVO_ADAPTIVE_PREDICTIONS_DIR"] = str(root / f"validation-{step}")
            Path(env["ZEVO_ADAPTIVE_PREDICTIONS_DIR"]).mkdir(exist_ok=True)
            for index in range(len(contract["members"])):
                (Path(env["ZEVO_ADAPTIVE_PREDICTIONS_DIR"]) / f"{index:03d}" / "predictions.csv").unlink(missing_ok=True)
            env["ZEVO_ADAPTIVE_METRICS_PATH"] = str(metrics_path)
            # Errors propagate without recording a negative observation. The
            # saved segment is reused on retry, never trained a second time.
            for member in contract["members"]:
                for artifact in member["files"].values():
                    file = Path(contract_path).parent / artifact["path"]
                    if hashlib.sha256(file.read_bytes()).hexdigest() != artifact["sha256"]:
                        raise ValueError("frozen Validation artifact changed")
            subprocess.run(evaluate_command, env=env, check=True)
            measurement = score_validation_predictions(
                contract_path, predictions_dir=env["ZEVO_ADAPTIVE_PREDICTIONS_DIR"],
                step=step, contract_sha256=contract_sha256)
            atomic_json(metrics_path, measurement)
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
        pending = ledger.get("pending")
        if not pending or pending["step"] != step:
            request_id = uuid.uuid4().hex
            pending = {**identity, "request_id": request_id, "step": step,
                       "decision_path": str(root / f"decision-{request_id}.json"),
                       "measurements": ledger["measurements"], "direction": direction}
            ledger["pending"] = pending
            atomic_json(ledger_path, ledger)
        deadline = time.monotonic() + decision_timeout
        while not Path(pending["decision_path"]).exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("agent decision timed out; full resumable state is retained")
            marker = {"phase": "adaptive_validation", "step": step, "total": planned_steps,
                      "t": time.time(), "adaptive_request": pending}
            print(f"__PROGRESS__:{ticket_id}:" + json.dumps(marker), flush=True)
            time.sleep(10)
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
