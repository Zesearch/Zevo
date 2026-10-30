"""Verified walltime handoff for an unchanged Train ticket and optimizer plan."""
from __future__ import annotations

import base64
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import shlex

TERMINAL = {"COMPLETED", "TIMEOUT", "PREEMPTED", "NODE_FAIL", "FAILED", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def authorized(ticket, row):
    if row is None or getattr(ticket, "agent_id", "") != "train":
        return False
    proof = (row.meta or {}).get("training_continuation") or {}
    return bool(proof.get("source_resource_id") == row.id
                and proof.get("ticket_id") == ticket.id
                and proof.get("verified") is True
                and str((row.meta or {}).get("scheduler_state") or "").split("+")[0].split(" ")[0] in TERMINAL)


def verify_files(root, request, ticket_id, executed_config, world_size):
    """Read metadata and inventory only; the next worker must verify real restore.

    Supports standard full-state Transformers/TRL checkpoints, including sharded
    weights. Unsupported optimizer layouts fail closed, never use weights alone.
    """
    import hashlib
    import json
    from pathlib import Path

    root = Path(root).resolve()
    def owned(path):
        value = Path(path).resolve()
        if not value.is_relative_to(root):
            raise ValueError("continuation artifact is outside this Train ticket")
        return value
    checkpoint = owned(request["checkpoint_path"])
    plan_path = owned(request["plan_path"])
    config = owned(executed_config)
    if hashlib.sha256(config.read_bytes()).hexdigest() != request["config_sha256"]:
        raise ValueError("executed configuration differs from continuation")
    plan = json.loads(plan_path.read_text())
    if (plan.get("ticket_id") != ticket_id
        or plan.get("config_sha256") != request["config_sha256"]
        or plan.get("planned_steps") != request["planned_steps"]):
        raise ValueError("continuation differs from frozen training plan")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if (state.get("global_step") != request["completed_steps"]
        or state.get("max_steps") != request["planned_steps"]
        or not 0 < request["completed_steps"] < request["planned_steps"]):
        raise ValueError("trainer state does not match continuation progress")
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "training_args.bin"]
    required += (["rng_state.pth"] if world_size == 1 else
                 [f"rng_state_{rank}.pth" for rank in range(world_size)])
    weights = next((name for name in ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors", "adapter_model.bin")
                    if (checkpoint / name).is_file()), None)
    if weights:
        required.append(weights)
    else:
        index = next((checkpoint / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
                      if (checkpoint / name).is_file()), None)
        if index is None:
            raise ValueError("continuation has no model weights")
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        if not shards:
            raise ValueError("empty checkpoint weight index")
        required += [index.name, *shards]
    for name in required:
        path = owned(checkpoint / name)
        if not path.is_relative_to(checkpoint) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or invalid full-state checkpoint file: {name}")
    return {**request, "checkpoint_path": str(checkpoint), "plan_path": str(plan_path),
            "ticket_id": ticket_id, "verified": True,
            "state_sha256": hashlib.sha256((checkpoint / "trainer_state.json").read_bytes()).hexdigest()}


async def verify_request(session, ticket, inp, output, row):
    request = output.continuation.model_dump()
    contract = inp.slurm_job
    meta = row.meta or {}
    state = str(meta.get("scheduler_state") or "").split("+")[0].split(" ")[0]
    if (contract.phase != "collect" or contract.bookkeeping_row_id != row.id
        or contract.job_id != row.instance_id or state not in TERMINAL):
        raise ValueError("continuation requires the exact terminal training workload")
    digest = hashlib.sha256(Path(output.train_config_path).read_bytes()).hexdigest()
    if digest != request["config_sha256"] or meta.get("training_config_sha256", digest) != digest:
        raise ValueError("continuation changed the executed training configuration")
    from zevo.contracts.configuration import load_train_config
    config = load_train_config(output.train_config_path)
    from zevo.engine.run.remote_jobs import _device_info_for_ticket, _ssh
    info = await _device_info_for_ticket(session, ticket)
    if info is None:
        raise ValueError("continuation has no verified device route")
    workdir = PurePosixPath(meta["remote_workdir"])
    if workdir.parent.name != ticket.id:
        raise ValueError("workload directory is not owned by this Train ticket")
    arguments = dict(root=str(workdir.parent), request=request, ticket_id=ticket.id,
                     executed_config=str(workdir / "train_config.yaml"), world_size=config.training.world_size)
    program = inspect.getsource(verify_files) + "\nimport json\nprint(json.dumps(verify_files(**" + repr(arguments) + ")))"
    code = "import base64;exec(base64.b64decode(" + repr(base64.b64encode(program.encode()).decode()) + "))"
    result = await _ssh(info, "python3 -c " + shlex.quote(code), timeout_seconds=60)
    if not result.get("ok"):
        raise ValueError("cannot verify full-state continuation: " + str(result.get("error", ""))[:1000])
    try:
        proof = json.loads(result.get("stdout", "").strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError("continuation verifier returned no evidence") from exc
    prior = meta.get("resumed_from") or {}
    if prior and (prior["config_sha256"] != digest or proof["completed_steps"] <= prior["completed_steps"]):
        raise ValueError("continuation made no progress or changed its plan")
    return {**proof, "source_resource_id": row.id, "source_job_id": row.instance_id}


async def queue_continuation(session, ticket, heartbeat, row, proof):
    """Commit the checkpoint binding and next activation together, without repair."""
    from datetime import datetime, timezone
    from zevo.engine.run.wakeup import queue_wakeup
    if proof["source_resource_id"] != row.id or proof["ticket_id"] != ticket.id:
        raise ValueError("continuation belongs to another workload")
    row.meta = {**dict(row.meta or {}), "training_continuation": proof}
    ticket.status = "queued"
    ticket.repair_route = ""
    heartbeat.activation_phase = "continue"
    heartbeat.finished_at = datetime.now(timezone.utc)
    await queue_wakeup(session, agent_id="train", ticket_id=ticket.id,
        source="training_continuation",
        reason=f"Resume the unchanged training plan at step {proof['completed_steps']}/{proof['planned_steps']}; preserve configuration and full trainer state.",
        payload={"continuation_resource_id": row.id})
