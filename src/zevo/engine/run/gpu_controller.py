"""Finite Slurm allocation controller; workload exit does not release hardware.

The backend alone publishes validated commands. Each command has an immutable
identity and its own outcome, so late events cannot complete a newer attempt.
The controller runs inside the allocation and owns the child process group.
"""
from __future__ import annotations

import base64
import json
import shlex
import math
import re
from pathlib import Path

# Kept stdlib-only: this program runs on the remote compute node, not the API.
CONTROLLER_SOURCE = r'''
import json, os, pathlib, signal, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
idle_seconds = int(sys.argv[2])
child = None
stopping = False

def stop(signum, frame):
    global stopping
    stopping = True
    if child is not None:
        try: os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError: pass

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
def forward(signum, frame):
    if child is not None:
        try: os.killpg(child.pid, signum)
        except ProcessLookupError: pass
for sig in (signal.SIGUSR1, signal.SIGUSR2):
    signal.signal(sig, forward)

def stop_slurm_steps():
    # srun workers can live outside the local process group, including on
    # other nodes. The controller is the sole consumer of this allocation.
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id: return True
    try:
        for attempt in range(6):
            listed = subprocess.run(["squeue", "--steps", "-h", "-j", job_id, "-o", "%i"],
                                    capture_output=True, text=True, timeout=10)
            if listed.returncode: return False
            steps = [line.strip() for line in listed.stdout.splitlines()
                     if line.strip().startswith(job_id + ".")
                     and line.strip().split(".", 1)[1].isdigit()]
            if not steps: return True
            for step in steps:
                subprocess.run(["scancel", "--signal=KILL", step],
                               capture_output=True, timeout=10)
            time.sleep(1)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return False

def stop_marked_workers(attempt):
    # Detached local workers can escape a process group. They still inherit
    # this command's unique marker, which is never placed on the controller.
    if not os.path.isdir("/proc"):
        return not bool(os.environ.get("SLURM_JOB_ID"))
    marker = ("ZEVO_GPU_ATTEMPT=" + attempt).encode()
    for iteration in range(20):
        found = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit() or int(entry) == os.getpid(): continue
            try:
                values = pathlib.Path("/proc", entry, "environ").read_bytes().split(b"\0")
                if marker in values: found.append(int(entry))
            except (OSError, PermissionError): pass
        if not found: return True
        for pid in found:
            try: os.kill(pid, signal.SIGKILL)
            except ProcessLookupError: pass
        time.sleep(0.1)
    return False

last_work = time.monotonic()
while not stopping:
    if (root / "release").exists(): break
    requests = sorted((root / "requests").glob("*.json"))
    pending = [p for p in requests if not (root / "outcomes" / p.name).exists()]
    if not pending:
        if idle_seconds > 0 and time.monotonic() - last_work >= idle_seconds: break
        time.sleep(1)
        continue
    request_path = pending[0]
    request = json.loads(request_path.read_text())
    output_path = root / "outcomes" / request_path.name
    log_out = request["stdout"].replace("%j", os.environ.get("SLURM_JOB_ID", "local"))
    log_err = request["stderr"].replace("%j", os.environ.get("SLURM_JOB_ID", "local"))
    rc = 125
    error = ""
    try:
        import hashlib
        script = pathlib.Path(request["script"])
        if hashlib.sha256(script.read_bytes()).hexdigest() != request["sha256"]:
            raise ValueError("workload script changed after validation")
        env = dict(os.environ, ZEVO_TICKET_ID=request["ticket_id"], ZEVO_RUN_ID=request["run_id"],
                   ZEVO_GPU_ATTEMPT=request_path.stem)
        with open(log_out, "a") as out, open(log_err, "a") as err:
            child = subprocess.Popen(["bash", str(script)], cwd=request["cwd"],
                                     env=env, stdout=out, stderr=err, start_new_session=True)
            while child.poll() is None:
                if stopping or (root / "cancel" / request_path.stem).exists():
                    try: os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try: os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                time.sleep(0.2)
            rc = child.returncode
    except Exception as exc:
        error = str(exc)
    finally:
        if child is not None:
            # Background workers must not outlive the attempt, even on success.
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait()
            child = None
    if not stop_slurm_steps() or not stop_marked_workers(request_path.stem):
        # Do not publish a reusable outcome if remote workers might survive.
        # Ending the batch job makes Slurm release the entire allocation.
        print("Zevo: worker cleanup unconfirmed; ending allocation", file=sys.stderr)
        sys.exit(125)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"exit_code": rc, "error": error}))
    os.replace(temporary, output_path)
    last_work = time.monotonic()
'''


def upload_text_command(path: str, content: str) -> str:
    """Atomically publish a small backend-owned file without shell interpolation."""
    encoded = base64.b64encode(content.encode()).decode()
    script = (
        "import base64,os,pathlib;os.umask(0o077);"
        f"p=pathlib.Path({path!r});p.parent.mkdir(parents=True,exist_ok=True);"
        "t=p.with_name(p.name+'.tmp');"
        f"t.write_bytes(base64.b64decode({encoded!r}));os.replace(t,p)"
    )
    return "python3 -c " + shlex.quote(script)


def controller_script(stage_script: str, directory: str, *, idle_seconds: int = 1800) -> str:
    """Preserve the site's validated SBATCH resource/container directives."""
    directives = [line for line in stage_script.splitlines() if line.startswith("#SBATCH")]
    encoded = base64.b64encode(CONTROLLER_SOURCE.encode()).decode()
    program = f"import base64;exec(base64.b64decode({encoded!r}))"
    return "\n".join([
        "#!/bin/bash", *directives,
        "exec python3 -c " + shlex.quote(program) + " " + shlex.quote(directory)
        + " " + str(idle_seconds), "",
    ])


def request_command(*, directory: str, request_id: str, script: str, sha256: str,
                    ticket_id: str, run_id: str, cwd: str, stdout: str, stderr: str) -> str:
    payload = dict(script=script, sha256=sha256, ticket_id=ticket_id, run_id=run_id,
                   cwd=cwd, stdout=stdout, stderr=stderr)
    root = shlex.quote(directory)
    return (
        f"mkdir -p {root}/requests {root}/outcomes {root}/cancel && "
        + upload_text_command(f"{directory}/requests/{request_id}.json", json.dumps(payload))
    )


def run_walltime_plan(plan, user_limit_hours=0):
    """Estimate plus explicit allowance, capped before rounding to Slurm minutes.

    Old artifacts have only time_limit_hours: preserve that already-budgeted
    value rather than invent an estimate or add a second safety margin.
    """
    estimate = getattr(plan, "estimated_run_hours", None)
    buffer = getattr(plan, "runtime_buffer_hours", None)
    legacy = estimate is None
    if legacy:
        estimate, buffer = float(plan.time_limit_hours), 0.0
    elif buffer is None:
        buffer = estimate * 0.25
    requested = estimate + buffer
    platform_cap = getattr(plan, "platform_max_runtime_hours", None)
    caps = [v for v in (platform_cap, user_limit_hours) if v is not None and v > 0]
    if requested <= 0 or not math.isfinite(requested):
        raise ValueError("a finite positive whole-run runtime estimate is required")
    minutes = math.ceil(requested * 60)
    if caps:
        minutes = min(minutes, math.floor(min(caps) * 60))
    if minutes < 1:
        raise ValueError("runtime cap is below Slurm's one-minute request precision")
    return dict(estimated_run_hours=estimate, runtime_buffer_hours=buffer,
                platform_max_runtime_hours=platform_cap, user_max_runtime_hours=user_limit_hours,
                requested_minutes=minutes, legacy_runtime_plan=legacy,
                estimate_evidence=getattr(plan, "runtime_estimate_evidence", ""),
                platform_limit_evidence=getattr(plan, "platform_runtime_limit_evidence", ""))


async def submit_workload(session, *, run, ticket, contract, info, heartbeat_id, sha256):
    """Reuse a healthy owned allocation or submit one finite controller job."""
    from sqlalchemy import select
    from zevo.db import InfraInstance, Run, Ticket
    from zevo.engine.run.remote_jobs import _ssh, _slurm_cli_bootstrap_command

    # Serialize publication inside one Run, including two concurrently ready stages.
    await session.execute(select(Run).where(Run.id == run.id).with_for_update())
    candidates = list((await session.execute(select(InfraInstance).where(
        InfraInstance.run_id == run.id, InfraInstance.provider == "cluster",
        InfraInstance.released_at.is_(None),
    ).order_by(InfraInstance.created_at.desc()))).scalars().all())
    active_consumers = list((await session.execute(
        select(InfraInstance).join(Ticket, Ticket.id == InfraInstance.ticket_id).where(
            InfraInstance.run_id == run.id, InfraInstance.provider == "cluster",
            Ticket.id != ticket.id,
            Ticket.status.notin_(["succeeded", "degraded", "failed", "cancelled", "skipped"]),
        )
    )).scalars().all())
    owner = None
    for row in candidates:
        meta = dict(row.meta or {})
        if not meta.get("controller_directory") or meta.get("allocation_owner_row_id"):
            continue
        if run.gpu_allocation_mode != "per_run" and row.ticket_id != ticket.id:
            continue
        checked = await _ssh(info, _slurm_cli_bootstrap_command(info.cluster.env_setup, required=("sbatch", "squeue", "sacct", "scancel")) +
                             f"squeue -h -j {shlex.quote(row.instance_id)} -o '%T'")
        if not checked.get("ok"):
            raise ValueError("cannot verify the retained allocation: " + checked.get("error", ""))
        if checked.get("stdout", "").strip() not in {"RUNNING", "PENDING", "CONFIGURING"}:
            # The provider no longer owns the allocation. It is safe to replace it.
            from datetime import datetime, timezone
            row.released_at = datetime.now(timezone.utc)
            row.release_reason = "retained allocation no longer live in Slurm"
            row.status = "released"
            continue
        if row.gpu_count < contract.num_gpus or int(meta.get("nodes", 1)) != contract.nodes:
            raise ValueError("stage does not fit the retained run allocation; resource plan must cover all stages")
        # No concurrent commands on one set of GPUs. The orchestrator can retry
        # after the active consumer completes, never overlap torchrun processes.
        if any((r.meta or {}).get("controller_directory") == meta["controller_directory"]
               for r in active_consumers):
            raise ValueError("retained allocation is busy with another stage")
        owner = row
        break
    directory = ((owner.meta or {})["controller_directory"] if owner is not None
                 else str(Path(info.cluster.workdir) / ".zevo-allocations" / heartbeat_id))
    outcome = f"{directory}/outcomes/{heartbeat_id}.json"
    publish = request_command(directory=directory, request_id=heartbeat_id,
                              script=contract.remote_script_path, sha256=sha256,
                              ticket_id=ticket.id, run_id=run.id, cwd=contract.remote_work_dir,
                              stdout=contract.stdout_path, stderr=contract.stderr_path)
    remote_script = shlex.quote(contract.remote_script_path)
    verify = (f"test -s {remote_script} && test \"$(sha256sum {remote_script} | awk '{{print $1}}')\" = "
              + shlex.quote(sha256) + " && ")
    publish = verify + publish
    walltime = dict((owner.meta or {}).get("allocation_walltime") or {}) if owner else {}
    if owner is None:
        wrapper_path = directory + "/controller.sbatch"
        body = Path(contract.script_path).read_text()
        if run.gpu_allocation_mode == "per_run":
            # The initial Data/Inference stage can need far less RAM and time
            # than Train. Allocate the validated train envelope from the start.
            plan = info.resource_plan
            def directive(name, default):
                match = re.search(r"^#SBATCH\s+--" + name + r"(?:=|\s+)(\S+)", body, re.M)
                return match.group(1) if match else default
            tasks = max(1, int(directive("ntasks-per-node", "1")))
            original_cpus = max(1, int(directive("cpus-per-task", "1")))
            def gib(value):
                match = re.fullmatch(r"([0-9.]+)([KMGT]?)", value.upper())
                if not match:
                    raise ValueError("unsupported Slurm memory unit in run allocation")
                amount, unit = match.groups()
                return float(amount) * {"K": 1 / 1024**2, "M": 1 / 1024, "": 1 / 1024, "G": 1, "T": 1024}[unit]
            requested_ram = max(gib(directive("mem", "0")),
                                gib(directive("mem-per-cpu", "0")) * tasks * original_cpus)
            memory_gib = max(plan.min_ram_gb, math.ceil(requested_ram))
            cpus_per_task = max(original_cpus, math.ceil(plan.min_cpus / tasks))
            body = re.sub(r"^#SBATCH\s+(?:--(?:mem|mem-per-cpu|cpus-per-task|time)(?:=|\s+)|-[ct]\s*).*\n?", "", body, flags=re.M)
            walltime = run_walltime_plan(plan, run.max_runtime_hours)
            envelope = (f"#SBATCH --mem={memory_gib}G\n"
                        f"#SBATCH --cpus-per-task={cpus_per_task}\n"
                        f"#SBATCH --time={walltime['requested_minutes']}\n")
            body = "#!/bin/bash\n" + envelope + body
        wrapper = controller_script(body, directory,
                                    idle_seconds=0 if run.gpu_allocation_mode == "per_run" else 1800)
        Path(contract.script_path).with_name("allocation-controller.sbatch").write_text(wrapper)
        command = publish + " && " + upload_text_command(wrapper_path, wrapper)
        command += " && " + _slurm_cli_bootstrap_command(info.cluster.env_setup, required=("sbatch", "squeue", "sacct", "scancel")) + "sbatch --parsable -- " + shlex.quote(wrapper_path)
    else:
        command = publish + " && printf '%s\\n' " + shlex.quote(owner.instance_id)
    result = await _ssh(info, command, timeout_seconds=30)
    if not result.get("ok"):
        raise ValueError("controller submission failed: " + str(result.get("error") or result.get("stdout")))
    job_id = result.get("stdout", "").strip().splitlines()[-1].split(";", 1)[0]
    if not job_id.isdecimal():
        raise ValueError(f"controller returned invalid job id: {job_id!r}")
    return job_id, {
        "allocation_walltime": walltime,
        "controller_directory": directory,
        "controller_outcome_path": outcome,
        "controller_request_id": heartbeat_id,
        "allocation_owner_row_id": owner.id if owner is not None else "",
        "allocation_owner_ticket_id": owner.ticket_id if owner is not None else ticket.id,
        "gpu_allocation_mode": run.gpu_allocation_mode or "per_stage",
    }
