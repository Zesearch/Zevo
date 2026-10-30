"""Finite Slurm allocation controller; supports retained or single-workload allocations.

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
import json, os, pathlib, shlex, signal, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
idle_seconds = int(sys.argv[2])
single_workload = len(sys.argv) > 3 and sys.argv[3] == "1"
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

def cleanup_error(message):
    print("Zevo: cleanup: " + message, file=sys.stderr, flush=True)

def workload_command(script):
    # SBATCH directives are comments to bash. Apply this workload's container
    # options to a step, never to the retained controller's batch environment.
    options = []
    for line in script.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            break
        if stripped.startswith("#SBATCH"):
            args = shlex.split(stripped[len("#SBATCH"):], comments=True)
            if args and args[0].startswith("--container"):
                options.extend(args)
    if not options:
        return ["bash", str(script)]
    # One launcher, as in the non-container path. Workloads may themselves use
    # srun/torchrun; overlap prevents the outer container step reserving CPUs
    # from those inner steps. Only one workload is dispatched at a time.
    return ["srun", "--overlap", "--nodes=1", "--ntasks=1",
            "--kill-on-bad-exit=1", *options, "bash", str(script)]

def stop_slurm_steps():
    # srun workers can live outside the local process group, including on
    # other nodes. The controller is the sole consumer of this allocation.
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id: return True
    try:
        for attempt in range(6):
            listed = subprocess.run(["squeue", "--steps", "-h", "-j", job_id, "-o", "%i"],
                                    capture_output=True, text=True, timeout=10)
            if listed.returncode:
                cleanup_error("squeue failed (exit %s): %s" % (listed.returncode, listed.stderr.strip()))
                return False
            steps = [line.strip() for line in listed.stdout.splitlines()
                     if line.strip().startswith(job_id + ".")
                     and line.strip().split(".", 1)[1].isdigit()]
            if not steps: return True
            for step in steps:
                cancelled = subprocess.run(["scancel", "--signal=KILL", step],
                                           capture_output=True, text=True, timeout=10)
                if cancelled.returncode:
                    cleanup_error("scancel %s failed (exit %s): %s" %
                                  (step, cancelled.returncode, cancelled.stderr.strip()))
            time.sleep(1)
    except (OSError, subprocess.TimeoutExpired) as exc:
        cleanup_error("Slurm step cleanup unavailable: %s: %s" % (type(exc).__name__, exc))
        return False
    cleanup_error("Slurm steps still present after cleanup: %s" % steps)
    return False

def stop_marked_workers(attempt):
    # Detached local workers can escape a process group. They still inherit
    # this command's unique marker, which is never placed on the controller.
    if not os.path.isdir("/proc"):
        cleanup_error("/proc unavailable for worker verification")
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
    cleanup_error("marked worker PIDs still present after SIGKILL: %s" % found)
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
        work = pathlib.Path(request["cwd"])
        ticket_root = work.parent if work.name.startswith("attempt-") and work.parent.name == request["ticket_id"] else work
        env["ZEVO_STAGE_RESUME_ROOT"] = str(ticket_root / ".zevo-resume")
        with open(log_out, "a") as out, open(log_err, "a") as err:
            print("__ATTEMPT__:%s:%s@%s" % (request["ticket_id"], request_path.stem, time.time()), file=out, flush=True)
            print("__PHASE__:%s:starting_workload@%s" % (request["ticket_id"], time.time()), file=out, flush=True)
            child = subprocess.Popen(workload_command(script), cwd=request["cwd"],
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
        diagnostic = root / ("cleanup-failure-" + request_path.name)
        diagnostic.write_text(json.dumps({"workload_exit_code": rc, "workload_error": error,
                                          "cleanup_confirmed": False}))
        print("Zevo: worker cleanup unconfirmed; workload exit=%s; ending allocation" % rc,
              file=sys.stderr, flush=True)
        sys.exit(125)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"exit_code": rc, "error": error}))
    os.replace(temporary, output_path)
    if single_workload or rc == 75:
        sys.exit(rc if 0 <= rc <= 255 else 1)
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


async def read_controller_headers(info, directory):
    """Read complete allocation directives, never a truncated script prefix."""
    from zevo.engine.run.remote_jobs import _ssh
    response = await _ssh(info, "sed -n '/^#SBATCH/p' " + shlex.quote(directory + "/controller.sbatch"),
                          timeout_seconds=30, stdout_limit=65536)
    if not response.get("ok") or response.get("stdout_truncated"):
        return None
    return response.get("stdout", "")


def controller_script(stage_script: str, directory: str, *, idle_seconds: int = 1800, single_workload: bool = False, env_setup: str = "") -> str:
    """Run the controller on the host; container directives belong to each step."""
    from zevo.engine.run.remote_jobs import _slurm_cli_bootstrap_command

    directives = [line for line in stage_script.splitlines() if line.startswith("#SBATCH")
                  and not re.match(r"#SBATCH\s+--container", line)]
    containerized = bool(re.search(r"^#SBATCH\s+--container", stage_script, re.M))
    # Container-specific setup can reference paths that are unavailable on the
    # host. The original workload script still executes that setup in its step.
    controller_setup = "" if containerized else env_setup
    bootstrap = _slurm_cli_bootstrap_command()
    encoded = base64.b64encode(CONTROLLER_SOURCE.encode()).decode()
    program = f"import base64;exec(base64.b64decode({encoded!r}))"
    return "\n".join([
        "#!/bin/bash", *directives, "set -e", controller_setup,
        "command -v python3 >/dev/null || { echo 'allocation controller requires python3 after env_setup' >&2; exit 127; }",
        bootstrap,
        "command -v srun >/dev/null || { echo 'allocation controller requires host srun' >&2; exit 127; }",
        "exec python3 -c " + shlex.quote(program) + " " + shlex.quote(directory)
        + " " + str(idle_seconds) + " " + str(int(single_workload)), "",
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


_TERMINAL_ALLOCATION_STATES = frozenset({
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
})


async def _retained_allocation_state(info, job_id: str) -> str:
    """Require positive scheduler evidence before replacing an allocation.

    Slurm removes finished jobs from slurmctld after MinJobAge. An exact
    squeue lookup then fails rather than returning an empty queue; accounting
    still records the terminal allocation. Transport failures are not absence.
    """
    from zevo.engine.run.remote_jobs import (
        _ssh, _slurm_cli_bootstrap_command, _slurm_login_command,
    )

    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("cannot verify the retained allocation: invalid stored job ID")
    bootstrap = _slurm_cli_bootstrap_command(
        info.cluster.env_setup, required=("squeue", "sacct"),
    )
    queued = await _ssh(info, _slurm_login_command(
        bootstrap + f"squeue -h -j {shlex.quote(job_id)} -o '%T'"))
    if not queued.get("ok"):
        # Match the Slurm diagnostic, not a generic SSH/permission error.
        missing = any(line.strip() == "slurm_load_jobs error: Invalid job id specified"
                      for line in str(queued.get("error", "")).splitlines())
        if not missing:
            raise ValueError("cannot verify the retained allocation: " + str(queued.get("error", "")))
    elif queued.get("stdout", "").strip():
        return queued["stdout"].strip().upper()

    accounting = await _ssh(info, _slurm_login_command(
        bootstrap + f"sacct -n -X -j {shlex.quote(job_id)} --format=JobIDRaw,State%40 -P"))
    if not accounting.get("ok"):
        raise ValueError("cannot verify the retained allocation: accounting lookup failed: "
                         + str(accounting.get("error", "")))
    states = set()
    for line in accounting.get("stdout", "").splitlines():
        columns = line.strip().split("|")
        if len(columns) >= 2 and columns[0].strip() == job_id and columns[1].strip():
            # CANCELLED may include 'by UID'. Ignore step rows and other jobs.
            states.add(columns[1].strip().split()[0].upper())
    if len(states) != 1:
        raise ValueError("cannot verify the retained allocation: accounting has no unambiguous job state")
    return states.pop()


def _checkpoint_allocation_handoff(run, ticket, consumer, train):
    """Only the verified checkpoint's own Inference may borrow a yielded Train GPU."""
    meta = consumer.meta or {}
    key = meta.get("checkpoint_yield")
    request = (ticket.payload or {}).get("mid_training_checkpoint") or {}
    stage = (run.lifecycle or {}).get("adaptive_checkpoints", {}).get(key, {})
    return bool(
        key and ticket.agent_id == "inference"
        and train is not None and train.agent_id == "train"
        and train.status == "waiting_external"
        and train.run_id == run.id and consumer.run_id == run.id
        and request.get("request_id") == key
        and request.get("ticket_id") == train.id == consumer.ticket_id
        and stage.get("request") == request
        and stage.get("job_row_id") == consumer.id
        and stage.get("inference_ticket_id") == ticket.id
        and not stage.get("resume_queued") and not meta.get("checkpoint_resume")
        and meta.get("scheduler_state") == "COMPLETED"
    )


async def submit_workload(session, *, run, ticket, contract, info, heartbeat_id, sha256, publication=None, continuation=None):
    """Reuse a healthy owned allocation or submit one finite controller job."""
    from sqlalchemy import select
    from zevo.db import InfraInstance, Run, Ticket
    from zevo.engine.run.remote_jobs import _ssh, _slurm_cli_bootstrap_command, _slurm_login_command

    # Serialize publication inside one Run, including two concurrently ready stages.
    await session.execute(select(Run).where(Run.id == run.id).with_for_update())
    candidates = list((await session.execute(select(InfraInstance).where(
        InfraInstance.run_id == run.id, InfraInstance.provider == "cluster",
        InfraInstance.released_at.is_(None),
    ).order_by(InfraInstance.created_at.desc()))).scalars().all())
    active_consumers = list((await session.execute(
        select(InfraInstance).join(Ticket, Ticket.id == InfraInstance.ticket_id).where(
            InfraInstance.run_id == run.id, InfraInstance.provider == "cluster",
            # A ticket can remain active across repair/collect activations after
            # its previous workload has been released. Only live resource claims
            # can block another stage from using the retained allocation.
            InfraInstance.released_at.is_(None),
            Ticket.id != ticket.id,
            Ticket.status.notin_(["succeeded", "degraded", "failed", "cancelled", "skipped"]),
        ).order_by(InfraInstance.created_at.desc())
    )).scalars().all())
    owner = None
    for row in candidates:
        meta = dict(row.meta or {})
        if not meta.get("controller_directory") or meta.get("allocation_owner_row_id"):
            continue
        if run.gpu_allocation_mode != "per_run" and row.ticket_id != ticket.id:
            continue
        state = await _retained_allocation_state(info, row.instance_id)
        if run.gpu_allocation_mode == "per_submission" and state not in _TERMINAL_ALLOCATION_STATES:
            raise ValueError("previous submission allocation is still active or terminating; retry after Slurm releases it")
        if state in _TERMINAL_ALLOCATION_STATES:
            # The provider no longer owns the allocation. It is safe to replace it.
            from datetime import datetime, timezone
            row.released_at = datetime.now(timezone.utc)
            row.release_reason = "retained allocation no longer live in Slurm"
            row.status = "released"
            continue
        if state not in {"RUNNING", "PENDING", "CONFIGURING"}:
            raise ValueError("cannot verify the retained allocation: allocation is not reusable or terminal: " + state)
        if row.gpu_count < contract.num_gpus or int(meta.get("nodes", 1)) != contract.nodes:
            raise ValueError("stage does not fit the retained run allocation; resource plan must cover all stages")
        # No concurrent commands on one set of GPUs. The orchestrator can retry
        # after the active consumer completes, never overlap torchrun processes.
        for consumer_index, consumer in enumerate(active_consumers):
            if (consumer.meta or {}).get("controller_directory") != meta["controller_directory"]:
                continue
            train = await session.get(Ticket, consumer.ticket_id)
            # Earlier completed segments remain as bookkeeping rows on the same
            # Train ticket. The verified current checkpoint supersedes them.
            current_checkpoint = next((other for other in active_consumers
                if other.ticket_id == consumer.ticket_id
                and (other.meta or {}).get("controller_directory") == meta["controller_directory"]
                and _checkpoint_allocation_handoff(run, ticket, other, train)), None)
            if (current_checkpoint is not None
                    and consumer_index > active_consumers.index(current_checkpoint)
                    and (consumer.meta or {}).get("scheduler_state") == "COMPLETED"):
                continue
            if not _checkpoint_allocation_handoff(run, ticket, consumer, train):
                raise ValueError("retained allocation is busy with another stage")
        if continuation and continuation.get("source_job_id") == row.instance_id:
            # The worker already yielded for walltime. Do not send it back into
            # the same exhausted window. Release this idle owned controller;
            # keep its row live until the watcher confirms actual termination.
            # active_consumers was checked above, so no other stage is killed.
            if publication is not None:
                publication["attempted"] = True
            release = await _ssh(info, upload_text_command(meta["controller_directory"] + "/release", "walltime continuation"))
            if not release.get("ok"):
                raise ValueError("could not release exhausted allocation for continuation")
            break
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
                                    idle_seconds=0 if run.gpu_allocation_mode == "per_run" else 1800,
                                    single_workload=run.gpu_allocation_mode == "per_submission",
                                    env_setup=info.cluster.env_setup)
        Path(contract.script_path).with_name("allocation-controller.sbatch").write_text(wrapper)
        command = publish + " && " + upload_text_command(wrapper_path, wrapper)
        command += " && " + _slurm_cli_bootstrap_command(info.cluster.env_setup, required=("sbatch", "squeue", "sacct", "scancel")) + "sbatch --parsable -- " + shlex.quote(wrapper_path)
    else:
        command = publish + " && printf '%s\\n' " + shlex.quote(owner.instance_id)
    if owner is not None:
        controller_body = (owner.meta or {}).get("controller_script_snapshot", "")
        if not controller_body:
            controller_body = await read_controller_headers(info, directory) or ""
    else:
        controller_body = wrapper
    # Force retirement writes before any external side effect. A deadlock here
    # is safely retryable; after publishing, an SSH failure is ambiguous.
    await session.flush()
    if publication is not None:
        publication["attempted"] = True
    result = await _ssh(info, _slurm_login_command(command), timeout_seconds=30)
    if not result.get("ok"):
        raise ValueError("controller submission failed: " + str(result.get("error") or result.get("stdout")))
    job_id = result.get("stdout", "").strip().splitlines()[-1].split(";", 1)[0]
    if not job_id.isdecimal():
        raise ValueError(f"controller returned invalid job id: {job_id!r}")
    return job_id, {
        "controller_script_snapshot": "\n".join(line for line in controller_body.splitlines() if line.startswith("#SBATCH")),
        "controller_workload_launch": ((owner.meta or {}).get("controller_workload_launch", "legacy_bash")
                                       if owner is not None else "host_srun_v1"),
        "allocation_walltime": walltime,
        "controller_directory": directory,
        "controller_outcome_path": outcome,
        "controller_request_id": heartbeat_id,
        "allocation_owner_row_id": owner.id if owner is not None else "",
        "allocation_owner_ticket_id": owner.ticket_id if owner is not None else ticket.id,
        "gpu_allocation_mode": run.gpu_allocation_mode or "per_stage",
    }
