"""Read-only, submission-time Slurm GPU capacity probe.

The snapshot is a sizing hint, not a reservation: Slurm remains responsible
for the actual allocation and may queue the job if capacity changes.
"""
from __future__ import annotations

import asyncio
import re
import shlex

from zevo.contracts.infrastructure import InfrastructureDeviceInfo
from zevo.engine.remote_transfer import ssh_base_args
from zevo.engine.run.resource_planning import SlurmCapacitySnapshot


_QOS_MARKER = "__ZEVO_QOS_LIMIT__"
_QUEUE_MARKER = "__ZEVO_RUNNING_QOS_JOBS__"
_GPU_GRES = re.compile(r"(?:^|,)gpu(?::[^:,()]+)?:([0-9]+)(?=,|\(|$)")
_GPU_TRES = re.compile(r"(?:^|,)gres/gpu=([0-9]+)(?=,|$)")
_TYPED_GPU_TRES = re.compile(r"(?:^|,)gres/gpu:[^=,]+=([0-9]+)(?=,|$)")


def _gpu_tres_count(value: str) -> int | None:
    generic = _GPU_TRES.search(value)
    if generic:
        return int(generic.group(1))
    typed = _TYPED_GPU_TRES.findall(value)
    return sum(map(int, typed)) if typed else None


def parse_slurm_capacity(
    output: str, *, qos: str = "",
) -> SlurmCapacitySnapshot:
    """Parse one bounded SSH response, rejecting incomplete scheduler data."""
    sections = output.split(f"\n{_QOS_MARKER}\n", 1)
    node_lines = sections[0].splitlines()
    if not node_lines:
        raise ValueError("sinfo returned no nodes")
    free_by_node: list[int] = []
    for line in node_lines:
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 4:
            raise ValueError("sinfo returned an unrecognized node row")
        _, total_gres, used_gres, state = fields
        if not state.lower().startswith(("idle", "mix")):
            continue
        totals = _GPU_GRES.findall(total_gres)
        used = _GPU_GRES.findall(used_gres)
        if not totals and total_gres in {"(null)", "N/A", ""}:
            continue  # CPU-only node in a mixed partition.
        if not totals or (used_gres not in {"(null)", "N/A", ""} and not used):
            raise ValueError("sinfo omitted GPU GRES accounting")
        free_by_node.append(max(0, sum(map(int, totals)) - sum(map(int, used))))

    qos_free: int | None = None
    if qos:
        if len(sections) != 2:
            raise ValueError("Slurm QoS limits were not returned")
        qos_sections = sections[1].split(f"\n{_QUEUE_MARKER}\n", 1)
        if len(qos_sections) != 2:
            raise ValueError("running QoS jobs were not returned")
        qos_rows = [line.split("|", 1) for line in qos_sections[0].splitlines() if line]
        matching = [row for row in qos_rows if len(row) == 2 and row[0].strip() == qos]
        if len(matching) != 1:
            raise ValueError("Slurm did not return the selected QoS")
        cap = _gpu_tres_count(matching[0][1].strip())
        if cap is not None:
            used_total = 0
            for line in qos_sections[1].splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) != 2 or parts[0] != qos:
                    raise ValueError("squeue returned an unrecognized QoS allocation")
                allocated = _gpu_tres_count(parts[1].strip())
                if allocated is None:
                    raise ValueError("squeue omitted running job GPU TRES")
                used_total += allocated
            qos_free = max(0, cap - used_total)
    return SlurmCapacitySnapshot(tuple(free_by_node), qos_free)


async def probe_slurm_capacity(info: InfrastructureDeviceInfo) -> SlurmCapacitySnapshot:
    """Query node headroom and (when configured) the selected QoS GPU cap."""
    if info.provider != "cluster" or info.cluster is None or info.ssh is None:
        raise ValueError("Slurm capacity probe requires a cluster SSH route")
    route = info.cluster
    partition = route.partition or info.resource_plan.slurm_partition
    qos = route.qos or info.resource_plan.slurm_qos
    sinfo = (
        "sinfo -N -h "
        + (f"-p {shlex.quote(partition)} " if partition else "")
        + "-O 'NodeList:40|,Gres:120|,GresUsed:120|,StateCompact:30'"
    )
    commands = ["set -e", sinfo]
    if qos:
        quoted_qos = shlex.quote(qos)
        commands.extend([
            f"printf '\\n{_QOS_MARKER}\\n'",
            f"sacctmgr -P -n show qos {quoted_qos} format=Name,GrpTRES",
            f"printf '\\n{_QUEUE_MARKER}\\n'",
            f"squeue -h -q {quoted_qos} -t RUNNING,COMPLETING,CONFIGURING "
            "-O QOS:50,TRES-Alloc:256",
        ])
    command = "\n".join(commands)
    process = await asyncio.create_subprocess_exec(
        *ssh_base_args(info.ssh), f"{info.ssh.user}@{info.ssh.host}", command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=45)
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise ValueError("Slurm capacity probe timed out") from exc
    if process.returncode != 0:
        raise ValueError("Slurm capacity probe command failed")
    if len(stdout) > 1_000_000:
        raise ValueError("Slurm capacity probe output exceeded 1 MB")
    return parse_slurm_capacity(stdout.decode("utf-8", errors="replace"), qos=qos)
