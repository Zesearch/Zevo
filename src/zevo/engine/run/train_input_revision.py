"""Positive evidence that a stopped submission never began its workload."""
from __future__ import annotations

import json
import shlex

from zevo.engine.run.remote_jobs import _device_info_for_ticket, _ssh, _slurm_login_command, _slurm_cli_bootstrap_command


async def submission_never_started(db, ticket, row):
    # A cancellation request alone is not confirmation that execution stopped.
    if row.released_at is None or row.provider != "cluster":
        return False
    meta = row.meta or {}
    info = await _device_info_for_ticket(db, ticket)
    if info is None or info.cluster is None:
        return False
    if meta.get("controller_directory"):
        # The retained allocation may have run other tickets. Check this exact
        # request's receipt, not the allocation's start time.
        from pathlib import PurePosixPath
        request = str(meta.get("controller_request_id") or "")
        path = str(meta.get("controller_outcome_path") or "")
        expected = str(PurePosixPath(meta["controller_directory"]) / "outcomes" / (request + ".json"))
        if not request or "/" in request or path != expected:
            return False
        result = await _ssh(info, "cat -- " + shlex.quote(path), stdout_limit=4096)
        try:
            receipt = json.loads(result.get("stdout", ""))
            return (result.get("ok") is True and receipt.get("exit_code") == 130
                    and receipt.get("error") == "cancelled before execution")
        except (ValueError, AttributeError):
            return False
    if not str(row.instance_id).isdigit():
        return False
    command = (_slurm_cli_bootstrap_command(info.cluster.env_setup, required=("sacct",))
               + "sacct -n -P -X -j " + shlex.quote(row.instance_id)
               + " --format=JobIDRaw,State,Start")
    result = await _ssh(info, _slurm_login_command(command), stdout_limit=4096)
    if not result.get("ok") or result.get("stdout_truncated"):
        return False
    records = [line.strip().split("|") for line in result.get("stdout", "").splitlines()]
    matches = [v for v in records if len(v) >= 3 and v[0] == row.instance_id]
    # Missing accounting history is uncertainty, not proof of no execution.
    return (len(matches) == 1 and matches[0][1].split(" ")[0] == "CANCELLED"
            and matches[0][2] in {"Unknown", "None", "N/A"})
