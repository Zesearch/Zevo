"""A successful infra Ticket must not carry the resolved resource plan in its
summary/error text.

The resolved plan's durable homes are ``device_info.json`` and the Ticket's
structured ``resource_plan`` artifact meta. Regression guard for the bug where a
succeeded infra Ticket showed the plan text (``... persisted to plan.json:
cloud_backend=lambda ... min_vram_gb=48 ...``) in an ERROR field.
"""
from __future__ import annotations

import json

from zevo.contracts.infrastructure import InfraResult, InfraTaskInput
from zevo.engine.run.runner import (
    _extract_summary_artifact_meta,
    _infra_provision_summary,
    _set_current_ticket_outcome_text,
)

RUN_ID = "run-abc123"
TICKET_ID = "infra-abc123-001"

# The exact style of free-text the infra agent used to dump into notes/error.
PLAN_DUMP_NOTES = (
    "Resource plan resolved before acquisition and persisted to plan.json: "
    "cloud_backend=lambda gpu_type=A100 num_gpus=1 min_vram_gb=48 "
    "min_ram_gb=64 min_cpus=8 time_limit_hours=8"
)


def _cluster_device_info() -> dict:
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "ticket_id": TICKET_ID,
        "purpose": "inference",
        "provider": "cluster",
        "cloud_backend": "",
        "instance_id": "",
        "auto_release": False,
        "host": "login.example.com",
        "ssh": {"host": "login.example.com", "port": 22, "user": "user", "key_path": "/k"},
        "cluster": {
            "jobid": "",
            "node": "",
            "requested_gpus": 1,
            "workdir": "/scratch/zevo/run",
            "hf_cache": "/scratch/zevo/hf_cache",
        },
        "instance": None,
        "gpu": None,
        "cuda": None,
        "cost": {"dph_total": 0},
        "resource_plan": {
            "purpose": "inference",
            "required_working_set_gib": 12,
            "host_memory_components_gib": {"model_and_runtime": 12},
            "host_memory_formula_gib": 32,
            "site_min_ram_gib": 0,
            "num_gpus": 1,
            "min_vram_gb": 48,
            "min_ram_gb": 32,
            "min_cpus": 8,
            "time_limit_hours": 8,
            "disk_gb": 0,
            "rationale": "SSH-backed Slurm system with A100 partition",
        },
        "probe_source": "ssh-environment",
        "probed_at": "2026-08-29T00:00:00Z",
    }


def _infra_task_input(work_dir: str) -> InfraTaskInput:
    return InfraTaskInput(
        ticket_id=TICKET_ID,
        run_id=RUN_ID,
        purpose="inference",
        provider="cluster",
        num_gpus=1,
        auto_release=False,
        instance_id="",
        work_dir=work_dir,
        remote_dir="/scratch/zevo",
        ssh_host="login.example.com",
        ssh_port=22,
        ssh_user="user",
        ssh_key_path="/k",
        gpu_lease_request_schema={},
        gpu_lease_grant_schema={},
        infra_instance_create_schema={},
        infra_instance_patch_schema={},
        infra_instance_response_schema={},
        device_info_schema={},
        device_info_validation_command="true",
    )


def _succeeded_provision(tmp_path):
    device_info = tmp_path / "device_info.json"
    device_info.write_text(json.dumps(_cluster_device_info()), encoding="utf-8")
    result = InfraResult(
        status="succeeded",
        ticket_id=TICKET_ID,
        operation="provision",
        device_info_path=str(device_info),
        error_message="",
        notes=PLAN_DUMP_NOTES,
    )
    inp = _infra_task_input(str(tmp_path))
    return _extract_summary_artifact_meta(
        result, inp=inp, agent_id="infrastructure",
        work_dir=str(tmp_path), run_id=RUN_ID,
    )


def test_success_summary_is_engine_authored_not_the_plan_dump(tmp_path) -> None:
    status, summary, artifact, meta = _succeeded_provision(tmp_path)

    assert status == "succeeded"
    # The agent's free-text plan dump must NOT become the Ticket summary/error.
    assert "persisted to plan.json" not in summary
    assert PLAN_DUMP_NOTES not in summary
    # The summary is the concise, engine-authored provision line.
    assert summary.startswith("provision succeeded on cluster")
    assert "num_gpus=1" in summary
    assert "min_vram_gb=48" in summary


def test_resolved_plan_remains_retrievable_from_structured_meta(tmp_path) -> None:
    _status, _summary, _artifact, meta = _succeeded_provision(tmp_path)

    plan = meta["resource_plan"]
    assert plan["num_gpus"] == 1
    assert plan["min_vram_gb"] == 48
    assert plan["min_ram_gb"] == 32
    assert plan["min_cpus"] == 8
    assert plan["time_limit_hours"] == 8
    assert plan["rationale"] == "SSH-backed Slurm system with A100 partition"


def test_succeeded_infra_ticket_error_message_is_empty(tmp_path) -> None:
    status, summary, _artifact, _meta = _succeeded_provision(tmp_path)

    class _Ticket:
        summary = ""
        error_message = ""

    ticket = _Ticket()
    _set_current_ticket_outcome_text(ticket, status=status, summary=summary)

    # error_message must be empty on a successful infra Ticket, and must never
    # carry the resolved plan.
    assert ticket.error_message == ""
    assert "persisted to plan.json" not in ticket.summary
    assert "min_vram_gb" in ticket.summary  # the clean engine line survives


def test_provision_summary_helper_does_not_leak_the_full_plan(tmp_path) -> None:
    from zevo.contracts.infrastructure import InfrastructureDeviceInfo

    info = InfrastructureDeviceInfo.model_validate(_cluster_device_info())
    line = _infra_provision_summary(info)

    assert "provision succeeded on cluster" in line
    assert "min_vram_gb=48" in line
    # A one-line summary, not the full plan object / rationale prose.
    assert info.resource_plan.rationale not in line
    assert "\n" not in line
