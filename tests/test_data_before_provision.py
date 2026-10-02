"""Data 0 runs before Infrastructure when it does not need the GPU host.

Smoke runs idled a freshly rented A10 for 5–12 minutes while Data fetched and
mapped rows that never touched it (run 7: provisioned 10:39, Data 10:40–10:46,
first GPU use 10:48). The engine already allows an iteration-0 Data ticket
without a device binding unless the source is a Hub id; these tests pin that,
so the playbook's new ordering rests on enforced behaviour.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from zevo.contracts.data import DataTaskInput
from zevo.contracts.tickets import DataPayload
from zevo.db.models import Run, Ticket
from zevo.engine.run.runner import _build_data_input


def _ticket(payload: dict) -> Ticket:
    return Ticket(id="data-r1-001", run_id="r1", agent_id="data", status="queued",
                  input_format="typed", lane="optimization", iteration=0, payload=payload)


def _run() -> Run:
    return Run(id="r1", metric="accuracy", validation_metric="accuracy", gpu_provider="cloud",
               decision_pins={"max_training_rows": 500})


def test_query_sourced_data_needs_no_device_binding() -> None:
    payload = DataPayload(operation="prepare_run_data", data_query="small trivia QA",
                          training_method="sft").model_dump()
    inp = _build_data_input(_ticket(payload), payload, {}, "/w", _run(), {})
    assert isinstance(inp, DataTaskInput)
    assert inp.device_info_path == ""
    assert inp.remote_data_output_dir == "" and inp.remote_data_helper_path == ""
    assert inp.max_training_rows == 500


def test_hub_sourced_data_requires_the_device_binding() -> None:
    payload = DataPayload(operation="prepare_run_data", dataset="org/trivia-train",
                          training_method="sft").model_dump()
    with pytest.raises((KeyError, ValueError, ValidationError), match="device_info"):
        _build_data_input(_ticket(payload), payload, {}, "/w", _run(), {})
