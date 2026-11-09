"""Hardware replacement requires a structured resource-plan review request."""
from __future__ import annotations

import pytest

from zevo.contracts.failure import FailureRecovery
from zevo.engine.run.failure_policy import HARDWARE_DEFECT_PREFIX, classify_failure


@pytest.mark.parametrize("message", [
    HARDWARE_DEFECT_PREFIX + " device 0: Remapping Failure Occurred: Yes, 8 uncorrectable rows",
    "vLLM engine died: NVRM: Xid 48 (PCI:0000:0a:00): An uncorrectable double bit error",
    "nvidia-smi: Volatile Uncorrected ECC errors 248781 on GPU 0",
    "row remapping failure on GPU 0 (pending=Yes, failure occurred=Yes)",
    "Retired Pages Pending : Yes on device 1",
    "Xid 79: GPU has fallen off the bus",
])
@pytest.mark.parametrize("agent_id", ["inference", "train", "infrastructure"])
def test_defective_gpu_goes_to_the_orchestrator(message: str, agent_id: str) -> None:
    assert classify_failure(message, agent_id=agent_id).route == "self"  # diagnostic only
    request = FailureRecovery(scope="resource_plan", reason="Verified defective GPU requires replacement",
                              evidence=[message], next_action="Request replacement of the assigned GPU")
    disposition = classify_failure(message, agent_id=agent_id, recovery=request)
    assert disposition.route == "orchestrator"
    assert disposition.code == "recovery_resource_plan"
    assert "replacement" in disposition.reason


@pytest.mark.parametrize("message", [
    "ECC: Enabled; 0 volatile uncorrected errors",
    "CUDA out of memory. Tried to allocate 2.00 GiB",
    "ValidationError: 1 validation error for InferenceResult",
    "torch.cuda.OutOfMemoryError while loading the model",
])
def test_healthy_gpu_messages_stay_with_the_agent(message: str) -> None:
    assert classify_failure(message, agent_id="train").route == "self"


def test_cancellation_still_wins_over_a_defect_report() -> None:
    disposition = classify_failure(
        HARDWARE_DEFECT_PREFIX + " Xid 48", agent_id="train", cancelled=True,
    )
    assert disposition.route == "terminal" and disposition.code == "cancelled"
