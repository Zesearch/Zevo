"""A defective GPU is routed to whoever can replace it, never self-repaired.

Smoke run 5b4a67e0 landed on a Lambda A10 with an exhausted row remapper
(Xid 48, ~249k uncorrectable ECC errors). Inference asked for a swap; the
orchestrator re-woke inference and then train on the same card, and the train
agent spent three activations finding a 13 GiB reservation that avoided the
bad rows: ~40 minutes and ~$10. The failure policy now names that failure and
sends it upstream.
"""
from __future__ import annotations

import pytest

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
    disposition = classify_failure(message, agent_id=agent_id)
    assert disposition.route == "orchestrator"
    assert disposition.code == "hardware_defect"
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
