from pathlib import Path

import pytest
from pydantic import BaseModel

from zevo.contracts._base import AgentTaskInput
from zevo.contracts.infrastructure import SlurmStageJobContract
from zevo.engine.agent.drivers._prompt import _payload_block
from zevo.engine.run.runner import _slurm_stage_result_problem, _stage_gpu_runtime_helper


class FutureGpuInput(AgentTaskInput):
    slurm_job: SlurmStageJobContract


class PlainInput(BaseModel):
    pass


def test_every_typed_specialist_receives_same_helper_and_driver_instructions(tmp_path):
    original = AgentTaskInput(ticket_id="future-12345678-001")
    staged = _stage_gpu_runtime_helper(original, str(tmp_path))
    assert original.gpu_runtime_helper_path == ""
    assert Path(staged.gpu_runtime_helper_path).read_text().startswith('"""Portable GPU runtime')
    prompt = _payload_block(staged)
    assert "ANY GPU" in prompt
    assert "CPU-only work does not require" in prompt
    assert "zevo_gpu_runtime.launch(argv" in prompt
    assert "Shared GPU execution contract" not in _payload_block(original)
    assert "gpu_runtime_helper_path" not in _stage_gpu_runtime_helper(PlainInput(), str(tmp_path)).model_dump()


@pytest.mark.parametrize("stage", ["data", "train", "inference", "future_gpu_stage"])
def test_gpu_submission_guard_is_independent_of_stage(tmp_path, stage):
    path = tmp_path / "work.sbatch"
    (tmp_path / "workload.py").write_text("print('work')\n")
    # Deliberately use a future stage as well: this guard depends on resources,
    # not the current finite list of scheduler-supported stages.
    contract = SlurmStageJobContract.model_construct(
        enabled=True, phase="submit", stage=stage, num_gpus=1, nodes=1, gpus_per_node=1,
        script_path=str(path), job_name="job", status_path="status", stdout_path="out", stderr_path="err",
        runtime_prologue="export TMPDIR=/tmp/job", lifecycle_prologue="export STATUS=status",
    )
    inp = FutureGpuInput.model_construct(ticket_id="future-12345678-001", slurm_job=contract,
                                        gpu_runtime_helper_path="/assigned/zevo_gpu_runtime.py")
    prefix = "#SBATCH --job-name=job\n#SBATCH --gpus=1\n#SBATCH --output=out\n#SBATCH --error=err\nexport TMPDIR=/tmp/job\nexport STATUS=status\n"
    path.write_text(prefix + "# zevo_gpu_runtime.py is only a comment\npython workload.py\n")
    assert "GPU workload must launch" in _slurm_stage_result_problem(inp, str(path), "workload.py")
    for launcher in ("zevo_gpu_runtime.py --", "zevo_parallel_inference.py", "zevo_adaptive_training.py"):
        path.write_text(prefix + f"python {launcher} python workload.py\n")
        assert _slurm_stage_result_problem(inp, str(path), "workload.py") == ""
    # Existing remote submissions are collected unchanged, even after upgrade.
    contract.phase = "collect"
    path.write_text(prefix + "python workload.py\n")
    assert _slurm_stage_result_problem(inp, str(path), "workload.py") == ""
