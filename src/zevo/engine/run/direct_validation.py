"""Pre-publication checks for the direct-host workload adapter."""
from pathlib import Path
import hashlib
import subprocess


def validate_launch(inp, body, executable_name, *, configuration_path=""):
    from zevo.engine.run.runner import (
        _python_undefined_name_problem, _python_memory_helper_problem,
        _python_stage_resume_problem, _python_checkpoint_helper_problem,
    )
    from zevo.contracts.configuration import load_train_config, load_inference_config
    from zevo.engine.run.inference_launch import verify_frozen_inputs
    job = inp.slurm_job
    script = Path(job.script_path)
    if "#SBATCH" in body or "sbatch " in body or "srun " in body:
        return "direct workloads must not submit Slurm jobs"
    checked = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, timeout=10)
    if checked.returncode:
        return "invalid workload shell syntax: " + checked.stderr[:1500]
    if executable_name not in body:
        return "workload launcher must invoke its assigned implementation"
    if job.phase == "collect":
        from zevo.engine.run.runner import _slurm_stage_implementation_sha256
        if (not job.implementation_sha256
                or _slurm_stage_implementation_sha256(job) != job.implementation_sha256):
            return "collected implementation differs from the registered execution"
        expected = getattr(inp, "slurm_launch_reuse", {}).get("expected_sha256")
        if expected and hashlib.sha256(body.encode()).hexdigest() != expected:
            return "executed workload launcher changed during collection"
        return ""
    problem = _python_undefined_name_problem(str(script.with_name(executable_name)))
    if problem:
        return problem
    commands = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    if getattr(inp, "gpu_runtime_helper_path", ""):
        from zevo.engine.run.python_capabilities import gpu_launch_problem
        problem = gpu_launch_problem(body, script.with_name(executable_name))
        if problem:
            return problem
    try:
        if executable_name == "predict.py":
            from zevo.contracts.configuration import validate_adaptive_vllm_memory_config
            from zevo.contracts.inference_core import validate_inference_core
            config = load_inference_config(configuration_path)
            validate_inference_core(config, inp.shared_inference_core)
            verify_frozen_inputs(inp.frozen_reuse_artifacts)
            if "--adaptive-vllm-memory" in inp.config_validation_command:
                validate_adaptive_vllm_memory_config(config)
                problem = _python_memory_helper_problem(str(script.with_name(executable_name)))
                if problem:
                    return problem
        elif executable_name == "train.py":
            config = load_train_config(configuration_path)
            if config.training.adaptive_evaluation != inp.adaptive_mid_training_evaluation:
                return "adaptive_evaluation must match the Run switch"
            if not 1 <= config.training.world_size <= job.num_gpus:
                return "training world_size exceeds the existing allocation"
            problem = _python_checkpoint_helper_problem(str(script.with_name(executable_name)))
            if problem:
                return problem
        if (getattr(inp, "stage_resume_helper_path", "") and executable_name in {"prepare_data.py", "predict.py"}
                and "zevo_parallel_inference.py" not in commands):
            return _python_stage_resume_problem(str(script.with_name(executable_name)))
    except (OSError, ValueError) as exc:
        return "workload configuration is invalid: " + str(exc)
    return ""
