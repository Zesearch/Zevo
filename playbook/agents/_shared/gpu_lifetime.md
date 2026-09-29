## GPU resource lifetime

Read `run_context.runtime.gpu_allocation_mode` (Infrastructure also receives
`gpu_allocation_mode` directly). In both modes, a code bug is repaired on the
same healthy resources. Preserve the error logs, environment, data and
checkpoints. Stop failed workers, correct the implementation and retry within
the engine's repair limit. Do not release resources or create a replacement
because a child process returned nonzero.

`per_stage` releases resources after the GPU stage's final validation;
`per_run` retains them across stages until the Run ends. The backend owns this
boundary. For Slurm, prepare the validated workload `.sbatch` file and return
`deferred` as before. The backend runs it under an allocation controller, so a
workload completion is distinct from allocation termination. A repair may use
the same JOBID with a new attempt directory. Never scancel the parent allocation
or submit a separate repair job yourself. Allocation expiry, preemption or
node loss requires replacement; code errors do not. Every allocation remains
subject to site walltime and provider policies. Per-run controllers have no
fixed idle handoff timeout; per-stage controllers retain a 30-minute idle limit.

In `per_run`, prepare the full run's resource envelope at Infrastructure time;
subsequent stages must fit that allocation. The controller uses the train-sized
GPU, CPU, RAM and walltime plan even if the first consumer is Data/Inference.
