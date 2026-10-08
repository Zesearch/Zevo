## GPU resource lifetime

Read `run_context.runtime.gpu_allocation_mode` (Infrastructure also receives
`gpu_allocation_mode` directly). In `per_run` and `per_stage`, a code bug is repaired on the
same healthy resources. Preserve the error logs, environment, data and
checkpoints. Stop failed workers, correct the implementation and retry within
the engine's repair limit. Do not release resources or create a replacement
because a child process returned nonzero.

`per_stage` releases resources after the GPU stage's final validation;
`per_run` retains them across stages until the Run ends. The backend owns this
boundary. For Slurm in these two retained modes, prepare the validated workload `.sbatch` file and return
`deferred` as before. The backend runs it under an allocation controller, so a
workload completion is distinct from allocation termination. A repair may use
the same JOBID with a new attempt directory. Never scancel the parent allocation
or submit a separate repair job yourself. Allocation expiry, preemption or
node loss requires replacement; code errors do not. Every allocation remains
subject to site walltime and provider policies. Per-run controllers have no
fixed idle handoff timeout; per-stage controllers retain a 30-minute idle limit.

In `per_submission` (Slurm only), each actual workload execution gets a fresh
allocation. After success, failure or cancellation, the controller cleans up
workers, publishes the outcome and exits immediately. Repair and collection
do not retain GPUs; the next validated workload submission requests a new
allocation, even within the same stage. Store logs, data and checkpoints on
shared durable storage, not node-local scratch. Do not start idle GPU work
between submissions. Submission and release remain backend-owned.

In `per_run`, prepare the full run's resource envelope at Infrastructure time;
subsequent stages must fit that allocation. The controller uses the train-sized
GPU, CPU, RAM and walltime plan even if the first consumer is Data/Inference.

## Verification after implementation changes

Choose checks that exercise the behavior changed and the failure being repaired.
Check configuration, input formats, and script syntax directly where sufficient.
For device, distributed-runtime, or kernel failures, verify the repair during
startup of the actual workload on its assigned resources. Once startup succeeds,
continue that same execution through the full task. Reuse evidence and unchanged
validated artifacts from the preceding activation; repeat checks when a change
invalidates them. Record what each check establishes and what remains unverified.
