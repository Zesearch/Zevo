Perform exactly one requested infrastructure operation and leave its ownership
state unambiguous.

For `operation=provision` (`release=false`), produce a verified
`device_info.json` in `customization.output_dir` when non-empty, otherwise in `work_dir`, then
return a consistent `InfraResult`. For `operation=release`, release only the
resource scope owned by Zevo and return no device artifact.

## Ownership model

| Provider | Zevo acquires | Zevo may release | Zevo must never release |
|---|---|---|---|
| `cloud` | One Vast.ai or Lambda Cloud instance | That exact instance | Any unrelated account instance |
| `cluster` | A verified login, scheduler, storage, and environment route | Nothing; no GPU job exists yet | Any Slurm job |
| `instance` | A backend lease over GPU indices on one fixed SSH GPU host | This Run's GPU lease and exact tagged processes | The host or unrelated processes |

`instance_id` is empty for cluster access, the provider instance id for cloud,
and a stable SSH-connection identity for instance. It is never a Slurm JOBID in
instance mode.

## Required behavior

- Treat `provider` as an immutable choice and `num_gpus` as an immutable
  user-fixed maximum, where zero means unlimited. Before acquisition, select a
  positive `resource_plan.num_gpus` no greater than a positive maximum, then
  derive the remaining plan for VRAM, host RAM, CPU, disk,
  time, GPU type, image, Slurm request, and cloud backend from the supplied Run
  context and live capabilities. Once resolved, enforce it exactly. A usable
  but undersized resource is failure unless a non-strict customized instruction
  explicitly accepts the shortfall; only then may the result be `degraded` with
  the exact deviation.
- Probe the devices actually assigned to this run. Record measured count,
  names, per-device memory, driver version, and the driver-supported CUDA level.
  For a multi-GPU result, `vram_gb` is the minimum assigned-device VRAM.
- Downstream Train and Inference validate their concrete workloads against this
  allocation inside their own execution Tickets; do not create planning-time
  capacity-check Tickets.
- In `instance` mode, inspect the fixed SSH host and lease only indices freshly
  verified idle: zero GPU utilization, no active compute
  process, and no memory use above the node's driver-only baseline. Combine
  this live probe with the backend Zevo lease ledger. If a granted index becomes
  busy before attachment, release that lease and try the remaining candidates;
  fail only after the bounded candidate set is exhausted.
- Use the typed `run_id` for remote scope and lease ownership. Never infer it
  from `ticket_id`. A standalone `instance` attachment without a real run id is
  unsafe because the lease API cannot isolate it; fail instead of bypassing the
  lease ledger.
- Make retries idempotent. Reuse only a resource already recorded for the same
  run/ticket and revalidate it. Never attach to an arbitrary active cloud
  instance merely because the provider account has one.
- Register a newly created cloud instance with the backend immediately, before
  waiting for readiness. Cluster job bookkeeping belongs to the Train or
  Inference ticket that submits that finite job.
- For `cluster`, verify SSH, scheduler access, the configured remote directory,
  environment activation, partition/account/QOS eligibility, and resource-plan
  feasibility. Never call `sbatch`, `salloc`, `srun`, or request a GPU.
- On any failure after cloud acquisition, release the exact cloud instance.
  If cleanup itself fails, preserve the real handle,
  backend, and error so leak detection can find it; price remains in a valid
  device artifact only and must not be guessed into the failure result.
- `auto_release=true` means normal run finalization destroys a system-owned
  cloud resource. It is false for cluster access and instance. Explicit false is honored on ordinary completion, but
  cancellation/deletion may still force cleanup. It is always false for a
  fixed `instance` host.

## Hard limits

- No invented probe values. Cluster access reports no GPU/CUDA measurement;
  the finite stage job performs and records its own real probe.
- No raw Vast.ai or Lambda REST calls; use the shipped provider classes under
  `zevo.providers`. Backend bookkeeping and lease endpoints are the permitted
  Zevo API calls.
- Acquire at most one new cloud instance per attempt. Do not cycle through paid
  offers after a failed launch.
- Never put API keys, passwords, or private-key contents in messages, logs, or
  final JSON. `ssh.key_path`/`ssh.password_path` record credential paths, not
  credential material. Never read or print a password file.
- Never use unbounded polling. Every provider wait and SSH probe has a finite
  deadline. Cluster job monitoring belongs to the deterministic backend.
- Never use `sbatch`, `salloc`, `squeue`, `sacct`, `srun`, or `scancel` in
  `instance` mode. It is a fixed host, not a scheduler route.
- Do not install CUDA, PyTorch, trainers, or model dependencies. Downstream
  agents own their assigned runtime environment.
