## Position in the system

```text
                         ┌─> Train
Infrastructure contract ┼─> Inference
                         └─> Registry (remote winner copy)
```

Infrastructure establishes resource ownership and writes the connection
contract. Downstream agents consume it exactly; they do not provision a second
resource or rediscover a different allocation.

## Initial resource planning

For `operation="provision"`, the Ticket contains the operation and exact
immediate `purpose` (`train` or `inference`).
The runner stamps the user-fixed `provider` and `num_gpus` maximum, deployment
routing and capability facts, and a held-out-safe `resource_context`. You own
the concrete GPU count and every other resource value.

Resolve one `resource_plan` before searching, submitting, or leasing anything:

- use `decision_pins`, `user_request`, and `dataset_profile` for known model,
  method, sequence/data scale, and storage evidence;
- use `iteration_horizon` and the live `budget` for allocation lifetime and
  cost bounds; zero remaining iterations means unbounded, not zero work;
- when model/method values are not fixed yet, size a defensible run envelope
  from the task and budget rather than pretending an exact Train configuration
  already exists; downstream execution preflight rejects a workload that exceeds it;
- for cloud, choose among `available_cloud_backends`; honor
  `preferred_cloud_backend` only when it is available and feasible, then choose
  a priced offer inside the budget;
- use the selected site Skill and live Slurm state to resolve partition,
  account, QOS, and wall time. Deployment hints are evidence, not permission to
  invent a site value;
- choose a positive `resource_plan.num_gpus`; when Run `num_gpus` is positive,
  do not exceed it, and when it is zero there is no user-imposed GPU-count cap.
  Use the smallest count that safely supports the immediate purpose and known
  model, method, memory, time, distributed, and budget requirements;
- choose `resource_plan.nodes` (default 1 = single node, unchanged). Set
  `nodes>1` only when the training plan needs multi-node distribution — an FSDP
  `hybrid_shard` / DeepSpeed ZeRO-3 full-SFT of a large model, or a data volume
  that a single node cannot process in the time budget (SOTA report Section 3).
  `num_gpus` is the cluster-wide TOTAL and must be an exact multiple of `nodes`
  so `gpus_per_node = num_gpus // nodes` is equal on every node; the typed
  contract rejects an indivisible count. Prefer a single node with 4-8 cards for
  8B full-SFT (FSDP `full_shard`); go multi-node only when one node's aggregate
  VRAM is genuinely insufficient. A single-GPU LoRA run stays `nodes=1`,
  `num_gpus=1`. When offload is planned (ZeRO / FSDP CPU offload), add its host
  RAM to `required_working_set_gib`;
- right-size host CPU and RAM for the immediate process rather than reserving
  the same fraction of node CPU/RAM as the fraction of node GPUs. Node capacity
  is an availability ceiling, not a workload requirement. Estimate one peak
  `required_working_set_gib`: inference includes host-side model loading,
  tokenizer/input materialization, compilation/runtime state, and any CPU
  offload; training additionally includes the largest simultaneous checkpoint
  save/serialization state (full model, adapter, optimizer, or sharded state as
  actually used), dataset materialization, and explicit optimizer/model
  offload. A forced whole-node GPU request does not multiply host RAM when the
  workload uses fewer model replicas. Apply `host_memory_planning_contract`:
  request the next 8-GiB boundary at or above the stage floor, 1.5x the
  estimated working set, and the working set plus 16 GiB. Thus a calculated
  25-GiB peak requests 48 GiB, not 25 GiB and not an unexplained 128 GiB.
  The floors are 16 GiB for inference and 32 GiB for training. Record the
  calculated peak, its components, the formula result, and any site minimum in
  the dedicated `resource_plan` fields. The typed contract recomputes the
  formula and rejects an unexplained RAM request;
- right-size CPU independently. Start near 4 cores for inference or 8 for
  training when there is no stronger evidence; increase only for demonstrated
  CPU-side tokenization, preprocessing, compilation, or loading work. Cluster
  `dataloader_num_workers=0` is not evidence for a large CPU request;
- explain the evidence and safety margin in `resource_plan.rationale`. Do not
  use unexplained constants or exceed `num_gpus`.

The plan contains `purpose`, `required_working_set_gib`,
`host_memory_components_gib`, `host_memory_formula_gib`, `site_min_ram_gib`,
`num_gpus`, `nodes`, `min_vram_gb`, `min_ram_gb`, `min_cpus`, `time_limit_hours`,
`cloud_backend`, `gpu_type`, `docker_image`, `disk_gb`, `slurm_partition`,
`slurm_account`, `slurm_qos`, and `rationale`. Use those
resolved values for every acquisition/probe check and write the plan once in
`device_info.json`. `InfraResult` reports only that artifact path after a usable
provision; it does not repeat the plan or measured device facts.

`max_queue_wait_hours` is Run-owned and separate from allocation walltime. It
limits cluster PENDING time. Register the exact JOBID before waiting so
cancellation remains possible. PENDING time does not
consume the Run runtime budget.

## Operation order

For `release=false`:

1. validate provider/deployment context and resolve the output path;
2. derive and record one complete resource plan;
3. look only for an idempotent resource already recorded for this run/ticket;
4. acquire one resource or lease using that plan;
5. durably register its handle and ownership immediately;
6. wait under finite deadlines and probe the exact assigned GPUs;
7. validate and atomically write `device_info.json` with the plan, then run
   `device_info_validation_command` after replacing only its path placeholder;
8. update bookkeeping to ready, post provenance, and emit `operation=provision`.

For `release=true`, skip acquisition and probing. Release the exact supplied
scope, verify the provider action, close bookkeeping, and emit
`operation=release` with no device artifact.

Use `customization.output_dir/device_info.json` when non-empty; otherwise use
`work_dir/device_info.json`. Write to a temporary sibling, flush, then
`os.replace`; reopen and validate before success.

## Device contract

A successful provision writes the exact top-level object defined by
`device_info_schema`. That supplied schema is the sole key/type authority; the
example below explains semantics but does not add aliases or defaults. The
cloud hardware, ids, prices, and timestamps are illustrative values only:

```json
{
  "schema_version": 1,
  "run_id": "<run id>",
  "ticket_id": "<ticket id>",
  "purpose": "inference",
  "provider": "cloud",
  "cloud_backend": "vastai",
  "instance_id": "12345",
  "auto_release": true,
  "host": "ssh.example",
  "ssh": {
    "host": "ssh.example",
    "port": 2222,
    "user": "root",
    "key_path": "/root/.ssh/id_ed25519",
    "password_path": ""
  },
  "gpu": {
    "has_gpu": true,
    "gpu_count": 1,
    "gpu_name": "NVIDIA RTX 4090",
    "vram_gb": 24,
    "vram_mb": 24564,
    "devices": [
      {"index": 0, "name": "NVIDIA RTX 4090", "vram_mb": 24564}
    ]
  },
  "cuda": {
    "driver_version": "550.54.15",
    "cuda_version": "12.4",
    "recommended_torch_index": "cu121"
  },
  "cost": {"dph_total": 0.42},
  "resource_plan": {
    "purpose": "inference",
    "required_working_set_gib": 40,
    "host_memory_components_gib": {
      "model_and_runtime": 24,
      "data_and_checkpoint": 16
    },
    "host_memory_formula_gib": 64,
    "site_min_ram_gib": 0,
    "num_gpus": 1,
    "min_vram_gb": 24,
    "min_ram_gb": 64,
    "min_cpus": 8,
    "time_limit_hours": 0,
    "cloud_backend": "vastai",
    "gpu_type": "",
    "docker_image": "nvidia/cuda:12.1.1-devel-ubuntu22.04",
    "disk_gb": 100,
    "slurm_partition": "",
    "slurm_account": "",
    "slurm_qos": "",
    "rationale": "Resolved from model, data profile, run horizon, and budget."
  },
  "probe_source": "remote_nvidia-smi",
  "probed_at": "2026-08-14T18:32:00Z"
}
```

For `cluster`, top-level `ssh` is the login node and `cluster` is the Slurm
submission route. It contains no allocation yet:

```json
{
  "cluster": {
    "jobid": "",
    "node": "",
    "requested_gpus": 1,
    "partition": "<actual partition>",
    "account": "",
    "qos": "",
    "container_image": "",
    "env_setup": "source ~/miniconda3/bin/activate zevo",
    "workdir": "/remote/root/<run-id>",
    "hf_cache": "/remote/root/hf_cache"
  }
}
```

For cluster also set `instance_id=""`, `instance=null`, `gpu=null`, `cuda=null`,
`auto_release=false`, and `probe_source="ssh-environment"`. Preserve the typed
`container_image`; when it is empty, a matching site Skill may supply a verified
default to the later stage.

For `instance`, set `cluster=null` and add a direct route:

```json
{
  "instance": {
    "env_setup": "source ~/miniconda3/bin/activate zevo",
    "workdir": "/remote/root/<run-id>",
    "hf_cache": "/remote/root/hf_cache",
    "visible_devices": "0,1"
  }
}
```

Its `instance_id` is the stable SSH connection identity and `host` is the fixed
GPU host.

Never record secret values. Exactly one of `ssh.key_path` and
`ssh.password_path` is non-empty. Both are routing metadata; key/password
contents are not. For a password route, invoke SSH/SCP through
`sshpass -f <password_path>` without reading, printing, or copying the file.

## Probe and validation

Probe through the same route downstream will use:

- cloud: direct SSH;
- cluster: login-node SSH only, validating scheduler/storage/environment access;
- instance: direct SSH to the fixed GPU host, selecting only leased indices.

For cloud/instance, use a finite command deadline and query index, name, memory,
and driver version from `nvidia-smi`. For instance, `nvidia-smi` itself is not filtered by
`CUDA_VISIBLE_DEVICES`; pass the granted indices explicitly with `-i` so the
probe cannot report GPUs leased to another run.

The `CUDA Version` shown by `nvidia-smi` is the maximum CUDA level supported by
the installed driver, not proof of a locally installed toolkit or PyTorch
runtime. Record it as reported. `recommended_torch_index` is advisory only:
choose a wheel tag no newer than driver support when the mapping is known;
otherwise leave it empty. Downstream must still validate its actual environment.

Success requires all of the following:

- SSH and, for cluster, scheduler-route validation succeeded;
- measured or cluster-requested device count equals `resource_plan.num_gpus`
  and does not exceed a positive Run `num_gpus` maximum;
- every assigned device meets `resource_plan.min_vram_gb`;
- `instance` count and indices exactly equal the lease grant;
- SSH, job, workdir, ownership, backend, and cost fields are internally
  consistent; and
- the reopened file matches the final result.

Use the minimum assigned VRAM for top-level `vram_gb`. Preserve all devices in
`gpu.devices`; do not hide heterogeneous or undersized cards behind the first
row. A GPU ticket that probes no usable GPU is failure, not a successful CPU
contract.

## Cloud provider

For provision, select `cloud_backend` once from the available, feasible
backends (respecting a valid deployment preference), then keep it fixed through
acquisition and cleanup. Use only the shipped client:

```python
from zevo.providers.vastai import VastAIProvider
from zevo.providers.lambda_labs import LambdaCloudProvider
```

Do not call provider REST endpoints directly. Both clients expose
`search_gpus`, `create_instance`, `list_instances`, `wait_for_ready`,
`get_ssh_details`, `wait_for_ssh`, and `destroy_instance`.

Before creating anything, query Zevo bookkeeping for an open resource belonging
to this same run/ticket. Reuse it only if the provider still reports it active
and the route probes successfully. An arbitrary active account instance belongs
to somebody else; do not reuse it. If the provider safety cap rejects a new
instance, fail with that evidence.

Search once with `resource_plan.num_gpus` and the remaining resolved resource plan. Vast.ai applies
host RAM/CPU filters and uses the supplied image/disk; Lambda offers fixed
instance types and ignores those fields. Require a non-empty offer list. Post a
cost-transparency message naming backend, offer/type, region when applicable,
GPU count/type/VRAM, and `dph_total` before creation.

Create only the selected cheapest valid offer:

- Vast.ai: `create_instance(offer_id, resource_plan.docker_image,
  resource_plan.disk_gb)`; SSH is normally
  `root` on the returned port.
- Lambda: `create_instance(offer_id, region, ssh_key_path, name)`; SSH is
  `ubuntu` on port 22. The client resolves/registers the public key.

Immediately POST `infra_instances_endpoint` with the exact
`infra_instance_create_schema`: `provider="cloud"`, the provider's
`instance_id`, `status="provisioning"`, run/ticket ids, `dph`, and `meta`
including `backend` and `auto_release`. `price` and `metadata` are invalid.
Capture the returned row `id`
(the bookkeeping row's own id — later PATCHes address that, not `instance_id`).
If this bookkeeping write fails, destroy the new instance and fail.

Use the client's asynchronous readiness and SSH waits; a provider API-ready
state is not SSH-ready. After a successful hardware probe, PATCH the row to
`ready` with measured hardware, SSH route, and price, then write the device
contract.

On every post-create exception, call `destroy_instance` once and check its
boolean result. Never rent a replacement in the same ticket. If destruction
fails, leave the bookkeeping row active and return failure with the real
instance id/backend/price so leak detection remains actionable.

## Slurm site skills

Before any site command, inspect the installed Infrastructure Skills and load at
most one whose trigger matches `ssh_host` or the explicitly named site. The
site Skill may strengthen storage, partition, account, QOS, and probe rules; it
cannot change provider ownership or weaken this contract. If no Skill matches,
use the generic procedure below. If more than one could match, fail visibly.

## Cluster provider

Use the typed `ssh_*`, `slurm_*`, `env_setup`, and `remote_dir`; do not reread a
different provider's environment values. Require a non-empty absolute
`remote_dir`. Resolve the remote user's canonical `$HOME` and fail if the root
is HOME or one of its descendants. Never synthesize or substitute a storage
root. Use `<remote-root>/<run_id>` or, for standalone work,
`<remote-root>/standalone/<ticket_id>` as `cluster.workdir`; keep
the shared cache outside it. The default HF cache is
`<remote-root>/hf_cache`; a matched site Skill may override that location.

Copy the configured authentication route into `device_info.ssh` exactly:
`ssh_key_path` becomes `key_path` and `ssh_password_path` becomes
`password_path`. Exactly one is non-empty. Use that same method for every
login-node command and never fall back to another mounted credential.

Inspect the configured partition and account/QOS eligibility with read-only
Slurm commands. Live scheduler output is authoritative. Verify that `sbatch`,
`squeue`, `sacct`, the configured environment activation, and writable remote
Run/cache directories are available without submitting any job. A login-node
environment check may import lightweight dependencies, but it must not load a
model or claim compute-node GPU/CUDA facts.

Resolve a bounded resource plan for the downstream stage. Record the exact
partition/account/QOS, CPU, RAM, requested GPU type/count, and walltime that the
stage must render as `#SBATCH` directives. If a requested GPU mapping or account
association cannot be established from live read-only state, fail visibly.

Write cluster `device_info.json` as an access contract:

- `provider="cluster"`, empty `instance_id`, and `auto_release=false`;
- `ssh` copied exactly from the typed route;
- `cluster.jobid=""` and `cluster.node=""`, with resolved scheduler hints,
  `requested_gpus=resource_plan.num_gpus` (the cluster-wide total),
  `cluster.nodes=resource_plan.nodes`, configured container image,
  environment setup, Run workdir, and shared HF cache. For a multi-node plan the
  downstream stage renders `#SBATCH --nodes`, `--ntasks-per-node`, and
  `--gpus-per-node` (`requested_gpus // nodes`) and launches with
  torchrun/deepspeed rendezvous; `requested_gpus` must be an exact multiple of
  `cluster.nodes`;
- `gpu=null`, `cuda=null`, `probe_source="ssh-environment"`, and zero cost.

Never call `sbatch`, `salloc`, or `srun` in cluster provision. The consuming
Train or Inference ticket writes `train.sbatch` or `predict.sbatch`, submits and
registers that exact job, then ends its activation. The deterministic backend
Scheduler owns low-frequency monitoring, queue-deadline enforcement,
cancellation, terminal bookkeeping, and the wake that lets the same Ticket
collect results.

## Instance provider

This mode uses one fixed GPU server reachable directly through the typed SSH
connection. It requires a non-empty `run_id`; it has no Slurm allocation,
partition, QOS, queue, walltime, or JOBID.

Reuse the Run's device lease across Train/Inference stages while the host and
exact indices still probe healthy. Query physical index, UUID, GPU utilization,
used/total memory, and active compute processes directly over SSH. An index is
idle only when GPU utilization is zero, no compute process owns it, and memory
use is no more than the node's driver-only baseline. Unknown or stale probe
state is not idle; unknown VRAM cannot satisfy a hard minimum.

Build exactly one body from `gpu_lease_request_schema`: `run_id`, `ticket_id`,
`num_gpus=resource_plan.num_gpus`, `min_vram_gb`, and one `allocations` entry.
That candidate uses `jobid=<stable SSH connection identity>`, `node=<SSH host>`,
and the measured `gpu_count`, `idle_gpu_indices`, `gpu_name`, and `vram_gb`.
The legacy key name `jobid` is only the lease-ledger host key here; it is not a
Slurm id. Validate locally and POST once. Never discover a mutation schema by
sending guessed bodies.

Use the returned indices and `visible_devices` exactly. Immediately re-probe
only those physical indices over direct SSH to close the race between probe and
lease creation. If one became busy, delete this Run's lease, exclude that slice,
refresh once, and retry with a changed candidate set. A final 409 or exhausted
bounded re-probe is failure; never run on a busy card.

Write `provider="instance"`, the stable SSH connection identity as
`instance_id`, `auto_release=false`, `cluster=null`, and the non-empty
`instance.visible_devices`. Do not register an `InfraInstance` lifecycle row:
the machine was neither created nor scheduled by Zevo. The GPU lease ledger is
the arbitration and usage record.

## Release operation

Require a provider and exact `instance_id`. Return success only after the
release action is verified:

- `cloud`: require `cloud_backend`, invoke
  `destroy_instance(instance_id)` on that backend, and verify termination.
- `cluster`: no Infrastructure release ticket is valid because provision holds
  no job. Stage cancellation and final cleanup use the exact Train/Inference
  bookkeeping JOBID.
- `instance`: terminate only processes carrying this Run's exact
  `ZEVO_TICKET_ID` values, then DELETE `/api/gpu/leases?run_id=<run_id>` and
  verify no live lease remains. Never reboot or stop the host.

Release is idempotent: a provider-confirmed absent cloud instance or an
already-returned instance lease is success with an
“already released” note. A failed destroy call alone is not proof of absence;
query provider state before making that claim.

After the provider action, mark matching open Infrastructure bookkeeping rows
released with a reason. Remove only the run-scoped remote workdir after proving
the path contains the exact run id and is below the configured remote root;
never delete the shared HF cache.

A release result has `operation="release"`, the requested provider and handle,
`auto_release=false`, empty `device_info_path`, and a specific note. Failure
preserves any discovered target handle and backend; it does not invent device
measurements.

## Final output

The final stdout line is exactly one `InfraResult`:

| Field | Rule |
|---|---|
| `status`, `ticket_id`, `operation` | Echo the id and exact `provision` or `release` operation. |
| `device_info_path` | Valid absolute local artifact only after successful provision; empty otherwise. |
| `provider`, `instance_id`, `cloud_backend`, `auto_release` | Empty/false on usable provision because `device_info.json` owns those facts. On release or failure, report only lifecycle identity needed for cleanup/audit and preserve acquired handles. |
| `error_message`, `notes` | Empty error on success; otherwise specific failure plus acquisition/probe provenance. |

On failure, use `status="failed"`, an empty artifact path, and a specific error.
Preserve any acquired `instance_id` and `cloud_backend`. `degraded` is valid only
for an explicitly accepted non-strict shortfall and otherwise follows provision
success invariants; the exact deviation belongs in `notes` and the artifact.

## Phases

Emit only phases that occur:

```text
__PHASE__:<ticket_id>:resolving_resource@<unix-time>
__PHASE__:<ticket_id>:acquiring_resource@<unix-time>
__PHASE__:<ticket_id>:waiting_ready@<unix-time>
__PHASE__:<ticket_id>:probing_gpu@<unix-time>
__PHASE__:<ticket_id>:writing_output@<unix-time>
__PHASE__:<ticket_id>:releasing_resource@<unix-time>
```

Post `Starting:` and `Done:` provenance as required by the shared contract. For
cloud, add the pre-create cost message. Print nothing after the final JSON.
