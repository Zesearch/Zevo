Infrastructure is a sibling dependency of Train, Inference, and Registry. It
does not consume a Data artifact. The runner resolves configuration and hands
you these fields directly:

| Field | Contract |
|---|---|
| `ticket_id` | Opaque work-order id. Echo it; do not parse it. |
| `run_id` | Parent run stamped by the runner; used for remote scope, leases, and bookkeeping. |
| `provider` | `cloud`, `cluster`, or `instance`; do not substitute another provider. |
| `cloud_backend` | Release-only lifecycle identity; empty during provision and for non-cloud providers. |
| `available_cloud_backends`, `preferred_cloud_backend` | Credentialed cloud capabilities plus an optional deployment preference. Infrastructure selects and reports the provision backend. |
| `release` | False provisions and probes. True releases the supplied scope. |
| `purpose` | Exact immediate `train` or `inference` consumer whose resource plan is being resolved. |
| `num_gpus` | User-fixed maximum GPU count stamped from Run. Zero means unlimited. Infrastructure selects a positive `resource_plan.num_gpus`, bounded by this value when it is positive. |
| `resource_context` | Engine-owned task, pin, data-profile, iteration-horizon, runtime, and live-budget evidence. Derive all other resource values from it. |
| `host_memory_planning_contract` | Engine-owned host-RAM formula: estimate the real stage peak, retain both 1.5x and +16 GiB headroom, round up to 8 GiB, and honor the 16-GiB inference or 32-GiB Train floor. `resource_plan` must expose the purpose, peak estimate, component evidence, exact formula result, optional site minimum, and final request; schema validation recomputes the result. |
| `gpu_lease_endpoint`, `openapi_endpoint` | Exact versioned API paths. The OpenAPI path is read-only; do not try `/openapi.json`. |
| `gpu_lease_request_schema`, `gpu_lease_grant_schema` | Exact instance-mode POST/response key and type authorities. The request's allocation list key is `allocations`. |
| `infra_instances_endpoint` | Exact bookkeeping collection path. Append only the returned row `id` for PATCH. |
| `infra_instance_create_schema`, `infra_instance_patch_schema`, `infra_instance_response_schema` | Exact POST/PATCH/response authorities. Use `dph` and `meta`; `price` and `metadata` do not exist. |
| `device_info_schema` | Exact machine-readable key/type authority for successful `device_info.json`; never infer its shape from an example or prior Run. |
| `device_info_validation_command` | Side-effect-free validator for the final artifact. Replace only its path placeholder; Run/Ticket/provider/GPU-maximum/purpose values are already inserted. |
| `ssh_host`, `ssh_port`, `ssh_user`, `ssh_key_path`, `ssh_password_path` | Login-node route for cluster/instance. Exactly one credential path is non-empty; never read or print its contents. |
| `slurm_partition`, `slurm_account`, `slurm_qos` | Deployment/site hints rather than Ticket resource requests. Site skills validate live availability and account/QOS pairing. |
| `env_setup`, `remote_dir` | Cluster/instance environment activation and remote root. Use configured non-empty values; do not append an extra `zevo`. |
| `container_image` | Optional cluster-specific Pyxis/Enroot image. Preserve it in `device_info.cluster`; a matching site Skill may provide a documented default only when this is empty. |
| `instance_id` | Exact cloud instance to release. Empty during a new provision and for cluster/instance access. |
| `auto_release` | Normal-finalization policy. False for release, cluster access, and `instance`; cloud defaults true. |
| `work_dir` | Persistent local ticket directory for `device_info.json`. |
| `customization` | Honor `customization` without weakening ownership, probe, or cleanup checks. |

Validate the lease body locally against `gpu_lease_request_schema` before its
single real POST. Never discover a mutation schema by sending guessed bodies:
`candidates`, `candidate_allocations`, and other aliases are invalid, and a
schema probe can acquire or partially persist a real resource.

Reuse the one successful Infra reference throughout a run. The orchestrator
must not create release tickets in the normal forward DAG; lifecycle code owns
terminal cleanup. Follow `platform.md` and emit exactly one `InfraResult`.
After usable provision, report only `device_info_path`; the reopened, validated
artifact is the sole authority for the resource plan, device measurements,
route, ownership, lifecycle policy, and price. Do not restate the resolved
resource plan (GPU count, VRAM, RAM, CPUs, cloud backend, etc.) in `notes`,
`summary`, or `error_message`: those fields are for concise execution notes and
failure detail only. The engine reads the plan back from `device_info.json` and
records it as the Ticket's structured `resource_plan` artifact meta, and writes
its own one-line provision summary; a succeeded infra Ticket must have an empty
`error_message`. For `cluster`, it records only
the verified route and planned scheduler resources; allocation id, node, GPU,
and CUDA are intentionally absent. Train and Inference each write and submit
their own finite `.sbatch` artifact, which releases its allocation on exit.
