## Optional managed compute

Registry is not restricted to CPU operations. Use GPU compute only when the
assigned preservation/verification work needs it; never modify or reselect the
measured champion. Metadata, file transfer and publication preparation do not
require a GPU job.

When compute is needed, use the supplied enabled `slurm_job` contract. Prepare
reproducible `register.py` and the exact assigned launch script; stage them under
`remote_work_dir`. For transport `slurm`, include the assigned job name, GPU
shape, stdout/stderr paths, lifecycle and runtime prologues. For `direct`, prepare
a foreground shell command on the assigned host, with no sbatch or srun.
Launch GPU work through the supplied `zevo_gpu_runtime.py`, staging that helper
beside the worker. Return `deferred` with `slurm_script_path` and
`register_script_path`; leave `registry_path` empty until collection validates
the complete output. The engine submits, monitors and wakes this same Ticket.
Never run GPU work in the agent, on the login node, or submit a job yourself.
If no enabled compute contract is supplied, report the missing compute route
instead of inventing one.

During `collect`, preserve the registered script, collect and validate the exact
execution's output, then produce the normal local registry manifest. Do not
launch another workload during collection. For CPU/file-only work with no
submitted execution, leave `slurm_script_path` empty and complete normally.
Use ticket-stable durable, atomic receipts for resumable preparation; verify
input checkpoint, operation, configuration, implementation and output checksums
before reuse. Repeat only work without a valid receipt. Publication remains
engine-owned and uses the existing verified save receipt for idempotency.

## Normal execution-window expiry

The engine distinguishes a registered workload's TIMEOUT/DEADLINE or the
system helper's reserved exit 75 from execution errors. Window expiry continues
this same Ticket without consuming a repair attempt or changing the experiment.
Cluster continuation waits for the old allocation's confirmed release before
requesting a new window. A process window on a healthy cloud/instance host
reuses that host. Do not return a failed result or request resource replacement
solely because the execution window expired.

Keep the input, recipe, configuration and workload identity unchanged. Restore
only checksummed, identity-matching batch/benchmark receipts from the stable
Ticket resume directory; repeat unfinished work. A missing or invalid receipt
is not completed work. Never report partial output as success. Actual execution
or artifact errors still use the normal bounded repair policy.

## Run-selected model destination

When `model_save_policy` is nonempty, prepare your artifacts FIRST. The engine
validates your result and then preserves the selected model at the Run's
chosen destination. A preservation failure becomes a normal failed activation:
you receive its diagnostic in the next auto-repair activation, fix the relevant
artifact, and return a corrected result. There is no separate upload retry loop
and you must not independently publish or handle upload credentials.

Prepare `registry.yaml` with the exact assigned `checkpoint_path` as
`model_path`, `storage=model_save_policy.weights`, and an empty `revision`.
This is a preparation manifest, not proof that publication succeeded. After a
verified save, the engine atomically replaces only these destination fields
with its verified receipt before registering the model. Preserve metrics and
provenance exactly. If supplied `saved_model_path` is nonempty, a previous
attempt already saved the model: reuse that path and `saved_model_revision`.

On repair, inspect the actual preservation error and the assigned artifacts.
For a model-card error, read the checkpoint README.md and write a corrected
copy inside `work_dir`, then return its absolute `model_card_path`. Use the
supplied provenance to resolve identifiers; do not guess a Hub ID from a
folder name. Preserve the card's other accurate information. The engine applies
this card only to an upload copy. Never mutate the measured checkpoint,
weights, tokenizer, or training configuration. Keep the corrected card path
in subsequent repair results. Errors requiring a different resource, input or
fixed configuration follow the shared failure-recovery rules.

Write the preparation and any repair in reproducible `register.py`. Report
`succeeded` when your artifacts are ready for backend validation; the engine
owns the final stage outcome, including upload failures. A saved HF/SSH URL
is valid and must not be probed with local filesystem APIs. These destination
rules supersede local-copy requirements below. Empty policy retains the
legacy local-copy contract for already launched Runs.

## Position and artifacts

```text
Train checkpoint ───────────┐
Evaluation metrics.json ────┼─> Registry ─> registry.yaml + retained local model
Infrastructure access ──────┘
```

Registry runs once and records the engine-selected Validation champion under
the Run's stable `M-<run_id[:8]>` identity. Candidate history already lives in
Score Events and Tickets. Training's remote directory is transient, so this
final remote checkpoint must be copied before infrastructure is released.

Use `customization.output_dir` as the model destination root when non-empty.
Otherwise use `<parent-of-work_dir>/models/<version_tag>`. Always write
`register.py` to `work_dir` and update the supplied `registry_path`.

## Required order of work

1. Resolve the inputs and output paths; create `work_dir` if needed.
2. Validate metrics and checkpoint provenance before changing the registry.
3. Copy the supplied `expected_version_tag` exactly; do not compare or select
   another iteration.
4. Stage and validate the supplied champion checkpoint.
5. Atomically write exactly that champion into this Ticket's manifest.
6. Re-read the committed snapshot.
7. Run the supplied `registry_validation_command`, replacing only
   `<absolute-registry-yaml-path>`, then emit one `RegisterResult`.

Emit these phase markers as work begins:

```text
__PHASE__:<ticket_id>:reading_metrics@<unix-time>
__PHASE__:<ticket_id>:persisting_model@<unix-time>
__PHASE__:<ticket_id>:writing_registry@<unix-time>
__PHASE__:<ticket_id>:validating_result@<unix-time>
```

## Metrics validation

Open `metrics_path` as UTF-8 JSON and require a non-empty top-level object.
`score` must be a finite `int` or `float`, not `bool`. Preserve
the complete object under `eval`; do not drop diagnostic or semantic metric
keys. Registry never invokes a scorer and never trusts a score copied into the
ticket text or model directory.

Failure to read a valid metrics file is ticket failure. Return no score field in
the Result; `metrics.json` and the selected manifest entry are the only score
authorities.

## Checkpoint validation

When `checkpoint_is_remote=false`, require `checkpoint_path` to be a readable,
non-empty local directory. When true:

- require readable `device_info_path` with non-empty `ssh.host`, `ssh.port`,
  `ssh.user`, and exactly one of `ssh.key_path` or `ssh.password_path`;
- inspect the exact remote directory with SSH before copying;
- copy the selected directory with
  `python -m zevo.engine.remote_transfer download --recursive` through the selected connection;
- copy only that checkpoint directory, never its parent run directory.

Validate the staged or local model according to its artifact shape:

- adapter: `adapter_config.json` and non-empty
  `adapter_model.safetensors` or `adapter_model.bin`;
- full model: `config.json` and non-empty weight data (`*.safetensors` or
  `pytorch_model*.bin`); when an index JSON exists, require every shard it names.

Reject an empty directory, broken symlink, missing referenced shard, or an
artifact inconsistent with `training_method`. Do not validate only the LoRA filename;
full fine-tunes are valid outputs too.

## Stable tag and idempotency

Require a non-empty `run_id` and copy the engine-owned tag without accepting an
override:

```text
version_tag = expected_version_tag
```

The API/database layer owns global tag uniqueness and proves that this Ticket's
lineage is the Validation champion. Do not search a global YAML file for a
collision, prior champion, or competing score. If this Ticket's own manifest
already contains the same `ticket_id`, iteration, provenance,
metrics, and committed checkpoint, verify and reuse it as an idempotent retry.
If that Ticket id matches but its facts differ, fail.

## Deterministic finalization

Write the supplied checkpoint's provenance, complete `eval`, current
`ticket_id`, selected iteration, a new `registered_at`, and
`retention: retained`. The selected manifest entry is the sole authority for
tag, model path, score, and retention; the Result does not repeat them.

## Safe model persistence

Copy into a unique temporary sibling such as
`.<version_tag>.partial-<uuid>` beneath the resolved models root. The stable
destination is `<models-root>/M-<run8>`. Resolve paths before mutation and
require staging, destination, and any backup to be direct children of that
exact root. Never use an unresolved variable, glob, or broad recursive delete.

After the staged copy passes checkpoint validation:

- if the destination is absent, atomically rename staging to it;
- if it contains the exact artifact for an idempotent retry, reuse it;
- otherwise fail: a once-per-Run Registry destination must not already contain
  a different model.

Commit and verify YAML after the atomic model move. If commit or verification
fails, remove only the exact uncommitted destination for this Ticket. Never
scan and delete “all other” model directories.

## Registry file and path rules

`registry_path` is the exact Ticket-local `<work_dir>/registry.yaml`. Do not
redirect it and do not read or update `data/runs/registry.yaml` or any other
shared file. Write exactly this shape:

```yaml
models:
  <expected_version_tag>: <selected RegistryEntry>
```

Reject any extra model entry. Write the complete document to a temporary file
in the same Ticket directory, flush and `fsync`, then `os.replace` it. A lock is
unnecessary because no other Ticket writes this path; global champion
serialization and prefix-collision protection belong to the database mirror.

Write paths in YAML as host-visible paths. In the standard deployment the
single conversion rule is:

```text
/app/<relative-path>  ->  <relative-path>
```

Thus `/app/data/runs/<run>/models/<tag>` becomes
`data/runs/<run>/models/<tag>`. Do not invent the obsolete `runs/...` or
`workspace/...` aliases. For a non-standard absolute output outside `/app`,
record the explicit absolute path rather than pretending it is repo-relative.

When reopening a stored repo-relative path, resolve it beneath `/app` and reject
`..` or any result that escapes the expected models root. Never pass the
display-form YAML string directly to a deletion command.

The supplied `registry_entry_schema` is the sole key/type authority for the
successful champion entry under `models[M-<run8>]`; the table below explains
semantics but does not add or rename keys:

| Key | Required value |
|---|---|
| `run_id`, `ticket_id`, `iteration` | Current final Registry Ticket and engine-selected champion iteration. |
| `base_model`, `training_method` | Actual Train identity; never Registry defaults. |
| `dataset_source`, `task_objective` | Supplied lineage and user objective. `dataset_source` is already host-visible; copy it verbatim. |
| `metric`, `metric_direction` | Validation metric name plus `max` or `min`, copied from the ticket and used for retention. |
| `model_path` | Selected champion's safe repo-relative display path, or explicitly allowed non-standard absolute path. It is never empty in the manifest. |
| `eval` | Complete validated metrics object, including finite top-level `score`. |
| `retention` | Always `retained`. |
| `registered_at` | New UTC timestamp for this final commit. |

After the atomic write, reopen the file and prove that it has only the expected
tag and that the selected entry's provenance, score, metric, target direction,
retention state, and model path equal the intended values. The runner mirrors
this snapshot into `registry_models` immediately after your successful result;
do not write Postgres yourself.

## Reproducible helper

Write the complete implementation to `<work_dir>/register.py` and execute that
file. It owns parsing, champion validation, safe copy, atomic
snapshot commit, post-write verification, and concise diagnostics; it does not
use a shared-file lock. Do not install PyYAML during the ticket and do not leave
the real mutation only in ad-hoc shell history. The helper may print diagnostic
lines before completion. Your agent's final
stdout object remains the only `RegisterResult`.

## Final output

Return one minimal `RegisterResult` after validating the committed state:

| Field | Rule |
|---|---|
| `status`, `ticket_id` | `succeeded` only after commit and reread; echo the whole Ticket id. |
| `registry_path` | Valid absolute local path to the committed YAML. |
| `register_script_path` | Absolute helper path when produced; absence is advisory, not proof that the committed registry/model is invalid. |
| `error_message`, `notes` | Empty error on success; notes state retention/idempotency and cleanup warnings. |

On failure use `status="failed"`, empty artifact/helper paths, and a specific
`error_message`. Never report failure after a registry commit merely because
cleanup of a rollback directory failed; report that cleanup warning in `notes`.

`status="succeeded"` means the Ticket-local selected-champion snapshot was
committed and re-read. The runner independently reopens the manifest and
verifies the selected local model and its exact lineage.
Print nothing after the object.
