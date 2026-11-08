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

Registry is the once-per-Run final stage after optimization stops. Its inputs are
already resolved; use the paths exactly as supplied.

| Field | Contract |
|---|---|
| `ticket_id` | This work order's opaque id. Echo it; do not parse it. |
| `run_id` | Required producing Run id stamped by the runner. It deterministically owns `M-<run8>`; never infer it from the Ticket id. |
| `expected_version_tag` | Engine-computed stable tag. Copy it exactly into the YAML key; do not derive or normalize it. |
| `registry_entry_schema` | Exact machine-readable key/type authority for `registry.yaml.models[expected_version_tag]`; never infer keys from an example or older registry. |
| `registry_validation_command` | Side-effect-free validator for the committed entry. Replace only its registry-path placeholder; Run/tag values are already inserted. |
| `checkpoint_path` | Exact Train checkpoint directory. |
| `checkpoint_is_remote` | When true, access and copy the checkpoint through `device_info_path`; when false, require a local directory. |
| `device_info_path` | Local `device_info.json` containing the remote access contract. Required for a remote checkpoint. |
| `train_config_path` | Exact `train_config.yaml` produced with this checkpoint; preserve it as configuration provenance. |
| `metrics_path` | Exact Evaluation `metrics.json`. Required and authoritative. |
| `base_model` | Base model derived by the runner from the validated Train configuration; record verbatim. |
| `training_method` | Method derived by the runner from the validated Train configuration; record verbatim. |
| `dataset_source` | Runner-resolved, host-visible original dataset/source lineage. Copy it verbatim; do not convert it or substitute a curated path. |
| `task_objective` | Original run objective; record verbatim. |
| `metric` | Required Validation metric name used to select this champion; persist it beside the score. |
| `metric_direction` | Task direction the engine already used to select this champion; record it verbatim. |
| `registry_path` | Exact Ticket-local `<work_dir>/registry.yaml` path. Write exactly one selected-champion snapshot; it is not a shared registry. |
| `work_dir` | Persistent ticket directory for `register.py`. |
| `customization` | Honor `customization` as defined in the shared contract without weakening provenance or safety checks. |

The canonical pipeline reaches Registry once, only after all intended
iterations and their Journals are complete. The API rejects any iteration other
than the engine-selected Validation champion.
Missing or invalid metrics therefore means the handoff is broken: return
`failed` instead of creating an unscored row.

Follow `platform.md`, then return the same `RegisterResult` in every mode. The
committed manifest must name this Ticket and supplied champion iteration; there
is no per-iteration incumbent/challenger comparison.
