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
