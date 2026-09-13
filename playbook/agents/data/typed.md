## Input

| Field | Contract |
|---|---|
| `operation` | `prepare_run_data`, private `prepare_holdout_data`, or Auto-mode `scope_problem`. |
| `run_id` / `ticket_id` | Echo the Ticket id in the result. |
| `task_objective` | `scope_problem` only: the user's objective. Empty otherwise. |
| `test_query` | `scope_problem` only: optional guidance for selecting or creating the Test population and format. Empty otherwise. |
| `constraints` | `scope_problem` only: optional user constraints. Empty otherwise. |
| `scoping_result_schema` | `scope_problem` only: exact JSON Schema of `scoping_result.json` (`ScopingResult`). |
| `scoping_result_validation_command` | `scope_problem` only: exact validator to run on the written file before success. |
| `dataset` | Exact user path/Hugging Face id; empty only when acquiring. |
| `dataset_split` / `dataset_config` | Exact supplied Hub slice/subset. |
| `data_query` | Natural-language discovery/preparation guidance used with the task objective when dataset is empty. |
| `dataset_source` | Provenance label. Resolve its exact realized identity into `data_recipe.json`; do not repeat it in `DataResult`. |
| `training_method` | Required semantic-shape selector for Run data; never a hyperparameter bundle. Empty for private held-out stripping. |
| `recipe_intent` | Exact selection/filter/sampling/weighting/transformation/mapping/seed request for this data version. Empty fields preserve baseline behavior. |
| `branch_transition` | Engine-validated exhausted-branch record. It may contain scalar score/plateau evidence, never Validation records, examples, fields, or statistics. |
| `data_intent_signature` | Engine-computed identity of source, method, recipe intent, and Data pins/suggestions. Use it to verify the work order; do not echo it in `DataResult`. |
| `expected_source_fingerprint` | For a local source, the engine-computed bare lowercase 64-character SHA-256. Copy it exactly into `data_recipe.source_fingerprint`; an empty value is permitted only for a remote/acquired source. |
| `expected_source_identity` | Engine-canonical source lineage. Copy it byte-for-byte into `data_recipe.source_identity`; never convert between container, host, relative, or absolute forms. |
| `data_recipe_schema` | Exact artifact-level realized-recipe schema for `data_recipe.json`. |
| `data_recipe_validation_command` | Exact side-effect-free recipe/artifact/source validator. Replace only the placeholders present; the engine already inserts a shell-quoted local source path. Use its printed signature. |
| `artifacts_validation_command` | Exact side-effect-free validator for the training artifact only. |
| `device_info_path` | The purpose=train remote route shared with Train. HF download, detect, analysis, transformation, and prepared rows stay there. |
| `remote_data_helper_path` | Small system helper to upload; it receipts/finalizes remote data without returning rows. |
| `remote_dataset_spec_schema` / `remote_dataset_spec_validation_command` | Exact remote HF source contract and validator. Pin an immutable revision. |
| `remote_hf_cache_path` | Remote cache for reusable raw Hugging Face downloads only. |
| `remote_data_output_dir` | Durable Run/data-intent directory for the prepared dataset and profile; validated retries share it. |
| `remote_preparation_receipt_path` | Exact compact receipt path. Validate it with the system helper before reuse and record it immediately after preparation. |
| `remote_required_environment` / `remote_timeout_seconds` | Exact cancellation markers and foreground execution deadline for remote Data. |
| `secret_environment_names` | Secret names that may be forwarded without printing values, currently `HF_TOKEN`. |
| Validation/scoring fields | Empty for `prepare_run_data` by contract. They are populated only for the private `prepare_holdout_data` operation, which cannot receive training-source fields. |
| `configuration_suggestions` | Advisory Data-only acquisition/curation guidance; no Train/Inference hyperparameters. |
| `configuration_pins` | Binding Data-only acquisition/curation values; no Train/Inference hyperparameters. |
| `work_dir` | Persistent output directory. |

## Output

For a Hugging Face source, successful `prepare_run_data` requires absolute
`remote_dataset_path`, `remote_data_profile_path`, local
`remote_dataset_spec_path`, `data_recipe_path`, and `prepare_script_path`; keep
`training_dataset_path` empty. The dataset/profile stay remote. Local-source
legacy preparation may still return `training_dataset_path`. Return all
Validation/scoring paths and answer fields empty; the engine attaches its frozen
scoring package only after validating this result.

Successful `prepare_holdout_data` requires only questions-only
`scoring_public_path` and preparation evidence; training/Validation paths must
remain empty.

Successful `scope_problem` (Auto mode) requires only `scoping_result_path`,
the absolute path of a `scoping_result.json` that passes
`scoping_result_validation_command`; every other artifact field stays empty.
The engine settles the Run's scoring contract from that file. Training data,
model, and method pins/queries are deliberately absent from this work order.

Run `artifacts_validation_command` after every output is final and before
returning success. Do not write a parallel ad-hoc Python/Bash validator or add
content-quality assertions that are absent from this command.

Return measured `n_rows_in`, `n_rows_out`, artifact paths, and exactly one
`DataResult`. `data_recipe.json` is the sole authority for source provenance,
ordered methods, and the realized recipe; the runner verifies it and derives
`data_signature`. For remote Data, the compact receipt binds that signature to
the exact final bytes and measured row count without copying rows back.

`data_recipe.dataset_name` is required display provenance. Use the stable
human-readable source name: the Hub repository for an acquired dataset or the
uploaded/local filename for a supplied dataset. Filtering, sampling, or
reformatting keeps that source name; describe the revision under `subset`,
`filters`, and `transformations` rather than hiding the name in `audit_steps`.
