## Work order

| Field | Contract |
|---|---|
| `operation` | Always `run_inference`. |
| `benchmark_id` | Engine-assigned identity of the primary benchmark. Copy it unchanged into the suite manifest and progress markers; never derive it from the display name. |
| `configuration_mode` | `select` only for baseline Validation; otherwise `reuse`. |
| `iteration` | The initial model Baseline is 0. A later Zevo-selected model may establish its Baseline at the next unused Train iteration; trained checkpoints are 1..N. |
| `model_source` | `base_model` or `checkpoint`. |
| `base_model` | Exact baseline model identity. |
| `checkpoint_path` | Empty for baseline; required for trained checkpoints. |
| `branch_transition` | Engine-validated exhausted-branch record; only a later Base-model baseline uses `level=base_model`. |
| `scoring_set` | Questions-only rows; preserve row order. |
| `sample_submission` | Exact output-column contract. |
| `inference_data_profile_path` | Required only when selecting the baseline config. |
| `inference_config_path` | Required whenever reusing the baseline config. |
| `inference_config_schema` | Exact machine-readable key/type authority for `inference_config.yaml`; never infer keys from examples or prior Runs. |
| `inference_mapping_contract` | Exact allowed/required keys and types inside `measurement.inference_config`. |
| `config_validation_command` | Side-effect-free validator to run after the optional tokenizer-only rendering check and before model generation or prediction work. |
| `memory_planning_contract` | Engine-owned formula and bounds for sizing vLLM from model weights plus the planned KV-cache workload. |
| `memory_helper_path` | System-owned helper used to derive the concrete GPU fraction and the matching free-memory preflight threshold. |
| `predictions_validation_command` | Exact side-effect-free output validator with assigned questions/sample paths already inserted. Replace only `<absolute-predictions-csv-path>` after copy-back. |
| `reusable_predict_script_path` | Prior verified `predict.py`; reuse when compatible, otherwise document why regeneration was required. |
| `configuration_suggestions` | Baseline-only high-level advice; reuse mode rejects new suggestions. |
| `configuration_pins` | Binding user values. |
| `device_info_path` | Cloud/instance device contract or cluster access route. |
| `slurm_job` | Engine-owned finite-job path/name, job-id stdout/stderr patterns, GPU count, queue deadline, phase, backend-observed JOBID/state, bookkeeping endpoints, and exact schemas. Enabled only for cluster. `submit` creates one job and returns deferred; `collect` never duplicates it. |
| `generation_backend` | Run-owned `hf` or `vllm`. |
| `work_dir` | Persistent local artifact directory. |
| `suite_members` | Additional benchmark contracts, each with its own engine-assigned `benchmark_id`. One GPU job owns the whole suite; never submit one job per member. |

`inference_config.yaml` must validate as `InferenceRunConfig` and include:

- schema version and actual base model;
- complete `prompt` (`prompt_framing`, `model_reasoning_type`, `system_prompt`);
- exact `template_kwargs` actually supported and passed to the tokenizer chat
  template; `{}` means pass no extra template kwargs;
- complete `measurement` (`inference_config`, explicit greedy/sampling strategy,
  max tokens, temperature, top-p, top-k, repetition penalty, and seed);
- generation backend and complete backend-specific `implementation_config`;
- tokenizer source, exact chat-template source/hash when chat framing is used,
  special-token ids, and verified `stop_token_ids` passed to the backend;
- `prompt_example`: ordered `<INPUT:field>` values, chat messages when used,
  and the exact template-rendered prompt ending at the generation boundary;
- accepted/adjusted/self-selected suggestion decisions with rationale.

Build `prompt_example` from synthetic placeholders through the same rendering
function and recorded `template_kwargs` used by `predict.py`. Never hard-code a
second mapping in the script. Never copy a Validation/Test answer or example
reference into it. For chat, the first message is the realized system prompt;
for completion/text, `messages` is empty.

`model_reasoning_type` records the selected execution mode, either `thinking`
or `non_thinking`. Baseline Inference verifies support against the exact model,
tokenizer, and template and honors explicit user requests. The rendered prompt,
template kwargs, and measured output must agree with that mode. Report an
unsupported request rather than silently substituting the model default.

Return exactly one `InferenceResult`. Its top-level artifact fields describe
the primary benchmark; on success, its `suite_members` must contain exactly
one result for every assigned non-primary member. A cluster submission returns
`status="deferred"` after the JOBID is registered; this is not success or
failure and carries no predictions. It posts `Waiting:`, never `Done:`; the
engine posts `Done:` only after a later collect Result and its artifacts pass
validation. For cluster success,
`slurm_script_path` is exactly `slurm_job.script_path`; for other providers it
is empty. On success, `inference_config_path`, `predictions_path`, and
`generation_diagnostics_path` are required absolute paths, and `n_requests`
must report all real generation calls, including sequential multi-turn calls.
The YAML is the sole authority
for base model, prompt, template, mapping, decoding, and backend configuration;
do not duplicate them in the Result. Report only `predict_script_path` when a
reproducible helper was produced. The engine compares it with the resolved
input artifact and records reuse provenance; do not echo or transform a source
path in the Result. A
missing helper is visible as an advisory artifact warning; it cannot relax
config or prediction checks.

For an assigned suite, keep each member in its own `work_dir` with its own
`inference_config.yaml`, `predictions.csv`, and
`generation_diagnostics.json`. Initialize the tokenizer and model engine once,
then process the primary member followed by `suite_members` in order. All
member configs must share the exact model/backend/tokenizer/chat-template
identity and vLLM `llm_kwargs`; prompt mapping, decoding controls, and output
schema remain member-specific. Emit progress before and after every member.

Run `predictions_validation_command` after `predictions.csv` is copied back and
before success. Do not recreate its checks with a basename, the current working
directory, or a separately authored Python/Bash snippet. Empty or unparseable
real model replies remain valid rows and are reported through diagnostics; the
command checks artifact shape and input alignment, not model quality.

`generation_diagnostics_path` is a JSON file with `schema_version=1` and one
ordered, dense `records` entry per real generation request. Each entry contains
only `request_index`, `row_index`, one-based `turn`, the backend-observed (or,
for Hugging Face, token-derived) `finish_reason` and `stop_reason`, and
`generated_tokens`; do not duplicate prompt, prediction, reference, or
held-out text. Never infer termination from visible decoded text. The
successful Result's `n_requests` must equal the number of records exactly.
