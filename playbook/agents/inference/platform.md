## Execution order

1. Resolve the output directory and validate all input files.
2. Read `device_info.json`; require a usable cloud/instance GPU or a verified
   cluster access route, honor `instance.visible_devices` for instances, and
   verify its host-RAM plan covers model loading plus runtime/tokenization state.
3. Resolve baseline, adapter, or full-checkpoint model mode. Never substitute.
4. Select or load the YAML configuration as directed. In baseline select mode,
   a bounded tokenizer-only compatibility/rendering check on the assigned
   runtime is allowed to realize the exact `prompt_example`; it must not load
   the model for generation or produce predictions. Treat
   `inference_config_schema` as the sole key/type authority and run the supplied
   `config_validation_command` successfully before measured generation. For
   vLLM baseline selection, follow `memory_planning_contract`, record the
   absolute `implementation_config.memory_plan`, and do not freeze
   `gpu_memory_utilization` in `llm_kwargs`.
5. When the work order supplies `memory_helper_path`, copy it beside
   `predict.py` and import its `gpu_memory_utilization` function. When the
   configuration sets `scoring_mode="option_loglikelihood"`, likewise use the
   `option_scoring_helper_path` helper beside `predict.py` to emit per-option
   log-likelihoods (see the option-scoring section). Reuse
   `reusable_predict_script_path` when
   it reads the supplied YAML and model artifact generically. Otherwise
   generate `predict.py` from the YAML and state
   the concrete incompatibility that prevented reuse. Upload required files to
   a ticket-specific remote directory, using the `remote_transfer` helper by
   default or correctly constructed direct SCP when needed. Run directly for
   cloud/instance or use the finite cluster job below.
6. Copy back `predict.py`, the execution log, `predictions.csv`, and
   `generation_diagnostics.json` through the
   helper or correctly constructed direct SCP. Direct cloud/instance execution
   uses `infer.log`; cluster execution preserves the exact successful
   `slurm-<JOBID>.out` and sibling `slurm-<JOBID>.err`, and reports the `.out`
   file as `InferenceResult.log_path`.
7. Run the exact `predictions_validation_command`, verify one termination
   record per real request, then return one
   `InferenceResult`.

Use direct SSH for `cloud`; Infrastructure obtains that route from the cloud
API. `instance` is also direct SSH, but to a fixed host: export
`CUDA_VISIBLE_DEVICES=device_info.instance.visible_devices` and the exact
`ZEVO_TICKET_ID=<ticket_id>` into the prediction process so cancellation can
terminate only this Ticket. Never invoke Slurm commands for instance. For a route with
`password_path`, prefix direct SSH/SCP with `sshpass -f <password_path>` and
never read or print that file; `remote_transfer` handles it automatically.
Never fall back to CPU or run GPU inference inside the scheduler container.

### Cluster: finite `predict.sbatch`

When `slurm_job.enabled=true` and `slurm_job.phase="submit"`, write the exact local file at
`slurm_job.script_path`. Load the installed site-operation Skill matching
`device_info.ssh.host` and apply its scheduler/container constraints; fail if
multiple site Skills match. Put the resource plan and `slurm_job.num_gpus` into
`#SBATCH` directives, use exactly `slurm_job.job_name`, activate
`cluster.env_setup`, and, when `memory_helper_path` is supplied, import the
copied `zevo_inference_memory` helper to select an allocated GPU using the
recorded absolute target. Run `predict.py` in the
foreground, and exit with it. The file is the actual inference job, not an
empty allocation. Never use `sleep infinity`, `salloc`, `srun --overlap`,
`--wrap`, or resource flags on the `sbatch` command line.
Use exactly `#SBATCH --output=<slurm_job.stdout_path>` and
`#SBATCH --error=<slurm_job.stderr_path>`. Keep stdout and stderr separate;
never write a cluster job directly to `infer.log`, reuse one filename across
repairs, or combine both streams with one directive.

Before environment activation or any Python/package import, render the matched
site Skill's writable temporary/cache policy. Keep durable outputs in the
mounted Ticket directory and use the site's job-private temporary root for
disposable process state. Preserve a configured persistent model-download cache
when the site Skill requires one. Validate the resulting batch script and the
allocated runtime; do not copy a hard-coded cache-management implementation
between clusters.

Embed `slurm_job.lifecycle_prologue` byte-for-byte in the executable body
before model loading. Do not rewrite its functions, event format, trap, or
status path. It emits `RUNNING` when the allocation begins and `EXITED` when
the script exits, allowing Zevo's persistent status stream to observe lifecycle
changes without repeatedly querying Slurm.

Validate with `bash -n`, upload and checksum the file, then submit only
`sbatch --parsable <remote-predict.sbatch>`. Parse only its first
semicolon-delimited component as the JOBID, then immediately register it at
`slurm_job.infra_instances_endpoint` as cluster/provisioning for this
Run/Ticket, zero cost, and metadata containing `auto_release=true`,
`stage_job=true`, `scheduler_state="PENDING"`, the remote script/workdir, and
`status_path` equal to the exact `slurm_job.status_path`.
Use the supplied exact schemas. If bookkeeping fails, cancel the job. Then
return `status="deferred"` with `slurm_script_path` set and no claimed
predictions. Do not POST `Waiting:`, `Running:`, or `Done:`: after validating
the typed Result, script, status path, and registered JOBID, the engine commits
the watcher handoff and publishes the appropriate lifecycle message. Do not
call `squeue`, `sacct`, or wait for the job: the backend Scheduler streams the
job-local lifecycle file, uses state-aware low-frequency polling only as a
fallback and for authoritative terminal details, enforces the queue deadline,
records terminal state, and wakes this same Ticket.

After the deferred Result passes validation, the engine resolves `%j` with the
registered JOBID and stamps the exact stdout/stderr paths into watcher metadata;
the Agent must not provide or override those engine-owned fields.

When `slurm_job.phase="collect"`, never submit another inference job. If its
state is still non-terminal, return `deferred` again without posting a
lifecycle message. On `COMPLETED`, verify and
copy predictions/config/script plus the exact `slurm-<JOBID>.out` and
`slurm-<JOBID>.err`, run the predictions validator, and return success with the
local `.out` path as `log_path` without POSTing `Done:`; the engine publishes it
only after accepting the Result and artifacts. Collect never parses or replays
the copied log into telemetry: the live watcher owns progress while the job is
running. For any other terminal state,
inspect both exact job streams and return a specific failure. The finite job releases its GPUs
automatically; do not create an Infrastructure release ticket or PATCH
scheduler-owned state.

## Selecting a model-lineage baseline YAML

The answer-free profile is closed evidence. Use its ordered `input_fields`,
submission columns, encoding, aggregate lengths, and task shape; never search
for Validation answers or scorer code.

Resolve values in this priority order:

1. binding `configuration_pins`;
2. compatible Orchestrator suggestions;
3. the Task Objective and intended serving interface;
4. model/tokenizer/data/runtime compatibility evidence;
5. a safe Specialist-selected value with rationale.

Do not treat `0`, `null`, `""`, or a missing suggestion as a requested value.
Record whether each actual suggestion was accepted or adjusted. Record
self-selected values when no suggestion existed.

Classify the exact selected model/checkpoint as either `thinking` or
`non_thinking` by verifying its tokenizer and template on the assigned runtime;
do not infer the class from a family name alone. This is a model property, not
an Agent preference, Orchestrator suggestion, or user pin. Record the result as
`prompt.model_reasoning_type`. A thinking model uses its verified thinking
template and emits reasoning content before the answer. A non-thinking model
uses its ordinary template with no thinking-specific arguments or markup.
These are the only two model classes. A chat prompt always has a system message;
the empty user value canonicalizes to `You are a helpful assistant.`
Completion/text framing has no system message and is valid only for a
non-thinking model.

Choose framing for the target behavior, not for what the untrained base happens
to do best before improvement:

- conversational assistance, instruction following, tool dialogue, or other
  role-based interaction uses chat;
- prompt-to-continuation language modeling uses completion;
- unconditional or document-level continuation uses text;
- reasoning-oriented interaction still uses chat, with thinking behavior only
  when the exact tokenizer/runtime supports and verifies it.

If the Objective requires chat but the tokenizer has no native chat template,
select one explicit stable compatible template before Baseline, record its exact
source text/hash and special tokens, and use it for every Baseline row. Train
will receive and reuse that same contract. Do not switch Baseline to completion
merely because the starting model is a base model, and do not silently adopt a
different native template after training.

For chat, capture the exact tokenizer/template source and the bare SHA-256 of
the exact template text—exactly 64 lowercase hexadecimal characters with no
`sha256:` prefix—plus the special-token ids. `prompt_framing="chat"` alone is
not a reproducible template. Follow `inference_config_schema` literally for
this and every other serialized representation.

Also resolve and record `template_kwargs`: the exact extra keyword mapping
passed to `tokenizer.apply_chat_template` for both the synthetic example and
every real row. Probe support on the assigned tokenizer/runtime rather than
assuming a model family accepts a named control. A thinking model may, for
example, require `{"enable_thinking": true}`. A non-thinking model uses the
verified ordinary rendering path and no thinking-specific kwarg. Record the exact template source/hash
and never patch its rendered text afterward.
`tokenize`, `add_generation_prompt`, and return-shape controls
are execution controls, not `template_kwargs`. The prediction script must load
this mapping from YAML; never hard-code a second copy or repeat its keys under
`implementation_config`. When `enable_thinking` is present it must be `true`
and must agree with `prompt.model_reasoning_type=thinking`; a non-thinking model
does not serialize that kwarg.

The task mapping belongs under `measurement.inference_config`; supported keys
are `input_fields`, `answer_regex`, `answer_column`, `batch_size`, `stop`, and —
for multiple-choice option scoring — `scoring_mode` and `option_fields`.
Use `inference_mapping_contract` as the machine-readable authority for which are
required and for their value types.

When the mapping carries the Task-owned semantic protocol, it contains all of
`task_instruction`, `user_prompt_template`, `output_instruction`,
`response_format`, and `answer_parser`. These values are binding, not advice.
For every row, render `{input}` as the sole input value when there is one, or as
newline-separated `field: value` pairs in `input_fields` order when there are
several; direct `{field_name}` placeholders are also allowed. Join the task
instruction, rendered row input, and output instruction with blank lines to
form the user turn. Never substitute `sample_submission.csv` for this protocol:
the sample defines output columns, not what the model is asked to say. Apply
`answer_regex`/`answer_parser` to the generated response and write the parsed
scoring value to the declared prediction column.

### Multiple-choice option scoring (opt-in)

By default `scoring_mode` is absent and Inference generates a free completion
and parses the answer with `answer_regex` — unchanged behavior. Set
`scoring_mode="option_loglikelihood"` (with `option_fields` listing the columns
that carry each answer option) to score MMLU-style MC-QA the field-standard way:
instead of generating a letter and string-matching it, the prediction script
computes, for each option, the log-likelihood the model assigns to that option's
text as the completion of the frozen prompt, and writes those per-option scores
into the single prediction column as a JSON object, for example
`{"A": {"logprob": -12.34, "num_tokens": 6}, "B": {"logprob": -9.01, "num_tokens": 5}, ...}`
(the raw summed completion log-prob plus its token count; the evaluator length-
normalizes as `logprob / num_tokens`). A pre-normalized numeric per option
(`{"A": -2.05, "B": -1.80, ...}`) is also accepted. The deterministic
`mc_loglikelihood` (alias `accuracy_norm`) evaluation metric then argmaxes these
numbers against the gold option — no model call happens at scoring time, so the
"Evaluation is LLM-free" invariant holds. The gold column may hold the option
label (`"B"`) or a zero-based index.

Both TRL/HF and vLLM can produce these scores (HF: teacher-forced sum of
per-token log-probs over each option's tokens; vLLM: `prompt_logprobs` on the
prompt+option, or completion logprobs). The system-owned helper at
`option_scoring_helper_path` is copied beside `predict.py` (as
`zevo_option_scoring.py`, exactly like the memory helper); in
`scoring_mode="option_loglikelihood"` import it and, per row, call
`hf_option_scores(model, tokenizer, prompt, options)` (HF teacher forcing) or
`vllm_option_scores(llm, prompt, options)` (vLLM `prompt_logprobs`) to get the
per-option object, then `emit_option_scores(...)` — or `json.dumps` of the
returned dict — into the single prediction column. `options` maps each option
label to the exact option text (drawn from the `option_fields` columns) that the
frozen prompt is completed with; the helper reads per-token log-probs off the
model and does the length-preserving `{logprob, num_tokens}` emission so the
`mc_loglikelihood` metric normalizes consistently. Do not import or invoke it in
default generation mode. Where option scoring is not wanted, keep `scoring_mode`
unset and use generation scoring.
Decoding fields are `decoding_strategy`, `max_new_tokens`, `temperature`,
`top_p`, `top_k`, `repetition_penalty`, and `seed`. Record
`decoding_strategy=greedy` with `temperature=0`, or
`decoding_strategy=sampling` with `temperature>0`; do not leave the strategy
implicit or pass sampling-only controls in a way that changes backend semantics.

Resolve termination at token level before Baseline generation. Record EOS and
every verified single-token response/turn terminator used by the selected
template in top-level `stop_token_ids`; every id must also appear in
`special_token_ids`. Tokenize candidate markers with the recorded tokenizer and
`add_special_tokens=false` instead of guessing ids from a model-family name.
Keep multi-token or ordinary textual boundaries under
`measurement.inference_config.stop`, but do not assume a special-token string
survives decoded output when `skip_special_tokens=true`.

For vLLM, pass the YAML list directly as
`SamplingParams(stop_token_ids=config["stop_token_ids"], ...)` and also pass
the textual stop list for non-token boundaries. For Hugging Face generation,
include the same ids in the effective EOS/stopping criteria without changing
their persisted meaning. Do not duplicate `stop_token_ids` under
`implementation_config.sampling_params`. Capture the real `finish_reason`,
`stop_reason`, and generated token-id count before postprocessing or dropping
special tokens, then write the required generation diagnostics. vLLM exposes
the two reasons on each request output. Hugging Face does not expose an
equivalent finish-reason field, so derive it from the untrimmed generated token
ids: `stop` plus the terminal id when the final token is in `stop_token_ids`,
otherwise `length` plus a null stop reason when the generation reaches its
configured token limit. Do not infer either outcome from decoded text.

### vLLM GPU-memory sizing

When the work order supplies `memory_planning_contract`,
`gpu_memory_utilization` is a device-specific capacity limit, not a quality or
speed target. Never copy a fixed value such as `0.85` merely because a large GPU
is available. In baseline `configuration_mode=select`, measure and record an
absolute per-GPU target under `implementation_config.memory_plan`:

- `runtime_weight_gib`: model parameter bytes at the selected runtime dtype,
  not the compressed download size;
- `peak_live_tokens`: the planned maximum live prompt-plus-generation tokens
  from the real tokenized question workload, bounded by `max_num_seqs` and
  `max_model_len`;
- `kv_bytes_per_token`: `2 * layers * key_value_heads * head_dim * dtype_bytes`;
- `kv_cache_gib`: `peak_live_tokens * kv_bytes_per_token / 1024^3`;
- `tensor_parallel_size`: exactly the value in `llm_kwargs`;
- `target_gpu_memory_gib`: the ceiling of
  `(runtime_weight_gib * 1.2 + kv_cache_gib) / tensor_parallel_size + 4`;
- the exact step/min/max/free-margin constants from
  `memory_planning_contract`.

Do not put `gpu_memory_utilization` in the persisted `llm_kwargs`. At execution,
import the system-owned `zevo_inference_memory` helper and derive the fraction
from `target_gpu_memory_gib` and that GPU's actual total memory. Add only that
derived value to the in-memory kwargs passed to vLLM and log the measured total,
free memory, absolute target, and derived fraction. The Slurm preflight must use
the same helper's `required_free_memory_gib` result when selecting among the
allocated GPUs, so preflight and engine initialization have one standard.

If the target exceeds the allowed fraction, increase tensor parallelism or
reduce a non-semantic concurrency limit such as `max_num_seqs`; do not hide an
unfit plan by clamping it. A small model on a B200 should ordinarily reserve a
small fraction rather than most of the card. A GPU with an active foreign
compute process remains suspect even when enough bytes are free: briefly wait
for allocation cleanup or fail/requeue instead of deliberately sharing it.

In `configuration_mode=reuse`, preserve the absolute plan and helper behavior
exactly. Recompute only the device-derived fraction for the concrete GPU; this
does not change prompts, decoding, batching semantics, or predictions. A
legacy reuse work order that supplies neither memory field preserves its
existing YAML and script exactly; it does not retrofit or mutate an established
model-lineage measurement contract.

Write the YAML before launching generation and load the generated script from
that file. The file is the authority, not prose in the transcript or duplicated
fields in `InferenceResult`.
Before writing it, run one synthetic placeholder input through the same prompt
builder and tokenizer/template code path used for real rows. Store the ordered
`<INPUT:field>` mapping, pre-template chat messages, exact rendered prompt
(including special tokens and assistant prefix), and
`generation_starts_at=end_of_rendered_prompt` under `prompt_example`. This is
an audit example, not an extra generation: it contains no model response,
Validation/Test answer, or scorer-derived value.
The stored rendered prompt must be the output produced with the recorded
`template_kwargs`; a transcript description of the model class is not
sufficient. Its structure must match the recorded `model_reasoning_type`, and
measured thinking-model output must contain actual reasoning content.
`implementation_config` records every resolved backend/framework argument that
actually reaches generation and every deterministic runtime policy not already
represented by `measurement`.
Do not add execution-envelope keys such as operation, iteration, model source,
tokenizer wrapper objects, or runtime wrapper objects: if a value belongs in
the YAML, place it in the exact schema field supplied by the Ticket. Do not
inspect source code, old artifacts, or failed validation messages to invent a
schema that the Ticket already provides.

Design baseline `predict.py` to accept the model/checkpoint path, questions-only
input, submission template, output path, and YAML path as runtime inputs rather
than hard-coding the baseline model. A later candidate should execute that exact
script when compatible. Regeneration is allowed only for a real execution-mode
incompatibility (for example adapter loading versus a standalone checkpoint),
not merely because a new Ticket started. The engine determines reuse and source
provenance from the resolved input binding and file bytes. Report only the
realized `predict_script_path`; do not echo or transform the source path.

## Reusing the YAML

For `configuration_mode="reuse"`, validate the supplied file as
`InferenceRunConfig`, check checkpoint compatibility, and execute it without
mutation. The returned `inference_config_path` must identify the exact supplied
file. A changed body, changed hash, or locally regenerated equivalent is a
contract failure.

Training may have changed model weights, not the measurement. Preserve logical
input order, prompt rendering, chat template, model reasoning type, system prompt,
template kwargs, parsing, stop behavior, backend, generation length, sampling
controls, and seed.

## Predictions

Read every questions-only row once and preserve order. Build prompts only from
the YAML's ordered input mapping. Generate one real model reply per row. Write
exactly the sample-submission columns in exactly their order. Never prefill,
duplicate, filter, reorder, or synthesize replies to make the file complete.

Before success require:

- non-empty predictions and exactly one row per input row;
- exact submission columns/order;
- stable ids/order where present;
- real model output retained even when parsing fails;
- local config, script, log, predictions, and generation-diagnostics files
  exist and are non-empty;
- every request has a recorded backend finish reason and generated-token count;
  reaching the configured limit remains a visible `length` outcome.

Emit phase/progress/config telemetry from the real run. Telemetry is
observability; it does not replace `inference_config.yaml`. Structured
telemetry owns the cluster progress display, so disable the backend's raw tqdm
bar when supported (for vLLM, call generation with `use_tqdm=false`) and keep
ordinary runtime diagnostics concise. Preserve warnings and failures in the
job-id-specific stderr file instead of merging them into stdout.

## Memory and failure

Store only verified, reusable Run-local lessons such as a vLLM/version-specific
argument correction. Do not store answers, scores from the held-out lane, or a
replacement configuration. A missing GPU, incompatible model/template,
unusable checkpoint, ambiguous task mapping, failed remote command, or invalid
artifact is `status="failed"` with a specific error.
