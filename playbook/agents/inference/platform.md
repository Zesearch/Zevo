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
   For one model replica, initialize the engine once and loop over the primary
   plus every member in that process. A cluster allocation can instead run
   independent replicas via `parallel_runner_path` below. Never submit
   another GPU job for a suite member. Atomically persist each member before
   moving on so a repaired execution can skip completed members. During every
   member emit progress with its exact engine-assigned `benchmark_id`, Task
   display `benchmark_name`, one-based `benchmark_index`, and
   `benchmark_total`, in addition to row `step` / `total`. Never reconstruct
   the ID from the name; the ID counts progress and the name is only a label.
   Put the first benchmark's config, predictions, and diagnostics in
   `primary_member_work_dir` (`work_dir/suite/000`) for every Inference ticket,
   including one-benchmark runs. Additional members use their assigned
   `suite/001`, `suite/002`, etc. directories. Keep ticket-wide `predict.py`,
   batch script, and execution logs in the top-level `work_dir`. An older work
   order without `primary_member_work_dir` keeps its original artifact paths.
6. Copy back `predict.py`, the execution log, the primary benchmark's
   `predictions.csv` and `generation_diagnostics.json` from its assigned
   directory through the
   helper or correctly constructed direct SCP. Direct cloud/instance execution
   uses `infer.log`; cluster execution preserves the exact successful
   `slurm-<JOBID>.out` and sibling `slurm-<JOBID>.err`, and reports the `.out`
   file as `InferenceResult.log_path`.
   Also copy each suite member's config, predictions, and diagnostics into its
   assigned `work_dir`.
7. Run the exact primary and per-member `predictions_validation_command`,
   verify one termination record per real request, then return one complete
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
multiple site Skills match. `device_info.resource_plan` is an Infrastructure
envelope, while `slurm_job` is the exact engine-selected request for this
Inference execution. Put `slurm_job.nodes` and `slurm_job.num_gpus` /
`slurm_job.gpus_per_node` into `#SBATCH` directives, use exactly
`slurm_job.job_name`, activate
`cluster.env_setup`, and, when `memory_helper_path` is supplied, import the
copied `zevo_inference_memory` helper to check allocated GPUs using the
recorded absolute target. Run the inference workload in the foreground and
exit with it. One suite is one `sbatch` and one allocation; parallel replicas
may each load the model once inside that allocation. The file is the actual inference job, not an
empty allocation. Never use `sleep infinity`, `salloc`, `srun --overlap`,
`--wrap`, or resource flags on the `sbatch` command line.
Size `#SBATCH --cpus-per-task` and `--mem` for simultaneous replicas using the
site limits and Infrastructure host-memory envelope. If host RAM/CPU can only
support one replica, run one even if a whole-node minimum reserves more GPUs.

For useful parallel work and `slurm_job.num_gpus >= 2 *
recommended_gpus_per_replica`, keep the YAML
`implementation_config.llm_kwargs.tensor_parallel_size` at the smallest
model-fitting size (normally `recommended_gpus_per_replica`), not the whole
allocation. Respect a larger frozen baseline YAML in reuse mode. Copy
`parallel_runner_path` beside the remote `predict.py`. Upload the normal
ordered `suite.json` manifest; each member has the exact `benchmark_id` from
the primary work order or `suite_members`, plus `name`, `config`, `questions`,
`sample_submission`, `output`, and `diagnostics`. `predict.py` must accept
`--model`, `--suite`, `--summary`, `--ticket-id` and emit per-member progress.
After validating every YAML, run the system helper in the batch foreground:

```
python zevo_parallel_inference.py --predict predict.py --model MODEL_PATH \
  --suite suite.json --summary suite_summary.json --ticket-id TICKET_ID \
  --allocated-gpus SLURM_JOB_GPU_COUNT --gpus-per-worker YAML_TENSOR_PARALLEL_SIZE
```

Use absolute paths in the real script. The helper isolates each worker's GPUs,
HOME, temporary directory, and writable compilation caches (including
FlashInfer, Triton, TorchInductor, CUDA and vLLM), splits large prepared CSV benchmarks when useful,
merges results in original row order, and marks each Benchmark complete as
soon as its shard artifacts are validated, even while a replica continues with
other Benchmarks. A final generated-row progress event alone is not completion.
If available host RAM or CPU cannot support all possible replicas, pass
`--max-workers N` with the safe limit derived from the site's allocation;
never let the requested GPU count alone imply that many model copies fit.
It never reads private answers. Do not preselect one `CUDA_VISIBLE_DEVICES`
before calling it: it needs Slurm's complete device mask. Run the system-owned
`required_free_memory_gib` preflight for every GPU group that will run a
replica. When only one replica fits, direct `predict.py --suite ...` remains
valid.
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
After the site policy, render `slurm_job.runtime_prologue` byte-for-byte before
environment activation or the workload. It enforces a writable, short
job-private temp root for process state; do not override its TMPDIR later.

Embed `slurm_job.lifecycle_prologue` byte-for-byte in the executable body
before model loading. Do not rewrite its functions, event format, trap, or
status path. It emits `RUNNING` when the allocation begins and `EXITED` when
the script exits, allowing Zevo's persistent status stream to observe lifecycle
changes without repeatedly querying Slurm.

Validate with `bash -n`, upload and checksum the file at the exact
`slurm_job.remote_script_path` inside `slurm_job.remote_work_dir`. Do not call
`sbatch` or create Infrastructure bookkeeping yourself. Return
`status="deferred"` with `slurm_script_path` set and no claimed
predictions. Do not POST `Waiting:`, `Running:`, or `Done:`: after validating
the typed Result, configuration, script, and paths, the engine submits the
uploaded file, registers the returned JOBID, commits the watcher handoff, and
publishes the appropriate lifecycle message. Do not
call `squeue`, `sacct`, or wait for the job: the backend Scheduler streams the
job-local lifecycle file, uses state-aware low-frequency polling only as a
fallback and for authoritative terminal details, enforces the queue deadline,
records terminal state, and wakes this same Ticket.

After the deferred Result passes validation, the engine submits the job,
resolves `%j` with its JOBID, and stamps the exact stdout/stderr paths into watcher metadata;
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

If that terminal failure came from generated implementation, command, or
runtime setup and produced no valid predictions, repair the generated files in
place. The next repair activation may carry `slurm_job.phase="submit"` with
`attempt=2`; return `deferred` and let the Engine validate the changed script
and create the distinct second JOBID. Never resubmit the terminal collect job
yourself. Only one automatic external re-execution is allowed for a Ticket.

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

Verify the exact model/checkpoint's supported reasoning modes and template on
the assigned runtime; do not infer support from a family name alone. Record
the selected execution mode as `prompt.model_reasoning_type` (`thinking` or
`non_thinking`). Model capability, template defaults, and the selected mode
are distinct. Honor explicit user requests, including those in `model_query`;
if unsupported or conflicting, report the conflict rather than silently
adjusting the request. Without a request, use verified model/runtime evidence
to select and explain the mode. A thinking mode emits reasoning before the
answer. A non-thinking mode may require an explicit disabling argument and
native empty reasoning delimiters. An omitted argument is not proof of disabled
thinking. A chat prompt always has a system message; an empty user value
canonicalizes to `You are a helpful assistant.` Completion/text framing has
no system message and is valid only for non-thinking mode.

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

### Structured conversation inputs

When the task explicitly asks to continue a conversation supplied in a record,
set `measurement.inference_config.conversation_field` to that input field.
Set `conversation_fallback_field` only when the task declares a plain prompt
fallback; both fields must be in `input_fields`. A field named `messages` alone
is not sufficient: quoted conversations in classification/analysis tasks remain
ordinary task data. In conversation mode, `inference_query` describes how to
interpret the record; do not interpolate it into an extra user turn.

Decode a JSON-string conversation or accept its already-parsed list. Preserve
role, content, and order, including historical assistant replies, then apply
the recorded chat template once with `add_generation_prompt=True`. A nonempty
conversation is authoritative: do not append `prompt` or duplicate the final
user turn. Only an empty/missing conversation uses the declared fallback as a
single user turn. Prepend the configured system message only if the history
has no system message. Never replace a record's existing system instructions.
Malformed JSON, unsupported role/content structures, or a missing final user
turn must produce a clear input error, not a silent flattening or fallback.
For the supported text-only system/user/assistant shape, the reference behavior
is `zevo.contracts.prompting.conversation_messages`; generated runtime code
must follow the same behavior without depending on unavailable remote modules.

Verify the real prompt builder with synthetic user/assistant/user history,
an empty-history fallback, and malformed JSON before generation (tokenizer-only,
no GPU model load). Assert the preserved turns and exactly one final user turn;
confirm the fallback is absent when history exists. The YAML `prompt_example`
uses a JSON string of synthetic role/content turns for `conversation_field`,
with every content exactly `<INPUT:field>` for that field. Other input values
keep their plain `<INPUT:field>` placeholders. Render this structured example
through the same path as real rows; never create a separate flat example to
satisfy validation. Record the input interpretation in YAML before Baseline
and preserve it in reuse mode.

For chat, capture the exact tokenizer/template source and the bare SHA-256 of
the exact template text—exactly 64 lowercase hexadecimal characters with no
`sha256:` prefix—plus the special-token ids. `prompt_framing="chat"` alone is
not a reproducible template. Follow `inference_config_schema` literally for
this and every other serialized representation.

Also resolve and record `template_kwargs`: the exact extra keyword mapping
passed to `tokenizer.apply_chat_template` for both the synthetic example and
every real row. Probe support on the assigned tokenizer/runtime rather than
assuming a model family accepts a named control. When supported,
`enable_thinking=true` must agree with thinking mode and `enable_thinking=false`
with non-thinking mode. Preserve explicit disabling arguments; omitting them
can restore a model's thinking default. Models without that control use their
verified native mechanism. Record the exact template source/hash and never
patch its rendered text afterward. Empty native `<think></think>` blocks can
represent disabled thinking; tag presence alone does not establish the mode.
`tokenize`, `add_generation_prompt`, and return-shape controls are execution
controls, not `template_kwargs`. The prediction script must load this mapping
from YAML; never hard-code a second copy or repeat its keys under
`implementation_config`. Verify the selected behavior on a bounded generation
probe before the full prediction job. If the probe contradicts an explicit
request, resolve or report it; do not label the configuration compliant.

The task mapping belongs under `measurement.inference_config`; supported keys
are `input_fields`, `answer_regex`, `answer_column`, `batch_size`, `stop`,
`conversation_field`, `conversation_fallback_field`, and —
for multiple-choice option scoring — `scoring_mode` and `option_fields`.
Use `inference_mapping_contract` as the machine-readable authority for which are
required and for their value types.

`inference_query` is the Task Test contract's exact user-turn instruction. It
is binding. If it contains `{input}` or `{field_name}` placeholders, render
them from the answer-free row. If it has no placeholders, append the rendered
row after a blank line. Record the same query in `measurement.inference_config`
and make `prompt_example` prove the exact rendered user turn. A sample
submission specifies prediction columns only; never use it as a prompt.

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

Do not pass the removed legacy `swap_space` argument to `vllm.LLM`. The
deterministic adaptive-memory validator rejects it before submission; use only
arguments supported by the installed vLLM `EngineArgs` contract.

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

Sample submissions illustrate the output schema. Populate identifiers from the
assigned inputs and output values from each actual model response, following
the Task's field definitions.

Apply deterministic extraction to the generated response, preserving the
requested answer structure and content. Retain the original decoded completion
in the configured raw-output artifact. Use an empty string for a missing
extracted value unless the Task specifies another missing-value representation.
Pin extraction behavior with the baseline and reuse it unchanged.

Before success require:

- a non-empty predictions file and exactly one row per input row; an empty
  extracted field is valid when extraction failed as described above;
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
