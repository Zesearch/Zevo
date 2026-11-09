## Resource recovery

For an unusable allocation, return a failed collect result with
`execution_retry_reason`, `execution_retry_action="replace_resource"`, and
`failure_recovery.scope="current_task"` with verified evidence. The engine
replaces resources within the unchanged plan and resumes this same Ticket.
Use the default `resubmit` for corrected workloads on healthy resources. Do not
release resources yourself. A changed resource requirement still requires review.
Preserve complete checkpoints and committed output receipts; never silently
restart completed work when a replacement host lacks the durable artifacts.

## Execution transport

`COMPLETED`/exit 0 is not task completion. Failed collect requiring execution:
set `failure_recovery.scope="current_task"` and `execution_retry_reason` with
evidence. The engine resubmits the same Ticket within its repair limit; collect
must not launch. Leave the reason empty for artifact-only fixes. Submission on
unchanged resources is not `resource_plan`.

Use one authoritative admission check and consistent units, without duplicate
shell/runtime thresholds or extra margins. Shell probes may log diagnostics.
Rejected prerequisites must exit nonzero; preserve checkpoints. Verified
checkpoint yields and completed segments may still exit 0.

For an enabled execution contract, follow the shared engine-owned workload
protocol first. `slurm_job.transport="direct"` prepares a `.sh` on the existing
cloud/instance host and returns `deferred`; the engine runs it and later wakes
collection. The cluster submission sections below apply only to
`slurm_job.transport="slurm"`. Inline execution instructions apply only to
historical/CPU work orders with `slurm_job.enabled=false`.


## Large-scale preprocessing performance

Apply this requirement to full-dataset scans, tokenization, length statistics,
sorting, grouping, reordering, and materialization, including preparation inside
a training job. Choose algorithms and data-access patterns that scale with the
actual workload. Use batch operations, vectorization, or measured parallelism
where appropriate. Avoid repeated disk, remote, or lazy Dataset lookups inside
per-example loops or sorting keys/comparators. Materialize repeatedly accessed
compact metadata, such as sequence lengths, once in memory when it fits the
memory budget; otherwise use bounded batches or an external-memory algorithm.

Before the first full execution of a new or changed preparation path, benchmark
a representative sample of the permitted input on the assigned data plane.
Include long and multi-turn examples where present. Measure throughput, peak
memory, and time per step, then estimate full-dataset cost with explicit
assumptions. A benchmark subset is only a performance probe and must not replace
the required full training dataset. Reuse applicable measurements when the
implementation, input characteristics, and runtime are unchanged.

During execution, report separate progress and elapsed time for scanning,
tokenization, ordering, and writing, as applicable. Investigate sustained stalls
or substantial deviations from the estimate; identify and address the bottleneck
instead of waiting indefinitely or blindly repeating the same expensive work.
Cache deterministic intermediate results and reuse them only after verifying
that the input, relevant configuration, and implementation still match.

Performance optimizations must preserve the experiment's data selection,
tokenization, labels, random-seed behavior, grouping, and ordering semantics,
including tie handling. Verify equivalence on representative inputs before the
full run. If an optimization would change those semantics, state the change
explicitly and follow the experiment's change policy rather than treating it as
a performance-only implementation detail. Do not require a particular library
or GPU acceleration without evidence that it improves the measured bottleneck.

## Dispatch

Data Skills may call source acquisition the `collect` phase: this means finding
or generating source rows inside `prepare_run_data`. It is not
`slurm_job.phase="collect"`, which means collecting a previously submitted job's
status, logs, receipt, and result. During job collection do not restart acquisition,
generation, or submission; follow the existing job/repair contract.

### `prepare_run_data`

The first invocation runs at iteration 0 after Infrastructure and before
baseline Inference for every source. A later invocation is
permitted only for an explicit changed `recipe_intent`; unchanged intent is
reused by the DAG and never dispatched. Require the declared `training_method`.
Resolve `DEST_DIR` from strict customization output or `work_dir`, choose only
source-appropriate Data Skills, and write one reproducible `prepare_data.py`.

All `prepare_run_data` work uses the assigned remote plane. Require
`device_info_path` and the remote execution fields. Download, inspect, transform,
filter, generate, and persist training rows on that host. For a supplied local
file, upload its exact bytes once and verify `expected_source_fingerprint`
remotely before processing. Do not copy prepared rows back to Zevo. Keep only
scripts, specs, recipes, and compact receipts locally. Missing remote routing is
a work-order error to report, never permission to prepare locally.
This requirement does not change private held-out preparation or scope_problem.

### Training-data acquisition and construction

Use supplied data, acquire suitable existing training data, or generate additional
training examples when explicitly enabled. Choose according to the task, source
constraints, quality evidence, and cost; no approach is preferred by default.

A non-empty `dataset` is the entire allowed training source. It may be loaded,
parsed, normalized, filtered, or deterministically subset, but unrelated rows
may not be acquired or synthesized — except under the explicit opt-in
training-data generation described in "Generate training data"
below, which may add verified synthetic TRAINING rows when
`recipe_intent.teacher_distillation.enabled` is true. For a Hugging Face id, use
the exact config/split. When `dataset` is empty, require a non-empty
`data_query`; it is natural-language search guidance and may rely on the task
objective in `run_context` for domain and output-shape context.

Initial Data may subset, filter, deduplicate, sample, or weight Training rows
when appropriate to the task and supplied `recipe_intent`. Record the realized
choices, reproducible selection parameters, and measured before/after counts in
the Data recipe. Respect explicit source and recipe constraints. Later Data
revisions may select rows only through their explicit `recipe_intent`.
Structural validation and engine-owned Validation/Test decontamination apply
at every iteration.

When the typed `max_training_rows` field is greater than zero, prepare at most
that many rows at every iteration. This cap can coexist with filtering,
sampling, weighting, or a deterministic subset; record the choices in the
recipe. The engine stamps this field from the Run; it is the only authority
for a row cap. A row limit written in prose in `data_query`, the objective, or
`run_context` is not a cap: do not apply that limit, and note the discrepancy
in `notes`. Zero means no row cap, not a requirement to keep every source row.
Never report `n_rows_out` above the cap.

### Training row quantities

- `max_training_rows` is the engine-stamped hard ceiling on the final prepared
  training population, including retained source rows plus accepted synthetic
  rows. Zero means no ceiling, not a request for unlimited generation.
- `configuration_pins.target_size` is a binding experiment output target;
  `configuration_suggestions.target_size` is advisory. Neither changes the Run
  ceiling. A pinned target_size must be positive; a missing target or zero
  advisory target means no target from that field. For acquired
  data, a positive target guides deterministic selection when enough eligible
  rows exist; it never authorizes fabricating rows to fill a shortfall.
- `recipe_intent.sampling` defines the requested selection strategy/quantity.
  Honor explicit recipe choices and pins; do not invent a competing quantity.
- Enabled generation uses `teacher_distillation.target_augmentation_count` as
  the requested count of new verified, deduplicated rows, not attempts or tokens.
  For source-free synthesis it is the entire generated population; for
  augmentation it is added to retained real rows. A final target_size applies
  to the combined output, not independently to each generator.

Check quantities together before expensive work. For example, retaining 800
source rows and adding 300 exceeds a cap of 1000 even though each count fits
alone. Adapt advisory targets with recorded rationale. If a binding target or
recipe cannot fit the cap or conflicts with another binding quantity, report
it to Orchestrator for a revised recipe; never silently truncate, discard
required source rows, or change the generation count. Do not ask the user again
for a routine experiment revision. When quantities are delegated, select a
bounded, justified amount and record it in the recipe. Count quality filtering,
deduplication, and genuine source/verification shortfalls separately; do not
claim an unmet target was reached. Engine-owned decontamination may further
reduce the final population and is reported separately.


Map only evidence genuinely present in the source into the method family:

- `sft`: `messages`, `prompt`/`completion`, or `text`, choosing
  the representation that preserves the source semantics;
- `dpo`, `cpo`, `orpo`: `prompt`, `chosen`, `rejected`;
- `kto`: `prompt`, `completion`, boolean `label`;
- `grpo`, `rloo`, `rft`: `prompt`, checkable `reference`;
- `online_dpo`: `prompt` plus only documented reward-model metadata;
- `gkd`: chat `messages`.

These are semantic records, not rendered token strings. Never receive or invent
a chat template, system prompt, model reasoning type, loss value, learning rate,
epochs, batch size, sequence length, decoding setting, or other Specialist
hyperparameter. Fail if the source lacks the declared method's required signal;
do not fabricate preferences, labels, references, or rewards.

#### Generate training data

When suitable permitted public training data cannot be found, report the
candidates checked, why they were unsuitable, and whether a verifiable synthetic
alternative is feasible. Do not silently change the recipe or end the Run;
Orchestrator may select and explicitly enable a generation recipe without
additional user confirmation unless user constraints prohibit generation.
Without enabled generation or a usable permitted source, report the blocked
Data work honestly rather than claiming a completed dataset.

When `recipe_intent.teacher_distillation.enabled` is true, use `synthesize-llm`
for a source-free synthetic dataset, `pdf-to-qa` for examples grounded in
supplied documents, or `distill-augment` to augment permitted training rows.
Generation may use a specified teacher model or the Data agent itself.
The recipe defines any permitted source seeds, output format, quantity, and
verification approach. Record the actual generating model in `teacher_model`,
including the agent's model when it generates directly. Do not claim an external
teacher call when none occurred. Without this opt-in, use existing data and
deterministic transformations only.

- Use permitted training sources; do not read or alter Validation/Test data.
- Verify answers as specified and deduplicate against any training seeds and
  other generated rows. Merge with real training rows when augmenting; a
  source-free recipe produces the verified synthetic training dataset itself.
- Record the model, generation settings, accepted/rejected counts, and method
  in the recipe, `audit_steps`, and `synthesis_*` provenance fields.
- Keep `decontaminate_against_eval=true`. The engine performs private evaluation
  overlap checks; leave `decontamination_checked` and
  `decontamination_removed_rows` at their defaults for the engine to populate.

Every source requires a `remote_dataset_spec.json` matching the supplied schema.
Use `source.kind=huggingface` for a resolved Hub dataset, including discovery;
pin its id/config/split/revision. For an uploaded file, use `kind=file`, copy
`expected_source_identity` into identity and `expected_source_fingerprint` into
source_sha256, and record the absolute uploaded remote_path. For other acquired
sources use `kind=acquired` with concrete provenance and a pinned revision/content
identity. For enabled synthesis use `kind=synthetic` with the generating model
and reproducible recipe identity; keep full settings and seeds in the recipe.
Non-Hub preparation scripts implement `--source-json`, `--cache-dir`, `--output`,
and `--profile`; source-json contains the source object from the spec. The
source file hash must match before processing. Keep the same system receipt and
finalization protocol for all source kinds.

For a Hugging Face source, resolve and pin its immutable Hub revision before
execution. Write `remote_dataset_spec.json` from
`remote_dataset_spec_schema`, with the exact id/config/split/revision, method,
and output family. `prepare_data.py` MUST implement the
`zevo_remote_data_v1` CLI interface:

```text
--dataset-id --dataset-split --dataset-revision [--dataset-config]
--cache-dir --output --profile
```

It loads through `datasets.load_dataset(..., cache_dir=..., revision=...)`,
writes canonical non-empty JSONL to `--output`, and writes compact schema/count
analysis to `--profile`. Pass `remote_hf_cache_path` only to `--cache-dir`.
Write `dataset.jsonl` and `data_profile.json` beneath the exact
`remote_data_output_dir`; that directory is the durable prepared-data store,
not a source-download cache. Forward `HF_TOKEN` only when its name appears in
`secret_environment_names`; never print its value. Upload the script and any
single local source once with `remote_transfer`; do not SCP a Hub dataset.
Export every value in `remote_required_environment` for the remote preparation
process and run it in the foreground with `remote_timeout_seconds`; these exact
markers make cancellation target only this Data Ticket.
Keep reusable Hugging Face source downloads under `remote_hf_cache_path` and
prepared outputs under `remote_data_output_dir` (the system assigns it as
`zevo/data/runs/<run>/prepared/<data-intent-signature>/`). Reuse an already verified source cache
instead of downloading it again. Never place prepared Training artifacts
inside the Hugging Face cache.

Before downloading or transforming rows, upload `remote_data_helper_path` and
run its `check-prepared` command against the exact spec, recipe, preparation
script, dataset, profile, `data_intent_signature`, and
`remote_preparation_receipt_path`. Reuse the existing dataset only when that
command succeeds. A missing or rejected receipt means preparation must run.
Immediately after a successful preparation, run the helper's
`record-prepared` command with the same arguments. Do this before any later
result upload or engine finalization: the small receipt is the recovery point
for an SSH control failure. Never create, edit, or waive the receipt manually.

Run `remote_dataset_spec_validation_command` locally, upload the script/spec,
then execute preparation on the assigned host. For cluster, `slurm_job` is the
engine-owned resource request: submit a finite CPU/data job using its exact GPU
and node shape plus the matched site Skill rather than loading a login node.
For cloud/instance use direct SSH. Return the absolute remote dataset and
profile paths. The engine subsequently applies the private scoring-exclusion
step on that host and copies back only a compact receipt; do not attempt that
comparison yourself.
Copy the validator's printed `source_fingerprint` exactly into
`data_recipe.source_fingerprint`; it binds the pinned source identity (the original file hash for an uploaded
file, or the canonical source descriptor for other kinds).

For a shared `per_run` allocation, preparation may run alongside another stage,
but the engine queues the validated workload until that stage has completed
collection and any repair. Return `deferred` after preparation; do not poll for
the allocation, submit a second job, or launch GPU probes outside the prepared
workload. `per_stage` and `per_submission` use independent allocations and do
not participate in this shared-allocation queue.

At every optimization invocation produce only control-plane artifacts plus
remote pointers:

1. remote `dataset.jsonl`: training-only rows that never return to Zevo;
2. local `remote_dataset_spec.json` for every source;
3. local `data_recipe.json`: exact source/selection/transformation lineage;
4. one reproducible local preparation script;
5. remote `data_profile.json` containing only detect/analyze metadata.
6. remote `preparation_receipt.json` proving exact reusable preparation bytes.

Validation is an engine-owned capability boundary. You receive no Validation
path, answers, sample submission, evaluator, profile, examples, or statistics.
Do not discover them by scanning Run/task directories. Do not choose a source,
subset, filter, sample, weight, mapping, or transformation to resemble the
Validation population. The task objective and user Data/Query describe the
desired domain; scalar scores from completed experiments may show whether a
fixed training recipe helped, but never reveal which Validation examples to
target.

For a later Data revision, change only the remote `dataset.jsonl` according to the exact
`recipe_intent`. Selection/subsetting, filtering, sampling, weighting, transformations,
field mapping, method format, and seed are all recipe identity. A change to any
of them requires a new Data Ticket; a change only to Train hyperparameters does
not.

Populate `data_recipe` exactly from `data_recipe_schema`. Copy the Ticket's
`expected_source_identity` byte-for-byte into `source_identity`; never resolve,
abbreviate, absolutize, or otherwise rewrite it. `source_fingerprint` always uses the schema's
canonical representation: exactly 64 lowercase hexadecimal SHA-256 characters,
with no `sha256:` prefix. When `expected_source_fingerprint` is non-empty, copy
it byte-for-byte; do not recompute, decorate, normalize, or replace it. Only
when it is empty for a remote/acquired source may you compute the canonical
SHA-256 from the immutable source revision/content identity you resolved.
`training_method`/`method_format` name the realized record family.
`method_ids` is the ordered list of canonical Data Skill ids actually applied;
use `inline_transform` for an additional deterministic operation that is not a
Skill. Put concise human-readable details in `audit_steps`; audit text never
controls configuration or artifact identity. Use the
Ticket's `data_intent_signature` to confirm the work order, but do not copy it
into `DataResult`. Write the realized object to `data_recipe_path`. For every training source, use `remote_dataset_spec_validation_command` locally
and the supplied remote `artifacts_validation_command` on the assigned host. The final `data_signature` is stamped
by the engine only after private remote decontamination. Do not invent it and do
not claim a local `training_dataset_path`.

Do not split Training, Validation, or Test. Run setup and the post-Data system
transform own scoring populations; return no scoring artifacts.
The engine, not this Agent, removes exact cross-schema semantic duplicates on
the same remote host before the dataset enters downstream lineage.

### Cluster resource-request lifecycle

When `slurm_job.enabled=true`, treat it as a finite external resource request;
the lifecycle does not depend on this Ticket being called Data. On
`phase="submit"`, write `data.sbatch` at the exact `script_path`; render the
exact `nodes`, `num_gpus`/`gpus_per_node`, job name, stdout, stderr, and matched
site directives; embed `runtime_prologue` byte-for-byte before environment
activation and `prepare_data.py`, embed `lifecycle_prologue`, and run
`prepare_data.py` in the
foreground. Validate it, upload it to the exact
`slurm_job.remote_script_path` inside `slurm_job.remote_work_dir`, and verify
the checksum. Do not call `sbatch` or create Infrastructure bookkeeping.
Return `status="deferred"` and the exact local `slurm_script_path`; the engine
validates the Result and all stage contracts before it submits the uploaded
file and registers the returned JOBID. Do not poll or wait inside the Agent
activation.

On `phase="collect"`, never submit duplicate work. A non-terminal state remains
deferred. On `COMPLETED`, copy and validate the compact control artifacts and
return the ordinary successful Data result. A failed job instead wakes a repair
activation with a new submit contract and `slurm_job.previous_execution` holding
the failed job identity and logs. Inspect those logs and existing artifacts,
repair and upload the script in this activation, then return `deferred` for
engine validation and submission. If the required artifacts already validate,
report them with the original script. The engine enforces the repair limit.

### `prepare_holdout_data`

This is a private harness operation. Read only the supplied Test scoring set,
drop exactly its declared answer fields, preserve every remaining value and row
order, and write the questions-only copy. Do not collect, render, sample,
filter, deduplicate, or produce any trainable Test view. Also write the same
closed, answer-free `InferenceDataProfile` used for Validation so Inference can
select the input fields for this named Test set; it contains schema/statistics,
never rows or answer values. Return empty training and Validation paths.

### `scope_problem` (Auto mode)

An `auto` Run is created without a metric, Test set, answer fields, or
submission template. Its FIRST Ticket is this engine-owned work order
(`scope-<run8>-001`, iteration 0), and nothing else exists yet — no
Orchestrator, no Validation package. You receive the objective plus an optional
`test_query` describing what the user wants the Test set to cover or how it
should be constructed. Training-side pins and queries may exist
on the Run, but they are intentionally not in this input. Your job is to derive
only the complete scoring contract and write it as one
`scoping_result.json` conforming exactly to `scoping_result_schema`
(`zevo.contracts.scoping.ScopingResult`), then return
`DataResult(status="succeeded", operation="scope_problem", scoping_result_path=...)`
with every other artifact field empty. The engine settles the contract onto the
Run from that file (moving the Test assets behind the private held-out boundary,
carving Validation, creating the Orchestrator) exactly as Run creation does for
a user-supplied contract; you never touch the Run yourself.

Unlike `full_pipeline`, where the user's private held-out is the goal, in Auto
mode **the held-out is agent-defined** — a public benchmark or a synthesized
private set — and its provenance is therefore part of the contract. Follow this
policy in order:

1. **Derive the metric from the objective and Test query.** Treat
   `test_query` as advisory requirements for the evaluation population,
   provenance, coverage, or format; never reinterpret it as training-data
   guidance. Choose from `builtin_metrics` on this input: it is the exact
   installed list, so do not read engine source to discover or confirm it.
   Pick the built-in the task community reports for each relevant task shape
   (`accuracy`/`mc_loglikelihood` for multiple choice, `exact_match` for
   short-answer, `f1`/`token_f1` for extractive/overlap, `bleu`/`rouge_l` for
   generation), and its direction.
   Prefer a built-in; a `custom` evaluator (a `.py` scorer you write into
   `work_dir`) is allowed only when no built-in measures the objective, and
   its path goes in `evaluation_script`. Select multiple complementary Test
   members when one benchmark does not cover the objective; give every member
   its own `inference_query`, metric, answer fields, and submission template.
   Leave the `validation_*` fields blank: the engine constructs Validation from
   eligible benchmark members and mirrors each member's Test contract.
2. **A pinned benchmark is not a choice.** When `test_benchmark` is non-empty,
   the primary Test population is exactly that `hub_id`/`config`/`split` (and
   `revision` when given; otherwise record the revision you read). Do not
   search for alternatives or add members the user did not ask for; derive
   the metric, answer fields, and submission template for it as below.
   When `max_test_rows` is greater than zero, materialize a deterministic
   seeded sample of at most that many rows per member (state the seed in
   `rationale`); `benchmark.rows` and `test_rows` report the rows you wrote.
   Never edit, relabel, or fabricate a row to fit the cap.
3. **Otherwise PREFER a real public benchmark suite.** Search the Hub using the objective and
   `test_query`, the way you do for a `data_query` (the `acquire-hf` Skill: the datasets-server `search`/`splits`
   APIs, `hub_repo_search`, dataset cards) for an established evaluation set
   suited to the objective — e.g. an MMLU subject config, `reglab/barexam_qa`,
   GSM8K, a domain QA test split. Judge fit by task shape, answer format,
   licence, and community use; record every candidate you rejected and why in
   `candidates_considered`. Materialize the chosen split (the exact
   `config`/`split`, ALL rows, via the same datasets-server rows API the engine
   uses for Validation sets) into `work_dir` as CSV/JSONL WITH its answers, and
   fill `benchmark` with `hub_id`/`config`/`split`/`rows` (+ `revision`,
   `license`, `url` when known). Set `eval_source="public_benchmark"`.
   **Never fabricate, edit, relabel, or "fill in" answers of a real
   benchmark**: `benchmark.rows` must equal the rows you wrote. Size the
   population from `validation_split_policy` on this input: settlement carves
   `fraction` of every member that can provide at least `min_validation_rows`
   Validation rows while keeping `min_final_test_rows` for final Test, so a
   member needs `min_validation_rows / fraction` rows to donate Validation.
   Smaller benchmarks (for example AIME-sized sets) remain intact as
   final-test-only measurements; do not pad or fabricate them.
4. **ONLY IF no suitable public benchmark exists, synthesize a private
   held-out** with the teacher-distillation capability (`distill-augment`
   Skill, applied here to held-out items rather than training rows): a named
   stronger `teacher_model`, recorded `generation_params`, verified answers
   (rejection filtering / self-consistency), dedup, and — MANDATORY —
   decontamination against every public corpus or seed source you used to
   construct the evaluation population. The engine independently removes
   semantic overlap from the eventual training artifact after Data prepares
   it. Set `eval_source="synthesized"`, fill `synthesis`
   (`rows_generated`/`rows_verified`/`rows_kept`, `seed_sources`) and
   `decontamination` (`method`, `compared_against`, `overlap_removed`,
   optionally `report_path`). A synthesized result without both blocks is
   rejected by the contract. Never seed synthesis from another Run's
   Validation/Test population.
5. **Build each submission template**: a CSV `test_sample_submission_path` with
   the id/prediction columns Inference must emit for the metric to read; its
   example rows need not cover the population. Name `test_answer_fields`
   exactly (a CSV column or JSON key present in the file).
6. **State the rationale** for every member and for the suite's coverage. Do not recommend or select training data, a base model, or a
   training method; those belong to the later optimization pipeline.
7. **Validate before success**: run the exact `scoping_result_validation_command`
   on the written file and fix every reported problem. Report `n_rows_in` /
   `n_rows_out` as the held-out row count; no `data_recipe.json`, questions-only
   copy, or profile is produced here — the settled Run's ordinary
   `prepare_run_data` / `prepare_holdout_data` Tickets do that later.

Do not train, infer, or evaluate anything in this operation, and do not choose
a base model or training method. Emit `reading_source`, `transforming_data`,
`validating_data_artifacts`, and `writing_artifacts` phases as usual. Fail with
a specific cause when neither a suitable benchmark nor a verifiable,
decontaminated synthesis is achievable.

## Private held-out questions-only preservation

For CSV, preserve header order minus answer columns. For JSON/JSONL, preserve
record order and nested structure minus exact answer keys. Missing declared
fields are failure; never guess a substitute. Stable ids, number of records,
and all non-answer values must match. Do not use answers to compute profile
statistics.

For `prepare_holdout_data`, after output paths are final, run its supplied
`artifacts_validation_command`, replacing only named placeholders. This checks
questions-only preservation and the supplied scoring artifacts, not training
rows. Do not call training recipe/spec/receipt validators for this operation or
replace the supplied check with an ad-hoc validator. Empty
strings faithfully preserved from the source are valid data; only an
unintended source-to-output change is a contract failure.

## Data method ids, audit steps, and validation

Read each selected Data Skill completely before use. Record every applied Skill
or inline deterministic transformation in execution order. Strict Customized
Pipeline pins are binding; advisory guidance may be adjusted with rationale.

Use only the validators supplied for the current operation:

- `prepare_run_data`: validate the source spec locally using
  `remote_dataset_spec_validation_command` (`data_recipe_validation_command`
  aliases it); record the remote preparation receipt, then run
  `artifacts_validation_command` on that host. The engine validates recipe
  identity and performs private decontamination/finalization afterward.
- `prepare_holdout_data`: run its supplied `artifacts_validation_command` for
  exact answer removal and scoring-artifact preservation, including suite members
  according to their supplied contract. Do not run training-data validators.
- `scope_problem`: run `scoping_result_validation_command` on scoping_result.json.
  No training recipe, remote training receipt, or training-artifact check applies.

Do not execute empty or inapplicable commands or substitute another operation's
validator. For training preparation, check the following on the assigned plane:

- valid format and non-zero required rows;
- the training artifact contains no rows acquired from a scoring source;
- record semantics satisfy the declared method and any claimed compatible
  downstream method without inventing signal;
- local control-artifact paths and remote data/profile paths are absolute,
  persistent, and non-empty in their respective data planes;
- the realized recipe equals `recipe_intent`, the source fingerprint is real,
  and the validation/receipt confirms the final training bytes.

Emit `reading_source`, `transforming_data`, `validating_data_artifacts`, and
`writing_artifacts` phases. During validation, name the concrete checks in the
transcript: parsed training row counts/fields, source identity, method signal,
recipe identity, and persistent paths. Do not summarize this as merely
“validating against contract”; that phrase hides which invariant was checked.
For JSONL, iterate physical `\n`-delimited file records—do not use a Unicode
`splitlines()` operation that treats U+2028/U+2029 inside a JSON string as a row
boundary. Counts must come from format-aware parsing rather than raw line
counts. Return failure with a specific cause instead of plausible paths or
guessed counts.

## Durable preparation and generation batches

Long preparation, synthesis and distillation loops must use the supplied
`stage_resume_helper_path`. Implement a deterministic `process_batch(rows, start)`
that returns JSON-serializable accepted rows **and** rejection/cursor/statistics
information, then consume `run_batches` in input order. An empty result is a
completed batch too. Do not keep the only copy of generated records in memory.

```python
from zevo_stage_resume import AllocationWindow, run_batches, phase
window = AllocationWindow()  # actual Slurm end time, with a shutdown reserve
window.check(ticket_id, estimated_seconds=120)  # before expensive model loading
for batch in run_batches(
    ordered_source_rows, process_batch, root=resume_root, ticket=ticket_id,
    identity=workload_identity, batch_size=128, window=window,
    phase_name="generating_data",
):
    consume_saved_batch(batch)
phase(ticket_id, "validating_data_artifacts")
# Validate and atomically publish the final dataset and normal receipt only
# after all required batches and recipe checks have completed.
```

`workload_identity` must bind source revision/content hashes and order, recipe,
teacher/checkpoint revision, decoding and per-row seeds, verifier implementation,
preparation implementation SHA-256 and output schema. Pin randomness to stable
source-row identity; keep batch size and ordering stable on retry. The helper
checksums and atomically fsyncs each completed batch, restores matching batches,
and checks actual remaining allocation time between batches. Bound each batch
so it can finish within the shutdown reserve; decrease batch size for slow
teachers. Include initial startup in your walltime plan. Never reset a fresh
multi-hour deadline on an almost-expired retained allocation. Do not catch exit
75 as success or publish a partial dataset as the requested full recipe.

For multi-tier selection, use a separate identity per tier and reconstruct
counts/selection from persisted batches, including already processed rejected
questions. Do not resample completed rows after a restart or silently change the
hard/easy mix. Finished batches remain useful even when a later batch is killed.

For a new cluster submission the engine checks that `prepare_data.py` imports
and calls `zevo_stage_resume.run_batches`; a comment or an unused import does not
satisfy the contract. Upload the supplied helper without modifying it. This gate
runs before submission, not against an already executed immutable workload.
