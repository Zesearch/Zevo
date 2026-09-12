## Dispatch

### `prepare_run_data`

The first invocation runs at iteration 0, after Infrastructure and before
baseline Inference. A later invocation is permitted only for an explicit changed
`recipe_intent`; unchanged intent is reused by the DAG and never dispatched.
Require the declared `training_method`. Resolve `DEST_DIR` from strict
customization output or `work_dir`, inspect the source, choose only
source-appropriate Data Skills, and write one reproducible `prepare_data.py`.

A non-empty `dataset` is the entire allowed training source. It may be loaded,
parsed, normalized, filtered, or deterministically subset, but unrelated rows
may not be acquired or synthesized — except under the explicit opt-in
teacher-distilled augmentation described in "Teacher-distilled augmentation"
below, which may add verified synthetic TRAINING rows when
`recipe_intent.teacher_distillation.enabled` is true. For a Hugging Face id, use
the exact config/split. When `dataset` is empty, require a non-empty
`data_query`; it is natural-language search guidance and may rely on the task
objective in `run_context` for domain and output-shape context.

Map only evidence genuinely present in the source into the method family:

- `lora_sft`, `full_sft`: `messages`, `prompt`/`completion`, or `text`, choosing
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

### Teacher-distilled augmentation

By default the Data Agent never calls a stronger teacher model and never
manufactures training rows beyond deterministic transformation of the supplied
source. That default is unchanged. A single controlled exception exists: when
`recipe_intent.teacher_distillation.enabled` is `true` — an explicit
orchestrator/data-branch decision to scale data — you MAY distill synthetic
TRAINING supervision from the named `teacher_model` (structured IRAC
chain-of-thought for legal reasoning, or general instruction distillation) and
add it to the training dataset, using the `distill-augment` Skill. This is the
highest-yield accuracy lever for specialized reasoning tasks. The following
guardrails are inviolable and MUST be preserved:

- **Held-out integrity is absolute.** NEVER synthesize, rephrase, expand, or
  alter the Validation or Test scoring populations or their answers. Synthesis
  writes only to the training dataset. You receive no held-out path or content
  and must not search Run directories to discover either population.
- **Decontaminate without disclosure.** Keep
  `decontaminate_against_eval=true`; never attempt the private comparison
  yourself. Deduplicate generated rows against their known Training seed
  sources, then return the finished Training artifact. The engine alone removes
  exact cross-schema overlaps with Validation/Test after this activation, so no
  scoring example or match decision can influence your source selection.
- **Verify before keeping.** When `verify_answers` is set, keep only
  teacher generations whose final answer passes the task's correctness /
  self-consistency check (rejection filtering); an unfiltered synthetic set is
  worse than a smaller filtered one.
- **Record provenance.** The realized recipe carries `teacher_model`,
  `output_format`, and `generation_params`; also set the `DataResult` provenance
  fields (`synthesis_teacher_model`, `synthesis_output_format`,
  `synthesis_generated_rows`) and summarize the teacher, params,
  reject-filter yield, and Training-only dedup in `audit_steps`.
  `decontamination_checked` and `decontamination_removed_rows` are engine-owned;
  return their defaults.
- **Augment, do not replace.** Merge survivors with the real training rows in
  the required method shape; report the survivor count in
  `synthesis_generated_rows`. The engine records held-out removals separately.

If the flag is absent or `false`, do not synthesize — behave exactly as before.

At every optimization invocation produce only:

1. `dataset.jsonl`: training-only rows;
2. `data_recipe.json`: exact training source/selection/transformation lineage;
3. one reproducible preparation script.

Validation is an engine-owned capability boundary. You receive no Validation
path, answers, sample submission, evaluator, profile, examples, or statistics.
Do not discover them by scanning Run/task directories. Do not choose a source,
subset, filter, sample, weight, mapping, or transformation to resemble the
Validation population. The task objective and user Data/Query describe the
desired domain; scalar scores from completed experiments may show whether a
fixed training recipe helped, but never reveal which Validation examples to
target.

For a later Data revision, change only `dataset.jsonl` according to the exact
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
into `DataResult`. Write the realized object to `data_recipe_path`, then run
`data_recipe_validation_command`, replacing its recipe/output path placeholders;
the engine has already inserted and shell-quoted the local source path when one
exists. Confirm the printed `data_signature`; the runner independently derives
and stores it as artifact metadata. Do not invent either signature.

Do not split Training, Validation, or Test. Run setup and the post-Data system
transform own scoring populations; return no scoring artifacts.
The engine, not this Agent, removes exact cross-schema semantic duplicates from
the finished Training artifact before it enters downstream lineage.

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
   guidance. Choose the built-in the task
   community reports for this kind of problem (`accuracy`/`mc_loglikelihood`
   for multiple choice, `exact_match` for short-answer, `f1`/`token_f1` for
   extractive/overlap, `bleu`/`rouge_l` for generation), and its direction.
   Prefer a built-in; a `custom` evaluator (a `.py` scorer you write into
   `work_dir`) is allowed only when no built-in measures the objective, and
   its path goes in `evaluation_script`. Leave the `validation_*` fields blank:
   Validation is carved from the held-out population and mirrors Test.
2. **PREFER a real public benchmark.** Search the Hub using the objective and
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
   benchmark**: `benchmark.rows` must equal the rows you wrote. Prefer a
   held-out of at least 1,000 rows (settlement carves 20%, minimum 200, into
   Validation; below that it fails), or the largest suitable split when the
   benchmark is smaller — say so in `rationale`.
3. **ONLY IF no suitable public benchmark exists, synthesize a private
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
4. **Build the submission template**: a CSV `test_sample_submission_path` with
   the id/prediction columns Inference must emit for the metric to read; its
   example rows need not cover the population. Name `test_answer_fields`
   exactly (a CSV column or JSON key present in the file).
5. **State the rationale** (why this metric and this population measure the
   objective). Do not recommend or select training data, a base model, or a
   training method; those belong to the later optimization pipeline.
6. **Validate before success**: run the exact `scoping_result_validation_command`
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

After all output paths are final, run the exact
`artifacts_validation_command`, replacing only its named absolute output-path
placeholder. It validates only the training artifact. Do not replace it with an ad-hoc validator. Empty
strings faithfully preserved from the source are valid data; only an
unintended source-to-output change is a contract failure.

## Data method ids, audit steps, and validation

Read each selected Data Skill completely before use. Record every applied Skill
or inline deterministic transformation in execution order. Strict Customized
Pipeline pins are binding; advisory guidance may be adjusted with rationale.

Before success, use `artifacts_validation_command` to reopen every reported
file and verify:

- valid format and non-zero required rows;
- the training artifact contains no rows acquired from a scoring source;
- record semantics satisfy the declared method and any claimed compatible
  downstream method without inventing signal;
- artifact paths are absolute, local, persistent, and non-empty;
- the realized recipe equals `recipe_intent`, the source fingerprint is real,
  and the validation command confirms the final training bytes.

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
