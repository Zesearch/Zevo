---
name: distill-augment
description: "Scale TRAINING data with teacher-distilled synthetic augmentation ONLY when recipe_intent.teacher_distillation.enabled is true (an explicit orchestrator/data-branch decision to scale data). Distill reasoning supervision (e.g. IRAC chain-of-thought for legal, or general instruction distillation) from the named stronger teacher, reject-filter to verified-correct rows, decontaminate against the frozen Validation/Test populations, and record provenance. Never synthesize or alter held-out Validation/Test data or their answers. Do nothing when the flag is absent or false."
---

# Teacher-distilled training-data augmentation

Use this Skill for the `prepare_run_data` operation ONLY when
`recipe_intent.teacher_distillation.enabled` is `true`. This is the controlled,
opt-in reversal of the blanket "no stronger teacher" prohibition: distilling
structured chain-of-thought rationales from a stronger teacher and SFT'ing on
them is the single biggest documented accuracy lever for specialized reasoning
tasks (the IRAC-distillation result on the Multi-state Bar Exam, arXiv
2504.04945). It augments the existing real training source; it does not replace
it, and it never touches held-out data.

If the flag is absent or `false`, this Skill does nothing — the run keeps its
existing (non-synthesized) behavior. Do not invent an augmentation the intent
did not authorize.

## Inviolable guardrails (never weaken these)

1. **Held-out integrity is absolute.** NEVER synthesize, rephrase, expand, or
   otherwise alter the Validation or Test scoring populations or their answers.
   Generation writes only to the TRAINING dataset. The frozen Validation
   artifacts are returned byte-for-byte unchanged.
2. **Decontaminate every generated row** against the frozen Validation/Test
   populations (`recipe_intent.teacher_distillation.decontaminate_against_eval`
   is always `true` when enabled — the contract forbids disabling it). Drop any
   generated row whose question/answer overlaps a scoring item by exact match or
   n-gram/embedding near-duplication. This matters especially for MMLU-derived
   evals, which are widely contaminated; an un-decontaminated "gain" can be pure
   leakage.
3. **Record provenance.** The realized recipe carries `teacher_model`,
   `output_format`, and `generation_params`; echo the teacher model, output
   format, generated-row count, and decontamination outcome in the
   `DataResult` provenance fields and in `audit_steps`.
4. **Augment TRAINING only.** `n_rows_out` includes the surviving synthetic
   rows; report them in `synthesis_generated_rows` and the removed count in
   `decontamination_removed_rows`.

## Workflow

1. Read `recipe_intent.teacher_distillation`: `teacher_model` (required),
   `output_format` (`irac_rationale` or `instruction_distillation`),
   `target_augmentation_count` (>= 1), `verify_answers`, and
   `generation_params`. Interpret the task with `run_context`.
2. Seed generation from the real training source (and `data_query` context),
   not from the scoring populations. For `irac_rationale`, prompt the teacher to
   reason in Issue/Rule/Application/Conclusion then emit the final answer in the
   task's required place; for `instruction_distillation`, distill faithful
   instruction/response supervision in the declared method's record family.
3. Generate in bounded batches up to `target_augmentation_count`. Vary topics,
   surface form, and difficulty; do not repeat one template.
4. **Reject-filter (when `verify_answers`):** keep only generations whose final
   answer passes the task's correctness / self-consistency check. Drop the rest.
   An unfiltered synthetic set is worse than a smaller filtered one.
5. **Decontaminate:** remove every surviving row that overlaps the frozen
   Validation/Test populations. Count removals.
6. **Dedup:** remove exact/near duplicates among generated rows and against the
   existing real training rows.
7. Merge survivors into `<DEST_DIR>/dataset.jsonl` in the exact required method
   shape, alongside the real rows. Parse every record and validate the method
   shape.

## Report

- Report `distill_augment` in `method_ids` (compose with the reformat Skills
  used to shape the merged rows).
- Set `synthesis_teacher_model`, `synthesis_output_format`,
  `synthesis_generated_rows` (surviving rows added), `decontamination_checked=true`,
  and `decontamination_removed_rows` in `DataResult`.
- Summarize the coverage plan, teacher/params, reject-filter yield, dedup, and
  decontamination overlap findings in `audit_steps`; keep only the concise
  outcome in `notes`.
