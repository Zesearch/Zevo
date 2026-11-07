---
name: synthesize-llm
description: "Author a bounded synthetic training dataset when no dataset was supplied, the operation is prepare_run_data, and recipe_intent.teacher_distillation.enabled is true. Use for an explicitly selected synthetic-data recipe, including when suitable permitted public data could not be found; honor user source and generation restrictions."
---

# Synthesize training records

Use this Skill for source acquisition (`collect` in the data recipe) when
there is no provided source. It is not used for Slurm job-result collection.

## Workflow

1. Require `recipe_intent.teacher_distillation.enabled=true` and a non-empty
   `data_query` or directed instruction. A failed public-data search alone does
   not enable generation; Orchestrator selects the recipe. Interpret it with
   the task objective in `run_context`, then extract the task,
   domain, language, difficulty range, answer form, and approximate size.
2. Confirm the selected training method's required supervision can be authored:
   targets for SFT, genuine contrasts for preference methods, boolean feedback
   for KTO, checkable references for GRPO/RLOO/RFT, or prompts for online DPO.
3. Define a coverage plan across topics, formats, and difficulty before writing
   rows. Avoid generating one template repeatedly.
4. Generate records in manageable batches in the required target shape using
   the recipe's specified model (the Data agent or a teacher). Record the actual
   generating model, settings, accepted/rejected counts, and verification
   evidence in the recipe and `synthesis_*` provenance fields. Follow the
   recipe's verification approach and honor user restrictions. Never read,
   synthesize from, or alter Validation/Test populations. Keep
   `decontaminate_against_eval=true`; private overlap checks belong to the
   engine. Leave their result fields at the engine-populated defaults.
5. Track normalized prompts and remove exact/near duplicates as an explicit
   filter operation.
6. Use the enabled recipe's positive `target_augmentation_count` as the verified
   generation target. Reconcile target_size and max_training_rows first using
   the Data platform's Training row quantities rules; zero target_size does
   not override the recipe's generation count. Report verification shortfalls
   honestly; do not claim examples that were not written.
7. Write `<DEST_DIR>/dataset.jsonl`, parse every record, validate the method
   shape, and measure the final count.

## Quality boundaries

- Keep every record on the requested task. Do not substitute an easier or more
  familiar domain.
- Vary surface form and reasoning structure without changing label semantics.
- For factual or high-stakes domains, prefer checkable, conservative content and
  avoid unsupported specificity.
- Make preference contrasts substantively different in quality, not trivial
  corruptions or formatting changes.
- Keep KTO classes sufficiently represented and document their counts.
- Ensure every RL/RFT reference is unambiguous and compatible with the intended
  reward check.
- Use boxed answers only when the task's evaluation representation calls for
  them.

## Report

Set `n_rows_in=0` and `n_rows_out` to the parsed final record count. Report
`synthesize_llm` in `method_ids`, and summarize the coverage plan,
class/difficulty distribution, deduplication, and any shortfall from
the recipe generation target and any final `target_size` in `audit_steps`; keep only the concise outcome in `notes`.
