---
name: synthesize-llm
description: "Author a bounded synthetic training dataset when no dataset was supplied, the operation is prepare_run_data, data_query is concrete, and no clean public dataset fits or synthesis was explicitly requested. Generate faithful supervision directly as the Data Agent; do not call an external model API unless a directive explicitly authorizes it."
---

# Synthesize training records

Use this Skill only for the `collect` phase when there is no provided source.

## Workflow

1. Require a non-empty `data_query` or directed instruction. Interpret it with
   the task objective in `run_context`, then extract the task,
   domain, language, difficulty range, answer form, and approximate size.
2. Confirm the selected training method's required supervision can be authored:
   targets for SFT, genuine contrasts for preference methods, boolean feedback
   for KTO, checkable references for GRPO/RLOO/RFT, or prompts for online DPO.
3. Define a coverage plan across topics, formats, and difficulty before writing
   rows. Avoid generating one template repeatedly.
4. Author records in manageable batches directly in the required target shape.
   Do not call an external generation API unless it is authorized: either a user
   directive, or `recipe_intent.teacher_distillation.enabled` is `true`. When
   teacher distillation is enabled, use the `distill-augment` Skill for the
   teacher call, reject-filtering, decontamination against the frozen
   Validation/Test populations, and provenance recording — never synthesize or
   alter held-out data, and augment TRAINING data only.
5. Track normalized prompts and remove exact/near duplicates as an explicit
   filter operation.
6. Stop at positive `target_size`. When it is zero, choose a bounded size based
   on distinct coverage you can actually validate; do not claim thousands of
   examples that were not written.
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
`target_size` in `audit_steps`; keep only the concise outcome in `notes`.
