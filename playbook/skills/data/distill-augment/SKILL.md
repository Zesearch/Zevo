---
name: distill-augment
description: "Generate additional training examples using a specified teacher model or the Data agent when recipe_intent.teacher_distillation.enabled is true. Verify, deduplicate, and record provenance; leave private evaluation overlap checks to the engine."
---

# Generate training data

For `prepare_run_data`, use this Skill only when
`recipe_intent.teacher_distillation.enabled` is true. Generation is an optional
way to augment permitted training sources, not a default or preferred strategy.
The Data agent may generate directly or call the specified teacher model.

## Workflow

1. Read the recipe's `teacher_model`, `output_format`,
   `target_augmentation_count`, `verify_answers`, and `generation_params`.
   `teacher_model` records the actual generating model, including the agent's own
   model for direct generation. Do not imply a model call that did not occur.
2. Use permitted training seeds and the task objective. Follow the requested
   answer format; `irac_rationale` applies only when explicitly selected, while
   `instruction_distillation` produces instruction/response examples.
3. Generate in bounded, resumable batches using the supplied stage-resume helper.
   When `verify_answers` is enabled, retain only examples passing the specified
   correctness or consistency checks. Record rejections and deduplicate against
   the training seeds and other generated examples.
4. Count target_augmentation_count as new verified, deduplicated rows. Before
   generation, reconcile retained real rows plus that target with any final
   target_size and max_training_rows using the Data platform's Training row
   quantities rules. Do not silently change a binding recipe to fit the cap.
   Merge accepted examples with the retained real rows in the required method
   format. Validate the dataset and record reproducible source/model/settings
   provenance and accepted/rejected counts.

## Data boundaries and reporting

Do not read, synthesize, or alter Validation/Test populations in this operation.
Keep `decontaminate_against_eval=true`; the engine performs private overlap checks.
Do not perform that comparison yourself or claim its outcome.

Report `distill_augment` in `method_ids`, and populate `synthesis_teacher_model`,
`synthesis_output_format`, and `synthesis_generated_rows`. Leave
`decontamination_checked` and `decontamination_removed_rows` at their defaults
for the engine. Summarize generation, verification, and training-only dedup in
`audit_steps`.
