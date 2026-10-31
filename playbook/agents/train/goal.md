Produce one primary final checkpoint descended from the explicitly selected
parent and one auditable `train_config.yaml`. Optionally retain a small,
hypothesis-driven set of intermediate model-weight checkpoints as future Run
branch points; this is never a requirement to save every epoch or step.

1. Load baseline `inference_config.yaml` and, for a checkpoint branch, the
   selected parent's `train_config.yaml`.
2. Select one coherent experimental direction. Several coupled changes are
   allowed only when they serve that same direction and are explained.
3. Resolve values in priority order: strict pins, compatible suggestions,
   selected-parent and Run-history evidence/memory, method Skill, measured runtime/data.
   Empty/zero/missing suggestions delegate the choice.
4. In a pipeline, execute the compatible method selected by the Orchestrator
   or pinned by the user. A standalone/freeform request without an Orchestrator
   selection delegates the method choice to Train. A pipeline request that omits
   the method also delegates that choice. Retain or change methods according to that experiment decision, never
   for diversity alone. If the method cannot run faithfully, report the evidence
   so the Orchestrator can revise the experiment. Choose unpinned method-specific
   values and full-parameter versus PEFT execution from evidence, resources, and
   expected benefit; explain the tradeoff. GKD teacher and Online DPO reward model
   references must be Hugging Face `owner/model` ids.
5. Derive the semantic loss contract from method and prompt framing, then fill
   only method-valid objective values.
6. Copy the complete Baseline prompt/template identity and `template_kwargs`,
   verify the exact rendered/token prefix and loss-mask boundary, add a synthetic
   `training_data_example` showing the method-shaped record, exact rendered
   sequence(s), and loss/context spans, then write and validate
   `train_config.yaml` before training.
7. Preflight the actual data/method/model/GPU/dependencies, train in the
   foreground, and verify the expected checkpoint interface. Select startup
   checks from changed execution paths and actual risk. Reuse applicable prior
   validation evidence. Establish working full-state recovery when continuation
   requires it; a weights-only export is insufficient. Record checks and results,
   then continue the planned execution without repeating unrelated verification.

Only after the complete training configuration is fixed, map the raw records
at `validation_dataset_path` using `validation_answer_fields` and the same
prompt/template renderer used for Training. Use the result only as trainer
evaluation data; it must never enter the optimization loss or influence the
training source/configuration choice. Store corrected, reusable runtime pitfalls in
Run-local Train memory. Return failure when the selected configuration cannot
be executed faithfully; do not mutate it silently after launch.
