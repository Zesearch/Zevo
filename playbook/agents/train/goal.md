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
   Empty/zero/missing hyperparameter suggestions delegate those values; method
   selection follows the separate priority below.
4. Resolve the method in order: user pin, explicit Orchestrator selection,
   inherited `active_training_method`, then Train choice when the activation
   contract delegates ownership to `worker` (including standalone/freeform).
   An omitted method does not discard an existing branch. If none applies,
   report an incomplete work order rather than silently choosing a method.
   Never change methods merely for diversity. If the method cannot run faithfully,
   report the evidence
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

By default, leave Trainer validation loss/reward and loss-based early stopping
off. They are optional; enable them only when useful for the chosen experiment
and record the realized settings and permitted evaluation-data source.
System-owned mid-training Validation, when enabled, owns intermediate scoring,
continuation decisions, and best-checkpoint selection. Store corrected, reusable runtime pitfalls in
Run-local Train memory. Return failure when the selected configuration cannot
be executed faithfully; do not mutate it silently after launch.
