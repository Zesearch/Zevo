---
name: dpo
description: "Run offline Direct Preference Optimization against a reference policy. Use when genuine prompt/chosen/rejected pairs are available and alignment should learn their relative ordering without an online reward model or rollout loop; prefer this stable baseline before specialized preference losses."
---

# DPO

Use `trl.DPOTrainer` to increase the chosen-vs-rejected log-probability margin
relative to the initial policy. DPO consumes fixed preference pairs and performs
no online generation.
API reference: https://huggingface.co/docs/trl/dpo_trainer

## Preconditions and data

Require every pipeline row to contain conversational `prompt`, `chosen`, and
`rejected` fields, with chosen/rejected assistant continuations. TRL also accepts
a consistent standard string representation in direct mode. Reject missing,
empty, identical, or trivially fabricated contrasts. Validate evaluation rows
against the same representation.

## Implement

Follow the Train Agent's shared platform contract:

```python
from trl import DPOConfig, DPOTrainer

args = DPOConfig(output_dir=OUTPUT_DIR, ...)
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    peft_config=peft_config_or_none,
    callbacks=[ZevoTrainerTelemetryCallback(ticket_id)],
)
```

With PEFT and `ref_model=None`, TRL uses the initial base policy as the implicit
reference. Default to PEFT in this system unless `method_config.use_peft` is
explicitly false. A full-parameter DPO run needs memory for both policy and
reference behavior; use precomputed reference log-probabilities only when the
installed TRL version and dataset type support it.

Never combine incompatible reference modes, including synchronized references
with PEFT-without-a-standalone-reference or with precomputed reference
log-probabilities.

## Implement and report the fixed objective

Read objective fields from the complete `loss_contract.objective_config`.
The validated `train_config.yaml` supplies the complete mapping; do not read these
fields from `method_config`, invent a missing execution value, or change them.

The canonical starting value is `loss_type="sigmoid"`. Select another current
TRL loss only while freezing an iteration, before its Data Ticket, from that
iteration's explicit hypothesis:

- `ipo` for identity-transform regularization;
- `hinge` for a hard margin;
- `robust` with non-zero `label_smoothing` for known label noise;
- `exo_pair` with valid smoothing for reverse-KL behavior;
- `sigmoid_norm` when length bias is the measured problem; or
- a documented multi-loss list only when all weights are explicit.

Do not expose the full installed loss menu as an invitation to guess. Check that
the selected value exists in the installed TRL version. A later
iteration may change loss type as its explicit `loss_contract` axis; never
change it inside an executing iteration.

When selecting the iteration config, `beta=0.1` is the recommended start and Train must choose a
concrete PEFT/full-training learning rate in `training_config`. During
execution, honor the exact values recorded in `train_config.yaml`.
`max_seq_len` is also resolved before execution and must preserve the combined
prompt/completion signal.

Report the exact `loss_type` list/string, weights, `beta`, smoothing, divergence,
reference strategy, PEFT mode, and max length. Monitor `rewards/accuracies` and
`rewards/margins`; a final loss alone does not show whether preferences separate.
Save and verify the adapter or full model selected by `use_peft`.
