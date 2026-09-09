---
name: kto
description: "Run Kahneman-Tversky Optimization on unpaired desirable/undesirable completions. Use when each prompt-completion example has a genuine boolean quality label but chosen/rejected pairs are unavailable; works best with both label classes and requires special batching for KL-based KTO."
---

# KTO

Use `trl.KTOTrainer` for unpaired binary feedback. Each example independently
marks one completion desirable or undesirable; do not synthesize pairs.

## Preconditions and data

Require every pipeline row to contain conversational `prompt`, assistant
`completion`, and boolean `label`. Require actual booleans, not truthy strings.
Validate the evaluation set the same way. Count both classes and fail when the
data cannot support the intended objective; at minimum, report severe imbalance.

Set tokenizer padding to the left and ensure a valid pad token, as required by
the trainer.

## Implement

Follow the Train Agent's shared platform contract:

```python
from trl import KTOConfig, KTOTrainer

args = KTOConfig(output_dir=OUTPUT_DIR, ...)
trainer = KTOTrainer(
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

With `ref_model=None`, the initial policy supplies the reference behavior.
Default to PEFT unless `method_config.use_peft=false`. Do not combine PEFT with
`use_liger_kernel=True`; do not combine synchronized reference updates with
PEFT-without-a-standalone-reference or precomputed reference log-probabilities.

## Implement and report the fixed objective

Read objective fields from the complete `loss_contract.objective_config`.
The validated `train_config.yaml` supplies complete selected values; never read them from
`method_config`, derive replacements from runtime data, or change them.

| loss | use |
|---|---|
| `kto` | default prospect-theoretic objective with an in-batch KL estimate |
| `apo_zero_unpaired` | only when the hypothesis is to raise desirable and lower undesirable likelihood without a KL estimate |

For KL-based KTO, set `train_sampling_strategy="sequential"`, require
per-device batch size greater than 1, and prefer at least 4 per step with an
effective batch of 16–128. Gradient accumulation does not repair a poor
per-step KL pairing.

Honor the exact ticket values, including `beta`, learning rate, and both class
weights. When selecting, start consideration at `beta=0.1`, learning rate `1e-6`, and
unit class weights. If observed class counts justify reweighting, that choice
must be declared as a later iteration's loss-contract axis before execution.

Report loss type, beta, both class counts and weights, per-step/effective batch,
sampling strategy, reference strategy, PEFT mode, and max length. Monitor
chosen/rejected rewards and margins. Verify the adapter or full checkpoint
selected by `use_peft`.
