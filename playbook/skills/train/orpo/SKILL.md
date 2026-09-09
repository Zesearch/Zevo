---
name: orpo
description: "Run reference-free Odds Ratio Preference Optimization, combining chosen-response NLL and rejected-response odds-ratio alignment in one stage. Use when genuine prompt/chosen/rejected pairs are available and a single-stage alternative to separate SFT plus reference-based DPO is desired."
---

# ORPO

Use experimental `ORPOTrainer`. ORPO combines NLL on the chosen response with a
relative odds-ratio penalty; it does not load a reference model and does not add
a separate SFT stage.
API reference: https://huggingface.co/docs/trl/orpo_trainer

## Preconditions and data

Require the same canonical preference rows as DPO/CPO:
`{"prompt":[...],"chosen":[assistant...],"rejected":[assistant...]}`.
Validate all training and evaluation rows. Fail on absent, empty, equal, or
fabricated contrasts.

## Implement

Follow the Train Agent's shared platform contract. Verify and report the TRL
version before using this experimental API:

```python
from trl.experimental.orpo import ORPOConfig, ORPOTrainer

args = ORPOConfig(output_dir=OUTPUT_DIR, ...)
trainer = ORPOTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    peft_config=peft_config_or_none,
    callbacks=[ZevoTrainerTelemetryCallback(ticket_id)],
)
```

Do not load a reference model. Default to PEFT unless
`method_config.use_peft=false`; deliberate full-parameter training must satisfy
the full-SFT memory guard. If the experimental symbols are unavailable, fail
with the installed version rather than switching objectives.

## Implement and report the fixed objective

Read objective fields from the complete `loss_contract.objective_config`.
The validated `train_config.yaml` supplies complete selected values; never read them from
`method_config`, invent a missing value, or change them.

ORPO has one objective family. Report it as
`chosen NLL + odds-ratio, beta=<resolved beta>`. `beta` controls the odds-ratio
term relative to chosen-response NLL. Use the ticket's already-resolved beta,
learning rate, and maximum sequence length exactly.

Report `beta`, max length, PEFT mode, and all resolved optimizer/training values.
Monitor `nll_loss`, `log_odds_ratio`, `rewards/accuracies`, and
`rewards/margins`. A rising preference margin with unstable NLL is not a clean
success; record it in `notes`. Verify the selected adapter/full checkpoint.
