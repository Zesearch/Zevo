---
name: rloo
description: "Run REINFORCE Leave-One-Out with a deterministic reward over prompt/reference data. Use when each prompt has a checkable reference and critic-free online RL is desired, using other sampled completions as each sample's baseline; requires at least two generations and preferably enough diversity for a stable baseline."
---

# RLOO

Use `trl.RLOOTrainer` to sample multiple completions, score them, and subtract the
mean reward of the other samples as a leave-one-out baseline. It is online RL
without a value/critic model.
API reference: https://huggingface.co/docs/trl/rloo_trainer

## Preconditions and reward

Require every pipeline row to contain conversational `prompt` and a non-empty
`reference`. This is intentionally stricter than TRL's prompt-only minimum: the
reference is the authorized programmatic reward for this system. Validate the
evaluation set identically.

Implement and unit-check a deterministic task-compatible reward. Accept
`completions`, `reference`, and `**kwargs`; handle malformed and conversational
outputs. Use boxed parsing only for explicitly boxed-answer tasks. Do not invent
a semantic judge or partial-credit rule.

Fail when the comparator is unjustified, `num_generations < 2`, or a probe shows
no usable reward variation.

## Implement

Follow the Train Agent's shared platform contract:

```python
from trl import RLOOConfig, RLOOTrainer

args = RLOOConfig(output_dir=OUTPUT_DIR, ...)
trainer = RLOOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    peft_config=peft_config_or_none,
    callbacks=[ZevoTrainerTelemetryCallback(ticket_id)],
)
```

Default to PEFT unless `method_config.use_peft=false`. Honor `generation_backend`
exactly: HF generation when `hf`; supported vLLM integration when `vllm`.
Colocate only after a memory check and never silently fall back to another
backend.

## Implement and report the fixed objective

Read objective and rollout fields from the complete
`loss_contract.objective_config`. The validated `train_config.yaml` supplies every
selected value; never read them from `method_config`, invent one, or change them.

RLOO has a fixed leave-one-out policy-gradient family. Report the shaping
configuration rather than inventing a `loss_type`:

- `beta` KL coefficient;
- `num_generations` and `normalize_advantages`;
- clipping bounds and optional reward clipping;
- completion length, sampling configuration, and optimizer iterations;
- reward definition/weights; and
- generation_backend/vLLM and PEFT/reference strategy.

When selecting the realized config, start consideration with `beta=0.05`, `num_generations=4`, one optimizer
iteration, learning rate `1e-6`, and masked truncated completions. Validate that
`num_generations >= 2`; do not replace values after writing `train_config.yaml`.

Report dataset prompts consumed as `n_examples`; do not multiply by sampled
completions. Keep `final_loss` as the real trainer loss and report mean reward,
reward variance, and zero-variance groups separately. Verify the adapter/full
policy artifact selected by `use_peft`.
