---
name: grpo
description: "Run Group Relative Policy Optimization with a deterministic reward over prompt/reference data. Use when each prompt has a trustworthy checkable reference, multiple sampled completions can produce useful within-group reward variance, and online RL is justified over simpler SFT or preference optimization."
---

# GRPO

Use `trl.GRPOTrainer` to sample a group of completions per prompt, score them,
and optimize group-relative advantages without a critic network.
API reference: https://huggingface.co/docs/trl/grpo_trainer

## Preconditions and reward

Require every pipeline row to contain conversational `prompt` and a non-empty,
checkable `reference`. Validate the optional evaluation set identically. TRL
only requires `prompt`, but this Skill requires `reference` because it defines
the pipeline's authorized reward signal.

Implement a deterministic reward whose signature accepts `completions`, the
dataset's `reference`, and `**kwargs`. Parse conversational and standard
completions correctly. Use task-compatible normalization; use boxed-answer
extraction only when the data/task explicitly uses that format. Do not invent
partial credit or compare raw strings when the task defines another equivalence.
Unit-check the reward on known match, mismatch, malformed, and empty outputs
before training.

Fail if the reference is ambiguous, the comparator cannot be justified, or a
probe batch yields no useful reward variance. Do not reward formatting alone
unless formatting is the declared objective.

## Implement

Follow the Train Agent's shared platform contract:

```python
from trl import GRPOConfig, GRPOTrainer

args = GRPOConfig(output_dir=OUTPUT_DIR, ...)
trainer = GRPOTrainer(
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

Default to PEFT unless `method_config.use_peft=false`. Respect `generation_backend`:
`hf` sets `use_vllm=False`; `vllm` sets it true with a supported mode. On the
shared single-node allocation, colocate is appropriate only when policy,
training state, KV cache, and rollout batch fit together. Fail rather than
silently changing a requested backend.

## Implement and report the fixed objective

Read objective and rollout fields from the complete
`loss_contract.objective_config`. The validated `train_config.yaml` supplies the
complete selected values; never read them from `method_config` or resolve
them again inside this iteration.

Prefer `loss_type="dapo"` unless a specific aggregation hypothesis says
otherwise. Relevant alternatives include:

- `dr_grpo` for a constant completion-length denominator;
- `grpo` only to reproduce the original sequence-normalized formulation; and
- `bnpo` only when local-batch normalization is intentional.

Keep `importance_sampling_level="token"` by default; use `"sequence"` when
sequence-level ratios are the explicit stability hypothesis. Set
`mask_truncated_completions=True` when truncated responses cannot receive a
valid reward.

Resolve/report at least: reward definition and weights, `num_generations`,
`beta`, loss type, importance-sampling level, clip values, reward scaling,
completion length, temperature, iterations, generation_backend/vLLM configuration, and
PEFT mode. When selecting, start consideration at `num_generations=4`, `beta=0`,
learning rate `1e-6`, and one optimizer iteration per generation batch, then
state the exact selected values. During execution, validate all divisibility
constraints imposed by the installed trainer without changing these values.

`TrainResult.n_examples` is the number of dataset prompts consumed, not prompts
times generations. `final_loss` is the trainer's real final training loss; put
mean reward, reward variance, zero-variance group rate, and completion metrics in
progress/config/notes rather than substituting reward for loss. Verify the saved
adapter/full policy artifact.
