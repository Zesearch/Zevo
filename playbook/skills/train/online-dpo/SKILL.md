---
name: online-dpo
description: "Run experimental Online DPO by sampling fresh completion pairs and ranking them with an explicit Hugging Face reward model. Use only for prompt-only data when method_config.reward_model is supplied and on-policy preferences are required; hosted judges and non-HF reward resources are unsupported."
---

# Online DPO

Use experimental `OnlineDPOTrainer`. It generates two current-policy responses,
ranks them with an external reward signal, and applies a DPO-style objective to
the resulting online pair.
API reference: https://huggingface.co/docs/trl/online_dpo_trainer

## Preconditions and data

Require every training and evaluation row to contain only a valid `prompt` plus
any explicitly documented metadata consumed by the reward. Pipeline prompts are
conversational message lists; consistent standard strings are acceptable in
direct mode.

Require `method_config.reward_model` as an explicit Hugging Face `owner/model`
repository id for a compatible sequence-classification reward model. Local
paths, URLs, uploads, Registry references, arbitrary Python reward callables,
and hosted judges are not supported. Do not invent a judge or use the policy's
own likelihood as preference evidence. A missing reward is a hard failure, not
permission to switch to DPO/GRPO.

Resolve the id through `huggingface_hub.snapshot_download(repo_id=reward_model)`
into the remote Hugging Face cache before loading it. This makes the declared
Hub source explicit and prevents a same-named relative directory from being
treated as the reward model. A missing/gated/inaccessible repository is a hard
failure and must name the id.

## Implement

Follow the Train Agent's shared platform contract. Verify/report the installed
TRL version, then:

```python
from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer

args = OnlineDPOConfig(output_dir=OUTPUT_DIR, ...)
trainer = OnlineDPOTrainer(
    model=model,
    ref_model=None,
    reward_funcs=reward,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    peft_config=peft_config_or_none,
    callbacks=[ZevoTrainerTelemetryCallback(ticket_id)],
)
```

Default to PEFT unless `method_config.use_peft=false`. Respect `generation_backend`
exactly: `hf` means `use_vllm=False`; `vllm` means `use_vllm=True` with the
supported integration mode. For a single shared GPU allocation, use colocate
only after checking memory and set a conservative utilization. If the requested
backend cannot initialize, fail rather than silently changing the run-level
generation_backend.

Experimental import/config failure must name `trl.__version__`; never substitute
offline DPO.

## Implement and report the fixed objective

Read objective and rollout fields from the complete
`loss_contract.objective_config`. The validated `train_config.yaml` supplies every
selected value; never read them from `method_config`, invent one, or change them.

Use `loss_type="sigmoid"` unless the experiment explicitly requests IPO.
Resolve/report:

- loss type and the complete `beta` schedule;
- reward model id and any `reward_weights`;
- `max_new_tokens`, combined `max_length`, sampling temperature/top-p/top-k;
- `missing_eos_penalty` when truncation without EOS should be penalized;
- generation_backend, vLLM mode/utilization when used; and
- PEFT/reference strategy.

When selecting, start consideration with learning rate `5e-7`, `max_new_tokens=64`,
`max_seq_len=512`, temperature `0.9`, and `beta=0.1`. During execution, validate compatibility
with the installed trainer, but do not replace the stored values.

Monitor the trainer's reward and preference metrics in addition to loss. Save
and verify the adapter/full artifact selected by `use_peft`.
