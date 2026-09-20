---
name: sft
description: "Supervised fine-tune the selected causal LM on chat, prompt-completion, or plain-text targets. Full-parameter training is the default; enable a LoRA adapter only when the Ticket or an explicit user instruction sets method_config.use_peft=true."
---

# Supervised Fine-Tuning

Use `trl.SFTTrainer` for supervised causal-language-model training. The single
method has two explicit execution modes:

- `method_config.use_peft=false` (default): optimize every model parameter and
  save a standalone Hugging Face model;
- `method_config.use_peft=true`: optimize a LoRA adapter and save only the
  adapter plus tokenizer.

Never infer PEFT from model size, GPU count, memory pressure, or a preference
for a smaller artifact. It requires a Ticket binding or explicit user
instruction. API reference: https://huggingface.co/docs/trl/sft_trainer

For compatibility, a resumed historical `full_sft` Ticket means
`use_peft=false`, and a historical `lora_sft` Ticket means `use_peft=true`.
Preserve its stored method id while finishing that Ticket; new work uses `sft`.

## Preconditions and data

Require the shape selected by `prompt_framing`:

| framing family | required row | required `loss_contract.target_scope` |
|---|---|---|
| `chat` / `chat:<id>` | `{"messages":[...]}` | explicit labels for each assistant turn |
| `completion` | `{"prompt":...,"completion":...}` | `completion_only_loss=True` |
| `text` | `{"text":...}` | all tokens; set neither mask |

Validate every row needed by the run and the optional evaluation dataset. Fail
on mixed or incompatible records; do not convert them inside Train.

For chat loss, expand each conversation into one example per assistant turn.
Render the preceding context with `add_generation_prompt=True` and the context
plus target assistant with `add_generation_prompt=False`, using the unchanged
template and exact `template_kwargs` loaded from baseline
`inference_config.yaml`. Require the prompt ids to be an exact prefix of the
full ids, mask that prefix with `-100`, and supervise only the remaining target.
Earlier assistant turns remain masked context. Fail on a prefix mismatch, empty
target, or target truncation. Do not patch Jinja, replace the instruct template,
or hand-write wrapper tokens.

## Prove the selected mode fits

For full-parameter training, count actual parameters and include weights,
gradients, optimizer/master states, activations, attention memory, evaluation,
saving, and allocator headroom. Use a measured probe or conservative estimate.
DDP is valid only when one full replica fits per GPU. Otherwise use a correctly
configured FSDP or DeepSpeed ZeRO strategy. Fail before step 1 when the assigned
resources cannot support the requested full-parameter mode; do not silently
switch to LoRA.

For PEFT, the base weights remain frozen, but the base model, activations,
adapter state, evaluation, and saving still must fit the selected strategy.

## Implement

Follow the Train Agent's shared platform contract. Resolve
`use_peft = method_config.use_peft` before constructing the trainer:

```python
from trl import SFTConfig, SFTTrainer

peft_config = None
if use_peft:
    from peft import LoraConfig
    peft_config = LoraConfig(
        r=resolved_r,
        lora_alpha=resolved_alpha,
        lora_dropout=resolved_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=resolved_target_modules,
    )

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=resolved_max_length,
    assistant_only_loss=False,  # labels are already explicit
    **loss_scope,
)
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    peft_config=peft_config,
    callbacks=[ZevoTrainerTelemetryCallback(ticket_id)],
)
```

Verify and report `trl.__version__` and the required `SFTConfig` fields. When
PEFT is active, verify `peft.__version__`, resolve target modules from the
actual architecture, and prefer PEFT-supported `"all-linear"` when appropriate.
Do not assume every model uses the same projection names.

When PEFT is inactive, do not quantize trainable weights or freeze model layers;
assert that the trainable parameter count equals the intended full model count.

Use the supplied system checkpoint helper for the final transaction. A
full-parameter run must atomically commit and reload a complete standalone
model. A PEFT run must atomically commit and reload `adapter_config.json` and
adapter weights without copying the frozen base weights. Save the tokenizer in
either mode.

## Implement and report the fixed objective

Read `loss_type` from `loss_contract.objective_config`; start from `nll` and do
not move it into `method_config`.

| value | use |
|---|---|
| `chunked_nll` | preferred when supported to reduce peak activation memory |
| `nll` | compatibility fallback |
| `dft` | only for an explicit Dynamic Fine-Tuning experiment |

Validate `max_seq_len` against actual rendered lengths and target preservation.
Keep packing off unless it is deliberate and compatible with the loss mask.
Honor every resolved Ticket value. Record `use_peft`, loss scope, loss type,
full/trainable parameter counts, precision, distributed strategy, and the exact
LoRA configuration when active. Fail if the saved artifact type does not match
`method_config.use_peft`.
