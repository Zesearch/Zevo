---
name: lora-sft
description: "Train the selected causal LM with supervised fine-tuning through a LoRA adapter. Use for ordinary target-bearing SFT data when a small portable checkpoint or low parameter-memory cost is preferred; supports chat messages, prompt-completion, and plain-text continuation according to prompt_framing."
---

# LoRA SFT

Use `trl.SFTTrainer` with `peft.LoraConfig` to optimize only adapter parameters.
Return an adapter that Inference loads on top of the unchanged `base_model`.
API reference: https://huggingface.co/docs/trl/sft_trainer

## Preconditions and data

Require the shape selected by `prompt_framing`:

| framing family | required row | required `loss_contract.target_scope` |
|---|---|---|
| `chat` / `chat:<id>` | `{"messages":[...]}` | explicit labels for each assistant turn |
| `completion` | `{"prompt":...,"completion":...}` | `completion_only_loss=True` |
| `text` | `{"text":...}` | all tokens; set neither mask |

Validate every row needed by the run, not only the first. Validate the optional
evaluation set against the same shape. Fail on mixed or incompatible records;
do not convert them inside Train.

For chat loss, expand each conversation into one example per assistant turn.
Render the preceding context with `add_generation_prompt=True` and the context
plus target assistant with `add_generation_prompt=False`, using the unchanged
template and exact `template_kwargs` loaded from baseline
`inference_config.yaml`. Never derive or hard-code those kwargs from the
thinking label. Require the prompt ids to be an
exact prefix of the full ids; mask that prefix with `-100` and supervise only
the remaining target. A non-thinking model uses the ordinary target; a thinking
model supervises its reasoning content before the response. Fail on a prefix
mismatch, empty target, or target truncation. Do
not patch Jinja, substitute an instruct template, or hand-write wrapper tokens.

## Implement

Follow the Train Agent's shared script, remote execution, telemetry, validation,
and output contract already loaded in your instructions. In the method block:

```python
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

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
    # binding ticket values + resolved shared config
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

Verify and report `trl.__version__` and the required `SFTConfig` fields before
launch. Fail rather than disabling a requested loss mask on an incompatible
installation.

Resolve target modules from the actual model architecture. Do not assume every
model uses `q_proj/k_proj/v_proj/o_proj`; prefer PEFT-supported `"all-linear"`
when appropriate or enumerate verified linear module names.

Save with `trainer.save_model(model_dir)` and save the tokenizer there. Verify
`adapter_config.json` plus adapter weights exist. Do not merge or save the frozen
base weights.

## Implement and report the fixed objective

Read the required `loss_type` from the complete
`loss_contract.objective_config`. When selecting, `nll` is the recommended
start; never read it from `method_config` or change it during execution.

The stored loss contract owns the loss scope; it is not a template inference or
experiment knob. Report the one
applicable mask and the actual rendered prompt structure in `__CONFIG__`.

Choose `SFTConfig.loss_type` deliberately:

| value | use |
|---|---|
| `chunked_nll` | preferred when supported; NLL with lower peak activation memory |
| `nll` | compatibility fallback, including Liger configurations incompatible with chunked NLL |
| `dft` | only for an explicitly intended Dynamic Fine-Tuning experiment |

Validate the selected `max_seq_len` against measured token lengths and target preservation. Keep
`packing=False` unless packing is deliberate and compatible with the loss mask.
If packing is enabled, report `packing`, `packing_strategy`, and `padding_free`.

Use the ticket's exact LoRA rank/alpha, learning rate, epochs, batch size, and
maximum length. Gradient checkpointing remains an implementation choice only
when it preserves that recorded configuration.

Record the exact values read from `SFTConfig` and `LoraConfig`, including target
modules and trainable parameter count, in `train_config.yaml`. Its
`loss_contract` must contain the actual mask flag (`assistant_only_loss`,
`completion_only_loss`, or all-token loss) plus the resolved `loss_type`;
`TrainResult` does not repeat this configuration.

Fail rather than claiming success if no adapter parameters are trainable, the
target modules do not exist, target tokens are truncated away, or the saved
directory contains full base weights instead of an adapter.
