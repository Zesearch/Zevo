---
name: full-sft
description: "Train every parameter of the selected causal LM with supervised fine-tuning. Use only when the model and sequence/batch configuration fit the assigned GPU strategy and a complete standalone checkpoint is required; accepts chat, prompt-completion, or plain-text SFT data according to prompt_framing."
---

# Full-parameter SFT

Use `trl.SFTTrainer` without PEFT. Optimize every trainable model parameter and
produce a standalone Hugging Face model directory.
API reference: https://huggingface.co/docs/trl/sft_trainer

## Preconditions and data

Apply the same framing/shape contract as LoRA SFT:

| framing family | required row | required `loss_contract.target_scope` |
|---|---|---|
| `chat` / `chat:<id>` | `{"messages":[...]}` | explicit labels for each assistant turn |
| `completion` | `{"prompt":...,"completion":...}` | `completion_only_loss=True` |
| `text` | `{"text":...}` | all tokens; set neither mask |

Validate all rows and the optional evaluation dataset. For chat, expand each
conversation into one example per assistant turn. For an assistant at index
`i`, render `messages[:i]` with `add_generation_prompt=True` and render
`messages[:i+1]` with `add_generation_prompt=False`, using the exact template
and `template_kwargs` loaded from baseline `inference_config.yaml`. Never derive
or hard-code those kwargs from the thinking label. Require the first
token sequence to be an exact prefix of the second; labels are `-100` over that
prefix and the remaining target token ids thereafter. Earlier assistant turns
remain context and are masked. This makes every trained reply begin from the
same generation prefix used by Inference. A non-thinking model uses the
ordinary target; a thinking model supervises its reasoning content before the
response. Fail on a prefix mismatch, empty target, or target truncation. Do not
patch Jinja, switch templates, or rely on generation
annotations to guess the loss mask.

Before training, prove that the model fits. Count actual parameters and include:

- weights, gradients, optimizer/master states;
- activations at the resolved batch and sequence length;
- attention memory when no memory-efficient kernel is active; and
- safety headroom for allocator fragmentation, evaluation, and saving.

Use a measured probe or conservative estimate, not the model name. If one full
replica fits per GPU, DDP is valid. If it does not, use a correctly configured
FSDP/ZeRO strategy; DDP does not pool model memory. Fail before step 1 when no
valid strategy fits, and recommend `lora_sft` in the error.

## Implement

Follow the Train Agent's shared platform contract already loaded in your
instructions. In the method block:

```python
from trl import SFTConfig, SFTTrainer

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
    callbacks=[ZevoTrainerTelemetryCallback(ticket_id)],
)
```

Verify and report `trl.__version__` and the required `SFTConfig` fields before
launch. Fail rather than dropping a loss mask or objective on an incompatible
installation.

Do not pass `peft_config`, quantize trainable weights, or freeze model layers.
Assert that the trainable parameter count equals the intended full model count.

Save with `trainer.save_model(model_dir)` using the distributed strategy's safe
save path, then save the tokenizer. Verify `config.json` and complete model
weights/shards exist and that the directory reloads as a standalone model.

## Implement and report the fixed objective

Read the required `loss_type` from the complete
`loss_contract.objective_config`. When selecting, `nll` is the recommended
start; never read it from `method_config` or change it during execution.

The stored loss contract owns the loss scope. Choose and report the
semantics-preserving TRL implementation:

| value | use |
|---|---|
| `chunked_nll` | preferred when supported to reduce peak activation memory |
| `nll` | compatibility fallback |
| `dft` | only for an explicit Dynamic Fine-Tuning experiment |

Validate the selected `max_seq_len` against actual rendered lengths and fail if it
silently removes target tokens. Keep packing off unless the recorded experiment
explicitly measures it and it is compatible with the loss scope.

Honor the exact ticket learning rate, epochs, and batch size. Memory-saving
implementation choices such as gradient checkpointing may change only when
they preserve the recorded mathematical configuration.

Report the loss scope, `loss_type`, full/trainable parameter counts, precision,
distributed strategy, memory estimate/probe, and any adjustment made to an
batch size. Never alter a ticket value to make the run
fit; fail clearly instead.
