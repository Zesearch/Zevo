---
name: cpo
description: "Run reference-free preference optimization with CPOTrainer, including CPO, SimPO, CPO-SimPO, or AlphaPO objectives. Use when genuine prompt/chosen/rejected pairs are available and a reference-model-free alternative to DPO is desired; use SimPO variants when length-normalized reward or an explicit target margin is the experiment."
---

# CPO

Use the current experimental TRL CPO implementation. It ranks `chosen` above
`rejected` without loading a reference model; the CPO objective can also include
behavior cloning on the chosen response.
API reference: https://huggingface.co/docs/trl/cpo_trainer

## Preconditions and data

Require every pipeline row to contain conversational:

```json
{"prompt":[{"role":"user","content":"..."}],
 "chosen":[{"role":"assistant","content":"..."}],
 "rejected":[{"role":"assistant","content":"..."}]}
```

Direct calls may use TRL's equivalent standard string format, but one dataset
must not mix representations. Require a meaningful preference contrast. Fail on
missing/empty/equal completions; never manufacture a rejected answer. Validate
the optional evaluation set identically.

## Implement

Follow the Train Agent's shared platform contract already in your instructions.
Verify and report `trl.__version__`, because this trainer is experimental, then:

```python
from trl.experimental.cpo import CPOConfig, CPOTrainer

args = CPOConfig(output_dir=OUTPUT_DIR, ...)
trainer = CPOTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    peft_config=peft_config_or_none,
    callbacks=[ZevoTrainerTelemetryCallback(ticket_id)],
)
```

Do not load or pass a reference model. In this system, default to a LoRA adapter
unless `method_config.use_peft` is explicitly false. Resolve LoRA fields from
the ticket/Skill and save an adapter; a deliberate full-parameter run must pass
the same memory guard as `full_sft` and save complete weights.

If the experimental import or requested config field is absent in the installed
TRL version, fail with the version and missing symbol. Do not silently replace
CPO with DPO or SFT.

## Implement and report the fixed objective

Read objective fields from the complete `loss_contract.objective_config`.
The validated `train_config.yaml` supplies the complete mapping; never read
them from `method_config`, invent a missing value, or change them in place.

| recipe | required config | use |
|---|---|---|
| CPO | `loss_type="sigmoid", cpo_alpha=1.0` | default reference-free preference + chosen-response BC |
| SimPO | `loss_type="simpo", cpo_alpha=0.0, simpo_gamma=<value>` | length-normalized implicit reward with target margin |
| CPO-SimPO | `loss_type="simpo", cpo_alpha>0` | SimPO ranking plus BC regularization |
| AlphaPO | `loss_type="alphapo"` plus resolved `alpha`/margin | explicit AlphaPO reward shaping experiment |
| IPO/hinge | `loss_type="ipo"` or `"hinge"` | only with a stated reason for identity or hard-margin behavior |

Plain CPO is the canonical starting recipe. A later iteration may select a
variant only when that iteration's `train_config.yaml` records `loss_contract`
as an explicit axis. Honor the concrete `beta`, `learning_rate`, and
`max_seq_len` already stored in YAML; never resolve them again while executing.

Record `loss_type`, `beta`, `cpo_alpha`, and every applicable
`simpo_gamma`/`alpha` value in `train_config.yaml` and `__CONFIG__`. Monitor
`nll_loss`, `rewards/accuracies`, and `rewards/margins`. Verify the saved
artifact type matches `use_peft`; `TrainResult` does not repeat objective fields.
