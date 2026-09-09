---
name: gkd
description: "Distill a specified stronger teacher into the selected student with TRL Generalized Knowledge Distillation, mixing offline and student-generated sequences under a generalized JSD objective. Use only when an explicit teacher with compatible tokenizer/vocabulary is available and both models fit the assigned GPU strategy."
---

# Generalized Knowledge Distillation

Use experimental `GKDTrainer` to optimize the ticket's `base_model` as the
student against a frozen teacher. GKD may mix dataset sequences with student
on-policy sequences and interpolates forward/reverse KL through generalized JSD.
API reference: https://huggingface.co/docs/trl/gkd_trainer

## Preconditions and data

Require every training and evaluation row to contain chat `messages` in the
canonical SFT form. GKD in this pipeline does not accept `completion` or `text`
framing. Fail on an incompatible run framing instead of converting it.

Require `method_config.teacher_model`: an explicit Hugging Face `owner/model`
repository id. Local paths, URLs, uploads, and Registry references are not
supported by this pipeline.
Do not silently use the student as its own teacher and do not choose an unrelated
teacher opportunistically. Confirm that teacher and student token distributions
are comparable—normally the same tokenizer/vocabulary family—and that the
teacher is actually distinct and plausibly stronger for the task.

Budget memory for student training state, frozen teacher, activations, and
generation. Fail before training if the pair cannot fit the assigned strategy.

Resolve the id through `huggingface_hub.snapshot_download(repo_id=teacher_model)`
into the remote Hugging Face cache before loading it. This makes the declared
Hub source explicit and prevents a same-named relative directory from being
treated as the teacher. A missing/gated/inaccessible repository is a hard
failure and must name the id.

## Implement

Follow the Train Agent's shared platform contract. Verify/report
`trl.__version__`, then:

```python
from trl.experimental.gkd import GKDConfig, GKDTrainer

args = GKDConfig(
    output_dir=OUTPUT_DIR,
    teacher_model_name_or_path=teacher_model,
    ...,
)
trainer = GKDTrainer(
    model=student,
    teacher_model=teacher,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    peft_config=peft_config_or_none,
    callbacks=[ZevoTrainerTelemetryCallback(ticket_id)],
)
```

Load the teacher frozen and in evaluation mode. Default the student to PEFT
unless `method_config.use_peft=false`; a full-parameter student needs the
full-SFT memory guard in addition to teacher memory. The run's `generation_backend` does
not imply vLLM here: current GKDConfig has no shared `use_vllm` contract, so use
the trainer's supported generation path.

If the experimental import/config is unavailable, fail with the installed TRL
version. Do not replace GKD with SFT or select another teacher.

## Implement and report the fixed objective

Read objective and distillation fields from the complete
`loss_contract.objective_config`. The validated `train_config.yaml` supplies every
selected value; never read them from `method_config`, invent one, or change them.

Resolve and report both axes:

- `lmbda`: fraction of student-generated on-policy sequences (`0` offline,
  `0.5` mixed default, `1` fully on-policy);
- `beta`: generalized-JSD interpolation (`0` forward-KL side, `0.5` balanced
  default, `1` reverse-KL side).

Use `seq_kd=True` only for an explicit sequence-level teacher-generation
experiment. Report `temperature`, `max_new_tokens`, SFT loss scope, max length,
teacher id, PEFT mode, and both model sizes. Preserve chat framing and use
assistant-only loss when the actual training template supports it.

Save only the trained student artifact and tokenizer—never the teacher. Verify
the adapter/full checkpoint selected by `use_peft` and name the teacher in
`train_config.yaml.method_selection_rationale` and `__CONFIG__`.
