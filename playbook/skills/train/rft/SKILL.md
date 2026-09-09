---
name: rft
description: "Run one rejection-sampling fine-tuning cycle: sample multiple current-policy completions for prompt/reference data, keep only deterministically correct outputs, then SFT on the survivors. Use when answers are checkable and the base model already succeeds often enough to create a non-empty high-quality supervised set."
---

# Rejection Sampling Fine-Tuning

RFT is a generate-filter-SFT procedure, not a dedicated TRL trainer. Execute one
cycle in this ticket: sample current-policy outputs, retain correct ones, then
train `SFTTrainer` on the retained set. Let later Train iterations decide
whether to repeat; do not run an open-ended ReST loop inside one ticket.
SFT API reference: https://huggingface.co/docs/trl/sft_trainer

## Preconditions and reward

Require every training row to contain conversational `prompt` and a non-empty,
checkable `reference`. Validate the optional validation set identically.

Define a deterministic answer parser/comparator and test it on known matches,
mismatches, malformed output, and empty output. Default to explicit exact
task-compatible normalization. Use
`loss_contract.objective_config.answer_parser="boxed"` only
when the task/data actually declares boxed answers. Do not invent partial credit
or a judge.

Probe a small prompt sample before committing to the full generation cost. Fail
if the base policy produces no verified positives; training on an empty or
unverified set is not RFT.

## Generate and filter

For each prompt, sample `num_samples` independent completions at non-zero
temperature using the exact run framing. Honor `generation_backend` exactly (`hf` or the
supported vLLM path).

Do not force chain-of-thought or `\\boxed{}` universally. Follow the frozen
Baseline Inference model classification and template exactly: generate
reasoning traces for a thinking model and preserve ordinary response format for
a non-thinking model. Never insert hidden references into prompts.

Keep a completion only when the deterministic comparator accepts it. Deduplicate
identical normalized responses and cap survivors per prompt so easy prompts do
not dominate. Record source prompts, generated samples, prompts with at least
one survivor, total survivors, and rejection reasons.

Convert only survivors into chat SFT rows by appending the sampled assistant
completion to the original prompt. Never include `reference` as model-visible
text.

## Train and save

Follow the Train Agent's shared platform contract. Train the derived set with
`SFTTrainer`, `SFTConfig`, the system-provided
`ZevoTrainerTelemetryCallback(ticket_id)`, assistant-only loss, and the actual
chat template. Default to a LoRA adapter unless `method_config.use_peft=false`.
Resolve SFT configuration as in `lora_sft`/`full_sft`; do not invoke those as
additional methods.

The original validation set is not an SFT loss dataset because it has no target
completion. Do not pass `{prompt, reference}` rows to `SFTTrainer` and do not
generate/filter them into training rows. If evaluation is requested, use them
only for clearly labeled generation metrics such as pass@k before/after; keep
them completely outside optimizer input.

## Implement and report the fixed objective

Read every field listed below from the complete
`loss_contract.objective_config`. The validated `train_config.yaml` supplies every
selected value; never read them from `method_config`, invent one, or change them.

Report both data-selection and SFT choices:

- `num_samples`, temperature and sampling parameters;
- answer parser/comparator, max survivors per prompt, and deduplication;
- source/survivor counts and survival rate;
- generation_backend/backend;
- SFT loss scope/type, max length, learning rate, epochs; and
- PEFT configuration or explicit full-parameter mode.

When selecting, start consideration with `num_samples=4`, temperature `0.8`, and one survivor
per prompt unless the stored contract explicitly selects another coherent
configuration. `TrainResult.n_examples` is the number of
surviving SFT rows actually consumed. `final_loss` is the SFT trainer loss, not
pass@k or filtering accuracy. Verify the adapter/full artifact selected by
`use_peft`.
