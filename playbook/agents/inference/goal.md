Produce one real prediction per supplied questions-only row under one durable,
comparable Inference configuration.

## Baseline selection

1. Read only the Task Objective in `run_context.task_objective`,
   `inference_data_profile_path`, `sample_submission`, model/runtime evidence,
   pins, suggestions, and relevant Run-local memory.
2. Verify model/tokenizer/template/backend compatibility.
3. Choose the intended serving interface from the Objective first, then choose
   all remaining unpinned values, including an explicit `greedy` or `sampling`
   decoding strategy. Classify the exact selected model/checkpoint as
   `thinking` or `non_thinking` from its verified tokenizer/template behavior.
   Use the matching template: a thinking model emits its reasoning span before
   the answer, while a non-thinking model uses ordinary rendering with no
   thinking markup. A conversational or instruction-following Objective uses a
   stable chat contract from Baseline through Train and later evaluation, even
   when the untrained base tokenizer does not ship a native chat template. Chat
   framing uses `You are a helpful assistant.` when no system prompt is supplied.
4. Verify the tokenizer ids for EOS and every single-token response/turn
   terminator used by the selected template. Record them in `stop_token_ids`;
   never rely on a special-token string surviving detokenization.
5. Record every realized field, the exact supported `template_kwargs`, each
   suggestion decision, and a synthetic
   `prompt_example` in `inference_config.yaml` before generation. The example
   must show input placeholders, pre-template messages, the actual system
   prompt, and the exact rendered text produced with those kwargs and handed to
   the model.
6. Generate predictions with that exact file and write one termination record
   per generation request containing its backend finish reason, stop reason,
   and generated-token count.

## Reuse

1. Load the supplied YAML.
2. Verify that the current checkpoint can execute it.
3. Use its prompt, template identity, `template_kwargs`, input mapping, parsing, decoding,
   generation backend, and seed exactly.
4. Verify the exact supplied YAML bytes and report its path; do not create a
   variant. The runner derives the hash from the artifact.

Success requires valid config, script, predictions, generation diagnostics,
row/request counts, output columns, and artifact copy-back. Preserve malformed model replies and report parse
diagnostics; never repair them into favorable answers.

Runtime errors that are corrected through reflection and may recur in this Run
should be written to Inference memory. Memory may help implementation but may
not change the reused measurement contract.
