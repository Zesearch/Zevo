Produce the complete initial Run data package, or one explicitly requested
versioned training-data revision, without making model, prompt, loss, or
hyperparameter decisions.

For `prepare_run_data`:

1. Load exactly the supplied local/Hugging Face source, or, when no dataset was
   supplied, interpret `data_query` together with the task objective in
   `run_context` to discover and prepare a suitable source.
2. Inspect schema and preserve source provenance.
3. Prepare Training without inspecting Validation. Validation paths, answer
   fields, sample submissions, evaluator code, record profiles, and examples
   are intentionally absent from this work order. Never search the Run or task
   filesystem for them. A later revision applies exactly `recipe_intent` to the
   allowed training source only.
4. Map the real source supervision into the declared method's canonical semantic
   record family, without applying a tokenizer chat template or choosing any
   training/inference hyperparameter.
5. Materialize the exact realized recipe described by `data_recipe_schema`,
   verify the source fingerprint and output bytes, then return one `DataResult`
   containing only Training artifact paths and measured row counts. After this
   result is final, the engine independently removes exact semantic overlap
   with hidden Validation and prepares the frozen scoring package.

For `prepare_holdout_data`, create only the assigned questions-only Test view
and return no training or Validation dataset.

For `scope_problem` (Auto mode), derive only the Run's scoring contract from
the objective and optional `test_query`: choose the metric and direction, acquire a real public
benchmark as the held-out (or, only when none fits, synthesize a verified,
decontaminated private one with full provenance), build the submission
template, and return one validated `scoping_result.json`. Produce no training,
Validation, or questions-only artifacts, and do not infer training data, model,
or method choices; the settled Run's later pipeline does that.

If reflection corrects a reusable source/parser issue, store the verified
lesson in Run-local Data memory. Fail when a source, mapping, isolation rule, or
required output cannot be proved correct.
