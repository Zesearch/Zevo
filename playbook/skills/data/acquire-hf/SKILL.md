---
name: acquire-hf
description: "Find and load a matching public HuggingFace training dataset when no dataset was supplied, the operation is prepare_run_data, and data_query names a task or domain likely covered by a public corpus. Do not use for a user-provided HuggingFace id; load that exact id as the provided source instead."
---

# Acquire a HuggingFace dataset

Use this Skill only for the `collect` phase. Compose it with an applicable
reformat Skill when the downloaded rows are not already in the required
training shape.

## Workflow

1. Require `dataset==""`, `operation=="prepare_run_data"`, and a non-empty
   `data_query`. Interpret it together with the task objective in `run_context`.
2. Search for a dataset that matches the requested task, domain, language, and
   answer type. Prefer a documented training split with a schema you can map
   without guessing.
3. Inspect the dataset card/configs/splits and sample records before selecting
   it. Do not select a merely adjacent dataset to avoid returning empty work.
4. Load a training split locally with `datasets.load_dataset`. Never run this
   acquisition on a remote GPU machine.
5. Record the exact dataset id, config, split/slice, and parsed row count.
6. Pass the loaded records through the Data Agent's required target renderer and
   optional training-only filter. Write final output under `DEST_DIR`.

An empty ticket `dataset_split` does not authorize validation or test data. Use
the selected dataset's training split. If the acquisition instruction explicitly
names a split/config, use it exactly and fail if it does not exist.

## Selection boundaries

- Never use a benchmark validation or test split merely because it is easier to
  load; it may be the population used to score the run.
- Never replace a non-empty `dataset`; a Hub id there is already the user's
  chosen source.
- Respect a strict `acquire_hf` directive. If no clean match exists, fail with
  the candidates checked and why they were unsuitable.
- In autonomous mode, when no clean public match exists, stop using this Skill
  and switch to `synthesize-llm`; record that fallback explicitly.
- Apply deterministic sampling only when the corpus exceeds a justified
  `target_size`. Keep the original parsed count as `n_rows_in`.

## Validate and report

- Parse representative and boundary records before full conversion.
- Verify the final JSONL against the Data Agent's declared method shape; prompt
  rendering and hyperparameters are not Data inputs.
- Report `acquire_hf` in `method_ids`; put the resolved repository/revision and
  materialization details in `audit_steps`.
- Name the exact source and selection in `notes`; never report a candidate you
  inspected but did not use.
