---
name: passthrough-jsonl
description: "Validate and materialize a provided JSONL dataset that already matches the semantic record family required by training_method. Use only when every record is already in target shape and no semantic field mapping is needed; otherwise use reformat-jsonl."
---

# Pass through target-shaped JSONL

This Skill performs no semantic transformation. It validates the complete
source and materializes the canonical training file under `DEST_DIR`.

## Workflow

1. Require a provided local JSONL source. A JSON array is not pass-through
   JSONL; use `reformat-jsonl` to materialize it one record per line.
2. Derive the exact required keys and value types from `training_method` using
   the Data Agent platform contract. No prompt or hyperparameter input is needed.
3. Parse every non-empty line as one JSON object. Fail on malformed JSON,
   heterogeneous shape, missing required values, or incompatible record types.
4. Copy the file to `<DEST_DIR>/dataset.jsonl`. Use a symlink only when the
   source is persistent and guaranteed to remain readable by downstream stages.
5. Parse the destination again and confirm byte-safe materialization and equal
   record counts.

## Boundaries

- Do not rewrite answers, add system messages, normalize text, filter rows, or
  repair malformed records under this Skill.
- Do not accept chat records for completion/text framing or accept SFT records
  for preference/RL methods merely because they are valid JSON.
- Do not use physical line count without parsing; blank or corrupt lines must
  not be counted as records.
- Handle scoring-set derivatives separately according to
  `scoring_outputs_needed`; they are never copied into the training dataset.

## Report

Set `n_rows_in == n_rows_out` to the parsed record count. Include
`passthrough_jsonl` in `method_ids`. State the verified record shape in
`audit_steps`; keep only the concise outcome in `notes`.
