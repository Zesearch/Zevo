---
name: reformat-jsonl
description: "Parse provided JSONL or JSON-array records that do not already match the required training shape, then map them into the semantic record family selected by training_method. Use for structured JSON field conversion; use passthrough-jsonl only when every JSONL record already matches exactly."
---

# Reformat JSON or JSONL training data

## Workflow

1. Detect the actual serialization:
   - JSONL: parse each non-empty line as one object;
   - JSON array: load the array and treat each element as one record.
2. Require object records and measure `n_rows_in` from parsed objects. Never use
   physical line count for a JSON array, pretty-printed JSON, or malformed JSONL.
3. Inspect the union of keys, value types, nulls, and representative records;
   do not assume the first record describes a heterogeneous file.
4. Map source fields to the semantic signals required by the selected training
   method. Fail when a required target, contrast, label, or reference is absent.
5. Write a deterministic script with a source normalizer and the Data platform's
   shared target renderer.
6. Materialize one JSON object per line at `<DEST_DIR>/dataset.jsonl`.
7. Parse every output record, validate exact keys/value types, and measure
   `n_rows_out`.

## Mapping boundaries

- Preserve task semantics, answer formatting, message order, and multi-turn
  context that the target shape supports.
- Do not flatten a conversation into completion/text framing without a clear
  deterministic input/target mapping.
- Do not fabricate missing rejected answers, KTO labels, or correctness
  references.
- Count and explain malformed or rejected source records; never skip them
  silently.
- Use a separate `filter` phase for deduplication, balancing, or sampling.

## Scoring derivatives

When `train_shaped` is requested, use a validation-specific source normalizer
and the identical target renderer. Preserve every validation row and answer;
never copy validation rows into the training output.

## Report

Report `reformat_jsonl` in `method_ids`. State whether the source was JSONL or
a JSON array, the semantic field mapping, parsed counts, and rejection reasons
in `audit_steps`; keep only the concise outcome in `notes`.
