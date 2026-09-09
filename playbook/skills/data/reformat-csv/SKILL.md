---
name: reformat-csv
description: "Map a provided CSV or Parquet table into the semantic record family required by training_method. Use when structured columns must be mapped into SFT, preference, feedback, prompt/reference, or prompt-only records; do not use when the source already is target-shaped JSONL."
---

# Reformat tabular training data

## Workflow

1. Parse the source with a format-aware library. For CSV, detect encoding and
   delimiter safely and respect quoted multiline fields; for Parquet, inspect
   the loaded schema.
2. Measure `n_rows_in` from parsed records, never from physical line count.
3. Inspect columns, data types, null rates, and representative values. Identify
   the semantic input, target, preference, label, or reference fields required
   by the selected training method.
4. Fail when a required signal cannot be mapped honestly. Do not infer an answer
   column solely from a similar name when several candidates exist.
5. Write a deterministic preparation script with:
   - a source-row normalizer;
   - the shared method-specific target renderer defined by the Data platform;
   - explicit validation and rejection counters.
6. Run the script and write UTF-8 JSONL to
   `<DEST_DIR>/dataset.jsonl`.
7. Parse the complete output, verify exact key/value types, and measure
   `n_rows_out`.

## Mapping boundaries

- Preserve the source's real answer representation and task meaning.
- Build genuine preference pairs only from real chosen/rejected evidence.
- Convert label fields to boolean for KTO only with a documented mapping.
- Require a valid reference for every GRPO/RLOO/RFT record.
- Preserve row identifiers only when the target trainer accepts them; otherwise
  keep traceability in the preparation log rather than adding arbitrary keys.
- Do not silently drop invalid rows. Count each rejection reason and fail if the
  remaining dataset is empty or materially different from the requested data.
- Apply deduplication, sampling, or balancing as a separate `filter` phase and
  record those operations separately.

## Scoring derivatives

When the ticket requests `train_shaped`, reuse the exact same target renderer
after a source-specific validation-row normalizer. Do not reuse training-row
field names blindly, and never filter or reorder validation rows.

## Report

Report `reformat_csv` in `method_ids`. State the source columns mapped to each
semantic field, parsed input/output counts, and explicit rejection counts in
`audit_steps`; keep only the concise outcome in `notes`.
