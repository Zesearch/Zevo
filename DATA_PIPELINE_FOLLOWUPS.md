# Data Pipeline Follow-ups

Status: items 1–5 and their regression coverage were implemented on 2026-09-13.
Tool-use supervision (item 6) remains intentionally deferred to dataset-specific
Data Query guidance and is not part of this change.

## 1. Align the durable remote data directory with engine validation

Status: complete. Prepared data now lives under the Run/data-intent directory;
engine validation accepts only that assigned root and no workdir hardlink is needed.

The engine assigns prepared Hugging Face data to:

`<zevo-root>/data/runs/<run-id>/<data-ticket-id>/`

However, remote result validation currently accepts only paths below the cluster
`workdir` or `hf_cache`. Because the durable data directory is a sibling of the
per-run workdir, the Data Agent had to hardlink its output back into:

`<zevo-root>/<run-id>/data/<data-ticket-id>/`

Required change:

- Treat the assigned `remote_data_output_dir` as an engine-authorized remote root.
- Finalize and report the dataset directly below the durable data directory.
- Remove the hardlink workaround into the per-run workdir.
- Keep raw Hugging Face files only in `cache/hf`; never place prepared data there.

## 2. Reuse completed Data artifacts across retries

Status: complete. A preparation receipt binds immutable source, recipe, script,
schema, row count and checksums. Replacement Tickets with the same intent share
the preparation directory and may reuse it only after receipt validation.

A transient SSH failure happened after the first ticket had already prepared its
dataset. The replacement ticket created the dataset again because the first
ticket had not committed any WorkProducts.

Required change:

- Persist a compact preparation receipt before later engine post-processing.
- On retry, discover candidate artifacts from an earlier ticket in the same Run.
- Reuse them only after validating source revision, recipe, schema, row count,
  checksum/data signature, and file existence.
- Never reuse a partial or semantically different artifact merely because the
  filename exists.

## 3. Avoid a full verified copy when decontamination changes nothing

Status: complete. Zero-change finalization publishes the immutable verified path
with an atomic hardlink, then reflink when available, with copy as fallback.

The engine currently writes another complete `verified/.../dataset.jsonl` even
when zero rows are removed. For this Run, each file is about 7.47 GB.

Required change:

- When verification removes zero rows and the filesystem supports it, use an
  atomic hardlink or reflink for the verified artifact.
- Fall back to a normal copy/rewrite only when required.
- Preserve an immutable verified path and receipt regardless of storage method.

## 4. Clean up superseded failed-ticket artifacts safely

Status: complete for new successful replacement Tickets. Unreferenced legacy
failed-ticket directories move to recoverable `.trash`; referenced paths and
directories containing preparation/final receipts are retained and audited.

The first failed ticket and second successful ticket currently leave prepared
and verified datasets behind. The unique physical data for this Run is roughly
28 GiB, excluding the shared Hugging Face cache.

Required change:

- After a replacement ticket succeeds, mark prior artifacts as superseded.
- Delete only artifacts proven unreferenced by WorkProducts, active tickets,
  checkpoints, or receipts.
- Prefer a retention window or trash state before permanent deletion.
- Expose what was retained, reused, or removed in the ticket audit trail.

## 5. Retry transient SSH control-plane failures in place

Status: complete. Idempotent control operations retry up to three times with
bounded exponential backoff; expensive dataset preparation is not rerun by this layer.

The first Data ticket failed because `beta.empireai.edu` closed an SSH control
connection with exit code 255 after the expensive preparation work completed.

Required change:

- Classify retryable SSH failures separately from deterministic data failures.
- Retry idempotent upload, finalize, existence-check, and receipt-download
  commands with bounded exponential backoff.
- Re-probe the route before escalating to a new Infrastructure/Data ticket.
- Keep the same prepared artifact and ticket when recovery is safe.

## 6. Preserve and train tool-use supervision explicitly

Status: deferred by product decision. Users may describe dataset-specific tool
format handling in Data Query; Zevo does not impose a universal renderer here.

`allenai/Dolci-Instruct-SFT` contains assistant tool calls whose text `content`
may be empty while the actual target is stored in `function_calls`. Tool
definitions are stored in `functions`, and tool results use the `environment`
role.

Required change:

- Normalize `functions`, `function_calls`, and `environment` turns into the
  selected model's explicit tool-use serialization.
- For OLMo Instruct formatting, preserve the `<functions>`, `</functions>`,
  `<function_calls>`, and `</function_calls>` tags.
- Include assistant function-call tokens in the loss target even when assistant
  `content` is empty.
- Mask system, user, tool definitions, and environment/tool-result context.
- Use the exact same renderer and template configuration in Train and Inference.
- Do not require additional user-facing configuration; resolve this from the
  model/tokenizer and normalized record shape.

## 7. Add regression checks for the combined change

Status: complete for items 1–5 plus question-only Train/Validation/Test
decontamination. Tool-use renderer tests remain deferred with item 6.

Required tests:

- Durable remote paths are accepted without a workdir hardlink.
- A retry reuses a fully validated earlier artifact and rejects partial or
  mismatched artifacts.
- Zero-change verification does not allocate another full dataset copy.
- Superseded cleanup never removes a referenced artifact.
- SSH exit-255 retry resumes control-plane post-processing without rerunning
  expensive data preparation.
- A tool-call-only assistant turn produces non-empty target labels.
- The rendered tool-use prompt is byte/token aligned between Train and Inference.
- Ordinary non-tool chat training remains unchanged.

## Current Run evidence

- Run: `0920f6c2-6303-47e2-9669-2b3a090d1e84`
- Successful Data ticket: `data-0920f6c2-002`
- Source rows: 2,152,112
- Prepared rows: 2,151,298
- Shared Hugging Face cache reused: 6.6 GB
- Prepared dataset size: 7,470,793,256 bytes
- Validation suite members materialized: 16
- Engine decontamination removed: 0 rows
