---
name: pdf-to-qa
description: "Extract and chunk user-provided PDF, text, or HTML documents, then author supervised or verifiable-answer examples grounded only in those documents. Use for document sources that require question/answer generation; do not use for plain text-continuation corpora or when the selected method requires preference/label signals the documents cannot support."
---

# Convert documents into grounded examples

Use the supplied documents as the complete source. Author examples from their
content; do not supplement them with unrelated facts.

## Workflow

1. Enumerate the provided files deterministically and reject unsupported or
   unreadable inputs.
2. Write an extraction script for the actual formats:
   - PDF: extract page text with a PDF parser;
   - HTML: remove navigation/markup while preserving meaningful structure;
   - text: decode explicitly and preserve section boundaries.
3. Record extraction failures by file/page. Do not silently treat empty pages as
   successful source material.
4. Normalize whitespace conservatively and chunk by semantic boundaries with a
   documented maximum size and small overlap only when context requires it.
5. Read each chunk and author diverse examples whose answers are supported by
   that chunk. Cover the source rather than repeating one easy question pattern.
6. Render examples through the Data Agent's method-specific target renderer and
   write `<DEST_DIR>/dataset.jsonl`.
7. Deduplicate normalized prompts and stop at `target_size` when it is positive;
   otherwise choose a size justified by source coverage and available distinct
   content.

## Method compatibility

- SFT chat/completion: author a grounded input and target in the task's natural
  answer form.
- GRPO/RLOO/RFT: use only questions with an unambiguous, checkable reference
  answer supported by the document.
- DPO/ORPO/CPO: use only when the source contains genuine alternatives,
  corrections, rankings, or other evidence for better/worse answers. Do not
  invent a rejected answer by corrupting the correct one.
- KTO: use only when the source supplies genuine desirable/undesirable labels.
- Online DPO: produce grounded prompts only when the downstream run has the
  required judge.
- Text framing: do not use this Q&A Skill; extract/chunk the corpus directly as
  text records under the Data platform workflow.

## Quality checks

- Trace each authored answer back to its source chunk during generation.
- Reject questions answerable only from outside knowledge or ambiguous context.
- Preserve code, formulae, units, and negation accurately.
- Use boxed answers only when the task's scorer expects a discrete boxed form.
- Parse every final JSONL record and validate the required method shape.

## Report

Use the parsed chunk count as `n_rows_in` and explain that unit in `notes`.
Use the final example count as `n_rows_out`. Report `pdf_to_qa` in
`method_ids`, and summarize files/pages/chunks processed and skipped in
`audit_steps`.
