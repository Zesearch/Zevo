# OLMO 32B Validation revision v2

Prepared for the twelve Validation sets in L1 setting
`186094f0-02ec-4046-821f-83236281a375`, task
`b333424a-c8a5-408d-acef-005da446c376`.

- MATH-Hard and GSM8K use the frozen v8 mathematical reference: a complete
  final boxed answer is required, with no unboxed-answer fallback.
- ARC, FOLIO, OpenBookQA and MMLU take the last explicit answer, including
  corrections on the same line. Ambiguous/malformed final selections fail.
  Each row retains its allowed labels, including ARC's numeric labels.
- The complete reference evaluator is embedded and hash-recorded, removing
  the dependency on a separate absolute `/app/data/evaluators/...` file.
- `RUNTIME` is explicitly declared. The existing compute bundle builder can
  inventory its packages and stage the runtime and NLTK assets. Slurm requires
  the supported container mount; instance/cloud hosts still require the
  documented pre-provisioned compatible runtime. This patch does not remove
  those infrastructure prerequisites.
- Code, IF and chat scoring paths and all twelve inference queries remain
  unchanged. Existing whole-evaluation deadline removal also applies here.

`manifest.json` records old/new hashes and the embedded dependency hash.
`setting-update.json` lists guarded before/after fields for all twelve sets.
Install the hash-verified artifact before applying the setting updates, verify
all before-values, and preserve existing frozen run/ticket contracts.
The update is prepared locally, not applied to production.

Re-score saved baseline predictions before comparing with scores under this
revision. No baseline inference rerun is required because queries are unchanged.

Rebuild:

```sh
python scripts/evaluators/repair_olmo_validation.py VALIDATION_AUDIT_DIR scripts/evaluators/olmo_v8 OUTPUT_DIR
PYTHONPATH=src ../Zevo/.venv/bin/python -m pytest tests/test_olmo_validation_revision.py -q
```

Tests cover actual CLI choice scoring after script relocation, integer/letter
option ranges, incomplete boxes, frozen dependency identity, and runtime/package
inventory inside the remote bundle. Tests use a simulated runtime inventory;
a real cluster/cloud execution and full dataset/judge run remain unperformed.
