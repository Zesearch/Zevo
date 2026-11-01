# OLMO 32B evaluator and query revision v8

This is the current prepared revision for task
`b333424a-c8a5-408d-acef-005da446c376` (`Olmo-3.1-32B-Instruct-SFT-judged-v2`).
It supersedes the OLMO artifacts in `olmo_v7` and `boxed_math_v1`, incorporating
all their scoring fixes. Earlier artifacts remain immutable.

Changes:

- BBH requires the last line `So the answer is ANSWER.`. MC answers use `(B)`;
  non-MC answers preserve numbers, signs, words and bracket sequences. No text
  may follow that line. Its revised query and scorer must activate together.
- OMEGA still requires a complete final boxed answer. Empty extraction scores
  zero. Exact rational comparison accepts decimal/fraction/LaTeX fraction
  equivalence without floating-point tolerances. Other expressions use the
  bundled symbolic comparator with its five-second per-comparison timeout.
  Commas in tuples are preserved; only scalar thousands grouping is removed.
- LiveCodeBench extraction recognizes both `livecodebench` (the actual sample
  submission marker) and `livecodebench_v3`. Empty prediction falls back to the
  complete response; an existing prediction remains authoritative.
- PopQA uses metric `alias_substring_match` and the displayed name
  `Knowledge · PopQA (alias substring match)`. Its underlying algorithm is
  unchanged. All other inference queries are unchanged.

`manifest.json` maps original production hashes to the new standalone scripts.
`task-update.json` contains the before/after fields for every test set and the
required baseline action. This is a prepared update, not an applied migration.

Activation requires checking all before-values, installing the artifacts under
new hash-verified paths, then updating the task configuration. Preserve frozen
in-flight contracts and historical results. Re-run baseline inference for BBH
because its query changed. For scorer-only changes, re-score saved predictions;
do not mix old/new baseline scores.

Reproduce from the original audited evaluator files and task inventory:

```sh
python scripts/evaluators/repair_olmo_protocol.py SOURCE_DIR OUTPUT_DIR TASK_INVENTORY_JSON
PYTHONPATH=src ../Zevo/.venv/bin/python -m pytest tests/test_olmo_evaluator_revision.py -q
```

Tests exercise the actual frozen embedded modules, aliases, prompt changes,
exact numeric equivalence, symbolic deadline dispatch, and previous v7
regressions. Full held-out runs and paid judge calls have not been performed.
The local test environment does not include SymPy/ANTLR; the symbolic parser
itself is the existing frozen implementation, not replaced by this revision.
