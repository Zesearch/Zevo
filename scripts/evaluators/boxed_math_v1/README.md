# Boxed math evaluator revision

These four standalone scripts replace the audited math-full, math-full-02,
math-full-v2, and OLMO v5 evaluators. `manifest.json` maps each immutable original
SHA-256 to its replacement. The scripts have not been installed in production.

Rules for model answers:

- Require `\boxed{...}`. Use the last boxed command; an empty or incomplete final
  box scores zero, even when an earlier intermediate box was valid.
- Parse balanced nested LaTeX braces and escaped literal braces in one pass.
- Keep existing mathematical-equivalence comparators after extraction. References
  do not need boxes. Fractions and zero-padded numbers remain supported.
- In the three math-full variants, parse numeric values as exact rationals and
  compare integer references exactly. Existing decimal tolerance remains.
- Minerva accepts recognized trailing unit symbols, including SI prefixes and
  powers. It no longer strips arbitrary expression suffixes such as `/5` or `+100`.
- OMEGA extracts only the boxed answer, eliminating the slow full-generation
  regex fallback. Its existing answer-comparison function remains unchanged.
- OLMO MATH/AIME also require boxed answers. BBH, AGIEval, MMLU, GPQA and PopQA
  implementations and configuration are unchanged.

This changes the scoring protocol. Do not mix new scores with old baseline
scores: re-score saved baseline predictions using the same replacement script.
Keep original evaluator objects and old results. Installation must update task
contracts to new paths/hashes; copying files alone does not activate them, and
changing a task contract must not mutate an in-flight ticket's frozen contract.

Rebuild from the original hash-named scripts (the builder verifies their hashes):

```sh
python scripts/evaluators/repair_math_protocol.py SOURCE_DIR OUTPUT_DIR
PYTHONPATH=src ../Zevo/.venv/bin/python -m pytest tests/test_boxed_math_evaluators.py -q
```

Validation covers frozen artifact hashes, malformed/absent/multiple/nested boxes,
large integer near misses, fractional equivalence, Minerva units, OMEGA long-text
performance, and actual CLI scoring for all three math-full variants. OLMO
extraction is tested from its embedded module; a complete production benchmark
re-score has not been performed. Existing optional symbolic-parser dependencies
are unchanged. Evaluation retry policy is outside this patch.
