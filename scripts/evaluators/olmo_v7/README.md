# OLMO 32B task evaluator revision v7

Scope: `Olmo-3.1-32B-Instruct-SFT-judged-v2`, task ID
`b333424a-c8a5-408d-acef-005da446c376`. These are standalone immutable replacement
scripts; `manifest.json` maps the original hashes to the new hashes.

The reasoning/knowledge/math script incorporates the prior boxed-only MATH,
AIME and OMEGA fixes and supersedes the OLMO artifact in `boxed_math_v1`.
BBH comparison preserves mathematical symbols. MMLU, AGIEval and GPQA use the
last explicit answer field or boxed choice, or a standalone option letter.
Ambiguous/unresolved final selections score zero; discussion of options is not
mined for an answer. PopQA's existing substring scoring is unchanged.

The instruction/Zebra/Alpaca script uses a JSON decoder to locate ZebraLogic
solution objects, respecting quoted braces and escapes and allowing malformed
prose before the JSON. IFEval and IFBench raise an evaluation error on internal
checker exceptions, with the instruction ID and chained cause. Ordinary false
checks still count as incorrect. AlpacaEval scoring is unchanged. The three
code benchmark scripts are unchanged.

The runner separately removes all whole-evaluation deadlines, including the
former default one-hour and four-hour limits and their environment overrides.
Individual code problems, API requests and transport commands retain deadlines.
Explicit cancellation and Slurm allocation wall time remain effective.

Rebuild with:

```sh
python scripts/evaluators/repair_olmo_protocol.py SOURCE_DIR OUTPUT_DIR
```

No production contracts or results have been changed. Install new immutable
paths/hashes for future evaluations, and re-score saved baseline predictions
under the same version before comparing scores. Do not mutate the frozen
contract of an in-flight evaluation. Full dataset and paid judge evaluations
have not been run as part of the local synthetic regression tests.
