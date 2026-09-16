# OLMo instruction-following Validation

This is a fixed 500-row, multi-constraint Validation subset of
[`allenai/IF_multi_constraints_upto5`](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5)
at revision `2e3a77407b7fce69f95b248d64a884e3ae1c2423` (ODC-BY-1.0).
It is not IFBench's held-out Test set. `manifest.json` records its source,
selection rules and content digest; `scripts/build_olmo_if_validation.py`
reproduces the selection.

Scoring uses only constraints verified by the official
[`allenai/IFBench`](https://github.com/allenai/IFBench) registry at pinned
commit `1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d` (Apache-2.0).
All constraints in a row must pass for that row to score 1. Unknown constraints
or verifier errors fail the Evaluation ticket rather than award a default pass.
Run setup checks question overlap with all final Test rows, and the Data stage
removes overlapping training rows before training.
