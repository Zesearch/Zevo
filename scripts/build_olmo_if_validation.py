#!/usr/bin/env python3
"""Build a pinned, independently sampled IF Validation asset for OLMo.

Only multi-constraint rows covered by the pinned official IFBench verifier are
eligible. Run setup independently rejects Validation/Test prompt overlap, and
Data removes any matching rows from the selected training population.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from ifbench import instructions_registry

from zevo.engine.artifact_validation import (
    _semantic_fingerprint,
    semantic_record_fingerprints,
)


SOURCE_ID = "allenai/IF_multi_constraints_upto5"
SOURCE_REVISION = "2e3a77407b7fce69f95b248d64a884e3ae1c2423"
SOURCE_FILE = "data/train-00000-of-00001.parquet"
VERIFIER_REVISION = "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d"
ELIGIBLE_CONSTRAINTS = frozenset({
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "detectable_format:title",
    "detectable_format:json_format",
    "detectable_format:constrained_response",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "punctuation:no_comma",
    "startend:quotation",
})


def build(parquet: Path, output_dir: Path, *, count: int, excluded: list[Path]) -> dict:
    missing = ELIGIBLE_CONSTRAINTS - set(instructions_registry.INSTRUCTION_DICT)
    if missing:
        raise ValueError(f"pinned verifier lacks required constraints: {sorted(missing)}")
    blocked = set()
    for path in excluded:
        blocked.update(semantic_record_fingerprints(str(path)))

    candidates: dict[str, tuple[str, dict]] = {}
    skipped_overlap = 0
    skipped_duplicate = 0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1024):
        for source in batch.to_pylist():
            messages = source.get("messages") or []
            if len(messages) != 1 or messages[0].get("role") != "user":
                continue
            prompt = str(messages[0].get("content") or "").strip()
            if not prompt or not prompt.isascii():
                continue
            try:
                raw = ast.literal_eval(source["ground_truth"])
            except (SyntaxError, ValueError, TypeError):
                continue
            if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
                continue
            spec = raw[0]
            ids = spec.get("instruction_id")
            kwargs = spec.get("kwargs")
            if (
                not isinstance(ids, list) or len(ids) < 2
                or not isinstance(kwargs, list) or len(kwargs) != len(ids)
                or not set(ids) <= ELIGIBLE_CONSTRAINTS
            ):
                continue
            if any(arg is not None and not isinstance(arg, dict) for arg in kwargs):
                continue
            fingerprint = _semantic_fingerprint({"instruction": prompt})
            if fingerprint in blocked:
                skipped_overlap += 1
                continue
            if fingerprint in candidates:
                skipped_duplicate += 1
                continue
            candidates[fingerprint] = (prompt, spec)
    if len(candidates) < count:
        raise ValueError(
            f"only {len(candidates)} clean, verifiable candidates remain; need {count}"
        )
    chosen = sorted(
        candidates.items(),
        key=lambda item: hashlib.sha256(
            ("zevo-olmo-if-validation-v1:" + item[0]).encode()
        ).hexdigest(),
    )[:count]
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = output_dir / "if_validation.jsonl"
    submission = output_dir / "if_validation_submission.csv"
    with dataset.open("w", encoding="utf-8") as handle:
        for index, (_fingerprint, (prompt, spec)) in enumerate(chosen, start=1):
            handle.write(json.dumps({
                "id": f"ifv-{index:04d}",
                "instruction": prompt,
                "constraint_spec": spec,
            }, ensure_ascii=False) + "\n")
    with submission.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "prediction"])
        writer.writerow(["ifv-0001", "<final response>"])
    manifest = {
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "source_file": SOURCE_FILE,
        "verifier_revision": VERIFIER_REVISION,
        "n_rows": count,
        "minimum_constraints_per_row": 2,
        "eligible_constraints": sorted(ELIGIBLE_CONSTRAINTS),
        "excluded_sources": [str(path) for path in excluded],
        "overlap_candidates_removed": skipped_overlap,
        "duplicate_candidates_removed": skipped_duplicate,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.count < 100:
        parser.error("Validation requires at least 100 rows")
    parquet = args.parquet or Path(hf_hub_download(
        repo_id=SOURCE_ID, repo_type="dataset", filename=SOURCE_FILE,
        revision=SOURCE_REVISION,
    ))
    print(json.dumps(build(parquet, args.out, count=args.count, excluded=args.exclude)))


if __name__ == "__main__":
    main()
