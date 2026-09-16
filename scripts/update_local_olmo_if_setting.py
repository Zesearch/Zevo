#!/usr/bin/env python3
"""Replace only FollowBench in the local OLMo L1 Validation setting."""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from zevo.api.routers.ui.tasks import SettingBody


ROOT = Path(__file__).resolve().parents[1]
TASK = "Olmo-3.1-32B-Instruct-SFT"
BASE = "http://localhost:8001/api"


def main() -> None:
    asset = ROOT / "assets" / "olmo" / "if_validation" / "if_validation.jsonl"
    sample = asset.with_name("if_validation_submission.csv")
    evaluator = ROOT / "examples" / "olmo_if_validation_evaluator.py"
    if not all(path.is_file() for path in (asset, sample, evaluator)):
        raise SystemExit("pinned IF Validation assets are missing")
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{BASE}/tasks/{TASK}/settings")
        response.raise_for_status()
        candidates = [item for item in response.json() if item.get("name") == "L1"]
        if len(candidates) != 1:
            raise SystemExit(f"expected one L1 setting, found {len(candidates)}")
        old = candidates[0]
        body = {
            key: old[key] for key in SettingBody.model_fields if key in old
        }
        suite = list(body.get("validation_sets") or [])
        matches = [
            index for index, item in enumerate(suite)
            if item.get("name") in {
                "Instruction Following · FollowBench",
                "Instruction Following · Verifiable IF",
            }
        ]
        if len(matches) != 1:
            raise SystemExit(f"expected one instruction-following member, found {len(matches)}")
        backup = ROOT / "data" / "setting-backups" / "olmo_l1_before_if_validation.json"
        replacing_followbench = (
            suite[matches[0]]["name"] == "Instruction Following · FollowBench"
        )
        if replacing_followbench and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_text(
                json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
        suite[matches[0]] = {
            "name": "Instruction Following · Verifiable IF",
            "test_set": "assets/olmo/if_validation/if_validation.jsonl",
            "inference_query": (
                "Follow {instruction} exactly. Return only the final response in "
                "prediction and copy id unchanged."
            ),
            "sample_submission": "assets/olmo/if_validation/if_validation_submission.csv",
            "metric_type": "custom",
            "metric": "instruction_following_accuracy",
            "metric_direction": "max",
            "answer_fields": ["constraint_spec"],
            "evaluation_script": "examples/olmo_if_validation_evaluator.py",
            "split": "",
            "config": "",
            "max_rows": 0,
            "source_rows": 500,
        }
        body["validation_sets"] = suite
        SettingBody.model_validate(body)
        response = client.patch(
            f"{BASE}/tasks/{TASK}/settings/{old['id']}", json=body,
        )
        response.raise_for_status()
        updated = response.json()
        selected = updated["validation_sets"][matches[0]]
        if (
            selected["metric"] != "instruction_following_accuracy"
            or selected["source_rows"] != 500
            or not selected["evaluator_sha256"]
        ):
            raise SystemExit("setting response did not freeze the expected IF scorer")
        print(json.dumps({
            "setting_id": updated["id"],
            "validation_member": selected["name"],
            "rows": selected["source_rows"],
            "evaluator_sha256": selected["evaluator_sha256"],
            "backup": str(backup) if backup.exists() else "",
        }))


if __name__ == "__main__":
    main()
