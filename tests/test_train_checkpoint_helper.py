from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


HELPER = Path(__file__).resolve().parents[1] / "playbook/runners/train_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("zevo_train_checkpoint_test", HELPER)
assert SPEC and SPEC.loader
checkpoint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checkpoint
SPEC.loader.exec_module(checkpoint)


def test_checkpoint_directory_is_validated_and_committed_atomically(tmp_path: Path) -> None:
    staging = tmp_path / "model.staging"
    final = tmp_path / "model"
    staging.mkdir()
    (staging / "config.json").write_text("{}", encoding="utf-8")
    (staging / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (staging / "model-00002-of-00002.safetensors").write_bytes(b"second")
    (staging / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }), encoding="utf-8")

    receipt = checkpoint.commit_checkpoint_directory(
        staging, final, metadata={"ticket_id": "train-1"},
    )

    assert not staging.exists()
    assert (final / checkpoint.MARKER_NAME).is_file()
    assert receipt["weight_bytes"] == len(b"firstsecond")
    assert receipt["metadata"] == {"ticket_id": "train-1"}
    assert checkpoint.commit_checkpoint_directory(staging, final) == receipt


def test_checkpoint_commit_rejects_missing_index_shard(tmp_path: Path) -> None:
    staging = tmp_path / "model.staging"
    staging.mkdir()
    (staging / "config.json").write_text("{}", encoding="utf-8")
    (staging / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (staging / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"b": "model-00002-of-00002.safetensors"}
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="missing shards"):
        checkpoint.commit_checkpoint_directory(staging, tmp_path / "model")


def test_checkpoint_commit_accepts_a_peft_adapter(tmp_path: Path) -> None:
    staging = tmp_path / "adapter.staging"
    staging.mkdir()
    (staging / "adapter_config.json").write_text("{}", encoding="utf-8")
    (staging / "adapter_model.safetensors").write_bytes(b"adapter")

    receipt = checkpoint.commit_checkpoint_directory(
        staging, tmp_path / "adapter",
    )

    assert receipt["weight_files"] == ["adapter_model.safetensors"]
    assert receipt["weight_bytes"] == len(b"adapter")


def test_checkpoint_commit_requires_sibling_directories(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(ValueError, match="must be siblings"):
        checkpoint.commit_checkpoint_directory(
            staging, tmp_path / "nested" / "model",
        )
