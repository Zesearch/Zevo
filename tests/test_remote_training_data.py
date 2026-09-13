from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from zevo.contracts.data import DataResult
from zevo.engine.artifact_validation import semantic_record_fingerprints
from zevo.engine.remote_training_data import (
    RemoteDatasetSpec,
    build_remote_training_package,
    finalize_remote_training_data,
    load_remote_dataset_receipt,
    load_remote_preparation_receipt,
    record_remote_preparation,
    remote_source_fingerprint,
    validate_remote_preparation,
)
from zevo.engine.run.runner import _run_remote_control_command


def _spec() -> RemoteDatasetSpec:
    return RemoteDatasetSpec.model_validate({
        "source": {
            "kind": "huggingface",
            "dataset_id": "allenai/Dolci-Instruct-SFT",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "config": "default",
            "split": "train",
        },
        "training_method": "full_sft",
        "method_format": "messages",
    })


def test_remote_source_identity_requires_an_explicit_revision() -> None:
    body = _spec().model_dump(mode="json")
    body["source"]["revision"] = ""
    with pytest.raises(ValidationError, match="revision"):
        RemoteDatasetSpec.model_validate(body)
    assert len(remote_source_fingerprint(_spec())) == 64


def test_remote_finalize_filters_in_place_and_returns_only_a_receipt(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "remote" / "staging" / "dataset.jsonl"
    prepared.parent.mkdir(parents=True)
    rows = [
        {"messages": [{"role": "user", "content": "keep me"},
                      {"role": "assistant", "content": "answer"}]},
        {"messages": [{"role": "user", "content": "held out question"},
                      {"role": "assistant", "content": "gold"}]},
    ]
    prepared.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    scoring = tmp_path / "validation.jsonl"
    scoring.write_text(json.dumps({
        "messages": [
            {"role": "user", "content": "held out question"},
            {"role": "assistant", "content": "a different gold answer"},
        ],
    }) + "\n", encoding="utf-8")
    script = tmp_path / "prepare_data.py"
    script.write_text("# deterministic remote preparation\n", encoding="utf-8")
    recipe = {
        "schema_version": 2,
        "dataset_name": "allenai/Dolci-Instruct-SFT",
        "source_identity": "allenai/Dolci-Instruct-SFT",
        "source_fingerprint": remote_source_fingerprint(_spec()),
        "training_method": "full_sft",
        "method_format": "messages",
        "method_ids": ["inline_transform"],
        "direction": "",
        "subset": "",
        "filters": [],
        "sampling": {},
        "weighting": {},
        "transformations": [],
        "field_mapping": {},
        "seed": 0,
        "teacher_distillation": {
            "enabled": False,
            "teacher_model": "",
            "output_format": "instruction_distillation",
            "target_augmentation_count": 0,
            "verify_answers": True,
            "decontaminate_against_eval": True,
            "generation_params": {},
        },
        "audit_steps": [],
    }
    package = build_remote_training_package(
        spec=_spec(),
        recipe=recipe,
        prepare_script_path=script,
        blocked_semantic_fingerprints=semantic_record_fingerprints(str(scoring)),
        data_intent_signature="a" * 64,
    )
    package_path = tmp_path / "remote_training_package.json"
    package_path.write_text(package.model_dump_json(), encoding="utf-8")
    output = tmp_path / "remote" / "verified" / "dataset.jsonl"
    receipt_path = tmp_path / "receipt.json"

    receipt = finalize_remote_training_data(
        package_path=package_path,
        prepared_path=prepared,
        output_path=output,
        receipt_path=receipt_path,
    )

    assert receipt.n_rows_before_decontamination == 2
    assert receipt.n_rows == 1
    assert receipt.decontamination_removed_rows == 1
    assert "keep me" in output.read_text(encoding="utf-8")
    assert "held out question" not in output.read_text(encoding="utf-8")
    assert load_remote_dataset_receipt(receipt_path) == receipt
    assert receipt.materialization_mode == "rewrite"


def test_remote_zero_change_finalize_uses_one_physical_dataset(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared" / "dataset.jsonl"
    prepared.parent.mkdir(parents=True)
    prepared.write_text(json.dumps({
        "messages": [
            {"role": "user", "content": "training only"},
            {"role": "assistant", "content": "answer"},
        ],
    }) + "\n", encoding="utf-8")
    profile = prepared.with_name("data_profile.json")
    profile.write_text('{"n_rows":1}\n', encoding="utf-8")
    script = tmp_path / "prepare_data.py"
    script.write_text("# deterministic remote preparation\n", encoding="utf-8")
    package = build_remote_training_package(
        spec=_spec(),
        recipe={"source": "fixture"},
        prepare_script_path=script,
        blocked_semantic_fingerprints=[],
        data_intent_signature="b" * 64,
    )
    package_path = tmp_path / "remote_training_package.json"
    package_path.write_text(package.model_dump_json(), encoding="utf-8")
    output = tmp_path / "verified" / "dataset.jsonl"

    receipt = finalize_remote_training_data(
        package_path=package_path,
        prepared_path=prepared,
        output_path=output,
        receipt_path=output.with_name("remote_data_receipt.json"),
        profile_path=profile,
        preparation_receipt_path=prepared.with_name("preparation_receipt.json"),
    )

    assert receipt.decontamination_removed_rows == 0
    assert receipt.materialization_mode == "hardlink"
    assert prepared.stat().st_ino == output.stat().st_ino
    preparation = load_remote_preparation_receipt(
        prepared.with_name("preparation_receipt.json")
    )
    assert preparation.dataset_sha256 == receipt.dataset_sha256


def test_preparation_receipt_reuses_only_exact_identity_and_bytes(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"question":"q","answer":"a"}\n', encoding="utf-8")
    profile = tmp_path / "data_profile.json"
    profile.write_text('{"n_rows":1}\n', encoding="utf-8")
    spec_path = tmp_path / "remote_dataset_spec.json"
    spec_path.write_text(_spec().model_dump_json(), encoding="utf-8")
    recipe_path = tmp_path / "data_recipe.json"
    recipe_path.write_text('{"source":"fixture"}\n', encoding="utf-8")
    script = tmp_path / "prepare_data.py"
    script.write_text("# deterministic\n", encoding="utf-8")
    receipt_path = tmp_path / "preparation_receipt.json"

    recorded = record_remote_preparation(
        data_intent_signature="c" * 64,
        spec_path=spec_path,
        recipe_path=recipe_path,
        prepare_script_path=script,
        dataset_path=dataset,
        profile_path=profile,
        receipt_path=receipt_path,
    )
    assert validate_remote_preparation(
        data_intent_signature="c" * 64,
        spec_path=spec_path,
        recipe_path=recipe_path,
        prepare_script_path=script,
        dataset_path=dataset,
        profile_path=profile,
        receipt_path=receipt_path,
    ) == recorded

    dataset.write_text('{"question":"changed","answer":"a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        validate_remote_preparation(
            data_intent_signature="c" * 64,
            spec_path=spec_path,
            recipe_path=recipe_path,
            prepare_script_path=script,
            dataset_path=dataset,
            profile_path=profile,
            receipt_path=receipt_path,
        )


def test_data_result_uses_one_local_or_remote_data_plane() -> None:
    common = {
        "status": "succeeded",
        "operation": "prepare_run_data",
        "ticket_id": "data-1",
        "data_recipe_path": "/local/data_recipe.json",
        "prepare_script_path": "/local/prepare_data.py",
        "error_message": "",
        "notes": "prepared remotely",
    }
    remote = DataResult(
        **common,
        remote_dataset_path="/remote/cache/dataset.jsonl",
        remote_dataset_spec_path="/local/remote_dataset_spec.json",
        remote_data_profile_path="/remote/cache/data_profile.json",
    )
    assert not remote.training_dataset_path
    with pytest.raises(ValidationError, match="exactly one"):
        DataResult(
            **common,
            training_dataset_path="/local/dataset.jsonl",
            remote_dataset_path="/remote/cache/dataset.jsonl",
            remote_dataset_spec_path="/local/remote_dataset_spec.json",
            remote_data_profile_path="/remote/cache/data_profile.json",
        )


@pytest.mark.asyncio
async def test_remote_control_exit_255_retries_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    return_codes = iter([255, 0])
    calls: list[tuple[str, ...]] = []

    class Process:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

        async def communicate(self) -> tuple[bytes, bytes]:
            if self.returncode:
                return b"", b"Connection closed by remote host"
            return b"ok", b""

        def kill(self) -> None:
            self.returncode = -9

    async def create_process(*command: str, **_kwargs: object) -> Process:
        calls.append(tuple(command))
        return Process(next(return_codes))

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(asyncio, "sleep", no_wait)

    attempts, stdout = await _run_remote_control_command(["ssh", "host", "true"])

    assert attempts == 2
    assert stdout == "ok"
    assert calls == [("ssh", "host", "true"), ("ssh", "host", "true")]
