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


@pytest.mark.parametrize("kind", ["file", "acquired", "synthetic"])
def test_non_hub_sources_keep_remote_preparation_and_finalization(tmp_path, kind):
    from zevo.engine.remote_training_data import file_sha256, validate_source_selection
    original = tmp_path / "uploaded.jsonl"
    original.write_text('{"prompt":"2+2", "completion":"4"}\n')
    body = _spec().model_dump()
    body["source"] = {"kind": kind, "identity": "original-source", "revision": "recipe-v1"}
    if kind == "file":
        body["source"].update(remote_path=str(original), source_sha256=file_sha256(original))
    spec = RemoteDatasetSpec.model_validate(body)
    if kind == "file":
        validate_source_selection(spec, source_identity="original-source", source_fingerprint=file_sha256(original))
        with pytest.raises(ValueError, match="differs"):
            validate_source_selection(spec, source_identity="original-source", source_fingerprint="0" * 64)
    recipe = {"source_identity": "original-source", "source_fingerprint": remote_source_fingerprint(spec)}
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json())
    script = tmp_path / "prepare.py"
    script.write_text("# reproducible preparation\n")
    dataset = tmp_path / "prepared.jsonl"
    dataset.write_text(original.read_text())
    profile = tmp_path / "profile.json"
    profile.write_text('{"rows": 1}')
    receipt_path = tmp_path / "prepared-receipt.json"
    kwargs = dict(data_intent_signature="a" * 64, spec_path=spec_path,
                  recipe_path=recipe_path, prepare_script_path=script,
                  dataset_path=dataset, profile_path=profile, receipt_path=receipt_path)
    receipt = record_remote_preparation(**kwargs)
    assert validate_remote_preparation(**kwargs) == receipt
    package = build_remote_training_package(spec=spec, recipe=recipe, prepare_script_path=script,
                                            blocked_semantic_fingerprints=[], data_intent_signature="a" * 64)
    package_path = tmp_path / "package.json"
    package_path.write_text(package.model_dump_json())
    final = finalize_remote_training_data(package_path=package_path, prepared_path=dataset,
        output_path=tmp_path / "final.jsonl", profile_path=profile, receipt_path=tmp_path / "receipt.json")
    assert final.n_rows == 1
    if kind == "file":
        original.write_text("changed source")
        with pytest.raises(ValueError, match="SHA-256"):
            record_remote_preparation(**kwargs)


@pytest.mark.parametrize("source", ["hub", "file", "acquired", "synthetic"])
def test_all_training_sources_build_remote_contract(tmp_path, source):
    from tests.test_cloud_device_remote_data_plane import _cloud
    from zevo.db.models import Run, Ticket
    from zevo.engine.run.runner import _build_data_input
    device = tmp_path / "device.json"
    from zevo.contracts.infrastructure import InfrastructureDeviceInfo
    device.write_text(InfrastructureDeviceInfo(**_cloud()).model_dump_json())
    uploaded = tmp_path / "source.jsonl"
    uploaded.write_text('{"prompt":"p", "completion":"a"}\n')
    payload = dict(operation="prepare_run_data", training_method="sft",
                   data_intent_signature="a" * 64, data_query="find suitable training data")
    payload["dataset"] = "org/train" if source == "hub" else str(uploaded) if source == "file" else ""
    run = Run(id="run-1", metric="accuracy", gpu_provider="cloud")
    ticket = Ticket(id="data-1", run_id=run.id, agent_id="data", lane="optimization")
    inp = _build_data_input(ticket, payload, {"device_info": {"path": str(device)}}, str(tmp_path), run)
    assert inp.device_info_path == str(device)
    assert inp.remote_dataset_spec_schema and inp.remote_data_output_dir
    assert "check-prepared" in inp.artifacts_validation_command
    assert bool(inp.expected_source_fingerprint) == (source == "file")


def test_uploaded_source_materializes_via_generic_remote_script(tmp_path):
    from zevo.engine.remote_training_data import file_sha256, materialize_remote_training_data
    uploaded = tmp_path / "source with spaces.jsonl"
    uploaded.write_text('{"prompt":"q", "completion":"answer"}\n')
    body = _spec().model_dump()
    body["source"] = dict(kind="file", identity="original-upload", revision="v1",
                          source_sha256=file_sha256(uploaded), remote_path=str(uploaded))
    spec = RemoteDatasetSpec.model_validate(body)
    script = tmp_path / "prepare.py"
    script.write_text('''import argparse, json, pathlib
p = argparse.ArgumentParser()
for name in ("source-json", "cache-dir", "output", "profile"):
    p.add_argument("--" + name, required=True)
a = p.parse_args()
source = json.loads(a.source_json)
pathlib.Path(a.output).write_bytes(pathlib.Path(source["remote_path"]).read_bytes())
pathlib.Path(a.profile).write_text('{"rows": 1}')
''')
    package = build_remote_training_package(spec=spec, recipe={"source_identity": "original-upload"},
        prepare_script_path=script, blocked_semantic_fingerprints=[], data_intent_signature="a" * 64)
    package_file = tmp_path / "package.json"
    package_file.write_text(package.model_dump_json())
    receipt = materialize_remote_training_data(package_path=package_file, prepare_script_path=script,
        output_path=tmp_path / "final.jsonl", cache_dir=tmp_path / "cache", receipt_path=tmp_path / "receipt.json")
    assert receipt.n_rows == 1
    assert Path(receipt.dataset_path).read_text() == uploaded.read_text()


@pytest.mark.parametrize("dataset", ["org/train", "/upload/data.jsonl", ""])
def test_training_data_binding_requires_infrastructure_for_every_source(dataset):
    from zevo.contracts.tickets import validate_bindings
    payload = {"operation": "prepare_run_data", "dataset": dataset, "data_query": "find data"}
    with pytest.raises(ValueError, match="device_info"):
        validate_bindings(agent_id="data", payload=payload, inputs={})
    result = validate_bindings(agent_id="data", payload=payload, inputs={
        "device_info": {"artifact_role": "device_info", "path": "/device.json"}})
    assert result["device_info"]["path"] == "/device.json"
