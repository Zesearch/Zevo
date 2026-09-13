"""Remote-first materialization for Hugging Face training datasets.

The scheduler is the control plane: it stores a small, immutable package that
names the Hub revision, preparation program, and hidden scoring fingerprints.
The assigned training host is the data plane: it downloads, inspects,
normalizes, decontaminates, and retains the actual JSONL dataset in a durable
Run/data-intent directory shared by exact retries. Hugging Face's reusable raw downloads remain in a
separate source cache. Only a compact receipt is copied back.

Data uploads this one file beside its preparation script, so the remote host
does not need the Zevo package installed. The host must provide Pydantic (for
the closed package/receipt schemas) plus the dependencies used by the generated
preparation script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RemoteHuggingFaceSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field("huggingface", pattern=r"^huggingface$")
    dataset_id: str = Field(min_length=3, pattern=r"^[^/\s]+/[^/\s]+$")
    revision: str = Field(
        min_length=1,
        description=(
            "Resolved immutable Hub commit/revision. Never leave this as an "
            "implicit moving default."
        ),
    )
    config: str = ""
    split: str = Field(min_length=1)


class RemoteDatasetSpec(BaseModel):
    """Data-authored, answer-blind plan for remote HF materialization."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(1, ge=1, le=1)
    source: RemoteHuggingFaceSource
    training_method: str = Field(min_length=1)
    method_format: str = Field(min_length=1)
    output_format: str = Field("jsonl", pattern=r"^jsonl$")
    preparation_interface: str = Field(
        "zevo_remote_data_v1", pattern=r"^zevo_remote_data_v1$",
    )


class RemoteTrainingPackage(BaseModel):
    """Engine-owned package carried to the training host (never raw rows)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(1, ge=1, le=1)
    source: RemoteHuggingFaceSource
    training_method: str = Field(min_length=1)
    method_format: str = Field(min_length=1)
    preparation_interface: str = Field(
        "zevo_remote_data_v1", pattern=r"^zevo_remote_data_v1$",
    )
    data_intent_signature: str = Field(pattern=_SHA256_PATTERN)
    prepare_script_sha256: str = Field(pattern=_SHA256_PATTERN)
    data_recipe_sha256: str = Field(pattern=_SHA256_PATTERN)
    data_signature: str = Field(pattern=_SHA256_PATTERN)
    blocked_semantic_fingerprints: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_fingerprints(self) -> "RemoteTrainingPackage":
        if any(
            len(value) != 64
            or value.lower() != value
            or any(ch not in "0123456789abcdef" for ch in value)
            for value in self.blocked_semantic_fingerprints
        ):
            raise ValueError(
                "blocked_semantic_fingerprints must contain bare lowercase SHA-256 values"
            )
        if self.blocked_semantic_fingerprints != sorted(
            set(self.blocked_semantic_fingerprints)
        ):
            raise ValueError(
                "blocked_semantic_fingerprints must be sorted and unique"
            )
        return self


class RemoteDatasetReceipt(BaseModel):
    """Small evidence file copied back after remote materialization."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(1, ge=1, le=1)
    data_signature: str = Field(pattern=_SHA256_PATTERN)
    dataset_path: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=_SHA256_PATTERN)
    profile_path: str = Field(min_length=1)
    n_rows_before_decontamination: int = Field(ge=1)
    n_rows: int = Field(ge=1)
    decontamination_removed_rows: int = Field(ge=0)
    record_fields: list[str] = Field(min_length=1)
    materialization_mode: Literal["rewrite", "hardlink", "reflink", "copy"] = (
        "rewrite"
    )

    @model_validator(mode="after")
    def validate_counts(self) -> "RemoteDatasetReceipt":
        if (
            self.n_rows_before_decontamination - self.decontamination_removed_rows
            != self.n_rows
        ):
            raise ValueError("remote dataset receipt row counts do not reconcile")
        if len(self.record_fields) != len(set(self.record_fields)):
            raise ValueError("remote dataset receipt fields must be unique")
        return self


class RemotePreparationReceipt(BaseModel):
    """Durable proof that expensive source preparation completed remotely."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(1, ge=1, le=1)
    data_intent_signature: str = Field(pattern=_SHA256_PATTERN)
    source: RemoteHuggingFaceSource
    prepare_script_sha256: str = Field(pattern=_SHA256_PATTERN)
    data_recipe_sha256: str = Field(pattern=_SHA256_PATTERN)
    preparation_signature: str = Field(pattern=_SHA256_PATTERN)
    dataset_path: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=_SHA256_PATTERN)
    profile_path: str = Field(min_length=1)
    profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    n_rows: int = Field(ge=1)
    record_fields: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_fields(self) -> "RemotePreparationReceipt":
        if len(self.record_fields) != len(set(self.record_fields)):
            raise ValueError("remote preparation receipt fields must be unique")
        return self


def load_remote_dataset_spec(path: str | Path) -> RemoteDatasetSpec:
    try:
        return RemoteDatasetSpec.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid remote dataset spec {path}: {exc}") from exc


def load_remote_training_package(path: str | Path) -> RemoteTrainingPackage:
    try:
        return RemoteTrainingPackage.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid remote training package {path}: {exc}") from exc


def load_remote_dataset_receipt(path: str | Path) -> RemoteDatasetReceipt:
    try:
        return RemoteDatasetReceipt.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid remote dataset receipt {path}: {exc}") from exc


def load_remote_preparation_receipt(path: str | Path) -> RemotePreparationReceipt:
    try:
        return RemotePreparationReceipt.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid remote preparation receipt {path}: {exc}") from exc


def remote_preparation_signature(
    *,
    data_intent_signature: str,
    source: RemoteHuggingFaceSource,
    prepare_script_sha256: str,
    data_recipe_sha256: str,
) -> str:
    return hashlib.sha256(_canonical_json({
        "data_intent_signature": data_intent_signature,
        "source": source.model_dump(mode="json"),
        "prepare_script_sha256": prepare_script_sha256,
        "data_recipe_sha256": data_recipe_sha256,
    })).hexdigest()


def remote_data_signature(
    *,
    spec: RemoteDatasetSpec,
    recipe: dict[str, Any],
    prepare_script_sha256: str,
    blocked_semantic_fingerprints: Iterable[str] = (),
) -> str:
    """Identity of immutable source + executable recipe + scoring exclusion.

    Remote bytes do not exist on the scheduler.  The receipt later binds this
    intent identity to the actual remote file SHA-256 and row count.
    """
    realized_recipe = dict(recipe)
    realized_recipe.pop("direction", None)
    realized_recipe.pop("audit_steps", None)
    realized_recipe.pop("dataset_name", None)
    value = {
        "remote_dataset_spec": spec.model_dump(mode="json"),
        "data_recipe": realized_recipe,
        "prepare_script_sha256": prepare_script_sha256,
        "blocked_semantic_fingerprints": sorted(set(blocked_semantic_fingerprints)),
    }
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def remote_source_fingerprint(spec: RemoteDatasetSpec) -> str:
    """Canonical immutable identity copied into DataRecipe.source_fingerprint."""
    return hashlib.sha256(_canonical_json(spec.source.model_dump(mode="json"))).hexdigest()


def build_remote_training_package(
    *,
    spec: RemoteDatasetSpec,
    recipe: dict[str, Any],
    prepare_script_path: str | Path,
    blocked_semantic_fingerprints: Iterable[str],
    data_intent_signature: str,
    recipe_path: str | Path | None = None,
) -> RemoteTrainingPackage:
    script_hash = file_sha256(prepare_script_path)
    recipe_for_hash = recipe
    if recipe_path is not None:
        try:
            loaded_recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid remote data recipe {recipe_path}: {exc}") from exc
        if not isinstance(loaded_recipe, dict):
            raise ValueError("remote data recipe must be a JSON object")
        recipe_for_hash = loaded_recipe
    recipe_hash = hashlib.sha256(_canonical_json(recipe_for_hash)).hexdigest()
    blocked = sorted(set(blocked_semantic_fingerprints))
    signature = remote_data_signature(
        spec=spec,
        recipe=recipe,
        prepare_script_sha256=script_hash,
        blocked_semantic_fingerprints=blocked,
    )
    return RemoteTrainingPackage(
        source=spec.source,
        training_method=spec.training_method,
        method_format=spec.method_format,
        data_intent_signature=data_intent_signature,
        prepare_script_sha256=script_hash,
        data_recipe_sha256=recipe_hash,
        data_signature=signature,
        blocked_semantic_fingerprints=blocked,
    )


_SEMANTIC_METADATA_KEYS = {
    "id", "row_id", "example_id", "index", "source", "split", "category",
    "num_turns", "role",
}

_QUESTION_FIELD_KEYS = {
    "question", "question_content", "prompt", "problem", "query", "input",
    "instruction", "context", "passage", "text", "choices", "options",
}

_ANSWER_FIELD_KEYS = {
    "answer", "answers", "response", "responses", "output", "completion",
    "target", "label", "labels", "solution", "rationale", "reference",
    "reference_answer", "expected", "gold", "ground_truth",
}


def _semantic_text(value: Any, *, key: str = "") -> list[str]:
    if key.lower() in _SEMANTIC_METADATA_KEYS:
        return []
    if isinstance(value, dict):
        parts: list[str] = []
        for child_key, child in value.items():
            parts.extend(_semantic_text(child, key=str(child_key)))
        return parts
    if isinstance(value, list):
        parts: list[str] = []
        for child in value:
            parts.extend(_semantic_text(child))
        return parts
    if isinstance(value, str) and value.strip():
        return [" ".join(value.split()).casefold()]
    return []


def _semantic_fingerprint(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if isinstance(message, dict):
                role = str(message.get("role") or "").strip().casefold()
                if role in {"user", "human"}:
                    parts.extend(_semantic_text(message.get("content")))
    elif isinstance(record.get("instruction"), dict):
        instructions = record["instruction"]
        parts = []
        for turn in instructions:
            parts.extend(_semantic_text(instructions.get(turn)))
    else:
        selected = {
            key: value for key, value in record.items()
            if key.casefold() in _QUESTION_FIELD_KEYS
        }
        if not selected:
            selected = {
                key: value for key, value in record.items()
                if key.casefold() not in _ANSWER_FIELD_KEYS
            }
        parts = _semantic_text(selected)
    text = "\n".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def _scan_jsonl(
    path: Path, *, blocked: set[str] | None = None,
) -> tuple[int, list[str], int, str]:
    """Validate a JSONL artifact without retaining a multi-GB dataset in RAM."""
    n_rows = 0
    removed = 0
    fields: list[str] = []
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(row, dict) or not row:
                raise ValueError(
                    f"remote prepared row {line_number} must be a non-empty object"
                )
            for key in row:
                if key not in fields:
                    fields.append(key)
            n_rows += 1
            if blocked and _semantic_fingerprint(row) in blocked:
                removed += 1
    if not n_rows:
        raise ValueError("remote preparation produced zero training rows")
    return n_rows, fields, removed, digest.hexdigest()


def _filter_jsonl_atomic(
    source: Path, destination: Path, *, blocked: set[str],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            with source.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid JSONL at line {line_number}: {exc}"
                        ) from exc
                    if _semantic_fingerprint(row) in blocked:
                        continue
                    output.write(json.dumps(
                        row, ensure_ascii=False, separators=(",", ":"),
                    ))
                    output.write("\n")
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _link_reflink_or_copy_atomic(source: Path, destination: Path) -> str:
    """Publish immutable bytes cheaply while retaining an independent path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent),
    )
    os.close(fd)
    os.unlink(temporary)
    try:
        try:
            os.link(source, temporary)
            mode = "hardlink"
        except OSError:
            try:
                subprocess.run(
                    ["cp", "--reflink=always", "--", str(source), temporary],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                mode = "reflink"
            except (OSError, subprocess.CalledProcessError):
                shutil.copy2(source, temporary)
                mode = "copy"
        os.replace(temporary, destination)
        return mode
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def record_remote_preparation(
    *,
    data_intent_signature: str,
    spec_path: str | Path,
    recipe_path: str | Path,
    prepare_script_path: str | Path,
    dataset_path: str | Path,
    profile_path: str | Path,
    receipt_path: str | Path,
) -> RemotePreparationReceipt:
    """Persist exact reusable evidence immediately after expensive preparation."""
    spec = load_remote_dataset_spec(spec_path)
    try:
        recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid remote data recipe {recipe_path}: {exc}") from exc
    if not isinstance(recipe, dict):
        raise ValueError("remote data recipe must be a JSON object")
    dataset = Path(dataset_path).resolve()
    profile = Path(profile_path).resolve()
    if not profile.is_file() or profile.stat().st_size == 0:
        raise ValueError("remote preparation profile is missing or empty")
    n_rows, fields, _removed, dataset_hash = _scan_jsonl(dataset)
    script_hash = file_sha256(prepare_script_path)
    recipe_hash = hashlib.sha256(_canonical_json(recipe)).hexdigest()
    receipt = RemotePreparationReceipt(
        data_intent_signature=data_intent_signature,
        source=spec.source,
        prepare_script_sha256=script_hash,
        data_recipe_sha256=recipe_hash,
        preparation_signature=remote_preparation_signature(
            data_intent_signature=data_intent_signature,
            source=spec.source,
            prepare_script_sha256=script_hash,
            data_recipe_sha256=recipe_hash,
        ),
        dataset_path=str(dataset),
        dataset_sha256=dataset_hash,
        profile_path=str(profile),
        profile_sha256=file_sha256(profile),
        n_rows=n_rows,
        record_fields=fields,
    )
    _write_json_atomic(Path(receipt_path).resolve(), receipt.model_dump(mode="json"))
    return receipt


def validate_remote_preparation(
    *,
    receipt_path: str | Path,
    data_intent_signature: str,
    spec_path: str | Path,
    recipe_path: str | Path,
    prepare_script_path: str | Path,
    dataset_path: str | Path,
    profile_path: str | Path,
) -> RemotePreparationReceipt:
    """Accept a cached preparation only when identity, schema and bytes match."""
    receipt = load_remote_preparation_receipt(receipt_path)
    spec = load_remote_dataset_spec(spec_path)
    try:
        recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid remote data recipe {recipe_path}: {exc}") from exc
    if not isinstance(recipe, dict):
        raise ValueError("remote data recipe must be a JSON object")
    dataset = Path(dataset_path).resolve()
    profile = Path(profile_path).resolve()
    script_hash = file_sha256(prepare_script_path)
    recipe_hash = hashlib.sha256(_canonical_json(recipe)).hexdigest()
    expected_signature = remote_preparation_signature(
        data_intent_signature=data_intent_signature,
        source=spec.source,
        prepare_script_sha256=script_hash,
        data_recipe_sha256=recipe_hash,
    )
    n_rows, fields, _removed, dataset_hash = _scan_jsonl(dataset)
    expected = {
        "data_intent_signature": data_intent_signature,
        "source": spec.source,
        "prepare_script_sha256": script_hash,
        "data_recipe_sha256": recipe_hash,
        "preparation_signature": expected_signature,
        "dataset_path": str(dataset),
        "dataset_sha256": dataset_hash,
        "profile_path": str(profile),
        "profile_sha256": file_sha256(profile),
        "n_rows": n_rows,
        "record_fields": fields,
    }
    mismatched = [
        name for name, value in expected.items()
        if getattr(receipt, name) != value
    ]
    if mismatched:
        raise ValueError(
            "remote preparation receipt does not match current artifacts: "
            + ", ".join(mismatched)
        )
    return receipt


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def materialize_remote_training_data(
    *,
    package_path: str | Path,
    prepare_script_path: str | Path,
    output_path: str | Path,
    cache_dir: str | Path,
    receipt_path: str | Path,
) -> RemoteDatasetReceipt:
    package = load_remote_training_package(package_path)
    prepare_script = Path(prepare_script_path).resolve()
    if file_sha256(prepare_script) != package.prepare_script_sha256:
        raise ValueError("remote prepare script differs from the engine-bound SHA-256")

    output = Path(output_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    profile = output.with_name("data_profile.json")
    if output.is_file() and receipt_file.is_file():
        previous = load_remote_dataset_receipt(receipt_file)
        if (
            previous.data_signature == package.data_signature
            and previous.dataset_path == str(output)
            and previous.dataset_sha256 == file_sha256(output)
        ):
            return previous

    output.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    raw = output.with_name(f".{output.name}.before-decontamination.jsonl")
    command = [
        sys.executable,
        str(prepare_script),
        "--dataset-id", package.source.dataset_id,
        "--dataset-split", package.source.split,
        "--dataset-revision", package.source.revision,
        "--cache-dir", str(cache),
        "--output", str(raw),
        "--profile", str(profile),
    ]
    if package.source.config:
        command.extend(["--dataset-config", package.source.config])
    environment = dict(os.environ)
    environment.setdefault("HF_HOME", str(cache))
    subprocess.run(command, check=True, env=environment)
    try:
        return finalize_remote_training_data(
            package_path=package_path,
            prepared_path=raw,
            output_path=output,
            receipt_path=receipt_file,
            profile_path=profile,
        )
    finally:
        try:
            raw.unlink()
        except OSError:
            pass


def finalize_remote_training_data(
    *,
    package_path: str | Path,
    prepared_path: str | Path,
    output_path: str | Path,
    receipt_path: str | Path,
    profile_path: str | Path = "",
    preparation_receipt_path: str | Path = "",
) -> RemoteDatasetReceipt:
    """Finalize an existing remote dataset without exposing scoring hashes to Data."""
    package = load_remote_training_package(package_path)
    prepared = Path(prepared_path).resolve()
    output = Path(output_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    profile = (
        Path(profile_path).resolve()
        if str(profile_path)
        else output.with_name("data_profile.json")
    )
    if output.is_file() and receipt_file.is_file():
        previous = load_remote_dataset_receipt(receipt_file)
        if (
            previous.data_signature == package.data_signature
            and previous.dataset_path == str(output)
            and previous.dataset_sha256 == file_sha256(output)
        ):
            return previous

    blocked = set(package.blocked_semantic_fingerprints)
    n_rows_before, fields, removed, prepared_hash = _scan_jsonl(
        prepared, blocked=blocked,
    )
    n_rows = n_rows_before - removed
    if not n_rows:
        raise ValueError(
            "all remotely prepared Training rows duplicate a hidden scoring population"
        )
    if not profile.is_file():
        _write_json_atomic(profile, {
            "schema_version": 1,
            "n_rows_before_decontamination": n_rows_before,
            "record_fields": fields,
        })

    if preparation_receipt_path:
        preparation_receipt = Path(preparation_receipt_path).resolve()
        expected_preparation = RemotePreparationReceipt(
            data_intent_signature=package.data_intent_signature,
            source=package.source,
            prepare_script_sha256=package.prepare_script_sha256,
            data_recipe_sha256=package.data_recipe_sha256,
            preparation_signature=remote_preparation_signature(
                data_intent_signature=package.data_intent_signature,
                source=package.source,
                prepare_script_sha256=package.prepare_script_sha256,
                data_recipe_sha256=package.data_recipe_sha256,
            ),
            dataset_path=str(prepared),
            dataset_sha256=prepared_hash,
            profile_path=str(profile),
            profile_sha256=file_sha256(profile),
            n_rows=n_rows_before,
            record_fields=fields,
        )
        if preparation_receipt.is_file():
            existing_preparation = load_remote_preparation_receipt(
                preparation_receipt
            )
            if existing_preparation != expected_preparation:
                raise ValueError(
                    "remote preparation receipt is partial, stale, or semantically different"
                )
        else:
            _write_json_atomic(
                preparation_receipt,
                expected_preparation.model_dump(mode="json"),
            )

    materialization_mode: Literal["rewrite", "hardlink", "reflink", "copy"]
    if removed:
        _filter_jsonl_atomic(prepared, output, blocked=blocked)
        materialization_mode = "rewrite"
        output_hash = file_sha256(output)
    else:
        materialization_mode = _link_reflink_or_copy_atomic(prepared, output)
        output_hash = prepared_hash
    receipt = RemoteDatasetReceipt(
        data_signature=package.data_signature,
        dataset_path=str(output),
        dataset_sha256=output_hash,
        profile_path=str(profile),
        n_rows_before_decontamination=n_rows_before,
        n_rows=n_rows,
        decontamination_removed_rows=removed,
        record_fields=fields,
        materialization_mode=materialization_mode,
    )
    _write_json_atomic(receipt_file, receipt.model_dump(mode="json"))
    return receipt


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zevo remote training-data helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-spec")
    validate.add_argument("spec_path")
    validate.add_argument("--dataset-id", required=True)
    validate.add_argument("--dataset-split", required=True)
    validate.add_argument("--dataset-config", default="")
    validate.add_argument("--training-method", required=True)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--package", required=True)
    materialize.add_argument("--prepare-script", required=True)
    materialize.add_argument("--output", required=True)
    materialize.add_argument("--cache-dir", required=True)
    materialize.add_argument("--receipt", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--package", required=True)
    finalize.add_argument("--prepared", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--receipt", required=True)
    finalize.add_argument("--profile", default="")
    finalize.add_argument("--preparation-receipt", default="")

    record_prepared = subparsers.add_parser("record-prepared")
    record_prepared.add_argument("--data-intent-signature", required=True)
    record_prepared.add_argument("--spec", required=True)
    record_prepared.add_argument("--recipe", required=True)
    record_prepared.add_argument("--prepare-script", required=True)
    record_prepared.add_argument("--dataset", required=True)
    record_prepared.add_argument("--profile", required=True)
    record_prepared.add_argument("--receipt", required=True)

    check_prepared = subparsers.add_parser("check-prepared")
    check_prepared.add_argument("--data-intent-signature", required=True)
    check_prepared.add_argument("--spec", required=True)
    check_prepared.add_argument("--recipe", required=True)
    check_prepared.add_argument("--prepare-script", required=True)
    check_prepared.add_argument("--dataset", required=True)
    check_prepared.add_argument("--profile", required=True)
    check_prepared.add_argument("--receipt", required=True)

    args = parser.parse_args(argv)
    if args.command == "validate-spec":
        spec = load_remote_dataset_spec(args.spec_path)
        expected = {
            "dataset_id": args.dataset_id,
            "split": args.dataset_split,
            "config": args.dataset_config,
            "training_method": args.training_method,
        }
        actual = {
            "dataset_id": spec.source.dataset_id,
            "split": spec.source.split,
            "config": spec.source.config,
            "training_method": spec.training_method,
        }
        if actual != expected:
            raise ValueError(
                f"remote dataset spec differs from the work order: {actual!r} != {expected!r}"
            )
        print(json.dumps({
            "spec": spec.model_dump(mode="json"),
            "source_fingerprint": remote_source_fingerprint(spec),
        }, sort_keys=True))
        return 0
    if args.command in {"record-prepared", "check-prepared"}:
        operation = (
            record_remote_preparation
            if args.command == "record-prepared"
            else validate_remote_preparation
        )
        receipt = operation(
            data_intent_signature=args.data_intent_signature,
            spec_path=args.spec,
            recipe_path=args.recipe,
            prepare_script_path=args.prepare_script,
            dataset_path=args.dataset,
            profile_path=args.profile,
            receipt_path=args.receipt,
        )
        print(receipt.model_dump_json())
        return 0

    if args.command == "materialize":
        receipt = materialize_remote_training_data(
            package_path=args.package,
            prepare_script_path=args.prepare_script,
            output_path=args.output,
            cache_dir=args.cache_dir,
            receipt_path=args.receipt,
        )
    else:
        receipt = finalize_remote_training_data(
            package_path=args.package,
            prepared_path=args.prepared,
            output_path=args.output,
            receipt_path=args.receipt,
            profile_path=args.profile,
            preparation_receipt_path=args.preparation_receipt,
        )
    print(receipt.model_dump_json())
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised on the remote host
    raise SystemExit(_main())
