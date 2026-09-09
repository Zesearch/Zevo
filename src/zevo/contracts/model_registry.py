"""
Input/output contracts for the Registry Agent.

The Registry Agent runs ONCE after the optimization loop and writes a structured
entry about the Validation-selected champion into a YAML registry file. This makes each run
addressable later (e.g. for iterative Zevo model improvement: "is this model
better than M-84969a2c?").

A version tag is the stable `M-<run8>` identity of the producing Run. A better
iteration history lives in score events and Tickets, not extra model identities.
The API verifies the selected iteration is the Run's Validation champion before
this Ticket can be created. The tag is also the value shown by the UI and used
as the registry key.

The single Registry Ticket writes its immutable snapshot to
``<work_dir>/registry.yaml``. The database is the global champion index; the
Ticket receives the already-selected checkpoint instead of reading or mutating
a shared YAML file. Snapshot format:

  models:
    M-84969a2c:
      run_id: <full-run-id>
      ticket_id: <registry-ticket-id>
      iteration: <ticket-iteration>
      base_model: <actual-model-id-or-path>
      training_method: <actual-train-method>
      dataset_source: <source-lineage>
      model_path: <retained-local-path>
      task_objective: <user-objective>
      metric: <task-metric-name>
      metric_direction: max | min
      retention: retained
      eval:
        score: <finite-number-on-the-task-defined-scale>
        <other-metric-keys>: <preserved-values>
      registered_at: <utc-timestamp>
"""

from __future__ import annotations

import math
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from zevo.contracts._base import AgentResult, AgentTaskInput


def model_tag_for_run(run_id: str) -> str:
    """Return the one stable Registry identity owned by a Run."""
    value = (run_id or "").strip()
    if not value:
        raise ValueError("Registry requires a non-empty run_id")
    return f"M-{value[:8]}"


class RegistryEvalRecord(BaseModel):
    """Authoritative score plus evaluator-specific diagnostics."""

    model_config = ConfigDict(extra="allow")
    score: float = Field(allow_inf_nan=False)


class RegistryEntry(BaseModel):
    """Exact durable shape of one retained Run champion."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    ticket_id: str = Field(min_length=1)
    iteration: int = Field(ge=1)
    base_model: str = Field(min_length=1)
    training_method: str = Field(min_length=1)
    dataset_source: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    task_objective: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    metric_direction: Literal["max", "min"]
    eval: RegistryEvalRecord
    retention: Literal["retained"] = "retained"
    registered_at: datetime = Field(description="UTC ISO-8601 commit timestamp.")


def validate_registry_entry(
    registry_path: str | Path, *, run_id: str, version_tag: str,
) -> RegistryEntry:
    """Validate the exact retained entry without touching the database."""
    source = Path(registry_path)
    try:
        body = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read registry YAML {source}: {exc}") from exc
    if not isinstance(body, dict) or not isinstance(body.get("models"), dict):
        raise ValueError("registry YAML must contain a top-level models mapping")
    entry_body: Any = body["models"].get(version_tag)
    if not isinstance(entry_body, dict):
        raise ValueError(f"registry YAML has no mapping entry for {version_tag!r}")
    entry = RegistryEntry.model_validate(entry_body)
    if entry.run_id != run_id:
        raise ValueError(
            f"registry entry run_id differs: expected {run_id!r}, got {entry.run_id!r}"
        )
    if version_tag != model_tag_for_run(run_id):
        raise ValueError(
            f"version_tag differs from Run-owned tag {model_tag_for_run(run_id)!r}"
        )
    if not math.isfinite(entry.eval.score):  # defensive around alternate loaders
        raise ValueError("registry entry eval.score must be finite")
    return entry


def _main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate a Zevo registry champion entry")
    parser.add_argument("command", choices=["validate-entry"])
    parser.add_argument("registry_path")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--version-tag", required=True)
    args = parser.parse_args(argv)
    try:
        entry = validate_registry_entry(
            args.registry_path,
            run_id=args.run_id,
            version_tag=args.version_tag,
        )
    except ValueError as exc:
        print(f"INVALID RegistryEntry: {exc}", file=sys.stderr)
        return 1
    print(
        "VALID RegistryEntry "
        f"version_tag={args.version_tag} score={entry.eval.score}"
    )
    return 0


# ---------- Input ----------

class RegisterTaskInput(AgentTaskInput):
    """Mirror of orchestrator.RegisterPayload (after ref resolution)."""

    run_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Producing run id, supplied by the runner. Use it for provenance, "
            "stable model identity, and run-scoped retention; do not infer it "
            "from ticket_id."
        ),
    )
    iteration: int = Field(
        ...,
        ge=0,
        description="Exact Ticket iteration whose checkpoint is being considered.",
    )
    expected_version_tag: str = Field(
        ...,
        min_length=3,
        max_length=10,
        pattern=r"^M-.{1,8}$",
        description=(
            "Engine-computed stable model tag for this Run. Copy it exactly as "
            "the registry YAML key; do not derive or normalize it."
        ),
    )
    registry_entry_schema: dict[str, Any] = Field(
        ...,
        description=(
            "Exact JSON Schema for the value stored at "
            "registry.yaml.models[expected_version_tag]. This is the sole "
            "entry-key/type authority; evaluator-specific fields are allowed "
            "only inside eval alongside its required finite score."
        ),
    )
    registry_validation_command: str = Field(
        ...,
        min_length=1,
        description=(
            "Side-effect-free validation command. Replace only "
            "<absolute-registry-yaml-path>; run_id and expected_version_tag are "
            "already shell-quoted and inserted by the engine."
        ),
    )
    checkpoint_path: str = Field(
        ...,
        description=(
            "Resolved path to the trained model, from the upstream "
            "train ticket. '' is invalid -- can't register a missing model."
        ),
    )
    checkpoint_is_remote: bool = Field(
        False,
        description=(
            "True when checkpoint_path lives on the infrastructure target and "
            "must be read/copied with the credentials in device_info_path. False "
            "means it is a local directory."
        ),
    )
    metrics_path: str = Field(
        ...,
        description=(
            "Resolved path to the eval metrics JSON file registered by the runner "
            "from EvaluationResult.metrics_path. Canonical typed tickets "
            "require a readable metrics file with a finite numeric score."
        ),
    )
    train_config_path: str = Field(
        ...,
        min_length=1,
        description="Resolved train_config.yaml from the exact checkpoint Ticket.",
    )

    device_info_path: str = Field(
        "",
        description=(
            "Resolved path to device_info.json from the upstream infra ticket. "
            "Needed when `checkpoint_path` is a REMOTE path: registry uses "
            "the ssh creds here to pull the BEST iteration's model back to the "
            "host. '' when the checkpoint is already local (nothing to pull)."
        ),
    )

    base_model: str = Field(..., description="HF model id, echoed from the train ticket.")
    training_method: str = Field(..., description="Training method, echoed from the train ticket.")
    dataset_source: str = Field(
        ...,
        description=(
            "Host-visible original dataset/source lineage resolved by the runner. "
            "Copy it verbatim into registry.yaml; do not convert or re-resolve it."
        ),
    )
    task_objective: str = Field(
        ...,
        description=(
            "Original task problem statement copied from the supervisor "
            "payload's top-level task_objective, not its composed Agent objective."
        ),
    )
    metric: str = Field(
        ...,
        min_length=1,
        description="Name of the task's authoritative evaluator value.",
    )
    metric_direction: Literal["max", "min"] = Field(
        ...,
        description=(
            "Authoritative score direction already used by the engine to select "
            "this final Validation champion."
        ),
    )

    registry_path: str = Field(
        ...,
        description=(
            "Exact Ticket-local path <work_dir>/registry.yaml. Write one immutable "
            "selected-champion snapshot here; never read or update a shared registry."
        ),
    )
    work_dir: str = Field(..., description="Directory where the register.py helper script is written.")


# ---------- Output ----------

class RegisterResult(AgentResult):
    """What the Registry Agent reports back."""

    status: Literal["succeeded", "failed"]

    registry_path: str = Field(..., description="Absolute path to the Ticket-local registry manifest that was committed.")
    register_script_path: str = Field(..., description="Path to the generated register.py. '' on failure.")


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess tests
    raise SystemExit(_main())
