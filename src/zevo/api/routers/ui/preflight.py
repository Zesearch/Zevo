"""POST /api/preflight — pre-run readiness check.

Given a UserRequest (or just a partial draft of one), return:
  - status: "ready" | "risky" | "blocked"
  - blockers[]: hard fails that will make the run die immediately
  - risks[]:    soft warnings the user should see before paying
  - info[]:     things we successfully verified, for transparency

Cheap (~100ms): file existence, env var presence, model-id family
lookup, eval-script importability. Does NOT call any provider API
or spawn any agents. Designed for inline UI use on the Create Run page.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.database import get_db
from zevo.db import Agent, InfraInstance
from zevo.contracts.customizations import RunCustomizations
from zevo.contracts.orchestrator import (
    BUILTIN_METRICS,
    UserRequest,
    inherit_test_validation_contract,
    scoring_asset_errors,
)
from zevo.contracts.training_methods import (
    AUXILIARY_MODEL_FIELD_BY_METHOD,
    is_huggingface_model_id,
    method_config_errors,
)
from zevo.holdout_storage import resolve_asset


router = APIRouter()


Severity = Literal["info", "risk", "blocker"]


class PreflightItem(BaseModel):
    severity: Severity
    code: str       # short machine-readable id, e.g. "dataset_missing"
    message: str    # one-line human description
    hint: str = ""  # actionable advice (optional)


class PreflightResponse(BaseModel):
    status: Literal["ready", "risky", "blocked"]
    items: list[PreflightItem]
    blockers: list[str]   # codes only, for quick counting
    risks: list[str]
    summary: str          # one-sentence verdict


class PreflightBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_request: UserRequest
    # Mirror CreateRunRequest: execution runtime is top-level and never part of
    # UserRequest. A missing provider resolves to the same safe default used by
    # run creation for a free-form run.
    gpu_provider: Literal["cluster", "cloud", "instance"] | None = None
    num_gpus: int | None = Field(None, ge=0)
    generation_backend: Literal["hf", "vllm"] | None = None
    customizations: RunCustomizations | None = None


# Training-model families accepted by the backend. The frontend agent-model
# catalog is a different concept: it chooses the model driving an Agent.
KNOWN_BASE_MODEL_PREFIXES = (
    # HF base-model families the train agent has been tested with
    "Qwen/", "meta-llama/", "mistralai/", "google/gemma", "microsoft/",
    "deepseek-ai/", "stabilityai/", "tiiuae/", "01-ai/", "allenai/",
)


def _exists(path: str) -> bool:
    if not path:
        return False
    return Path(path).exists()


def _ok(items: list[PreflightItem], code: str, msg: str) -> None:
    items.append(PreflightItem(severity="info", code=code, message=msg))


def _risk(items: list[PreflightItem], code: str, msg: str, hint: str = "") -> None:
    items.append(PreflightItem(severity="risk", code=code, message=msg, hint=hint))


def _block(items: list[PreflightItem], code: str, msg: str, hint: str = "") -> None:
    items.append(PreflightItem(severity="blocker", code=code, message=msg, hint=hint))


def _check_objective(req: UserRequest, items: list[PreflightItem]) -> None:
    obj = (req.task_objective or "").strip()
    if not obj:
        _block(items, "task_objective_empty",
               "Task objective is empty.",
               "Describe what model behaviour you want, e.g. 'Fine-tune a Q&A model on the pasta dataset'.")
    elif len(obj) < 10:
        _risk(items, "task_objective_terse",
              f"Task objective is only {len(obj)} chars; orchestrator may need to ask for clarification.")
    else:
        _ok(items, "task_objective_ok", f"Task objective set ({len(obj)} chars).")


def _check_dataset(req: UserRequest, items: list[PreflightItem]) -> None:
    if req.dataset:
        if not _exists(req.dataset):
            _block(items, "dataset_missing",
                   f"Dataset path {req.dataset!r} does not exist (checked from backend container).",
                   "Check the path or re-upload via /files.")
        else:
            size = Path(req.dataset).stat().st_size
            if size == 0:
                _block(items, "dataset_empty",
                       f"Dataset file is 0 bytes: {req.dataset}")
            elif size < 1024:
                _risk(items, "dataset_tiny",
                      f"Dataset is only {size} bytes; you may want more data for meaningful training.")
            else:
                _ok(items, "dataset_ok", f"Dataset present ({size:,} bytes).")
    elif req.data_query:
        _ok(items, "dataset_acquire",
            "No dataset path provided; the data agent will acquire from the query.")
    else:
        _risk(items, "no_data_source",
              "Neither `dataset` nor `data_query` is set; the data agent will need to make decisions.",
              "Either upload a dataset or describe what to acquire in `data_query`.")


def _check_eval(req: UserRequest, items: list[PreflightItem]) -> None:
    contracts = (
        (
            "test", "Test", req.metric_type, req.metric,
            req.evaluation_script,
        ),
        (
            "validation", "Validation", req.validation_metric_type,
            req.validation_metric, req.validation_evaluation_script,
        ),
    )
    for code, label, metric_type, metric, eval_script in contracts:
        if not metric_type:
            _block(
                items, f"{code}_metric_type_required",
                f"Choose the {label} metric type.",
            )
            continue
        if not metric.strip():
            _block(
                items, f"{code}_metric_required",
                f"Choose the {label} metric.",
            )
            continue
        if metric_type == "builtin":
            if metric.strip().lower() not in BUILTIN_METRICS:
                _block(
                    items, f"unknown_{code}_builtin_metric",
                    f"{label} metric {metric!r} is not a Zevo built-in metric.",
                )
            else:
                _ok(
                    items, f"{code}_builtin_metric",
                    f"{label} will use built-in {metric.strip().lower()}.",
                )
            continue
        if not eval_script:
            _block(
                items, f"{code}_custom_metric_requires_script",
                f"Custom {label} metrics require an evaluator script.",
            )
            continue
        if not _exists(eval_script):
            _block(
                items, f"{code}_evaluation_script_missing",
                f"{label} evaluator path {eval_script!r} does not exist.",
            )
            continue
        # Static preflight proves readability and syntax. The fixed protocol
        # and sample-submission binding are verified by the actual runner.
        try:
            src = Path(eval_script).read_text(encoding="utf-8")
            if not src.strip():
                _block(
                    items, f"{code}_evaluation_script_empty",
                    f"{label} evaluator is empty: {eval_script}",
                )
                continue
            compile(src, eval_script, "exec")
            _ok(
                items, f"{code}_evaluation_script_ok",
                f"The frozen {label} evaluator is readable Python.",
            )
        except SyntaxError as e:
            _block(
                items, f"{code}_evaluation_script_syntax",
                f"{label} evaluator has invalid Python syntax: "
                f"{e.msg} (line {e.lineno}).",
            )
        except OSError as e:
            _block(
                items, f"{code}_evaluation_script_unreadable",
                f"{label} evaluator cannot be read: {e}",
            )


def _check_test_set(req: UserRequest, items: list[PreflightItem]) -> None:
    if not req.test_set:
        # scoring_asset_errors emits the canonical required-field blocker. This
        # helper owns path validation only, so do not duplicate that message.
        return
    resolved = resolve_asset(req.test_set)
    if not _exists(resolved):
        _block(items, "test_set_missing",
               f"test_set path {req.test_set!r} does not exist.")
        return
    if not req.validation_set:
        try:
            from zevo.engine.method.validation_split import (
                MIN_VALIDATION_ROWS, _n_validation, _read_raw,
            )
            _columns, rows = _read_raw(Path(resolved))
            selected = _n_validation(len(rows))
            sample_path = Path(resolve_asset(req.test_sample_submission))
            with sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
                sample = csv.DictReader(handle)
                sample_columns = list(sample.fieldnames or [])
                sample_rows = sum(1 for _ in sample)
            if not sample_columns or sample_rows < 1:
                raise ValueError(
                    "Test sample submission must be a non-empty CSV with a "
                    "header and at least one example row"
                )
            if len(sample_columns) != len(set(sample_columns)):
                raise ValueError("Test sample submission contains duplicate columns")
        except Exception as exc:
            _block(
                items, "validation_split_too_small",
                f"Validation cannot be derived from this Test set: {exc}",
                "Upload a separate Validation set with its answer fields and sample submission.",
            )
        else:
            _ok(
                items, "validation_split_ready",
                f"Zevo will derive {selected} Validation rows (20% of Test; minimum {MIN_VALIDATION_ROWS}).",
            )


def _check_scoring_assets(req: UserRequest, items: list[PreflightItem]) -> None:
    for i, message in enumerate(scoring_asset_errors(req)):
        _block(items, f"scoring_assets_{i + 1}", message)


def _check_base_model(req: UserRequest, items: list[PreflightItem]) -> None:
    bm = (req.base_model or "").strip()
    if not bm:
        _ok(items, "base_model_default",
            "No base_model specified; Orchestrator must recommend one and baseline Inference will validate it.")
        return
    if not is_huggingface_model_id(bm):
        _block(
            items, "base_model_not_huggingface",
            f"base_model={bm!r} is not a Hugging Face owner/model id.",
            "Use an explicit id such as 'Qwen/Qwen2.5-1.5B-Instruct'; "
            "aliases, URLs, paths, uploads, and Registry tags are not "
            "supported here.",
        )
        return
    if any(bm.startswith(p) or p.lower() in bm.lower() for p in KNOWN_BASE_MODEL_PREFIXES):
        _ok(items, "base_model_known", f"base_model={bm!r} matches a known family.")
    else:
        _risk(items, "base_model_unknown",
              f"base_model={bm!r} doesn't match a known family; download/load may fail.",
              "Pick a HF model id like 'Qwen/Qwen2.5-1.5B-Instruct' or 'meta-llama/Llama-3.1-8B-Instruct'.")


def _check_method_config(req: UserRequest, items: list[PreflightItem]) -> None:
    errors = method_config_errors(
        req.training_method, req.method_config, require_dependencies=True,
    )
    if errors:
        for index, message in enumerate(errors, 1):
            _block(
                items,
                f"method_config_{index}",
                message,
                "Choose the required auxiliary model from Hugging Face and use its owner/model id.",
            )
        return

    method = (req.training_method or "").strip().lower()
    field = AUXILIARY_MODEL_FIELD_BY_METHOD.get(method)
    if field:
        model_id = str((req.method_config or {}).get(field) or "").strip()
        _ok(items, f"{field}_ok", f"{field}={model_id!r} is a Hugging Face model id.")
        _risk(
            items,
            "auxiliary_model_memory",
            f"{method} loads an auxiliary model in addition to the trainable policy.",
            "Ensure the selected GPU allocation has enough memory for both models and training state.",
        )


def _remote_root_error(prefix: str) -> str:
    root = os.environ.get(f"ZEVO_{prefix}_REMOTE_DIR", "").strip()
    user = os.environ.get(f"ZEVO_{prefix}_SSH_USER", "").strip()
    if not root:
        return f"ZEVO_{prefix}_REMOTE_DIR is required"
    if root.startswith("~") or not root.startswith("/"):
        return f"ZEVO_{prefix}_REMOTE_DIR must be an absolute remote path"
    if user and any(
        root == home or root.startswith(home + "/")
        for home in (f"/home/{user}", f"/Users/{user}")
    ):
        return f"ZEVO_{prefix}_REMOTE_DIR must be outside the remote user's HOME"
    return ""


def _check_infra(
    gpu_provider: str | None,
    items: list[PreflightItem],
    num_gpus: int | None = None,
    generation_backend: str | None = None,
) -> None:
    # Free-form run creation resolves a missing provider to `instance`. Never
    # infer a billable provider merely because credentials happen to exist.
    gp = (gpu_provider or "instance").strip().lower()
    if gp == "cloud":
        # The preference is deployment context, not a provision payload value.
        # With no preference, Infrastructure chooses among every credentialed
        # backend using live feasibility, price, and budget evidence.
        backend = str(
            os.environ.get("ZEVO_CLOUD_BACKEND", "")
        ).strip().lower()
        has_vast = bool(os.environ.get("VASTAI_API_KEY", "").strip())
        has_lambda = bool(
            os.environ.get("LAMBDA_API_KEY", "").strip()
            or os.environ.get("LAMBDA_CLOUD_API_KEY", "").strip()
        )
        if backend not in ("", "vastai", "lambda"):
            _block(
                items,
                "cloud_backend_invalid",
                "ZEVO_CLOUD_BACKEND must be empty, vastai, or lambda.",
                "Clear it to let Infrastructure choose, or set a supported preference.",
            )
        elif backend == "lambda":
            if not has_lambda:
                _block(items, "cloud_no_credentials",
                       "gpu_provider=cloud, ZEVO_CLOUD_BACKEND=lambda but LAMBDA_API_KEY is not set.",
                       "Add LAMBDA_API_KEY via the /settings page (and optionally LAMBDA_SSH_KEY_NAME), "
                       "then restart the backend + scheduler.")
            else:
                _ok(items, "cloud_ready",
                    "gpu_provider=cloud (Lambda Cloud) and LAMBDA_API_KEY is present.")
        elif backend == "vastai":
            if not has_vast:
                _block(items, "cloud_no_credentials",
                       "gpu_provider=cloud (Vast.ai) but VASTAI_API_KEY is not set in the backend container.",
                       "Add VASTAI_API_KEY via the /settings page, then restart the backend + scheduler. "
                       "(Or set ZEVO_CLOUD_BACKEND=lambda + LAMBDA_API_KEY to rent on Lambda Cloud instead.)")
            else:
                _ok(items, "cloud_ready",
                    "gpu_provider=cloud (Vast.ai) and VASTAI_API_KEY is present.")
        elif not (has_vast or has_lambda):
            _block(
                items,
                "cloud_no_credentials",
                "gpu_provider=cloud but no supported cloud credential is configured.",
                "Add VASTAI_API_KEY or LAMBDA_API_KEY via Settings, then restart backend + scheduler.",
            )
        else:
            available = [
                name for name, enabled in (
                    ("Vast.ai", has_vast), ("Lambda Cloud", has_lambda),
                ) if enabled
            ]
            _ok(
                items,
                "cloud_ready",
                "gpu_provider=cloud; Infrastructure will select from: "
                + ", ".join(available) + ".",
            )
    elif gp == "cluster":
        if not os.environ.get("ZEVO_CLUSTER_SSH_HOST", "").strip():
            _block(items, "cluster_no_host",
                   "gpu_provider=cluster but ZEVO_CLUSTER_SSH_HOST is not set in the backend container.",
                   "Set ZEVO_CLUSTER_SSH_HOST (and optionally _PORT/_USER/_KEY) in .env, then "
                   "restart the backend + scheduler.")
        elif error := _remote_root_error("CLUSTER"):
            _block(items, "cluster_remote_dir", error)
        else:
            _ok(items, "cluster_ready",
                f"gpu_provider=cluster; will SSH to {os.environ.get('ZEVO_CLUSTER_SSH_HOST')} "
                "and probe nvidia-smi at run time.")
    elif gp == "instance":
        if not os.environ.get("ZEVO_INSTANCE_SSH_HOST", "").strip():
            _block(items, "instance_no_host",
                   "gpu_provider=instance but ZEVO_INSTANCE_SSH_HOST is not set in the backend container.",
                   "Set ZEVO_INSTANCE_SSH_HOST + ZEVO_INSTANCE_SSH_USER in .env, then restart the backend + "
                   "scheduler. The host must be a fixed directly reachable GPU machine, not a Slurm login node.")
        elif error := _remote_root_error("INSTANCE"):
            _block(items, "instance_remote_dir", error)
        else:
            _ok(items, "instance_ready",
                f"gpu_provider=instance; will use the fixed GPU host "
                f"{os.environ.get('ZEVO_INSTANCE_SSH_HOST')} directly without Slurm.")

    # These are Run-envelope fields, so preflight must resolve and report them
    # instead of merely accepting them in its request schema. Provider-specific
    # capacity is verified by Infrastructure against the real allocation/offer;
    # here we make the effective upper bound explicit before launch.
    effective_gpus = max(0, int(num_gpus or 0))
    effective_backend = (generation_backend or "vllm").strip().lower()
    _ok(
        items,
        "gpu_count",
        (
            f"Run may use up to {effective_gpus} GPU(s)."
            if effective_gpus else "Run has no GPU-count limit."
        ),
    )
    _ok(items, "generation_backend", f"Generation backend is {effective_backend}.")


def _check_dataset_profile(req: UserRequest, items: list[PreflightItem]) -> None:
    """If the user pointed at a file under data/files/, lift
    the cached profile and surface any issues the profiler flagged.
    This is the bridge from F.1-F.4 → preflight: warnings the
    orchestrator can also see in its payload."""
    if not req.dataset or not Path(req.dataset).exists():
        return
    try:
        from zevo.engine.dataset_profiler import load_cached, is_cache_valid
    except ImportError:
        return
    p = Path(req.dataset)
    dataset_dir = p.parent
    if not (dataset_dir / ".profile.json").exists():
        # Not auto-profiled (e.g. PDF, or upload didn't trigger the
        # background task yet) — silently skip; the agent will inspect.
        return
    cached = load_cached(dataset_dir)
    if cached is None or not is_cache_valid(dataset_dir, p, cached):
        return

    # Mirror profile-reported issues into preflight items so the
    # Create-Run page surfaces them.
    for issue in cached.issues:
        sev = "blocker" if issue.severity == "error" else "risk"
        # Don't double-block: if the profile says the Data Agent can proceed,
        # demote 'error' issues to risks (they're recoverable).
        if cached.ready_for_data and sev == "blocker":
            sev = "risk"
        if sev == "blocker":
            _block(items, f"dataset_{issue.code}",
                   f"Dataset issue: {issue.message}",
                   "See the file profile on /files for details.")
        else:
            _risk(items, f"dataset_{issue.code}",
                  f"Dataset note: {issue.message}")

    # If profile classified the task with high confidence, surface that
    # as info so the user knows what the orchestrator will see.
    if cached.task_type.confidence >= 0.8:
        _ok(items, "dataset_profile_ok",
            f"Dataset profiled: {cached.task_type.label} "
            f"({cached.expected_format}), {cached.n_rows:,} rows, "
            f"recommended split {cached.split_recommendation.train_size}/"
            f"{cached.split_recommendation.val_size}.")


async def _check_orphaned_instances(items: list[PreflightItem], db: AsyncSession) -> None:
    """Warn if there are infra_instances rows older than 4h that haven't
    been released — likely leaks from prior runs costing real money."""
    now = datetime.now(timezone.utc)
    rows = (await db.execute(
        select(InfraInstance).where(
            InfraInstance.released_at.is_(None),
            InfraInstance.dph > 0,  # ignore free local probes
        )
    )).scalars().all()
    orphans: list[tuple[str, float]] = []
    for r in rows:
        age_h = (now - r.created_at).total_seconds() / 3600.0 if r.created_at else 0
        if age_h >= 4.0:
            spent = (r.dph or 0.0) * age_h
            orphans.append((f"{r.provider}:{r.instance_id}", spent))
    if orphans:
        total = sum(s for _, s in orphans)
        _risk(items, "orphaned_instances",
              f"{len(orphans)} GPU instance(s) active >4h, costing ~${total:.2f} so far.",
              "Check /infra/instances and release them via the infrastructure agent.")


def _driver_ready(driver: str) -> tuple[bool, str]:
    if driver == "claude_cli":
        try:
            has_claude_session = Path(
                "/root/.claude/.credentials.json"
            ).is_file()
        except OSError:
            has_claude_session = False
        ready = bool(
            os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
            or os.environ.get("ANTHROPIC_API_KEY", "").strip()
            or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
            or has_claude_session
        )
        return ready, "Claude subscription login or Anthropic API key"
    if driver == "codex_cli":
        from zevo.engine.agent.drivers.codex_auth import codex_auth_mode
        mode = codex_auth_mode("/root/.codex/auth.json")
        ready = mode in ("chatgpt", "api_key") or bool(os.environ.get("OPENAI_API_KEY", "").strip())
        return ready, "Codex ChatGPT login or OpenAI API key"
    if driver == "bedrock":
        ready = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()) or bool(
            os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
            and os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
        )
        return ready, "AWS Bedrock credentials"
    if driver == "openrouter":
        return bool(os.environ.get("OPENROUTER_API_KEY", "").strip()), "OpenRouter API key"
    if driver == "stub":
        return True, "stub driver"
    return False, f"supported driver (got {driver or 'unset'})"


async def _check_provider_creds(items: list[PreflightItem], db: AsyncSession) -> None:
    """Validate every public LLM Agent's driver, model, and credential.

    The execution-role table also contains deterministic system runners so
    Ticket foreign keys have one stable target. Those rows have manifests, not
    Agent identities, and must never be passed to ``load_agent``.
    """
    from zevo.engine.agent.loader import list_agent_ids, load_agent

    public_agent_ids = list_agent_ids()
    rows = (await db.execute(
        select(Agent)
        .where(Agent.id.in_(public_agent_ids))
        .order_by(Agent.id)
    )).scalars().all()
    rows_by_id = {row.id: row for row in rows}
    unregistered = sorted(set(public_agent_ids) - set(rows_by_id))
    if unregistered:
        _block(
            items,
            "agents_missing",
            "Missing Agent configurations: " + ", ".join(unregistered) + ".",
            "Restart the backend to seed Agent configurations from the Playbook.",
        )
        return
    if not public_agent_ids:
        _block(items, "agents_missing", "No Agent configurations exist in the database.")
        return
    missing: dict[str, list[str]] = {}
    ready_drivers: set[str] = set()
    for agent_id in public_agent_ids:
        row = rows_by_id[agent_id]
        blueprint = load_agent(agent_id)
        driver = (row.default_driver or blueprint.default_driver or "").strip()
        model = (row.default_model or blueprint.default_model or "").strip()
        if not model:
            missing.setdefault("a configured model", []).append(row.id)
            continue
        ready, requirement = _driver_ready(driver)
        if ready:
            ready_drivers.add(driver)
        else:
            missing.setdefault(requirement, []).append(row.id)
    for index, (requirement, agent_ids) in enumerate(missing.items(), 1):
        _block(
            items,
            f"agent_provider_not_ready_{index}",
            f"{', '.join(agent_ids)} require {requirement}.",
            "Configure the effective driver/model on Agents and its credential in Settings.",
        )
    if not missing:
        _ok(items, "provider_creds_ok", f"Configured Agent drivers are ready: {', '.join(sorted(ready_drivers))}.")


@router.post("/preflight", response_model=PreflightResponse)
async def preflight(body: PreflightBody, db: AsyncSession = Depends(get_db)) -> PreflightResponse:
    items: list[PreflightItem] = []
    req = inherit_test_validation_contract(body.user_request)

    _check_objective(req, items)
    _check_dataset(req, items)
    _check_dataset_profile(req, items)         # B.3: hoist profiler signals
    _check_test_set(req, items)
    _check_scoring_assets(req, items)
    _check_eval(req, items)
    _check_base_model(req, items)
    _check_method_config(req, items)
    _check_infra(
        body.gpu_provider,
        items,
        num_gpus=body.num_gpus,
        generation_backend=body.generation_backend,
    )
    await _check_provider_creds(items, db)
    await _check_orphaned_instances(items, db)  # B.3: surface GPU leaks

    blockers = [i.code for i in items if i.severity == "blocker"]
    risks = [i.code for i in items if i.severity == "risk"]

    if blockers:
        status = "blocked"
        summary = (
            f"{len(blockers)} blocker(s) must be resolved before this run can start: "
            f"{', '.join(blockers)}"
        )
    elif risks:
        status = "risky"
        summary = f"Run can start, but {len(risks)} risk(s) flagged: {', '.join(risks)}"
    else:
        status = "ready"
        summary = "All preflight checks passed. Safe to start."

    return PreflightResponse(
        status=status, items=items,
        blockers=blockers, risks=risks, summary=summary,
    )
