"""SQLAlchemy 2.0 models for Zevo.

Schema overview (see plan: peaceful-crafting-bentley.md, "Postgres schema"):

  agents            - registered agent definitions (one row per agent identity.md)
  runs              - pipeline runs initiated by `run create` or POST /runs
  tickets           - work units; one row per ticket in a run's DAG
  ticket_messages   - human/agent conversation attached to a ticket
  ticket_notices    - structured system notices attached to a ticket
  heartbeat_results - one structured agent result per activation
  agent_memory_entries - Run-scoped lessons shared across one Agent's Tickets
  work_products     - role-addressed pointers to artifacts an agent produced
  heartbeat_runs    - one row per CLI invocation; logs + driver + model + exit code
  execution_events  - streaming phase/progress events from agent markers
  registry_models   - global query index mirrored from Registry Ticket manifests

Design notes
------------
- We use Postgres in production. SQLite is supported for local dev / tests --
  JSONB falls back to JSON, ENUMs become strings.
- All timestamps are stored as TIMESTAMP WITH TIME ZONE in Postgres,
  TIMESTAMP in SQLite (driver-side).
- tickets.payload follows the worker contracts documented in the Playbook and
  built by `zevo.engine.run.runner`. We do not introduce a parallel
  per-task-type table.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON


# Dialect-aware JSON column: JSONB on Postgres, plain JSON elsewhere.
JsonCol = JSON().with_variant(JSONB(), "postgresql")


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    """When this row was written — not when its transaction opened.

    `server_default=func.now()` alone is wrong for anything the harness writes.
    In Postgres `now()` is `transaction_timestamp()`, fixed for the life of the
    transaction, and one heartbeat IS one transaction: the runner opens it when a
    stage starts and commits when the stage ends, twenty minutes later for a
    training run. Every row written in between — the result message, the work
    products — came out stamped with the moment the stage BEGAN. A ticket then
    read, in timestamp order: created, `{"status": "succeeded"}`, "Starting…",
    "Progress…", "Done…" — the outcome filed seven seconds before the agent said
    it was starting, and on a supervisor ticket nearly an hour early.

    Evaluated per INSERT on the Python side, so the time is the time. The server
    default stays for rows inserted outside the ORM (migrations, raw SQL), where
    the transaction is short and the distinction does not arise.
    """
    return datetime.now(timezone.utc)


class Agent(Base):
    """One row per agent (mirrors playbook/agents/<id>/identity.md)."""

    __tablename__ = "agents"
    __table_args__ = (
        CheckConstraint(
            "sandbox = 'none' OR "
            "(sandbox = 'openshell' AND id = 'orchestrator' "
            "AND default_driver = 'claude_cli')",
            name="ck_agents_sandbox_capability",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # kebab-case slug
    name: Mapped[str] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(String(128))
    reports_to: Mapped[str] = mapped_column(String(64), default="")
    identity_path: Mapped[str] = mapped_column(Text)
    # Empty means inherit identity.md. A non-empty value is a runtime override
    # set through `agent set` inside the Zevo shell or through the UI.
    default_driver: Mapped[str] = mapped_column(String(32), default="")
    default_model: Mapped[str] = mapped_column(String(256), default="")
    # Per-agent execution terminal. OpenShell is currently valid only for the
    # claude_cli orchestrator; workers require paths/SSH outside its policy.
    sandbox: Mapped[str] = mapped_column(String(16), default="none", server_default="none")
    output_schema: Mapped[str] = mapped_column(String(256), default="")
    tools: Mapped[list[str]] = mapped_column(JsonCol, default=list)
    # How many heartbeats may run in parallel for this agent. Default 1
    # (serial). Bumping helps inference / data-creation; keep at 1 for
    # supervisor and Data work where ordering matters. server_default is set
    # so existing rows backfill cleanly on the alembic upgrade.
    max_concurrent_runs: Mapped[int] = mapped_column(
        Integer, default=1, server_default="1"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow,
    )


class Task(Base):
    """A reusable problem definition: goal plus held-out scoring assets.

    These used to live only in fixtures/user_requests.py as Python literals,
    which meant the catalogue could not be edited without a code change. A
    fresh database is seeded with the shipped catalogue frozen in the
    b2c3d4e5f6a8 migration, so every task -- shipped or added later -- is
    ordinary data the user can edit, add to, or delete.

    Training data, model, method, validation assets, runtime and budgets belong
    to TaskSetting or Run. Keeping them off Task prevents hidden defaults from
    changing a run launched through a different UI entry point.
    """

    __tablename__ = "tasks"
    __table_args__ = (
        CheckConstraint(
            "metric_direction IN ('max', 'min')", name="ck_tasks_metric_direction"
        ),
        CheckConstraint(
            "metric_type IN ('builtin', 'custom')", name="ck_tasks_metric_type"
        ),
    )

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_objective: Mapped[str] = mapped_column(Text, default="")
    # One Task may measure several named Test contracts. Each item owns its
    # data, inference query, submission shape, answer fields, metric, target,
    # and optional frozen custom evaluator; the harness reports their
    # unweighted mean as the suite headline.
    test_sets: Mapped[list[dict[str, Any]]] = mapped_column(
        JsonCol, default=list, server_default="[]"
    )
    # The scalar columns below are the primary (first) Test projection used by
    # the existing optimization/Validation lane. The held-out harness reads
    # every item from ``test_sets``.
    test_set: Mapped[str] = mapped_column(Text, default="")
    # Which of `test_set`'s columns hold the answers. The data agent drops
    # exactly these to produce the questions-only copy inference is given, so a
    # task declares its scoring files ONCE instead of shipping a hand-stripped
    # duplicate of each — the arrangement that came before this, and that went
    # stale the moment either file was edited.
    test_answer_fields: Mapped[list[str]] = mapped_column(JsonCol, default=list)
    test_sample_submission: Mapped[str] = mapped_column(Text, default="")
    # Headline/primary projection for existing Run reporting columns. Each
    # suite member owns its scorer; these columns project a one-member suite
    # and use the harness aggregate for a multi-member suite.
    metric_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="builtin", server_default="builtin"
    )
    evaluation_script: Mapped[str] = mapped_column(Text, default="")
    evaluator_sha256: Mapped[str] = mapped_column(
        String(64), default="", server_default=""
    )
    # Human-readable name of the authoritative value produced by the scorer,
    # e.g. accuracy, f1, loss, or pass@1.  Its scale is task-defined.
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    # Which numeric direction wins for this task's scorer.
    metric_direction: Mapped[str] = mapped_column(
        String(3), nullable=False
    )
    # The original tasks migration left this nullable at the database level.
    # Normal inserts still receive now(), but the annotation must match the
    # deployed schema so Alembic does not invent an ALTER on every check.
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        server_default=func.now(), default=_utcnow,
    )


class TaskSetting(Base):
    """One way of attacking a task: what a run pins down, and what it leaves open.

    A task is the PROBLEM (objective + the files that score it). Which base
    model, which training method and whether the training data is handed over
    are the SETTING, and one task has many — that is what the autonomy level
    counts, so the level is derived from these three rather than stored.

    Rows are created only when the user explicitly saves a launch configuration
    or writes one down before running it. Deleting one never touches past runs;
    each Run keeps its own request snapshot.
    """

    __tablename__ = "task_settings"
    __table_args__ = (
        CheckConstraint(
            "validation_metric_type IN ('builtin', 'custom')",
            name="ck_task_settings_validation_metric_type",
        ),
        CheckConstraint(
            "validation_metric_direction IN ('max', 'min')",
            name="ck_task_settings_validation_metric_direction",
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # Plain text, like runs.task_name: deleting a Task does not erase historical
    # Settings or Runs.
    task_name: Mapped[str] = mapped_column(String(64), index=True, default="")
    # User-chosen name; blank at creation receives the next non-reused `sN`.
    name: Mapped[str] = mapped_column(String(32), default="", server_default="")
    # The three decisions the level counts.
    dataset: Mapped[str] = mapped_column(Text, default="")
    # When `dataset` is a HuggingFace id: which slice, and which named subset.
    # Same two axes as the validation pair below, for the same reason — a repo
    # is not one table, and "the training data" does not name a set on its own.
    dataset_split: Mapped[str] = mapped_column(String(64), default="", server_default="")
    dataset_config: Mapped[str] = mapped_column(String(64), default="", server_default="")
    base_model: Mapped[str] = mapped_column(String(128), default="")
    training_method: Mapped[str] = mapped_column(String(64), default="")
    # Method-specific pinned values. Auxiliary model values are Hugging Face
    # ids, not local artifacts; validation lives at every API boundary that
    # writes this column.
    method_config: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    # User-pinned portions of Specialist-owned choices. Empty values are
    # declared in Intents; supplied values remain Setting-owned because
    # changing one changes what is being compared, not merely where it runs.
    prompt_framing: Mapped[str] = mapped_column(String(256), default="", server_default="")
    system_prompt: Mapped[str] = mapped_column(Text, default="", server_default="")
    loss_objective_config: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    inference_config: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    decoding_config: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    # The set the run OPTIMIZES against. It belongs here rather than on the task
    # for the same reason `dataset` does: the task is the PROBLEM, and the
    # problem is defined by what a run is judged on — the held-out test set.
    # Which rows you tune against on the way there is part of how you attack it.
    # Empty means "carve 20% from Test at run creation"; Test must be large
    # enough to leave at least 200 Validation rows.
    validation_set: Mapped[str] = mapped_column(Text, default="", server_default="")
    # When `validation_set` is a HuggingFace id rather than a path: which slice
    # of the repo, and which named subset. A hub repo is not one table, and the
    # split that is a validation set is exactly the thing that has to be said —
    # defaulting to `train` here would tune on training rows. Empty split lets
    # run creation pick the repo's validation-like split, and fail if it has
    # none rather than guess.
    validation_split: Mapped[str] = mapped_column(String(64), default="", server_default="")
    validation_config: Mapped[str] = mapped_column(String(64), default="", server_default="")
    # Required when a named validation_set is stored. Empty only accompanies
    # the carve path, whose shape is resolved by the Data Agent.
    validation_answer_fields: Mapped[list[str]] = mapped_column(JsonCol, default=list)
    # The submission template for THIS validation set: the columns inference has
    # to emit. Required for a named set; a carve gets one from the Data Agent.
    validation_sample_submission: Mapped[str] = mapped_column(Text, default="", server_default="")
    # Validation belongs to the Setting/Run. It may use a different metric
    # implementation and scale from that Run's held-out Test metric.
    validation_metric_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="builtin", server_default="builtin"
    )
    validation_metric: Mapped[str] = mapped_column(
        String(64), nullable=False, default="token_f1", server_default="token_f1"
    )
    validation_metric_direction: Mapped[str] = mapped_column(
        String(3), nullable=False, default="max", server_default="max"
    )
    validation_evaluation_script: Mapped[str] = mapped_column(
        Text, default="", server_default=""
    )
    validation_evaluator_sha256: Mapped[str] = mapped_column(
        String(64), default="", server_default=""
    )
    # Natural-language guidance for Zevo-owned selections. These describe the
    # search; unlike dataset/base_model/training_method they do not pin an exact
    # branch.
    data_query: Mapped[str] = mapped_column(Text, default="")
    model_query: Mapped[str] = mapped_column(Text, default="", server_default="")
    method_query: Mapped[str] = mapped_column(Text, default="", server_default="")
    # No `gpu_provider` or `generation_backend` here: which box the work runs on and
    # which server generates the tokens are facts about a machine, not about a
    # line of attack. The same experiment on a cluster and on a rented GPU is
    # the same experiment, so they are chosen per run instead.
    # 0 = unlimited on both, same as everywhere else.
    iteration_budget: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    max_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, server_default="0.0")
    stop_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), default=_utcnow)


class Run(Base):
    """A single pipeline execution (kicked off by `run create` / POST /runs)."""

    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint(
            "metric_direction IN ('max', 'min')", name="ck_runs_metric_direction"
        ),
        CheckConstraint(
            "validation_metric_direction IN ('max', 'min')",
            name="ck_runs_validation_metric_direction",
        ),
        CheckConstraint(
            "mode IN ('full_pipeline', 'customized_pipeline', 'single_stage', 'auto')",
            name="ck_runs_mode",
        ),
        CheckConstraint(
            "status IN ('planning', 'running', 'success', 'degraded', 'failed', 'halted', 'cancelled')",
            name="ck_runs_status",
        ),
        CheckConstraint(
            "generation_backend IN ('hf', 'vllm')",
            name="ck_runs_generation_backend",
        ),
        CheckConstraint(
            "gpu_provider IN ('instance', 'cluster', 'cloud')",
            name="ck_runs_gpu_provider",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # The user's problem statement and the composed instruction Agents receive
    # are stored separately; no reader has to reverse-engineer a string seam.
    task_objective: Mapped[str] = mapped_column(Text, default="", server_default="")
    agent_objective: Mapped[str] = mapped_column(Text, default="", server_default="")
    task_name: Mapped[str] = mapped_column(String(64), default="")  # predefined package, or the user's own name
    # What the user calls THIS execution, as opposed to the task it executes:
    # many runs share one task_name, so the list needs something per-run to read.
    # Current API/CLI launch contracts require a non-empty value.
    run_name: Mapped[str] = mapped_column(String(128), default="", server_default="")
    # Exact reusable Setting selected or matched at launch. The Run still owns
    # a complete immutable request snapshot; this id is naming provenance for
    # UI grouping, not a live lookup that can rewrite history.
    setting_id: Mapped[str | None] = mapped_column(
        ForeignKey("task_settings.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # Immutable display-name snapshot. Deleting the reusable Setting may clear
    # its FK, but must not erase what a historical Run was called.
    setting_name: Mapped[str] = mapped_column(String(32), default="", server_default="")
    # One product-level launch vocabulary across API, UI and Playbook.
    # full_pipeline = autonomous orchestration; customized_pipeline = pipeline
    # with per-Agent customizations; single_stage = one directly assigned Ticket;
    # auto = agent-derived scoring with ordinary training-side ownership.
    mode: Mapped[str] = mapped_column(
        String(32), default="full_pipeline", server_default="full_pipeline"
    )
    # Whether the scoring contract (metric, direction, Validation/Test split)
    # is on this row yet. Every mode except `auto` settles it at creation, so
    # the default is true; an `auto` Run is created false and flips to true when
    # the engine settles the Data agent's ScopingResult (zevo.engine.run.scoping).
    # Until then `metric` is '' and `holdout` is {}: nothing may stamp a
    # pipeline payload or wake the Orchestrator.
    scoring_settled: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true"
    )
    # Immutable snapshot of the Task-owned held-out Test metric. It reports the
    # final result but never selects an iteration.
    metric_direction: Mapped[str] = mapped_column(
        String(3), default="max", server_default="max"
    )
    # Immutable snapshot of Task.metric, paired with metric_direction.
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    # Validation selects the champion and controls stop-threshold comparisons.
    # A separately supplied set may differ from Test; a Test-derived set
    # inherits Test's metric and direction during Run creation.
    validation_metric: Mapped[str] = mapped_column(
        String(64), nullable=False, default="token_f1", server_default="token_f1"
    )
    validation_metric_direction: Mapped[str] = mapped_column(
        String(3), nullable=False, default="max", server_default="max"
    )
    # Per-Agent customizations. Empty outside customized_pipeline runs.
    customizations: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    # Values explicitly supplied by the user are Run-global constraints.
    # Specialist-selected values live in inference_config.yaml and the
    # per-iteration train_config.yaml artifacts.
    decision_pins: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    # Exact tokenizer/template signatures are comparable only inside one base
    # model lineage. Different base models may legitimately ship different
    # native templates, while a baseline and descendants of the same base must
    # render identically.
    model_lineages: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    # planning | running | success | degraded | failed | halted | cancelled.
    # `degraded` = the run produced a registered model AND something failed:
    # real work came out of it, but not all of it, so it is neither a clean
    # success nor a write-off. Terminal status vocabulary lives in contracts/tickets.py.
    status: Mapped[str] = mapped_column(String(32), default="planning")
    summary: Mapped[str] = mapped_column(Text, default="")
    halted_reason: Mapped[str] = mapped_column(Text, default="")
    # Graceful cancel. `cancel_requested_at` is set the moment the operator
    # asks to keep the weights; the run stays `running` (its tickets already
    # cancelled) while the daemon rescues the champion checkpoint per
    # `cancel_policy`, then flips it to `cancelled` and records what happened
    # in `cancel_outcome`. See engine/run/cancel_rescue.py.
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancel_policy: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    cancel_outcome: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    registry_version_tag: Mapped[str] = mapped_column(String(128), default="")
    # The reactive orchestrator's supervisor Ticket, reused for every wake.
    supervisor_ticket_id: Mapped[str] = mapped_column(String(64), default="")
    # Zevo model-improvement iteration loop. After each eval, the
    # orchestrator decides whether to spin a new train/infer/eval cycle
    # with tweaked hyperparams to chase stop_threshold. iterations_completed
    # bumps after each registered model in this run.
    # 0 = no hard iteration cap; the orchestrator may still stop on evidence,
    # target, budget, or operator intervention.
    iteration_budget: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    iterations_completed: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    stop_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Best VALIDATION score in validation_metric_direction — the champion-selection
    # number. NULL = none yet; negative values remain valid task scores.
    best_validation_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # ── the held-out goal, written ONLY by the harness ───────────────────────
    # The orchestrator optimizes validation; the test set is scored behind its
    # back (see runner._spawn_holdout_tickets) so the loop cannot fit itself to
    # the number it is judged by. This is the held-out score of whichever
    # iteration currently wins on VALIDATION — not the latest or max test
    # score. ScoreEvent carries the complete held-out series.
    champion_test_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # BOTH lanes' paths, kept HERE rather than in the orchestrator's payload.
    # This is the Agent API capability boundary: the orchestrator is handed
    # validation paths only, and these fields are never returned to Agent-shaped
    # API callers. It is not a claim of per-process filesystem sandboxing.
    #
    #   test contract    test_metric_type, test_evaluation_script,
    #                    test_evaluator_sha256
    #   test lane        test_set, test_answer_fields, test_public,
    #                    test_sample_submission
    #   validation lane  validation_metric_type, validation_evaluation_script,
    #                    validation_evaluator_sha256, validation_set,
    #                    validation_answer_fields,
    #                    validation_public, validation_sample_submission,
    #                    validation_needs_metric_binding, validation_rows,
    #                    validation_source
    #   training data    dataset_split, dataset_config
    #   note             why the carve did what it did
    #
    # `*_public` is the questions-only copy of each set. It is written by the
    # RUNNER (see `_record_prepared_scoring_data`), not by whoever created the run: the
    # copy is the data agent's output, recorded the first time it is produced so
    # every iteration is scored on the same rows. It was once a file the user
    # hand-maintained beside the original, which went stale the moment either
    # side was edited and told nobody.
    holdout: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    # Per-iteration decision+performance log. Evaluation upserts the factual
    # model/method/score fields; the orchestrator enriches the same row with its
    # action/result/analysis/next decision prose.
    # `score` is the VALIDATION score. The engine adds `test_score` to
    # the entry once the held-out lane lands, and strips it back out of the
    # copy it hands the orchestrator.
    # This column predates the JSONB convention and its migration deliberately
    # created PostgreSQL JSON. Keep the ORM type identical to the deployed
    # column; changing it requires an explicit data migration.
    history: Mapped[list[Any]] = mapped_column(
        JSON, default=list, server_default="[]"
    )
    # G.1 — hard $ cap. 0.0 = no cap (existing runs). When set, the
    # backend rejects ticket spawn + the orchestrator reads the live budget
    # snapshot and closes successfully when a retained model exists, otherwise
    # fails specifically because no usable result was produced. Spent =
    # sum(heartbeat_runs.estimated_cost_usd) +
    # sum(infra_instances.dph * uptime_seconds/3600) for the run.
    max_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, server_default="0.0")
    # Run-only wall-clock cap in hours. 0.0 = unlimited. Unlike iteration and
    # cost limits, this is deliberately not part of TaskSetting: a deadline is
    # about this launch, not a reusable experimental configuration.
    max_runtime_hours: Mapped[float] = mapped_column(Float, default=0.0, server_default="0.0")
    # Maximum time Zevo may leave an automatically submitted cluster job in
    # queue. Queue wait is excluded from max_runtime_hours and every launch has
    # an explicit, bounded allowance greater than 0 and at most 168 hours.
    max_queue_wait_hours: Mapped[float] = mapped_column(
        Float, default=24.0, server_default="24.0"
    )
    # #7 — Zevo model-improvement loop policy. min_delta_per_iter triggers
    # plateau stop when 2+ consecutive iterations move the score by
    # less than this. regression_tolerance triggers regression stop
    # when the score worsens by more than this from prior best.
    # 0.0 (default) = no auto-stop on those signals.
    min_delta_per_iter: Mapped[float] = mapped_column(Float, default=0.0, server_default="0.0")
    regression_tolerance: Mapped[float] = mapped_column(Float, default=0.0, server_default="0.0")
    # Generation backend for the run: 'hf' (transformers+peft) or 'vllm'. Set at
    # run creation from POST /runs / CLI --generation-backend; the runner stamps it onto
    # the inference + train (GRPO) ticket inputs.
    #
    # `default` is 'vllm', matching the launch form — every path that creates a run sets the value
    # explicitly, so this only decides what a row with no opinion gets, and it
    # should be the same answer the rest of the system gives.
    generation_backend: Mapped[str] = mapped_column(String(16), default="vllm", server_default="vllm")
    # Maximum GPUs this run may use at once, from the launch form. Infrastructure
    # selects a concrete count at or below this limit. Under `instance`, that
    # concrete count is the lease size taken from one user allocation.
    #
    # Zero means the launch form was blank and imposes no GPU-count limit.
    # The maximum is resolved once at Run creation. It has no Ticket-payload
    # duplicate; the runner stamps it onto Infrastructure's typed input.
    num_gpus: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    gpu_provider: Mapped[str] = mapped_column(
        String(16), default="instance", server_default="instance"
    )
    # Verified SSH profile selected for a cluster/instance run. NULL means use
    # the matching deployment-level ZEVO_CLUSTER_* / ZEVO_INSTANCE_* fallback.
    # Real FK: deleting the SshHost sets this back to NULL (ON DELETE SET NULL)
    # instead of leaving a dangling id. NULL (not '') so the FK stays satisfiable.
    ssh_host_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("ssh_hosts.id", ondelete="SET NULL"),
        nullable=True, default=None, index=True,
    )
    # When gpu_provider == 'cloud', pin Vast.ai vs Lambda for this run
    # ('' | 'vastai' | 'lambda'). Empty falls back to the ZEVO_CLOUD_BACKEND
    # deployment default. Resolved once at Run creation.
    cloud_backend: Mapped[str] = mapped_column(String(16), default="", server_default="")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    tickets: Mapped[list["Ticket"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    memory_entries: Mapped[list["AgentMemoryEntry"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Ticket(Base):
    """One independently executable work unit within a Run.

    Cross-ticket data flow has one representation: ``inputs``. Each entry names
    the source Ticket, artifact role, resolved WorkProduct id, and resolved path.
    Scheduling dependencies are therefore derived rather than copied into a
    second dependency list.
    """

    __tablename__ = "tickets"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'repairing', 'awaiting_input', 'waiting_external', 'succeeded', 'degraded', 'failed', 'skipped', 'cancelled')",
            name="ck_tickets_status",
        ),
        CheckConstraint(
            "input_format IN ('typed', 'freeform')",
            name="ck_tickets_input_format",
        ),
        CheckConstraint(
            "lane IN ('optimization', 'held_out_test')",
            name="ck_tickets_lane",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # e.g. 'data-001'
    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    # queued | running | repairing | awaiting_input | waiting_external | succeeded | degraded | failed |
    # skipped | cancelled
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    # typed = validate the Agent payload; freeform = a natural-language request.
    input_format: Mapped[str] = mapped_column(
        String(16), default="typed", server_default="typed"
    )
    # optimization tickets are visible to the supervisor. held_out_test tickets
    # are engine-owned and hidden from it.
    lane: Mapped[str] = mapped_column(
        String(24), default="optimization", server_default="optimization", index=True
    )
    # Explicit iteration ownership belongs on the Ticket envelope, not hidden in
    # an agent-specific payload.
    iteration: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    payload: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    customization: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    # role -> {source_ticket_id, artifact_role, work_product_id, path}
    inputs: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    supervisor_wake_deferred: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false"
    )
    # Automatic repair is bounded and auditable on the Ticket itself.
    repair_attempts: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0"
    )
    repair_route: Mapped[str] = mapped_column(
        String(16), default="", server_default=""
    )
    summary: Mapped[str] = mapped_column(Text, default="")  # final result summary
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), default=_utcnow, onupdate=_utcnow)

    run: Mapped[Run] = relationship(back_populates="tickets")
    messages: Mapped[list["TicketMessage"]] = relationship(back_populates="ticket", cascade="all, delete-orphan")
    notices: Mapped[list["TicketNotice"]] = relationship(back_populates="ticket", cascade="all, delete-orphan")
    results: Mapped[list["HeartbeatResult"]] = relationship(back_populates="ticket", cascade="all, delete-orphan")
    work_products: Mapped[list["WorkProduct"]] = relationship(back_populates="ticket", cascade="all, delete-orphan")
    heartbeats: Mapped[list["HeartbeatRun"]] = relationship(back_populates="ticket", cascade="all, delete-orphan")
    execution_events: Mapped[list["ExecutionEvent"]] = relationship(back_populates="ticket", cascade="all, delete-orphan")
    memory_entries: Mapped[list["AgentMemoryEntry"]] = relationship(
        back_populates="source_ticket", cascade="all, delete-orphan"
    )


class AgentMemoryEntry(Base):
    """One append-only lesson scoped to a Run, Agent, and execution lane."""

    __tablename__ = "agent_memory_entries"
    __table_args__ = (
        CheckConstraint(
            "lane IN ('optimization', 'held_out_test')",
            name="ck_agent_memory_lane",
        ),
        CheckConstraint(
            "kind IN ('verified_fact', 'pitfall', 'runtime_finding', "
            "'experiment_finding', 'artifact_reference', 'recommendation')",
            name="ck_agent_memory_kind",
        ),
        CheckConstraint(
            "visibility IN ('agent_local', 'shared_candidate')",
            name="ck_agent_memory_visibility",
        ),
        CheckConstraint(
            "status IN ('active', 'superseded', 'accepted', 'dismissed')",
            name="ck_agent_memory_status",
        ),
        Index(
            "ix_agent_memory_scope",
            "run_id", "agent_id", "lane", "status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    agent_id: Mapped[str] = mapped_column(String(64), index=True)
    lane: Mapped[str] = mapped_column(
        String(24), default="optimization", server_default="optimization"
    )
    iteration: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    kind: Mapped[str] = mapped_column(String(32), index=True)
    key: Mapped[str] = mapped_column(String(64), index=True)
    summary: Mapped[str] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    visibility: Mapped[str] = mapped_column(
        String(24), default="agent_local", server_default="agent_local"
    )
    applies_to: Mapped[dict[str, str]] = mapped_column(JsonCol, default=dict)
    status: Mapped[str] = mapped_column(
        String(16), default="active", server_default="active", index=True
    )
    source_ticket_id: Mapped[str] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True
    )
    supersedes_id: Mapped[str | None] = mapped_column(
        ForeignKey("agent_memory_entries.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow
    )

    run: Mapped[Run] = relationship(back_populates="memory_entries")
    source_ticket: Mapped[Ticket] = relationship(back_populates="memory_entries")


class TicketMessage(Base):
    """Conversation only: a user or Agent message attached to a Ticket."""

    __tablename__ = "ticket_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ticket_id: Mapped[str] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True
    )
    author: Mapped[str] = mapped_column(String(64))  # agent_id, 'user', 'system'
    body: Mapped[str] = mapped_column(Text)
    triggered_wakeup_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), default=_utcnow)

    ticket: Mapped[Ticket] = relationship(back_populates="messages")


class TicketNotice(Base):
    """A structured system warning/information record; never prompt dialogue."""

    __tablename__ = "ticket_notices"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ticket_id: Mapped[str] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True
    )
    code: Mapped[str] = mapped_column(String(64), default="")
    severity: Mapped[str] = mapped_column(String(16), default="info")
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow
    )

    ticket: Mapped[Ticket] = relationship(back_populates="notices")


class HeartbeatResult(Base):
    """The validated structured result emitted by one Agent activation."""

    __tablename__ = "heartbeat_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ticket_id: Mapped[str] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True
    )
    heartbeat_id: Mapped[str] = mapped_column(String(36), default="", index=True)
    agent_id: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(16), default="failed")
    output: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow
    )

    ticket: Mapped[Ticket] = relationship(back_populates="results")


class WorkProduct(Base):
    """Role-addressed artifact pointer consumed through Ticket.inputs."""

    __tablename__ = "work_products"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ticket_id: Mapped[str] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(48), index=True)
    path: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), default=_utcnow)

    ticket: Mapped[Ticket] = relationship(back_populates="work_products")


class HeartbeatRun(Base):
    """One CLI invocation. Lets us answer: 'what driver/model ran this ticket, when?'"""

    __tablename__ = "heartbeat_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ticket_id: Mapped[str] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True
    )
    agent_id: Mapped[str] = mapped_column(String(64), index=True)
    driver: Mapped[str] = mapped_column(String(32))  # one of the registered drivers (see engine.agent.drivers)
    model: Mapped[str] = mapped_column(String(256))
    # Immutable semantic operation executed by this heartbeat.
    operation: Mapped[str] = mapped_column(String(64), default="", server_default="")
    # Why this activation exists within the operation. External stages normally
    # have separate submit and collect activations; retries and checkpoint
    # continuations must remain distinguishable in the audit trail and UI.
    activation_phase: Mapped[str] = mapped_column(
        String(32), default="", server_default="",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        default=_utcnow, index=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_code: Mapped[int] = mapped_column(Integer, default=-1)
    stdout_path: Mapped[str] = mapped_column(Text, default="")
    stderr_path: Mapped[str] = mapped_column(Text, default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    # The exact configuration resolved for this activation. It is execution
    # metadata, not a synthetic phase called ``__config__``.
    resolved_config: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)

    # Per-heartbeat usage + cost. Sourced from Codex
    # `turn_completed.usage` events as they flow through the runner's
    # event_sink. estimated_cost_usd applies the current per-model price
    # table at heartbeat-close time and is frozen on the row -- if
    # prices change later, historical rows keep showing what they cost
    # at the time.
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    reasoning_output_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, server_default="0.0")

    ticket: Mapped[Ticket] = relationship(back_populates="heartbeats")


class ExecutionEvent(Base):
    """Streaming phase/progress event from agent markers.

    Inspired by TuneLLM/training/scripts/train_unsloth.py and
    TuneLLM/agent/agent/job_handler.py. One row per logical phase or
    attempt/phase/step reading; replayed progress markers are merged within an
    attempt so the UI plots each measured step once without splicing a restarted
    training process into the failed curve.
    """

    __tablename__ = "execution_events"
    __table_args__ = (
        Index(
            "ux_execution_progress_step",
            "heartbeat_id", "attempt_id", "phase", "current_step",
            unique=True,
            postgresql_where=text("event_type = 'progress'"),
            sqlite_where=text("event_type = 'progress'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ticket_id: Mapped[str] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True
    )
    # Exact activation that emitted this event. A ticket can be activated more
    # than once, so ticket_id alone is not an execution identity.
    heartbeat_id: Mapped[str] = mapped_column(
        ForeignKey("heartbeat_runs.id", ondelete="CASCADE"), index=True
    )
    # One concrete process launch inside the activation. A Train agent can
    # repair an implementation-only runtime failure and launch Trainer again in
    # the same heartbeat; the attempt UUID keeps both histories intact.
    attempt_id: Mapped[str] = mapped_column(String(36), nullable=False)
    event_type: Mapped[str] = mapped_column(String(16), default="phase")
    phase: Mapped[str] = mapped_column(String(64))
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    loss: Mapped[float] = mapped_column(Float, default=-1.0)
    extras: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), default=_utcnow)

    ticket: Mapped[Ticket] = relationship(back_populates="execution_events")


class AgentWakeupRequest(Base):
    """One row per "this agent needs to wake up" event.

    Every code path that creates / reassigns a ticket, every UI
    "Run Heartbeat" click, every cron tick, every agent-to-agent handoff
    inserts a row here. The wakeup_daemon drains the queue: groups by
    agent_id, respects `max_concurrent_runs`, takes a per-WAKEUP advisory
    lock, and runs the ticket in-process.

    Coalescing: if a queued wakeup already exists for the same
    (agent_id, ticket_id), the new one is marked 'coalesced' instead of
    duplicating work — and `uq_wakeup_queued_per_ticket` below makes that
    a database guarantee rather than a best-effort SELECT-then-INSERT.
    """

    __tablename__ = "agent_wakeup_requests"
    __table_args__ = (
        Index(
            "uq_wakeup_queued_per_ticket",
            "agent_id", "ticket_id",
            unique=True,
            postgresql_where=text("status = 'queued' AND ticket_id IS NOT NULL"),
            sqlite_where=text("status = 'queued' AND ticket_id IS NOT NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    # Nullable: a "no-ticket" wakeup is an alarm-clock asking the agent
    # to scan its own inbox.
    ticket_id: Mapped[str | None] = mapped_column(
        ForeignKey("tickets.id", ondelete="SET NULL"), nullable=True, index=True
    )

    source: Mapped[str] = mapped_column(String(32))
    # 'assignment' | 'on_demand' | 'handoff' | 'retry' | 'scheduler' |
    # 'cron' | 'reassignment'
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    # 'queued' | 'running' | 'completed' | 'failed' | 'coalesced'

    heartbeat_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("heartbeat_runs.id", ondelete="SET NULL"), nullable=True
    )

    trigger_detail: Mapped[str] = mapped_column(String(128), default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)

    scheduled_for: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class RegistryModel(Base):
    """Queryable global champion mirrored from Registry Ticket manifests."""

    __tablename__ = "registry_models"
    __table_args__ = (
        # Mirror the DB invariant the migrations already enforce (and that Task/
        # Run/TaskSetting all declare). The model had dropped it, leaving a
        # model-vs-DB drift; declaring it here realigns them and keeps sqlite
        # create_all schemas consistent with the migrated Postgres schema.
        CheckConstraint(
            "metric_direction IN ('max', 'min')",
            name="ck_registry_models_metric_direction",
        ),
    )

    version_tag: Mapped[str] = mapped_column(String(128), primary_key=True)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    base_model: Mapped[str] = mapped_column(String(128))
    training_method: Mapped[str] = mapped_column(String(64))
    dataset_source: Mapped[str] = mapped_column(Text, default="")
    model_path: Mapped[str] = mapped_column(Text, default="")
    task_objective: Mapped[str] = mapped_column(Text, default="")
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_direction: Mapped[str] = mapped_column(String(3), nullable=False)
    eval: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), default=_utcnow)
    # J.1 — back-link to the run that produced this model. Nullable so
    # historic rows (registered before this column existed) still load.
    # Lineage endpoint joins on this to surface dataset/infra/metrics.
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), nullable=True, index=True
    )


class AuditEvent(Base):
    """One operations-history row. Lightweight: who did what, to which
    object, before+after JSON for diffable rows. Used to answer
    'when did this run get cancelled, by whom?' and 'who flipped
    data to claude_cli?'.

    Append-only. No PII scanning, no auth integration — just structured
    history. event_type is free-form (`run.create`, `run.cancel`,
    `agent.patch`, `secrets.set`, `secrets.clear`, ...).
    """

    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    actor: Mapped[str] = mapped_column(String(64), default="anonymous")
    target_type: Mapped[str] = mapped_column(String(32), default="")
    target_id: Mapped[str] = mapped_column(String(128), default="", index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    before: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    after: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow, index=True
    )


class ScoreEvent(Base):
    """One eval-result data point in a run's history.

    Two series share this table, told apart by `split`. The engine writes the
    `validation` row from every completed Evaluation ticket and writes the
    `test` row from the private held-out lane. Agents can read validation only;
    the dashboard can observe both.
    """

    __tablename__ = "score_events"
    __table_args__ = (
        CheckConstraint(
            "split IN ('validation', 'test')", name="ck_score_events_split"
        ),
        CheckConstraint(
            "source IN ('baseline', 'trained', 'validation')",
            name="ck_score_events_source",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    iteration: Mapped[int] = mapped_column(Integer, default=0)
    # Which set produced this number: 'validation' (the run's own signal) or
    # 'test' (the held-out goal, harness-written). Rows from before the split
    # existed are 'test': back then the loop tuned against the test set.
    split: Mapped[str] = mapped_column(String(16), default="validation", server_default="test")
    # 'baseline' (pre-train base model probe), 'trained' (post-train eval),
    # 'validation' (mid-train val score, optional).
    source: Mapped[str] = mapped_column(String(32), default="trained")
    score: Mapped[float] = mapped_column(Float, nullable=False)
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    # Free-form numeric extras (f1, em, bleu, etc.) the eval agent can
    # surface without us adding new columns each time.
    extras: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow, index=True
    )


class InfraInstance(Base):
    """A GPU instance the infrastructure agent has provisioned or
    connected to. Backend-side bookkeeping ONLY — the agent is still in
    charge of provisioning, probing, and releasing. This table just
    gives the harness:

      - leak detection (instances active > N hours)
      - cumulative GPU cost per run
      - a single source of truth for "what's currently running"

    The infra agent calls POST /api/infra/instances on provision,
    PATCH on release. Diagnostic host probes do not get a row because they
    are not Run infrastructure and have no rental lifecycle.
    """

    __tablename__ = "infra_instances"
    __table_args__ = (
        Index("ix_infra_instances_provider_instance_id", "provider", "instance_id"),
    )

    # Composite-uniqueness handled by (provider, instance_id); use a
    # surrogate primary key so we can still write before the provider
    # returns its id.
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # Cloud instance id or Slurm job id. Empty only before the provider returns
    # the system-owned handle.
    instance_id: Mapped[str] = mapped_column(String(128), default="")
    provider: Mapped[str] = mapped_column(String(32))
    # 'cloud' | 'cluster' | 'instance' — kept open-ended for future providers.
    status: Mapped[str] = mapped_column(String(32), default="provisioning")
    # 'provisioning' | 'ready' | 'released' | 'failed'
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    ticket_id: Mapped[str | None] = mapped_column(
        ForeignKey("tickets.id", ondelete="SET NULL"), nullable=True, index=True
    )
    gpu_name: Mapped[str] = mapped_column(String(64), default="")
    gpu_count: Mapped[int] = mapped_column(Integer, default=0)
    vram_gb: Mapped[int] = mapped_column(Integer, default=0)
    dph: Mapped[float] = mapped_column(Float, default=0.0)
    # Dollars-per-hour at provision time. Multiply by (released_at - created_at)
    # for cumulative spend.
    ssh_host: Mapped[str] = mapped_column(String(128), default="")
    ssh_port: Mapped[int] = mapped_column(Integer, default=0)
    ssh_user: Mapped[str] = mapped_column(String(64), default="")
    meta: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow, index=True
    )
    ready_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    released_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    release_reason: Mapped[str] = mapped_column(Text, default="")


class GpuLease(Base):
    """One GPU on one fixed `instance` host, held by one run.

    Why this exists
    ---------------
    `cloud` and `cluster` need no such table: each run ACQUIRES its own
    hardware (rent a box / `sbatch`), and the acquisition is itself the
    arbitration. `instance` mode has no acquisition step because the fixed host
    is already online. Without this table, concurrent Zevo Runs could both read
    the host's full GPU count and each launch an N-way
    torchrun on the same N cards.

    One row = one card = at most one live holder. That last part is enforced
    by the partial unique index below, not by the code that reads free cards
    and then writes: two requests can both see card 3 free, and only the index
    stops them both taking it.

    A lease is scoped to the RUN, not the ticket — train, inference and
    evaluation are separate tickets working on the same model, and making them
    re-bid between stages would let another run take the cards mid-pipeline.
    The reconciler releases leases when their owning Run becomes terminal.
    """

    __tablename__ = "gpu_leases"
    __table_args__ = (
        # THE guarantee: at most one live lease per (allocation, card). Partial
        # so released rows stay as history without blocking the next holder.
        Index(
            "ux_gpu_lease_live_card",
            "jobid",
            "gpu_index",
            unique=True,
            postgresql_where=text("released_at IS NULL"),
            sqlite_where=text("released_at IS NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # Stable identity of the fixed host reported by Infrastructure. The legacy
    # column name remains for migration compatibility.
    jobid: Mapped[str] = mapped_column(String(64), index=True)
    # Fixed host name. A slice is only ever handed out within one machine because
    # CUDA_VISIBLE_DEVICES cannot span machines.
    node: Mapped[str] = mapped_column(String(128), default="")
    # 0-based index as `nvidia-smi -L` reports it INSIDE the allocation, which is
    # what CUDA_VISIBLE_DEVICES takes.
    gpu_index: Mapped[int] = mapped_column(Integer)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=True, index=True
    )
    ticket_id: Mapped[str | None] = mapped_column(
        ForeignKey("tickets.id", ondelete="SET NULL"), nullable=True
    )
    gpu_name: Mapped[str] = mapped_column(String(64), default="")
    vram_gb: Mapped[int] = mapped_column(Integer, default=0)
    acquired_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow, index=True
    )
    released_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    release_reason: Mapped[str] = mapped_column(Text, default="")


class TranscriptEvent(Base):
    """One structured event from a Codex CLI subprocess.

    Replaces the "raw stdout text" model with a typed timeline that the
    Paperclip-style UI can render with proper styling (executed commands,
    tool calls, phase chips, progress bars) and that the CLI can replay.

    `seq` is a monotonic counter per heartbeat so reconnects can order
    backlog deterministically even when DB-side timestamps collide.

    `type` examples:
      - 'session_created' - session started
      - 'agent_message'   - assistant text (markdown-ish)
      - 'tool_call'       - agent invoking a tool (shell/edit/read/...)
      - 'tool_result'     - tool returned (stdout / exit code)
      - 'phase'           - synthesized from Ticket-stamped __PHASE__ markers
      - 'progress'        - synthesized from Ticket-stamped __PROGRESS__ markers
      - 'reasoning'       - agent reasoning text (if exposed)
      - 'task_complete'   - the agent's terminal event
      - 'finished'        - heartbeat finished (added by the harness)
      - 'raw'             - any unparseable stdout line (safety net)
    """

    __tablename__ = "transcript_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    heartbeat_id: Mapped[str] = mapped_column(
        ForeignKey("heartbeat_runs.id", ondelete="CASCADE"), index=True
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow
    )
    seq: Mapped[int] = mapped_column(Integer, default=0)
    # Provider-native structured event names can be longer than the small Zevo
    # event vocabulary (for example claude_system_background_tasks_changed).
    type: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict)


class SshHost(Base):
    """A verified SSH connection used by a cluster or instance Run.

    A private key remains in the host's read-only ``~/.ssh`` bind mount; a
    password lives in a Zevo-owned 0600 file. Only the selected credential
    path is stored here, so backend and scheduler resolve the same connection
    without copying key material through Ticket payloads.
    """

    __tablename__ = "ssh_hosts"
    __table_args__ = (
        CheckConstraint(
            "category IN ('cluster', 'instance')",
            name="ck_ssh_hosts_category",
        ),
        CheckConstraint(
            "status <> 'verified' OR "
            "((key_path <> '' AND password_path = '') OR "
            "(key_path = '' AND password_path <> ''))",
            name="ck_ssh_hosts_verified_credential",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    label: Mapped[str] = mapped_column(String(128), default="")
    host: Mapped[str] = mapped_column(String(255))
    port: Mapped[int] = mapped_column(Integer, default=22, server_default="22")
    username: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(
        String(16), default="instance", server_default="instance"
    )
    key_path: Mapped[str] = mapped_column(Text, default="", server_default="")
    password_path: Mapped[str] = mapped_column(Text, default="", server_default="")
    remote_dir: Mapped[str] = mapped_column(Text, default="", server_default="")
    env_setup: Mapped[str] = mapped_column(Text, default="", server_default="")
    # Optional Pyxis/Enroot image used by finite Slurm jobs on this connection.
    # It is connection data rather than a global site constant: two clusters may
    # require entirely different registries or local squashfs paths.
    container_image: Mapped[str] = mapped_column(Text, default="", server_default="")
    # Legacy encrypted inline-key storage. New and re-verified connections use
    # key_path exclusively; retaining this column avoids destroying existing
    # operator data during migration.
    private_key: Mapped[str] = mapped_column(Text, default="")
    # 'unverified' | 'verified' | 'failed'. Verified rows have exactly one
    # credential file: a host-mounted private key or a managed password.
    status: Mapped[str] = mapped_column(
        String(16), default="unverified", server_default="unverified"
    )
    gpu_info: Mapped[dict[str, Any]] = mapped_column(
        JsonCol, default=dict, nullable=False
    )  # parsed nvidia-smi
    last_error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow
    )
    last_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class EnvironmentSshVerification(Base):
    """Last local connectivity check for an env-backed SSH connection.

    The row deliberately contains no host, username, path, or key.  A hash
    binds the result to the active connection values so editing `.env`
    automatically makes the old verification stale.
    """

    __tablename__ = "environment_ssh_verifications"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(
        String(16), default="unverified", server_default="unverified"
    )
    last_error: Mapped[str] = mapped_column(Text, default="")
    last_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


# ──────────────────────── Ticket delete-safety guard ────────────────────────
# Cascade-deletes on Ticket also remove its work products, messages, and events.
# That's usually right, but a manual `DELETE FROM tickets WHERE id=...`
# can silently orphan downstream tickets whose ArtifactBinding points at the
# deleted ticket. We don't block the delete (legitimate
# cleanup must work), but we log a stderr WARN so operators see it.
import sys as _sys_warn
from sqlalchemy import event as _sa_event, text as _sa_text


@_sa_event.listens_for(Ticket, "before_delete")
def _warn_on_ticket_delete_with_downstream_bindings(mapper, connection, target: "Ticket") -> None:  # noqa: ARG001
    try:
        if not target.id or not target.run_id:
            return
        # Postgres only (JSONB ::text cast). SQLite test runs skip silently.
        if connection.dialect.name != "postgresql":
            return
        q = _sa_text(
            "SELECT id FROM tickets WHERE run_id = :rid AND id <> :tid AND "
            "inputs::text LIKE :pat LIMIT 5"
        )
        rows = connection.execute(
            q, {"rid": target.run_id, "tid": target.id, "pat": f"%{target.id}%"}
        ).fetchall()
        ids = [r[0] for r in rows]
        if ids:
            print(
                f"[db.Ticket.delete] WARN: deleting ticket {target.id!r} but "
                f"{len(ids)} other ticket(s) in run {target.run_id[:8]} "
                f"reference it in their inputs: {ids}. Their bindings may break.",
                file=_sys_warn.stderr, flush=True,
            )
    except Exception as e:
        print(
            f"[db.Ticket.delete] guard error (continuing): {type(e).__name__}: {e}",
            file=_sys_warn.stderr, flush=True,
        )
