"""Auto-generate a Markdown model card from a RegistryModel row.

Pulls together the registry entry + run + score history + lineage so
the result is self-contained: a reader knows what data + what infra +
what method + what eval score produced this model, without clicking
through five pages.

Pure-function: takes loaded ORM objects + dicts, returns a string.
The endpoint glues this to the DB.
"""
from __future__ import annotations

import re
import math
from datetime import datetime
from typing import Any


_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)


def _rel_path(p: Any) -> str:
    """Show a path relative to its run-id (UUID) segment so absolute host /
    cluster prefixes (usernames, home dirs) aren't exposed in the card."""
    if not isinstance(p, str) or not p:
        return str(p or "")
    m = _UUID.search(p)
    if m:
        return p[m.start():]
    return re.sub(
        r"^/(?:Users|home)/[^/]+/|^/orange/[^/]+/[^/]+/|^/tmp/[^/]+/", "", p
    )


def _fmt_date(d: Any) -> str:
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d %H:%M UTC")
    if isinstance(d, str):
        return d.replace("T", " ").split(".")[0]
    return "—"


def _fmt_score(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v:.4g}"


def _fmt_dollar(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"${v:.4f}" if v < 1 else f"${v:.2f}"


def render_model_card(
    *,
    registry: dict,                              # RegistryModel fields as dict
    run: dict | None = None,                     # joined Run fields, or None
    score_history: list[dict] | None = None,     # ScoreEvent rows
    audit: list[dict] | None = None,             # audit_events for this version_tag
    budget: dict | None = None,                  # BudgetSnapshot dict, or None
) -> str:
    """Return a Markdown model card.

    All arguments are plain dicts so this is trivially testable without
    a DB and so the endpoint can serialize ORM objects however it likes.
    """
    score_history = score_history or []
    audit = audit or []

    eval_block = registry.get("eval") or {}
    # The registry's own eval block comes from the run's eval ticket, and that
    # ticket scored the VALIDATION set — the set the loop tuned against. The
    # held-out number lives on the run, written by the harness after the fact,
    # and it is the one that belongs at the top of a model card: it is what
    # this checkpoint scores on data no part of the run could fit to.
    # The registry keeps the validation-selected champion, so the card's final
    # score is that champion's held-out result rather than the latest probe.
    score = (
        run.get("champion_test_score")
        if run else None
    )

    lines: list[str] = []

    # ─── header ─────────────────────────────────────────────────────────────
    # No id/title line here — the drawer header already shows the run id, and the
    # standalone-download filename carries it. Start straight at the metadata.
    if registry.get("registered_at"):
        lines.append(f"**Registered:** {_fmt_date(registry['registered_at'])}")
    lines.append("")

    # ─── headline metrics ───────────────────────────────────────────────────
    lines.append("## Performance")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    metric = str((run or {}).get("metric") or "score")
    lines.append(f"| Final held-out test {metric} | {_fmt_score(score)} |")
    if run and run.get("improvement") is not None:
        gain = float(run["improvement"])
        lines.append(
            f"| Improvement over held-out baseline | {'+' if gain >= 0 else ''}{_fmt_score(gain)} |"
        )
    if run and run.get("baseline_test_score") is not None:
        lines.append(
            f"| Held-out baseline {metric} | {_fmt_score(run['baseline_test_score'])} |"
        )
    if run and run.get("best_validation_score") is not None:
        lines.append(
            f"| Best validation across iterations | {_fmt_score(run['best_validation_score'])} |"
        )
    if eval_block:
        for k, v in eval_block.items():
            if k == "score":
                continue
            lines.append(f"| {k} | {v} |")
    lines.append("")

    # ─── training config ────────────────────────────────────────────────────
    lines.append("## Training")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Base model | `{registry.get('base_model', '?')}` |")
    lines.append(f"| Training method | `{registry.get('training_method', '?')}` |")
    if registry.get("dataset_source"):
        lines.append(f"| Dataset source | `{registry['dataset_source']}` |")
    if registry.get("model_path"):
        # Not "Adapter path": a full fine-tune saves whole weights, not a delta,
        # and the field was renamed to `model_path` precisely because the run
        # may be either. The label is what a reader believes.
        lines.append(f"| Model path | `{_rel_path(registry['model_path'])}` |")
    lines.append("")

    # ─── run + cost ─────────────────────────────────────────────────────────
    if run:
        lines.append("## Run")
        lines.append("")
        lines.append("| | |")
        lines.append("|---|---|")
        # Run id is the card title already — don't repeat it here.
        lines.append(f"| Status | `{run.get('status', '?')}` |")
        if run.get("iterations_completed"):
            lines.append(f"| Iterations completed | {run['iterations_completed']} / {run.get('iteration_budget', '?')} |")
        if run.get("started_at"):
            lines.append(f"| Started | {_fmt_date(run['started_at'])} |")
        if run.get("finished_at"):
            lines.append(f"| Finished | {_fmt_date(run['finished_at'])} |")
        if budget:
            lines.append(f"| Spend | {_fmt_dollar(budget.get('spent_usd'))} (LLM {_fmt_dollar(budget.get('llm_cost_usd'))} · GPU {_fmt_dollar(budget.get('gpu_cost_usd'))}) |")
            if budget.get("max_cost_usd"):
                lines.append(f"| Budget cap | {_fmt_dollar(budget['max_cost_usd'])} |")
            if budget.get("max_runtime_hours"):
                lines.append(
                    f"| Time limit | {budget['max_runtime_hours']:.3g} h "
                    f"(elapsed {budget.get('elapsed_runtime_hours', 0):.3g} h) |"
                )
        lines.append("")

    # ─── score history ──────────────────────────────────────────────────────
    if score_history:
        lines.append("## Score history")
        lines.append("")
        lines.append("| Iteration | Source | Score | Notes |")
        lines.append("|---:|---|---:|---|")
        for e in score_history:
            lines.append(
                f"| {e.get('iteration', '—')} | {e.get('source', '?')} | "
                f"{_fmt_score(e.get('score'))} | "
                f"{(e.get('notes') or '')[:80]} |"
            )
        lines.append("")

    # ─── audit trail ────────────────────────────────────────────────────────
    if audit:
        lines.append("## History")
        lines.append("")
        for e in audit[:10]:
            lines.append(
                f"- {_fmt_date(e.get('ts'))} · `{e.get('event_type', '?')}` "
                f"by `{e.get('actor', 'anonymous')}` — {e.get('summary', '')}"
            )
        lines.append("")

    # ─── footer ─────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("_Auto-generated by ZEVO model-card renderer._")

    return "\n".join(lines)
