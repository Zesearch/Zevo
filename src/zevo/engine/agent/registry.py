"""Seed LLM Agent and system-runner role records. Idempotent."""
from __future__ import annotations

import asyncio
from typing import Any, Callable

import frontmatter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.engine.agent.loader import REPO_ROOT, list_agent_ids, load_agent
from zevo.db import Agent, get_session_factory


def _tool_name(t: Callable[..., Any] | Any) -> str:
    """Best-effort short name for a tool callable.

    openai-agents wraps function tools in a FunctionTool object whose
    `__name__` falls back to `repr(self)` (with the full description
    inlined) -- ugly when serialized to the API. Pull the cleaner `.name`
    if it exists, else the function's __name__.
    """
    name = getattr(t, "name", None)
    if isinstance(name, str) and name:
        return name
    return getattr(t, "__name__", t.__class__.__name__)


async def seed_agents(session: AsyncSession) -> int:
    """Upsert execution roles from Agent identities and runner manifests."""
    touched = 0
    for agent_id in list_agent_ids():
        bp = load_agent(agent_id)
        existing = (
            await session.execute(select(Agent).where(Agent.id == agent_id))
        ).scalar_one_or_none()
        if existing is None:
            row = Agent(
                id=agent_id,
                name=bp.name,
                title=bp.title,
                reports_to=bp.reports_to,
                identity_path=str(bp.identity_path),
                default_driver=bp.default_driver,
                default_model=bp.default_model,
                output_schema=(
                    f"{bp.output_schema.__module__}:{bp.output_schema.__name__}"
                    if bp.output_schema is not None
                    else ""
                ),
                tools=[_tool_name(t) for t in bp.tools],
            )
            session.add(row)
        else:
            existing.name = bp.name
            existing.title = bp.title
            existing.reports_to = bp.reports_to
            existing.identity_path = str(bp.identity_path)
            # LLM Agent fields remain user-mutable through the API/UI. Seed
            # only a blank value; a non-empty DB value is an explicit runtime
            # override and survives later blueprint changes.
            if not existing.default_driver:
                existing.default_driver = bp.default_driver
            if not existing.default_model:
                existing.default_model = bp.default_model
            existing.output_schema = (
                f"{bp.output_schema.__module__}:{bp.output_schema.__name__}"
                if bp.output_schema is not None
                else ""
            )
            existing.tools = [_tool_name(t) for t in bp.tools]
        touched += 1

    # Evaluation has no Agent identity or prompt. Its manifest lives under
    # playbook/runners and its DB role exists only because Ticket.agent_id and
    # HeartbeatRun.agent_id use the execution-role table as a foreign key.
    runner_path = REPO_ROOT / "playbook" / "runners" / "evaluation.md"
    doc = frontmatter.load(runner_path)
    meta = dict(doc.metadata)
    runner_id = str(meta.get("id") or "evaluation")
    existing = (
        await session.execute(select(Agent).where(Agent.id == runner_id))
    ).scalar_one_or_none()
    values = {
        "name": str(meta.get("name") or "Evaluation Runner"),
        "title": str(meta.get("title") or "Quality Evaluator"),
        "reports_to": str(meta.get("reports_to") or "system"),
        "identity_path": str(runner_path),
        "default_driver": str(meta.get("driver") or "evaluation_runner"),
        "default_model": str(meta.get("model") or "no-llm"),
        "output_schema": str(
            meta.get("output_schema")
            or "zevo.contracts.evaluation:EvaluationResult"
        ),
    }
    if existing is None:
        session.add(Agent(id=runner_id, tools=[], sandbox="none", **values))
    else:
        for key, value in values.items():
            setattr(existing, key, value)
        existing.tools = []
        existing.sandbox = "none"
    touched += 1
    await session.commit()
    return touched


async def seed_agents_main() -> int:
    Session = get_session_factory()
    async with Session() as s:
        return await seed_agents(s)


if __name__ == "__main__":
    n = asyncio.run(seed_agents_main())
    print(f"seeded {n} execution roles")
