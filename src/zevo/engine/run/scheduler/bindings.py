"""Resolve Ticket input bindings to exact Work Products.

The Ticket envelope owns all cross-ticket data flow. Agent payloads never carry
Ticket ids or fields whose meaning changes from id to path.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.contracts.tickets import ArtifactBinding
from zevo.db import Ticket, WorkProduct


def source_ticket_ids(inputs: dict[str, Any]) -> list[str]:
    return sorted({
        str(value.get("source_ticket_id") or "")
        for value in inputs.values()
        if isinstance(value, dict) and value.get("source_ticket_id")
    })


async def resolve_input_bindings(
    session: AsyncSession,
    *,
    run_id: str,
    inputs: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    parsed = {
        name: ArtifactBinding.model_validate(value).model_dump()
        for name, value in (inputs or {}).items()
    }
    source_ids = source_ticket_ids(parsed)
    if not source_ids:
        return parsed

    tickets = (await session.execute(
        select(Ticket).where(Ticket.run_id == run_id, Ticket.id.in_(source_ids))
    )).scalars().all()
    by_ticket = {ticket.id: ticket for ticket in tickets}

    products = (await session.execute(
        select(WorkProduct)
        .where(WorkProduct.ticket_id.in_(source_ids))
        .order_by(WorkProduct.created_at.desc())
    )).scalars().all()
    by_id = {product.id: product for product in products}
    by_role: dict[tuple[str, str], WorkProduct] = {}
    for product in products:
        key = (product.ticket_id, product.role)
        current = by_role.get(key)
        # A Train Ticket may expose a few intermediate branch points under the
        # same semantic checkpoint role. An ordinary binding always resolves
        # to its final checkpoint; an Orchestrator can select an intermediate
        # one explicitly by WorkProduct id.
        if current is None or (
            product.role == "checkpoint"
            and (product.meta or {}).get("checkpoint_kind") == "final"
            and (current.meta or {}).get("checkpoint_kind") != "final"
        ):
            by_role[key] = product

    resolved: dict[str, dict[str, Any]] = {}
    for name, binding in parsed.items():
        source_id = binding["source_ticket_id"]
        if not source_id:
            resolved[name] = binding
            continue
        source = by_ticket.get(source_id)
        if source is None:
            raise ValueError(
                f"input {name!r} references ticket {source_id!r} outside run {run_id[:8]}"
            )
        if source.status not in {"succeeded", "degraded"}:
            raise ValueError(
                f"input {name!r} source ticket {source_id!r} is {source.status}, "
                "not succeeded/degraded"
            )
        requested_product_id = str(binding.get("work_product_id") or "")
        product = (
            by_id.get(requested_product_id)
            if requested_product_id
            else by_role.get((source_id, binding["artifact_role"]))
        )
        if product is None:
            qualifier = (
                f"WorkProduct {requested_product_id!r}"
                if requested_product_id
                else f"artifact role {binding['artifact_role']!r}"
            )
            raise ValueError(
                f"input {name!r} requires {qualifier} "
                f"from {source_id!r}, but that Work Product does not exist"
            )
        if product.ticket_id != source_id or product.role != binding["artifact_role"]:
            raise ValueError(
                f"input {name!r} WorkProduct {product.id!r} does not match "
                f"source_ticket_id={source_id!r} and "
                f"artifact_role={binding['artifact_role']!r}"
            )
        resolved[name] = {
            **binding,
            "work_product_id": product.id,
            "path": product.path,
        }
    return resolved


def input_path(inputs: dict[str, Any], name: str, *, required: bool = True) -> str:
    value = inputs.get(name) if isinstance(inputs, dict) else None
    path = str(value.get("path") or "") if isinstance(value, dict) else ""
    if required and not path:
        raise KeyError(f"resolved Ticket input {name!r} has no path")
    return path
