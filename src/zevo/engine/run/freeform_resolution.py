"""Resolve natural-language work before executing a specialist contract.

Freeform changes how a request is authored, not the execution or acceptance
contract. Resolution is audited separately and never returns a fabricated
specialist success. Missing inputs fail before expensive work begins.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from zevo.contracts.tickets import ArtifactBinding, PIPELINE_PAYLOAD_BY_AGENT, validate_stored_payload
from zevo.engine.run.scheduler.bindings import resolve_input_bindings


class FreeformResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticket_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, ArtifactBinding] = Field(default_factory=dict)
    explanation: str
    missing_information: str = ""


def resolution_blueprint(blueprint, inp, bindings):
    from zevo.contracts.tickets import InfrastructureProvisionPayload, InfrastructureReleasePayload
    if inp.agent_id == "infrastructure":
        schemas = [InfrastructureProvisionPayload.model_json_schema(), InfrastructureReleasePayload.model_json_schema()]
    else:
        model = PIPELINE_PAYLOAD_BY_AGENT.get(inp.agent_id)
        if model is None:
            raise ValueError(f"freeform resolution is not supported for {inp.agent_id}")
        schemas = [model.model_json_schema()]
    instructions = (
        "Resolve the user's freeform request into the supplied typed payload and input bindings. "
        "This activation is preparation only: inspect authorized attachments, and if necessary write "
        "small configuration/profile files in work_dir. Do not train, generate datasets, provision, "
        "release, register, submit jobs, or access scoring populations during resolution. "
        "Do not invent missing model/data/compute identities. Copy explicit user constraints into pins; "
        "record choices for delegated details without turning your defaults into user pins. "
        "Use existing bindings unchanged. Additional local paths must be attachments, paths explicitly "
        "named by the user, or small files prepared in work_dir. Never expose hidden scoring paths. "
        "If essential information cannot be resolved, explain it in missing_information. "
        "The engine validates and builds the execution contract before the normal specialist runs. "
        "For Data, interpret prose as intent and put exact machine constraints in recipe_intent.constraints. "
        "A standalone local Data transformation does not require a scoring suite or GPU. "
        "For Train, bind the prepared dataset, verified device_info and inference configuration; "
        "a standalone prompt configuration may be prepared from explicit user requirements. "
        "Leave data_signature empty: the engine binds verified input bytes.\n\n"
        + "Payload schemas: " + json.dumps(schemas)
        + "\nAvailable bindings: " + json.dumps(bindings)
    )
    return replace(blueprint, instructions=instructions, output_schema=FreeformResolution)


def authorize_paths(resolution, inp, existing, *, standalone):
    allowed = {str(Path(p).resolve()) for p in inp.attachments}
    allowed.update(str(Path(v['path']).resolve()) for v in existing.values() if v.get('path'))
    root = Path(inp.work_dir).resolve()
    for name, binding in resolution.inputs.items():
        if name in existing:
            previous = ArtifactBinding.model_validate(existing[name])
            if binding != previous:
                raise ValueError(f"freeform resolution cannot replace bound input {name}")
        if binding.source_ticket_id:
            continue  # The normal resolver verifies Run ownership and artifacts.
        if not standalone and name not in existing:
            raise ValueError("pipeline freeform inputs require existing WorkProduct bindings")
        path = Path(binding.path).resolve()
        explicitly_named = binding.path in inp.request and binding.path.startswith('/')
        if str(path) not in allowed and root not in path.parents and not explicitly_named:
            raise ValueError(f"freeform input {name} is not an authorized attachment or prepared artifact")
        if not path.exists():
            raise ValueError(f"freeform input {name} does not exist: {path}")
    dataset = str(resolution.payload.get('dataset') or '')
    if dataset.startswith('/') and str(Path(dataset).resolve()) not in allowed and dataset not in inp.request:
        raise ValueError("freeform Data source must be an authorized attachment or explicitly named path")


async def resolve_work_order(*, driver, blueprint, inp, ticket, run, session, bindings,
                             driver_kwargs, build_input):
    result = await driver.run_agent(blueprint=resolution_blueprint(blueprint, inp, bindings),
                                   input_payload=inp, **driver_kwargs)
    resolved = FreeformResolution.model_validate(result.output.model_dump())
    if result.exit_code or resolved.ticket_id != ticket.id:
        raise ValueError("freeform resolution failed or returned another Ticket identity")
    if resolved.missing_information:
        raise ValueError("freeform request needs information: " + resolved.missing_information)
    standalone = run.mode == 'single_stage'
    authorize_paths(resolved, inp, bindings, standalone=standalone)
    new_inputs = {**bindings, **{k: v.model_dump() for k, v in resolved.inputs.items()}}
    from zevo.db.models import Ticket as SourceTicket
    for binding in new_inputs.values():
        if binding.get('source_ticket_id'):
            source = await session.get(SourceTicket, binding['source_ticket_id'])
            if source is None or source.lane == 'held_out_test' or source.id == ticket.id:
                raise ValueError("freeform resolution cannot bind hidden or self-produced inputs")
    new_inputs = await resolve_input_bindings(session, run_id=run.id, inputs=new_inputs)
    payload = dict(resolved.payload)
    from zevo.api.routers.shared.tickets import _resolve_data_source_selection, _stamp_pipeline_payload
    if not standalone:
        payload = _stamp_pipeline_payload(run=run, agent_id=ticket.agent_id, payload=payload)
    payload = validate_stored_payload(agent_id=ticket.agent_id, input_format='typed', payload=payload)
    if ticket.agent_id == 'data':
        payload = _resolve_data_source_selection(payload)
    if ticket.agent_id == 'train':
        from zevo.db.models import WorkProduct
        from zevo.engine.remote_training_data import file_sha256
        binding = new_inputs.get('training_dataset') or {}
        product = await session.get(WorkProduct, binding['work_product_id']) if binding.get('work_product_id') else None
        if product:
            payload['data_signature'] = (product.meta or {}).get('data_signature', '')
        elif standalone and binding.get('path'):
            payload['data_signature'] = file_sha256(binding['path'])
    payload = validate_stored_payload(agent_id=ticket.agent_id, input_format='typed', payload=payload)
    from zevo.contracts.tickets import validate_bindings
    if not (standalone and ticket.agent_id == 'data' and not new_inputs):
        new_inputs = validate_bindings(agent_id=ticket.agent_id, payload=payload, inputs=new_inputs)
    if not standalone:
        from types import SimpleNamespace
        from zevo.api.routers.shared.tickets import validate_pipeline_payload, validate_pipeline_bindings
        body = SimpleNamespace(agent_id=ticket.agent_id, iteration=ticket.iteration)
        await validate_pipeline_payload(session, owning_run=run, body=body, payload=payload,
                                        current_ticket_id=ticket.id)
        payload, new_inputs = await validate_pipeline_bindings(session, owning_run=run, body=body,
            payload=payload, inputs=new_inputs, current_ticket_id=ticket.id)
    await session.refresh(ticket)
    await session.refresh(run)
    if ticket.status in {'cancelled', 'awaiting_input'}:
        raise ValueError("freeform activation was stopped before execution")
    if run.status in {'cancelled', 'success', 'failed', 'degraded'} or run.cancel_requested_at:
        raise ValueError("Run stopped before freeform execution")
    original = (ticket.input_format, ticket.payload, ticket.inputs)
    ticket.input_format, ticket.payload, ticket.inputs = 'typed', payload, new_inputs
    try:
        task_input = await build_input(payload, new_inputs)
    except Exception:
        ticket.input_format, ticket.payload, ticket.inputs = original
        raise
    from zevo.db.models import TicketMessage
    session.add(TicketMessage(ticket_id=ticket.id, author='user', body=inp.request))
    audit = {'request': inp.request, 'attachments': inp.attachments,
             'resolution': resolved.model_dump(mode='json'), 'payload': payload, 'inputs': new_inputs}
    (Path(inp.work_dir) / 'freeform_resolution.json').write_text(json.dumps(audit, indent=2))
    await session.commit()
    return task_input
