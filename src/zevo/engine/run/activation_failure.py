"""Persist unexpected runner failures without leaving a stale running activation."""
from datetime import datetime, timezone
from functools import wraps
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from zevo.db import HeartbeatRun, HeartbeatResult, Run, Ticket
from zevo.contracts.tickets import TERMINAL_RUN_STATUSES

log = logging.getLogger(__name__)
ACTIVATION_KEY = 'zevo_active_heartbeat'


def track_activation_failures(run):
    @wraps(run)
    async def guarded(session, *args, **kwargs):
        session.info.pop(ACTIVATION_KEY, None)
        try:
            return await run(session, *args, **kwargs)
        except Exception as exc:
            # Cache scalars before rollback expires the ORM objects. Do not
            # accidentally close an activation belonging to an earlier call.
            identity = session.info.get(ACTIVATION_KEY)
            if identity:
                factory = async_sessionmaker(session.bind, expire_on_commit=False)
                try:
                    await session.rollback()
                    await persist_activation_failure(factory, identity, exc)
                except Exception:
                    log.exception('Could not persist runner failure for %s', identity)
            raise
        finally:
            session.info.pop(ACTIVATION_KEY, None)
    return guarded


async def persist_activation_failure(factory, identity, error):
    # SQLAlchemy's full exception includes SQL parameter values. Persist only
    # the database exception text, not arbitrary artifact contents/parameters.
    cause = getattr(error, 'orig', None) or error
    message = f'Runner failed: {type(error).__name__}: {cause}'[:2000]
    ticket_id, heartbeat_id = identity
    async with factory() as db:
        ticket = (await db.scalars(select(Ticket).where(
            Ticket.id == ticket_id).with_for_update())).one_or_none()
        hb = await db.get(HeartbeatRun, heartbeat_id)
        if ticket is None or hb is None or hb.ticket_id != ticket_id or hb.finished_at:
            return
        hb.finished_at = datetime.now(timezone.utc)
        hb.exit_code = -1
        hb.error_message = message
        newest = (await db.scalars(select(HeartbeatRun.id).where(
            HeartbeatRun.ticket_id == ticket_id).order_by(
                HeartbeatRun.started_at.desc()).limit(1))).first()
        # Preserve cancellation, external jobs, completed results, and any
        # newer activation. A backend failure still needs a bounded diagnosis;
        # it does not authorize an Agent to modify engine code.
        retry = False
        handled = False
        run = await db.get(Run, ticket.run_id)
        if newest == heartbeat_id and ticket.status == 'running':
            ticket.status = 'failed'
            ticket.error_message = message
            ticket.summary = message
            if run is not None and run.status not in TERMINAL_RUN_STATUSES:
                from zevo.engine.run.runner import _apply_ticket_failure_policy
                retry = await _apply_ticket_failure_policy(
                    db, ticket, error_message=message, origin='runner',
                    can_diagnose=hb.driver not in {'evaluation_runner', 'holdout_data_runner'},
                )
                handled = True
                db.add(HeartbeatResult(
                    ticket_id=ticket.id, heartbeat_id=hb.id, agent_id=ticket.agent_id,
                    status='failed', output={'status': 'failed', 'ticket_id': ticket.id,
                                             'error_message': message},
                ))
            else:
                ticket.repair_route = 'terminal'
        await db.commit()
        if handled:
            from zevo.engine.run.runner import _queue_ticket_repair, _maybe_wake_supervisor
            if retry:
                await _queue_ticket_repair(db, ticket)
            else:
                await _maybe_wake_supervisor(db, ticket)
