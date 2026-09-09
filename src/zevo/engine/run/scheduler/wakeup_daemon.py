"""Wake-up daemon -- drains agent_wakeup_requests.

Loop (every ~1s):
  1. SELECT queued wakeups WHERE scheduled_for <= now()
  2. Group by agent_id
  3. For each agent: count currently-running, respect max_concurrent_runs
  4. For each available slot:
       try acquire the per-wakeup advisory lock
       mark wakeup 'running', run the ticket in-process (own task + session)
  5. When the runner finishes, mark wakeup 'completed' or 'failed'

Cron tick (every 5 min, configurable):
  Enqueue a wakeup for one unattended queued Ticket per agent in this daemon's
  lane. Lets agents resume work if event-driven enqueueing dropped one.

Lock model: Postgres advisory locks, one key per wakeup. The daemon runs
the ticket runner IN-PROCESS (no subprocess): each wakeup gets its own DB
session, the lock is taken on a dedicated connection (see
zevo.engine.run.wakeup.advisory_lock), and PostgreSQL releases it if that
connection closes unexpectedly.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import os
from datetime import datetime, timezone

from sqlalchemy import bindparam, or_, select, text, true as sa_true, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.db import (
    Agent,
    AgentWakeupRequest,
    Run,
    Ticket,
    TicketNotice,
    get_session_factory,
)
from zevo.paths import work_dir_root
from zevo.engine.run.runner import run_ticket
from zevo.engine.run.scheduler.reconciler import reconcile_runs_and_tickets
from zevo.engine.run.wakeup import advisory_lock, queue_wakeup
from zevo.contracts.tickets import TERMINAL_RUN_STATUSES


log = logging.getLogger(__name__)

# How long a wakeup may sit behind a held agent lock before the daemon
# treats "busy" as suspicious rather than routine.
_LOCK_STUCK_SECONDS = 300.0

# The wakeups each agent is working on right now: agent_id -> {wakeup_id:
# task}. A SET per agent, capped by that agent's `max_concurrent_runs` —
# this used to be a single entry, which pinned every agent to one wakeup
# at a time and made the configured limit unreachable. The drain loop
# never waits on any of these. Tasks are kept so they are not garbage
# collected mid-flight, ids so the stale sweep can tell "still working"
# from "abandoned".
_inflight: dict[str, dict[str, asyncio.Task]] = {}

_SCHEDULER_LANES = frozenset({"optimization", "held_out_test"})


def _scheduler_lane(value: str | None = None) -> str:
    """The only Ticket lane this daemon is allowed to execute."""
    lane = (value or os.environ.get("ZEVO_SCHEDULER_LANE") or "optimization").strip()
    if lane not in _SCHEDULER_LANES:
        raise RuntimeError(
            "ZEVO_SCHEDULER_LANE must be 'optimization' or 'held_out_test', "
            f"not {lane!r}"
        )
    return lane


def _wakeup_lane_clause(lane: str):
    """SQL predicate assigning legacy no-ticket wakeups to optimization."""
    if lane == "optimization":
        return or_(
            Ticket.lane == "optimization",
            AgentWakeupRequest.ticket_id.is_(None),
        )
    return Ticket.lane == "held_out_test"


def _inflight_count(agent_id: str) -> int:
    return len(_inflight.get(agent_id, ()))


def _inflight_done(agent_id: str, wakeup_id: str) -> None:
    live = _inflight.get(agent_id)
    if live is not None:
        live.pop(wakeup_id, None)
        if not live:
            _inflight.pop(agent_id, None)


async def _process_wakeup(session: AsyncSession, w: AgentWakeupRequest) -> None:
    """Run one wakeup: mark running -> execute -> mark done/failed.

    Robust to any exception inside `run_ticket` or downstream code: we
    always end with the wakeup row in a terminal status ('completed' or
    'failed'), in a fresh transaction, so a single bad wakeup cannot
    leak slot accounting and cannot poison the daemon's session.
    """
    wakeup_id = w.id
    agent_id = w.agent_id
    ticket_id = w.ticket_id

    w.status = "running"
    try:
        await session.commit()
    except SQLAlchemyError:
        await session.rollback()
        raise

    error_message = ""
    try:
        payload = w.payload if isinstance(w.payload, dict) else {}
        driver_name = str(payload.get("driver") or "")
        model_override = str(payload.get("model") or "")
        # work_dir_root() already honors ZEVO_WORK_DIR. Ticket artifacts,
        # including a Registry Ticket's manifest, live beneath that root.
        work_root = str(payload.get("work_dir") or work_dir_root())
        if ticket_id:
            await run_ticket(
                session,
                ticket_id=ticket_id,
                work_dir_root=work_root,
                driver_name=driver_name,
                model_override=model_override,
            )
        else:
            # Join on the run: a ticket queued before the user cancelled is
            # still sitting at `queued`, and picking it up restarts work on a run
            # that is over. `run_ticket` re-checks the ticket, but nothing was
            # re-checking the run.
            tk = (await session.execute(
                select(Ticket).join(Run, Run.id == Ticket.run_id).where(
                    Ticket.agent_id == agent_id,
                    Ticket.status == "queued",
                    Ticket.lane == _scheduler_lane(),
                    Run.status.notin_(list(TERMINAL_RUN_STATUSES)),
                ).order_by(Ticket.created_at).limit(1)
            )).scalar_one_or_none()
            if tk is not None:
                await run_ticket(
                    session,
                    ticket_id=tk.id,
                    work_dir_root=work_root,
                    driver_name=driver_name,
                    model_override=model_override,
                )
    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        log.exception("wakeup %s failed", wakeup_id)
        # The session may be in a partially-failed state; clear it so
        # the terminal-status write below uses a clean transaction.
        try:
            await session.rollback()
        except SQLAlchemyError:
            pass

    await _mark_wakeup_terminal(wakeup_id, error_message)


async def _mark_wakeup_terminal(wakeup_id: str, error_message: str) -> None:
    """Mark a wakeup completed/failed in its OWN fresh session.

    Using a new session here is deliberate -- the caller's session may
    be poisoned by a prior failure. We never want a terminal-status
    write to be the thing that kills the daemon.
    """
    Session = get_session_factory()
    try:
        async with Session() as s:
            w = (await s.execute(
                select(AgentWakeupRequest).where(AgentWakeupRequest.id == wakeup_id)
            )).scalar_one_or_none()
            if w is None:
                return
            w.status = "failed" if error_message else "completed"
            w.finished_at = datetime.now(timezone.utc)
            if error_message:
                w.reason = ((w.reason or "") + " | " + error_message)[:2000]
            await s.commit()
    except SQLAlchemyError:
        log.exception("could not mark wakeup %s terminal (%s); "
                      "stale-sweep will reclaim", wakeup_id, error_message[:80])


async def _drain_once(lane: str | None = None) -> int:
    """Pick up queued wakeups, respect slot pool + locks, run them.

    Returns the number of wakeups actually processed this tick.
    """
    Session = get_session_factory()
    processed = 0
    lane = _scheduler_lane(lane)

    async with Session() as s:
        # Pull a batch of queued, grouped by agent
        rows = (await s.execute(
            select(AgentWakeupRequest)
            .outerjoin(Ticket, AgentWakeupRequest.ticket_id == Ticket.id)
            .where(AgentWakeupRequest.status == "queued")
            .where(AgentWakeupRequest.scheduled_for <= datetime.now(timezone.utc))
            .where(_wakeup_lane_clause(lane))
            .order_by(AgentWakeupRequest.scheduled_for)
            .limit(50)
        )).scalars().all()

        if not rows:
            return 0

        # Group by agent
        by_agent: dict[str, list[AgentWakeupRequest]] = {}
        for r in rows:
            by_agent.setdefault(r.agent_id, []).append(r)

        # Currently-running counts (this run; cross-instance contention
        # is mediated by the advisory lock anyway).
        running = (await s.execute(
            select(AgentWakeupRequest)
            .outerjoin(Ticket, AgentWakeupRequest.ticket_id == Ticket.id)
            .where(AgentWakeupRequest.status == "running")
            .where(_wakeup_lane_clause(lane))
        )).scalars().all()
        running_by_agent: dict[str, int] = {}
        for r in running:
            running_by_agent[r.agent_id] = running_by_agent.get(r.agent_id, 0) + 1

        # max_concurrent_runs per agent
        agent_rows = (await s.execute(select(Agent))).scalars().all()
        slots_per_agent = {a.id: max(1, a.max_concurrent_runs) for a in agent_rows}

    # Hand each agent's wakeup to a background task rather than awaiting
    # it here. An agent's work can block for a long time on something
    # outside our control -- a remote transfer, a provider readiness check, or
    # an SSH that never answers -- and awaiting it inline froze the whole
    # daemon: one stuck agent and every other agent's wakeups sat queued,
    # which looks from the UI like "I launched a run and nothing happens".
    # Agents are independent, so they proceed independently. An individual
    # Agent may also use several slots up to max_concurrent_runs; only a single
    # wakeup id is serialized by its own advisory lock.
    for agent_id, candidates in by_agent.items():
        cap = slots_per_agent.get(agent_id, 1)
        # Both terms matter and they count different things. `running_by_agent`
        # comes from the DB, so it also sees work a second daemon started;
        # `_inflight_count` sees tasks this process began that have not marked
        # themselves running yet. Using either alone lets the agent overshoot
        # its cap during the gap between the two.
        slots = cap - max(running_by_agent.get(agent_id, 0), _inflight_count(agent_id))
        for w in candidates[:max(0, slots)]:
            if w.id in _inflight.get(agent_id, {}):
                continue
            task = asyncio.create_task(_run_one(agent_id, w.id, w.created_at))
            _inflight.setdefault(agent_id, {})[w.id] = task
            task.add_done_callback(
                lambda _t, a=agent_id, i=w.id: _inflight_done(a, i)
            )
            processed += 1

    return processed


def _wakeup_lock_key(wakeup_id: str) -> str:
    """The advisory-lock key for one wakeup.

    The reclaim query below rebuilds this string in SQL, so the two must
    agree exactly; changing the prefix here without changing it there makes
    leaked locks invisible rather than making anything fail.
    """
    return f"wakeup:{wakeup_id}"


async def _run_one(agent_id: str, wakeup_id: str, queued_at: datetime) -> None:
    """Take the agent's lock and run one wakeup, in its own session.

    Its own session so advisory locks are acquired and released cleanly
    and one bad row cannot contaminate sibling wakeups.
    """
    Session = get_session_factory()
    try:
        async with Session() as s:
            async with advisory_lock(s, _wakeup_lock_key(wakeup_id)) as got:
                if not got:
                    # Keyed on the WAKEUP, so this no longer means "the agent
                    # is busy" -- a sibling wakeup for the same agent holds its
                    # own key and does not contend here. It means someone else
                    # is already processing THIS one, or its lock leaked. A
                    # leak now strands one wakeup instead of the whole agent,
                    # but it still never clears itself, so say so.
                    waited = (datetime.now(timezone.utc) - queued_at).total_seconds()
                    if waited > _LOCK_STUCK_SECONDS:
                        log.warning(
                            "[wakeup] wakeup %s (%s) has been locked for %.0fs and is "
                            "still queued; the lock may be held by a stale connection",
                            wakeup_id[:8], agent_id, waited)
                    else:
                        log.info("[wakeup] %s already being processed, skipping",
                                 wakeup_id[:8])
                    return
                w = (await s.execute(
                    select(AgentWakeupRequest).where(AgentWakeupRequest.id == wakeup_id)
                )).scalar_one_or_none()
                if w is None or w.status != "queued":
                    return
                # Different wakeup rows may target the same long-lived Ticket
                # (most visibly the stable Orchestrator Ticket).  The outer
                # lock prevents duplicate handling of one queue row; this
                # inner lock prevents overlapping heartbeats for one Ticket.
                ticket_key = f"ticket:{w.ticket_id}" if w.ticket_id else ""
                if ticket_key:
                    async with advisory_lock(s, ticket_key) as ticket_got:
                        if not ticket_got:
                            log.info(
                                "[wakeup] ticket %s is already running; defer %s",
                                w.ticket_id, wakeup_id[:8],
                            )
                            return
                        log.info("[wakeup] %s -> running wakeup %s (ticket=%s, source=%s)",
                                 agent_id, w.id[:8], w.ticket_id, w.source)
                        await _process_wakeup(s, w)
                else:
                    log.info("[wakeup] %s -> running wakeup %s (ticket=None, source=%s)",
                             agent_id, w.id[:8], w.source)
                    await _process_wakeup(s, w)
    except asyncio.CancelledError:
        raise
    except Exception:
        # Last-resort guard: NO wakeup may ever take down the daemon.
        # Log it, mark it failed, keep going.
        log.exception("[wakeup] hard failure processing %s; marking failed",
                      wakeup_id[:8])
        await _mark_wakeup_terminal(
            wakeup_id, f"daemon hard failure during wakeup {wakeup_id}"
        )


async def _cron_tick(interval_seconds: int, lane: str | None = None) -> None:
    """Every `interval_seconds`, enqueue a 'cron' wakeup per agent that
    has at least one queued ticket sitting unattended.
    """
    Session = get_session_factory()
    lane = _scheduler_lane(lane)
    async with Session() as s:
        agents = (await s.execute(select(Agent))).scalars().all()
        for a in agents:
            # Only enqueue if there's actually a queued ticket -- otherwise
            # it just thrashes the daemon.
            tk = (await s.execute(
                select(Ticket).where(
                    Ticket.agent_id == a.id,
                    Ticket.status == "queued",
                    Ticket.lane == lane,
                    ~select(TicketNotice.id).where(
                        TicketNotice.ticket_id == Ticket.id,
                        TicketNotice.code
                        == "ticket.awaiting_supervisor_completion",
                    ).exists(),
                ).limit(1)
            )).scalar_one_or_none()
            if tk is None:
                continue
            # A pending wakeup for this Ticket already covers the scan. Using
            # a concrete Ticket id is what lets two daemons claim disjoint
            # lanes and also lets queue_wakeup coalesce duplicates.
            pending = (await s.execute(
                select(AgentWakeupRequest).where(
                    AgentWakeupRequest.agent_id == a.id,
                    AgentWakeupRequest.ticket_id == tk.id,
                    AgentWakeupRequest.status == "queued",
                ).limit(1)
            )).scalar_one_or_none()
            if pending is not None:
                continue
            await queue_wakeup(
                s, agent_id=a.id, ticket_id=tk.id,
                source="cron", reason=f"alarm-clock tick ({interval_seconds}s)",
            )


async def _check_stuck_infra_instances(
    stuck_after_hours: float = 4.0,
    *,
    cooldown_seconds: int = 1800,
) -> int:
    """Find infra_instances that have been active (no released_at)
    longer than `stuck_after_hours` AND cost real money (dph > 0). For
    each, post a TicketNotice on the parent run so the orchestrator sees
    the warning on its next wake AND the user sees it in the UI.

    This STUCK detector never auto-releases — the rental may belong to a
    legitimately long-running, non-terminal run. Terminal reconciliation still
    honors device_info.auto_release; here we only surface the signal.

    `cooldown_seconds` debounces: a stuck instance gets at most one
    warning notice per cooldown window (we tag notices with
    `stuck_infra.<instance_id>` and look for the most recent one before
    re-posting).
    """
    from datetime import datetime, timedelta, timezone
    from zevo.db import InfraInstance, TicketNotice

    Session = get_session_factory()
    posted = 0
    async with Session() as s:
        now = datetime.now(timezone.utc)
        threshold = now - timedelta(hours=stuck_after_hours)
        rows = (await s.execute(
            select(InfraInstance).where(
                InfraInstance.released_at.is_(None),
                InfraInstance.created_at < threshold,
                InfraInstance.dph > 0,
            )
        )).scalars().all()
        for r in rows:
            if not r.run_id:
                # Nowhere to post; log only.
                log.warning(
                    "[stuck-infra] instance %s (%s:%s) active for %.1fh; "
                    "no run_id linked, can't post notice",
                    r.id, r.provider, r.instance_id,
                    (now - r.created_at).total_seconds() / 3600.0,
                )
                continue
            tag = f"__STUCK_INFRA_WARN__:{r.id}"
            # Cooldown: skip if we warned for THIS instance within the window.
            cutoff = now - timedelta(seconds=cooldown_seconds)
            recent = (await s.execute(
                select(TicketNotice).where(
                    TicketNotice.body.contains(tag),
                    TicketNotice.created_at >= cutoff,
                )
            )).scalars().first()
            if recent is not None:
                continue
            age_h = (now - r.created_at).total_seconds() / 3600.0
            spend = (r.dph or 0.0) * age_h
            # Pick an arbitrary ticket on the run to attach the notice to.
            tk = (await s.execute(
                select(Ticket).where(Ticket.run_id == r.run_id).limit(1)
            )).scalar_one_or_none()
            if tk is None:
                continue
            s.add(TicketNotice(
                ticket_id=tk.id,
                code="infrastructure.stuck", severity="warning",
                body=(
                    f"{tag}\n\n"
                    f"⚠ Stuck infra: instance {r.provider}:{r.instance_id} "
                    f"has been active for {age_h:.1f}h "
                    f"({r.gpu_count}× {r.gpu_name or 'GPU'}, "
                    f"~${spend:.2f} spent at ${r.dph:.3f}/h). "
                    f"Either release it via the infrastructure agent or "
                    f"explain why it must stay up."
                ),
            ))
            posted += 1
        if posted:
            await s.commit()
    return posted


async def _reap_leaked_wakeup_locks(lane: str | None = None) -> int:
    """Hand back per-wakeup advisory locks that nobody is using any more.

    A session-level advisory lock outlives the transaction that took it,
    so any path that loses the unlock parks it on a pooled connection
    forever and that agent silently stops being woken. There is more than
    one such path -- a session poisoned by a failed statement, a task
    cancelled while the agent is mid-work (you cannot await an unlock
    while being cancelled) -- and `advisory_lock` cannot close all
    of them from the inside. They all look identical from out here
    though: the lock is held, its holder has been sitting idle, and
    nobody claims to be working for that agent. Terminating the holder
    is what gives the lock back; the pool just reconnects.

    Deliberately conservative -- a lock is only leaked if its wakeup is
    still `queued` (a running one is real work, possibly another daemon's),
    it is not in this daemon's `_inflight`, and its holder has been idle
    long enough to rule out the moment between taking the lock and marking
    the wakeup running.

    Since the key became per-wakeup, a leak strands ONE wakeup rather than
    silencing an agent completely — the agent's next wakeup takes a
    different key and runs. That makes this reclaim less urgent, and it is
    also why it must still work: a stranded wakeup is now quiet enough to
    go unnoticed.
    """
    Session = get_session_factory()
    lane = _scheduler_lane(lane)
    async with Session() as s:
        # Only `queued` wakeups are worth looking at. A lock leaked on one
        # that already ran is inert — nothing will ever ask for that key
        # again — and restricting the join keeps it off the whole history
        # table. The key is rebuilt exactly as `_wakeup_lock_key` writes it.
        rows = (await s.execute(text("""
            SELECT w.id AS wakeup_id, w.agent_id, l.pid
            FROM pg_locks l
            JOIN pg_stat_activity act ON act.pid = l.pid
            JOIN agent_wakeup_requests w
              ON (hashtext('wakeup:' || w.id)::bigint & 4294967295) = l.objid
             AND ((hashtext('wakeup:' || w.id)::bigint >> 32) & 4294967295) = l.classid
            LEFT JOIN tickets t ON t.id = w.ticket_id
            WHERE l.locktype = 'advisory'
              AND l.granted
              AND act.datname = current_database()
              AND act.pid <> pg_backend_pid()
              AND act.state IN ('idle', 'idle in transaction')
              AND act.state_change < now() - interval '30 seconds'
              AND w.status = 'queued'
              AND (t.lane = :lane OR (:lane = 'optimization' AND w.ticket_id IS NULL))
        """).bindparams(lane=lane))).mappings().all()

        reaped = 0
        for r in rows:
            if r["wakeup_id"] in _inflight.get(r["agent_id"], {}):
                continue
            log.warning(
                "[wakeup] wakeup %s (%s) left its lock behind on an idle "
                "connection; reclaiming it so the wakeup can run",
                r["wakeup_id"][:8], r["agent_id"])
            await s.execute(
                text("SELECT pg_terminate_backend(:pid)").bindparams(pid=r["pid"])
            )
            reaped += 1
        return reaped


async def _reap_leaked_ticket_locks(lane: str | None = None) -> int:
    """Hand back per-ticket advisory locks (`ticket:<id>`) that leaked.

    The inner `ticket:` lock (taken inside the wakeup lock) stops overlapping
    heartbeats for one Ticket. Like the wakeup lock it is session-level, so a
    poisoned/cancelled path can strand it on an idle pooled connection. Because
    the Orchestrator supervisor is a single long-lived Ticket re-run on every
    child completion, a stranded `ticket:` lock means every future supervisor
    wakeup defers forever (`ticket_got == False`) and the run silently dies with
    no recovery — the exact gap the `wakeup:` reaper does NOT cover.

    Conservative, mirroring the wakeup reaper: a lock is leaked only if (a) its
    holder connection has been idle > 30s (rules out the acquire->mark-running
    window and, combined with (c), any legitimate hold), (b) the Ticket's Run is
    still active (a terminal run's lock is inert), and (c) NO wakeup for that
    Ticket is currently `running` — a running wakeup is real work holding the
    lock legitimately (possibly another daemon's). `_process_wakeup` flips the
    wakeup to `running` (committed) before `run_ticket`, so (c) is true for the
    entire processing duration. Terminating the holder returns the lock.
    """
    Session = get_session_factory()
    lane = _scheduler_lane(lane)
    async with Session() as s:
        rows = (await s.execute(text("""
            SELECT t.id AS ticket_id, l.pid
            FROM pg_locks l
            JOIN pg_stat_activity act ON act.pid = l.pid
            JOIN tickets t
              ON (hashtext('ticket:' || t.id)::bigint & 4294967295) = l.objid
             AND ((hashtext('ticket:' || t.id)::bigint >> 32) & 4294967295) = l.classid
            JOIN runs r ON r.id = t.run_id
            WHERE l.locktype = 'advisory'
              AND l.granted
              AND act.datname = current_database()
              AND act.pid <> pg_backend_pid()
              AND act.state IN ('idle', 'idle in transaction')
              AND act.state_change < now() - interval '30 seconds'
              AND r.status NOT IN :terminal
              AND t.lane = :lane
              AND NOT EXISTS (
                  SELECT 1 FROM agent_wakeup_requests w
                  WHERE w.ticket_id = t.id AND w.status = 'running'
              )
        """).bindparams(
            bindparam("terminal", value=sorted(TERMINAL_RUN_STATUSES), expanding=True),
            bindparam("lane", value=lane),
        ))).mappings().all()

        reaped = 0
        for r in rows:
            log.warning(
                "[wakeup] ticket %s left its lock behind on an idle connection "
                "(run still active); reclaiming it so the supervisor can run again",
                r["ticket_id"])
            await s.execute(
                text("SELECT pg_terminate_backend(:pid)").bindparams(pid=r["pid"])
            )
            reaped += 1
        return reaped


async def _sweep_stale_wakeups(
    stale_after_seconds: int = 300,
    lane: str | None = None,
) -> int:
    """Mark `running` wakeups older than `stale_after_seconds` as `failed`.

    Without this, a crashed runner/daemon leaves a wakeup in `running`
    forever, which our slot accounting then counts against the agent's
    max_concurrent_runs, blocking new wakeups for that agent.

    Wakeups this daemon is still working on are exempt however old they
    are. Training legitimately outlives any cutoff, and now that a long
    agent no longer blocks the loop, the sweep actually gets to run while
    one is mid-flight -- without the exemption it would declare live work
    dead and free the slot underneath it.
    """
    live = {wid for per_agent in _inflight.values() for wid in per_agent}
    lane = _scheduler_lane(lane)
    Session = get_session_factory()
    async with Session() as s:
        cutoff = datetime.now(timezone.utc) - _dt.timedelta(seconds=stale_after_seconds)
        result = await s.execute(
            update(AgentWakeupRequest)
            .where(
                AgentWakeupRequest.status == "running",
                AgentWakeupRequest.created_at < cutoff,
                (
                    or_(
                        AgentWakeupRequest.ticket_id.in_(
                            select(Ticket.id).where(Ticket.lane == lane)
                        ),
                        AgentWakeupRequest.ticket_id.is_(None),
                    )
                    if lane == "optimization"
                    else AgentWakeupRequest.ticket_id.in_(
                        select(Ticket.id).where(Ticket.lane == lane)
                    )
                ),
                AgentWakeupRequest.id.notin_(live) if live else sa_true(),
            )
            .values(
                status="failed",
                finished_at=datetime.now(timezone.utc),
                reason=AgentWakeupRequest.reason.op("||")(
                    " | swept as stale by wakeup-daemon"
                ),
            )
        )
        await s.commit()
        return int(result.rowcount or 0)


async def serve_forever(
    *,
    poll_interval_s: float = 1.0,
    cron_interval_s: int = 300,
    reconcile_interval_s: int = 5,
    # 30 min: training heartbeats download model weights + pip install torch
    # on the remote GPU + run the actual training, easily exceeding any
    # short cutoff. A heartbeat is only "stale" if it far outlasts a real
    # training run. Genuinely-crashed runners still get swept, just later.
    stale_after_seconds: int = 1800,
    lane: str | None = None,
) -> None:
    """Daemon main loop. Polls every poll_interval_s; fires inbox-cron
    every cron_interval_s; maintains external-job event streams and sweeps
    stale `running` wakeups + tickets + runs on every reconcile_interval_s.
    """
    lane = _scheduler_lane(lane)
    maintenance = lane == "optimization"
    log.info(
        "[wakeup-daemon] starting; lane=%s, poll=%ss, cron=%ss, "
        "reconcile=%ss, stale=%ss",
        lane, poll_interval_s, cron_interval_s,
        reconcile_interval_s, stale_after_seconds,
    )

    # Run the reconciler ONCE at startup so any crashed-from-previous-boot
    # tickets/runs get cleaned up before the daemon starts new work.
    if maintenance:
        try:
            report = await reconcile_runs_and_tickets(
                stale_ticket_seconds=stale_after_seconds,
            )
            if any(report.values()):
                log.info("[reconciler:boot] %s", report)
        except Exception as e:
            log.exception("[reconciler:boot] failed: %s", e)

    # A fresh daemon boot means ANY `running` wakeup is orphaned — the process
    # that owned it died (crash/restart). Sweep them IMMEDIATELY (age 0) rather
    # than waiting up to `stale_after_seconds`: an orphaned `running` wakeup eats
    # the agent's concurrency slot the whole time, and for an agent with
    # max_concurrent_runs=1 (e.g. orchestrator) that blocks it entirely, so the
    # queue looks "stuck, waiting for daemon". Safe at startup: nothing is
    # actually executing yet (single scheduler instance per deployment).
    try:
        orphaned = await _sweep_stale_wakeups(0, lane)
        if orphaned:
            log.info("[wakeup-daemon:boot] swept %d orphaned running wakeup(s)", orphaned)
    except Exception as e:
        log.exception("[wakeup-daemon:boot] orphan sweep failed: %s", e)

    # Immediately re-wake any Agent that has an unattended `queued` Ticket but
    # no pending wakeup (e.g. a ticket whose wakeup was consumed then lost to a
    # mid-run restart). Without this it would wait a full cron_interval (~5 min).
    try:
        await _cron_tick(cron_interval_s, lane)
    except Exception as e:
        log.exception("[wakeup-daemon:boot] initial cron tick failed: %s", e)

    last_cron = 0.0
    last_reconcile = 0.0
    last_stuck_infra = 0.0
    # Run the stuck-infra detector every 10 minutes (cheap; debounced
    # via notice cooldown so we don't spam runs with warnings).
    STUCK_INFRA_INTERVAL_S = 600.0
    loop_count = 0
    while True:
        try:
            swept = await _sweep_stale_wakeups(stale_after_seconds, lane)
            if swept:
                log.info("[wakeup-daemon] swept %d stale running wakeup(s)", swept)
        except Exception as e:
            log.exception("[wakeup-daemon] stale sweep failed: %s", e)
        try:
            await _reap_leaked_wakeup_locks(lane)
        except Exception as e:
            log.exception("[wakeup-daemon] lock reap failed: %s", e)
        try:
            await _reap_leaked_ticket_locks(lane)
        except Exception as e:
            log.exception("[wakeup-daemon] ticket lock reap failed: %s", e)
        try:
            n = await _drain_once(lane)
            if n > 0:
                log.info("[wakeup-daemon] started %d wakeup(s) this tick", n)
        except Exception as e:
            log.exception("[wakeup-daemon] drain failed: %s", e)

        loop_count += 1
        now_loop = loop_count * poll_interval_s
        if now_loop - last_cron >= cron_interval_s:
            try:
                await _cron_tick(cron_interval_s, lane)
                last_cron = now_loop
            except Exception as e:
                log.exception("[wakeup-daemon] cron tick failed: %s", e)

        if maintenance and now_loop - last_reconcile >= reconcile_interval_s:
            try:
                report = await reconcile_runs_and_tickets(
                    stale_ticket_seconds=stale_after_seconds,
                )
                if any(report.values()):
                    log.info("[reconciler] %s", report)
                last_reconcile = now_loop
            except Exception as e:
                log.exception("[reconciler] failed: %s", e)

        if maintenance and now_loop - last_stuck_infra >= STUCK_INFRA_INTERVAL_S:
            try:
                n = await _check_stuck_infra_instances()
                if n > 0:
                    log.warning(
                        "[stuck-infra] posted %d warning notice(s) for "
                        "long-lived instances", n,
                    )
                last_stuck_infra = now_loop
            except Exception as e:
                log.exception("[stuck-infra] check failed: %s", e)

        await asyncio.sleep(poll_interval_s)
