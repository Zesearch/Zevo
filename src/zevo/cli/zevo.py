"""Canonical command handlers for the interactive Zevo shell.

Users run ``zevo`` once and then enter commands without repeating the
executable name. The host shell dispatches each command to this Typer app.

Each command also has (or will have) a REST equivalent at /api/*.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zevo.engine.agent.loader import list_agent_ids, load_agent  # noqa: E402
from zevo.engine.agent.registry import seed_agents  # noqa: E402
from zevo.db import (  # noqa: E402
    Agent as AgentRow,
    TicketMessage,
    ExecutionEvent,
    RegistryModel,
    Run,
    Ticket,
    WorkProduct,
    get_session_factory,
)
from zevo.contracts.tickets import TERMINAL_RUN_STATUSES  # noqa: E402
from sqlalchemy import select, desc  # noqa: E402


console = Console()
app = typer.Typer(help="Zevo — a self-improving system for evolving language models.", add_completion=False)
agent_app = typer.Typer(help="Inspect agents.")
ticket_app = typer.Typer(help="Inspect tickets.")
model_app = typer.Typer(help="Inspect individual saved models.")
run_app = typer.Typer(help="Start and inspect orchestrator-driven pipeline Runs.")
price_app = typer.Typer(help="Show the cost price tables (model $/Mtok, GPU $/hour).")
dataset_app = typer.Typer(help="The Files catalogue: the same file sets the Files page lists.")
task_app = typer.Typer(help="The task catalogue — the same rows the Tasks page lists.")
app.add_typer(agent_app, name="agent")
app.add_typer(ticket_app, name="ticket")
app.add_typer(model_app, name="model")
app.add_typer(run_app, name="run")
app.add_typer(price_app, name="price")
app.add_typer(dataset_app, name="file")
app.add_typer(task_app, name="task")

# The files catalogue on disk (the same root the Files page serves).
# `file` and the shorthand resolver read it directly, so both work
# with the backend down. Resolved via zevo.paths so ZEVO_FILES_DIR is the
# ONE override, same as everywhere else.
from zevo.paths import files_root as _files_root
_FILES_DIR = Path(_files_root())


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """Zevo — a self-improving system for evolving language models."""
    if ctx.invoked_subcommand is None:
        from zevo.cli.shell import CommandRunner, interactive_shell

        interactive_shell(CommandRunner(REPO_ROOT, "backend", local=True))
    elif os.environ.get("ZEVO_INTERACTIVE_COMMAND") != "1":
        console.print("[yellow]Start Zevo first by running:[/] [bold cyan]zevo[/]")
        raise typer.Exit(2)


# ----------- price tables ------------------------------------------------

@price_app.command("models")
def price_models() -> None:
    """Show the per-model token price table ($ per 1M tokens), env overrides applied."""
    from zevo.engine.cost.pricing import PRICES_USD_PER_MTOK, price
    t = Table(box=None, pad_edge=False)
    t.add_column("model", style="cyan")
    t.add_column("input", justify="right", style="magenta")
    t.add_column("cached in", justify="right")
    t.add_column("output", justify="right", style="magenta")
    for m in sorted(PRICES_USD_PER_MTOK):
        t.add_row(
            m,
            f"${price(m, 'input'):.2f}",
            f"${price(m, 'cached_input'):.2f}",
            f"${price(m, 'output'):.2f}",
        )
    console.print(t)
    console.print(
        "[dim]edit [/]src/zevo/engine/cost/pricing.py[dim] (PRICES_USD_PER_MTOK) or override: [/]"
        "ZEVO_PRICE_<MODEL>_<INPUT|OUTPUT>_USD_PER_MTOK"
    )


@price_app.command("gpu")
def price_gpu() -> None:
    """Show the per-GPU-type hourly price table ($ per GPU-hour), env overrides applied."""
    from zevo.engine.cost.pricing import GPU_HOURLY_USD, gpu_hourly
    t = Table(box=None, pad_edge=False)
    t.add_column("gpu", style="cyan")
    t.add_column("$/GPU-hour", justify="right", style="magenta")
    for g in sorted(GPU_HOURLY_USD, key=lambda k: GPU_HOURLY_USD[k], reverse=True):
        t.add_row(g, f"${gpu_hourly(g):.2f}")
    console.print(t)
    console.print(
        "[dim]cost = $/GPU-hour × gpu_count × uptime. edit [/]src/zevo/engine/cost/pricing.py[dim] "
        "(GPU_HOURLY_USD) or override: [/]ZEVO_GPU_PRICE_<TYPE>_USD_PER_HOUR"
    )


# ----------- agent commands ----------------------------------------------

@agent_app.command("list")
def agent_list() -> None:
    """Every agent, with the model it runs on (UI: Agents).

    Reads the agents TABLE — the same rows the Agents page lists, so a driver or
    model changed at runtime shows here immediately. The on-disk definitions
    under playbook/agents/<id>/identity.md are what seeded it; `agent show <id>`
    prints those.
    """
    asyncio.run(_agent_list())


async def _agent_list() -> None:
    from zevo.db import Agent
    Session = get_session_factory()
    async with Session() as s:
        rows = list((await s.execute(select(Agent).order_by(Agent.id))).scalars().all())
    if not rows:
        console.print("[dim]no agents — run `seed-agents`[/dim]")
        return
    table = Table(box=None, pad_edge=False)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("title", no_wrap=True)
    table.add_column("driver", no_wrap=True)
    table.add_column("model", no_wrap=True)
    table.add_column("skills", justify="right")
    for r in rows:
        driver = r.default_driver or ""
        model = r.default_model or ""
        skills = 0
        try:
            blueprint = load_agent(r.id)
            skills = len(blueprint.skills)
            driver = driver or blueprint.default_driver
            model = model or blueprint.default_model
        except Exception:  # noqa: BLE001 — a row without files still lists
            pass
        table.add_row(
            r.id, r.title,
            driver or "[red](not set)[/red]",
            model or "[red](not set)[/red]",
            str(skills),
        )
    console.print(table)


@agent_app.command("skills")
def agent_skills(
    agent: str = typer.Argument(
        "", help="Agent id (e.g. train, data). Empty = every agent that has skills."
    ),
) -> None:
    """List each agent's method skills (from playbook/skills/<agent>/<skill>/SKILL.md).

    `method` is the id the orchestrator sets in a ticket's payload; `skill` is the
    folder/SKILL.md name (method with underscores→hyphens)."""
    ids = [agent] if agent else list_agent_ids()
    shown = False
    for aid in ids:
        try:
            bp = load_agent(aid)
        except Exception as e:  # unknown id / load error — report, keep going
            console.print(f"[red]{aid}:[/] {e}")
            continue
        if not bp.skills:
            if agent:
                console.print(f"[yellow]{aid}[/] has no skills.")
            continue
        shown = True
        table = Table(box=None, pad_edge=False)
        table.add_column("method", style="cyan")
        table.add_column("skill", style="green")
        table.add_column("description")
        for c in bp.skills:
            table.add_row(c.method, c.name, c.description)
        console.print(table)
    if not shown and not agent:
        console.print("[dim]No agents have skills.[/]")


@agent_app.command("status")
def agent_status(
    api_base: str = typer.Option(
        os.environ.get("ZEVO_API_BASE", "http://localhost:8001"), "--api-base"),
) -> None:
    """Show each agent's EFFECTIVE runtime driver + model.

    These are the live values the scheduler will use on the next heartbeat:
    the agents DB row (mutable via `agent set` / the Agents UI), which
    falls back to the identity.md frontmatter when an override is empty.
    """
    asyncio.run(_agent_status(api_base.rstrip("/")))


async def _agent_status(api_base: str) -> None:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{api_base}/api/agents")
            r.raise_for_status()
            rows = r.json()
    except Exception as e:  # noqa: BLE001 -- surface any connectivity error plainly
        console.print(f"[red]could not reach the backend at {api_base}[/]: {e}")
        console.print("[yellow]is the stack up? `docker compose up -d`[/]")
        raise typer.Exit(1)

    table = Table(box=None, pad_edge=False)
    table.add_column("id", style="cyan")
    table.add_column("driver", style="green")
    table.add_column("model", style="magenta")
    table.add_column("sandbox")
    any_unset = False
    for a in rows:
        aid = str(a.get("id", ""))
        db_driver = a.get("default_driver") or ""
        db_model = a.get("default_model") or ""
        # DB row wins; an empty field falls back to the identity.md frontmatter
        # (which itself may now be empty). Track where the value came from.
        driver, model = db_driver, db_model
        if not (driver and model):
            try:
                bp = load_agent(aid)
            except Exception:  # noqa: BLE001
                bp = None
            if bp is not None:
                driver = driver or bp.default_driver
                model = model or bp.default_model
        # Neither DB nor frontmatter set it -> the runner will refuse this agent.
        if not (driver and model):
            any_unset = True
            table.add_row(
                aid,
                driver or "[red](not set)[/]",
                model or "[red](not set)[/]",
                str(a.get("sandbox") or "none"),
            )
            continue
        table.add_row(aid, driver, model, str(a.get("sandbox") or "none"))
    console.print(table)
    if any_unset:
        console.print(
            "[red]some agents have no driver/model set[/] — they will fail to run. "
            "Set one with [bold]agent set <id> --driver <d> --model <m>[/]."
        )


@agent_app.command("show")
def agent_show(agent_id: str) -> None:
    """Print an agent's blueprint + first 40 lines of instructions."""
    bp = load_agent(agent_id)
    console.print(f"[bold cyan]{bp.id}[/]  {bp.title}")
    console.print(f"  model:         {bp.default_model}")
    console.print(f"  driver:        {bp.default_driver}")
    console.print(f"  output_schema: {bp.output_schema}")
    console.print(f"  tools:         {[t.__name__ if hasattr(t, '__name__') else type(t).__name__ for t in bp.tools]}")
    console.print(f"  identity.md:     {bp.identity_path}")
    console.rule("instructions (head)")
    for line in bp.instructions.splitlines()[:40]:
        console.print(line)
    console.rule()


@agent_app.command("instructions")
def agent_instructions(
    agent_id: str = typer.Argument(...),
    input_format: str = typer.Option("typed", "--format", help="typed or freeform"),
) -> None:
    """Print the complete assembled instructions shown by the UI."""
    if input_format not in ("typed", "freeform"):
        console.print("[red]--format must be typed or freeform.[/]")
        raise typer.Exit(1)
    data = asyncio.run(_api_get(
        f"/api/agents/{agent_id}/instructions?input_format={input_format}"
    ))
    if data is None:
        raise typer.Exit(1)
    console.print(data.get("content", ""))


@app.command("daemon")
def daemon_cmd(
    poll: float = typer.Option(1.0, "--poll", help="Wakeup-queue poll interval seconds."),
    cron: int = typer.Option(300, "--cron", help="Cron tick interval seconds (alarm-clock inbox scan)."),
    lane: str = typer.Option(
        "", "--lane",
        help="Ticket lane: optimization or held_out_test (defaults to ZEVO_SCHEDULER_LANE).",
    ),
) -> None:
    """Long-running heartbeat daemon for exactly one execution lane."""
    # Route the daemon's log.info() lines to stdout as structured JSON; without
    # this they fall through the default WARNING root logger and never surface.
    from zevo.api.observability import configure_logging
    configure_logging()
    from zevo.engine.run.scheduler.wakeup_daemon import serve_forever
    asyncio.run(serve_forever(
        poll_interval_s=poll, cron_interval_s=cron, lane=lane or None,
    ))


wakeups_app = typer.Typer(help="Inspect the agent wake-up queue.")
app.add_typer(wakeups_app, name="wakeups")


@wakeups_app.command("list")
def wakeups_list(
    agent: str = typer.Option("", "--agent"),
    status: str = typer.Option("", "--status"),
    limit: int = typer.Option(30, "--limit"),
) -> None:
    """List recent wake-up requests."""
    asyncio.run(_wakeups_list(agent, status, limit))


async def _wakeups_list(agent: str, status: str, limit: int) -> None:
    from sqlalchemy import desc as sql_desc
    from zevo.db import AgentWakeupRequest as W
    Session = get_session_factory()
    async with Session() as s:
        q = select(W).order_by(sql_desc(W.created_at)).limit(limit)
        if agent: q = q.where(W.agent_id == agent)
        if status: q = q.where(W.status == status)
        rows = (await s.execute(q)).scalars().all()
        t = Table(box=None, pad_edge=False)
        t.add_column("id");      t.add_column("agent")
        t.add_column("ticket");  t.add_column("source")
        t.add_column("status");  t.add_column("created")
        for r in rows:
            t.add_row(r.id[:8], r.agent_id, r.ticket_id or "-",
                      r.source, r.status, str(r.created_at)[:19])
        console.print(t)


@wakeups_app.command("show")
def wakeups_show(wakeup_id: str) -> None:
    """One wake-up request, with the payload the daemon will act on."""
    asyncio.run(_wakeups_show(wakeup_id))


async def _wakeups_show(wakeup_id: str) -> None:
    from zevo.db import AgentWakeupRequest as W
    Session = get_session_factory()
    async with Session() as s:
        # Accept short prefix
        rows = (await s.execute(select(W).where(W.id.like(f"{wakeup_id}%")))).scalars().all()
        if not rows:
            console.print(f"[red]no wakeup with id-prefix {wakeup_id!r}[/]")
            raise typer.Exit(1)
        for w in rows:
            console.print(f"[bold]wakeup[/] {w.id}")
            console.print(f"  agent:          {w.agent_id}")
            console.print(f"  ticket:         {w.ticket_id or '-'}")
            console.print(f"  source:         {w.source}")
            console.print(f"  status:         {w.status}")
            console.print(f"  heartbeat_run:  {w.heartbeat_run_id or '-'}")
            console.print(f"  trigger_detail: {w.trigger_detail}")
            console.print(f"  reason:         {w.reason}")
            console.print(f"  created_at:     {w.created_at}")
            console.print(f"  finished_at:    {w.finished_at or '-'}")


@app.command("server")
def server_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev)."),
) -> None:
    """Run the FastAPI backend (uvicorn)."""
    import uvicorn
    uvicorn.run("zevo.api.main:app", host=host, port=port, reload=reload)


@agent_app.command("run")
def agent_run(
    agent_id: str = typer.Argument(...),
    ticket_id: str = typer.Option(..., "--ticket"),
    driver: str = typer.Option("", "--driver"),
    model: str = typer.Option("", "--model"),
    work_dir: str = typer.Option("", "--work-dir"),
) -> None:
    """Run a single heartbeat for one ticket. Repair / standalone path."""
    async def go() -> None:
        from zevo.engine.run.runner import run_ticket
        from zevo.db import Ticket
        from zevo.db import get_session_factory
        Session = get_session_factory()
        async with Session() as s:
            tk = (await s.execute(select(Ticket).where(Ticket.id == ticket_id))).scalar_one_or_none()
            if tk is None:
                console.print(f"[red]no ticket {ticket_id}[/]")
                raise typer.Exit(1)
            if tk.agent_id != agent_id:
                console.print(f"[yellow]ticket {ticket_id} is assigned to {tk.agent_id}, not {agent_id} -- continuing anyway[/]")
            await run_ticket(
                s, ticket_id=ticket_id,
                work_dir_root=work_dir,
                driver_name=driver, model_override=model,
            )
            await s.refresh(tk)
            console.print(f"[bold]ticket {ticket_id}[/] -> {tk.status}: {escape(str(tk.summary or '')[:120])}")
    asyncio.run(go())


@agent_app.command("invoke")
def agent_invoke(
    agent_id: str = typer.Argument(...),
    driver: str = typer.Option("", "--driver"),
    model: str = typer.Option("", "--model"),
    work_dir: str = typer.Option("", "--work-dir"),
    foreground: bool = typer.Option(
        False, "--foreground",
        help="Run inline in this process instead of enqueuing a wakeup. Useful for dev/debug.",
    ),
    watch: bool = typer.Option(
        False, "--watch", "-w",
        help="After enqueuing, poll the wakeup until a heartbeat runs, then tail its stdout live.",
    ),
) -> None:
    """Pulse an agent. Default: enqueue a wakeup; the daemon picks it
    up. With --foreground: run inline in this process. With --watch:
    enqueue then stream the resulting heartbeat's transcript live."""
    async def go() -> None:
        from zevo.db import Ticket, get_session_factory
        Session = get_session_factory()
        async with Session() as s:
            candidate = (await s.execute(
                select(Ticket)
                .where(Ticket.agent_id == agent_id,
                       Ticket.status == "queued")
                .order_by(Ticket.created_at)
                .limit(1)
            )).scalar_one_or_none()
            if candidate is None:
                console.print(f"[yellow]no queued tickets assigned to {agent_id}[/]")
                console.print(f"[dim]hint: create one with `agent task {agent_id} \"...\"`[/]")
                return
            if foreground:
                from zevo.engine.run.runner import run_ticket
                console.print(f"[dim]invoking {agent_id} on ticket {candidate.id} [foreground][/]")
                await run_ticket(s, ticket_id=candidate.id, work_dir_root=work_dir,
                                 driver_name=driver, model_override=model)
                await s.refresh(candidate)
                console.print(f"[bold]ticket {candidate.id}[/] -> {candidate.status}: {escape(str(candidate.summary or '')[:140])}")
            else:
                from zevo.engine.run.wakeup import queue_wakeup
                w = await queue_wakeup(
                    s, agent_id=agent_id, ticket_id=candidate.id,
                    source="on_demand",
                    reason=f"manual agent invoke (driver={driver!r}, model={model!r})",
                    payload={"driver": driver, "model": model},
                )
                console.print(f"[green]enqueued wakeup {w.id[:8]}[/] for ticket {candidate.id}")
                if watch:
                    await _watch_wakeup_until_done(w.id)
                else:
                    console.print(f"[dim]tail with: heartbeats tail <id>  (or --watch next time)[/]")
    asyncio.run(go())


async def _watch_wakeup_until_done(wakeup_id: str) -> None:
    """Poll a wakeup; when its heartbeat_run_id is set, tail it live."""
    from zevo.db import AgentWakeupRequest, HeartbeatRun
    Session = get_session_factory()
    console.print(f"[dim]waiting for daemon to pick up wakeup...[/]")
    for _ in range(120):
        async with Session() as s:
            w = (await s.execute(
                select(AgentWakeupRequest).where(AgentWakeupRequest.id == wakeup_id)
            )).scalar_one()
            if w.heartbeat_run_id:
                console.print(f"[dim]heartbeat {w.heartbeat_run_id[:8]} -- tailing[/]")
                await _heartbeats_tail(w.heartbeat_run_id, follow=True, interval=1.0)
                return
            if w.status in ("completed", "failed"):
                console.print(f"[dim]wakeup -> {w.status} (no heartbeat to tail)[/]")
                return
        await asyncio.sleep(1)
    console.print("[yellow]gave up waiting after 120s[/]")


# ----------- heartbeats commands ----------------------------------------

heartbeats_app = typer.Typer(help="Inspect + tail agent heartbeats (live transcript).")
app.add_typer(heartbeats_app, name="heartbeats")


@heartbeats_app.command("list")
def heartbeats_list(
    agent: str = typer.Option("", "--agent"),
    ticket: str = typer.Option("", "--ticket"),
    limit: int = typer.Option(30, "--limit"),
) -> None:
    """Recent heartbeats — one row per agent invocation."""
    asyncio.run(_heartbeats_list(agent, ticket, limit))


async def _heartbeats_list(agent: str, ticket: str, limit: int) -> None:
    from sqlalchemy import desc as sql_desc
    from zevo.db import HeartbeatRun
    Session = get_session_factory()
    async with Session() as s:
        q = select(HeartbeatRun).order_by(sql_desc(HeartbeatRun.started_at)).limit(limit)
        if agent: q = q.where(HeartbeatRun.agent_id == agent)
        if ticket: q = q.where(HeartbeatRun.ticket_id == ticket)
        rows = (await s.execute(q)).scalars().all()
        t = Table(box=None, pad_edge=False)
        t.add_column("id"); t.add_column("agent"); t.add_column("ticket")
        t.add_column("driver"); t.add_column("model"); t.add_column("exit")
        t.add_column("started")
        for r in rows:
            t.add_row(r.id[:8], r.agent_id, r.ticket_id, r.driver,
                      r.model[:18], str(r.exit_code), str(r.started_at)[:19])
        console.print(t)


@heartbeats_app.command("tail")
def heartbeats_tail(
    heartbeat_id: str = typer.Argument(..., help="heartbeat id (or 8-char prefix)"),
    follow: bool = typer.Option(True, "--follow/--no-follow", "-f/-F"),
    interval: float = typer.Option(1.0, "--interval"),
    api_base: str = typer.Option(
        os.environ.get("ZEVO_API_BASE", "http://localhost:8001"), "--api-base",
        help="Backend base URL for live WS streaming.",
    ),
    raw_stdout: bool = typer.Option(
        False, "--raw",
        help="Tail the raw stdout log file instead of structured events.",
    ),
) -> None:
    """Stream a heartbeat's live transcript to your terminal.

    By default uses the structured event WebSocket (matches the web UI).
    Pass --raw to inspect the process stdout log instead."""
    asyncio.run(_heartbeats_tail(heartbeat_id, follow, interval, api_base, raw_stdout))


async def _heartbeats_tail(
    prefix: str, follow: bool, interval: float, api_base: str, raw_stdout: bool,
) -> None:
    from zevo.db import HeartbeatRun
    from pathlib import Path
    Session = get_session_factory()

    # Resolve id-prefix locally so we can pass full id to the WS endpoint.
    async with Session() as s:
        hb = (await s.execute(
            select(HeartbeatRun).where(HeartbeatRun.id.like(f"{prefix}%"))
        )).scalar_one_or_none()
    if hb is None:
        console.print(f"[red]no heartbeat with id-prefix {prefix!r}[/]")
        raise typer.Exit(1)
    console.print(
        f"[dim]tailing heartbeat[/] {hb.id[:8]} agent={hb.agent_id} ticket={hb.ticket_id}"
    )

    if not raw_stdout:
        # Structured event stream (matches web UI).
        await _stream_heartbeat_ws(hb.id, api_base=api_base.rstrip("/"))
        return

    # --raw: the durable process log, useful when event parsing itself failed.
    cursor = 0
    while True:
        async with Session() as s:
            hb = (await s.execute(
                select(HeartbeatRun).where(HeartbeatRun.id == hb.id)
            )).scalar_one()
        if hb.stdout_path:
            p = Path(hb.stdout_path)
            if p.exists():
                data = p.read_text(encoding="utf-8", errors="replace")
                if len(data) > cursor:
                    print(data[cursor:], end="", flush=True)
                    cursor = len(data)
        if not follow or hb.finished_at is not None:
            if hb.finished_at is not None:
                console.print(f"\n[dim]heartbeat finished -- exit_code={hb.exit_code}[/]")
            break
        await asyncio.sleep(interval)


@agent_app.command("task")
def agent_task(
    agent_id: str = typer.Argument(..., help="Which agent should handle this."),
    nl_request: str = typer.Argument(..., help="What you want done, in plain English."),
    attach: list[str] = typer.Option(
        None, "--attach", "-a",
        help="Local file to upload + attach. Repeatable.",
    ),
    run_id: str = typer.Option("", "--run", help="Attach the ticket to an existing run id."),
    task_name: str = typer.Option(
        "", "--task", help="Required when --run is omitted: reusable task name.",
    ),
    run_name: str = typer.Option(
        "", "--run-name", help="Required when --run is omitted: name of this execution.",
    ),
    watch: bool = typer.Option(
        True, "--watch/--no-watch", "-w/-W",
        help="After creating the ticket, stream the agent's live transcript.",
    ),
) -> None:
    """Describe a task in natural language; the agent figures out how to do it.

    Example:

        agent task data \\
            "Make a 100-row chat-JSONL dataset from this PDF" \\
            --attach ./notes.pdf --watch
    """
    asyncio.run(_agent_task(
        agent_id=agent_id,
        nl_request=nl_request,
        attachments_local=attach or [],
        run_id=run_id, task_name=task_name, run_name=run_name,
        watch=watch,
        api_base=_API_BASE,
    ))


async def _agent_task(
    *, agent_id: str, nl_request: str, attachments_local: list[str],
    run_id: str, task_name: str, run_name: str, watch: bool, api_base: str,
) -> None:
    import httpx
    if agent_id == "evaluation":
        console.print(
            "[red]evaluation is a deterministic pipeline runner and cannot "
            "accept a natural-language agent task.[/]"
        )
        raise typer.Exit(1)
    # 1) Upload attachments
    uploaded_paths: list[str] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for local in attachments_local:
            p = Path(local).expanduser()
            if not p.exists():
                console.print(f"[red]attachment not found:[/] {p}")
                raise typer.Exit(1)
            with p.open("rb") as fh:
                r = await client.post(
                    f"{api_base}/api/attachments",
                    files={"file": (p.name, fh)},
                )
            if r.status_code >= 400:
                console.print(f"[red]upload failed:[/] {r.status_code} {r.text}")
                raise typer.Exit(1)
            j = r.json()
            uploaded_paths.append(j["path"])
            console.print(
                f"[dim]uploaded[/] {p.name} -> [dim]{j['path']}[/] ({j['size_bytes']} B)"
            )

        # 2) POST the freeform ticket
        body: dict = {
            "agent_id": agent_id,
            "input_format": "freeform",
            "payload": {
                "request": nl_request,
                "attachments": uploaded_paths,
            },
        }
        if run_id:
            body["run_id"] = run_id
        else:
            if not task_name.strip() or not run_name.strip():
                console.print("[red]--task and --run-name are required when --run is omitted.[/]")
                raise typer.Exit(1)
            body["task_name"] = task_name.strip()
            body["run_name"] = run_name.strip()
        r = await client.post(f"{api_base}/api/tickets", json=body)
        if r.status_code >= 400:
            console.print(f"[red]ticket create failed:[/] {r.status_code} {r.text}")
            raise typer.Exit(1)
        ticket = r.json()
        ticket_id = ticket["id"]
        console.print(
            f"[green]created ticket[/] [bold]{ticket_id}[/] "
            f"-> agent {agent_id} (freeform). status={ticket.get('status','queued')}"
        )

    if not watch:
        console.print(
            f"[dim]tail later with[/] heartbeats tail <heartbeat-id> "
            f"or pass --watch next time"
        )
        return

    # 4) Wait for a heartbeat to appear for this ticket, then stream its WS.
    await _wait_then_stream(ticket_id=ticket_id, api_base=api_base)


# ----------- run commands (full pipeline) --------------------------------

# Where to reach the backend API. Inside the scheduler/backend containers this
# is http://backend:8000 (set via ZEVO_API_BASE); from the host it falls back to
# the docker-mapped port. The UI link we print is always the host-facing frontend.
_API_BASE = os.environ.get("ZEVO_API_BASE", "http://localhost:8001")
_UI_BASE = os.environ.get("ZEVO_UI_BASE", "http://localhost:5173")

_UR_DEFAULTS = {
    "task_objective": "",
    "metric": "",
    "metric_type": "builtin",
    "evaluation_script": "",
    "evaluator_sha256": "",
    "validation_metric": "token_f1",
    "validation_metric_type": "builtin",
    "validation_metric_direction": "max",
    "validation_evaluation_script": "",
    "validation_evaluator_sha256": "",
    "training_method": "",
    "dataset": "",
    "data_query": "",
    "base_model": "",
    "model_query": "",
    "method_query": "",
    "test_set": "",
    "test_answer_fields": [],
    "validation_set": "",
    "validation_answer_fields": [],
    "validation_sample_submission": "",
    "test_sample_submission": "",
    "constraints": [],
}


async def _task_by_name(name: str):
    """The task row, or None. The catalogue lives in the database — the same
    rows the Tasks page lists, so the CLI and the UI cannot disagree."""
    from zevo.db import Task
    Session = get_session_factory()
    async with Session() as s:
        return await s.get(Task, name)


def _columns(spec: str) -> list[str]:
    """`"answer, gold"` -> `["answer", "gold"]`. Blank entries dropped, because
    a trailing comma is a typo and not a column called "".""" 
    return [c.strip() for c in (spec or "").split(",") if c.strip()]


def _resolve_data_ref(ref: str) -> str:
    """Turn a catalogue shorthand into the path a run can read.

    `medqa-usmle/train.csv` is how the UI names a file, and typing the full
    /app/data/files/... prefix at a terminal is nobody's idea of a good
    time. A hub id looks the same (`trl-lib/Capybara`), so the shorthand only
    wins when that dataset and file actually exist — otherwise the string is
    passed through untouched.
    """
    r = (ref or "").strip()
    if not r or r.startswith("/") or r.count("/") != 1:
        return r
    ds, _, fname = r.partition("/")
    candidate = _FILES_DIR / ds / fname
    return str(candidate) if candidate.is_file() else r


@run_app.command("create")
def run_create(
    objective: str = typer.Argument(
        "",
        help="CUSTOM task: describe it in plain English. "
             "Leave empty and use --task to run a predefined task instead.",
    ),
    task: str = typer.Option(
        "", "--task", "-t",
        help="REQUIRED. Task name. A predefined task name (med / math "
             "/ tiny / …) runs it as-is — list them with `task list`. "
             "Any other name defines a custom task, which also needs an OBJECTIVE.",
    ),
    run_name: str = typer.Option(
        "", "--run-name", help="REQUIRED. Human-readable name for this execution.",
    ),
    # ---- custom-task fields (each its own flag) ----
    base_model: str = typer.Option(
        "", "--base-model", help="HF model id, e.g. Qwen/Qwen2.5-7B-Instruct.",
    ),
    model_query: str = typer.Option(
        "", "--model-query",
        help="Natural-language guidance for Zevo when --base-model is blank.",
    ),
    gpu_provider: str = typer.Option(
        "", "--gpu-provider", help="where to get the GPU: cluster (SSH login node + finite Slurm jobs) | cloud (rent through Vast.ai or Lambda Cloud) | instance (fixed directly reachable GPU host, no Slurm). Empty defaults to instance.",
    ),
    num_gpus: int = typer.Option(
        0, "--num-gpus", help="maximum GPUs this run may use at once. Infrastructure chooses an actual positive count at or below this limit for cloud, cluster, or instance. Missing or 0 means unlimited.",
    ),
    dataset: str = typer.Option(
        "", "--dataset",
        help="training data: a path, a catalogue shorthand like medqa-usmle/train.csv, "
             "or a HuggingFace id. Or use --attach to upload one.",
    ),
    dataset_split: str = typer.Option(
        "", "--dataset-split", help="HuggingFace training split; blank uses catalogue/train resolution.",
    ),
    dataset_config: str = typer.Option(
        "", "--dataset-config", help="Named HuggingFace training dataset configuration.",
    ),
    data_query: str = typer.Option(
        "", "--data-query", help="What Data should acquire when --dataset is blank.",
    ),
    test_set: str = typer.Option(
        "", "--test-set",
        help="test set WITH the answers — what evaluation scores against. "
             "A path, or a catalogue shorthand like medqa-usmle/test.csv.",
    ),
    answer_fields: str = typer.Option(
        "", "--answer-fields",
        help="where the ground truth lives in the test set, comma-separated: a "
             "column of a CSV, a key of a JSON record. The data agent drops "
             "these to build the questions-only copy inference is given.",
    ),
    validation_set: str = typer.Option(
        "", "--validation-set",
        help="validation set WITH the answers — what the run TUNES on. When "
             "empty, Zevo moves 20% of Test into Validation before the Run; "
             "the split must contain at least 200 rows.",
    ),
    validation_split: str = typer.Option(
        "", "--validation-split", help="HuggingFace validation-like split.",
    ),
    validation_config: str = typer.Option(
        "", "--validation-config", help="Named HuggingFace validation dataset configuration.",
    ),
    validation_answer_fields: str = typer.Option(
        "", "--validation-answer-fields",
        help="where the ground truth lives in the validation set. Required "
             "when --validation-set is supplied.",
    ),
    validation_sample_submission: str = typer.Option(
        "", "--validation-sample-submission",
        help="submission template for the validation set: the columns inference "
             "must emit. Required when --validation-set is supplied.",
    ),
    test_sample_submission: str = typer.Option(
        "", "--test-sample-submission", help="held-out submission-template CSV path (defines output columns).",
    ),
    metric_type: str = typer.Option(
        "", "--metric-type",
        help="Override the held-out Test metric implementation: builtin or custom. Omitted uses the Task default.",
    ),
    evaluation_script: str = typer.Option(
        "", "--evaluation-script",
        help="Custom Python evaluator used only for held-out Test.",
    ),
    validation_metric_type: str = typer.Option(
        "", "--validation-metric-type",
        help="Validation metric implementation: builtin or custom. A saved Setting supplies it when omitted.",
    ),
    validation_metric: str = typer.Option(
        "", "--validation-metric",
        help="Validation metric value name used to compare iterations.",
    ),
    validation_target: str = typer.Option(
        "", "--validation-target",
        help="Validation direction: Max or Min.",
    ),
    validation_evaluation_script: str = typer.Option(
        "", "--validation-evaluation-script",
        help="Custom Python evaluator used only for Validation.",
    ),
    training_method: str = typer.Option(
        "", "--training-method", help="pin an installed Train Skill method id (empty = Zevo owns the hierarchical Method branch).",
    ),
    method_query: str = typer.Option(
        "", "--method-query",
        help="Natural-language guidance for Zevo when --training-method is blank.",
    ),
    teacher_model: str = typer.Option(
        "", "--teacher-model",
        help="Hugging Face owner/model id required by --training-method gkd.",
    ),
    reward_model: str = typer.Option(
        "", "--reward-model",
        help="Hugging Face owner/model id required by --training-method online_dpo.",
    ),
    use_peft: Optional[bool] = typer.Option(
        None, "--use-peft/--no-use-peft",
        help="Pin adapter or full-parameter training for methods that support PEFT.",
    ),
    prompt_framing: str = typer.Option(
        "", "--prompt-framing",
        help="Customized Pipeline only: pin chat, chat:<template-model>, completion, or text.",
    ),
    system_prompt: str = typer.Option(
        "", "--system-prompt", help="Customized Pipeline only: exact chat system turn.",
    ),
    loss_objective_config: str = typer.Option(
        "", "--loss-objective-config",
        help="Customized Pipeline only: JSON object or @file.json with fixed loss values.",
    ),
    inference_config: str = typer.Option(
        "", "--inference-config",
        help="Customized Pipeline only: JSON object or @file.json with fixed task mapping and parsing values.",
    ),
    decoding_config: str = typer.Option(
        "", "--decoding-config",
        help="Customized Pipeline only: JSON object or @file.json with fixed generation values.",
    ),
    attach: list[str] = typer.Option(
        None, "--attach", "-a", help="Local file to upload + use as the dataset. Repeatable.",
    ),
    # ---- run-level knobs (both modes) ----
    iterations: Optional[int] = typer.Option(
        None, "--iterations", help="Max trained loops (Train→Inference→Evaluation→Registry). Omitted or 0 = no hard iteration cap.",
    ),
    max_cost: Optional[float] = typer.Option(
        None, "--max-cost", help="Hard $ cap on the whole run (LLM tokens + GPU rental). Omitted or 0 = no cap.",
    ),
    max_runtime_hours: Optional[float] = typer.Option(
        None, "--max-runtime-hours",
        help="Wall-clock limit for this Run in hours. Omitted or 0 = unlimited; never saved in a Setting.",
    ),
    max_queue_wait_hours: Optional[float] = typer.Option(
        None, "--max-queue-wait-hours",
        help="Maximum Slurm PENDING time per automatic submission (>0, max 168 hours). Omitted = 24 hours. Queue wait does not consume Run runtime.",
    ),
    stop_threshold: Optional[float] = typer.Option(
        None, "--stop-threshold",
        help="Optional validation-score threshold on the Validation metric's own scale; omit to disable threshold stopping.",
    ),
    metric: str = typer.Option(
        "", "--metric",
        help="Held-out Test value name, e.g. accuracy, benchmark_average, or loss. Omitted uses the Task default.",
    ),
    metric_direction: str = typer.Option(
        "", "--target",
        help="Held-out Test direction: Max or Min. Omitted uses the Task default.",
    ),
    watch: bool = typer.Option(
        True, "--watch/--no-watch", "-w/-W",
        help="After starting, stream the run-level timeline in the terminal.",
    ),
    repeat: Optional[int] = typer.Option(
        None, "--repeat", "-n",
        help="Launch the SAME run this many times (category-1 robustness → a success ratio). Default: 1. >1 implies --no-watch.",
    ),
    generation_backend: str = typer.Option(
        "", "--generation-backend",
        help="Generation backend for inference + rollout-based training: hf | vllm. Empty = vllm.",
    ),
    mode: str = typer.Option(
        "full_pipeline", "--mode",
        help="full_pipeline or customized_pipeline.",
    ),
    customizations: str = typer.Option(
        "", "--customizations",
        help="JSON file containing RunCustomizations; required for customized_pipeline.",
    ),
    setting: str = typer.Option(
        "", "--setting", help="Saved setting id or exact name for this Task.",
    ),
    save_setting: bool = typer.Option(
        False, "--save-setting", help="Save this Task/configuration for reuse.",
    ),
    setting_name: str = typer.Option(
        "", "--setting-name", help="Name used with --save-setting (max 32 characters).",
    ),
    allow_risky: bool = typer.Option(
        False, "--allow-risky", help="Launch when preflight has warnings but no blockers.",
    ),
) -> None:
    """Start a Full Pipeline or Customized Pipeline Run.

    1) Predefined task (the catalogue names the problem; the per-field
       flags below still apply and become this run's settings):

        run create --task med --run-name med-lora-try1

    2) CUSTOM task (define it with per-field flags):

        run create "Fine-tune on basketball-rules Q&A, target 80%" \\
            --task basketball --run-name basketball-try1 \\
            --base-model Qwen/Qwen2.5-7B-Instruct --gpu-provider cloud \\
            --attach ./basketball.jsonl --test-set /app/data/files/bball/test.csv \\
            --target max --iterations 3 --max-cost 80

    Either way the orchestrator drives every agent, the daemon runs it, and the
    UI shows it live.
    """
    # A bare `run create` inside the command shell prompts for its two identity
    # fields. Non-interactive calls keep the strict missing-field errors below.
    if sys.stdin.isatty():
        if not task.strip():
            task = typer.prompt("Task")
        if not run_name.strip():
            run_name = typer.prompt("Run name")

    # The run executes server-side (scheduler daemon), independent of this CLI —
    # so a plain Ctrl-C here only stops the local watcher, NOT the run. Capture
    # the started run id and, on Ctrl-C, cancel the run (stop tickets + release
    # any GPU) so interrupting `run create` actually stops the task.
    started: list[str] = []
    try:
        asyncio.run(_run_create(
            objective=objective,
            task=task,
            run_name=run_name,
            base_model=base_model,
            model_query=model_query,
            gpu_provider=gpu_provider,
            num_gpus=num_gpus,
            dataset=dataset,
            dataset_split=dataset_split,
            dataset_config=dataset_config,
            data_query=data_query,
            test_set=test_set,
            answer_fields=answer_fields,
            validation_set=validation_set,
            validation_split=validation_split,
            validation_config=validation_config,
            validation_answer_fields=validation_answer_fields,
            validation_sample_submission=validation_sample_submission,
            test_sample_submission=test_sample_submission,
            metric_type=metric_type,
            evaluation_script=evaluation_script,
            validation_metric_type=validation_metric_type,
            validation_metric=validation_metric,
            validation_metric_direction=validation_target,
            validation_evaluation_script=validation_evaluation_script,
            training_method=training_method,
            method_query=method_query,
            teacher_model=teacher_model,
            reward_model=reward_model,
            use_peft=use_peft,
            prompt_framing=prompt_framing,
            system_prompt=system_prompt,
            loss_objective_config=loss_objective_config,
            inference_config=inference_config,
            decoding_config=decoding_config,
            attachments_local=attach or [],
            iterations=iterations,
            max_cost=max_cost,
            max_runtime_hours=max_runtime_hours,
            max_queue_wait_hours=max_queue_wait_hours,
            stop_threshold=stop_threshold,
            metric=metric,
            metric_direction=metric_direction,
            watch=watch,
            repeat=repeat,
            generation_backend=generation_backend,
            mode=mode, customizations_path=customizations,
            setting_ref=setting, save_setting=save_setting,
            setting_name=setting_name, allow_risky=allow_risky,
            api_base=_API_BASE,
            started=started,
        ))
    except KeyboardInterrupt:
        rid = started[-1] if started else ""
        if rid:
            console.print(f"\n[yellow]interrupted — cancelling run[/] {rid} …")
            try:
                asyncio.run(_run_cancel(rid, api_base=_API_BASE))
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]cancel failed:[/] {e} — cancel manually: run cancel {rid[:8]}")
        else:
            console.print("\n[yellow]interrupted before the run started.[/]")
        raise typer.Exit(130)


async def _run_create(
    *, objective: str, task: str, run_name: str,
    base_model: str, model_query: str = "", gpu_provider: str, num_gpus: int,
    dataset: str, test_set: str,
    dataset_split: str, dataset_config: str, data_query: str,
    answer_fields: str, validation_set: str, validation_answer_fields: str,
    validation_split: str, validation_config: str,
    validation_sample_submission: str,
    test_sample_submission: str, metric_type: str, evaluation_script: str,
    validation_metric_type: str, validation_metric: str,
    validation_metric_direction: str, validation_evaluation_script: str,
    training_method: str, method_query: str = "",
    teacher_model: str, reward_model: str, use_peft: Optional[bool],
    prompt_framing: str, system_prompt: str,
    loss_objective_config: str, inference_config: str, decoding_config: str,
    attachments_local: list[str],
    iterations: Optional[int], max_cost: Optional[float],
    max_runtime_hours: Optional[float], max_queue_wait_hours: Optional[float] = None,
    stop_threshold: Optional[float],
    metric: str, metric_direction: str,
    watch: bool, api_base: str, repeat: Optional[int] = None, generation_backend: str = "",
    mode: str = "full_pipeline", customizations_path: str = "",
    setting_ref: str = "", save_setting: bool = False, setting_name: str = "",
    allow_risky: bool = False,
    started: Optional[list] = None,
) -> None:
    import httpx
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Only send run-level knobs the user actually set; otherwise the backend
        # resolves uncapped limits and the system runtime defaults.
        run_opts: dict = {}
        if iterations is not None:
            run_opts["iteration_budget"] = iterations
        if stop_threshold is not None:
            run_opts["stop_threshold"] = stop_threshold
        if max_cost is not None:
            run_opts["max_cost_usd"] = max_cost
        if max_runtime_hours is not None:
            run_opts["max_runtime_hours"] = max_runtime_hours
        if max_queue_wait_hours is not None:
            run_opts["max_queue_wait_hours"] = max_queue_wait_hours
        if generation_backend:
            run_opts["generation_backend"] = generation_backend
        if gpu_provider:
            run_opts["gpu_provider"] = gpu_provider
        # A run-level knob like the ones above, NOT a user_request override:
        # the runner reads it off the Run row to stamp the infra ticket.
        if num_gpus:
            run_opts["num_gpus"] = num_gpus
        mode = mode.strip().lower()
        if mode not in ("full_pipeline", "customized_pipeline"):
            console.print("[red]--mode must be full_pipeline or customized_pipeline.[/]")
            raise typer.Exit(1)
        detailed_flags = {
            "--prompt-framing": prompt_framing,
            "--system-prompt": system_prompt,
            "--loss-objective-config": loss_objective_config,
            "--inference-config": inference_config,
            "--decoding-config": decoding_config,
        }
        supplied_details = [
            flag for flag, value in detailed_flags.items() if value.strip()
        ]
        if mode == "full_pipeline" and supplied_details:
            console.print(
                "[red]Full Pipeline does not accept detailed prompt, loss, or "
                "inference flags.[/] Use --mode customized_pipeline for: "
                + ", ".join(supplied_details)
            )
            raise typer.Exit(1)
        run_opts["mode"] = mode
        if len(setting_name) > 32:
            console.print("[red]--setting-name must be at most 32 characters.[/]")
            raise typer.Exit(1)
        if save_setting:
            run_opts.update(save_setting=True, setting_name=setting_name.strip())
        customizations_body: dict = {}
        if customizations_path:
            p = Path(customizations_path).expanduser()
            try:
                customizations_body = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                console.print(f"[red]could not read customizations JSON:[/] {exc}")
                raise typer.Exit(1)
        if mode == "customized_pipeline":
            if not customizations_path:
                console.print("[red]customized_pipeline requires --customizations FILE.json.[/]")
                raise typer.Exit(1)
            run_opts["customizations"] = customizations_body
        elif customizations_path:
            console.print("[red]--customizations is valid only with --mode customized_pipeline.[/]")
            raise typer.Exit(1)

        # Every run is named, predefined or not — the backend rejects a blank
        # task_name, and unnamed runs used to show as "(unnamed)" in every list.
        task = (task or "").strip()
        if not task:
            console.print("[red]--task is required.[/] Name the run: a task from the catalogue "
                          "(see `task list`) or any name of your own for a custom task.")
            raise typer.Exit(1)
        run_name = (run_name or "").strip()
        if not run_name:
            console.print("[red]--run-name is required.[/] Name this execution separately from its task.")
            raise typer.Exit(1)
        metric_direction = (metric_direction or "").strip().lower()
        metric = (metric or "").strip()
        metric_type = (metric_type or "").strip().lower()
        if metric_type not in ("", "builtin", "custom"):
            console.print("[red]--metric-type must be builtin or custom.[/]")
            raise typer.Exit(1)
        if metric_direction not in ("", "max", "min"):
            console.print("[red]--target must be Max or Min.[/]")
            raise typer.Exit(1)
        if metric_type == "custom" and not evaluation_script.strip():
            console.print("[red]custom Test metric requires --evaluation-script.[/]")
            raise typer.Exit(1)
        validation_metric_type = (validation_metric_type or "").strip().lower()
        validation_metric = (validation_metric or "").strip()
        validation_metric_direction = (validation_metric_direction or "").strip().lower()
        if validation_metric_type not in ("", "builtin", "custom"):
            console.print("[red]--validation-metric-type must be builtin or custom.[/]")
            raise typer.Exit(1)
        if validation_metric_direction not in ("", "max", "min"):
            console.print("[red]--validation-target must be Max or Min.[/]")
            raise typer.Exit(1)
        if validation_metric_type == "custom" and not validation_evaluation_script.strip():
            console.print("[red]custom Validation metric requires --validation-evaluation-script.[/]")
            raise typer.Exit(1)
        method_config: dict[str, object] = {}
        if teacher_model.strip():
            method_config["teacher_model"] = teacher_model.strip()
        if reward_model.strip():
            method_config["reward_model"] = reward_model.strip()
        if use_peft is not None:
            method_config["use_peft"] = use_peft

        def _json_object(raw: str, label: str) -> dict:
            text = raw.strip()
            if not text:
                return {}
            try:
                value = (
                    json.loads(Path(text[1:]).expanduser().read_text(encoding="utf-8"))
                    if text.startswith("@") else json.loads(text)
                )
            except (OSError, json.JSONDecodeError) as exc:
                console.print(f"[red]{label} must be JSON or @file.json:[/] {exc}")
                raise typer.Exit(1)
            if not isinstance(value, dict):
                console.print(f"[red]{label} must decode to a JSON object.[/]")
                raise typer.Exit(1)
            return value

        decision_pin_overrides = {
            "prompt_framing": prompt_framing.strip(),
            "system_prompt": system_prompt.strip(),
            "loss_objective_config": _json_object(
                loss_objective_config, "--loss-objective-config",
            ),
            "inference_config": _json_object(inference_config, "--inference-config"),
            "decoding_config": _json_object(decoding_config, "--decoding-config"),
        }
        decision_pin_overrides = {
            key: value for key, value in decision_pin_overrides.items()
            if value not in ("", {})
        }
        from zevo.contracts.training_methods import method_config_errors
        config_errors = method_config_errors(
            training_method, method_config, require_dependencies=True,
        )
        if config_errors:
            console.print(f"[red]{'; '.join(config_errors)}[/]")
            raise typer.Exit(1)

        # The catalogue is the tasks TABLE, the same rows the Tasks page lists.
        # It used to be fixtures/user_requests.py, so a task added in the UI was
        # invisible here and `--task <it>` demanded an objective.
        task_row = await _task_by_name(task)
        predefined = task_row is not None
        if not predefined and not metric:
            console.print("[red]--metric is required for a custom task.[/]")
            raise typer.Exit(1)
        elif not predefined and not metric_direction:
            console.print(
                "[red]--target is required for a custom task.[/] "
                "Choose Max when higher scores are better, or Min when lower scores are better."
            )
            raise typer.Exit(1)

        overrides = {
            "dataset": _resolve_data_ref(dataset),
            "dataset_split": dataset_split.strip(),
            "dataset_config": dataset_config.strip(),
            "data_query": data_query.strip(),
            "model_query": model_query.strip(),
            "method_query": method_query.strip(),
            "test_set": _resolve_data_ref(test_set),
            "test_answer_fields": _columns(answer_fields),
            "validation_set": _resolve_data_ref(validation_set),
            "validation_split": validation_split.strip(),
            "validation_config": validation_config.strip(),
            "validation_answer_fields": _columns(validation_answer_fields),
            "validation_sample_submission": _resolve_data_ref(validation_sample_submission),
            "test_sample_submission": _resolve_data_ref(test_sample_submission),
            "metric": metric,
            "metric_direction": metric_direction,
            "metric_type": metric_type,
            "evaluation_script": _resolve_data_ref(evaluation_script),
            "validation_metric_type": validation_metric_type,
            "validation_metric": validation_metric,
            "validation_metric_direction": validation_metric_direction,
            "validation_evaluation_script": _resolve_data_ref(validation_evaluation_script),
            "base_model": base_model,
            "training_method": training_method,
            "method_config": method_config,
            **decision_pin_overrides,
        }
        overrides = {k: v for k, v in overrides.items() if v}

        selected_setting = None
        if setting_ref:
            if not predefined:
                console.print("[red]--setting requires an existing Task.[/]")
                raise typer.Exit(1)
            from zevo.db import TaskSetting
            Session = get_session_factory()
            async with Session() as session:
                candidates = (await session.execute(
                    select(TaskSetting).where(TaskSetting.task_name == task)
                )).scalars().all()
            selected_setting = next(
                (s for s in candidates if s.id == setting_ref or s.name == setting_ref), None,
            )
            if selected_setting is None:
                console.print(f"[red]no setting {setting_ref!r} on Task {task!r}.[/]")
                raise typer.Exit(1)
            run_opts["setting_id"] = selected_setting.id

        if predefined:
            # ---- A catalogue task. Naming it is enough; any flag given on top
            # adjusts THIS run without editing the task, which is what the
            # Launch-run dialog does too. ----
            if objective:
                console.print(f"[yellow]{task!r} is in the catalogue; ignoring the OBJECTIVE argument.[/]")
            from zevo.api.routers.ui.tasks import task_to_user_request
            ur = task_to_user_request(task_row).model_dump()
            if attachments_local:
                uploaded_paths: list[str] = []
                for local in attachments_local:
                    p = Path(local).expanduser()
                    if not p.is_file():
                        console.print(f"[red]attachment not found:[/] {p}")
                        raise typer.Exit(1)
                    with p.open("rb") as fh:
                        uploaded = await client.post(
                            f"{api_base}/api/attachments",
                            files={"file": (p.name, fh)},
                        )
                    if uploaded.status_code >= 400:
                        console.print(f"[red]upload failed:[/] {uploaded.status_code} {uploaded.text}")
                        raise typer.Exit(1)
                    uploaded_paths.append(uploaded.json()["path"])
                ur["dataset"] = uploaded_paths[0]
            if selected_setting is not None:
                setting_pins = {
                    "prompt_framing": selected_setting.prompt_framing or "",
                    "system_prompt": selected_setting.system_prompt or "",
                    "loss_objective_config": dict(selected_setting.loss_objective_config or {}),
                    "inference_config": dict(selected_setting.inference_config or {}),
                    "decoding_config": dict(selected_setting.decoding_config or {}),
                }
                if mode == "full_pipeline" and any(
                    value not in ("", {}) for value in setting_pins.values()
                ):
                    console.print(
                        "[red]This Setting pins detailed Specialist choices and "
                        "can only be launched with --mode customized_pipeline.[/]"
                    )
                    raise typer.Exit(1)
                ur.update({
                    "dataset": selected_setting.dataset or "",
                    "dataset_split": selected_setting.dataset_split or "",
                    "dataset_config": selected_setting.dataset_config or "",
                    "data_query": selected_setting.data_query or "",
                    "model_query": selected_setting.model_query or "",
                    "method_query": selected_setting.method_query or "",
                    "validation_set": selected_setting.validation_set or "",
                    "validation_split": selected_setting.validation_split or "",
                    "validation_config": selected_setting.validation_config or "",
                    "validation_answer_fields": list(selected_setting.validation_answer_fields or []),
                    "validation_sample_submission": selected_setting.validation_sample_submission or "",
                    "validation_metric_type": selected_setting.validation_metric_type,
                    "validation_metric": selected_setting.validation_metric,
                    "validation_metric_direction": selected_setting.validation_metric_direction,
                    "validation_evaluation_script": selected_setting.validation_evaluation_script or "",
                    "validation_evaluator_sha256": selected_setting.validation_evaluator_sha256 or "",
                    "base_model": selected_setting.base_model or "",
                    "training_method": selected_setting.training_method or "",
                    "method_config": dict(selected_setting.method_config or {}),
                    **setting_pins,
                })
                if iterations is None:
                    run_opts["iteration_budget"] = selected_setting.iteration_budget
                if max_cost is None:
                    run_opts["max_cost_usd"] = selected_setting.max_cost_usd
                if stop_threshold is None:
                    run_opts["stop_threshold"] = selected_setting.stop_threshold
                # A Setting id is immutable provenance. The Launch UI clears
                # the selected id as soon as any Setting-owned value changes;
                # mirror that behavior for CLI launches.
                setting_was_edited = bool(
                    overrides
                    or attachments_local
                    or iterations is not None
                    or max_cost is not None
                    or stop_threshold is not None
                )
                if setting_was_edited:
                    run_opts.pop("setting_id", None)
                    console.print(
                        "[dim]selected Setting was edited; launching the adjusted "
                        "configuration without its Setting id.[/]"
                    )
            if overrides or attachments_local:
                if attachments_local:
                    console.print("[yellow]--attach with a catalogue task overrides its training data.[/]")
                ur.update(overrides)
                console.print(f"[dim]overriding:[/] {', '.join(sorted(overrides))}")
            body: dict = {"task_name": task, "run_name": run_name, "user_request": ur, **run_opts}
        else:
            # ---- Custom task: build a UserRequest from objective + per-field flags. ----
            if not objective.strip():
                console.print(f"[red]{task!r} is not a predefined task,[/] so it needs an OBJECTIVE "
                              "describing it. Run `task list` to see what is there.")
                raise typer.Exit(1)
            uploaded_paths: list[str] = []
            for local in attachments_local:
                p = Path(local).expanduser()
                if not p.exists():
                    console.print(f"[red]attachment not found:[/] {p}")
                    raise typer.Exit(1)
                with p.open("rb") as fh:
                    r = await client.post(
                        f"{api_base}/api/attachments",
                        files={"file": (p.name, fh)},
                    )
                if r.status_code >= 400:
                    console.print(f"[red]upload failed:[/] {r.status_code} {r.text}")
                    raise typer.Exit(1)
                j = r.json()
                uploaded_paths.append(j["path"])
                console.print(f"[dim]uploaded[/] {p.name} -> [dim]{j['path']}[/] ({j['size_bytes']} B)")

            ur = dict(_UR_DEFAULTS)
            ur["task_objective"] = objective
            ur["metric"] = metric
            ur["metric_direction"] = metric_direction
            ur["data_query"] = objective  # let Data derive from the goal if no dataset
            if uploaded_paths:
                ur["dataset"] = uploaded_paths[0]
            ur.update(overrides)
            # Custom tasks carry their name too, so the run is never "(unnamed)".
            body = {"task_name": task, "run_name": run_name, "user_request": ur, **run_opts}

        preflight_body = {
            "user_request": body["user_request"],
            "gpu_provider": body.get("gpu_provider"),
            "num_gpus": body.get("num_gpus"),
            "generation_backend": body.get("generation_backend"),
            "customizations": body.get("customizations"),
        }
        pf = await client.post(f"{api_base}/api/preflight", json=preflight_body)
        if pf.status_code >= 400:
            console.print(f"[red]preflight failed:[/] {pf.status_code} {pf.text}")
            raise typer.Exit(1)
        verdict = pf.json()
        if verdict.get("status") == "blocked":
            console.print(f"[red]preflight blocked:[/] {verdict.get('summary', '')}")
            for item in verdict.get("items", []):
                if item.get("severity") == "blocker":
                    console.print(f"  [red]•[/] {item.get('message', '')}")
            raise typer.Exit(1)
        if verdict.get("status") == "risky" and not allow_risky:
            console.print(f"[yellow]preflight warnings:[/] {verdict.get('summary', '')}")
            console.print("[dim]Review them in the UI or rerun with --allow-risky.[/]")
            raise typer.Exit(1)

        label = f"task: {task}" if predefined else f"custom task: {task}"
        # `repeat` is the one run-level knob no task carries: the fixtures module
        # this used to read had a per-package repeat, but no package ever set it,
        # so every task resolved to 1. Ask for more with --repeat.
        n = max(1, repeat if repeat is not None else 1)
        run_ids: list[str] = []
        for i in range(n):
            r = await client.post(f"{api_base}/api/runs", json=body)
            if r.status_code >= 400:
                console.print(f"[red]run create failed{f' (#{i+1})' if n > 1 else ''}:[/] {r.status_code} {r.text}")
                raise typer.Exit(1)
            out = r.json()
            rid = out.get("run_id") or out.get("id")
            run_ids.append(rid)
            if started is not None:
                started.append(rid)   # surfaced to the caller so Ctrl-C can cancel it
            tag = f" — {i + 1}/{n}" if n > 1 else ""
            console.print(f"[green]started run[/] [bold]{rid}[/] [dim]({label}{tag})[/]")
        run_id = run_ids[-1]

    if n > 1:
        console.print(
            f"[dim]launched {n} runs; the leaderboard aggregates their success ratio. "
            f"watch any with[/] run watch <id>"
        )
        return
    console.print(f"[dim]watch in the UI:[/] {_UI_BASE}/runs/{run_id}")
    if not watch:
        console.print(f"[dim]tail later with[/] run watch {run_id}")
        return
    await _stream_run_timeline(run_id, api_base=api_base)


# ----------- ticket commands ---------------------------------------------

@ticket_app.command("list")
def ticket_list(
    run_id: str = typer.Option("", "--run", help="Filter by run id."),
    status: str = typer.Option("", "--status", help="Filter by status."),
) -> None:
    """Tickets, newest first — each step of a run (UI: Tickets)."""
    asyncio.run(_ticket_list(run_id, status))


async def _ticket_list(run_id: str, status: str) -> None:
    Session = get_session_factory()
    async with Session() as s:
        q = select(Ticket).order_by(desc(Ticket.created_at))
        if run_id:
            q = q.where(Ticket.run_id == run_id)
        if status:
            q = q.where(Ticket.status == status)
        rows = (await s.execute(q.limit(100))).scalars().all()
        # Keep the canonical Agent id here; Ticket has no separate stage field.
        t = Table(box=None, pad_edge=False)
        t.add_column("id", style="cyan", no_wrap=True)
        t.add_column("run", no_wrap=True)
        t.add_column("agent", no_wrap=True)
        t.add_column("status", no_wrap=True)
        t.add_column("created", no_wrap=True)
        t.add_column("summary", max_width=48, overflow="ellipsis", no_wrap=True)
        for r in rows:
            t.add_row(r.id, r.run_id[:8], r.agent_id, r.status,
                      str(r.created_at)[:19], (r.summary or "").strip())
        console.print(t)


@ticket_app.command("show")
def ticket_show(ticket_id: str) -> None:
    """One ticket: its contract, messages, and produced artifacts."""
    asyncio.run(_ticket_show(ticket_id))


async def _ticket_show(ticket_id: str) -> None:
    Session = get_session_factory()
    async with Session() as s:
        tk = (await s.execute(select(Ticket).where(Ticket.id == ticket_id))).scalar_one_or_none()
        if tk is None:
            console.print(f"[red]no ticket {ticket_id}[/]")
            raise typer.Exit(1)
        console.print(f"[bold]Ticket[/] {tk.id}")
        console.print(f"  agent:     {tk.agent_id}")
        console.print(f"  status:    {tk.status}")
        console.print(f"  format:    {tk.input_format}")
        console.print(f"  lane:      {tk.lane}")
        console.print(f"  iteration: {tk.iteration}")
        console.print(f"  summary:   {tk.summary}")
        console.print("  payload:")
        console.print(json.dumps(tk.payload, indent=2))
        console.print("  customization:")
        console.print(json.dumps(tk.customization, indent=2))
        console.print("  inputs:")
        console.print(json.dumps(tk.inputs, indent=2))

        wps = (await s.execute(select(WorkProduct).where(WorkProduct.ticket_id == tk.id))).scalars().all()
        if wps:
            console.rule("work products")
            for wp in wps:
                console.print(f"  [{wp.role}] {wp.path}")
        messages = (await s.execute(
            select(TicketMessage).where(TicketMessage.ticket_id == tk.id).order_by(TicketMessage.created_at)
        )).scalars().all()
        if messages:
            console.rule("messages")
            for c in messages:
                console.print(f"[dim]{c.created_at}[/] [bold]{c.author}[/]: {c.body[:400]}")
        events = (await s.execute(
            select(ExecutionEvent).where(ExecutionEvent.ticket_id == tk.id).order_by(ExecutionEvent.ts)
        )).scalars().all()
        if events:
            console.rule("execution events")
            for p in events:
                console.print(f"  {p.ts} {p.phase} step={p.current_step}/{p.total_steps} loss={p.loss}")


@ticket_app.command("cancel")
def ticket_cancel(
    ticket_id: str = typer.Argument(...),
    api_base: str = typer.Option(
        os.environ.get("ZEVO_API_BASE", "http://localhost:8001"), "--api-base",
    ),
) -> None:
    """Cancel a running ticket: kills the agent driver subprocess and marks it cancelled."""
    asyncio.run(_ticket_cancel(ticket_id, api_base))


async def _ticket_cancel(ticket_id: str, api_base: str) -> None:
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(f"{api_base}/api/tickets/{ticket_id}/cancel")
        if r.status_code >= 400:
            console.print(f"[red]cancel failed:[/] {r.status_code} {r.text}")
            raise typer.Exit(1)
        d = r.json()
        if d.get("process_killed"):
            console.print(f"[green]cancelled[/] ticket {ticket_id} — process killed")
        elif d.get("status") in (
            "succeeded", "degraded", "failed", "skipped", "cancelled",
        ):
            console.print(f"[yellow]ticket {ticket_id} is already {d['status']}[/]")
        else:
            console.print(f"[green]marked cancelled[/] ticket {ticket_id} (process may not be in this host)")


# ----------- model commands ----------------------------------------------

@app.command("models")
def models() -> None:
    """Every model version Zevo has registered (UI: the Models page)."""
    asyncio.run(_registry_list())


async def _registry_list() -> None:
    rows = await _api_get("/api/models")
    if rows is None:
        raise typer.Exit(1)
    t = Table(box=None, pad_edge=False)
    t.add_column("tag")
    t.add_column("iteration", justify="right")
    t.add_column("base model")
    t.add_column("method")
    t.add_column("test score")
    t.add_column("improvement")
    for row in rows:
        test = row.get("champion_test_score")
        gain = row.get("improvement")
        t.add_row(
            row.get("version_tag", ""), str(row.get("iteration", 0)),
            row.get("base_model", ""), row.get("training_method", ""),
            _score_text(test, row.get("metric")),
            _score_text(gain, row.get("metric"), signed=True),
        )
    console.print(t)


@model_app.command("show")
def model_show(version_tag: str = typer.Argument(...)) -> None:
    """Show the same model facts as the Models detail panel."""
    row = asyncio.run(_api_get(f"/api/models/{version_tag}"))
    if row is None:
        raise typer.Exit(1)
    for label, key in (
        ("Tag", "version_tag"), ("Run", "run_id"), ("Task", "task_name"),
        ("Iteration", "iteration"), ("Base Model", "base_model"),
        ("Training Method", "training_method"), ("Dataset", "dataset_source"),
        ("Model Path", "model_path_abs"), ("Test Score", "champion_test_score"),
        ("Baseline Test Score", "baseline_test_score"),
        ("Improvement", "improvement"),
        ("Registered", "registered_at"),
    ):
        console.print(f"[dim]{label}:[/] {row.get(key, '')}")
    console.print_json(data=row.get("eval") or {})


@model_app.command("card")
def model_card(version_tag: str = typer.Argument(...)) -> None:
    """Print the generated model card."""
    async def go() -> None:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{_api_base()}/api/models/{version_tag}/card")
        if response.status_code >= 400:
            console.print(f"[red]{response.status_code}[/] {response.text}")
            raise typer.Exit(1)
        console.print(response.text)
    asyncio.run(go())


@model_app.command("compare")
def model_compare(a: str = typer.Argument(...), b: str = typer.Argument(...)) -> None:
    """Compare two model tags using the Models API."""
    from urllib.parse import quote
    result = asyncio.run(_api_get(
        f"/api/models/compare?a={quote(a)}&b={quote(b)}"
    ))
    if result is None:
        raise typer.Exit(1)
    console.print(f"[bold]{result.get('headline_summary', '')}[/]")
    table = Table(box=None, pad_edge=False)
    table.add_column("field")
    table.add_column(a)
    table.add_column(b)
    for field, cell in (result.get("fields") or {}).items():
        if cell.get("differs"):
            table.add_row(field, str(cell.get("a", "")), str(cell.get("b", "")))
    console.print(table)


# ----------- admin -------------------------------------------------------

@app.command("seed-agents")
def seed_agents_cmd() -> None:
    """Sync agents table with playbook/agents/<id>/identity.md (idempotent)."""
    Session = get_session_factory()

    async def go() -> None:
        async with Session() as s:
            n = await seed_agents(s)
            console.print(f"[green]seeded {n} agents[/]")
    asyncio.run(go())




@run_app.command("list")
def run_list(
    show_all: bool = typer.Option(
        False, "--all", "-a",
        help="Include finished runs too (default: only active/running ones)."),
    limit: int = typer.Option(40, "--limit"),
    api_base: str = typer.Option(
        os.environ.get("ZEVO_API_BASE", "http://localhost:8001"), "--api-base"),
) -> None:
    """List runs — only the active (still-running) ones by default; `--all` for every run."""
    asyncio.run(_run_list(show_all=show_all, limit=limit, api_base=api_base.rstrip("/")))


async def _run_list(*, show_all: bool, limit: int, api_base: str) -> None:
    import httpx
    from rich.table import Table

    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{api_base}/api/runs", params={"limit": limit})
    if r.status_code >= 400:
        console.print(f"[red]failed to list runs:[/] {r.status_code} {r.text}")
        raise typer.Exit(1)
    runs = r.json()
    if not show_all:
        runs = [x for x in runs if str(x.get("status", "")) not in TERMINAL_RUN_STATUSES]

    table = Table(box=None, pad_edge=False)
    table.add_column("run id", style="cyan", no_wrap=True)
    table.add_column("status")
    table.add_column("task")
    table.add_column("iters", justify="right")
    table.add_column("test score", justify="right")
    table.add_column("improvement", justify="right")
    table.add_column("started")
    color = {"running": "yellow", "success": "green", "failed": "red",
             "cancelled": "dim", "halted": "red"}
    for x in runs:
        st = str(x.get("status", ""))
        # The same paired held-out outcomes as Runs in the web UI.
        score = x.get("champion_test_score")
        score_s = _score_text(score, x.get("metric"))
        baseline = x.get("baseline_test_score")
        champion = x.get("champion_test_score")
        gain = None
        if isinstance(baseline, (int, float)) and isinstance(champion, (int, float)):
            gain = (
                baseline - champion
                if x.get("metric_direction") == "min"
                else champion - baseline
            )
        gain_s = _score_text(gain, x.get("metric"), signed=True)
        iters = f"{x.get('iterations_completed', 0)}/{x.get('iteration_budget', 0)}"
        started = str(x.get("started_at") or "")[:19].replace("T", " ")
        table.add_row(
            x.get("id", ""), f"[{color.get(st, 'white')}]{st}[/]",
            x.get("task_name") or "-", iters, score_s, gain_s, started,
        )
    console.print(table)
    if not runs:
        console.print("[dim]no active runs.[/]" if not show_all else "[dim]no runs yet.[/]")
    elif not show_all:
        console.print("[dim]finished runs hidden — add [/][cyan]--all[/][dim] to see them.[/]")


async def _resolve_run_id(client, api_base: str, run_id: str) -> str:
    """Accept a full UUID or a unique prefix (the UI/`run list` show short ids).

    Exact id wins; otherwise match runs whose id starts with `run_id`. Errors
    on no match or an ambiguous prefix.
    """
    exact = await client.get(f"{api_base}/api/runs/{run_id}")
    if exact.status_code == 200:
        return run_id
    listed = await client.get(f"{api_base}/api/runs", params={"limit": 200})
    runs = listed.json() if listed.status_code == 200 else []
    matches = [str(x.get("id", "")) for x in runs if str(x.get("id", "")).startswith(run_id)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        console.print(f"[red]no run matching[/] {run_id!r}")
        raise typer.Exit(1)
    console.print(f"[red]ambiguous run id[/] {run_id!r} — matches {len(matches)}:")
    for m in matches[:10]:
        console.print(f"  {m}")
    raise typer.Exit(1)


@run_app.command("watch")
def run_watch(
    run_id: str = typer.Argument(...),
    api_base: str = typer.Option(
        os.environ.get("ZEVO_API_BASE", "http://localhost:8001"), "--api-base"),
) -> None:
    """Stream a run's unified timeline (all agents interleaved). Accepts an id prefix."""
    asyncio.run(_run_watch(run_id, api_base=api_base.rstrip("/")))


async def _run_watch(run_id: str, *, api_base: str) -> None:
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as client:
        full = await _resolve_run_id(client, api_base, run_id)
    await _stream_run_timeline(full, api_base=api_base)


@run_app.command("show")
def run_show(
    run_id: str = typer.Argument(...),
    api_base: str = typer.Option(
        os.environ.get("ZEVO_API_BASE", "http://localhost:8001"), "--api-base",
    ),
) -> None:
    """Show one Run's request, budget, scores, Tickets, and artifacts."""
    asyncio.run(_run_show(run_id, api_base=api_base.rstrip("/")))


async def _run_show(run_id: str, *, api_base: str) -> None:
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        full = await _resolve_run_id(client, api_base, run_id)

        async def get(suffix: str) -> dict | list:
            response = await client.get(f"{api_base}/api/runs/{full}{suffix}")
            if response.status_code >= 400:
                console.print(
                    f"[red]could not load {suffix or 'run'}:[/] "
                    f"{response.status_code} {response.text}"
                )
                raise typer.Exit(1)
            return response.json()

        detail = await get("")
        request = await get("/request")
        budget = await get("/budget")
        scores = await get("/scores")
        artifacts = await get("/artifacts")

    assert isinstance(detail, dict)
    assert isinstance(artifacts, list)
    console.print(f"[bold cyan]{detail.get('run_name') or full}[/]  {full}")
    facts = Table(box=None, pad_edge=False)
    facts.add_column("field", style="dim")
    facts.add_column("value")
    for label, key in (
        ("Status", "status"), ("Run Mode", "mode"), ("Task", "task_name"),
        ("Setting", "setting_name"), ("Setting ID", "setting_id"), ("Metric", "metric"),
        ("Target", "metric_direction"),
        ("Iterations Completed", "iterations_completed"),
        ("Iteration Budget", "iteration_budget"),
        ("Best Validation Score", "best_validation_score"),
        ("Champion Test Score", "champion_test_score"),
        ("Model Tag", "registry_version_tag"), ("Started", "started_at"),
        ("Queue Wait", "queue_wait_seconds"),
        ("Finished", "finished_at"),
    ):
        value = detail.get(key, "")
        if key == "metric_direction":
            value = str(value).capitalize()
        facts.add_row(label, str(value))
    console.print(facts)
    console.rule("request")
    console.print_json(data=request)
    console.rule("budget")
    console.print_json(data=budget)
    console.rule("scores")
    console.print_json(data=scores)
    console.rule("governance")
    console.print_json(data={
        "decision_pins": detail.get("decision_pins", {}),
        "model_lineages": detail.get("model_lineages", {}),
    })
    console.rule("tickets")
    for ticket in detail.get("tickets", []):
        console.print(
            f"  [cyan]{ticket.get('id', '')}[/]  "
            f"iteration={ticket.get('iteration', 0)}  "
            f"lane={ticket.get('lane', '')}  status={ticket.get('status', '')}"
        )
    console.rule("artifacts")
    for artifact in artifacts:
        console.print(
            f"  [cyan]{artifact.get('id', '')}[/]  "
            f"[{artifact.get('role', '')}] {artifact.get('path', '')} "
            f"([dim]{artifact.get('size_bytes', 0)} B[/])"
        )


@run_app.command("artifact")
def run_artifact(
    run_id: str = typer.Argument(..., help="Run id (a unique prefix is enough)."),
    artifact_id: str = typer.Argument(..., help="Artifact id shown by `run show`."),
    api_base: str = typer.Option(
        os.environ.get("ZEVO_API_BASE", "http://localhost:8001"), "--api-base",
    ),
) -> None:
    """Show one artifact's metadata and the same safe preview as the Run page."""
    asyncio.run(_run_artifact(run_id, artifact_id, api_base=api_base.rstrip("/")))


async def _run_artifact(run_id: str, artifact_id: str, *, api_base: str) -> None:
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        full = await _resolve_run_id(client, api_base, run_id)
        response = await client.get(
            f"{api_base}/api/runs/{full}/artifacts/{artifact_id}"
        )
    if response.status_code >= 400:
        console.print(
            f"[red]could not load artifact:[/] {response.status_code} {response.text}"
        )
        raise typer.Exit(1)
    artifact = response.json()
    facts = Table(box=None, pad_edge=False)
    facts.add_column("field", style="dim")
    facts.add_column("value")
    for key in (
        "id", "ticket_id", "role", "path", "local_path", "exists",
        "size_bytes", "created_at", "preview_kind",
    ):
        facts.add_row(key, str(artifact.get(key, "")))
    console.print(facts)
    if artifact.get("meta"):
        console.rule("metadata")
        console.print_json(data=artifact["meta"])
    console.rule("preview")
    console.print(escape(str(artifact.get("preview", ""))))


@run_app.command("cancel")
def run_cancel(
    run_id: str = typer.Argument(...),
    api_base: str = typer.Option(
        os.environ.get("ZEVO_API_BASE", "http://localhost:8001"), "--api-base"),
) -> None:
    """Cancel a running pipeline: marks the run + every in-flight ticket
    cancelled, and the scheduler kills the agent driver subprocesses within ~3s.
    Accepts a full run id or a unique prefix (as shown by `run list`)."""
    asyncio.run(_run_cancel(run_id, api_base=api_base.rstrip("/")))


async def _run_cancel(run_id: str, *, api_base: str) -> None:
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as client:
        run_id = await _resolve_run_id(client, api_base, run_id)
        r = await client.post(f"{api_base}/api/runs/{run_id}/cancel")
        if r.status_code >= 400:
            console.print(f"[red]cancel failed:[/] {r.status_code} {r.text}")
            raise typer.Exit(1)
        d = r.json()
        if d.get("note") == "already terminal":
            console.print(f"[yellow]run {run_id} is already {d['status']}[/]")
            return
        n = len(d.get("tickets_cancelled") or [])
        console.print(
            f"[green]cancelled run[/] [bold]{run_id}[/] "
            f"-- {n} ticket(s) flipped to cancelled; "
            f"scheduler will SIGTERM the agent driver subprocesses within ~3s."
        )


async def _stream_run_timeline(run_id: str, *, api_base: str) -> None:
    """Open the run-timeline WebSocket and pretty-print events."""
    import websockets
    ws_url = api_base.replace("http://", "ws://").replace("https://", "wss://")
    url = f"{ws_url}/api/ws/runs/{run_id}/timeline"
    console.print(f"[dim]streaming run timeline for {run_id}...[/]")
    try:
        async with websockets.connect(url, max_size=None) as ws:
            async for raw in ws:
                try:
                    ev = json.loads(raw)
                except Exception:
                    continue
                _print_event(ev, show_agent=True)
                if ev.get("type") == "run_finished":
                    return
    except Exception as e:
        console.print(f"[yellow]WS dropped: {e}; falling back to REST polling[/]")
        await _stream_run_timeline_poll(run_id, api_base=api_base)


async def _stream_run_timeline_poll(run_id: str, *, api_base: str) -> None:
    """REST fallback: poll /runs/{id}/timeline?since_ts=ISO."""
    import httpx
    since = ""
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            params = {"since_ts": since} if since else {}
            r = await client.get(f"{api_base}/api/runs/{run_id}/timeline", params=params)
            if r.status_code >= 400:
                console.print(f"[red]poll error: {r.status_code}[/]")
                return
            rows = r.json()
            for ev in rows:
                _print_event(ev, show_agent=True)
                since = ev.get("ts") or since
                if ev.get("type") == "run_finished":
                    return
            await asyncio.sleep(1.5)


async def _wait_then_stream(*, ticket_id: str, api_base: str) -> None:
    """Poll for the next heartbeat on this ticket, then WS-tail it."""
    import httpx
    console.print(f"[dim]waiting for daemon to spawn a heartbeat for {ticket_id}...[/]")
    deadline_s = 180  # 3 min cap; user can Ctrl-C anytime
    interval = 1.0
    waited = 0.0
    hb_id = ""
    async with httpx.AsyncClient(timeout=10.0) as client:
        while waited < deadline_s:
            r = await client.get(
                f"{api_base}/api/heartbeats",
                params={"ticket_id": ticket_id, "limit": 1},
            )
            if r.status_code < 400:
                rows = r.json()
                # Prefer the most recently started, and only proceed if it
                # actually exists (some installs return [] for a moment).
                if rows:
                    hb_id = rows[0]["id"]
                    break
            await asyncio.sleep(interval)
            waited += interval
    if not hb_id:
        console.print(
            "[yellow]no heartbeat appeared yet. Check `heartbeats list` "
            "in a moment.[/]"
        )
        return
    console.print(f"[dim]heartbeat[/] [bold]{hb_id[:8]}[/] [dim]-- streaming...[/]")
    await _stream_heartbeat_ws(hb_id, api_base=api_base)


async def _stream_heartbeat_ws(heartbeat_id: str, *, api_base: str) -> None:
    """Open the WebSocket and pretty-print events as they arrive."""
    import websockets
    # Convert http(s) -> ws(s).
    ws_url = api_base.replace("http://", "ws://").replace("https://", "wss://")
    url = f"{ws_url}/api/ws/heartbeats/{heartbeat_id}"
    try:
        async with websockets.connect(url, max_size=None) as ws:
            async for raw in ws:
                try:
                    ev = json.loads(raw)
                except Exception:
                    continue
                _print_event(ev)
                if ev.get("type") == "finished":
                    return
    except Exception as e:
        console.print(f"[yellow]WS dropped: {e}; falling back to REST polling[/]")
        await _stream_heartbeat_poll(heartbeat_id, api_base=api_base)


async def _stream_heartbeat_poll(heartbeat_id: str, *, api_base: str) -> None:
    """Fallback when WS is unavailable: REST poll /events?since_seq=N."""
    import httpx
    since = 0
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            r = await client.get(
                f"{api_base}/api/heartbeats/{heartbeat_id}/events",
                params={"since_seq": since},
            )
            if r.status_code >= 400:
                console.print(f"[red]poll error: {r.status_code}[/]")
                return
            rows = r.json()
            for ev in rows:
                _print_event(ev)
                since = max(since, ev.get("seq", since))
                if ev.get("type") == "finished":
                    return
            await asyncio.sleep(1.0)


def _print_event(ev: dict, show_agent: bool = False) -> None:
    """Render one transcript event to the terminal with rich styling.

    When show_agent is True (run-level timeline), prefix each line with
    the agent id so interleaved multi-agent output is readable.
    """
    ev_type = str(ev.get("type") or "raw")
    payload = ev.get("payload") or {}
    ts = str(ev.get("ts") or "")[11:19]  # HH:MM:SS
    prefix = f"[dim]{ts}[/]"
    if show_agent:
        agent = ev.get("agent_id") or ""
        if agent:
            prefix = f"[dim]{ts}[/] [cyan]{agent}[/]"

    # Run-level timeline event types (only appear with show_agent).
    if ev_type == "ticket_created":
        console.print(
            f"{prefix} [blue]📋 created[/] {ev.get('ticket_id','')} · "
            f"{ev.get('agent_id', '')} · {payload.get('lane', '')}"
        )
        return
    if ev_type == "ticket_status":
        st = payload.get("status", "")
        color = {
            "succeeded": "green", "degraded": "magenta", "failed": "red",
            "cancelled": "red", "skipped": "dim", "running": "yellow",
            "awaiting_input": "cyan", "waiting_external": "yellow", "queued": "blue",
        }.get(st, "dim")
        console.print(f"{prefix} [{color}]→ {ev.get('ticket_id','')}: {st}[/]")
        return
    if ev_type == "message":
        # escape() — agent text can contain '[...]' (e.g. a remote path
        # /orange/.../zevo) that Rich would mis-parse as a markup tag and crash.
        body = escape(str(payload.get('body', ''))[:160])
        console.print(f"{prefix} [dim]💬 {escape(str(payload.get('author','')))}: {body}[/]")
        return
    if ev_type == "heartbeat_finished":
        ec = payload.get("exit_code", -1)
        color = "green" if ec == 0 else "red"
        console.print(f"{prefix} [{color}]■ heartbeat finished[/] exit={ec}")
        return
    if ev_type == "run_finished":
        st = payload.get("status", "")
        color = "green" if st == "success" else "red"
        console.print(f"{prefix} [{color}]◆ run {st}[/] {escape(str(payload.get('halted_reason','')))}")
        return
    if ev_type == "cancelled":
        console.print(f"{prefix} [red]⊘ {escape(str(payload.get('message','cancelled')))}[/]")
        return

    if ev_type == "heartbeat_started":
        console.print(
            f"{prefix} [green]●[/] heartbeat started "
            f"agent={payload.get('agent_id')} "
            f"driver={payload.get('driver')} model={payload.get('model')}"
        )
    elif ev_type == "agent_message":
        msg = payload.get("message") or payload.get("text") or payload.get("content") or ""
        if msg:
            console.print(f"{prefix} {escape(str(msg))}")
    elif ev_type in ("agent_reasoning", "reasoning"):
        msg = payload.get("text") or payload.get("message") or ""
        if msg:
            console.print(f"{prefix} [italic dim]{escape(str(msg))}[/]")
    elif ev_type == "tool_call":
        tool = payload.get("tool") or payload.get("name") or "tool"
        cmd = payload.get("command") or payload.get("args") or payload.get("input") or ""
        cmd_s = cmd if isinstance(cmd, str) else json.dumps(cmd)
        console.print(f"{prefix} [yellow]⚙ {escape(str(tool))}[/] [dim]{escape(cmd_s[:200])}[/]")
    elif ev_type == "tool_result":
        out = payload.get("output") or payload.get("stdout") or ""
        ec = payload.get("exit_code")
        tag = f" exit={ec}" if ec is not None else ""
        snippet = out.strip().splitlines()[:3] if isinstance(out, str) else []
        body = "\n".join(snippet)[:280]
        console.print(f"{prefix} [dim]↳ tool result{tag}[/]")
        if body:
            console.print(f"   [dim]{escape(body)}[/]")
    elif ev_type == "phase":
        console.print(
            f"{prefix} [magenta]▸ phase:[/] [bold]{payload.get('phase','')}[/]"
        )
    elif ev_type == "progress":
        step = payload.get("step")
        total = payload.get("total")
        loss = payload.get("loss")
        bits = []
        if step is not None: bits.append(f"step {step}" + (f"/{total}" if total else ""))
        if loss is not None and loss >= 0: bits.append(f"loss {float(loss):.4f}")
        console.print(f"{prefix} [cyan]·[/] " + "  ".join(bits))
    elif ev_type == "stderr":
        t = payload.get("text") or ""
        if t.strip():
            console.print(f"{prefix} [red]{t.rstrip()}[/]")
    elif ev_type == "finished":
        ec = payload.get("exit_code", -1)
        err = payload.get("error_message") or ""
        color = "green" if ec == 0 else "red"
        console.print(f"{prefix} [{color}]■ heartbeat finished[/] exit={ec} {err}")
    elif ev_type == "heartbeat_alive":
        msg = payload.get("message") or "still running…"
        secs = payload.get("silent_seconds")
        extra = f" ({int(secs)}s)" if secs else ""
        console.print(f"{prefix} [dim yellow]⟳ {msg}{extra}[/]")
    elif ev_type == "raw":
        t = payload.get("text") or ""
        if t.strip():
            console.print(f"{prefix} [dim]{t}[/]")
    else:
        # Unknown event type -- show concise summary.
        keys = ", ".join(list(payload.keys())[:4])
        console.print(f"{prefix} [dim]ⓘ {ev_type}[/] [dim]({keys})[/]")


@agent_app.command("set")
def agent_set(
    agent_id: str = typer.Argument(...),
    provider: str = typer.Option("", "--provider", "--driver",
        help="Driver to use: claude_cli | codex_cli | bedrock | openrouter."),
    model: str = typer.Option("", "--model",
        help="Model id to pass to the driver (driver-specific format). "
             "Examples: claude-opus-5, claude-sonnet-5."),
    sandbox: Optional[str] = typer.Option(
        None, "--sandbox",
        help="Execution terminal: none or openshell. OpenShell is available only for the claude_cli orchestrator.",
    ),
    reset: bool = typer.Option(
        False, "--reset",
        help="Clear DB overrides and fall back to identity.md frontmatter on the next heartbeat.",
    ),
    api_base: str = typer.Option(
        os.environ.get("ZEVO_API_BASE", "http://localhost:8001"), "--api-base"),
) -> None:
    """Change an Agent's runtime driver, model, or sandbox.

    Takes effect on the agent's NEXT invocation (no restart needed).
    The change is persisted in the agents DB row; the seeder respects
    user-set values across container restarts.

    Use `all` as agent_id to apply the same config to every LLM agent. The
    deterministic Evaluation runner is excluded.
    """
    asyncio.run(_agent_set(
        agent_id, provider, model, sandbox, reset, api_base.rstrip("/"),
    ))


async def _agent_set(
    agent_id: str, provider: str, model: str, sandbox: Optional[str],
    reset: bool, api_base: str,
) -> None:
    import httpx
    if reset and (provider or model or sandbox is not None):
        console.print("[red]--reset cannot be combined with --provider/--driver, --model, or --sandbox[/]")
        raise typer.Exit(1)
    body: dict = {"default_driver": "", "default_model": ""} if reset else {}
    if provider:
        body["default_driver"] = provider
    if model:
        body["default_model"] = model
    if sandbox is not None:
        value = sandbox.strip().lower()
        if value not in ("none", "openshell"):
            console.print("[red]--sandbox must be none or openshell.[/]")
            raise typer.Exit(1)
        body["sandbox"] = value
    if not body:
        console.print("[yellow]nothing to update -- pass --provider/--driver, --model, --sandbox, or --reset[/]")
        raise typer.Exit(1)

    async with httpx.AsyncClient(timeout=10.0) as client:
        targets = [agent_id]
        if agent_id in ("all", "*"):
            r = await client.get(f"{api_base}/api/agents")
            if r.status_code >= 400:
                console.print(f"[red]agent list failed:[/] {r.status_code} {r.text}")
                raise typer.Exit(1)
            targets = [str(a["id"]) for a in r.json() if a.get("id") != "evaluation"]
        for target in targets:
            r = await client.patch(f"{api_base}/api/agents/{target}", json=body)
            if r.status_code >= 400:
                console.print(f"[red]update failed for {target}:[/] {r.status_code} {r.text}")
                raise typer.Exit(1)
            d = r.json()
            shown_driver = d.get("default_driver", "") or "(identity.md)"
            shown_model = d.get("default_model", "") or "(identity.md)"
            console.print(
                f"[green]updated[/] {target}: "
                f"provider=[bold]{shown_driver}[/] "
                f"model=[bold]{shown_model}[/] "
                f"sandbox=[bold]{d.get('sandbox', 'none')}[/]"
            )


@agent_app.command("drivers")
def agent_drivers() -> None:
    """Every driver with its readiness — the same credential check the UI shows.

    Readiness comes from the auth-status endpoint's own function rather than a
    second copy of the rules here. The list used to be hand-written, which is
    why it went on claiming two drivers after codex_cli and openrouter shipped.
    """
    import shutil
    from rich.table import Table as _Table
    from zevo.api.routers.ui.auth_status import get_auth_status

    # What each driver actually talks to, and — for the two that shell out —
    # the binary that has to exist. A credential alone is not readiness there:
    # without the binary the subprocess never starts.
    KIND = {
        "claude_cli": ("subprocess (`claude`)", "claude", "ZEVO_CLAUDE_BIN"),
        "codex_cli":  ("subprocess (`codex`)",  "codex",  "ZEVO_CODEX_BIN"),
        "bedrock":    ("boto3 Converse",        "",       ""),
        "openrouter": ("HTTP chat-completions", "",       ""),
    }

    status = asyncio.run(get_auth_status())
    t = _Table(box=None, pad_edge=False)
    t.add_column("name"); t.add_column("kind"); t.add_column("ready?"); t.add_column("auth")
    for d in status.drivers:
        kind, binary, bin_env = KIND.get(d.driver, ("", "", ""))
        have_bin = True
        if binary:
            have_bin = bool(os.environ.get(bin_env, "").strip()) or shutil.which(binary) is not None
        found = ", ".join(f"{c.name} {c.preview}".strip() for c in d.creds if c.present)
        notes = found or "[dim]no credential set[/dim]"
        if binary and not have_bin:
            notes = f"`{binary}` not on PATH; {notes}"
        t.add_row(d.driver, kind, "[green]yes[/]" if (d.ready and have_bin) else "[red]no[/]", notes)
    t.add_row("stub", "test-only (no LLM)", "[green]yes[/]",
              "returns canned artifacts; used by the e2e tests")
    console.print(t)


@agent_app.command("ping")
def agent_ping(
    agent_id: str = typer.Argument(...),
) -> None:
    """Send a no-ticket cron-style wakeup -- agent scans its inbox."""
    async def go() -> None:
        from zevo.engine.run.wakeup import queue_wakeup
        from zevo.db import get_session_factory
        Session = get_session_factory()
        async with Session() as s:
            w = await queue_wakeup(
                s, agent_id=agent_id, ticket_id=None,
                source="cron", reason="manual agent ping (cron tick)",
            )
            console.print(f"[green]ping queued[/] wakeup={w.id[:8]} status={w.status}")
    asyncio.run(go())

# =========================================================================
# One command group per page of the UI.
#
# The web app and the terminal show the same system, so they should offer the
# same nouns: `dashboard` is the Dashboard, `task list` is the Tasks
# page, and so on. Anything that reads the database or the catalogue directory
# works with the backend down; anything that reads live provider state goes
# through the API, and says so when it cannot reach it.
# =========================================================================


def _api_base() -> str:
    return os.environ.get("ZEVO_API_BASE", "http://localhost:8001")


async def _api_get(path: str):
    """GET from the backend, or None when it is not reachable."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{_api_base()}{path}")
        if r.status_code >= 400:
            console.print(f"[red]{r.status_code}[/red] {path}")
            return None
        return r.json()
    except Exception as e:  # noqa: BLE001 — the backend being down is normal
        console.print(f"[red]backend unreachable[/red] at {_api_base()} ({type(e).__name__})")
        return None


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


_PERCENTAGE_METRICS = {
    "accuracy", "exact_match", "em", "f1", "f1_micro", "f1_macro",
    "token_f1", "precision", "recall", "bleu", "rouge", "rouge_l",
    "pass_rate", "win_rate", "pass@1", "pass_at_1", "mc_loglikelihood",
    "accuracy_norm", "suite_average",
}


def _is_percentage_metric(metric: str | None) -> bool:
    key = (metric or "").strip().lower().replace(" ", "_").replace("-", "_")
    return (
        key in _PERCENTAGE_METRICS
        or key.endswith("_accuracy")
        or key.endswith("_f1")
        or key.endswith("_precision")
        or key.endswith("_recall")
    )


def _score_text(v, metric: str | None = None, *, signed: bool = False) -> str:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return "—"
    value = float(v)
    prefix = "+" if signed and value >= 0 else ""
    if _is_percentage_metric(metric):
        return f"{prefix}{value * 100:.1f}%"
    return f"{prefix}{value:.3f}"


# ----------- tasks -------------------------------------------------------
#
# Tasks are ROWS, not code. The shipped catalogue is frozen in migration
# b2c3d4e5f6a8, which seeds a fresh database once and is never consulted again —
# a task added in the UI and a task the product shipped are the same kind of
# thing, and both are edited in the table.


def _level_of(dataset: str, base_model: str, training_method: str) -> str:
    """Same canonical ownership ladder the API applies."""
    pins = tuple(bool((v or "").strip()) for v in (dataset, base_model, training_method))
    return {
        (True, True, True): "L1",
        (False, True, True): "L2",
        (False, True, False): "L3",
        (False, False, False): "L4",
    }.get(pins, "Custom")


def _short_ref(path: str) -> str:
    """`/app/data/files/gsm8k/train.csv` -> `gsm8k/train.csv`."""
    v = (path or "").strip()
    if not v:
        return ""
    try:
        return str(Path(v).relative_to(_FILES_DIR))
    except ValueError:
        return v[len("/app/"):] if v.startswith("/app/") else v


async def _task_rows() -> list:
    from zevo.db import Task
    Session = get_session_factory()
    async with Session() as s:
        return list((await s.execute(select(Task).order_by(Task.name))).scalars().all())


@task_app.command("list")
def task_list() -> None:
    """Every task in the catalogue (UI: Tasks)."""
    rows = asyncio.run(_task_rows())
    if not rows:
        console.print("[dim]no tasks[/dim]")
        return
    t = Table(box=None, pad_edge=False)
    t.add_column("name", style="bold cyan", no_wrap=True)
    t.add_column("metric", no_wrap=True)
    t.add_column("target", no_wrap=True)
    t.add_column("test set", max_width=28, overflow="ellipsis", no_wrap=True)
    t.add_column("task objective", max_width=32, overflow="ellipsis", no_wrap=True)
    for r in rows:
        t.add_row(
            r.name,
            r.metric,
            r.metric_direction.capitalize(),
            _short_ref(r.test_set),
            r.task_objective or "",
        )
    console.print(t)


@task_app.command("show")
def task_show(name: str = typer.Argument(..., help="Task name.")) -> None:
    """One task, field by field."""
    rows = asyncio.run(_task_rows())
    row = next((r for r in rows if r.name == name), None)
    if row is None:
        console.print(f"[red]no task {name!r}[/red] — see `task list`")
        raise typer.Exit(1)
    console.print(f"[bold]{row.name}[/bold]")
    console.print(f"\n{row.task_objective}\n")
    from zevo.api.routers.ui.tasks import _stored_test_sets

    t = Table(box=None, pad_edge=False)
    t.add_column("Test set", style="bold cyan", no_wrap=True)
    t.add_column("data")
    t.add_column("inference query", max_width=42)
    t.add_column("metric", no_wrap=True)
    t.add_column("target", no_wrap=True)
    t.add_column("answer fields")
    t.add_column("evaluator")
    t.add_column("sample submission")
    for item in _stored_test_sets(row):
        t.add_row(
            item.name,
            _short_ref(item.test_set),
            item.inference_query,
            f"{item.metric} ({item.metric_type})",
            item.metric_direction,
            ", ".join(item.answer_fields),
            _short_ref(item.evaluation_script),
            _short_ref(item.sample_submission),
        )
    console.print(t)


def _task_test_sets(raw: str) -> list[dict]:
    """Read the compact CLI suite value: JSON inline, or ``@file.json``."""
    value = (raw or "").strip()
    try:
        parsed = (
            json.loads(Path(value[1:]).expanduser().read_text(encoding="utf-8"))
            if value.startswith("@") else json.loads(value)
        )
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"[red]--test-sets must be JSON or @file.json:[/] {exc}")
        raise typer.Exit(1)
    if not isinstance(parsed, list) or not parsed:
        console.print("[red]--test-sets must decode to a non-empty JSON array.[/]")
        raise typer.Exit(1)
    suite = []
    for index, item in enumerate(parsed, 1):
        if not isinstance(item, dict):
            console.print(f"[red]Test set {index} must be a JSON object.[/]")
            raise typer.Exit(1)
        normalized = dict(item)
        normalized["test_set"] = _resolve_data_ref(str(item.get("test_set") or ""))
        normalized["sample_submission"] = _resolve_data_ref(
            str(item.get("sample_submission") or "")
        )
        if str(item.get("metric_type") or "builtin") == "custom":
            normalized["evaluation_script"] = _resolve_data_ref(
                str(item.get("evaluation_script") or "")
            )
        suite.append(normalized)
    return suite


@task_app.command("add")
def task_add(
    name: str = typer.Argument(..., help="What to call the task."),
    objective: str = typer.Argument(..., help="What the fine-tuned model should do."),
    test_sets: str = typer.Option(
        ...,
        "--test-sets",
        help=(
            "JSON array, or @file.json. Each item has name, test_set, "
            "inference_query, sample_submission, metric, metric_direction, and "
            "answer_fields. Set metric_type=custom and evaluation_script for Other."
        ),
    ),
) -> None:
    """Add a task. It appears in the UI immediately."""
    asyncio.run(_task_add(
        name=name, task_objective=objective, test_sets=test_sets,
    ))


async def _task_add(**kw) -> None:
    import httpx
    body = {
        "name": kw["name"],
        "task_objective": kw["task_objective"],
        "test_sets": _task_test_sets(kw["test_sets"]),
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{_api_base()}/api/tasks", json=body)
    if r.status_code >= 400:
        console.print(f"[red]{r.status_code}[/red] {r.text}")
        raise typer.Exit(1)
    d = r.json()
    console.print(f"[green]added[/green] {d['name']}")


@task_app.command("edit")
def task_edit(
    name: str = typer.Argument(..., help="Task name."),
    objective: Optional[str] = typer.Option(None, "--objective"),
    test_sets: Optional[str] = typer.Option(
        None, "--test-sets", help="Replacement JSON array, or @file.json.",
    ),
) -> None:
    """Edit the same Task-owned Test suite as the UI."""
    body: dict = {}
    if objective is not None: body["task_objective"] = objective
    if test_sets is not None: body["test_sets"] = _task_test_sets(test_sets)
    if not body:
        console.print("[yellow]nothing to update[/]")
        raise typer.Exit(1)
    result = asyncio.run(_api("PATCH", f"/api/tasks/{name}", json=body))
    if result is None:
        raise typer.Exit(1)
    console.print(f"[green]updated[/] {name}")


setting_app = typer.Typer(help="Manage a Task's reusable Settings.")
task_app.add_typer(setting_app, name="setting")


@setting_app.command("list")
def task_setting_list(task: str = typer.Argument(...)) -> None:
    """List saved Settings for a Task."""
    rows = asyncio.run(_api_get(f"/api/tasks/{task}/settings"))
    if rows is None:
        raise typer.Exit(1)
    table = Table(box=None, pad_edge=False)
    table.add_column("id"); table.add_column("name"); table.add_column("model")
    table.add_column("method"); table.add_column("iterations"); table.add_column("stop threshold")
    for row in rows:
        table.add_row(
            row.get("id", ""), row.get("name", ""), row.get("base_model", ""),
            row.get("training_method", ""), str(row.get("iteration_budget", 0)),
            "—" if row.get("stop_threshold") is None else str(row["stop_threshold"]),
        )
    console.print(table)


@setting_app.command("show")
def task_setting_show(task: str, setting: str) -> None:
    """Print one Setting as JSON."""
    rows = asyncio.run(_api_get(f"/api/tasks/{task}/settings"))
    row = next((r for r in (rows or []) if r.get("id") == setting or r.get("name") == setting), None)
    if row is None:
        console.print(f"[red]no setting {setting!r} on Task {task!r}[/]")
        raise typer.Exit(1)
    console.print_json(data=row)


def _setting_json(path: str) -> dict:
    try:
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"[red]could not read setting JSON:[/] {exc}")
        raise typer.Exit(1)
    if not isinstance(value, dict):
        console.print("[red]setting JSON must be an object.[/]")
        raise typer.Exit(1)
    editable = {
        "name", "dataset", "dataset_split", "dataset_config",
        "validation_set", "validation_split", "validation_config",
        "validation_answer_fields",
        "validation_sample_submission", "validation_metric_type",
        "validation_metric", "validation_metric_direction",
        "validation_evaluation_script", "base_model", "training_method",
        "method_config", "data_query", "model_query", "method_query",
        "iteration_budget", "max_cost_usd",
        "stop_threshold", "prompt_framing", "system_prompt",
        "loss_objective_config", "inference_config", "decoding_config",
    }
    return {key: item for key, item in value.items() if key in editable}


@setting_app.command("add")
def task_setting_add(task: str, json_file: str = typer.Option(..., "--json")) -> None:
    """Create a Setting from the API-shaped JSON object."""
    result = asyncio.run(_api("POST", f"/api/tasks/{task}/settings", json=_setting_json(json_file)))
    if result is None: raise typer.Exit(1)
    console.print(f"[green]added[/] {result.get('name', result.get('id', ''))}")


@setting_app.command("edit")
def task_setting_edit(task: str, setting_id: str, json_file: str = typer.Option(..., "--json")) -> None:
    """Replace a Setting's editable fields from an API-shaped JSON object."""
    result = asyncio.run(_api(
        "PATCH", f"/api/tasks/{task}/settings/{setting_id}", json=_setting_json(json_file),
    ))
    if result is None: raise typer.Exit(1)
    console.print(f"[green]updated[/] {result.get('name', setting_id)}")


@setting_app.command("rm")
def task_setting_rm(task: str, setting_id: str, yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Delete a saved Setting without changing historical Runs."""
    if not yes and not typer.confirm(f"Delete setting {setting_id!r} from Task {task!r}?"):
        raise typer.Exit(0)
    result = asyncio.run(_api("DELETE", f"/api/tasks/{task}/settings/{setting_id}"))
    if result is None: raise typer.Exit(1)
    console.print(f"[green]deleted[/] {setting_id}")


@task_app.command("rm")
def task_rm(
    name: str = typer.Argument(..., help="Task name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Delete a task. Runs already started from it keep their name and history."""
    if not yes and not typer.confirm(f"Delete task {name!r}?"):
        raise typer.Exit(0)
    asyncio.run(_task_rm(name))


async def _task_rm(name: str) -> None:
    import httpx
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.delete(f"{_api_base()}/api/tasks/{name}")
    if r.status_code >= 400:
        console.print(f"[red]{r.status_code}[/red] {r.text}")
        raise typer.Exit(1)
    console.print(f"[green]deleted[/green] {name}")


# ----------- datasets ----------------------------------------------------
#
# The catalogue is a directory tree, not a table, so these read the files the
# API serves and work with the backend down.

_META_FILES = {"source.json", ".profile.json"}
_IGNORED_FILE_NAMES = {".DS_Store"}
_IGNORED_DIR_NAMES = {"__pycache__", "__MACOSX"}


def _dataset_dirs() -> list[Path]:
    if not _FILES_DIR.is_dir():
        return []
    return sorted(p for p in _FILES_DIR.iterdir() if p.is_dir())


def _dataset_files(d: Path) -> list[Path]:
    """Return every real catalogue file, including files in split folders.

    ``source.json`` and ``.profile.json`` are catalogue metadata only when they
    live at the file-set root. Generated cache files are never user inputs.
    """
    files: list[Path] = []
    for f in d.rglob("*"):
        if not f.is_file():
            continue
        relative = f.relative_to(d)
        if len(relative.parts) == 1 and f.name in _META_FILES:
            continue
        if f.name in _IGNORED_FILE_NAMES or f.suffix.lower() == ".pyc":
            continue
        if any(part in _IGNORED_DIR_NAMES for part in relative.parts[:-1]):
            continue
        files.append(f)
    return sorted(files, key=lambda f: f.relative_to(d).as_posix())


def _source_of(d: Path) -> dict:
    f = d / "source.json"
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


@dataset_app.command("list")
def dataset_list() -> None:
    """Every file set in the catalogue (UI: Files)."""
    dirs = _dataset_dirs()
    if not dirs:
        console.print(f"[dim]no file sets under {_FILES_DIR}[/dim]")
        return
    t = Table(box=None, pad_edge=False)
    t.add_column("name", style="bold cyan", no_wrap=True)
    t.add_column("files", justify="right")
    t.add_column("size", justify="right")
    t.add_column("what it is", max_width=36, overflow="ellipsis", no_wrap=True)
    for d in dirs:
        files = _dataset_files(d)
        t.add_row(d.name, str(len(files)),
                  _fmt_bytes(sum(f.stat().st_size for f in files)),
                  _source_of(d).get("note") or "")
    console.print(t)


@dataset_app.command("show")
def dataset_show(
    name: str = typer.Argument(..., help="File-set name."),
    rows: int = typer.Option(0, "--rows", "-n", help="Also print the first N rows of each CSV/JSONL."),
) -> None:
    """One file set: its files, and which Tasks use it."""
    d = _FILES_DIR / name
    if not d.is_dir():
        console.print(f"[red]no file set {name!r} under {_FILES_DIR}[/red]")
        raise typer.Exit(1)

    src = _source_of(d)
    console.print(f"[bold]{name}[/bold]" + (f" — {src['note']}" if src.get("note") else ""))
    console.print(f"[dim]{d}[/dim]\n")

    t = Table(box=None, pad_edge=False)
    t.add_column("file")
    t.add_column("size", justify="right")
    t.add_column("rows", justify="right")
    for f in _dataset_files(d):
        n = "-"
        if f.suffix.lower() in (".csv", ".jsonl"):
            try:
                with f.open("r", encoding="utf-8", errors="replace") as fh:
                    count = sum(1 for _ in fh)
                # a CSV's header is not a row
                n = str(count - 1 if f.suffix.lower() == ".csv" and count else count)
            except OSError:
                n = "?"
        t.add_row(f.relative_to(d).as_posix(), _fmt_bytes(f.stat().st_size), n)
    for r in src.get("remote") or []:
        # The split is part of the identity: one repo listed twice IS two files.
        where = r.get("split") or "train"
        if r.get("config"):
            where = f"{r['config']}/{where}"
        t.add_row(
            f"{r.get('id', '?')} [dim]({where} · {r.get('kind', 'remote')})[/dim]",
            "-", "-",
        )
    console.print(t)

    if rows > 0:
        for f in _dataset_files(d):
            if f.suffix.lower() not in (".csv", ".jsonl"):
                continue
            console.print(f"\n[bold]{f.relative_to(d).as_posix()}[/bold] — first {rows}")
            with f.open("r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i >= rows + (1 if f.suffix.lower() == ".csv" else 0):
                        break
                    console.print(f"  [dim]{line.rstrip()[:200]}[/dim]")

    asyncio.run(_dataset_users(name))


async def _dataset_users(name: str) -> None:
    """Which tasks reference this dataset — the reason to keep or drop it."""
    from zevo.db import Task
    prefix = str(_FILES_DIR / name) + "/"
    Session = get_session_factory()
    async with Session() as s:
        tasks = (await s.execute(select(Task).order_by(Task.name))).scalars().all()
        users = []
        for task in tasks:
            refs = [task.test_set, task.test_sample_submission]
            refs.extend(
                str(item.get(key) or "")
                for item in (task.test_sets or [])
                if isinstance(item, dict)
                for key in ("test_set", "sample_submission")
            )
            if any((ref or "").startswith(prefix) for ref in refs):
                users.append(task.name)
    console.print(f"\n[dim]used by:[/dim] {', '.join(users) if users else '(no task)'}")


# ----------- the remaining pages ----------------------------------------


@app.command("dashboard")
def dashboard(
    window: str = typer.Option(
        "all", "--window", help="1d | 1w | 1m | all (applies to Time and Cost).",
    ),
) -> None:
    """System vitals and task-separated held-out model outcomes."""
    if window not in ("1d", "1w", "1m", "all"):
        console.print("[red]--window must be 1d, 1w, 1m, or all[/]")
        raise typer.Exit(2)
    asyncio.run(_dashboard(window=window))


async def _dashboard(*, window: str) -> None:
    from datetime import datetime, timedelta, timezone

    runs = await _api_get("/api/runs?limit=500")
    models = await _api_get("/api/models")
    cost = await _api_get(f"/api/cost/total?window={window}")
    if runs is None or models is None or cost is None:
        raise typer.Exit(1)

    active = sum(1 for r in runs if r.get("status") == "running")
    run_ids = {str(r.get("id", "")) for r in runs}
    model_count = sum(
        1 for m in models if m.get("run_id") and str(m.get("run_id")) in run_ids
    )

    window_delta = {
        "1d": timedelta(days=1), "1w": timedelta(days=7),
        "1m": timedelta(days=30), "all": None,
    }[window]
    cutoff = datetime.now(timezone.utc) - window_delta if window_delta else None
    run_time = 0.0
    terminal = {"success", "degraded", "failed", "cancelled", "halted"}
    for run in runs:
        duration = run.get("duration_s")
        if run.get("status") not in terminal or not isinstance(duration, (int, float)) or duration < 0:
            continue
        if cutoff is not None:
            try:
                finished = datetime.fromisoformat(str(run.get("finished_at", "")).replace("Z", "+00:00"))
            except ValueError:
                continue
            if finished.tzinfo is None:
                finished = finished.replace(tzinfo=timezone.utc)
            if finished < cutoff:
                continue
        run_time += float(duration)

    # Match ConsoleOverview: a saved champion from a usable terminal run, with
    # both held-out endpoints. Group by task + metric + direction so unlike
    # evaluation contracts are never averaged together.
    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for run in runs:
        if run.get("status") not in ("success", "degraded"):
            continue
        if not run.get("registry_version_tag"):
            continue
        baseline = run.get("baseline_test_score")
        best = run.get("champion_test_score")
        if not isinstance(baseline, (int, float)) or not isinstance(best, (int, float)):
            continue
        direction = str(run.get("metric_direction") or "max")
        lift = float(baseline) - float(best) if direction == "min" else float(best) - float(baseline)
        key = (
            str(run.get("task_name") or "—"),
            str(run.get("metric") or "score"),
            direction,
        )
        grouped.setdefault(key, []).append((lift, float(best)))

    def duration_text(seconds: float | None) -> str:
        if seconds is None:
            return "—"
        total = int(round(seconds))
        if total >= 3600:
            return f"{total // 3600}h {(total % 3600) // 60}m"
        if total >= 60:
            return f"{total // 60}m {total % 60}s"
        return f"{total}s"

    vitals = Table(show_header=False, box=None, padding=(0, 3, 0, 0))
    vitals.add_column(style="dim", no_wrap=True)
    vitals.add_column()
    vitals.add_row("ACTIVE RUNS", str(active))
    vitals.add_row("SAVED MODELS", str(model_count))
    vitals.add_row("RUN TIME", f"{duration_text(run_time)}  [dim]({window})[/]")
    vitals.add_row(
        "COST", f"${cost.get('total_usd', 0):.2f}  [dim]({window} · "
        f"tokens ${cost.get('agent_usd', 0):.2f} · GPU ${cost.get('gpu_usd', 0):.2f})[/]",
    )
    console.print(vitals)

    console.rule("Model Improvement by Zevo")
    stats = Table(box=None, pad_edge=False)
    stats.add_column("task", style="cyan")
    stats.add_column("metric")
    stats.add_column("target")
    stats.add_column("runs", justify="right")
    stats.add_column("average improvement", justify="right")
    stats.add_column("average test score", justify="right")
    stats.add_column("best test score", justify="right")
    for (task, metric, direction), outcomes in sorted(grouped.items()):
        gains = [gain for gain, _score in outcomes]
        scores = [score for _gain, score in outcomes]
        best = min(scores) if direction == "min" else max(scores)
        stats.add_row(
            task,
            metric.replace("_", " ").title(),
            direction.capitalize(),
            str(len(outcomes)),
            _score_text(sum(gains) / len(gains), metric, signed=True),
            _score_text(sum(scores) / len(scores), metric),
            _score_text(best, metric),
        )
    if not grouped:
        stats.add_row("—", "—", "—", "0", "—", "—", "—")
    console.print(stats)


@app.command("leaderboard")
def leaderboard(
    by: str = typer.Option("base", "--by", help="base | harness — what an entrant is."),
) -> None:
    """Per task, which entrant did best (UI: Leaderboard)."""
    asyncio.run(_leaderboard(by))


async def _leaderboard(by: str) -> None:
    if by not in ("base", "harness"):
        console.print("[red]--by must be base or harness[/red]")
        raise typer.Exit(1)
    data = await _api_get(f"/api/leaderboard?by={by}")
    if data is None:
        raise typer.Exit(1)
    cells = sorted(
        data.get("cells", []),
        key=lambda c: (
            c["task_name"],
            tuple(c.get("setting_key") or []) if by == "harness" else (),
            c["champion_test_score_rank"],
        ),
    )
    if not cells:
        console.print("[dim]no scored runs yet[/dim]")
        return
    t = Table(box=None, pad_edge=False)
    t.add_column("task", style="bold cyan", no_wrap=True)
    # Same two columns the page shows: the model a row IS, then what it came
    # from. Neither held-out result is collapsed into the other: each prints
    # its own dense rank within the same scope as the web board.
    t.add_column("model", no_wrap=True)
    t.add_column("base" if by == "base" else "harness", no_wrap=True)
    t.add_column("test score", justify="right")
    t.add_column("improvement", justify="right")
    t.add_column("cost", justify="right")
    for c in cells:
        imp = c.get("improvement")
        run_id = str(c.get("run_id") or "")
        entrant = str(c.get("model") or "")
        t.add_row(c["task_name"], f"M-{run_id[:8]}" if run_id else "—",
                  entrant.split("/")[-1],
                  _score_text(c["champion_test_score"], c.get("metric")),
                  _score_text(imp, c.get("metric"), signed=True),
                  f"${c.get('cost_usd', 0):.2f}")
    console.print(t)


@app.command("hardware")
def hardware() -> None:
    """GPUs Zevo can train on (UI: Hardware)."""
    asyncio.run(_hardware())


async def _hardware() -> None:
    local = await _api("POST", "/api/hardware/detect-local", json={})
    if local is None:
        raise typer.Exit(1)
    console.print("[bold]Local GPU[/]")
    local_table = Table(box=None, pad_edge=False)
    local_table.add_column("detected", no_wrap=True)
    local_table.add_column("GPUs", justify="right")
    local_table.add_column("model")
    local_table.add_column("VRAM", justify="right")
    local_table.add_column("driver")
    local_table.add_column("CUDA")
    local_table.add_row(
        "[green]yes[/green]" if local.get("has_gpu") else "[dim]no[/dim]",
        str(local.get("gpu_count", 0)),
        str(local.get("gpu_name") or "—"),
        f"{local.get('vram_gb', 0):g} GB" if local.get("has_gpu") else "—",
        str(local.get("driver_version") or "—"),
        str(local.get("cuda_version") or "—"),
    )
    console.print(local_table)
    if local.get("error"):
        console.print(f"[dim]{local['error']}[/]")

    data = await _api_get("/api/hardware/cloud/backends")
    if data is None:
        raise typer.Exit(1)
    backends = data.get("backends", [])
    if not backends:
        console.print("[dim]no cloud backends[/dim]")
        return
    console.print("\n[bold]Cloud GPUs[/]")
    t = Table(box=None, pad_edge=False)
    t.add_column("backend", style="bold cyan", no_wrap=True)
    t.add_column("credentials", no_wrap=True)
    t.add_column("rented now", justify="right")
    for b in backends:
        rented = "—"
        if b.get("configured"):
            # Only ask a backend for its instances when a key exists; without
            # one the call is a guaranteed error, not information.
            live = await _api_get(f"/api/hardware/cloud/{b['id']}/instances")
            if live is not None:
                rented = str(len(live.get("instances", [])))
        t.add_row(b.get("label") or b["id"],
                  "[green]set[/green]" if b.get("configured") else "[dim]missing[/dim]",
                  rented)
    console.print(t)


@app.command("settings")
def settings() -> None:
    """Which credentials are set (UI: Settings). Values are never printed."""
    asyncio.run(_settings())


async def _settings() -> None:
    data = await _api_get("/api/settings")
    if data is None:
        raise typer.Exit(1)
    entries = data.get("entries", [])
    if not entries:
        console.print("[dim]nothing configured[/dim]")
        return
    console.print(f"[dim]{data.get('env_path', '')}[/dim]\n")
    t = Table(box=None, pad_edge=False)
    t.add_column("key", style="bold cyan", no_wrap=True)
    t.add_column("set", justify="center", no_wrap=True)
    # `preview` is the masked tail the API returns; the value itself is never
    # sent, so there is nothing here to leak into a terminal history.
    t.add_column("preview", no_wrap=True)
    for e in entries:
        t.add_row(e.get("name", "?"),
                  "[green]yes[/green]" if e.get("present") else "[dim]no[/dim]",
                  e.get("preview") or "")
    console.print(t)


@app.command("levels")
def levels() -> None:
    """What L1-L4 mean (UI: Levels)."""
    t = Table(box=None, pad_edge=False)
    t.add_column("level", style="bold cyan", no_wrap=True)
    t.add_column("training data", no_wrap=True)
    t.add_column("training model", no_wrap=True)
    t.add_column("training method", no_wrap=True)
    t.add_column("what Zevo decides")
    for lvl, owners, blurb in (
        ("L1", (1, 1, 1), "Optimize only inside the user-pinned data/model/method branch"),
        ("L2", (0, 1, 1), "Exhaust the current Data branch before selecting another Data"),
        ("L3", (0, 1, 0), "Exhaust Data branches before advancing to another Method"),
        ("L4", (0, 0, 0), "Exhaust Data, then Method, then Base-model branches"),
    ):
        t.add_row(lvl, *[("User" if o else "[bold]Zevo[/bold]") for o in owners], blurb)
    console.print(
        "Every task rests on three decisions: the training data, the model, and the method.\n"
        "Each is either given by the user or made by Zevo. The more Zevo decides, the higher\n"
        "the level. User values stay pinned for the Run. Zevo retains the active branch until\n"
        "its inner search is exhausted, then advances Data -> Method -> Base model.\n"
    )
    console.print(t)

# =========================================================================
# Write commands, one per action the UI offers.
#
# Everything that changes state goes through the API rather than the database,
# so the CLI and the UI take exactly the same code path — a task added here is
# validated by the same handler, and a rented box is booked by the same client.
# The two outward-facing ones (renting hardware, writing credentials) confirm
# first and never take a secret as an argument.
# =========================================================================


async def _api(method: str, path: str, **kw):
    """Call the backend and return the parsed body, or None on failure."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.request(method, f"{_api_base()}{path}", **kw)
    except Exception as e:  # noqa: BLE001 — the backend being down is normal
        console.print(f"[red]backend unreachable[/red] at {_api_base()} ({type(e).__name__})")
        return None
    if r.status_code >= 400:
        console.print(f"[red]{r.status_code}[/red] {r.text}")
        return None
    if r.status_code == 204 or not r.content:
        return {}
    return r.json()


# ----------- datasets: the write half ------------------------------------


@dataset_app.command("add")
def dataset_add(
    name: str = typer.Argument(..., help="File-set name — the folder files go into."),
    files: list[str] = typer.Argument(None, help="Local files to upload. Repeatable."),
) -> None:
    """Create a file set, or add files to one that exists."""
    asyncio.run(_dataset_add(name, list(files or [])))


async def _dataset_add(name: str, files: list[str]) -> None:
    import httpx
    if not files:
        console.print("[red]give at least one file[/red] — or use `file add-remote`.")
        raise typer.Exit(1)
    for local in files:
        f = Path(local).expanduser()
        if not f.is_file():
            console.print(f"[red]not a file:[/red] {f}")
            raise typer.Exit(1)
        async with httpx.AsyncClient(timeout=300) as client:
            with f.open("rb") as fh:
                r = await client.post(
                    f"{_api_base()}/api/files",
                    files={"file": (f.name, fh)}, data={"name": name},
                )
        if r.status_code >= 400:
            console.print(f"[red]{r.status_code}[/red] {r.text}")
            raise typer.Exit(1)
        console.print(f"[green]uploaded[/green] {f.name} -> {name}")


@dataset_app.command("add-remote")
def dataset_add_remote(
    name: str = typer.Argument(..., help="File-set name."),
    ident: str = typer.Argument(
        ..., help="Hugging Face dataset id or an http(s) URL.",
    ),
    kind: str = typer.Option(
        "huggingface", "--kind", help="Remote kind: huggingface or url.",
    ),
    role: str = typer.Option(
        "train", "--role",
        help="What this is in the file set: train | validation | test.",
    ),
    split: str = typer.Option(
        "", "--split",
        help="Which slice to load — slicing syntax works ('train[:2000]'). "
             "Empty means train. Defaults to --role when that is given.",
    ),
    config: str = typer.Option(
        "", "--config", help="Named subset, for repos that ship several.",
    ),
) -> None:
    """Name a Hugging Face dataset or URL as one of this file set's files.

    Nothing is downloaded — the id, split and config are recorded, and the data
    agent loads exactly that when a run needs it. List the same repo twice to
    pull two slices of it: its train split as training data, its validation
    split as the set runs tune on.
    """
    kind = kind.strip().lower()
    if kind not in ("huggingface", "url"):
        console.print("[red]--kind must be huggingface or url.[/]")
        raise typer.Exit(1)
    if role not in ("train", "validation", "test"):
        console.print("[red]--role must be train, validation, or test.[/]")
        raise typer.Exit(1)
    d = asyncio.run(_api(
        "POST", f"/api/files/{name}/remote",
        json={
            "id": ident, "kind": kind, "role": role,
            "split": split or role, "config": config,
        },
    ))
    if d is None:
        raise typer.Exit(1)
    console.print(f"[green]added[/green] {ident} ({split or role}) to {name}")


@dataset_app.command("note")
def dataset_note(
    name: str = typer.Argument(..., help="File-set name."),
    text: str = typer.Argument(..., help="One line: what this dataset is."),
) -> None:
    """Set the one-line description shown on the file-set card."""
    d = asyncio.run(_api("PUT", f"/api/files/{name}/note", json={"note": text}))
    if d is None:
        raise typer.Exit(1)
    console.print(f"[green]updated[/green] {name}")


@dataset_app.command("edit-remote")
def dataset_edit_remote(
    name: str = typer.Argument(..., help="File-set name."),
    ident: str = typer.Argument(..., help="Hugging Face id or URL already listed."),
    current_split: str = typer.Option("", "--current-split", help="Select one entry when the id appears more than once."),
    role: Optional[str] = typer.Option(None, "--role", help="train, validation, or test."),
    split: Optional[str] = typer.Option(None, "--split"),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Edit a remote file entry without deleting and recreating it."""
    from urllib.parse import quote, urlencode
    if role is not None and role not in ("train", "validation", "test"):
        console.print("[red]--role must be train, validation, or test.[/]")
        raise typer.Exit(1)
    body = {key: value for key, value in {
        "role": role, "split": split, "config": config,
    }.items() if value is not None}
    if not body:
        console.print("[yellow]nothing to update[/]")
        raise typer.Exit(1)
    query = f"?{urlencode({'split': current_split})}" if current_split else ""
    result = asyncio.run(_api(
        "PATCH", f"/api/files/{quote(name, safe='')}/remote/{quote(ident, safe='')}{query}",
        json=body,
    ))
    if result is None:
        raise typer.Exit(1)
    console.print(f"[green]updated[/] {ident} on {name}")


@dataset_app.command("rm")
def dataset_rm(
    name: str = typer.Argument(..., help="File-set name."),
    file: str = typer.Option("", "--file", "-f", help="Delete just this file."),
    remote: str = typer.Option(
        "", "--remote", help="Stop listing this remote id or URL.",
    ),
    split: str = typer.Option(
        "", "--split",
        help="With --remote: which entry, when the same id is listed under "
             "several splits.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Delete a local file, remote entry, or the whole file set."""
    what = f"file {file!r} from {name}" if file else (
        f"remote entry {remote!r} from {name}"
        if remote else f"the whole file set {name!r}"
    )
    if not yes and not typer.confirm(f"Delete {what}?"):
        raise typer.Exit(0)
    if file:
        path = f"/api/files/{name}/contents/{file}"
    elif remote:
        from urllib.parse import quote, urlencode
        query = f"?{urlencode({'split': split})}" if split else ""
        path = (
            f"/api/files/{quote(name, safe='')}/remote/"
            f"{quote(remote, safe='')}{query}"
        )
    else:
        path = f"/api/files/{name}"
    if asyncio.run(_api("DELETE", path)) is None:
        raise typer.Exit(1)
    console.print(f"[green]deleted[/green] {what}")


@dataset_app.command("profile")
def dataset_profile(
    name: str = typer.Argument(..., help="File-set name."),
    refresh: bool = typer.Option(False, "--refresh", help="Re-run the profiler, ignoring the cache."),
) -> None:
    """The structured profile the data agent receives as context."""
    if refresh:
        if asyncio.run(_api("POST", f"/api/files/{name}/profile/refresh")) is None:
            raise typer.Exit(1)
    d = asyncio.run(_api("GET", f"/api/files/{name}/profile"))
    if d is None:
        raise typer.Exit(1)
    t = Table(box=None, pad_edge=False)
    t.add_column("field", style="dim", no_wrap=True)
    t.add_column("value")
    tt = d.get("task_type") or {}
    t.add_row("file", str(d.get("file", "")))
    t.add_row("task type", f"{tt.get('label', '?')}  [dim]conf {tt.get('confidence', 0):.0%}[/dim]")
    t.add_row("format", str(d.get("expected_format", "")))
    t.add_row("rows", f"{d.get('n_rows', 0):,}  [dim]{d.get('n_columns', 0)} cols[/dim]")
    t.add_row("columns", ", ".join(c.get("name", "") for c in d.get("columns", [])))
    split = d.get("split_recommendation") or {}
    if split:
        t.add_row("split", str(split.get("rationale", "")))
    t.add_row("ready for data", "yes" if d.get("ready_for_data") else "no")
    for issue in d.get("issues", []):
        t.add_row("issue", f"[yellow]{issue.get('code', '')}[/yellow] {issue.get('detail', '')}")
    if d.get("notes"):
        t.add_row("notes", str(d["notes"]))
    console.print(t)


# ----------- runs + tickets: the write half ------------------------------


@run_app.command("rm")
def run_rm(
    run_id: str = typer.Argument(..., help="Run id (a prefix is enough)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Delete a run and its history. In-flight work is cancelled and its GPU released first."""
    # The help says a prefix is enough, and `run watch` / `run cancel` mean it.
    # This one used to hand the prefix straight to the API, which only knows
    # full ids, so it 404'd on exactly the short id `run list` prints.
    async def go() -> None:
        import httpx
        base = _api_base().rstrip("/")
        async with httpx.AsyncClient(timeout=30) as client:
            full = await _resolve_run_id(client, base, run_id)
        if not yes and not typer.confirm(f"Delete run {full[:8]}? This cannot be undone."):
            raise typer.Exit(0)
        if await _api("DELETE", f"/api/runs/{full}") is None:
            raise typer.Exit(1)
        console.print(f"[green]deleted[/green] {full}")

    asyncio.run(go())


@ticket_app.command("rerun")
def ticket_rerun(
    ticket_id: str = typer.Argument(..., help="Ticket id."),
    strategy: str = typer.Option(
        "fresh", "--strategy",
        help="fresh | from_checkpoint.",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Run even when the failure classifier says the input is structural.",
    ),
) -> None:
    """Run a terminal ticket again, using the UI's retry policy and strategies."""
    if strategy not in ("fresh", "from_checkpoint"):
        console.print("[red]--strategy must be fresh or from_checkpoint[/]")
        raise typer.Exit(2)
    d = asyncio.run(_api(
        "POST", f"/api/tickets/{ticket_id}/rerun",
        json={"strategy": strategy, "actor": "cli", "force": force},
    ))
    if d is None:
        raise typer.Exit(1)
    status = d.get("status", "")
    color = "green" if status == "queued" else "yellow"
    console.print(f"[{color}]{status or 'rerun requested'}[/] {ticket_id}")
    if d.get("note"):
        console.print(f"[dim]{d['note']}[/]")


@ticket_app.command("retry-status")
def ticket_retry_status(ticket_id: str = typer.Argument(..., help="Ticket id.")) -> None:
    """Classify the latest failure before deciding whether to rerun it."""
    d = asyncio.run(_api("GET", f"/api/tickets/{ticket_id}/retry-status"))
    if d is None:
        raise typer.Exit(1)
    color = {
        "transient": "green", "structural": "red", "cancelled": "cyan",
    }.get(str(d.get("verdict", "")), "yellow")
    console.print(
        f"[{color}]{d.get('verdict', 'unknown')}[/]  "
        f"code={d.get('code', 'unknown')}  "
        f"retryable={str(bool(d.get('retryable'))).lower()}  "
        f"exit={d.get('last_exit_code', 0)}"
    )
    console.print(str(d.get("reason", "")))
    if d.get("recovery"):
        console.print(f"[dim]Recovery: {escape(str(d['recovery']))}[/]")
    if d.get("last_error"):
        console.rule("last error")
        console.print(escape(str(d["last_error"])))


@ticket_app.command("heartbeat")
def ticket_heartbeat(ticket_id: str = typer.Argument(..., help="Ticket id.")) -> None:
    """Queue one on-demand heartbeat for a Ticket, exactly like the Ticket page."""
    d = asyncio.run(_api(
        "POST", f"/api/tickets/{ticket_id}/heartbeat", json={},
    ))
    if d is None:
        raise typer.Exit(1)
    console.print(
        f"[green]{d.get('status', 'queued')}[/] heartbeat for {ticket_id}  "
        f"[dim]wakeup={d.get('wakeup_id', '')}[/]"
    )


@heartbeats_app.command("cancel")
def heartbeat_cancel(
    heartbeat_id: str = typer.Argument(..., help="Heartbeat id (a unique prefix is enough)."),
) -> None:
    """Stop one running heartbeat without cancelling its whole Ticket."""
    d = asyncio.run(_api(
        "POST", f"/api/heartbeats/{heartbeat_id}/cancel", json={},
    ))
    if d is None:
        raise typer.Exit(1)
    status = str(d.get("status", ""))
    color = "green" if status == "cancelled" else "yellow"
    console.print(f"[{color}]{status or 'cancel requested'}[/] heartbeat {heartbeat_id}")


@ticket_app.command("message")
def ticket_message(
    ticket_id: str = typer.Argument(..., help="Ticket id."),
    body: str = typer.Argument(..., help="What to say. The agent reads it on its next turn."),
) -> None:
    """Send a message on a ticket.

    On a live run the agent wakes up and reads it. On a finished one the
    message is only recorded — say which happened, so nobody waits on work
    that is never going to start.
    """
    d = asyncio.run(_api("POST", f"/api/tickets/{ticket_id}/messages",
                         json={"author": "cli", "body": body}))
    if d is None:
        raise typer.Exit(1)
    if d.get("woke_agent"):
        console.print(f"[green]posted[/green] on {ticket_id}; the agent will pick it up")
    else:
        console.print(f"[green]posted[/green] on {ticket_id} "
                      f"[dim](run is finished — noted, not acted on; "
                      f"use `ticket rerun` to re-run it)[/]")


# ----------- hardware: search, rent, destroy -----------------------------


@app.command("gpu-search")
def gpu_search(
    backend: str = typer.Argument("vastai", help="vastai | lambda."),
    min_vram: float = typer.Option(16.0, "--min-vram", help="GB per GPU."),
    gpu_type: str = typer.Option("", "--gpu-type", help="e.g. A100, H100."),
    num_gpus: int = typer.Option(1, "--num-gpus"),
    max_dph: float = typer.Option(0.0, "--max-dph", help="$/hour cap. 0 = none."),
    limit: int = typer.Option(10, "--limit"),
) -> None:
    """Rentable machines on a cloud backend (UI: Hardware -> search)."""
    offers = asyncio.run(_api("POST", f"/api/hardware/cloud/{backend}/search", json={
        "min_vram_gb": min_vram, "gpu_type": gpu_type,
        "num_gpus": num_gpus, "max_dph": max_dph, "limit": limit,
    }))
    if offers is None:
        raise typer.Exit(1)
    if not offers:
        console.print("[dim]no offers matched[/dim]")
        return
    t = Table(box=None, pad_edge=False)
    t.add_column("offer id", style="cyan", no_wrap=True)
    t.add_column("gpu", no_wrap=True)
    t.add_column("n", justify="right")
    t.add_column("vram", justify="right")
    t.add_column("$/hour", justify="right")
    t.add_column("region", no_wrap=True)
    for o in offers:
        t.add_row(str(o.get("id", "")), o.get("gpu_name", ""), str(o.get("num_gpus", "")),
                  f"{o.get('gpu_ram_gb', 0):.0f} GB", f"${o.get('dph_total', 0):.3f}",
                  o.get("region", "") or "—")
    console.print(t)
    console.print(f"\n[dim]rent one:[/dim] gpu-rent {backend} <offer id>")


@app.command("gpu-rent")
def gpu_rent(
    backend: str = typer.Argument(..., help="vastai | lambda."),
    offer_id: str = typer.Argument(..., help="An id from `gpu-search`."),
    region: str = typer.Option("", "--region", help="Lambda needs one; Vast.ai ignores it."),
    disk_gb: int = typer.Option(50, "--disk-gb"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Rent a machine. This SPENDS MONEY on your provider account."""
    if not yes and not typer.confirm(
        f"Rent offer {offer_id} on {backend}? This starts billing immediately."
    ):
        raise typer.Exit(0)
    d = asyncio.run(_api("POST", f"/api/hardware/cloud/{backend}/rent", json={
        "offer_id": offer_id, "disk_gb": disk_gb, "region": region,
    }))
    if d is None:
        raise typer.Exit(1)
    console.print(f"[green]rented[/green] instance {d.get('instance_id')}  "
                  f"[dim]destroy it with:[/dim] gpu-destroy {backend} {d.get('instance_id')}")


@app.command("gpu-destroy")
def gpu_destroy(
    backend: str = typer.Argument(..., help="vastai | lambda."),
    instance_id: str = typer.Argument(..., help="Instance id."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Destroy a rented machine and stop its billing."""
    if not yes and not typer.confirm(f"Destroy {instance_id} on {backend}?"):
        raise typer.Exit(0)
    if asyncio.run(_api("DELETE", f"/api/hardware/cloud/{backend}/instances/{instance_id}")) is None:
        raise typer.Exit(1)
    console.print(f"[green]destroyed[/green] {instance_id}")


# ----------- settings ----------------------------------------------------


@app.command("set-cloud-backend")
def set_cloud_backend(
    backend: str = typer.Argument(
        ..., help="vastai | lambda | auto (auto clears the explicit selection).",
    ),
) -> None:
    """Choose the non-secret cloud GPU backend, matching the Settings selector."""
    normalized = backend.strip().lower()
    if normalized not in ("vastai", "lambda", "auto"):
        console.print("[red]backend must be vastai, lambda, or auto[/]")
        raise typer.Exit(2)
    value = "" if normalized == "auto" else normalized
    d = asyncio.run(_api(
        "POST", "/api/settings",
        json={"values": {"ZEVO_CLOUD_BACKEND": value}},
    ))
    if d is None:
        raise typer.Exit(1)
    console.print(
        f"[green]cloud backend set to[/] {normalized}  "
        "[dim](restart backend and scheduler before launching work)[/]"
    )


@app.command("set-secret")
def set_secret(
    key: str = typer.Argument(..., help="Key name, e.g. VASTAI_API_KEY. See `settings`."),
    clear: bool = typer.Option(False, "--clear", help="Unset the key instead."),
) -> None:
    """Set one credential. The value is PROMPTED FOR, never passed as an argument.

    An API key on a command line ends up in shell history, in `ps` output, and
    in any terminal recording. Typing it at a hidden prompt does not.
    """
    value = ""
    if not clear:
        value = typer.prompt(f"value for {key}", hide_input=True)
        if not value.strip():
            console.print("[red]empty — nothing written.[/red] Use --clear to unset.")
            raise typer.Exit(1)
    d = asyncio.run(_api("POST", "/api/settings", json={"values": {key: value}}))
    if d is None:
        raise typer.Exit(1)
    console.print(f"[green]{'cleared' if clear else 'set'}[/green] {key}"
                  + ("" if clear else "  [dim](restart the stack for agents to pick it up)[/dim]"))


def main() -> None:
    """Installed entry point; public commands are accepted only inside the shell."""
    if len(sys.argv) > 1 and os.environ.get("ZEVO_INTERACTIVE_COMMAND") != "1":
        console.print("[yellow]Zevo commands run inside the interactive shell.[/]")
        console.print("Start it with: [bold cyan]zevo[/]")
        raise SystemExit(2)
    app(prog_name="zevo")


if __name__ == "__main__":
    main()
