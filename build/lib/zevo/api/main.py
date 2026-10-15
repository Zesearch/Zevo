"""FastAPI backend for Zevo.

Mounts routers under /api/* matching the plan's REST surface. Lifespan
hook seeds the agents table from playbook/agents/<id>/identity.md so a fresh DB is immediately
usable.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from zevo.api.config import settings
from zevo.api.csrf import CsrfOriginMiddleware
from zevo.api.observability import (
    RequestContextMiddleware,
    configure_logging,
    init_error_tracking,
)
# Routers, grouped by WHO CALLS THEM — the one thing you need to know before
# changing one. `ui` is safe to reshape with the page in front of you; `agent`
# is a contract with code running in a container that you cannot see; `shared`
# is both at once, which is exactly the case worth marking.
from zevo.api.routers.ui import (
    agents,
    attachments,
    auth_status,
    cost,
    failure_modes,
    files as files_router,
    hardware,
    heartbeat_events,
    heartbeats,
    leaderboard,
    models as models_router,
    preflight,
    settings as settings_router,
    ssh_hardware,
    tasks,
    wakeups,
    ws,
)
from zevo.api.routers.shared import memory, run_timeline, runs, tickets
from zevo.api.routers.agent import gpu_leases, infra
from zevo.engine.agent.registry import seed_agents_main


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        n = await seed_agents_main()
        print(f"[backend] seeded {n} execution roles from Playbook manifests")
    except Exception as e:
        # Non-fatal: lets the API come up even if identity.md / DB seed has issues.
        print(f"[backend] WARN: seed_agents failed at startup: {e}")
    try:
        n = await ssh_hardware.sync_ssh_connections_with_env()
        print(f"[backend] synchronized {n} SSH connection profiles with .env")
    except Exception as e:
        # Connection sync must not make the entire UI unavailable. The Settings
        # page will still expose the database rows and their previous status.
        print(f"[backend] WARN: SSH connection .env sync failed at startup: {e}")
    yield


def create_application() -> FastAPI:
    # Route the root logger to stdout as structured JSON (request-id correlated)
    # before the app is built, so startup + every request logs consistently.
    # init_error_tracking() is a no-op unless SENTRY_DSN is set.
    configure_logging()
    init_error_tracking()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Automated fine-tuning system with CLI runners, REST API, and dashboard.",
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Block cross-site browser POST/PUT/PATCH/DELETE (CSRF) by Origin/Referer.
    # Non-browser callers (CLI, agents, curl) send no Origin and pass through.
    # Added before RequestContextMiddleware so a blocked request is still logged.
    app.add_middleware(CsrfOriginMiddleware)
    # Assigns/propagates X-Request-ID and emits one structured access line per
    # request. Added after CORS so preflight responses still get CORS headers.
    app.add_middleware(RequestContextMiddleware)

    app.include_router(agents.router, prefix=settings.api_prefix, tags=["agents"])
    app.include_router(runs.router, prefix=settings.api_prefix, tags=["runs"])
    app.include_router(tickets.router, prefix=settings.api_prefix, tags=["tickets"])
    app.include_router(models_router.router, prefix=settings.api_prefix, tags=["models"])
    app.include_router(files_router.router, prefix=settings.api_prefix, tags=["files"])
    app.include_router(hardware.router, prefix=settings.api_prefix, tags=["hardware"])
    app.include_router(ssh_hardware.router, prefix=settings.api_prefix, tags=["ssh-hardware"])
    app.include_router(tasks.router, prefix=settings.api_prefix, tags=["tasks"])
    app.include_router(leaderboard.router, prefix=settings.api_prefix, tags=["leaderboard"])
    app.include_router(wakeups.router, prefix=settings.api_prefix, tags=["wakeups"])
    app.include_router(heartbeats.router, prefix=settings.api_prefix, tags=["heartbeats"])
    app.include_router(attachments.router, prefix=settings.api_prefix, tags=["attachments"])
    app.include_router(heartbeat_events.router, prefix=settings.api_prefix, tags=["heartbeat_events"])
    app.include_router(run_timeline.router, prefix=settings.api_prefix, tags=["run_timeline"])
    app.include_router(memory.router, prefix=settings.api_prefix, tags=["memory"])
    app.include_router(cost.router, prefix=settings.api_prefix, tags=["cost"])
    app.include_router(auth_status.router, prefix=settings.api_prefix, tags=["auth"])
    app.include_router(settings_router.router, prefix=settings.api_prefix, tags=["settings"])
    app.include_router(preflight.router, prefix=settings.api_prefix, tags=["preflight"])
    app.include_router(infra.router, prefix=settings.api_prefix, tags=["infra"])
    app.include_router(gpu_leases.router, prefix=settings.api_prefix, tags=["gpu"])
    app.include_router(failure_modes.router, prefix=settings.api_prefix, tags=["failure-modes"])
    app.include_router(ws.router, prefix=settings.api_prefix, tags=["ws"])
    return app


app = create_application()


@app.get("/")
async def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": f"{settings.api_prefix}/docs",
    }


@app.get(settings.api_prefix, include_in_schema=False)
async def api_root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": f"{settings.api_prefix}/docs",
        "openapi": f"{settings.api_prefix}/openapi.json",
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}
