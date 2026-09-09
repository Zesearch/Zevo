"""The CSRF guard must block cross-site browser mutations while leaving every
non-browser caller (CLI, agents, curl) and the dashboard itself untouched.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zevo.api.csrf import CsrfOriginMiddleware


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(CsrfOriginMiddleware)

    @app.get("/thing")
    async def get_thing():
        return {"ok": True}

    @app.post("/thing")
    async def post_thing():
        return {"ok": True}

    return app


@pytest.fixture()
def client():
    return TestClient(_app())


def test_get_is_never_blocked(client):
    # Even a cross-site Origin is fine on a safe method.
    r = client.get("/thing", headers={"origin": "https://evil.example"})
    assert r.status_code == 200


def test_post_with_no_origin_allowed(client):
    # CLI / agent / curl: no Origin, no Referer.
    r = client.post("/thing")
    assert r.status_code == 200


def test_post_cross_site_origin_blocked(client):
    r = client.post("/thing", headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    assert "CSRF" in r.json()["detail"]


def test_post_cross_site_via_referer_blocked(client):
    r = client.post("/thing", headers={"referer": "https://evil.example/attack.html"})
    assert r.status_code == 403


def test_post_same_origin_allowed(client):
    # Dashboard talking to its own /api: Origin netloc == Host.
    r = client.post(
        "/thing",
        headers={"origin": "http://testserver", "host": "testserver"},
    )
    assert r.status_code == 200


def test_post_allowlisted_origin_allowed(client, monkeypatch):
    # An origin in the CORS allow-list is fine even if not same-origin.
    r = client.post("/thing", headers={"origin": "http://localhost:5173"})
    assert r.status_code == 200


def test_disable_escape_hatch(monkeypatch):
    monkeypatch.setenv("ZEVO_DISABLE_CSRF_GUARD", "1")
    c = TestClient(_app())
    r = c.post("/thing", headers={"origin": "https://evil.example"})
    assert r.status_code == 200
