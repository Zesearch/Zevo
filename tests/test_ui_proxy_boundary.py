from pathlib import Path

import pytest
import yaml
from starlette.datastructures import Headers

from zevo.api import ui_access


@pytest.mark.parametrize("worker_header", ["x-zevo-worker", "x-zevo-service"])
def test_proxy_token_never_upgrades_worker_identity(monkeypatch, worker_header):
    monkeypatch.setattr(ui_access, "_ui_access_token", lambda: "ui-secret")
    assert ui_access._trusted(Headers({"x-zevo-ui-access": "ui-secret"}))
    assert not ui_access._trusted(Headers({"x-zevo-ui-access": "ui-secret", worker_header: "worker-token"}))


def test_execution_services_cannot_directly_reach_dashboard_proxy():
    compose = yaml.safe_load((Path(__file__).parents[1] / "docker-compose.yml").read_text())
    services = compose["services"]
    dashboard = set(services["web"]["networks"])
    for name in ("scheduler", "holdout-scheduler"):
        assert not dashboard.intersection(services[name]["networks"])
    assert dashboard.intersection(services["backend"]["networks"])
    assert set(services["scheduler"]["networks"]).intersection(services["backend"]["networks"])
