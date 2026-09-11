"""An uploaded private key is stored by Zevo (0600, under its credential
root) and the connection points at that copy, so the browser need not be on
the host that holds ~/.ssh."""
import os

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.ui import ssh_hardware
from zevo.api.routers.ui.ssh_hardware import (
    CreateSshHostBody, UpdateSshHostBody, create_ssh_host, update_ssh_host,
)
from zevo.db import Base, SshHost

KEY = "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----\n"


def _body(**over):
    base = dict(
        name="Uploaded box", category="instance", host="gpu.example.org",
        username="researcher", remote_parent_dir="/srv/work",
        env_setup="conda activate zevo", private_key=KEY,
    )
    base.update(over)
    return CreateSshHostBody(**base)


def test_uploaded_key_counts_as_the_one_credential():
    _body()
    with pytest.raises(ValidationError):
        _body(password="also")
    with pytest.raises(ValidationError):
        _body(private_key="")


def _isolate(tmp_path, monkeypatch):
    async def verified(**_kwargs):
        return {"ok": True, "error": ""}
    monkeypatch.setattr(ssh_hardware, "verify_ssh_workspace", verified)
    monkeypatch.setattr(ssh_hardware, "SSH_CREDENTIAL_ROOT", tmp_path / "ssh-connections")
    monkeypatch.setattr(ssh_hardware, "INFRA_SKILLS_ROOT", tmp_path / "skills")
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(ssh_hardware, "ENV_PATH", env_path)


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.mark.asyncio
async def test_create_stores_uploaded_key_0600_and_reports_it(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    Session = await _session()
    async with Session() as db:
        dto = await create_ssh_host(_body(), db)
        row = (await db.execute(select(SshHost).where(SshHost.id == dto.id))).scalar_one()
    assert dto.private_key_uploaded is True
    assert dto.authentication == "private_key"
    assert row.key_path.startswith(str(tmp_path / "ssh-connections"))
    assert open(row.key_path, encoding="utf-8").read() == KEY
    assert os.stat(row.key_path).st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_update_with_uploaded_key_replaces_password(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    Session = await _session()
    async with Session() as db:
        created = await create_ssh_host(_body(private_key="", password="pw"), db)
        row = (await db.execute(select(SshHost).where(SshHost.id == created.id))).scalar_one()
        password_path = row.password_path
        assert os.path.exists(password_path)
        dto = await update_ssh_host(created.id, UpdateSshHostBody(
            name="Uploaded box", category="instance", host="gpu.example.org",
            username="researcher", remote_parent_dir="/srv/work",
            env_setup="conda activate zevo", private_key=KEY,
        ), db)
        row = (await db.execute(select(SshHost).where(SshHost.id == created.id))).scalar_one()
    assert dto.private_key_uploaded is True and dto.authentication == "private_key"
    assert row.password_path == "" and not os.path.exists(password_path)
    assert os.stat(row.key_path).st_mode & 0o777 == 0o600
