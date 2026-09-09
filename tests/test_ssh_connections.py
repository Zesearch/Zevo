"""Launch-time guards for Cluster/Instance SSH connections.

Covers the create_run validation that a run targeting a selected SSH connection
must name a connection and that connection must be verified. These guards
raise before any pipeline work, so no run/ticket setup is needed.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.hardware.ssh_verify import verify_ssh_workspace
from zevo.api.routers.shared.runs import CreateRunRequest, create_run
from zevo.api.routers.ui import ssh_hardware
from zevo.api.routers.ui.ssh_hardware import (
    CreateSshHostBody,
    UpdateSshHostBody,
    _connection_env_prefix,
    _connection_env_values,
    _host_private_key_path,
    _infra_skill_display_path,
    _read_infra_skill_instructions,
    _remove_managed_credential,
    _remove_infra_skill,
    _remote_zevo_dir,
    _resolve_host_private_key_path,
    _write_infra_skill,
    _write_password,
    update_ssh_host,
)
from zevo.db.models import Agent, Base, SshHost, Task


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


def _seed(db) -> None:
    db.add(Agent(
        id="orchestrator", name="Orchestrator", title="Supervisor",
        identity_path="playbook/agents/orchestrator/identity.md",
    ))
    db.add(Task(
        metric="accuracy", name="t", task_objective="Improve the model.",
        metric_direction="max", test_set="/data/test.csv",
        test_answer_fields=["answer"], test_sample_submission="/data/sample.csv",
    ))


def test_ssh_and_local_are_not_gpu_provider_categories() -> None:
    for removed in ("ssh", "local"):
        with pytest.raises(ValidationError):
            CreateRunRequest(task_name="t", run_name="r", gpu_provider=removed)


def test_connection_contract_uses_host_key_path_without_copying(
    tmp_path, monkeypatch,
) -> None:
    host_ssh_root = tmp_path / "host-home" / ".ssh"
    host_ssh_root.mkdir(parents=True)
    key_path = host_ssh_root / "zevo_cluster_ed25519"
    key_path.write_text("test private key", encoding="utf-8")
    monkeypatch.setattr(ssh_hardware, "HOST_SSH_ROOT", host_ssh_root)
    monkeypatch.setattr(ssh_hardware, "CONTAINER_SSH_ROOT", tmp_path / "container-ssh")

    body = CreateSshHostBody(
        name="Research cluster",
        category="cluster",
        host="login.example.org",
        port=22,
        username="researcher",
        private_key_path=str(key_path),
        remote_parent_dir="/scratch/researcher",
        env_setup="conda activate zevo",
    )
    assert body.category == "cluster"
    assert _remote_zevo_dir(body.remote_parent_dir) == "/scratch/researcher/zevo"
    assert _remote_zevo_dir("/scratch/researcher/zevo") == "/scratch/researcher/zevo"

    resolved_key_path = _resolve_host_private_key_path(body.private_key_path)
    assert resolved_key_path == str(key_path.resolve())
    assert key_path.read_text(encoding="utf-8") == "test private key"

    row = SshHost(
        label=body.name,
        category=body.category,
        host=body.host,
        port=body.port,
        username=body.username,
        key_path=resolved_key_path,
        remote_dir=_remote_zevo_dir(body.remote_parent_dir),
        env_setup=body.env_setup,
    )
    assert row.host == "login.example.org"
    assert row.key_path == resolved_key_path
    assert row.remote_dir == "/scratch/researcher/zevo"
    assert row.env_setup == "conda activate zevo"


def test_password_connection_stores_only_a_mode_0600_password_file(tmp_path) -> None:
    body = CreateSshHostBody(
        name="Password box",
        category="instance",
        host="gpu.example.org",
        username="researcher",
        password="correct horse battery staple",
        remote_parent_dir="/srv/work",
        env_setup="conda activate zevo",
    )
    password_path = tmp_path / "password-box" / "password"
    _write_password(password_path, body.password)
    assert password_path.read_text(encoding="utf-8") == body.password + "\n"
    assert os.stat(password_path).st_mode & 0o777 == 0o600

    row = SshHost(
        label=body.name,
        category=body.category,
        host=body.host,
        port=body.port,
        username=body.username,
        key_path="",
        password_path=str(password_path),
        remote_dir="/srv/work/zevo",
        env_setup=body.env_setup,
    )
    assert row.key_path == ""
    assert row.password_path == str(password_path)


def test_connection_materializes_and_updates_a_private_infrastructure_skill(
    tmp_path, monkeypatch,
) -> None:
    skill_root = tmp_path / "playbook" / "skills" / "infrastructure"
    monkeypatch.setattr(ssh_hardware, "INFRA_SKILLS_ROOT", skill_root)
    row = SshHost(
        id="beta-connection-id",
        label="Research Beta",
        category="cluster",
        host="beta.example.org",
        port=22,
        username="researcher",
        key_path="/root/.ssh/id_ed25519",
        remote_dir="/projects/team/zevo",
        env_setup="source /projects/team/env/bin/activate",
    )

    instructions = "Use the b200 partition. Request at least four GPUs."
    uploaded = f"""---
name: research-beta
description: Site rules for Research Beta.
---

{instructions}
"""
    _write_infra_skill(row, uploaded)
    path = skill_root / "Research-Beta" / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    assert "name: Research-Beta" in text
    assert "SSH host is beta.example.org" in text
    assert "Treat this resource as a Slurm cluster" in text
    assert instructions in text
    assert row.username not in text
    assert row.remote_dir not in text
    assert _read_infra_skill_instructions(row) == instructions
    assert _infra_skill_display_path(row).endswith(
        "/Research-Beta/SKILL.md"
    )

    # A connection rename regenerates matching metadata, retains the
    # operator-authored section, and moves the directory to the new name.
    row.label = "Research Beta renamed"
    _write_infra_skill(row, previous_path=path)
    renamed_path = skill_root / "Research-Beta-renamed" / "SKILL.md"
    updated = renamed_path.read_text(encoding="utf-8")
    assert "# Research Beta renamed" in updated
    assert instructions in updated
    assert not path.exists()

    _remove_infra_skill(row)
    assert not renamed_path.exists()
    _write_infra_skill(row)
    assert not renamed_path.exists(), "verification must not invent an empty site Skill"


def test_connection_env_values_contain_paths_but_never_credentials() -> None:
    row = SshHost(
        id="my-instance-id",
        label="My instance",
        category="instance",
        host="gpu.example.org",
        port=2222,
        username="researcher",
        key_path="",
        password_path="/app/data/ssh-connections/my-instance/password",
        remote_dir="/srv/work/zevo",
        env_setup="conda activate zevo",
    )
    values = _connection_env_values(row)
    prefix = _connection_env_prefix("my-instance-id")
    assert values == {
        f"{prefix}_ID": "my-instance-id",
        f"{prefix}_NAME": "My instance",
        f"{prefix}_CATEGORY": "instance",
        f"{prefix}_SSH_HOST": "gpu.example.org",
        f"{prefix}_SSH_PORT": "2222",
        f"{prefix}_SSH_USER": "researcher",
        f"{prefix}_SSH_KEY": "",
        f"{prefix}_SSH_PASSWORD_FILE": (
            "/app/data/ssh-connections/my-instance/password"
        ),
        f"{prefix}_REMOTE_DIR": "/srv/work/zevo",
        f"{prefix}_ENV_SETUP": "conda activate zevo",
        f"{prefix}_CONTAINER_IMAGE": "",
    }


def test_same_category_connections_have_independent_env_blocks() -> None:
    rows = [
        SshHost(
            id=connection_id,
            label=label,
            category="cluster",
            host=host,
            port=22,
            username="researcher",
            key_path="/root/.ssh/id_ed25519",
            remote_dir=f"/projects/{connection_id}/zevo",
            env_setup="source /projects/env/bin/activate",
        )
        for connection_id, label, host in (
            ("alpha-id", "Cluster Alpha", "alpha.example.org"),
            ("beta-id", "Cluster Beta", "beta.example.org"),
        )
    ]
    alpha = _connection_env_values(rows[0])
    beta = _connection_env_values(rows[1])
    assert set(alpha).isdisjoint(beta)
    assert next(v for k, v in alpha.items() if k.endswith("_SSH_HOST")) == "alpha.example.org"
    assert next(v for k, v in beta.items() if k.endswith("_SSH_HOST")) == "beta.example.org"


def test_connection_rejects_both_or_neither_credential() -> None:
    base = {
        "name": "box",
        "category": "instance",
        "host": "gpu.example.org",
        "username": "researcher",
        "remote_parent_dir": "/srv/work",
        "env_setup": "conda activate zevo",
    }
    with pytest.raises(ValidationError):
        CreateSshHostBody(**base)
    with pytest.raises(ValidationError):
        CreateSshHostBody(
            **base,
            private_key_path="/Users/me/.ssh/id_ed25519",
            password="secret",
        )

    update = UpdateSshHostBody(**base)
    assert update.private_key_path == ""
    assert update.password == ""
    with pytest.raises(ValidationError):
        UpdateSshHostBody(
            **base,
            private_key_path="/Users/me/.ssh/id_ed25519",
            password="secret",
        )


def test_host_key_path_maps_to_read_only_container_mount(tmp_path, monkeypatch) -> None:
    container_root = tmp_path / "container-ssh"
    container_root.mkdir()
    mapped_key = container_root / "zevo_cluster_ed25519"
    mapped_key.write_text("test private key", encoding="utf-8")
    monkeypatch.setattr(
        ssh_hardware,
        "HOST_SSH_ROOT",
        Path("/Users/researcher/.ssh"),
    )
    monkeypatch.setattr(ssh_hardware, "CONTAINER_SSH_ROOT", container_root)

    resolved = _resolve_host_private_key_path(
        "/Users/researcher/.ssh/zevo_cluster_ed25519"
    )
    assert resolved == str(mapped_key.resolve())
    assert _host_private_key_path(resolved) == (
        "/Users/researcher/.ssh/zevo_cluster_ed25519"
    )


def test_host_key_path_rejects_public_key_and_paths_outside_ssh(
    tmp_path, monkeypatch,
) -> None:
    host_ssh_root = tmp_path / "home" / ".ssh"
    host_ssh_root.mkdir(parents=True)
    public_key = host_ssh_root / "id_ed25519.pub"
    public_key.write_text("ssh-ed25519 test", encoding="utf-8")
    monkeypatch.setattr(ssh_hardware, "HOST_SSH_ROOT", host_ssh_root)
    monkeypatch.setattr(ssh_hardware, "CONTAINER_SSH_ROOT", tmp_path / "container-ssh")

    with pytest.raises(HTTPException):
        _resolve_host_private_key_path(str(public_key))
    with pytest.raises(HTTPException):
        _resolve_host_private_key_path(str(tmp_path / "outside-key"))


def test_managed_credential_cleanup_never_deletes_host_key(
    tmp_path, monkeypatch,
) -> None:
    managed_root = tmp_path / "managed"
    managed_password = managed_root / "connection" / "password"
    managed_password.parent.mkdir(parents=True)
    managed_password.write_text("secret\n", encoding="utf-8")
    host_key = tmp_path / "host-home" / ".ssh" / "id_ed25519"
    host_key.parent.mkdir(parents=True)
    host_key.write_text("private key", encoding="utf-8")
    monkeypatch.setattr(ssh_hardware, "SSH_CREDENTIAL_ROOT", managed_root)

    _remove_managed_credential(str(host_key))
    _remove_managed_credential(str(managed_password))

    assert host_key.is_file()
    assert not managed_password.exists()


@pytest.mark.asyncio
async def test_edit_updates_connection_and_retains_existing_credential(
    tmp_path, monkeypatch,
) -> None:
    key_path = tmp_path / "id_ed25519"
    key_path.write_text("private key", encoding="utf-8")

    async def verified(**_kwargs):
        return {"ok": True, "error": ""}

    monkeypatch.setattr(ssh_hardware, "verify_ssh_workspace", verified)
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(ssh_hardware, "ENV_PATH", env_path)
    monkeypatch.setattr(
        ssh_hardware,
        "INFRA_SKILLS_ROOT",
        tmp_path / "playbook" / "skills" / "infrastructure",
    )
    Session = await _session()
    async with Session() as db:
        db.add(SshHost(
            id="edit-me",
            label="Old name",
            category="instance",
            host="old.example.org",
            port=22,
            username="old-user",
            key_path=str(key_path),
            remote_dir="/old/zevo",
            env_setup="old-env",
            container_image="/shared/runtime.sqsh",
            status="verified",
        ))
        await db.commit()

        result = await update_ssh_host(
            "edit-me",
            UpdateSshHostBody(
                name="New name",
                category="cluster",
                host="new.example.org",
                port=2222,
                username="new-user",
                remote_parent_dir="/new/work",
                env_setup="conda activate new-env",
            ),
            db,
        )

        row = await db.get(SshHost, "edit-me")
        assert row is not None
        assert row.key_path == str(key_path)
        assert row.password_path == ""
        assert row.remote_dir == "/new/work/zevo"
        assert row.container_image == "/shared/runtime.sqsh"
        assert result.name == "New name"
        assert result.category == "cluster"
        assert result.status == "verified"
        env = env_path.read_text(encoding="utf-8")
        prefix = _connection_env_prefix("edit-me")
        assert f'{prefix}_NAME="New name"' in env
        assert f"{prefix}_CATEGORY=cluster" in env
        assert f"{prefix}_SSH_HOST=new.example.org" in env
        assert f"{prefix}_SSH_KEY={key_path}" in env


@pytest.mark.asyncio
async def test_workspace_verification_checks_all_three_steps(tmp_path, monkeypatch) -> None:
    key_path = tmp_path / "id_ed25519"
    key_path.write_text("test key", encoding="utf-8")
    captured: list[str] = []

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"zevo-ssh-workspace-ok", b""

    async def fake_exec(*args, **_kwargs):
        captured.extend(str(arg) for arg in args)
        return FakeProcess()

    monkeypatch.setattr(
        "zevo.api.hardware.ssh_verify.asyncio.create_subprocess_exec",
        fake_exec,
    )
    result = await verify_ssh_workspace(
        host="login.example.org",
        port=22,
        username="researcher",
        key_path=str(key_path),
        remote_dir="/scratch/researcher/zevo",
        env_setup="conda activate zevo",
    )
    assert result == {"ok": True, "error": ""}
    remote_command = captured[-1]
    assert "mkdir -p -- /scratch/researcher/zevo" in remote_command
    assert "conda activate zevo" in remote_command
    assert "test -d /scratch/researcher/zevo" in remote_command


@pytest.mark.asyncio
async def test_password_verification_uses_password_file_not_password_in_argv(
    tmp_path, monkeypatch,
) -> None:
    password_path = tmp_path / "password"
    password_path.write_text("top-secret-password\n", encoding="utf-8")
    captured: list[str] = []

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"zevo-ssh-workspace-ok", b""

    async def fake_exec(*args, **_kwargs):
        captured.extend(str(arg) for arg in args)
        return FakeProcess()

    monkeypatch.setattr(
        "zevo.api.hardware.ssh_verify.asyncio.create_subprocess_exec",
        fake_exec,
    )
    result = await verify_ssh_workspace(
        host="gpu.example.org",
        port=22,
        username="researcher",
        key_path="",
        password_path=str(password_path),
        remote_dir="/srv/work/zevo",
        env_setup="conda activate zevo",
    )
    assert result == {"ok": True, "error": ""}
    assert captured[:4] == ["sshpass", "-f", str(password_path), "ssh"]
    assert "top-secret-password" not in captured


@pytest.mark.asyncio
async def test_ssh_launch_rejects_an_unverified_connection() -> None:
    Session = await _session()
    async with Session() as db:
        _seed(db)
        db.add(SshHost(
            id="h1", label="box", category="instance", host="h.example",
            username="u", status="unverified",
        ))
        await db.commit()
        with pytest.raises(HTTPException) as ei:
            await create_run(
                CreateRunRequest(task_name="t", run_name="r",
                                 gpu_provider="instance", ssh_host_id="h1"),
                db,
            )
        assert ei.value.status_code == 422


@pytest.mark.asyncio
async def test_ssh_launch_rejects_a_missing_connection_id() -> None:
    Session = await _session()
    async with Session() as db:
        _seed(db)
        await db.commit()
        with pytest.raises(HTTPException) as ei:
            await create_run(
                CreateRunRequest(task_name="t", run_name="r",
                                 gpu_provider="instance", ssh_host_id="does-not-exist"),
            db,
        )
        assert ei.value.status_code == 422


@pytest.mark.asyncio
async def test_launch_rejects_connection_category_mismatch() -> None:
    Session = await _session()
    async with Session() as db:
        _seed(db)
        db.add(SshHost(
            id="h1", label="cluster", category="cluster", host="h.example",
            username="u", status="verified", key_path="/k",
            remote_dir="/scratch/zevo", env_setup="conda activate zevo",
        ))
        await db.commit()
        with pytest.raises(HTTPException) as ei:
            await create_run(
                CreateRunRequest(
                    task_name="t", run_name="r", gpu_provider="instance",
                    ssh_host_id="h1",
                ),
                db,
            )
        assert ei.value.status_code == 422
