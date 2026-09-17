"""Credential endpoints must report PRESENCE only -- never any slice of a
secret. A first-6/last-4 redaction still leaks entropy and confirms a key's
shape, so /auth-status and /settings must return no preview for real secrets.
"""
from __future__ import annotations

import pytest

from zevo.api.routers.ui.auth_status import _env_cred


def test_env_cred_never_previews_secret(monkeypatch) -> None:
    secret = "sk-ant-oat01-SUPERSECRETVALUE-do-not-leak-0123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    cred = _env_cred("ANTHROPIC_API_KEY")
    assert cred.present is True
    assert cred.preview == ""            # no redaction, no chars
    # No contiguous run of the secret may appear anywhere on the object.
    blob = cred.model_dump_json()
    for i in range(0, len(secret) - 5):
        assert secret[i:i + 6] not in blob


def test_settings_secret_entry_has_no_preview() -> None:
    # The settings GET builds SecretEntry with preview="" for non-PLAIN keys.
    # Assert the allowlist logic: only PLAIN_KEYS may echo their value.
    from zevo.api.routers.ui.settings import PLAIN_KEYS
    assert "ANTHROPIC_API_KEY" not in PLAIN_KEYS
    assert "OPENAI_API_KEY" not in PLAIN_KEYS
    assert "GOOGLE_CLOUD_VERTEX_API_KEY" not in PLAIN_KEYS
    assert "AWS_BEARER_TOKEN_BEDROCK" not in PLAIN_KEYS
    # non-secret config that is intentionally shown in full
    assert "AWS_REGION" in PLAIN_KEYS
    assert "ZEVO_DEFAULT_COMPUTE" in PLAIN_KEYS
    assert "ZEVO_EVALUATION_JUDGE_PROVIDER" in PLAIN_KEYS


def test_default_compute_setting_names_one_concrete_picker_target() -> None:
    from zevo.api.routers.ui.settings import ALLOWED_KEYS, KEY_FORMATS

    key = "ZEVO_DEFAULT_COMPUTE"
    assert key in ALLOWED_KEYS
    pattern = KEY_FORMATS[key]
    assert pattern.fullmatch("cloud:vastai")
    assert pattern.fullmatch("cloud:lambda")
    assert pattern.fullmatch("environment:cluster")
    assert pattern.fullmatch("connection:1843b8b3-6090-4221-b318-5f58299e9104")
    assert not pattern.fullmatch("cluster")
    assert not pattern.fullmatch("connection:")


@pytest.mark.asyncio
async def test_vertex_judge_settings_accept_express_key_without_echoing_it(tmp_path, monkeypatch) -> None:
    from zevo.api.routers.ui import settings as settings_module

    monkeypatch.setattr(settings_module, "ENV_PATH", tmp_path / ".env")
    key = "AIza" + "A" * 35
    result = await settings_module.set_secrets(settings_module.SetSecretsBody(values={
        "ZEVO_EVALUATION_JUDGE_PROVIDER": "vertex_ai",
        "GOOGLE_CLOUD_VERTEX_API_KEY": key,
    }))
    assert set(result.updated) == {"ZEVO_EVALUATION_JUDGE_PROVIDER", "GOOGLE_CLOUD_VERTEX_API_KEY"}
    assert key not in result.model_dump_json()
    assert settings_module._read_env(tmp_path / ".env")["GOOGLE_CLOUD_VERTEX_API_KEY"] == key
