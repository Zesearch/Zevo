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
    assert "AWS_BEARER_TOKEN_BEDROCK" not in PLAIN_KEYS
    # non-secret config that is intentionally shown in full
    assert "AWS_REGION" in PLAIN_KEYS
