"""Settings resolution and API-key gating — deterministic, no network, no key."""

import pytest

from adaptrna_agentic.settings import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    ROLES,
    Settings,
    require_api_key,
)

_MODEL_VARS = ["ADAPTRNA_MODEL"] + [f"ADAPTRNA_MODEL_{role.upper()}" for role in ROLES]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No model overrides in the environment, and a guaranteed-absent .env file."""
    for var in _MODEL_VARS + ["ADAPTRNA_MAX_TOKENS"]:
        monkeypatch.delenv(var, raising=False)

    return tmp_path / "does_not_exist.env"


def test_defaults_are_opus_for_every_role(clean_env):
    settings = Settings.from_env(env_file=clean_env)

    assert settings.models == {role: DEFAULT_MODEL for role in ROLES}
    assert settings.models == {role: "anthropic:claude-opus-5" for role in ROLES}
    assert settings.max_tokens == DEFAULT_MAX_TOKENS


def test_global_override_applies_to_every_role(clean_env, monkeypatch):
    monkeypatch.setenv("ADAPTRNA_MODEL", "anthropic:claude-sonnet-5")

    settings = Settings.from_env(env_file=clean_env)

    assert set(settings.models.values()) == {"anthropic:claude-sonnet-5"}


def test_per_role_override_beats_global(clean_env, monkeypatch):
    monkeypatch.setenv("ADAPTRNA_MODEL", "anthropic:claude-sonnet-5")
    monkeypatch.setenv("ADAPTRNA_MODEL_VERIFIER", "anthropic:claude-opus-5")

    settings = Settings.from_env(env_file=clean_env)

    assert settings.model_for("verifier") == "anthropic:claude-opus-5"
    assert settings.model_for("orchestrator") == "anthropic:claude-sonnet-5"
    assert settings.model_for("toolsmith") == "anthropic:claude-sonnet-5"


def test_dotenv_file_is_loaded_but_real_env_wins(clean_env, monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ADAPTRNA_MODEL=anthropic:claude-haiku-4-5\nADAPTRNA_MAX_TOKENS=4096\n"
    )
    monkeypatch.setenv("ADAPTRNA_MODEL", "anthropic:claude-sonnet-5")

    settings = Settings.from_env(env_file=env_file)

    # The real environment variable wins; the .env fills what the environment lacks.
    assert set(settings.models.values()) == {"anthropic:claude-sonnet-5"}
    assert settings.max_tokens == 4096


def test_unknown_role_raises_listing_valid_roles(clean_env):
    settings = Settings.from_env(env_file=clean_env)

    with pytest.raises(KeyError) as excinfo:
        settings.model_for("banana")

    for role in ROLES:
        assert role in str(excinfo.value)


def test_require_api_key_missing_is_actionable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY") as excinfo:
        require_api_key()

    # The message must carry the fix, not just the complaint.
    assert ".env" in str(excinfo.value)


def test_require_api_key_present_passes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    require_api_key()
