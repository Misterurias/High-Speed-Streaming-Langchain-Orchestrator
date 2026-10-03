"""Tests for app/config.py and app/llm.py: settings, secrets, and building models."""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config
from app.config import MissingApiKeyError, Secrets, provider_of
from app.llm import build_chat_model, build_orchestrator_from_config

APP_DIR = Path(__file__).resolve().parent.parent / "app"


@pytest.fixture
def no_env_keys(monkeypatch, tmp_path):
    """Run with no API keys anywhere: none in the environment, and a working
    directory with no .env file (so your real .env isn't picked up)."""
    for var in config.PROVIDER_KEY_ENV.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


class TestOneSourceOfTruth:
    def test_no_model_name_appears_outside_config_py(self):
        """Swapping models must only require editing config.py."""
        model_name = re.compile(r"claude-[a-z0-9.-]+|gpt-[0-9a-z.-]+", re.IGNORECASE)
        offenders = [
            f"{path.relative_to(APP_DIR)}: {match}"
            for path in APP_DIR.rglob("*")
            if path.suffix in {".py", ".js", ".html", ".css"}
            and "vendor" not in path.parts
            and path.name != "config.py"
            for match in model_name.findall(path.read_text())
        ]
        assert offenders == []

    def test_both_roles_use_the_shared_model_by_default(self):
        assert config.ROUTER.model == config.MODEL
        assert config.ANSWERER.model == config.MODEL

    def test_public_config_comes_straight_from_the_constants(self):
        public = config.public_config()
        assert public["max_query_chars"] == config.MAX_QUERY_CHARS
        assert public["max_history_turns"] == config.MAX_HISTORY_TURNS
        assert config.ANSWERER.model.endswith(public["model"])


class TestModelStrings:
    @pytest.mark.parametrize("model, provider", [("anthropic:x", "anthropic"), ("openai:y", "openai")])
    def test_provider_is_read_from_the_prefix(self, model, provider):
        assert provider_of(model) == provider

    @pytest.mark.parametrize("bad", ["claude-haiku", "google:gemini", "anthropic:", ":x"])
    def test_malformed_or_unsupported_models_are_rejected(self, bad):
        with pytest.raises(ValueError):
            provider_of(bad)


class TestSecrets:
    def test_key_is_returned_for_the_matching_provider(self):
        secrets = Secrets(_env_file=None, anthropic_api_key="sk-ant", openai_api_key="sk-oai")
        assert secrets.key_for("anthropic:m") == "sk-ant"
        assert secrets.key_for("openai:m") == "sk-oai"

    @pytest.mark.parametrize("value", [None, "", "   "], ids=["unset", "empty", "whitespace"])
    def test_missing_key_gives_a_clear_error(self, value, no_env_keys):
        secrets = Secrets(_env_file=None, anthropic_api_key=value)
        with pytest.raises(MissingApiKeyError, match="ANTHROPIC_API_KEY"):
            secrets.key_for("anthropic:m")

    def test_keys_are_hidden_when_printed(self):
        secrets = Secrets(_env_file=None, anthropic_api_key="sk-very-secret")
        assert "sk-very-secret" not in repr(secrets)
        assert "sk-very-secret" not in str(secrets)

    def test_startup_fails_fast_when_a_needed_key_is_missing(self, no_env_keys):
        with pytest.raises(MissingApiKeyError):
            config.load_secrets()

    def test_keys_are_read_from_a_dot_env_file(self, no_env_keys, tmp_path):
        (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-from-dotenv\n")
        assert config.load_secrets().key_for("anthropic:m") == "sk-from-dotenv"


class TestBuildingModels:
    """Creating a client doesn't contact the provider, so no network is needed."""

    @pytest.mark.parametrize(
        "model, class_name",
        [("anthropic:claude-test", "ChatAnthropic"), ("openai:gpt-test", "ChatOpenAI")],
    )
    def test_role_settings_are_applied(self, model, class_name):
        secrets = Secrets(_env_file=None, anthropic_api_key="sk-a", openai_api_key="sk-o")
        role = config.ModelRole(model=model, temperature=0.0, max_tokens=123)
        llm = build_chat_model(role, secrets)
        assert type(llm).__name__ == class_name
        assert llm.temperature == 0.0
        assert llm.max_tokens == 123

    def test_full_pipeline_builds_from_config(self, no_env_keys, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert build_orchestrator_from_config() is not None

    def test_server_startup_builds_models_from_config(self, no_env_keys, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from app.main import create_app

        with TestClient(create_app()) as client:  # runs the real startup code
            assert client.app.state.orchestrator is not None
