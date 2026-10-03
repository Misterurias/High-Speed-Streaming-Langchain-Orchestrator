"""
The one place to change how the app behaves.

There are two kinds of settings, and they live in two different places on purpose:

1. CHOICES (which model, how long answers can be, request limits)
   Plain Python constants in this file. To swap models, change `MODEL` below.
   Nothing else in the codebase names a model, so this is the only edit needed.

2. SECRETS (API keys)
   Never written in code. They are read from environment variables or from a
   local `.env` file, which is git-ignored.

The browser UI also reads the limits and model name from this file, through
GET /api/config, so the front end never hard-codes them either.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

# Written as "provider:model-name", the format LangChain's `init_chat_model`
# understands. Supported providers: "anthropic" and "openai".
#   Examples:  "anthropic:claude-haiku-4-5"   "openai:gpt-4o-mini"
MODEL = "anthropic:claude-haiku-4-5"


@dataclass(frozen=True)
class ModelRole:
    """How one job in the pipeline uses a model."""

    model: str
    temperature: float  # 0 = always the same answer, higher = more varied
    max_tokens: int     # upper limit on the length of the model's reply


# The app calls a model for two different jobs. Both use MODEL by default;
# point either one at a different model string to split them (for example, a
# smaller, faster model just for routing).
ROUTER = ModelRole(
    model=MODEL,
    temperature=0.0,  # classification should be deterministic
    max_tokens=200,   # it only returns {"route": ..., "expression": ...}
)
ANSWERER = ModelRole(
    model=MODEL,
    temperature=0.3,
    max_tokens=1024,
)

LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 2

# ─────────────────────────────────────────────────────────────────────────────
# Request limits (enforced by the API, and shown to the UI)
# ─────────────────────────────────────────────────────────────────────────────

MAX_QUERY_CHARS = 2000    # longest question a user can send
MAX_HISTORY_TURNS = 20    # how many earlier messages are sent back as context
MAX_TURN_CHARS = 8000     # longest single earlier message

# ─────────────────────────────────────────────────────────────────────────────
# Secrets
# ─────────────────────────────────────────────────────────────────────────────

# Which environment variable holds the key for each provider.
PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class MissingApiKeyError(RuntimeError):
    """Raised at startup when a model is configured but its key isn't set."""


class Secrets(BaseSettings):
    """API keys, read from the environment or `.env`. Field names map to
    environment variables case-insensitively (anthropic_api_key ← ANTHROPIC_API_KEY).
    SecretStr hides the value if the object is ever printed or logged."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    def key_for(self, model: str) -> str:
        """Return the API key for a "provider:model" string, or raise a clear error."""
        provider = provider_of(model)
        env_var = PROVIDER_KEY_ENV[provider]
        secret: SecretStr | None = getattr(self, env_var.lower())
        if secret is None or not secret.get_secret_value().strip():
            raise MissingApiKeyError(
                f"{env_var} is not set, but model '{model}' needs it. "
                f"Add it to your .env file (see .env.example)."
            )
        return secret.get_secret_value()


def provider_of(model: str) -> str:
    """'anthropic:claude-haiku-4-5' -> 'anthropic'. Rejects unknown providers early."""
    provider, sep, name = model.partition(":")
    if not sep or not name or provider not in PROVIDER_KEY_ENV:
        raise ValueError(
            f"Model '{model}' must look like 'provider:model-name' with provider "
            f"one of {sorted(PROVIDER_KEY_ENV)}."
        )
    return provider


@lru_cache
def load_secrets() -> Secrets:
    """Read the keys once, and fail at startup (not on the first user request)
    if a key needed by ROUTER or ANSWERER is missing."""
    secrets = Secrets()
    for role in (ROUTER, ANSWERER):
        secrets.key_for(role.model)
    return secrets


def public_config() -> dict:
    """The non-secret settings the browser needs. Served at GET /api/config."""
    return {
        "model": ANSWERER.model.partition(":")[2],
        "router_model": ROUTER.model.partition(":")[2],
        "max_query_chars": MAX_QUERY_CHARS,
        "max_history_turns": MAX_HISTORY_TURNS,
    }
