"""Configuration, loaded only from environment variables or a local .env file.

No secret ever lives in the code. Keys are stored as SecretStr so they are
masked if settings are ever logged or printed.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Which environment variable holds the key for each provider prefix.
_PROVIDER_KEY_FIELDS = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # "provider:model" strings understood by LangChain's init_chat_model.
    llm_model: str = "anthropic:claude-haiku-4-5"
    # The router only classifies, so it can use a smaller/faster model.
    router_model: str | None = None

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    answer_max_tokens: int = 1024

    @property
    def effective_router_model(self) -> str:
        return self.router_model or self.llm_model

    def api_key_for(self, model: str) -> SecretStr:
        provider = model.split(":", 1)[0] if ":" in model else ""
        field = _PROVIDER_KEY_FIELDS.get(provider)
        if field is None:
            raise ValueError(
                f"Model '{model}' must be written as 'provider:model' with provider "
                f"one of {sorted(_PROVIDER_KEY_FIELDS)}"
            )
        key = getattr(self, field)
        if key is None or not key.get_secret_value().strip():
            raise ValueError(f"{field.upper()} is not set (needed for model '{model}')")
        return key

    @model_validator(mode="after")
    def _require_keys(self) -> "Settings":
        # Fail fast at startup instead of on the first user request.
        self.api_key_for(self.llm_model)
        self.api_key_for(self.effective_router_model)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
