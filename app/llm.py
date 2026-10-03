"""Builds the real chat models from Settings. Kept separate so tests can
swap in fake models without touching any API keys."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from app.chains import build_llm_router, build_orchestrator
from app.config import Settings


def _make_model(settings: Settings, model: str, **overrides) -> BaseChatModel:
    return init_chat_model(
        model,
        api_key=settings.api_key_for(model).get_secret_value(),
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        **overrides,
    )


def build_from_settings(settings: Settings) -> Runnable:
    # Clients are created once at startup and reused across requests
    # (connection pooling instead of a new HTTP client per call).
    answer_llm = _make_model(
        settings, settings.llm_model, temperature=0.3, max_tokens=settings.answer_max_tokens
    )
    router_llm = _make_model(
        settings, settings.effective_router_model, temperature=0, max_tokens=200
    )
    return build_orchestrator(answer_llm, build_llm_router(router_llm))
