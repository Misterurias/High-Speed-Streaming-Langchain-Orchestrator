"""
Turns the model choices in config.py into real LangChain chat-model objects.

This is the only file that talks to `init_chat_model`. Everything else receives
ready-made model objects, which is what lets the tests pass in fake models
and never need an API key.
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from app import config
from app.chains import build_orchestrator
from app.routing import build_llm_router


def build_chat_model(role: config.ModelRole, secrets: config.Secrets) -> BaseChatModel:
    """Create one chat model for a role (router or answerer).

    `init_chat_model` reads the provider from the "provider:model" string, so
    the same code builds an Anthropic or an OpenAI client.
    """
    return init_chat_model(
        role.model,
        api_key=secrets.key_for(role.model),
        temperature=role.temperature,
        max_tokens=role.max_tokens,
        timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=config.LLM_MAX_RETRIES,
    )


def build_orchestrator_from_config() -> Runnable:
    """Build the full pipeline with the models named in config.py.

    Called once when the server starts. The model clients are reused for every
    request, so each request doesn't pay to open a new connection.
    """
    secrets = config.load_secrets()
    router_llm = build_chat_model(config.ROUTER, secrets)
    answer_llm = build_chat_model(config.ANSWERER, secrets)
    return build_orchestrator(answer_llm, build_llm_router(router_llm))
