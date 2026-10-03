"""
Shared test helpers. pytest loads this file automatically before any test.

The key idea: tests never call a real model. They use
  * fake_llm(...)     a fake chat model that streams a fixed reply word by word
  * fake_router(...)  a fake router that returns a fixed RouteDecision
so the whole suite runs offline, instantly, and without an API key.
"""

from __future__ import annotations

import json
from itertools import cycle

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app import config
from app.chains import build_orchestrator
from app.main import create_app
from app.routing import RouteDecision


class RecordingFakeLLM(GenericFakeChatModel):
    """A fake chat model that also remembers the messages it was sent,
    so tests can check what the prompt looked like."""

    last_messages: list = []

    async def _astream(self, messages, *args, **kwargs):
        self.last_messages = messages
        async for chunk in super()._astream(messages, *args, **kwargs):
            yield chunk


def fake_llm(reply: str = "The answer is here.") -> RecordingFakeLLM:
    return RecordingFakeLLM(messages=cycle([AIMessage(content=reply)]))


def fake_router(decision, calls: list | None = None) -> RunnableLambda:
    """A router that always returns `decision`. If `decision` is an exception,
    it raises it instead. Each call's input is appended to `calls`."""

    async def route(inputs):
        if calls is not None:
            calls.append(inputs)
        if isinstance(decision, Exception):
            raise decision
        return decision

    return RunnableLambda(route)


GENERAL = RouteDecision(route="general")


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Turn a raw SSE response body into [(event_name, data_dict), ...]."""
    events, name = [], None
    for line in body.splitlines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name:
            events.append((name, json.loads(line.split(":", 1)[1].strip())))
            name = None
    return events


def make_client(llm=None, router=None) -> TestClient:
    """A test HTTP client for the app, wired to fake models."""
    orchestrator = build_orchestrator(llm or fake_llm(), router or fake_router(GENERAL))
    return TestClient(create_app(orchestrator))


@pytest.fixture(autouse=True)
def fresh_secrets_cache():
    """load_secrets() caches its result; clear it so tests don't affect each other."""
    config.load_secrets.cache_clear()
    yield
    config.load_secrets.cache_clear()
