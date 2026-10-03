"""FastAPI app: a chat UI at GET / and POST /ask, which streams the answer as
Server-Sent Events.

Request body:
    {"query": "...", "history": [{"role": "user"|"assistant", "content": "..."}, ...]}
"history" is optional; {"query": "..."} alone works.

Event stream for one request:
    event: route   data: {"route": "math", "routed_by": "llm", "expression": "...", "result": "..."}
    event: token   data: {"text": "The answer"}        (many of these, as the LLM generates)
    event: done    data: {"request_id": "...", "route": "math", "ttft_ms": 412, "total_ms": 980}
or, if something fails mid-stream:
    event: error   data: {"request_id": "...", "message": "..."}
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from app.chains import ANSWER_TAG, ROUTER_RUN_NAME, Plan

logger = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).parent / "static"
MAX_QUERY_CHARS = 2000
MAX_TURN_CHARS = 8000
MAX_HISTORY_TURNS = 20  # older turns are dropped by the client; this caps abuse


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_TURN_CHARS)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value

    def history_messages(self) -> list[BaseMessage]:
        return [
            HumanMessage(t.content) if t.role == "user" else AIMessage(t.content)
            for t in self.history
        ]


def _chunk_text(chunk: Any) -> str:
    """Extract text from an AIMessageChunk. Providers differ: OpenAI sends a
    string, Anthropic can send a list of content blocks."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") in (None, "text")
        )
    return ""


def _sse(event: str, payload: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(payload)}


async def stream_answer(
    orchestrator: Runnable, query: str, history: list[BaseMessage], request_id: str
) -> AsyncIterator[dict]:
    """Run the LangChain pipeline and translate its events into SSE events."""
    started = time.perf_counter()
    first_token_ms: float | None = None
    route: str | None = None

    try:
        inputs = {"query": query, "history": history}
        async for ev in orchestrator.astream_events(inputs, version="v2"):
            kind = ev["event"]

            if kind == "on_chain_end" and ev["name"] == ROUTER_RUN_NAME:
                plan: Plan = ev["data"]["output"]
                route = plan.route
                logger.info("[%s] routed: %s", request_id, plan.model_dump(exclude_none=True))
                yield _sse("route", plan.model_dump(exclude_none=True))

            elif kind == "on_chat_model_stream" and ANSWER_TAG in ev.get("tags", []):
                text = _chunk_text(ev["data"]["chunk"])
                if text:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    yield _sse("token", {"text": text})

        total_ms = (time.perf_counter() - started) * 1000
        logger.info("[%s] done route=%s ttft=%.0fms total=%.0fms",
                    request_id, route, first_token_ms or -1, total_ms)
        yield _sse("done", {
            "request_id": request_id,
            "route": route,
            "ttft_ms": round(first_token_ms) if first_token_ms is not None else None,
            "total_ms": round(total_ms),
        })
    except Exception:
        # Headers (200) are already sent, so errors are reported in-stream.
        # Internal details go to the logs, not to the client.
        logger.exception("[%s] stream failed", request_id)
        yield _sse("error", {"request_id": request_id, "message": "The request failed. Please try again."})


def create_app(orchestrator: Runnable | None = None) -> FastAPI:
    """App factory. Pass an orchestrator to inject fakes in tests; otherwise the
    real one is built from environment settings at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if orchestrator is not None:
            app.state.orchestrator = orchestrator
        else:
            from app.config import get_settings
            from app.llm import build_from_settings

            settings = get_settings()  # raises at startup if a key is missing
            app.state.orchestrator = build_from_settings(settings)
            logger.info("Models ready: answer=%s router=%s",
                        settings.llm_model, settings.effective_router_model)
        yield

    app = FastAPI(title="Streaming LangChain Orchestrator", version="1.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def chat_ui() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask")
    async def ask(body: AskRequest, request: Request) -> EventSourceResponse:
        request_id = uuid.uuid4().hex[:12]
        return EventSourceResponse(
            stream_answer(
                request.app.state.orchestrator, body.query, body.history_messages(), request_id
            ),
            ping=15,  # keep-alive comments so proxies don't close idle streams
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering so chunks flush immediately
                "X-Request-ID": request_id,
            },
        )

    return app


app = create_app()
