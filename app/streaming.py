"""
Step 3: turn the LangChain pipeline's progress into Server-Sent Events (SSE).

What SSE is
-----------
A normal HTTP response is sent all at once. With SSE, the server keeps the
response open and writes small messages into it over time. Each message is
plain text, a named event plus a JSON payload, separated by a blank line:

    event: token
    data: {"text": "Hello"}

The browser (static/js/sse.js) reads these as they arrive and updates the page.

What one request's stream looks like
------------------------------------
    event: route   data: {"route": "math", "routed_by": "llm", "expression": "3*12-5", "result": "31"}
    event: token   data: {"text": "You'd"}
    event: token   data: {"text": " have 31"}
    ...                                      (one per chunk the model generates)
    event: done    data: {"request_id": "...", "route": "math", "ttft_ms": 412, "total_ms": 980}

If something breaks mid-answer, the stream ends with an `error` event instead
of `done`. (The HTTP status is already 200 by then, so errors must travel
inside the stream.)

Where the events come from
--------------------------
`orchestrator.astream_events(...)` yields a play-by-play of everything
LangChain does: every step starting and ending, every token from every model.
We only forward two kinds:
    * the router step ending        → one `route` event (which chain was picked)
    * a token from the ANSWER model  → a `token` event
Tokens from the router model are internal and are skipped.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from app.chains import ANSWER_TAG
from app.routing import ROUTER_STEP_NAME, Plan

logger = logging.getLogger("orchestrator")

# The four SSE event names. The browser listens for exactly these.
ROUTE, TOKEN, DONE, ERROR = "route", "token", "done", "error"


def sse_event(event: str, payload: dict[str, Any]) -> dict[str, str]:
    """One SSE message in the shape sse-starlette expects."""
    return {"event": event, "data": json.dumps(payload)}


def chunk_text(chunk: Any) -> str:
    """Get the text out of one streamed model chunk.

    Providers differ: OpenAI sends a plain string, Anthropic can send a list
    of content blocks like [{"type": "text", "text": "Hi"}].
    """
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


async def stream_answer(
    orchestrator: Runnable,
    query: str,
    history: list[BaseMessage],
    request_id: str,
) -> AsyncIterator[dict[str, str]]:
    """Run the pipeline and yield SSE events as things happen.

    This is an async generator: each `yield` sends one event to the browser
    immediately, while the model is still generating the rest.
    """
    started = time.perf_counter()
    first_token_ms: float | None = None
    route: str | None = None

    try:
        async for event in orchestrator.astream_events(
            {"query": query, "history": history}, version="v2"
        ):
            # The router step just finished: tell the browser which route won.
            if event["event"] == "on_chain_end" and event["name"] == ROUTER_STEP_NAME:
                plan: Plan = event["data"]["output"]
                route = plan.route
                logger.info("[%s] routed: %s", request_id, plan.model_dump(exclude_none=True))
                yield sse_event(ROUTE, plan.model_dump(exclude_none=True))

            # The answering model produced a chunk of text: forward it.
            elif event["event"] == "on_chat_model_stream" and ANSWER_TAG in event.get("tags", []):
                text = chunk_text(event["data"]["chunk"])
                if text:
                    if first_token_ms is None:
                        first_token_ms = elapsed_ms(started)
                    yield sse_event(TOKEN, {"text": text})

        total_ms = elapsed_ms(started)
        logger.info("[%s] done route=%s ttft=%sms total=%.0fms",
                    request_id, route, _round(first_token_ms), total_ms)
        yield sse_event(DONE, {
            "request_id": request_id,
            "route": route,
            "ttft_ms": _round(first_token_ms),  # time to first token: what users feel as speed
            "total_ms": round(total_ms),
        })

    except Exception:
        # Full details go to the server log; the browser only gets a safe message.
        logger.exception("[%s] stream failed", request_id)
        yield sse_event(ERROR, {
            "request_id": request_id,
            "message": "The request failed. Please try again.",
        })


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _round(value: float | None) -> int | None:
    return round(value) if value is not None else None
