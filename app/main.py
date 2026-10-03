"""
The web server: its routes and startup. Start here when reading the code.

    GET  /             the chat page (static/index.html)
    GET  /static/...   the page's CSS, JavaScript, and libraries
    GET  /api/config   non-secret settings the page needs (model name, limits)
    GET  /health       "I'm alive" check for hosting platforms
    POST /ask          ask a question; the answer streams back as SSE

How a question is answered, file by file:

    schemas.py    validates the request body
    routing.py    step 1: decide math vs. general (calls the calculator, safe_math.py)
    chains.py     step 2: run the matching answer chain
    streaming.py  step 3: send progress to the browser as SSE events

Models are built once at startup from config.py (via llm.py) and reused.

Run it:   uvicorn app.main:app --reload --reload-dir app
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.runnables import Runnable
from sse_starlette.sse import EventSourceResponse

from app import config
from app.schemas import AskRequest
from app.streaming import stream_answer

logger = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).parent / "static"


def create_app(orchestrator: Runnable | None = None) -> FastAPI:
    """Build the FastAPI app.

    Normally the pipeline is built from config.py at startup. Tests pass in
    their own `orchestrator` made with fake models instead.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs once when the server starts, before any request is handled.
        if orchestrator is not None:
            app.state.orchestrator = orchestrator
        else:
            from app.llm import build_orchestrator_from_config

            app.state.orchestrator = build_orchestrator_from_config()
            logger.info("Models ready: router=%s answerer=%s",
                        config.ROUTER.model, config.ANSWERER.model)
        yield

    app = FastAPI(title="Streaming LangChain Orchestrator", version="2.0.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def chat_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/config")
    async def public_settings() -> dict:
        return config.public_config()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask")
    async def ask(body: AskRequest, request: Request) -> EventSourceResponse:
        # By the time we get here, FastAPI has already validated `body`.
        request_id = uuid.uuid4().hex[:12]
        events = stream_answer(
            request.app.state.orchestrator,
            body.query,
            body.history_as_messages(),
            request_id,
        )
        # EventSourceResponse sends each event from the generator as soon as it
        # is yielded. If the browser disconnects, it cancels the generator, which
        # cancels the in-flight model call.
        return EventSourceResponse(
            events,
            ping=15,  # send a keep-alive comment every 15s so idle streams aren't cut
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # tell proxies like nginx not to buffer chunks
                "X-Request-ID": request_id,
            },
        )

    return app


app = create_app()
