# Streaming LangChain Orchestrator

A FastAPI service with one endpoint, `POST /ask`, that uses LangChain to route a query to either a **math tool chain** or a **general LLM chain**, and streams the answer back token by token as **Server-Sent Events**.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then add your API key to .env
uvicorn app.main:app --reload --reload-dir app
```

`--reload-dir app` matters: plain `--reload` watches the whole folder, including `.venv`, and any `.py` change there restarts the server and cuts off in-flight streams. For demos and production, drop `--reload` entirely.

### Chat UI

Open **http://localhost:8000** for a chat interface: answers stream in token by token with Markdown formatting, each reply shows which route handled it (Calculator with the computed expression, or General), and follow-up questions keep the conversation context. The send button turns into a Stop button while an answer is streaming.

The page is a single HTML file (`app/static/index.html`) with no build step. It reads the SSE stream with `fetch` (the browser's `EventSource` can't send POST bodies). Markdown is rendered with `marked` and sanitized with `DOMPurify`, both bundled in `app/static/vendor/` so the app works offline.

### API

```bash
curl -N -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "If I buy 3 boxes of 12 cookies and eat 5, how many are left?"}'

# or the pretty terminal client
python scripts/ask.py "Explain what an API is in two sentences"
```

For multi-turn chat, send earlier turns as optional `history` (max 20 turns). The router and the answer prompts both see it, so a follow-up like "now double that" resolves against the earlier answer:

```json
{"query": "Now double that",
 "history": [{"role": "user", "content": "What is 3 * 12?"},
             {"role": "assistant", "content": "3 * 12 = 36."}]}
```

Run the tests (no API key needed; they use fake models):

```bash
pytest -q
```

## How it works

```
POST /ask {"query": "..."}
      │
      ▼
 router (LangChain Runnable)
   1. fast path ── query is already an expression ("12*(3+4)")? → math, no LLM call
   2. LLM router ── structured output: {route, expression}
   3. safe_math ─── evaluates the expression deterministically
      │            (any failure → falls back to general, never errors)
      ▼
 RunnableBranch
   ├─ math    → MATH_PROMPT (with verified result) → LLM → stream
   └─ general → GENERAL_PROMPT → LLM → stream
      │
      ▼
 astream_events → SSE:  route → token, token, token… → done
```

The whole pipeline is one LangChain Runnable (`app/chains.py`), built from `RunnablePassthrough.assign`, `RunnableLambda`, `RunnableBranch`, prompt templates, and `with_structured_output`. The API layer (`app/main.py`) consumes it with `astream_events` and turns LangChain events into SSE events.

### SSE event format

| event   | data |
|---------|------|
| `route` | `{"route": "math", "routed_by": "llm", "expression": "3*12-5", "result": "31"}` |
| `token` | `{"text": "There are"}` (one per streamed chunk) |
| `done`  | `{"request_id": "…", "route": "math", "ttft_ms": 410, "total_ms": 980}` |
| `error` | `{"request_id": "…", "message": "The request failed. Please try again."}` |

`routed_by` is `fast_path`, `llm`, or `fallback`, so clients and logs can see exactly how each decision was made.

## Design decisions

**Speed**
- **Routing and extraction in one call.** The router returns both the route and the math expression, so a math question costs one router call plus the streamed answer, not three calls.
- **Fast path.** Pure expressions skip the router LLM entirely.
- **Optional smaller router model.** Set `ROUTER_MODEL` to a cheaper, faster model; classification doesn't need the big one.
- **Clients built once at startup** (FastAPI lifespan) and reused, so requests don't pay for new HTTP connections.
- **True streaming.** Tokens are forwarded the moment the model emits them. `X-Accel-Buffering: no` stops nginx-style proxies from buffering, and keep-alive pings stop idle streams from being cut.
- **Client disconnects cancel the work.** sse-starlette cancels the generator, which cancels the in-flight LLM request instead of generating tokens nobody reads.

**Correctness**
- **The LLM never does the arithmetic.** It only writes the expression; `app/safe_math.py` computes it, and the answer prompt is told not to contradict the verified result.

**Security**
- **No keys in code.** Keys are read from environment variables or a git-ignored `.env` via `pydantic-settings`, stored as `SecretStr` (masked if logged), and the app **fails at startup** if the needed key is missing.
- **No `eval()`.** The math tool parses the expression with Python's `ast` and only allows numbers, basic operators, and an allow-list of math functions. Code injection like `__import__('os')` is rejected, and limits block "exponent bombs" like `9**9**9`.
- **Input validation.** Queries must be 1–2000 non-blank characters (422 otherwise).
- **No internal errors leaked.** Stream failures are logged server-side with a request ID; the client only gets a generic message plus the ID.

**Resilience**
- Router failure, bad expressions, or a math route with no expression all **degrade to the general chain** instead of failing the request.
- Model timeouts and retries are configurable (`LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`).

**Provider-agnostic**
- Models are `provider:model` strings passed to LangChain's `init_chat_model`, so switching between Anthropic and OpenAI is a config change.

## Project layout

```
app/
  main.py       FastAPI app, /ask endpoint, LangChain events → SSE, serves the chat UI
  static/       Chat UI (index.html) and vendored marked + DOMPurify
  chains.py     Router, math chain, general chain, orchestrator Runnable
  safe_math.py  AST-based safe calculator (the math tool)
  llm.py        Builds real models from settings
  config.py     Env/.env settings with startup validation
scripts/ask.py  Demo client that prints tokens as they stream
tests/          34 tests: math tool safety, end-to-end SSE, history, and the chat page (fake models)
```

## What I'd add for production

Per-client rate limiting and API-key auth on the endpoint, OpenTelemetry/LangSmith tracing, a semantic cache for repeated questions, and an eval set to measure router accuracy.
