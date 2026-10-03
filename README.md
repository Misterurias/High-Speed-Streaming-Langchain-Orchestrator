# Streaming LangChain Orchestrator

A chat app with a FastAPI backend. Each question is **routed by LangChain** either to a calculator tool (math) or to a general LLM chain, and the answer **streams back word by word** over Server-Sent Events.

**New to the code?** Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). It explains LangChain, SSE, and routing from scratch, and follows one question through every file.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env              # then paste your API key into .env
uvicorn app.main:app --reload --reload-dir app
```

Open **http://localhost:8000**.

`--reload-dir app` restarts the server when your code changes, but ignores `.venv`. Plain `--reload` watches everything, and a file change in `.venv` would restart the server mid-answer. For demos, leave `--reload` off.

## Configuration: one place

**All settings live in [`app/config.py`](app/config.py).** To swap models, change one line:

```python
MODEL = "anthropic:claude-haiku-4-5"
```

The router and the answerer both use `MODEL` unless you point one of them elsewhere. Limits (`MAX_QUERY_CHARS`, `MAX_HISTORY_TURNS`) are there too, and the browser reads them from `GET /api/config`, so the front end never hard-codes them.

`.env` holds **only secrets** (API keys). A test fails if a model name appears in any file other than `config.py`.

## Testing

Tests use fake models, so they run offline, in a few seconds, with no API key.

```bash
pytest                              # whole suite, every test by name, coverage table at the end
pytest tests/test_routing.py        # one file
pytest tests/test_routing.py::TestFallbacks    # one group
pytest -k calculator                # every test whose name contains "calculator"
pytest --no-cov -q                  # quick run: dots only, no coverage
```

There is one test file per app module:

| Test file | Checks |
|---|---|
| `test_safe_math.py` | The calculator gets answers right, and blocks code injection and inputs too large to compute |
| `test_routing.py` | Fast path, LLM routing, and every fallback |
| `test_chains.py` | What the answering model is sent for each route, including history |
| `test_streaming.py` | Event order, chunking, router tokens kept internal, safe errors |
| `test_api.py` | HTTP routes, request validation, static files, `/api/config` |
| `test_config.py` | One source of truth, secrets handling, building real model clients |

Settings for `pytest` (verbose output and coverage) are in `pyproject.toml`. When you run a single file, the coverage table still lists every module, so the modules that file doesn't test will show low numbers.

## Project layout

```
app/
  config.py       every setting: models, limits; keys come from .env
  main.py         web server and routes (start reading here)
  schemas.py      shape of a valid /ask request
  routing.py      step 1: math or general?
  safe_math.py    the calculator tool
  prompts.py      every prompt sent to a model
  chains.py       step 2: answer chains, and how all steps connect
  streaming.py    step 3: progress → SSE events
  llm.py          builds model clients from config.py
  static/
    index.html    page structure
    css/          styles.css
    js/           main.js (entry), api.js, sse.js, chat-view.js, markdown.js
    vendor/       marked + DOMPurify (bundled, no CDN)
docs/
  ARCHITECTURE.md how it all works
scripts/ask.py    terminal client: python scripts/ask.py "What is 15% of 240?"
tests/            one file per app module, plus shared fakes in conftest.py
```

## API

`POST /ask` with `{"query": "...", "history": [...]}` (`history` is optional) returns an SSE stream:

| event | data |
|---|---|
| `route` | `{"route": "math", "routed_by": "llm", "expression": "3*12-5", "result": "31"}` |
| `token` | `{"text": "You'd have"}` (one per chunk) |
| `done` | `{"request_id": "…", "route": "math", "ttft_ms": 410, "total_ms": 980}` |
| `error` | `{"request_id": "…", "message": "The request failed. Please try again."}` |

```bash
curl -N -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "If I buy 3 boxes of 12 cookies and eat 5, how many are left?"}'
```

## Design decisions

- **The LLM never does arithmetic.** It writes the expression; `safe_math.py` computes it with an allow-list parser (no `eval()`), and the answer prompt is told not to contradict the result.
- **One router call does two jobs.** Classification and writing the expression happen in the same structured-output call, and bare expressions skip the router entirely.
- **Failures degrade, not error.** Any router or calculator problem falls back to the general chain.
- **Only answer tokens stream.** The router's model output is internal and filtered out by tag.
- **Secrets stay secret.** Keys load from the environment as `SecretStr`, the app refuses to start without them, and stream errors never expose internal details.
- **No build step for the front end.** Plain HTML, CSS, and ES modules. Markdown is rendered with `marked` and sanitized with `DOMPurify`, both bundled locally.
