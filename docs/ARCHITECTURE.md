# How this app works

A guide for someone new to the codebase, or new to LangChain or streaming. After reading it, you should be able to follow any question from the moment it's typed to the moment the last word appears, and know which file to open for each step.

## The 30-second version

You type a question. The server decides whether it needs **math** or is a **general** question. Math questions get their number from a real calculator, and the model only explains it. General questions go straight to the model. Either way, the answer is sent back **a few words at a time** as the model writes it, instead of all at once at the end.

```
 Browser                                 Server
 ───────                                 ──────
 type question ──── POST /ask ─────────▶ 1. ROUTE      math or general?
                                         2. ANSWER     run the matching chain
 badge appears ◀─── event: route ──────┐ 3. STREAM     send progress as it happens
 words appear  ◀─── event: token ×N ───┤
 timing shown  ◀─── event: done  ──────┘
```

## Map of the code

Each file has one job. Read them in this order.

| File | Job | Read it to learn |
|---|---|---|
| `app/config.py` | **Every setting.** Which model, limits, timeouts. API keys come from `.env`. | How to swap models |
| `app/main.py` | The web server and its routes. | What URLs exist |
| `app/schemas.py` | The shape of a valid request to `/ask`. | What the browser must send |
| `app/routing.py` | **Step 1.** Decide math vs. general. | How routing works |
| `app/safe_math.py` | The calculator the router uses. | Why we don't use `eval()` |
| `app/prompts.py` | The text of every prompt sent to a model. | What the model is told |
| `app/chains.py` | **Step 2.** The answer chains, and how all steps connect. | How LangChain pieces fit |
| `app/streaming.py` | **Step 3.** Turn progress into SSE events. | How streaming works |
| `app/llm.py` | Build real model clients from `config.py`. | Where API calls are set up |
| `app/static/` | The chat page: `index.html`, `css/`, `js/`. | The browser side |
| `tests/` | One test file per app file. | What each part promises to do |

## Three ideas you need

### 1. LangChain Runnables and the `|` pipe

LangChain's main idea is that every building block (a prompt template, a model, an output parser, or one of our own functions) is a **Runnable**. Runnables share the same methods:

- `.invoke(input)` runs it and returns the whole result
- `.astream(input)` returns the result piece by piece
- `.astream_events(input)` returns a play-by-play of every step inside it (this is what streaming uses)

Because they share an interface, they connect with `|`, like a Unix pipe: the output of the left side becomes the input of the right side.

```python
GENERAL_PROMPT | answer_llm | StrOutputParser()
#  fill in the     send it to     pull the text out
#  question        the model      of the reply
```

The result of a pipe is also a Runnable, so whole pipelines can be piped together. `chains.py` builds the full app this way. It uses two more LangChain building blocks:

- **`RunnablePassthrough.assign(plan=...)`** runs a step and *adds* its result to the input dictionary. The router's output becomes `state["plan"]` while `query` and `history` pass through untouched.
- **`RunnableBranch`** is an if/else for Runnables: if `plan.route == "math"`, run the math chain; otherwise, run the general chain.

> **Why use LangChain instead of calling the API directly?** Each piece is swappable. The tests replace the real models with fakes without touching any other code, `init_chat_model` switches between Anthropic and OpenAI from a string, and `astream_events` provides step-level progress for free. A hand-written version would need its own plumbing for all three.

### 2. Structured output: making the model reply in a fixed shape

The router has to return data, not prose. `routing.py` defines the shape as a Pydantic class:

```python
class RouteDecision(BaseModel):
    route: Literal["math", "general"]
    expression: str | None
```

`router_llm.with_structured_output(RouteDecision)` sends that shape to the model as a schema it must fill in, then checks the reply and converts it into a `RouteDecision` object. The field descriptions in the class are instructions the model reads.

### 3. Server-Sent Events (SSE): sending an answer in pieces

A normal HTTP response is delivered all at once. **SSE** keeps the response open, and the server writes small text messages into it as it goes:

```
event: token
data: {"text": "You'd have"}

event: token
data: {"text": " 31 cookies"}

```

Each message is an `event:` line plus a `data:` line, followed by a blank line. On the server, `stream_answer` in `streaming.py` is an **async generator**: each `yield` sends one message immediately. On the browser, `js/sse.js` reads the response body as it arrives and splits it at the blank lines.

> **Why SSE and not WebSockets?** WebSockets are two-way, for apps like multiplayer games where both sides talk constantly. A chat answer only flows server → browser, and SSE is plain HTTP, so it works through ordinary proxies and needs no extra protocol.

## The life of one question

Following `"If I buy 3 boxes of 12 cookies and eat 5, how many are left?"` through the system.

**1. The browser sends it.** `js/main.js` → `ask()` shows your message, adds an empty reply with a "Routing…" badge, and calls `streamAnswer()` in `js/api.js`. That sends:

```json
POST /ask
{"query": "If I buy 3 boxes of 12 cookies and eat 5, how many are left?", "history": []}
```

**2. The server checks it.** FastAPI validates the body against `AskRequest` in `schemas.py`. A blank or overlong query never reaches our code; it's rejected with a 422.

**3. The route handler starts the stream.** `main.py` → `ask()` creates a request ID and returns an `EventSourceResponse` wrapping `stream_answer(...)`. From here on, everything the generator yields goes straight to the browser.

**4. Step 1, routing.** `stream_answer` runs the pipeline with `astream_events`. The first step is the router (`routing.py` → `decide_route`):

- *Fast path?* Is the query a bare expression like `12*(3+4)`? No, it has words.
- *Ask the router model.* It replies `RouteDecision(route="math", expression="3*12-5")`.
- *Calculate.* `safe_math.evaluate("3*12-5")` returns `31`.

The result is `Plan(route="math", routed_by="llm", expression="3*12-5", result="31")`.

**5. The browser learns the route.** `streaming.py` sees the router step end and yields:

```
event: route
data: {"route": "math", "routed_by": "llm", "expression": "3*12-5", "result": "31"}
```

The browser turns the "Routing…" badge into the green **Calculator · 3*12-5 = 31**.

**6. Step 2, answering.** `RunnableBranch` in `chains.py` picks the math chain. The model receives `MATH_PROMPT` from `prompts.py` with the verified result filled in, and is told not to recompute it. It starts writing: *"You'd have **31 cookies** left…"*

**7. Step 3, streaming the words.** Each chunk the answering model produces arrives in `stream_answer` as an `on_chat_model_stream` event. Only chunks from the model tagged `"answer"` are forwarded, because the router's own model output is internal:

```
event: token    data: {"text": "You'd"}
event: token    data: {"text": " have **31"}
...
```

In the browser, each token is appended to the answer, and the whole answer is re-rendered as Markdown at most once per screen frame (`js/markdown.js`, `js/main.js`). That's why bold text "snaps" into place once its closing `**` arrives.

**8. Done.** The pipeline finishes and the server yields:

```
event: done
data: {"request_id": "a1b2c3d4e5f6", "route": "math", "ttft_ms": 412, "total_ms": 980}
```

The browser shows the timing under the reply and saves the exchange to `history`, so a follow-up like "now double that" has context.

## Every routing outcome

| Query | Path | Badge |
|---|---|---|
| `12 * (3 + 4)` | Fast path: calculated directly, **no router model call** | Calculator |
| `3 boxes of 12, eat 5?` | Router says math → calculator computes | Calculator |
| `Explain what an API is` | Router says general | General |
| `What is a prime number?` | Router says general (a question *about* math, nothing to compute) | General |
| Router model errors or times out | Fallback | General · fallback |
| Router replies with no usable decision | Fallback | General · fallback |
| Router writes an expression the calculator rejects | Fallback | General · fallback |

A fallback never fails the request; the general chain answers instead. The reason is logged and included as `note` in the `route` event.

## How to…

**Swap the model.** Change `MODEL` in `app/config.py`. To use a different model for routing only, change `ROUTER = ModelRole(model=...)`. For OpenAI, use `"openai:<model-name>"` and add `OPENAI_API_KEY` to `.env`. A test (`test_no_model_name_appears_outside_config_py`) fails if a model name is ever written anywhere else.

**Change a limit.** Edit `MAX_QUERY_CHARS` or `MAX_HISTORY_TURNS` in `app/config.py`. The API enforces it, and the browser picks it up from `/api/config`.

**Change what the model is told.** Edit `app/prompts.py`.

**Add a new route** (for example, "weather"):
1. Add `"weather"` to `Route` in `routing.py` and describe it in `ROUTER_PROMPT`.
2. Write a `build_weather_chain()` in `chains.py`.
3. Add a `(is_weather, build_weather_chain(...))` pair to the `RunnableBranch`.
4. Give it a badge in `js/chat-view.js` → `showRoute()` and a color in `css/styles.css`.
5. Add tests in `tests/test_routing.py` and `tests/test_chains.py`.

## Things that can bite you

- **`--reload` restarts mid-answer.** Plain `uvicorn --reload` watches the whole folder, including `.venv`. If files there change (for example, while iCloud syncs your Desktop), the server restarts and cuts off streams. Use `--reload-dir app`.
- **Proxies can buffer the stream.** Servers like nginx may collect a response before forwarding it, which quietly turns streaming into one big chunk at the end. The `X-Accel-Buffering: no` header in `main.py` turns that off for nginx.
- **Errors arrive inside the stream.** By the time an answer is streaming, the HTTP status (200) has already been sent. Failures arrive as an `error` event, with details in the server log under the request ID.
