# Loom talking points (~4 minutes)

Keep `app/chains.py`, `app/main.py`, and a terminal open. Not a script to read, just the order to hit things.

## 1. What it does (20s)
- One endpoint, `POST /ask`. LangChain decides if the query is math or general, then streams the answer back over Server-Sent Events.
- Live demo right away: run `python scripts/ask.py "If I buy 3 boxes of 12 cookies and eat 5, how many are left?"`. Point out the `route` event first, then tokens arriving one by one, then `done` with time-to-first-token.
- Run a general one too: `python scripts/ask.py "Explain what an API is in two sentences"`.

## 2. Routing with LangChain (`chains.py`, ~90s)
- The whole pipeline is one Runnable: `RunnablePassthrough.assign(plan=router) | RunnableBranch(math_chain, general_chain)`.
- Router has three layers:
  1. Fast path: if the query is already an expression like `12*(3+4)`, no LLM call at all.
  2. One LLM call with `with_structured_output(RouteDecision)`. It returns the route *and* the expression in the same call, so math doesn't need an extra round trip.
  3. Fallbacks: if the router errors or the expression is bad, it drops to the general chain instead of failing.
- Key point: the LLM never does the arithmetic. It writes the expression, and `safe_math.py` computes it. The answer prompt gets the verified result and is told not to contradict it.

## 3. The math tool (`safe_math.py`, ~30s)
- No `eval()`. It parses with Python's `ast` and only allows numbers, operators, and a list of math functions.
- Show the test cases: `__import__('os')` gets rejected, `9**9**9` gets blocked before it can hang the server.

## 4. Streaming (`main.py`, ~60s)
- `astream_events` gives every event in the pipeline. I forward only tokens from models tagged `"answer"`, so router tokens don't leak into the user's stream.
- `EventSourceResponse` sends proper SSE. Pings keep the connection alive, and `X-Accel-Buffering: no` stops proxies from buffering chunks.
- If the client disconnects, the generator gets cancelled, which cancels the LLM call.
- Errors mid-stream come back as an `error` event with a request ID; real details stay in server logs.

## 5. Security and config (`config.py`, ~30s)
- Keys only come from environment variables or `.env` (git-ignored, with `.env.example` committed).
- Stored as `SecretStr`, and the app refuses to start if the key is missing.
- Models are `provider:model` strings, so swapping Anthropic and OpenAI is a config change.

## 6. Wrap up (20s)
- Run `pytest -q`: 30 tests, using fake models, so no key needed.
- Next steps for production: rate limiting and auth, tracing with LangSmith, caching, and an eval set for router accuracy.
