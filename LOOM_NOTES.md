# Loom talking points (~5 minutes)

An order to walk through, not a script. Have the browser, a terminal, and the editor open.

## 1. Demo first (45s)
- Open http://localhost:8000. Click the cookies suggestion: the badge says **Calculator · 3*12-5 = 31**, then the answer streams in.
- Ask "now double that" to show follow-ups use history (the router resolves "that" to 31).
- Ask "Explain what an API is in two sentences": purple **General** badge, Markdown formatting.
- Start a long question and press Stop to show cancellation.

## 2. My design goals (15s)
Say the four goals out loud, then show each one:
1. **Understandable:** a newcomer can follow a question through the code.
2. **One source of truth:** models and limits live in one file.
3. **Tested and visible:** every test has a readable name, with coverage.
4. **Separated front end:** HTML, CSS, and JS each in their own files.

## 3. Understandable: follow one question (2 min)
Open `docs/ARCHITECTURE.md` briefly to show it exists, then walk the code in request order:
- `main.py`: the routes, and the docstring listing which file handles each step.
- `routing.py`: the diagram at the top. Fast path → one LLM call with structured output (`RouteDecision`) → calculator. Point out that every failure falls back to general.
- `safe_math.py`: no `eval()`. It parses into a tree and only allows math.
- `chains.py`: `RunnablePassthrough.assign` adds the plan, `RunnableBranch` is an if/else, `|` pipes steps together.
- `streaming.py`: `astream_events` gives a play-by-play; I forward the router result as `route` and only the answer model's tokens as `token`.
- Front end: `js/main.js` header lists every JS file's job; `sse.js` explains why fetch is used instead of EventSource (POST bodies).

## 4. One source of truth (30s)
- `config.py`: change `MODEL` and both router and answerer switch.
- `.env` holds only the key.
- The header in the UI shows the model name, read from `/api/config`, so the front end doesn't hard-code it either.
- Show `test_no_model_name_appears_outside_config_py`: it fails if anyone writes a model name somewhere else.

## 5. Tests (45s)
- Run `pytest tests/test_routing.py` to show named tests for one file.
- Run `pytest` for the whole suite: every test listed by name, then the coverage table at 100%.
- Mention: fake models, so no API key, no network, runs in seconds.

## 6. Wrap up (15s)
Production next steps: authentication and rate limiting on `/ask`, tracing with LangSmith, an eval set to measure routing accuracy, and caching repeated questions.
