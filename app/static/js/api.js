/**
 * Everything that talks to the server lives here.
 *
 *   loadConfig()     GET /api/config   → model name and limits (set in app/config.py)
 *   streamAnswer()   POST /ask         → the answer, as a stream of events
 */

import { readSSE } from "./sse.js";

/** Fetch non-secret settings, so the page never hard-codes limits or model names. */
export async function loadConfig() {
  const res = await fetch("/api/config");
  if (!res.ok) throw new Error(`Config request failed: ${res.status}`);
  return res.json();
}

/**
 * Ask a question and yield the server's events as they arrive:
 *
 *   { event: "route", data: { route, routed_by, expression?, result? } }   first
 *   { event: "token", data: { text } }                                     many
 *   { event: "done",  data: { ttft_ms, total_ms, ... } }                   last
 *   { event: "error", data: { message } }                                  instead of done
 *
 * Pass an AbortSignal to cancel. Aborting closes the connection, which also
 * cancels the model call on the server.
 */
export async function* streamAnswer({ query, history, signal }) {
  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, history }),
    signal,
  });
  if (!res.ok) throw new Error(`Server returned ${res.status}`);
  yield* readSSE(res);
}
