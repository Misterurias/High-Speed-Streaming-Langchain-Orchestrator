/**
 * The chat page's entry point: connects the input box, the server, and the screen.
 *
 * Files and their jobs:
 *   main.js       this file: conversation state and event handlers
 *   api.js        talks to the server (POST /ask, GET /api/config)
 *   sse.js        splits the streamed response into individual events
 *   chat-view.js  updates what's on screen
 *   markdown.js   turns the model's Markdown into safe HTML
 *
 * What happens when you press Enter:
 *   1. Your message is shown, and an empty reply with a "Routing…" badge appears.
 *   2. The question + earlier conversation are POSTed to /ask.
 *   3. As events stream back:
 *        route → the badge shows Calculator or General
 *        token → the text is appended and the reply re-rendered
 *        done  → timing is shown under the reply
 *   4. The finished exchange is added to `history` for the next question.
 */

import { loadConfig, streamAnswer } from "./api.js";
import {
  addAssistantMessage, addUserMessage, autoGrow, hideEmptyState, setSendButton,
} from "./chat-view.js";

// ── State ──────────────────────────────────────────────────────────────────

/** Finished exchanges, sent with each question so follow-ups have context. */
const history = []; // [{ role: "user" | "assistant", content: "..." }]

/** Defaults until /api/config loads. The real values come from app/config.py. */
let settings = { max_query_chars: 2000, max_history_turns: 20 };

/** Lets the Stop button cancel the request in progress (null when idle). */
let currentRequest = null;

// ── Page elements ──────────────────────────────────────────────────────────

const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendButton = document.getElementById("send");

// ── Asking a question ──────────────────────────────────────────────────────

async function ask(query) {
  hideEmptyState();
  addUserMessage(query);
  const reply = addAssistantMessage();

  currentRequest = new AbortController();
  refreshSendButton();

  let answer = "";
  let footnote = "";
  let completed = false; // true if we got "done" or the user pressed Stop
  let failed = false;    // true if the server sent an "error" event
  const renderSoon = throttleToAnimationFrames(() => reply.setText(answer));

  try {
    const events = streamAnswer({
      query,
      history: history.slice(-settings.max_history_turns),
      signal: currentRequest.signal,
    });

    for await (const { event, data } of events) {
      if (event === "route") {
        reply.showRoute(data);
      } else if (event === "token") {
        answer += data.text;
        renderSoon();
      } else if (event === "done") {
        completed = true;
        footnote = formatTiming(data);
      } else if (event === "error") {
        failed = true;
        reply.showError(data.message);
      }
    }
    // The stream ended without "done" or "error": the server likely restarted.
    if (!completed && !failed) reply.showError("The connection closed before the answer finished.");
  } catch (err) {
    if (err.name === "AbortError") {
      completed = true; // keep the partial answer as part of the conversation
      footnote = "Stopped";
    } else {
      reply.showError("Couldn't reach the server. Is it running?");
    }
  } finally {
    reply.setText(answer); // final render, in case a frame was still pending
    reply.finish(footnote);
    if (completed && answer.trim()) {
      history.push({ role: "user", content: query }, { role: "assistant", content: answer });
    }
    currentRequest = null;
    refreshSendButton();
    input.focus();
  }
}

/**
 * Re-rendering Markdown on every token is wasteful, because tokens can arrive
 * faster than the screen refreshes. This runs `fn` at most once per frame
 * (~60 times a second), no matter how many tokens arrived in between.
 */
function throttleToAnimationFrames(fn) {
  let queued = false;
  return () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; fn(); });
  };
}

function formatTiming({ ttft_ms, total_ms }) {
  const firstToken = ttft_ms != null ? `first token ${ttft_ms} ms · ` : "";
  return `${firstToken}total ${(total_ms / 1000).toFixed(1)} s`;
}

// ── Input handling ─────────────────────────────────────────────────────────

function submit() {
  if (currentRequest) {           // while streaming, the button means Stop
    currentRequest.abort();
    return;
  }
  const query = input.value.trim();
  if (!query) return;
  input.value = "";
  autoGrow(input);
  ask(query);
}

function refreshSendButton() {
  setSendButton(sendButton, { busy: currentRequest !== null, hasText: input.value.trim() !== "" });
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  submit();
});

input.addEventListener("input", () => {
  autoGrow(input);
  refreshSendButton();
});

input.addEventListener("keydown", (e) => {
  // Enter sends; Shift+Enter adds a new line. isComposing skips Enter presses
  // that confirm input-method text (e.g. typing Japanese).
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!currentRequest) submit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    if (!currentRequest) ask(button.dataset.query);
  });
});

document.getElementById("newChat").addEventListener("click", () => {
  currentRequest?.abort();
  location.reload(); // the conversation only lives in this page, so reloading clears it
});

// ── Startup ────────────────────────────────────────────────────────────────

refreshSendButton();

loadConfig()
  .then((config) => {
    settings = config;
    input.maxLength = config.max_query_chars;
    document.getElementById("modelName").textContent = config.model;
  })
  .catch(() => { /* keep the defaults; the server still enforces its limits */ });
