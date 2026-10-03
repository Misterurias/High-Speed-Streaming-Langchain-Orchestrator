/**
 * Everything that changes what's on screen. No networking here.
 *
 * Each assistant reply is built once with addAssistantMessage(), which returns
 * a small object with methods to update that reply as events arrive:
 *
 *     const reply = addAssistantMessage();
 *     reply.showRoute(plan);          // on the "route" event
 *     reply.setText(answerSoFar);     // on each "token" event
 *     reply.finish("total 1.2 s");    // on "done"
 */

import { renderMarkdown } from "./markdown.js";

const thread = document.getElementById("thread");
const scroller = document.getElementById("scroller");

const ICON_SEND =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
const ICON_STOP =
  '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';

/** Remove the welcome screen (called when the first message is sent). */
export function hideEmptyState() {
  document.getElementById("empty")?.remove();
}

/** Add the user's message. textContent shows it exactly as typed, never as HTML. */
export function addUserMessage(text) {
  const el = createElement("div", "msg user");
  const bubble = createElement("div", "bubble");
  bubble.textContent = text;
  el.appendChild(bubble);
  thread.appendChild(el);
  scrollToBottom({ force: true });
}

/** Add an empty assistant reply and return the controls to fill it in. */
export function addAssistantMessage() {
  const el = createElement("div", "msg assistant streaming");
  const badge = createElement("div", "badge pending");
  badge.innerHTML = '<span class="dot"></span>Routing…';
  const content = createElement("div", "content");
  el.append(badge, content);
  thread.appendChild(el);
  scrollToBottom({ force: true });

  return {
    /** Show which chain is answering, from the server's "route" event. */
    showRoute(plan) {
      if (plan.routed_by === "fallback") {
        badge.className = "badge fallback";
        badge.textContent = "General · fallback";
        badge.title = plan.note || "";
      } else if (plan.route === "math") {
        badge.className = "badge math";
        badge.textContent = "Calculator · ";
        const code = document.createElement("code");
        code.textContent = `${plan.expression} = ${plan.result}`;
        badge.appendChild(code);
        badge.title = plan.routed_by === "fast_path"
          ? "Routed instantly, no LLM call"
          : "Routed by the LLM";
      } else {
        badge.className = "badge general";
        badge.textContent = "General";
      }
    },

    /** Replace the reply with the full answer so far, rendered as Markdown. */
    setText(markdownText) {
      content.innerHTML = renderMarkdown(markdownText);
      scrollToBottom();
    },

    /** Stop the cursor, drop a badge that never resolved, and add a footnote. */
    finish(footnote) {
      el.classList.remove("streaming");
      if (badge.classList.contains("pending")) badge.remove();
      if (footnote) {
        const meta = createElement("div", "meta");
        meta.textContent = footnote;
        el.appendChild(meta);
      }
    },

    showError(message) {
      const box = createElement("div", "error-box");
      box.textContent = message;
      el.appendChild(box);
    },
  };
}

/** Switch the send button between "send" and "stop" modes. */
export function setSendButton(button, { busy, hasText }) {
  button.innerHTML = busy ? ICON_STOP : ICON_SEND;
  button.setAttribute("aria-label", busy ? "Stop" : "Send");
  button.disabled = !busy && !hasText;
}

/** Grow the input box with its content, up to the CSS max-height. */
export function autoGrow(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
}

/**
 * Keep the newest text in view, but only if the user is already near the
 * bottom. If they scrolled up to reread something, don't yank them down.
 */
function scrollToBottom({ force = false } = {}) {
  const distanceFromBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
  if (force || distanceFromBottom < 120) scroller.scrollTop = scroller.scrollHeight;
}

function createElement(tag, className) {
  const el = document.createElement(tag);
  el.className = className;
  return el;
}
