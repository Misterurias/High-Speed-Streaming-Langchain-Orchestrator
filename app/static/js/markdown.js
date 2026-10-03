/**
 * Turns the model's Markdown into safe HTML.
 *
 * The model doesn't send formatting, it sends plain text containing Markdown
 * symbols:   "You'd have **31 cookies** left."
 *
 *   1. marked.parse()      Markdown → HTML   "You'd have <strong>31 cookies</strong> left."
 *   2. DOMPurify.sanitize  removes anything executable (<script>, onerror=...),
 *                          because this HTML comes from model output and will be
 *                          inserted into the page.
 *
 * Both libraries are loaded by index.html as globals (window.marked, window.DOMPurify).
 */

export function renderMarkdown(text) {
  if (window.marked && window.DOMPurify) {
    // breaks: true → a single newline becomes a line break, like in chat apps.
    return window.DOMPurify.sanitize(window.marked.parse(text, { breaks: true }));
  }
  // Libraries failed to load: show plain text safely rather than breaking.
  return escapeHtml(text).replace(/\n/g, "<br>");
}

function escapeHtml(text) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return text.replace(/[&<>"']/g, (ch) => map[ch]);
}
