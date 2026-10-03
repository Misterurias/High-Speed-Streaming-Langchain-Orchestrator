/**
 * Reads a Server-Sent Events (SSE) stream from a fetch() response.
 *
 * The server sends text like this, one message per blank-line-separated block:
 *
 *     event: token
 *     data: {"text": "Hello"}
 *
 *     event: token
 *     data: {"text": " world"}
 *
 * Why not the browser's built-in EventSource?
 *   EventSource can only make GET requests. We need POST, to send the question
 *   and conversation history in the request body. So we read the response body
 *   ourselves and split it into messages.
 *
 * The tricky part: network chunks don't line up with messages. One chunk can
 * hold half a message, or three and a half. So incoming text goes into a
 * buffer, and only complete messages (ending in a blank line) are taken out.
 */

/**
 * Yield each complete SSE message as { event, data } as soon as it arrives.
 * `data` is parsed from JSON.
 */
export async function* readSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) return;

    // Normalize Windows-style line endings so we only have to look for "\n\n".
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    // Take out every complete message currently in the buffer.
    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const message = parseBlock(block);
      if (message) yield message;
    }
  }
}

/** Turn one "event: …\ndata: …" block into { event, data }, or null. */
function parseBlock(block) {
  let event = "message";
  let data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith(":")) continue; // a comment, e.g. the server's keep-alive ping
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  return data ? { event, data: JSON.parse(data) } : null;
}
