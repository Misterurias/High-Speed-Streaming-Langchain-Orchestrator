"""Tiny terminal client for demos: prints tokens as they arrive.

Usage:  python scripts/ask.py "What is 15% of 240?"
Quote the question: zsh treats ? and * as filename patterns otherwise.
"""

import json
import sys

import httpx

URL = "http://localhost:8000/ask"


def main() -> int:
    query = " ".join(sys.argv[1:]) or "What is 15% of 240?"
    event = None
    finished = False
    try:
        with httpx.stream("POST", URL, json={"query": query}, timeout=60) as resp:
            if resp.status_code != 200:
                print(f"[http {resp.status_code}] {resp.read().decode()}")
                return 1
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    if event == "route":
                        print(f"[route] {data}\n")
                    elif event == "token":
                        print(data["text"], end="", flush=True)
                    elif event == "done":
                        finished = True
                        print(f"\n\n[done] ttft={data['ttft_ms']}ms total={data['total_ms']}ms")
                    elif event == "error":
                        finished = True
                        print(f"\n[error] {data['message']} (request {data['request_id']})")
    except httpx.ConnectError:
        print(f"[client] Could not connect to {URL}. Is the server running?")
        return 1
    except (httpx.RemoteProtocolError, httpx.ReadError):
        print("\n[client] The server closed the connection mid-stream. "
              "Check the server terminal: it most likely restarted or crashed.")
        return 1

    if not finished:
        print("\n[client] Stream ended without a 'done' event.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
