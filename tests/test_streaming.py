"""Tests for app/streaming.py: turning the pipeline's progress into SSE events."""

import asyncio
import json

from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableLambda

from app.chains import build_orchestrator
from app.routing import RouteDecision
from app.streaming import chunk_text, stream_answer
from tests.conftest import GENERAL, fake_llm, fake_router


def collect(orchestrator, query="hello") -> list[tuple[str, dict]]:
    async def gather():
        return [
            (e["event"], json.loads(e["data"]))
            async for e in stream_answer(orchestrator, query, [], "req123")
        ]
    return asyncio.run(gather())


class TestEventOrder:
    def test_stream_is_route_then_tokens_then_done(self):
        events = collect(build_orchestrator(fake_llm("one two three"), fake_router(GENERAL)))
        names = [name for name, _ in events]
        assert names[0] == "route"
        assert set(names[1:-1]) == {"token"}
        assert names[-1] == "done"

    def test_answer_arrives_in_several_chunks_that_join_back_up(self):
        events = collect(build_orchestrator(fake_llm("Rome was founded in 753 BC."), fake_router(GENERAL)))
        tokens = [data["text"] for name, data in events if name == "token"]
        assert len(tokens) > 1
        assert "".join(tokens) == "Rome was founded in 753 BC."

    def test_route_event_describes_the_plan(self):
        router = fake_router(RouteDecision(route="math", expression="6*7"))
        events = collect(build_orchestrator(fake_llm(), router))
        assert events[0] == ("route", {"route": "math", "routed_by": "llm", "expression": "6*7", "result": "42"})

    def test_done_event_reports_timing_and_request_id(self):
        events = collect(build_orchestrator(fake_llm(), fake_router(GENERAL)))
        done = events[-1][1]
        assert done["request_id"] == "req123"
        assert done["route"] == "general"
        assert isinstance(done["ttft_ms"], int) and isinstance(done["total_ms"], int)


class TestOnlyAnswerTokensAreStreamed:
    def test_tokens_from_the_router_model_are_not_sent_to_the_user(self):
        """The router also calls a model. Its output must stay internal."""
        router_model = fake_llm("ROUTER-INTERNAL-OUTPUT")

        async def router_that_uses_a_model(inputs):
            async for _ in router_model.astream("classify"):
                pass
            return GENERAL

        orchestrator = build_orchestrator(fake_llm("visible answer"), RunnableLambda(router_that_uses_a_model))
        text = "".join(data["text"] for name, data in collect(orchestrator) if name == "token")
        assert text == "visible answer"

    def test_empty_chunks_are_not_sent(self):
        """Models sometimes stream empty chunks (e.g. at the start). They're skipped."""

        class EmptyFirstChunk(type(fake_llm())):
            async def _astream(self, messages, *args, **kwargs):
                from langchain_core.outputs import ChatGenerationChunk
                yield ChatGenerationChunk(message=AIMessageChunk(content=""))
                async for chunk in super()._astream(messages, *args, **kwargs):
                    yield chunk

        from itertools import cycle
        from langchain_core.messages import AIMessage
        llm = EmptyFirstChunk(messages=cycle([AIMessage(content="hi there")]))
        tokens = [d["text"] for n, d in collect(build_orchestrator(llm, fake_router(GENERAL))) if n == "token"]
        assert "" not in tokens and "".join(tokens) == "hi there"


class TestErrors:
    def test_failure_mid_stream_ends_with_a_safe_error_event(self):
        async def broken(_):
            raise RuntimeError("secret internal detail")

        events = collect(RunnableLambda(broken))
        assert events[-1][0] == "error"
        assert events[-1][1]["request_id"] == "req123"
        assert "secret internal detail" not in events[-1][1]["message"]


class TestChunkText:
    def test_reads_plain_string_content(self):
        assert chunk_text(AIMessageChunk(content="hello")) == "hello"

    def test_reads_list_of_content_blocks_and_skips_non_text(self):
        chunk = AIMessageChunk(content=[
            {"type": "text", "text": "Hi "},
            {"type": "tool_use", "id": "x", "name": "t", "input": {}},
            {"type": "text", "text": "there"},
        ])
        assert chunk_text(chunk) == "Hi there"

    def test_returns_empty_string_for_anything_else(self):
        class OddChunk:
            content = 42
        assert chunk_text(object()) == ""
        assert chunk_text(OddChunk()) == ""
