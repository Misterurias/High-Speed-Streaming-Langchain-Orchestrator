"""Tests for app/chains.py: the answer chains and the orchestrator that connects them.

These check what the answering model is actually sent for each route.
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from app.chains import build_orchestrator
from app.routing import RouteDecision
from tests.conftest import GENERAL, fake_llm, fake_router


def stream_text(orchestrator, inputs) -> str:
    """Stream the pipeline and join the chunks. The router step is async,
    so the pipeline is run with `astream` inside an event loop."""
    async def gather():
        return "".join([chunk async for chunk in orchestrator.astream(inputs)])
    return asyncio.run(gather())


def run(llm, router, query, history=None) -> str:
    return stream_text(build_orchestrator(llm, router), {"query": query, "history": history or []})


class TestMathChain:
    def test_model_is_given_the_verified_result_to_explain(self):
        llm = fake_llm("That's 31.")
        router = fake_router(RouteDecision(route="math", expression="3*12-5"))
        answer = run(llm, router, "How many cookies are left?")

        assert answer == "That's 31."
        last_prompt = llm.last_messages[-1].content
        assert "Expression: 3*12-5" in last_prompt
        assert "Verified result: 31" in last_prompt

    def test_math_prompt_tells_model_not_to_recompute(self):
        llm = fake_llm()
        run(llm, fake_router(RouteDecision(route="math", expression="2*2")), "2 times 2?")
        assert "Do NOT recompute" in llm.last_messages[0].content


class TestGeneralChain:
    def test_model_is_given_the_question_directly(self):
        llm = fake_llm("Rome was founded in 753 BC.")
        answer = run(llm, fake_router(GENERAL), "Tell me about Rome")
        assert answer == "Rome was founded in 753 BC."
        assert llm.last_messages[-1].content == "Tell me about Rome"


class TestConversationHistory:
    def test_history_sits_between_system_prompt_and_new_question(self):
        llm = fake_llm()
        history = [HumanMessage("What is the capital of France?"), AIMessage("Paris.")]
        run(llm, fake_router(GENERAL), "How many people live there?", history)

        contents = [m.content for m in llm.last_messages]
        assert contents[1:] == ["What is the capital of France?", "Paris.", "How many people live there?"]

    def test_history_is_optional(self):
        orchestrator = build_orchestrator(fake_llm("hi"), fake_router(GENERAL))
        assert stream_text(orchestrator, {"query": "hello"}) == "hi"
