"""Tests for app/routing.py: how a query is sent to math or general.

These call the router step directly (no web server) to check each decision path.
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.routing import Plan, RouteDecision, build_llm_router, make_router_step
from tests.conftest import GENERAL, fake_router


def decide(router, query, history=None) -> Plan:
    step = make_router_step(router)
    return asyncio.run(step.ainvoke({"query": query, "history": history or []}))


class TestFastPath:
    def test_bare_expression_is_calculated_without_calling_the_llm(self):
        calls = []
        plan = decide(fake_router(GENERAL, calls), "12 * (3 + 4)")
        assert calls == []
        assert plan == Plan(route="math", routed_by="fast_path", expression="12 * (3 + 4)", result="84")

    def test_trailing_equals_or_question_mark_is_ignored(self):
        plan = decide(fake_router(GENERAL), "2+2=")
        assert plan.expression == "2+2" and plan.result == "4"


class TestLLMRouter:
    def test_math_decision_is_computed_by_the_calculator(self):
        router = fake_router(RouteDecision(route="math", expression="3 * 12 - 5"))
        plan = decide(router, "3 boxes of 12 cookies, I eat 5. How many left?")
        assert plan == Plan(route="math", routed_by="llm", expression="3 * 12 - 5", result="31")

    def test_general_decision_goes_to_general_chain(self):
        plan = decide(fake_router(GENERAL), "Tell me about Rome")
        assert plan == Plan(route="general", routed_by="llm")

    def test_router_receives_conversation_history(self):
        calls = []
        history = [HumanMessage("What is 3 * 12?"), AIMessage("36")]
        decide(fake_router(GENERAL, calls), "Now double that", history)
        assert calls[0]["history"] == history
        assert calls[0]["query"] == "Now double that"


class TestFallbacks:
    """Every failure degrades to the general chain, so the user still gets an answer."""

    @pytest.mark.parametrize(
        "router_reply, note",
        [
            (RuntimeError("provider down"), "router unavailable"),
            (None, "router gave no decision"),
            (RouteDecision(route="math", expression=None), "router chose math but gave no expression"),
        ],
        ids=["router_raises", "router_returns_nothing", "math_without_expression"],
    )
    def test_router_problems_fall_back_to_general(self, router_reply, note):
        plan = decide(fake_router(router_reply), "some question")
        assert plan == Plan(route="general", routed_by="fallback", note=note)

    def test_expression_the_calculator_rejects_falls_back_to_general(self):
        router = fake_router(RouteDecision(route="math", expression="__import__('os')"))
        plan = decide(router, "do something sneaky")
        assert plan.route == "general" and plan.routed_by == "fallback"
        assert plan.note.startswith("calculator could not evaluate")


class TestRouterChain:
    def test_llm_router_is_prompt_piped_into_structured_output(self):
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        class StructuredFake(GenericFakeChatModel):
            def with_structured_output(self, schema, **kwargs):
                from langchain_core.runnables import RunnableLambda
                return RunnableLambda(lambda _: schema(route="general"))

        router = build_llm_router(StructuredFake(messages=iter([])))
        result = router.invoke({"query": "hi", "history": []})
        assert result == RouteDecision(route="general")
