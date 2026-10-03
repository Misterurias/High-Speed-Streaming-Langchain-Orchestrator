"""
Step 1 of every request: decide WHICH chain should answer.

The result is a `Plan`, for example:
    Plan(route="math", routed_by="llm", expression="3*12-5", result="31")
    Plan(route="general", routed_by="llm")

How the decision is made, cheapest check first:

    user query
        │
        ├─ 1. FAST PATH ── is the query already a bare expression like "12*(3+4)"?
        │                    yes → compute it now, route = math. No LLM call at all.
        │
        ├─ 2. LLM ROUTER ─ one model call returns a RouteDecision:
        │                    {"route": "math", "expression": "3*12-5"}  or  {"route": "general"}
        │                    Routing and "write the expression" happen in the SAME call,
        │                    so a math question doesn't need an extra round trip.
        │
        └─ 3. CALCULATOR ─ safe_math evaluates the expression. The LLM never does
                             the arithmetic itself, because models get math wrong.

If anything goes wrong (the router call fails, returns nothing usable, or writes
an expression the calculator rejects), the plan becomes route="general" with
routed_by="fallback". The user still gets an answer instead of an error.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from app.prompts import ROUTER_PROMPT
from app.safe_math import MathError, evaluate, format_result, is_bare_expression

logger = logging.getLogger(__name__)

# The LangChain run name of the router step. streaming.py listens for the
# event that ends this step so it can tell the browser which route was chosen.
ROUTER_STEP_NAME = "router"

Route = Literal["math", "general"]


class RouteDecision(BaseModel):
    """The exact shape the router LLM must reply with.

    `with_structured_output(RouteDecision)` turns this class into a schema the
    model is forced to follow, then validates the reply back into this object.
    The field descriptions below are sent to the model as instructions.
    """

    route: Route = Field(
        description="'math' if answering needs an arithmetic calculation, otherwise 'general'."
    )
    expression: str | None = Field(
        default=None,
        description=(
            "Only when route is 'math': ONE arithmetic expression that computes the answer, "
            "using numbers, + - * / // % ** ( ), the constants pi and e, and the functions "
            "sqrt, abs, round, floor, ceil, log, log10, exp, sin, cos, tan, factorial. "
            "No variables, units, or words. Null when route is 'general'."
        ),
    )


class Plan(BaseModel):
    """The router's final decision, after the calculator has run.

    It's also sent to the browser as the first SSE event (`route`), which is
    what the green "Calculator" / purple "General" badge in the UI shows.
    """

    route: Route
    routed_by: Literal["fast_path", "llm", "fallback"]
    expression: str | None = None
    result: str | None = None
    note: str | None = None  # why a fallback happened, for logs and debugging


def build_llm_router(router_llm: BaseChatModel) -> Runnable:
    """prompt → model → RouteDecision.

    The `|` operator is LangChain's pipe: the output of the left side becomes
    the input of the right side, like a Unix pipe.
    """
    return ROUTER_PROMPT | router_llm.with_structured_output(RouteDecision)


def make_router_step(llm_router: Runnable) -> Runnable:
    """Wrap the decision logic as a named LangChain step.

    Naming it (ROUTER_STEP_NAME) lets streaming.py spot the moment it finishes.
    """

    async def decide_route(inputs: dict[str, Any]) -> Plan:
        query: str = inputs["query"]
        history = inputs.get("history") or []

        # 1. Fast path: the query is already math, so there's nothing to classify.
        if is_bare_expression(query):
            return _calculate(query.strip().rstrip("=?").strip(), routed_by="fast_path")

        # 2. Ask the router model.
        try:
            decision = await llm_router.ainvoke({"query": query, "history": history})
        except Exception:
            logger.exception("Router call failed; falling back to general chain")
            return _fallback("router unavailable")

        # Models occasionally ignore the schema and reply in plain text; the
        # parser then returns None instead of a RouteDecision.
        if not isinstance(decision, RouteDecision):
            logger.warning("Router returned no structured decision: %r", decision)
            return _fallback("router gave no decision")

        if decision.route == "general":
            return Plan(route="general", routed_by="llm")

        if not decision.expression:
            return _fallback("router chose math but gave no expression")

        # 3. The calculator does the arithmetic.
        try:
            return _calculate(decision.expression, routed_by="llm")
        except MathError as exc:
            logger.info("Calculator rejected %r: %s", decision.expression, exc)
            return _fallback(f"calculator could not evaluate the expression ({exc})")

    return RunnableLambda(decide_route, name=ROUTER_STEP_NAME)


def _calculate(expression: str, routed_by: Literal["fast_path", "llm"]) -> Plan:
    result = format_result(evaluate(expression))
    return Plan(route="math", routed_by=routed_by, expression=expression, result=result)


def _fallback(note: str) -> Plan:
    return Plan(route="general", routed_by="fallback", note=note)
