"""LangChain orchestration: route the query, then run the matching chain.

The whole pipeline is a single LangChain Runnable:

    {"query", "history"} ──> assign(plan = router) ──> RunnableBranch
                                                         ├─ plan.route == "math"  -> math_chain
                                                         └─ otherwise             -> general_chain

The router step (named "router") is:
  1. Fast path: if the query is already an arithmetic expression, route to
     math with no LLM call at all.
  2. Otherwise one small LLM call with structured output returns both the
     route AND, for math, the expression to compute. Routing and extraction
     happen in the same call, so math questions don't need an extra round trip.
     The router sees the conversation history, so follow-ups like
     "now double that" resolve against earlier answers.
  3. The expression is computed by the deterministic safe_math tool (the LLM
     never does the arithmetic). If extraction or evaluation fails, or the
     router call itself errors, the request degrades to the general chain
     instead of failing.

Both answer chains use the same model tagged "answer", so the API layer can
stream exactly the answer tokens and ignore the router's internal tokens.

"history" is optional: a list of prior HumanMessage/AIMessage turns. Callers
that only send {"query": ...} get single-turn behavior.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field

from app.safe_math import MathError, evaluate, format_result, is_bare_expression

logger = logging.getLogger(__name__)

ANSWER_TAG = "answer"
ROUTER_RUN_NAME = "router"

Route = Literal["math", "general"]


class RouteDecision(BaseModel):
    """Structured output the router LLM must return."""

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
    """What the router decided. Sent to the client as the first SSE event."""

    route: Route
    routed_by: Literal["fast_path", "llm", "fallback"]
    expression: str | None = None
    result: str | None = None
    note: str | None = None


HISTORY = MessagesPlaceholder("history", optional=True)

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query router. Classify the user's LATEST message, using the "
            "earlier conversation only to resolve references like 'that' or 'it'.\n"
            "- route='math' when the answer requires computing a number (arithmetic, "
            "percentages, unit-free word problems). Also write the single expression "
            "that computes it, substituting any numbers taken from the conversation.\n"
            "- route='general' for everything else, including questions ABOUT math "
            "concepts that need no calculation (e.g. 'what is a prime number?').",
        ),
        HISTORY,
        ("human", "{query}"),
    ]
)

GENERAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, concise assistant. Answer clearly and accurately. "
            "Use Markdown when it helps (lists, code blocks, bold).",
        ),
        HISTORY,
        ("human", "{query}"),
    ]
)

MATH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a math assistant. A verified calculator has already computed the "
            "answer. Do NOT recompute it and never contradict it. State the answer and "
            "show the expression that was used. If the user asked for an explanation, "
            "explain briefly; otherwise keep it to 1-3 sentences.",
        ),
        HISTORY,
        (
            "human",
            "Question: {query}\nExpression: {expression}\nVerified result: {result}",
        ),
    ]
)


def build_llm_router(router_llm: BaseChatModel) -> Runnable:
    """prompt -> LLM -> RouteDecision (validated by Pydantic)."""
    return ROUTER_PROMPT | router_llm.with_structured_output(RouteDecision)


def _compute(expression: str, routed_by: Literal["fast_path", "llm"]) -> Plan:
    result = format_result(evaluate(expression))
    return Plan(route="math", routed_by=routed_by, expression=expression, result=result)


def build_orchestrator(answer_llm: BaseChatModel, llm_router: Runnable) -> Runnable:
    """Assemble the routing + answering pipeline as one Runnable."""

    async def plan_query(inputs: dict[str, Any]) -> Plan:
        query: str = inputs["query"]
        history = inputs.get("history") or []

        # 1. Fast path: "12 * (3 + 4)" needs no model to classify.
        if is_bare_expression(query):
            return _compute(query.strip().rstrip("=?").strip(), "fast_path")

        # 2. LLM router (classification + expression extraction in one call).
        try:
            decision = await llm_router.ainvoke({"query": query, "history": history})
        except Exception:
            logger.exception("Router call failed; falling back to general chain")
            return Plan(route="general", routed_by="fallback", note="router unavailable")

        # Models occasionally reply in plain text instead of the structured
        # format; the parser then returns None. Treat that as "no decision".
        if not isinstance(decision, RouteDecision):
            logger.warning("Router returned no structured decision: %r", decision)
            return Plan(route="general", routed_by="fallback", note="router gave no decision")

        if decision.route == "general":
            return Plan(route="general", routed_by="llm")

        # 3. Deterministic tool does the arithmetic.
        if decision.expression:
            try:
                return _compute(decision.expression, "llm")
            except MathError as exc:
                logger.info("Math tool rejected %r: %s", decision.expression, exc)
                note = f"math tool could not evaluate the expression ({exc})"
        else:
            note = "router chose math but gave no expression"
        return Plan(route="general", routed_by="fallback", note=note)

    answer = answer_llm.with_config(tags=[ANSWER_TAG])

    general_chain = (GENERAL_PROMPT | answer | StrOutputParser()).with_config(
        run_name="general_chain"
    )

    math_chain = (
        RunnableLambda(
            lambda x: {
                "query": x["query"],
                "history": x.get("history") or [],
                "expression": x["plan"].expression,
                "result": x["plan"].result,
            },
            name="math_inputs",
        )
        | MATH_PROMPT
        | answer
        | StrOutputParser()
    ).with_config(run_name="math_chain")

    return (
        RunnablePassthrough.assign(plan=RunnableLambda(plan_query, name=ROUTER_RUN_NAME))
        | RunnableBranch(
            (lambda x: x["plan"].route == "math", math_chain),
            general_chain,
        )
    ).with_config(run_name="orchestrator")
