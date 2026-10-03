"""
Step 2: the two answer chains, and the orchestrator that connects everything.

What LangChain gives us here
----------------------------
Every piece below is a "Runnable": an object you can call with `.invoke()`
(get the whole result), `.stream()` (get it piece by piece), or
`.astream_events()` (get a play-by-play of every step, which streaming.py uses).
Prompts, models, parsers, and our own Python functions all share that interface,
so they can be snapped together with `|` and the result is a Runnable too.

The full pipeline
-----------------

    input: {"query": "...", "history": [...]}
       │
       ▼
    RunnablePassthrough.assign(plan=router_step)
       │   Runs the router (routing.py) and ADDS its result to the input dict:
       │   {"query": ..., "history": ..., "plan": Plan(route="math", ...)}
       ▼
    RunnableBranch  ── an if/else for Runnables
       ├─ if plan.route == "math" →  math_chain:     MATH_PROMPT    | model | to-text
       └─ otherwise               →  general_chain:  GENERAL_PROMPT | model | to-text
       │
       ▼
    output: the answer text (streamed token by token)
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

from app.prompts import GENERAL_PROMPT, MATH_PROMPT
from app.routing import make_router_step

# A label attached to the answering model. The router also calls a model, but
# only the tokens tagged with this label are streamed to the user.
ANSWER_TAG = "answer"


def build_general_chain(answer_llm: BaseChatModel) -> Runnable:
    """Ordinary questions: prompt → model → plain text."""
    return (GENERAL_PROMPT | answer_llm | StrOutputParser()).with_config(run_name="general_chain")


def build_math_chain(answer_llm: BaseChatModel) -> Runnable:
    """Math questions: the calculator already ran in the router, so this chain
    only hands the verified result to the model to explain."""

    def pick_math_inputs(state: dict[str, Any]) -> dict[str, Any]:
        plan = state["plan"]
        return {
            "query": state["query"],
            "history": state.get("history") or [],
            "expression": plan.expression,
            "result": plan.result,
        }

    return (
        RunnableLambda(pick_math_inputs, name="math_inputs")
        | MATH_PROMPT
        | answer_llm
        | StrOutputParser()
    ).with_config(run_name="math_chain")


def build_orchestrator(answer_llm: BaseChatModel, llm_router: Runnable) -> Runnable:
    """Connect router → branch → answer chain into one Runnable.

    The models are passed in rather than created here, so tests can pass fakes.
    """
    tagged_llm = answer_llm.with_config(tags=[ANSWER_TAG])

    def is_math(state: dict[str, Any]) -> bool:
        return state["plan"].route == "math"

    return (
        RunnablePassthrough.assign(plan=make_router_step(llm_router))
        | RunnableBranch(
            (is_math, build_math_chain(tagged_llm)),
            build_general_chain(tagged_llm),  # the default branch
        )
    ).with_config(run_name="orchestrator")
