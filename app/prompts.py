"""
Every prompt the app sends to a model, in one place.

A LangChain `ChatPromptTemplate` is a reusable message list with blanks in it.
`{query}` is filled with the user's question, and `MessagesPlaceholder("history")`
is replaced with the earlier conversation (or nothing, on the first message).

There are three prompts, one per job:
    ROUTER_PROMPT   decides math vs. general         (used by routing.py)
    GENERAL_PROMPT  answers ordinary questions        (used by chains.py)
    MATH_PROMPT     explains a pre-computed result    (used by chains.py)
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# The earlier conversation. `optional=True` means callers can leave it out.
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

# The calculator has already produced the number before this prompt runs.
# The model's only job is to explain it, never to redo the arithmetic.
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
        ("human", "Question: {query}\nExpression: {expression}\nVerified result: {result}"),
    ]
)
