"""
The shape of a request to POST /ask, and the rules it must follow.

FastAPI checks every incoming body against `AskRequest` before our code runs.
A body that breaks a rule (blank query, too long, unknown role) is rejected
with a 422 response automatically.

Example body:
    {
      "query": "Now double that",
      "history": [
        {"role": "user", "content": "What is 3 * 12?"},
        {"role": "assistant", "content": "3 * 12 = 36."}
      ]
    }
`history` is optional; {"query": "..."} on its own is valid.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator

from app.config import MAX_HISTORY_TURNS, MAX_QUERY_CHARS, MAX_TURN_CHARS


class ChatTurn(BaseModel):
    """One earlier message. Only "user" and "assistant" are allowed, so a
    client can't slip in a fake "system" instruction."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_TURN_CHARS)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)

    @field_validator("query")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value

    def history_as_messages(self) -> list[BaseMessage]:
        """Convert to the message objects LangChain prompts expect."""
        return [
            HumanMessage(turn.content) if turn.role == "user" else AIMessage(turn.content)
            for turn in self.history
        ]
