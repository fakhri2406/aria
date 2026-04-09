"""Base dataclass for versioned prompt templates."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate


@dataclass(frozen=True)
class VersionedPrompt:
    """A versioned prompt stored as system + human message pair."""

    name: str
    version: str
    system_template: str
    human_template: str

    def to_chat_prompt(self) -> ChatPromptTemplate:
        """Build a LangChain ChatPromptTemplate from this prompt."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_template),
                ("human", self.human_template),
            ]
        )

    def __repr__(self) -> str:
        return f"VersionedPrompt(name={self.name!r}, version={self.version!r})"
