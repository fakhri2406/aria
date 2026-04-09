"""Prompt registry — central lookup for versioned agent prompts."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from aria.prompts.base import VersionedPrompt


class PromptRegistry:
    """Stores and retrieves versioned prompts by agent name."""

    def __init__(self) -> None:
        self._prompts: dict[str, VersionedPrompt] = {}

    def register(self, agent_name: str, prompt: VersionedPrompt) -> None:
        """Register a prompt for a given agent name."""
        self._prompts[agent_name] = prompt

    def get(self, agent_name: str) -> ChatPromptTemplate:
        """Return the ChatPromptTemplate for *agent_name*.

        Raises ``KeyError`` if the agent name is not registered.
        """
        return self._prompts[agent_name].to_chat_prompt()

    def list_prompts(self) -> list[str]:
        """Return a human-readable list of registered prompts (name + version)."""
        return [repr(p) for p in self._prompts.values()]

    @classmethod
    def default(cls) -> PromptRegistry:
        """Create a registry pre-loaded with all default V1 prompts."""
        from aria.prompts.critic_prompt import CRITIC_PROMPT_V1
        from aria.prompts.planner_prompt import PLANNER_PROMPT_V1
        from aria.prompts.synthesizer_prompt import SYNTHESIZER_PROMPT_V1

        registry = cls()
        registry.register("planner", PLANNER_PROMPT_V1)
        registry.register("critic", CRITIC_PROMPT_V1)
        registry.register("synthesizer", SYNTHESIZER_PROMPT_V1)
        return registry
