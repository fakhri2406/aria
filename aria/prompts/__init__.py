"""Versioned prompt templates for ARIA agents."""

from aria.prompts.base import VersionedPrompt
from aria.prompts.critic_prompt import CRITIC_PROMPT_V1
from aria.prompts.planner_prompt import PLANNER_PROMPT_V1
from aria.prompts.registry import PromptRegistry
from aria.prompts.synthesizer_prompt import SYNTHESIZER_PROMPT_V1

__all__ = [
    "CRITIC_PROMPT_V1",
    "PLANNER_PROMPT_V1",
    "PromptRegistry",
    "SYNTHESIZER_PROMPT_V1",
    "VersionedPrompt",
]
