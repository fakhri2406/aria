"""Tests for aria/prompts/ — VersionedPrompt, PromptRegistry, and V1 constants."""

from __future__ import annotations

import dataclasses

import pytest
from langchain_core.prompts import ChatPromptTemplate

from aria.prompts.base import VersionedPrompt
from aria.prompts.critic_prompt import CRITIC_PROMPT_V1
from aria.prompts.planner_prompt import PLANNER_PROMPT_V1
from aria.prompts.registry import PromptRegistry
from aria.prompts.synthesizer_prompt import SYNTHESIZER_PROMPT_V1


class TestVersionedPrompt:
    def test_frozen(self):
        prompt = VersionedPrompt(
            name="test", version="v1", system_template="sys", human_template="human"
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            prompt.name = "changed"  # type: ignore[misc]

    def test_to_chat_prompt_returns_template(self):
        prompt = VersionedPrompt(
            name="test",
            version="v1",
            system_template="You are helpful.",
            human_template="Question: {question}",
        )
        template = prompt.to_chat_prompt()
        assert isinstance(template, ChatPromptTemplate)
        assert "question" in template.input_variables

    def test_repr(self):
        prompt = VersionedPrompt(
            name="planner", version="v2", system_template="s", human_template="h"
        )
        assert repr(prompt) == "VersionedPrompt(name='planner', version='v2')"


class TestPromptRegistry:
    def test_register_and_get(self):
        registry = PromptRegistry()
        prompt = VersionedPrompt(
            name="agent", version="v1", system_template="sys", human_template="{input}"
        )
        registry.register("agent", prompt)
        template = registry.get("agent")
        assert isinstance(template, ChatPromptTemplate)

    def test_get_missing_raises_keyerror(self):
        registry = PromptRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_prompts(self):
        registry = PromptRegistry()
        prompt = VersionedPrompt(
            name="agent", version="v1", system_template="s", human_template="h"
        )
        registry.register("agent", prompt)
        prompts = registry.list_prompts()
        assert len(prompts) == 1
        assert "agent" in prompts[0]
        assert "v1" in prompts[0]

    def test_default_registry(self):
        registry = PromptRegistry.default()
        prompts = registry.list_prompts()
        assert len(prompts) == 3

        names = " ".join(prompts)
        assert "planner" in names
        assert "critic" in names
        assert "synthesizer" in names

        for agent in ("planner", "critic", "synthesizer"):
            assert isinstance(registry.get(agent), ChatPromptTemplate)


class TestPromptConstants:
    def test_planner_prompt_v1(self):
        assert PLANNER_PROMPT_V1.name == "planner"
        assert PLANNER_PROMPT_V1.version == "v1"
        assert "{question}" in PLANNER_PROMPT_V1.human_template

    def test_critic_prompt_v1(self):
        assert CRITIC_PROMPT_V1.name == "critic"
        assert CRITIC_PROMPT_V1.version == "v1"
        assert "{question}" in CRITIC_PROMPT_V1.human_template
        assert "{findings}" in CRITIC_PROMPT_V1.human_template

    def test_synthesizer_prompt_v1(self):
        assert SYNTHESIZER_PROMPT_V1.name == "synthesizer"
        assert SYNTHESIZER_PROMPT_V1.version == "v1"
        for var in ("{question}", "{summaries}", "{entities}", "{facts}", "{critique}"):
            assert var in SYNTHESIZER_PROMPT_V1.human_template
