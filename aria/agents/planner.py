"""Planner agent — decomposes a research question into sub-queries."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from aria.graph.state import ARIAState


class PlannerAgent:
    """Decomposes the research question into 3-5 targeted sub-queries."""

    def __init__(self, llm: ChatAnthropic, prompt_template: ChatPromptTemplate) -> None:
        self._chain = prompt_template | llm

    async def __call__(self, state: ARIAState) -> dict:
        print("[ARIA] Planner: decomposing research question...")
        response = await self._chain.ainvoke({"question": state["question"]})
        content = str(response.content)

        lines = [
            line.strip().lstrip("0123456789.-) ")
            for line in content.strip().splitlines()
            if line.strip()
        ]
        sub_queries = [line for line in lines if line][:5]

        print(f"[ARIA] Planner: generated {len(sub_queries)} sub-queries")
        return {"sub_queries": sub_queries}
