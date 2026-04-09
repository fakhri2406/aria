"""Critic agent — evaluates research findings for coverage gaps."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from aria.graph.state import ARIAState


class CriticAgent:
    """Evaluates findings for relevance and completeness, flags gaps."""

    def __init__(self, llm: ChatAnthropic, prompt_template: ChatPromptTemplate) -> None:
        self._chain = prompt_template | llm

    async def __call__(self, state: ARIAState) -> dict:
        iteration = state["iteration"] + 1
        print(f"[ARIA] Critic: evaluating findings (iteration {iteration})...")

        summaries = "\n".join(r.summary for r in state["nlp_results"] if r.summary)
        response = await self._chain.ainvoke(
            {
                "question": state["question"],
                "findings": summaries or "(no findings)",
            }
        )

        critique = str(response.content).strip()
        has_gaps = critique.startswith("GAPS_FOUND")
        print(f"[ARIA] Critic: {'gaps found' if has_gaps else 'passed'}")
        return {"critique": critique, "iteration": iteration}
