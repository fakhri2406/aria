"""Synthesizer agent — compiles findings into a structured Markdown report."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from aria.graph.state import ARIAState


class SynthesizerAgent:
    """Produces a final structured Markdown report from all findings."""

    def __init__(self, llm: ChatAnthropic, prompt_template: ChatPromptTemplate) -> None:
        self._chain = prompt_template | llm

    async def __call__(self, state: ARIAState) -> dict:
        print("[ARIA] Synthesizer: compiling final report...")

        summaries = "\n\n".join(r.summary for r in state["nlp_results"] if r.summary)

        entities_parts: list[str] = []
        for r in state["nlp_results"]:
            for name, label in r.entities.named_entities:
                entities_parts.append(f"- {name} ({label})")
        entities_text = "\n".join(entities_parts)

        facts = "\n".join(f"- {fact}" for r in state["nlp_results"] for fact in r.unique_facts)

        response = await self._chain.ainvoke(
            {
                "question": state["question"],
                "summaries": summaries or "(none)",
                "entities": entities_text or "(none)",
                "facts": facts or "(none)",
                "critique": state.get("critique", ""),
            }
        )

        report = str(response.content).strip()
        print("[ARIA] Synthesizer: report complete")
        return {"report": report, "status": "done"}
