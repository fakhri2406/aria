"""Researcher agent — gathers information using MCP tools and NLP pipeline."""

from __future__ import annotations

import asyncio

from aria.graph.state import ARIAState
from aria.mcp.registry import MCPToolRegistry
from aria.nlp.pipeline import NLPPipeline


class ResearcherAgent:
    """Executes sub-queries via MCP tools and processes results through NLP."""

    def __init__(self, tool_registry: MCPToolRegistry, nlp: NLPPipeline) -> None:
        self._registry = tool_registry
        self._nlp = nlp

    async def __call__(self, state: ARIAState) -> dict:
        print("[ARIA] Researcher: gathering information...")
        tools = self._registry.get_langchain_tools()
        raw_findings: list[str] = []
        nlp_results = []

        for i, query in enumerate(state["sub_queries"], 1):
            print(f"[ARIA] Researcher: processing sub-query {i}/{len(state['sub_queries'])}")
            query_results: list[str] = []

            for tool in tools:
                try:
                    result = await tool.ainvoke({"query": query})
                    if result:
                        query_results.append(str(result))
                except Exception as exc:
                    print(f"[ARIA] Researcher: tool '{tool.name}' failed: {exc}")

            combined = "\n".join(query_results)
            if combined.strip():
                raw_findings.append(combined)
                nlp_result = await asyncio.to_thread(self._nlp.process, combined)
                nlp_results.append(nlp_result)

        print(f"[ARIA] Researcher: collected {len(raw_findings)} findings")
        return {"raw_findings": raw_findings, "nlp_results": nlp_results}
