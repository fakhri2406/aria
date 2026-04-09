"""ARIA orchestrator — wires up all subsystems and runs the agent graph."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Self

from aria.config import Settings
from aria.graph.builder import build_graph
from aria.graph.state import ARIAState
from aria.mcp.registry import MCPToolRegistry
from aria.mcp.servers import get_default_servers
from aria.nlp.pipeline import NLPPipeline
from aria.prompts.registry import PromptRegistry

logger = logging.getLogger(__name__)

REPORTS_DIR = Path.cwd() / "reports"


class ARIAOrchestrator:
    """Async context manager that connects MCP servers, builds the graph, and runs queries."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._nlp = NLPPipeline()
        self._tool_registry = MCPToolRegistry(get_default_servers())
        self._prompt_registry = PromptRegistry.default()

    async def __aenter__(self) -> Self:
        await self._tool_registry.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._tool_registry.__aexit__(*exc)

    async def run(self, question: str) -> str:
        """Execute the full ARIA pipeline and return the Markdown report."""
        graph = build_graph(
            self._settings,
            self._tool_registry,
            self._nlp,
            self._prompt_registry,
        )

        initial_state: ARIAState = {
            "question": question,
            "sub_queries": [],
            "raw_findings": [],
            "nlp_results": [],
            "critique": "",
            "iteration": 0,
            "report": "",
            "status": "started",
        }

        result = await graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": uuid.uuid4().hex}},
        )

        report: str = result["report"]
        self._save_report(report)
        return report

    @staticmethod
    def _save_report(report: str) -> Path:
        """Write the report to disk and return the file path."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"aria_{timestamp}.md"
        path.write_text(report, encoding="utf-8")
        logger.info("Report saved to %s", path)
        return path
