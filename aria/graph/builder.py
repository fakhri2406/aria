"""Builds and compiles the ARIA LangGraph agent graph."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from aria.agents.critic import CriticAgent
from aria.agents.planner import PlannerAgent
from aria.agents.researcher import ResearcherAgent
from aria.agents.synthesizer import SynthesizerAgent
from aria.config import Settings
from aria.graph.state import ARIAState
from aria.mcp.registry import MCPToolRegistry
from aria.nlp.pipeline import NLPPipeline
from aria.prompts.registry import PromptRegistry


def _should_continue(state: ARIAState) -> str:
    """Route back to researcher if gaps remain and iterations allow."""
    if state["iteration"] < 2 and state["critique"].startswith("GAPS_FOUND"):
        return "researcher"
    return "synthesizer"


def build_graph(
    settings: Settings,
    tool_registry: MCPToolRegistry,
    nlp: NLPPipeline,
    prompt_registry: PromptRegistry | None = None,
):
    """Construct and compile the ARIA agent graph.

    Returns a compiled LangGraph ``CompiledGraph`` ready for ``ainvoke``.
    """
    if prompt_registry is None:
        prompt_registry = PromptRegistry.default()

    llm = ChatAnthropic(
        model_name=settings.claude_model,
        api_key=SecretStr(settings.anthropic_api_key),
        timeout=None,
        stop=None,
    )

    planner = PlannerAgent(llm, prompt_registry.get("planner"))
    researcher = ResearcherAgent(tool_registry, nlp)
    critic = CriticAgent(llm, prompt_registry.get("critic"))
    synthesizer = SynthesizerAgent(llm, prompt_registry.get("synthesizer"))

    graph = StateGraph(ARIAState)
    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("critic", critic)
    graph.add_node("synthesizer", synthesizer)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        _should_continue,
        {"researcher": "researcher", "synthesizer": "synthesizer"},
    )
    graph.add_edge("synthesizer", END)

    return graph.compile(checkpointer=MemorySaver())
