"""Builds and compiles the ARIA LangGraph agent graph."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
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


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research planning assistant. Decompose the user's research "
            "question into 3-5 specific, searchable sub-queries that together cover "
            "the topic comprehensively. Output ONLY the sub-queries, one per line, "
            "numbered 1-5. Do not add any other text.",
        ),
        ("human", "{question}"),
    ]
)

CRITIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research critic. Evaluate whether the findings adequately "
            "answer the research question. If there are significant coverage gaps, "
            'start your response with "GAPS_FOUND" followed by what is missing. '
            'If the findings are sufficient, start with "PASS" followed by a brief '
            "quality assessment.",
        ),
        ("human", "Research question: {question}\n\nFindings:\n{findings}"),
    ]
)

SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research report writer. Produce a well-structured Markdown "
            "report with these sections: Executive Summary, Key Findings, Detailed "
            "Analysis, and Conclusions. Use the provided summaries, entities, facts, "
            "and critic assessment to write a thorough yet concise report.",
        ),
        (
            "human",
            "Research question: {question}\n\n"
            "Summaries:\n{summaries}\n\n"
            "Key entities:\n{entities}\n\n"
            "Key facts:\n{facts}\n\n"
            "Critic assessment:\n{critique}",
        ),
    ]
)


def _should_continue(state: ARIAState) -> str:
    """Route back to researcher if gaps remain and iterations allow."""
    if state["iteration"] < 2 and state["critique"].startswith("GAPS_FOUND"):
        return "researcher"
    return "synthesizer"


def build_graph(
    settings: Settings,
    tool_registry: MCPToolRegistry,
    nlp: NLPPipeline,
):
    """Construct and compile the ARIA agent graph.

    Returns a compiled LangGraph ``CompiledGraph`` ready for ``ainvoke``.
    """
    llm = ChatAnthropic(
        model_name=settings.claude_model,
        api_key=SecretStr(settings.anthropic_api_key),
        timeout=None,
        stop=None,
    )

    planner = PlannerAgent(llm, PLANNER_PROMPT)
    researcher = ResearcherAgent(tool_registry, nlp)
    critic = CriticAgent(llm, CRITIC_PROMPT)
    synthesizer = SynthesizerAgent(llm, SYNTHESIZER_PROMPT)

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
