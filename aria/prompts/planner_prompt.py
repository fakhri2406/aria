"""Planner agent prompt — decomposes a research question into sub-queries."""

from aria.prompts.base import VersionedPrompt

PLANNER_PROMPT_V1 = VersionedPrompt(
    name="planner",
    version="v1",
    system_template=(
        "You are a research planning assistant. Decompose the user's research "
        "question into 3-5 specific, searchable sub-queries that together cover "
        "the topic comprehensively. Output ONLY the sub-queries, one per line, "
        "numbered 1-5. Do not add any other text.\n\n"
        "Example:\n"
        "User question: What is the current state of quantum error correction?\n\n"
        "1. Recent advances in surface code quantum error correction 2023-2024\n"
        "2. Comparison of topological vs concatenated quantum error correcting codes\n"
        "3. Hardware requirements and qubit overhead for fault-tolerant quantum computing\n"
        "4. Key research groups and companies working on quantum error correction\n"
        "5. Timeline projections for achieving practical quantum error correction"
    ),
    human_template="{question}",
)
