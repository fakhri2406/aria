# ARIA (Agentic Research & Intelligence Assistant)

## What It Does

ARIA takes a natural-language research question, breaks it into sub-queries, gathers information from the web and academic papers, and produces a structured Markdown report. It uses four specialised AI agents coordinated through a LangGraph state machine: a **Planner** decomposes the question, a **Researcher** collects data via MCP tool servers, a **Critic** evaluates coverage gaps (triggering re-research if needed), and a **Synthesizer** writes the final report. An NLP pipeline (entity extraction, summarisation, semantic deduplication) processes raw findings between agent steps.

```
CLI Report ─▶ Orchestrator ─▶ Planner ─▶ Researcher ─▶ Critic ─▶ Synthesizer ─▶ Report
```

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Agent orchestration | LangGraph + LangChain | State-machine graph connecting four agents with conditional looping |
| LLM | Claude (via `langchain-anthropic`) | Reasoning backbone for planning, critiquing, and synthesising |
| Tool integration | Model Context Protocol (MCP) | Connects agents to external data sources over stdio |
| Web search | Tavily MCP server | Retrieves web results for research sub-queries |
| Academic search | Custom FastMCP server | Queries arXiv API for relevant papers |
| NLP — entities | spaCy (`en_core_web_sm`) | Named entity recognition, noun phrase extraction, key term ranking |
| NLP — summarisation | HuggingFace Transformers (`bart-large-cnn`) | Extractive summarisation with automatic chunking for long texts |
| NLP — deduplication | sentence-transformers (`all-MiniLM-L6-v2`) | Cosine-similarity-based removal of near-duplicate findings |
| CLI | Typer + Rich | Terminal interface with spinners, coloured panels, and config display |
| Configuration | Pydantic Settings + python-dotenv | Typed settings loaded from `.env` with validation |
| Prompt management | Custom `VersionedPrompt` registry | Versioned, template-based prompts decoupled from agent logic |

## Architecture

ARIA runs as four async agents on a compiled LangGraph `StateGraph`:

1. **Planner** — receives the research question and produces 3–5 specific sub-queries using an LLM chain.
2. **Researcher** — for each sub-query, invokes all discovered MCP tools (Tavily web search, arXiv paper search), then passes the raw results through the NLP pipeline (entity extraction, summarisation, semantic deduplication).
3. **Critic** — evaluates whether the collected findings adequately answer the original question. Outputs `PASS` or `GAPS_FOUND`. If gaps are found and fewer than 2 iterations have run, the graph loops back to the Researcher.
4. **Synthesizer** — compiles all processed findings, entities, facts, and the critic's assessment into a structured Markdown report with Executive Summary, Key Findings, Detailed Analysis, and Conclusions.

The **MCP layer** (`MCPToolRegistry`) connects to tool servers over stdio, discovers available tools at runtime, and wraps them as LangChain `StructuredTool` objects with Pydantic-validated arguments. New data sources can be added by registering additional MCP server configurations.

The **NLP pipeline** runs synchronously (offloaded to a thread in the async context) and chains three stages: spaCy entity extraction, BART summarisation (with automatic chunking for texts over 1024 tokens), and sentence-transformer-based semantic deduplication.

## Quick Start

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), Node.js (for Tavily MCP server)

```bash
# Clone and install
git clone https://github.com/<your-username>/aria.git
cd aria
uv pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_sm

# Configure environment
cp .env.example .env
# Edit .env — set at minimum:
#   ANTHROPIC_API_KEY=sk-...
#   TAVILY_API_KEY=tvly-...    (optional, enables web search)

# Run a research query
aria run "What are the latest advances in quantum error correction?"

# Other commands
aria version          # Print version
aria config           # Show current settings (keys masked)
aria tools            # List discovered MCP tools
```

```bash
# Development
make dev              # Install with dev dependencies
pytest                # Run tests
ruff check .          # Lint
ruff format .         # Format
```

## Project Structure

```
aria/
├── __init__.py
├── __main__.py              # Typer CLI (run, version, config, tools)
├── config.py                # Pydantic Settings from .env
├── orchestrator.py          # Async context manager wiring all subsystems
├── agents/
│   ├── planner.py           # Decomposes question → sub-queries
│   ├── researcher.py        # Gathers data via MCP tools + NLP
│   ├── critic.py            # Evaluates coverage, flags gaps
│   └── synthesizer.py       # Produces final Markdown report
├── mcp/
│   ├── client.py            # Async stdio MCP client
│   ├── servers.py           # Server configs (Tavily, ArXiv)
│   ├── arxiv_server.py      # FastMCP server for arXiv search
│   └── registry.py          # Tool discovery + LangChain wrapping
├── nlp/
│   ├── pipeline.py          # NLPPipeline orchestrator
│   ├── extractor.py         # spaCy entity extraction
│   ├── summarizer.py        # BART summarisation with chunking
│   └── deduplicator.py      # Sentence-transformer deduplication
├── prompts/
│   ├── base.py              # VersionedPrompt dataclass
│   ├── planner_prompt.py    # Planner V1 prompt
│   ├── critic_prompt.py     # Critic V1 prompt
│   ├── synthesizer_prompt.py# Synthesizer V1 prompt
│   └── registry.py          # PromptRegistry lookup
└── graph/
    ├── state.py             # ARIAState TypedDict
    └── builder.py           # build_graph() → compiled StateGraph
tests/
├── test_nlp_pipeline.py     # NLP component tests
├── test_graph_state.py      # State contract tests
└── test_prompts.py          # Prompt + registry tests
reports/                     # Generated reports (gitignored)
```
