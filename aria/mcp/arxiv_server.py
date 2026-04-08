"""MCP server exposing arXiv paper search as a tool."""

from __future__ import annotations

import asyncio

import arxiv
from mcp.server.fastmcp import FastMCP

server = FastMCP("arxiv-search")


def _search_sync(query: str, max_results: int) -> list[dict]:
    """Run a synchronous arXiv search (called via asyncio.to_thread)."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [
        {
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "summary": result.summary,
            "published": result.published.isoformat(),
            "url": result.entry_id,
        }
        for result in client.results(search)
    ]


@server.tool()
async def fetch_arxiv_papers(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv for academic papers matching the query.

    Returns a list of papers with title, authors, summary, published date, and URL.
    """
    return await asyncio.to_thread(_search_sync, query, max_results)


if __name__ == "__main__":
    server.run()
