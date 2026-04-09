"""Typed state definition for the ARIA agent graph."""

from __future__ import annotations

from typing import TypedDict

from aria.nlp.pipeline import NLPResult


class ARIAState(TypedDict):
    """State flowing through the ARIA LangGraph agent graph."""

    question: str
    sub_queries: list[str]
    raw_findings: list[str]
    nlp_results: list[NLPResult]
    critique: str
    iteration: int
    report: str
    status: str
