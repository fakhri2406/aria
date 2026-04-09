"""Tests for aria/graph/state.py — ARIAState TypedDict contract."""

from __future__ import annotations

from aria.graph.state import ARIAState
from aria.nlp.pipeline import NLPResult


EXPECTED_KEYS = {
    "question",
    "sub_queries",
    "raw_findings",
    "nlp_results",
    "critique",
    "iteration",
    "report",
    "status",
}


class TestARIAState:
    def test_has_all_required_keys(self):
        assert set(ARIAState.__annotations__) == EXPECTED_KEYS

    def test_accepts_valid_dict(self):
        state: ARIAState = {
            "question": "What is quantum computing?",
            "sub_queries": ["sub1", "sub2"],
            "raw_findings": ["finding1"],
            "nlp_results": [NLPResult()],
            "critique": "PASS: looks good",
            "iteration": 1,
            "report": "# Report",
            "status": "done",
        }
        assert state["question"] == "What is quantum computing?"
        assert state["iteration"] == 1
        assert len(state["nlp_results"]) == 1

    def test_field_types(self):
        from typing import get_type_hints

        hints = get_type_hints(ARIAState)
        assert hints["question"] is str
        assert hints["critique"] is str
        assert hints["iteration"] is int
        assert hints["report"] is str
        assert hints["status"] is str
        assert hints["sub_queries"] == list[str]
        assert hints["raw_findings"] == list[str]
        assert hints["nlp_results"] == list[NLPResult]
