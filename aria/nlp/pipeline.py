"""Composable NLP pipeline combining extraction, summarization, and deduplication."""

from __future__ import annotations

from pydantic import BaseModel

from aria.nlp.deduplicator import SemanticDeduplicator
from aria.nlp.extractor import EntityExtractor, ExtractedEntities
from aria.nlp.summarizer import TextSummarizer


class NLPResult(BaseModel):
    """Combined output of the NLP pipeline."""

    summary: str = ""
    entities: ExtractedEntities = ExtractedEntities()
    unique_facts: list[str] = []


class NLPPipeline:
    """Runs entity extraction, summarization, and semantic deduplication on text."""

    def __init__(self) -> None:
        self._extractor = EntityExtractor()
        self._summarizer = TextSummarizer()
        self._deduplicator = SemanticDeduplicator()

    def process(self, text: str) -> NLPResult:
        """Process *text* through the full NLP pipeline."""
        if not text.strip():
            return NLPResult()

        entities = self._extractor.extract(text)
        summary = self._summarizer.summarize(text)
        unique_facts = self._deduplicator.deduplicate(entities.noun_phrases + entities.key_terms)

        return NLPResult(
            summary=summary,
            entities=entities,
            unique_facts=unique_facts,
        )
