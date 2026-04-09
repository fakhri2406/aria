"""NLP pipeline — entity extraction, summarization, and semantic deduplication."""

from aria.nlp.deduplicator import SemanticDeduplicator
from aria.nlp.extractor import EntityExtractor, ExtractedEntities
from aria.nlp.pipeline import NLPPipeline, NLPResult
from aria.nlp.summarizer import TextSummarizer

__all__ = [
    "EntityExtractor",
    "ExtractedEntities",
    "NLPPipeline",
    "NLPResult",
    "SemanticDeduplicator",
    "TextSummarizer",
]
