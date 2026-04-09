"""Entity extraction using spaCy NLP pipeline."""

from __future__ import annotations

from collections import Counter

import spacy
from pydantic import BaseModel


class ExtractedEntities(BaseModel):
    """Entities, noun phrases, and key terms extracted from text."""

    named_entities: list[tuple[str, str]] = []
    noun_phrases: list[str] = []
    key_terms: list[str] = []


class EntityExtractor:
    """Extracts named entities, noun phrases, and key terms from text using spaCy."""

    def __init__(self) -> None:
        self._nlp: spacy.language.Language | None = None

    def _load(self) -> spacy.language.Language:
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def extract(self, text: str) -> ExtractedEntities:
        """Extract entities, noun phrases, and key terms from *text*."""
        if not text.strip():
            return ExtractedEntities()

        nlp = self._load()
        doc = nlp(text)

        named_entities = [(ent.text, ent.label_) for ent in doc.ents]
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]

        noun_counts: Counter[str] = Counter()
        for token in doc:
            if token.pos_ == "NOUN" and not token.is_stop and not token.is_punct:
                noun_counts[token.lower_] += 1
        key_terms = [term for term, _ in noun_counts.most_common(10)]

        return ExtractedEntities(
            named_entities=named_entities,
            noun_phrases=noun_phrases,
            key_terms=key_terms,
        )
