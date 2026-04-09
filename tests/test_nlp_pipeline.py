"""Tests for aria/nlp/ — EntityExtractor, TextSummarizer, SemanticDeduplicator, NLPPipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aria.nlp.deduplicator import SemanticDeduplicator
from aria.nlp.extractor import EntityExtractor, ExtractedEntities
from aria.nlp.pipeline import NLPPipeline, NLPResult
from aria.nlp.summarizer import TextSummarizer


class TestEntityExtractor:
    def test_extract_empty_text(self):
        ext = EntityExtractor()
        assert ext.extract("") == ExtractedEntities()
        assert ext.extract("   ") == ExtractedEntities()

    def test_extract_named_entities(self):
        ext = EntityExtractor()

        ent = MagicMock()
        ent.text = "Google"
        ent.label_ = "ORG"

        doc = MagicMock()
        doc.ents = [ent]
        doc.noun_chunks = []
        doc.__iter__ = lambda self: iter([])

        mock_nlp = MagicMock(return_value=doc)
        with patch.object(ext, "_load", return_value=mock_nlp):
            result = ext.extract("Google is a company.")

        assert result.named_entities == [("Google", "ORG")]

    def test_extract_noun_phrases(self):
        ext = EntityExtractor()

        chunk = MagicMock()
        chunk.text = "quantum computing"

        doc = MagicMock()
        doc.ents = []
        doc.noun_chunks = [chunk]
        doc.__iter__ = lambda self: iter([])

        mock_nlp = MagicMock(return_value=doc)
        with patch.object(ext, "_load", return_value=mock_nlp):
            result = ext.extract("Advances in quantum computing.")

        assert result.noun_phrases == ["quantum computing"]

    def test_extract_key_terms(self):
        ext = EntityExtractor()

        def make_token(lower: str, pos: str = "NOUN", is_stop: bool = False):
            t = MagicMock()
            t.pos_ = pos
            t.is_stop = is_stop
            t.is_punct = False
            t.lower_ = lower
            return t

        tokens = [
            make_token("model"),
            make_token("model"),
            make_token("data"),
            make_token("the", pos="DET", is_stop=True),
        ]

        doc = MagicMock()
        doc.ents = []
        doc.noun_chunks = []
        doc.__iter__ = lambda self: iter(tokens)

        mock_nlp = MagicMock(return_value=doc)
        with patch.object(ext, "_load", return_value=mock_nlp):
            result = ext.extract("Some text about models and data.")

        assert result.key_terms[0] == "model"
        assert "data" in result.key_terms


class TestTextSummarizer:
    def test_summarize_empty_text(self):
        s = TextSummarizer()
        assert s.summarize("") == ""
        assert s.summarize("   ") == ""

    def test_summarize_short_text(self):
        s = TextSummarizer()

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(100))

        mock_pipe = MagicMock()
        mock_pipe.tokenizer = mock_tokenizer
        mock_pipe.return_value = [{"summary_text": "Short summary."}]

        with patch.object(s, "_load", return_value=mock_pipe):
            result = s.summarize("Some input text.")

        assert result == "Short summary."

    def test_summarize_long_text_chunks(self):
        s = TextSummarizer()

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = [
            list(range(1500)),
            list(range(900)),
            list(range(600)),
        ]
        mock_tokenizer.decode.side_effect = ["chunk one text", "chunk two text"]

        mock_pipe = MagicMock()
        mock_pipe.tokenizer = mock_tokenizer
        mock_pipe.side_effect = [
            [{"summary_text": "Summary A."}],
            [{"summary_text": "Summary B."}],
        ]

        with patch.object(s, "_load", return_value=mock_pipe):
            result = s.summarize("A very long input text.")

        assert result == "Summary A. Summary B."


class TestSemanticDeduplicator:
    def test_deduplicate_empty_list(self):
        d = SemanticDeduplicator()
        assert d.deduplicate([]) == []

    def test_deduplicate_no_duplicates(self):
        d = SemanticDeduplicator()

        mock_model = MagicMock()
        mock_model.encode.return_value = "embeddings"

        sim_row0 = MagicMock()
        sim_row0.__getitem__ = lambda self, j: 0.0 if j != 0 else 1.0
        sim_row1 = MagicMock()
        sim_row1.__getitem__ = lambda self, j: 0.0 if j != 1 else 1.0

        mock_sim = MagicMock()
        mock_sim.__getitem__ = lambda self, i: [sim_row0, sim_row1][i]

        with patch.object(d, "_load", return_value=mock_model), \
             patch("aria.nlp.deduplicator.util.cos_sim", return_value=mock_sim):
            result = d.deduplicate(["apples", "quantum mechanics"])

        assert result == ["apples", "quantum mechanics"]

    def test_deduplicate_removes_duplicates(self):
        d = SemanticDeduplicator()

        mock_model = MagicMock()
        mock_model.encode.return_value = "embeddings"

        matrix = [
            [1.0, 0.1, 0.95],
            [0.1, 1.0, 0.2],
            [0.95, 0.2, 1.0],
        ]
        mock_sim = MagicMock()
        mock_sim.__getitem__ = lambda self, i: MagicMock(
            __getitem__=lambda s, j: matrix[i][j]
        )

        with patch.object(d, "_load", return_value=mock_model), \
             patch("aria.nlp.deduplicator.util.cos_sim", return_value=mock_sim):
            result = d.deduplicate(["machine learning", "quantum physics", "ML models"])

        assert result == ["machine learning", "quantum physics"]


class TestNLPPipeline:
    def test_process_empty_text(self):
        with patch.object(EntityExtractor, "_load"), \
             patch.object(TextSummarizer, "_load"), \
             patch.object(SemanticDeduplicator, "_load"):
            pipeline = NLPPipeline()
            result = pipeline.process("")

        assert result == NLPResult()
        assert result.summary == ""
        assert result.entities == ExtractedEntities()
        assert result.unique_facts == []

    def test_process_delegates_to_components(self):
        entities = ExtractedEntities(
            named_entities=[("Python", "LANGUAGE")],
            noun_phrases=["machine learning"],
            key_terms=["model"],
        )

        pipeline = NLPPipeline()

        with patch.object(pipeline._extractor, "extract", return_value=entities) as mock_ext, \
             patch.object(pipeline._summarizer, "summarize", return_value="A summary.") as mock_sum, \
             patch.object(pipeline._deduplicator, "deduplicate", return_value=["machine learning", "model"]) as mock_dedup:
            result = pipeline.process("Some research text about machine learning.")

        mock_ext.assert_called_once_with("Some research text about machine learning.")
        mock_sum.assert_called_once_with("Some research text about machine learning.")
        mock_dedup.assert_called_once_with(["machine learning", "model"])

        assert result.summary == "A summary."
        assert result.entities == entities
        assert result.unique_facts == ["machine learning", "model"]
