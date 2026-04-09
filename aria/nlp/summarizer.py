"""Extractive summarization using HuggingFace BART."""

from __future__ import annotations

import logging
from typing import Any, cast

from transformers import PreTrainedTokenizerBase, pipeline as hf_pipeline
from transformers.pipelines.base import Pipeline

logger = logging.getLogger(__name__)


class TextSummarizer:
    """Summarizes text using facebook/bart-large-cnn, with automatic chunking."""

    MODEL = "facebook/bart-large-cnn"
    MAX_TOKENS = 1024
    CHUNK_TOKENS = 900

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None

    def _load(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = hf_pipeline(
                "summarization",
                model=self.MODEL,
                device="cpu",
            )
        return self._pipeline

    def summarize(
        self,
        text: str,
        max_length: int = 180,
        min_length: int = 60,
    ) -> str:
        """Summarize *text*. Long texts are chunked automatically."""
        if not text.strip():
            return ""

        pipe = self._load()
        assert pipe.tokenizer is not None
        tokenizer: PreTrainedTokenizerBase = pipe.tokenizer
        token_ids = list(tokenizer.encode(text, add_special_tokens=False))
        n_tokens = len(token_ids)

        if n_tokens <= self.MAX_TOKENS:
            result = cast(
                list[dict[str, Any]],
                pipe(
                    text,
                    max_length=max_length,
                    min_length=min(min_length, max_length),
                    do_sample=False,
                ),
            )
            return result[0]["summary_text"]

        chunks = self._chunk_tokens(token_ids, tokenizer)
        summaries: list[str] = []
        for chunk_text in chunks:
            chunk_ids = list(tokenizer.encode(chunk_text, add_special_tokens=False))
            ratio = len(chunk_ids) / n_tokens
            chunk_max = max(30, int(max_length * ratio))
            chunk_min = max(10, min(int(min_length * ratio), chunk_max))
            chunk_result = cast(
                list[dict[str, Any]],
                pipe(chunk_text, max_length=chunk_max, min_length=chunk_min, do_sample=False),
            )
            summaries.append(chunk_result[0]["summary_text"])

        return " ".join(summaries)

    def _chunk_tokens(
        self,
        token_ids: list[int],
        tokenizer: PreTrainedTokenizerBase,
    ) -> list[str]:
        """Split token ids into ~CHUNK_TOKENS-sized pieces, decoded back to text."""
        chunks: list[str] = []
        for start in range(0, len(token_ids), self.CHUNK_TOKENS):
            chunk_ids = token_ids[start : start + self.CHUNK_TOKENS]
            chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        return chunks
