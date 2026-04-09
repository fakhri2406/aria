"""Semantic deduplication using sentence-transformers."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer, util


class SemanticDeduplicator:
    """Removes near-duplicate strings using cosine similarity of embeddings."""

    MODEL = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.MODEL)
        return self._model

    def deduplicate(self, items: list[str], threshold: float = 0.85) -> list[str]:
        """Return *items* with near-duplicates removed (keeps first occurrence)."""
        if not items:
            return []

        model = self._load()
        embeddings = model.encode(items, convert_to_tensor=True)
        sim_matrix = util.cos_sim(embeddings, embeddings)

        removed: set[int] = set()
        result: list[str] = []

        for i in range(len(items)):
            if i in removed:
                continue
            result.append(items[i])
            for j in range(i + 1, len(items)):
                if j not in removed and float(sim_matrix[i][j]) > threshold:
                    removed.add(j)

        return result
