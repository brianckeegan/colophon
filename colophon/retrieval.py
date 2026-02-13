"""Deterministic lexical retrieval used for evidence ranking in local runs."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import Source


@dataclass(slots=True)
class RetrievalHit:
    """Retrieved bibliography source and its relevance score.

    Parameters
    ----------
    source : Source
        Retrieved bibliography source.
    score : float
        Similarity score for the query/source pair.
    """

    source: Source
    score: float


class SimpleRetriever:
    """Tiny lexical retriever suitable for deterministic local tests and demos."""

    def __init__(self, sources: list[Source]) -> None:
        """Init.

        Parameters
        ----------
        sources : list[Source]
            Parameter description.
        """
        self._sources = sources
        self._doc_vectors = [self._term_counts(source.text) for source in sources]

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Search.

        Parameters
        ----------
        query : str
            Parameter description.
        top_k : int
            Parameter description.

        Returns
        -------
        list[RetrievalHit]
            Return value description.
        """
        query_vector = self._term_counts(query)
        scored: list[RetrievalHit] = []

        for source, vector in zip(self._sources, self._doc_vectors, strict=True):
            score = self._cosine_similarity(query_vector, vector)
            if score > 0:
                scored.append(RetrievalHit(source=source, score=score))

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _term_counts(text: str) -> dict[str, float]:
        """Term counts.

        Parameters
        ----------
        text : str
            Parameter description.

        Returns
        -------
        dict[str, float]
            Return value description.
        """
        counts: dict[str, float] = {}
        for token in _tokenize(text):
            counts[token] = counts.get(token, 0.0) + 1.0
        return counts

    @staticmethod
    def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
        """Cosine similarity.

        Parameters
        ----------
        left : dict[str, float]
            Parameter description.
        right : dict[str, float]
            Parameter description.

        Returns
        -------
        float
            Return value description.
        """
        if not left or not right:
            return 0.0

        dot = sum(value * right.get(key, 0.0) for key, value in left.items())
        left_norm = math.sqrt(sum(v * v for v in left.values()))
        right_norm = math.sqrt(sum(v * v for v in right.values()))

        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)


def _tokenize(value: str) -> list[str]:
    """Tokenize.

    Parameters
    ----------
    value : str
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    tokens: list[str] = []
    for chunk in value.split():
        token = chunk.strip(".,;:!?()[]{}\"'\n\t ").lower()
        if token:
            tokens.append(token)
    return tokens
