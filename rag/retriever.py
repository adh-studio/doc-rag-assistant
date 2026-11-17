"""Recherche semantique : embarque la question puis interroge la base.

Le `Retriever` orchestre embedder et base vectorielle. Il applique aussi le
garde-fou de pertinence : si le meilleur score reste sous le seuil, on renvoie
une liste vide, ce qui declenchera une reponse "je ne sais pas" en aval.
"""

from __future__ import annotations

from .embeddings import Embedder
from .vectorstore import SearchHit, VectorStore


class Retriever:
    """Associe un fournisseur d'embeddings a une base vectorielle."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int,
        min_score: float,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._min_score = min_score

    def retrieve(self, question: str) -> list[SearchHit]:
        """Retourne les passages pertinents pour la question (filtres par score)."""
        question = question.strip()
        if not question:
            return []
        query_vector = self._embedder.embed([question])[0]
        hits = self._store.search(query_vector, self._top_k)
        # Garde-fou : on ne conserve que les passages au dessus du seuil.
        return [hit for hit in hits if hit.score >= self._min_score]
