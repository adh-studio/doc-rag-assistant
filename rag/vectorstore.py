"""Base vectorielle locale minimaliste (numpy + similarite cosinus).

Implementation volontairement simple et sans dependance serveur : les vecteurs
sont stockes dans un tableau numpy et la recherche top-k se fait par produit
scalaire (les vecteurs etant normalises L2, cosinus = produit scalaire).

En production, cette classe est interchangeable avec FAISS, pgvector ou Chroma
sans toucher au reste du pipeline : seule l'interface `search` est consommee
par le `Retriever`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .documents import Chunk


# Noms des fichiers de persistance dans le dossier d'index.
_META_FILE = "chunks.json"
_VECTORS_FILE = "vectors.npy"


@dataclass(frozen=True)
class SearchHit:
    """Resultat de recherche : un chunk et son score de similarite."""

    chunk: Chunk
    score: float


class VectorStore:
    """Stocke les vecteurs de chunks et effectue la recherche par similarite."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("Le nombre de chunks et de vecteurs doit coincider")
        self._chunks = chunks
        self._vectors = vectors.astype(np.float32)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        """Retourne les `top_k` chunks les plus proches, tries par score."""
        if len(self._chunks) == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        # Produit scalaire vecteurs x requete == cosinus (vecteurs normalises).
        scores = self._vectors @ query
        k = min(top_k, len(scores))
        # argpartition pour l'efficacite, puis tri des k candidats.
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [SearchHit(chunk=self._chunks[i], score=float(scores[i])) for i in top_idx]

    def save(self, index_dir: str | Path) -> None:
        """Persiste chunks (JSON lisible) et vecteurs (binaire numpy)."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        meta = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "doc_path": c.doc_path,
                "text": c.text,
            }
            for c in self._chunks
        ]
        (index_dir / _META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.save(index_dir / _VECTORS_FILE, self._vectors)

    @classmethod
    def load(cls, index_dir: str | Path) -> "VectorStore":
        """Recharge une base vectorielle persistee."""
        index_dir = Path(index_dir)
        meta_path = index_dir / _META_FILE
        vectors_path = index_dir / _VECTORS_FILE
        if not meta_path.exists() or not vectors_path.exists():
            raise FileNotFoundError(
                f"Index introuvable dans {index_dir}. Lancez d'abord l'ingestion."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks = [
            Chunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                doc_title=item["doc_title"],
                doc_path=item["doc_path"],
                text=item["text"],
            )
            for item in meta
        ]
        vectors = np.load(vectors_path)
        return cls(chunks=chunks, vectors=vectors)
