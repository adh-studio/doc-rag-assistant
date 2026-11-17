"""Orchestration du pipeline RAG de bout en bout.

Expose deux operations :
- `ingest(dossier)` : lit les documents, les decoupe, calcule les embeddings et
  persiste l'index vectoriel.
- `ask(question)` : recharge l'index, recupere les passages pertinents et
  genere une reponse citant ses sources.

Les composants (embedder, store, retriever, generator) sont injectes via les
`Settings`, ce qui rend chaque etage testable et remplacable isolement.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .documents import build_chunks, load_documents
from .embeddings import Embedder, build_embedder
from .generator import Answer, build_generator
from .retriever import Retriever
from .vectorstore import VectorStore


# Fichier de persistance du fournisseur d'embeddings (vocabulaire TF-IDF, etc.).
_EMBEDDER_FILE = "embedder.pkl"


@dataclass(frozen=True)
class IngestReport:
    """Bilan d'une ingestion."""

    documents: int
    chunks: int
    index_dir: str


class Pipeline:
    """Point d'entree unique du pipeline RAG."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings.from_env()

    @property
    def settings(self) -> Settings:
        return self._settings

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest(self, folder: str | Path) -> IngestReport:
        """Construit et persiste l'index vectoriel a partir d'un dossier."""
        documents = load_documents(folder)
        if not documents:
            raise ValueError(f"Aucun document exploitable dans {folder}")
        chunks = build_chunks(
            documents,
            chunk_size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("Les documents n'ont produit aucun chunk")

        embedder = build_embedder(self._settings)
        corpus = [chunk.text for chunk in chunks]
        embedder.fit(corpus)
        vectors = embedder.embed(corpus)

        store = VectorStore(chunks=chunks, vectors=vectors)
        store.save(self._settings.index_dir)
        self._save_embedder(embedder)

        return IngestReport(
            documents=len(documents),
            chunks=len(chunks),
            index_dir=str(self._settings.index_dir),
        )

    # ------------------------------------------------------------------ #
    # Interrogation
    # ------------------------------------------------------------------ #
    def ask(self, question: str) -> Answer:
        """Repond a une question en s'appuyant sur l'index persiste."""
        store = VectorStore.load(self._settings.index_dir)
        embedder = self._load_embedder()
        retriever = Retriever(
            embedder=embedder,
            store=store,
            top_k=self._settings.top_k,
            min_score=self._settings.min_score,
        )
        hits = retriever.retrieve(question)
        generator = build_generator(self._settings.allow_claude_cli)
        return generator.generate(question, hits)

    # ------------------------------------------------------------------ #
    # Persistance du fournisseur d'embeddings
    # ------------------------------------------------------------------ #
    def _save_embedder(self, embedder: Embedder) -> None:
        path = Path(self._settings.index_dir) / _EMBEDDER_FILE
        with path.open("wb") as handle:
            pickle.dump(embedder, handle)

    def _load_embedder(self) -> Embedder:
        path = Path(self._settings.index_dir) / _EMBEDDER_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Fournisseur d'embeddings introuvable dans {self._settings.index_dir}. "
                "Relancez l'ingestion."
            )
        # L'index n'est jamais charge depuis une source non fiable : l'usage de
        # pickle reste limite a nos propres artefacts generes localement.
        with path.open("rb") as handle:
            return pickle.load(handle)
