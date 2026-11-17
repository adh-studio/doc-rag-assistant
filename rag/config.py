"""Configuration centralisee du pipeline RAG.

Toute la configuration passe par un objet immuable `Settings`. Les valeurs
sont lues depuis l'environnement avec des valeurs par defaut sures pour un
fonctionnement 100% hors ligne (mode TF-IDF + generation extractive).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_flag(name: str, default: bool) -> bool:
    """Lit une variable d'environnement booleenne de maniere tolerante."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "oui", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Parametres du pipeline, injectes dans chaque composant (inversion de
    dependance : aucun composant ne lit l'environnement directement)."""

    # Emplacement de l'index vectoriel persiste.
    index_dir: Path = Path(".rag_index")

    # Decoupage des documents.
    chunk_size: int = 500  # nombre de caracteres vises par chunk
    chunk_overlap: int = 80  # recouvrement entre deux chunks consecutifs

    # Recherche.
    top_k: int = 3
    # Score cosinus minimal pour considerer un passage comme pertinent.
    # En dessous, le garde-fou "je ne sais pas" se declenche.
    min_score: float = 0.05

    # Choix du fournisseur d'embeddings.
    # Si une cle API est presente et que le mode distant est autorise, on
    # utilise le fournisseur distant ; sinon repli TF-IDF local deterministe.
    embeddings_api_key: str | None = None
    allow_remote_embeddings: bool = True

    # Choix du generateur.
    # La generation via la CLI `claude` n'est tentee que si ce drapeau est
    # vrai ; les tests la desactivent pour rester hors ligne et deterministes.
    allow_claude_cli: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        """Construit les parametres depuis l'environnement du processus."""
        index_dir = Path(os.environ.get("RAG_INDEX_DIR", ".rag_index"))
        return cls(
            index_dir=index_dir,
            chunk_size=int(os.environ.get("RAG_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.environ.get("RAG_CHUNK_OVERLAP", "80")),
            top_k=int(os.environ.get("RAG_TOP_K", "3")),
            min_score=float(os.environ.get("RAG_MIN_SCORE", "0.05")),
            embeddings_api_key=os.environ.get("RAG_EMBEDDINGS_API_KEY"),
            allow_remote_embeddings=_env_flag("RAG_ALLOW_REMOTE_EMBEDDINGS", True),
            allow_claude_cli=_env_flag("RAG_ALLOW_CLAUDE_CLI", True),
        )

    @property
    def use_remote_embeddings(self) -> bool:
        """Vrai uniquement si une cle est fournie et le mode distant permis."""
        return bool(self.embeddings_api_key) and self.allow_remote_embeddings
