"""Fixtures partagees : pipeline hors ligne sur le corpus de demo."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import Settings
from rag.pipeline import Pipeline

# Racine du projet (le dossier parent de tests/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Parametres de test : index temporaire, 100% hors ligne.

    - TF-IDF local force (aucune cle, mode distant interdit).
    - CLI claude desactivee : generation extractive deterministe.
    """
    return Settings(
        index_dir=tmp_path / "index",
        allow_remote_embeddings=False,
        embeddings_api_key=None,
        allow_claude_cli=False,
    )


@pytest.fixture
def pipeline(settings: Settings) -> Pipeline:
    """Pipeline dont l'index est deja construit sur le corpus de demo."""
    pipe = Pipeline(settings)
    pipe.ingest(CORPUS_DIR)
    return pipe
