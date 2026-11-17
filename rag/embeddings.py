"""Fournisseurs d'embeddings.

Deux modes interchangeables derriere la meme interface `Embedder` :

1. `LocalTfidfEmbedder` : repli 100% local et deterministe (TF-IDF scikit-learn,
   vecteurs normalises L2). Aucun reseau, ideal pour les tests et le hors ligne.
2. `RemoteEmbedder` : fournisseur distant compatible OpenAI, active uniquement
   si une cle API est fournie. Non utilise par les tests.

Le choix est fait par `build_embedder` en fonction des `Settings`, ce qui
respecte le principe d'inversion de dependance : les composants en aval ne
connaissent que l'interface, jamais l'implementation concrete.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Settings


# Mots vides francais (sans accents, coherents avec strip_accents="unicode").
# Les retirer evite que des mots fonctionnels tres frequents gonflent la
# similarite : une question hors sujet obtient alors un score quasi nul, ce qui
# fait correctement declencher le garde-fou "je ne sais pas".
FRENCH_STOPWORDS = [
    "a", "au", "aux", "avec", "ce", "ces", "cette", "ceci", "cela", "comme",
    "comment", "dans", "de", "des", "du", "donc", "elle", "elles", "en", "est",
    "et", "etre", "il", "ils", "je", "la", "le", "les", "leur", "leurs", "lui",
    "ma", "mais", "me", "meilleur", "meilleure", "meme", "mes", "moi", "mon",
    "ne", "ni", "nos", "notre", "nous", "on", "ou", "par", "pas", "plus", "pour",
    "quand", "que", "quel", "quelle", "quelles", "quels", "qui", "sa", "sans",
    "se", "ses", "si", "son", "sont", "sur", "ta", "te", "tes", "toi", "ton",
    "tu", "un", "une", "vos", "votre", "vous", "y", "etait", "ete", "cet",
    "aussi", "tout", "tous", "toute", "toutes", "entre", "vers", "chez",
]


@runtime_checkable
class Embedder(Protocol):
    """Contrat commun a tous les fournisseurs d'embeddings."""

    # Dimension des vecteurs produits (connue apres `fit`).
    dim: int

    def fit(self, corpus: list[str]) -> None:
        """Prepare le fournisseur sur le corpus (vocabulaire TF-IDF, etc.)."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Retourne une matrice (len(texts), dim) de vecteurs normalises L2."""
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalise chaque ligne pour que cosinus = produit scalaire."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class LocalTfidfEmbedder:
    """Embeddings TF-IDF locaux et deterministes (scikit-learn).

    Le vocabulaire est appris sur le corpus lors de `fit`. A la requete, on
    projette la question dans le meme espace via `transform`.
    """

    def __init__(self) -> None:
        # Vectoriseur TF-IDF : minuscules, bi-grammes, normalisation L2.
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            norm="l2",
            min_df=1,
            strip_accents="unicode",
            stop_words=FRENCH_STOPWORDS,
        )
        self._fitted = False
        self.dim = 0

    def fit(self, corpus: list[str]) -> None:
        if not corpus:
            raise ValueError("Le corpus d'ajustement ne peut pas etre vide")
        self._vectorizer.fit(corpus)
        self._fitted = True
        self.dim = len(self._vectorizer.get_feature_names_out())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit() doit etre appele avant embed()")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # transform renvoie une matrice creuse deja normalisee L2.
        matrix = self._vectorizer.transform(texts).toarray().astype(np.float32)
        return matrix


class RemoteEmbedder:
    """Fournisseur d'embeddings distant compatible OpenAI.

    Active uniquement lorsqu'une cle API est presente. Conserve pour montrer
    l'interchangeabilite : en production on brancherait ici OpenAI, Voyage,
    Cohere, etc. Les tests hors ligne ne l'instancient jamais.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        endpoint: str = "https://api.openai.com/v1/embeddings",
        dim: int = 1536,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self.dim = dim

    def fit(self, corpus: list[str]) -> None:
        # Un modele distant est deja entraine : rien a apprendre localement.
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payload = json.dumps({"model": self._model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        vectors = [item["embedding"] for item in body["data"]]
        matrix = np.asarray(vectors, dtype=np.float32)
        self.dim = matrix.shape[1]
        return _l2_normalize(matrix)


def build_embedder(settings: Settings) -> Embedder:
    """Fabrique le fournisseur d'embeddings selon la configuration."""
    if settings.use_remote_embeddings and settings.embeddings_api_key:
        return RemoteEmbedder(api_key=settings.embeddings_api_key)
    return LocalTfidfEmbedder()
