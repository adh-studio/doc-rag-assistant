"""Generation de la reponse a partir des passages recuperes.

Strategie en chaine (chain of responsibility) :
1. `ClaudeCliGenerator` : utilise la CLI `claude -p` (abonnement, sans cle API)
   si elle est disponible et autorisee.
2. `ExtractiveGenerator` : repli deterministe, hors ligne, qui compose la
   reponse directement a partir des passages trouves.

Dans tous les cas la reponse CITE ses sources (numero + titre du document) et
un garde-fou renvoie "je ne sais pas" quand aucun passage pertinent n'existe.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Protocol

from .vectorstore import SearchHit


# Message renvoye quand le corpus ne permet pas de repondre.
UNKNOWN_ANSWER = (
    "Je ne sais pas : le corpus ne contient pas d'information pertinente "
    "pour repondre a cette question."
)


@dataclass(frozen=True)
class Answer:
    """Reponse finale : texte + sources citees + passages utilises."""

    text: str
    sources: list[str]
    hits: list[SearchHit]

    @property
    def is_unknown(self) -> bool:
        return not self.sources


def _unique_sources(hits: list[SearchHit]) -> list[str]:
    """Liste ordonnee et dedupliquee des citations (par document)."""
    seen: set[int] = set()
    sources: list[str] = []
    for hit in hits:
        if hit.chunk.doc_id not in seen:
            seen.add(hit.chunk.doc_id)
            sources.append(hit.chunk.citation())
    return sources


class Generator(Protocol):
    """Contrat d'un moteur de generation."""

    def generate(self, question: str, hits: list[SearchHit]) -> Answer:
        ...


class ExtractiveGenerator:
    """Repli deterministe : compose la reponse a partir des passages.

    Aucune dependance reseau, sortie reproductible : parfait pour les tests et
    le fonctionnement hors ligne.
    """

    def generate(self, question: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return Answer(text=UNKNOWN_ANSWER, sources=[], hits=[])
        sources = _unique_sources(hits)
        lines = [
            "D'apres le corpus, voici les elements pertinents :",
            "",
        ]
        for hit in hits:
            lines.append(f"- {hit.chunk.text} ({hit.chunk.citation()})")
        lines.append("")
        lines.append("Sources : " + " ; ".join(sources))
        return Answer(text="\n".join(lines), sources=sources, hits=hits)


class ClaudeCliGenerator:
    """Generation via la CLI `claude -p` (abonnement, sans cle API).

    En cas d'indisponibilite ou d'erreur, delegue au generateur de repli fourni
    afin de garantir une reponse dans tous les cas.
    """

    def __init__(self, fallback: Generator, timeout: int = 60) -> None:
        self._fallback = fallback
        self._timeout = timeout

    @staticmethod
    def is_available() -> bool:
        """Vrai si l'executable `claude` est present dans le PATH."""
        return shutil.which("claude") is not None

    def _build_prompt(self, question: str, hits: list[SearchHit]) -> str:
        contexte = "\n\n".join(
            f"{hit.chunk.citation()}\n{hit.chunk.text}" for hit in hits
        )
        return (
            "Tu es un assistant documentaire. Reponds a la question en te basant "
            "UNIQUEMENT sur les passages ci-dessous. Cite tes sources sous la forme "
            "[numero] titre. Si les passages ne suffisent pas, dis que tu ne sais pas.\n\n"
            f"Passages :\n{contexte}\n\n"
            f"Question : {question}\n\nReponse :"
        )

    def generate(self, question: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return Answer(text=UNKNOWN_ANSWER, sources=[], hits=[])
        if not self.is_available():
            return self._fallback.generate(question, hits)
        prompt = self._build_prompt(question, hits)
        try:
            result = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=True,
            )
        except (subprocess.SubprocessError, OSError):
            # Toute defaillance (reseau, timeout, binaire absent) -> repli sur.
            return self._fallback.generate(question, hits)
        text = result.stdout.strip()
        if not text:
            return self._fallback.generate(question, hits)
        sources = _unique_sources(hits)
        return Answer(text=text, sources=sources, hits=hits)


def build_generator(allow_claude_cli: bool) -> Generator:
    """Construit la chaine de generation selon la configuration."""
    extractive = ExtractiveGenerator()
    if allow_claude_cli:
        return ClaudeCliGenerator(fallback=extractive)
    return extractive
