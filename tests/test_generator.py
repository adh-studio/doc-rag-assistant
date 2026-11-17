"""Tests unitaires de la couche de generation."""

from __future__ import annotations

from rag.documents import Chunk
from rag.generator import (
    ClaudeCliGenerator,
    ExtractiveGenerator,
    build_generator,
)
from rag.vectorstore import SearchHit


def _hit(doc_id: int, title: str, text: str, score: float) -> SearchHit:
    chunk = Chunk(
        chunk_id=0,
        doc_id=doc_id,
        doc_title=title,
        doc_path=f"/{title}.md",
        text=text,
    )
    return SearchHit(chunk=chunk, score=score)


def test_extractif_compose_et_cite() -> None:
    gen = ExtractiveGenerator()
    hits = [_hit(2, "Tarifs", "La formule Essentiel coute 9 euros.", 0.6)]
    answer = gen.generate("prix ?", hits)
    assert "9 euros" in answer.text
    assert answer.sources == ["[2] Tarifs"]
    assert not answer.is_unknown


def test_extractif_sans_passage_renvoie_inconnu() -> None:
    gen = ExtractiveGenerator()
    answer = gen.generate("question ?", [])
    assert answer.is_unknown
    assert "Je ne sais pas" in answer.text


def test_sources_dedupliquees_par_document() -> None:
    gen = ExtractiveGenerator()
    hits = [
        _hit(1, "Doc", "Passage A.", 0.5),
        _hit(1, "Doc", "Passage B.", 0.4),
    ]
    answer = gen.generate("q ?", hits)
    # Deux passages du meme document -> une seule citation.
    assert answer.sources == ["[1] Doc"]


def test_claude_cli_repli_quand_indisponible() -> None:
    # Quand la CLI est absente, le generateur delegue au repli extractif.
    extractive = ExtractiveGenerator()
    gen = ClaudeCliGenerator(fallback=extractive)
    hits = [_hit(3, "Securite", "Chiffrement AES-256.", 0.7)]
    answer = gen.generate("chiffrement ?", hits)
    # Le repli produit une reponse citant la source, quel que soit l'etat de la CLI.
    assert "AES-256" in answer.text
    assert answer.sources == ["[3] Securite"]


def test_build_generator_respecte_la_config() -> None:
    assert isinstance(build_generator(allow_claude_cli=False), ExtractiveGenerator)
    assert isinstance(build_generator(allow_claude_cli=True), ClaudeCliGenerator)
