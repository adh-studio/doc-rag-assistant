"""Tests unitaires du decoupage et du chargement des documents."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.documents import build_chunks, chunk_text, load_documents


def test_chunk_text_court_renvoie_un_seul_chunk() -> None:
    texte = "Un texte tres court."
    assert chunk_text(texte, chunk_size=500, overlap=50) == [texte]


def test_chunk_text_decoupe_et_recouvre() -> None:
    texte = "mot " * 200  # 800 caracteres environ
    chunks = chunk_text(texte, chunk_size=200, overlap=40)
    # Plusieurs chunks attendus.
    assert len(chunks) > 1
    # Aucun chunk ne depasse la taille demandee (coupe sur frontiere de mot).
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_text_parametres_invalides() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_load_documents_trie_et_titre(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("# Second\nContenu B", encoding="utf-8")
    (tmp_path / "a.md").write_text("# Premier\nContenu A", encoding="utf-8")
    (tmp_path / "ignore.png").write_bytes(b"\x00")

    docs = load_documents(tmp_path)
    # Ordre alphabetique deterministe : a.md puis b.md.
    assert [d.title for d in docs] == ["Premier", "Second"]
    # Les fichiers non textuels sont ignores.
    assert len(docs) == 2


def test_build_chunks_attribue_des_ids_uniques(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("# Titre\n" + ("phrase. " * 300), encoding="utf-8")
    docs = load_documents(tmp_path)
    chunks = build_chunks(docs, chunk_size=200, overlap=40)
    ids = [c.chunk_id for c in chunks]
    assert ids == list(range(len(chunks)))
    assert all(c.doc_id == 1 for c in chunks)
