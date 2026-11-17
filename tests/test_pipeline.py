"""Tests d'integration du pipeline RAG complet (hors ligne)."""

from __future__ import annotations

from pathlib import Path

from rag.config import Settings
from rag.pipeline import Pipeline


def test_ingestion_cree_l_index(settings: Settings) -> None:
    pipe = Pipeline(settings)
    report = pipe.ingest(Path(__file__).resolve().parent.parent / "corpus")

    assert report.documents == 4
    assert report.chunks >= 4
    index_dir = settings.index_dir
    # Les trois artefacts de persistance doivent exister.
    assert (index_dir / "chunks.json").exists()
    assert (index_dir / "vectors.npy").exists()
    assert (index_dir / "embedder.pkl").exists()


def test_question_tarif_renvoie_le_bon_passage(pipeline: Pipeline) -> None:
    answer = pipeline.ask("Combien coute la formule Essentiel ?")
    # Le passage pertinent contient le prix.
    assert "9 euros" in answer.text
    # La source citee est bien le document des tarifs.
    assert any("Tarifs" in source for source in answer.sources)
    assert not answer.is_unknown


def test_question_securite_cite_la_bonne_source(pipeline: Pipeline) -> None:
    answer = pipeline.ask("Quel algorithme de chiffrement est utilise ?")
    assert "AES-256" in answer.text
    assert any("Securite" in source for source in answer.sources)


def test_le_meilleur_passage_est_dans_le_top_k(pipeline: Pipeline) -> None:
    answer = pipeline.ask("Ou se trouve le siege social ?")
    # Au moins un passage recupere mentionne Lyon.
    assert any("Lyon" in hit.chunk.text for hit in answer.hits)
    # Le nombre de passages respecte top_k.
    assert len(answer.hits) <= pipeline.settings.top_k


def test_garde_fou_je_ne_sais_pas(pipeline: Pipeline) -> None:
    # Question totalement hors corpus : aucune information pertinente.
    answer = pipeline.ask("Quelle est la meilleure recette de curry thailandais ?")
    assert answer.is_unknown
    assert "Je ne sais pas" in answer.text
    assert answer.sources == []


def test_reponse_cite_toujours_une_source_quand_pertinent(pipeline: Pipeline) -> None:
    answer = pipeline.ask("L'API impose-t-elle une limite de debit ?")
    assert not answer.is_unknown
    # Toute reponse pertinente cite au moins une source.
    assert len(answer.sources) >= 1
    assert "600 requetes" in answer.text


def test_persistance_rechargeable(settings: Settings) -> None:
    # Un premier pipeline construit l'index.
    Pipeline(settings).ingest(Path(__file__).resolve().parent.parent / "corpus")
    # Un second pipeline (nouvel objet) doit pouvoir repondre sans reingestion.
    autre = Pipeline(settings)
    answer = autre.ask("Combien coute la formule Business ?")
    assert "29 euros" in answer.text
