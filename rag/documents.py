"""Modeles de donnees et ingestion des documents.

On lit un dossier de fichiers markdown/texte, puis on decoupe chaque document
en chunks qui se recouvrent legerement pour ne pas couper une idee en deux.
Chaque chunk conserve un lien vers sa source (titre + chemin) afin de pouvoir
citer l'origine dans la reponse finale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# Extensions de fichiers considerees comme du texte exploitable.
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


@dataclass(frozen=True)
class Chunk:
    """Un passage indexable issu d'un document."""

    chunk_id: int
    doc_id: int
    doc_title: str
    doc_path: str
    text: str

    def citation(self) -> str:
        """Libelle court utilise pour citer la source dans une reponse."""
        return f"[{self.doc_id}] {self.doc_title}"


@dataclass
class Document:
    """Un document source, identifie par son titre et son chemin."""

    doc_id: int
    title: str
    path: str
    content: str
    chunks: list[Chunk] = field(default_factory=list)


def _derive_title(path: Path, content: str) -> str:
    """Titre = premier titre markdown rencontre, sinon nom de fichier."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return path.stem


def _normalize_whitespace(text: str) -> str:
    """Compacte les espaces multiples tout en gardant les sauts de paragraphe."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Reduit les espaces/tabulations repetes sans supprimer les retours ligne.
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Decoupe un texte en fenetres de `chunk_size` caracteres qui se
    recouvrent de `overlap` caracteres.

    On coupe de preference sur une frontiere de mot pour ne pas tronquer un
    terme au milieu, ce qui ameliore la qualite des embeddings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size doit etre strictement positif")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap doit etre dans [0, chunk_size[")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        # Recherche d'une coupure propre (espace) proche de la fin de fenetre.
        if end < length:
            boundary = text.rfind(" ", start + overlap, end)
            if boundary != -1 and boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        # Avance en conservant un recouvrement.
        start = max(end - overlap, start + 1)
    return chunks


def load_documents(folder: str | Path) -> list[Document]:
    """Charge et trie les documents d'un dossier (ordre deterministe)."""
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {folder}")

    paths = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    documents: list[Document] = []
    for doc_id, path in enumerate(paths, start=1):
        raw = path.read_text(encoding="utf-8")
        content = _normalize_whitespace(raw)
        title = _derive_title(path, raw)
        documents.append(
            Document(doc_id=doc_id, title=title, path=str(path), content=content)
        )
    return documents


def build_chunks(documents: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    """Transforme la liste de documents en une liste plate de chunks indexables."""
    chunks: list[Chunk] = []
    next_id = 0
    for doc in documents:
        pieces = chunk_text(doc.content, chunk_size, overlap)
        for piece in pieces:
            chunk = Chunk(
                chunk_id=next_id,
                doc_id=doc.doc_id,
                doc_title=doc.title,
                doc_path=doc.path,
                text=piece,
            )
            doc.chunks.append(chunk)
            chunks.append(chunk)
            next_id += 1
    return chunks
