"""Interface en ligne de commande du pipeline RAG.

Usage :
    python -m rag ingest <dossier>
    python -m rag ask "votre question"
"""

from __future__ import annotations

import argparse
import sys

from .config import Settings
from .pipeline import Pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Assistant RAG : repond a des questions sur un corpus en citant ses sources.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Indexe un dossier de documents")
    p_ingest.add_argument("folder", help="Dossier contenant les documents (.md, .txt)")

    p_ask = sub.add_parser("ask", help="Pose une question au corpus indexe")
    p_ask.add_argument("question", help="Question en langage naturel")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    pipeline = Pipeline(settings)

    if args.command == "ingest":
        report = pipeline.ingest(args.folder)
        print(
            f"Index construit : {report.documents} document(s), "
            f"{report.chunks} chunk(s) -> {report.index_dir}"
        )
        return 0

    if args.command == "ask":
        answer = pipeline.ask(args.question)
        print(answer.text)
        if answer.sources:
            print()
            print("Sources citees : " + " ; ".join(answer.sources))
        return 0

    parser.error("commande inconnue")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
