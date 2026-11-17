"""Permet d'invoquer le paquet avec `python -m rag`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
