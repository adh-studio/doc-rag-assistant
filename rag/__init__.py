"""Assistant RAG : repond a des questions sur un corpus en citant ses sources.

Le paquet expose les briques du pipeline (ingestion, embeddings, index
vectoriel, recherche, generation) derriere des interfaces claires afin de
rester testable hors ligne et interchangeable en production.
"""

from .pipeline import Pipeline
from .config import Settings

__all__ = ["Pipeline", "Settings"]
__version__ = "1.0.0"
