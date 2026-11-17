# doc-rag-assistant

Assistant RAG (Retrieval-Augmented Generation) qui repond a des questions
posees en langage naturel sur un corpus de documents, en **citant
systematiquement ses sources**. Il fonctionne entierement hors ligne grace a un
repli deterministe (TF-IDF + generation extractive), et sait brancher un
fournisseur d'embeddings distant ou la CLI `claude` quand ils sont disponibles.

## Contexte

Les equipes passent un temps considerable a chercher une information dans une
documentation eparpillee. Ce projet montre comment construire, de bout en bout,
un assistant qui recupere les passages pertinents d'un corpus et compose une
reponse sourcee, avec un garde-fou qui dit honnetement "je ne sais pas" quand le
corpus ne contient pas la reponse (lutte contre l'hallucination).

Le corpus de demonstration decrit un produit fictif (Nimbus Backup, solution de
sauvegarde cloud) : presentation, tarifs, securite et API.

## Stack

- **Python 3.13**
- **scikit-learn** pour les embeddings TF-IDF (repli local deterministe)
- **numpy** pour l'index vectoriel et la similarite cosinus
- **pytest** pour les tests
- **GitHub Actions** pour l'integration continue

## Architecture

Chaque etage du pipeline est un composant isole, derriere une interface claire,
injecte via un objet de configuration. On peut remplacer une brique (embeddings,
base vectorielle, generateur) sans toucher aux autres.

```
corpus/ ──► documents.py ──► embeddings.py ──► vectorstore.py ──► retriever.py ──► generator.py ──► reponse citee
           (ingestion,        (TF-IDF local     (numpy + cosinus,   (top-k +          (CLI claude ou
            decoupage)         ou API distante)   persiste)           garde-fou)        repli extractif)
```

| Fichier | Responsabilite | Principe |
|---|---|---|
| `rag/config.py` | Parametres immuables, lus depuis l'environnement | Inversion de dependance |
| `rag/documents.py` | Chargement et decoupage en chunks avec recouvrement | Responsabilite unique |
| `rag/embeddings.py` | Fournisseurs TF-IDF local / distant derriere une interface | Ouvert/ferme, Liskov |
| `rag/vectorstore.py` | Index numpy persiste, recherche par similarite | Interchangeable (FAISS, pgvector...) |
| `rag/retriever.py` | Embarque la question, interroge l'index, filtre par score | Responsabilite unique |
| `rag/generator.py` | Generation en chaine (claude CLI puis repli extractif) | Chain of responsibility |
| `rag/pipeline.py` | Orchestration ingestion / interrogation | Facade |
| `rag/cli.py` | Interface en ligne de commande | Separation UI / domaine |

### Deux modes d'embeddings

1. **Local TF-IDF (defaut)** : deterministe, sans reseau, teste hors ligne.
   Les mots vides francais sont retires pour que le garde-fou de pertinence
   fonctionne correctement.
2. **Distant** : fournisseur compatible OpenAI, active uniquement si
   `RAG_EMBEDDINGS_API_KEY` est defini. En production on y brancherait OpenAI,
   Voyage, Cohere, etc.

### Base vectorielle

Implementation locale simple (numpy + cosinus, persistee en fichiers
`chunks.json` + `vectors.npy`). Elle est **interchangeable** avec FAISS,
pgvector ou Chroma en production : seule l'interface `search` est consommee.

### Generation

Moteur en chaine : (1) CLI `claude -p` si disponible et autorisee (abonnement,
sans cle API) ; (2) repli **extractif deterministe** qui compose la reponse a
partir des passages trouves. Dans les deux cas, la reponse cite ses documents
sources (numero + titre).

## Installation

```bash
# Python 3.13 requis
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
# 1. Indexer un dossier de documents (.md, .txt)
python -m rag ingest corpus

# 2. Poser une question
python -m rag ask "Combien coute la formule Business ?"
python -m rag ask "Quel algorithme de chiffrement est utilise ?"
```

Variables d'environnement utiles : `RAG_INDEX_DIR`, `RAG_TOP_K`,
`RAG_MIN_SCORE`, `RAG_EMBEDDINGS_API_KEY`, `RAG_ALLOW_CLAUDE_CLI`.

## Tests

Commande unique, hors ligne, deterministe :

```bash
python -m pytest
```

Les tests couvrent : l'ingestion cree bien l'index, une question factuelle
renvoie le bon passage dans le top-k, la reponse cite la bonne source, le
garde-fou "je ne sais pas" se declenche hors sujet, et la persistance est
rechargeable.

## Demo web

Une page statique (`docs/index.html`) illustre le pipeline sous forme de schema.
Elle peut etre publiee via GitHub Pages (voir `PUBLISH.md`).

## Ce que ce projet demontre pour un poste Full Stack

- **IA / RAG** : pipeline complet LLM + embeddings + base vectorielle + recherche
  par similarite + generation sourcee ; prompt engineering ; garde-fou
  anti-hallucination ; developpement assiste par IA (CLI claude branchable).
- **Backend Python** : conception orientee objet, API interne claire, CLI.
- **Genie logiciel** : principes SOLID, clean architecture, patterns (facade,
  chain of responsibility, inversion de dependance), tests unitaires et
  d'integration reels.
- **Bases de donnees** : index vectoriel persiste, documente comme
  interchangeable avec pgvector / FAISS / Chroma.
- **DevOps / CI** : integration continue GitHub Actions, configuration par
  environnement, execution reproductible.
- **Securite et robustesse** : fonctionnement hors ligne par defaut, cles
  jamais stockees en clair, repli sur en cas de defaillance d'un service externe.
- **Frontend** : page de presentation responsive, accessible, theme clair/sombre.
