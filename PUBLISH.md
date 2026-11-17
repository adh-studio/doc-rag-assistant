# Publication du depot

Le depot local est deja initialise et commite. Pour le publier sur GitHub
(compte `adh-studio`) :

```bash
cd /Users/wize/fullstack-entretien/projets/doc-rag-assistant

# Cree le depot public et pousse la branche main
gh repo create doc-rag-assistant \
  --public \
  --description "Assistant RAG hors ligne qui repond sur un corpus en citant ses sources (Python, embeddings, base vectorielle)." \
  --source . \
  --push
```

## Activer GitHub Pages (demo web)

La demo statique se trouve dans `docs/index.html`. Pour la publier :

```bash
# Active Pages sur la branche main, dossier /docs
gh api -X POST repos/adh-studio/doc-rag-assistant/pages \
  -f "source[branch]=main" \
  -f "source[path]=/docs"
```

La page sera disponible sous quelques minutes a l'adresse :
`https://adh-studio.github.io/doc-rag-assistant/`

## Verifier la CI

Apres le push, l'onglet **Actions** du depot execute automatiquement le
workflow `CI` (installation Python 3.13 + `pytest`). Il doit passer au vert.

```bash
gh run list --repo adh-studio/doc-rag-assistant
```
