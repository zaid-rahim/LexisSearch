# LexisSearch — Semantic Legal Search Engine

A neural semantic search engine built over a corpus of legal documents, powered by **sentence-transformers (MPNet)** and deployed live on Streamlit Cloud. Type any legal concept in plain English and the engine returns the most semantically relevant documents ranked by cosine similarity — not keyword matching.

**Live demo:** https://semanticenginesearch.streamlit.app

---

## What it does

Traditional keyword search fails when someone searches "right to a fair hearing" but the document says "due process entitlement." Semantic search understands that these mean the same thing because it works on **meaning**, not character patterns.

This engine:
- Embeds every document in the corpus as a 768-dimensional vector using `all-mpnet-base-v2`
- At query time, embeds the query using the same model
- Computes cosine similarity between the query vector and every document vector in real time
- Returns the top 5 most semantically relevant results with scores, a visual score bar, and a match distribution chart

---

## Architecture

```
Raw legal documents
        │
        ▼
  Text preprocessing
  (cleaning, deduplication)
        │
        ▼
  SentenceTransformer
  all-mpnet-base-v2
  (768-dim embeddings)
        │
        ▼
  Doc_Embeddings.npy        ← precomputed, saved to disk
  (n_docs × 768 matrix)
        │
        ▼
  Streamlit app
        │
  User query → embed → cosine similarity → top-5 ranked results
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Embedding model | `sentence-transformers/all-mpnet-base-v2` |
| Similarity metric | Cosine similarity (NumPy, from scratch) |
| Preprocessing | Pandas, Python |
| Embedding computation | Kaggle GPU notebook |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |
| Storage | NumPy `.npy` matrix (precomputed embeddings) |

---

## Why MPNet over MiniLM?

`all-mpnet-base-v2` achieves higher semantic accuracy than the lighter `all-MiniLM-L6-v2` at the cost of inference speed. For a search engine over legal text — where precision matters and real-time generation isn't required — the quality tradeoff is worth it.

| Model | Embedding dim | Speed | Quality |
|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | Fast | Good |
| all-mpnet-base-v2 | 768 | Slower | Better |

---

## Cosine similarity — implemented from scratch

```python
def cosine_top_k(query_vec, doc_vecs, k=5):
    q = query_vec.reshape(1, -1)
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)
    d_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    sims = np.dot(q_norm, d_norm.T).flatten()
    idx  = np.argsort(sims)[::-1][:k]
    return idx, sims[idx]
```

The `+ 1e-10` epsilon prevents division by zero for zero-norm vectors. Normalization is applied to both query and document vectors so the dot product equals cosine similarity.

---

## How to run locally

```bash
git clone https://github.com/Zaid-Rahim/lexis-search
cd lexis-search
pip install -r requirements.txt
streamlit run app.py
```

**requirements.txt:**
```
streamlit
sentence-transformers
numpy
pandas
```

You will also need `Doc_Embeddings.npy` in the root directory. Generate it by running the preprocessing notebook on your corpus:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embeddings = model.encode(documents, show_progress_bar=True)
np.save("Doc_Embeddings.npy", embeddings)
```

---

## Project structure

```
├── app.py                  # Streamlit frontend
├── sementic-similarity     # Kaggle notebook — preprocessing + embedding generation
│   search.ipynb
├── Doc_Embeddings.npy      # Precomputed document embeddings (768-dim per doc)
├── requirements.txt
└── README.md
```

---

## Key learnings

- **Semantic vs keyword search:** The engine correctly retrieves documents about "custodial torture" when searching "physical abuse in detention" — zero keyword overlap, high semantic similarity
- **Why precompute embeddings:** Encoding thousands of documents at query time would take minutes; precomputing and saving as `.npy` makes retrieval instantaneous
- **Cosine vs dot product:** Raw dot product is affected by vector magnitude; cosine normalizes for it, making similarity purely about direction (meaning)
- **Non-English query behavior:** Urdu queries against an English-trained model produce low similarity scores — direct evidence of why multilingual embedding models exist

---

## Context

Week 3 of a 12-week AI/LLM engineering roadmap. Built after implementing transformers from scratch (Week 2) and training a GPT on Urdu poetry. Next: building a full RAG pipeline with LangChain and a vector database.
