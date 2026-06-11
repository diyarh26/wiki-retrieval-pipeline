# Retrieval Experiments

## Phase 2 - Starter Baseline

Date: 2026-06-11

### Environment

- Python environment: local `.venv`
- Installed packages: `numpy`, `sentence-transformers`, `faiss-cpu`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Public query file: corrected course file with 29 unique queries
- Corpus: 27,074 pages

### Starter Pipeline

- One retrieval vector per page.
- Text format: `title + content`.
- Query-time retrieval: brute-force NumPy dot product over normalized vectors.
- No chunking, section handling, FAISS search, lexical scoring, or reranking.

### Artifacts

- `artifacts/index_vectors.npy` - page-level dense vectors, 41,585,792 bytes
- `artifacts/index_meta.json` - page IDs and chunk IDs, 536,744 bytes

### Result

```text
public_queries=29
mean_ndcg@10=0.3289
query_phase_time=4.22s
```

This is the baseline to beat. It confirms that one full-page embedding is not strong enough for fact-linking questions over long pages.

## Phase 3 Target Design

Use the chunking strategy discussed by the team:

- Keep short pages as one chunk.
- Split long pages by section-like headings first.
- Split long sections into about 200-word chunks with about 40-word overlap.
- Prefix every chunk with title and section context.
- Search many chunks, then aggregate scores back to page IDs.
- Start with FAISS `IndexFlatIP` for exact inner-product search over normalized MiniLM embeddings.

Initial page scoring target:

```text
page_score = max_chunk_score + 0.10 * average_top_3_chunk_scores
```
