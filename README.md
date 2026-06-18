# Section B Retrieval Pipeline

## Task

This repository contains the Section B Wikipedia retrieval system. Given a
natural-language query, the pipeline returns the top 10 page IDs using committed
offline artifacts; grading should not rebuild indexes.

The submitted method is a clean hybrid retriever: it combines dense semantic
retrieval, BM25 lexical retrieval, title-focused dense chunks, title/lead field
BM25, generic lexical evidence, and source-agreement reranking.

## Final Result

Public self-test:

```text
mean_ndcg@10=0.4501
```

Reference scoring checkpoint:

```text
Final branch: main
Tag: score-04501-clean
Commit: b66d2f3744fd9a772c6e77d2810a7fac519c623c
```

Safe fallback:

```text
Tag: safe-04300
Commit: 24f628b316d07bdaf36ea72c6b25d2f4c3f522c4
Score: 0.4300
```

Video link: https://1drv.ms/v/c/cd6dfd0f8d9fc96e/IQDIH48K6evxQZSImqTnZ1qtAR2pV_Z5EjH73nEY7dF0mow?e=GjCp6h

## Fresh Clone Evaluation

No rebuild is required. After cloning, pull Git LFS artifacts and run the public
evaluation script:

```bash
git clone https://github.com/diyarh26/wiki-retrieval-pipeline.git
cd wiki-retrieval-pipeline
git lfs pull
pip install -r requirements.txt
python scripts/eval_public.py
```

Expected output:

```text
public_queries=29
mean_ndcg@10=0.4501
```

If using the course VM or an existing virtual environment, replace `python` with
that interpreter, for example:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/student/wiki-retrieval-pipeline/.venv/bin/python scripts/eval_public.py
```

For local development only, the offline build workflow is:

```bash
pip install -r requirements.txt
python scripts/build_index.py
python scripts/eval_public.py
```

The grading workflow should not need to rebuild the index because the required
artifacts are already committed under `artifacts/`.

## Required Artifacts

The active submission path loads these artifact groups from `artifacts/`:

| Artifact | Purpose |
| --- | --- |
| `faiss.index` | Main dense chunk FAISS index. |
| `chunk_meta.json` | Page/chunk metadata aligned with `faiss.index`. |
| `index_vectors.npy` | Dense page vectors for page-level semantic retrieval. |
| `index_meta.json` | Page-id metadata aligned with `index_vectors.npy`. |
| `bm25_*` | Page-level BM25 lexical index. |
| `chunk_bm25_*` | Chunk-level BM25 lexical index. |
| `title_lead_bm25_*` | Title/lead field BM25 lexical index. |
| `faiss_title190.index` | Title-focused dense chunk FAISS index. |
| `chunk_meta_title190.json` | Metadata aligned with `faiss_title190.index`. |
| `page_features.json` | Generic page metadata artifact retained for compatibility. |

Large binary artifacts are stored with Git LFS. Staff should not need to rebuild
indexes during grading.

More detail is in [docs/artifacts.md](docs/artifacts.md).

## Pipeline

1. Chunking
   - `chunk.py` splits each Wikipedia page into section-aware text chunks.
   - The main dense index uses standard chunks.
   - The title-190 dense artifact adds stronger title context.

2. Embedding
   - `embed.py` embeds chunks and queries with
     `sentence-transformers/all-MiniLM-L6-v2`.

3. Indexing
   - `index.py` loads the main dense FAISS index and page vectors.
   - `lexical.py`, `chunk_lexical.py`, and `field_lexical.py` load the BM25
     artifacts.

4. Retrieval
   - Candidate pages come from MiniLM dense page retrieval, dense chunk
     retrieval, page BM25, expanded-query BM25, chunk BM25, title+lead field
     BM25, and title-190 dense chunks.
   - The reranker uses a candidate pool of 120 pages.

5. Reranking
   - `retrieve.py` applies a global weighted fusion over normalized source
     scores, source ranks, rare-token coverage, phrase overlap, source count,
     and top-20 source agreement.
   - The final version is dense-led, but chunk BM25, field BM25, phrase
     overlap, and source agreement are important stabilizers.

## Main Experiment History

| Experiment | Public mean NDCG@10 | Decision |
| --- | ---: | --- |
| Starter baseline | 0.3289 | Historical baseline. |
| Clean baseline after removing overfit logic | 0.3783 | Clean reset point. |
| Title+lead BM25 | 0.3883 | Kept as general field lexical evidence. |
| Field BM25 / candidate improvements | 0.4076 | Kept as general candidate improvement. |
| Source consensus safe baseline | 0.4300 | Tagged `safe-04300`. |
| Title chunks + candidate pool | 0.4338 | Kept as clean recall improvement. |
| Reranker rebalance | 0.4501 | Final submission candidate. |
| Full 180/45/lead chunking | 0.4059 | Rejected; lower score. |
| Dense180 variants | 0.4453-0.4501 | Rejected; worse or neutral with extra artifact cost. |
| Pure RRF variants | below final | Rejected; less stable than weighted fusion. |
| Lead-only cross-encoder | 0.4393 | Rejected as unstable despite improvement. |

The old archived `archive/old-05809` branch reached 0.5809, but it was not used
for submission because the audit found template/family/signature/query-trigger
logic that should not generalize. See
[docs/old_05809_audit.md](docs/old_05809_audit.md).

## Cleanliness / Anti-Overfit Notes

The final retrieval path does not use:

```text
query IDs
hardcoded page IDs
public-query phrase triggers
public-query-specific rules
synthetic family extraction
first-24-word signatures
family-aware ranking
cross-encoder reranking
template-specific routing
```

All final improvements are global retrieval methods: broader clean candidate
generation, title-focused dense evidence, field BM25, chunk BM25, and generic
weighted source fusion.

## Team Members

- Aleen Nijim
- Diyar Husayyan

## Collaboration Note

Development used iterative experimentation with Codex assistance. The submitted
branch keeps only clean, general retrieval changes; suspicious archived logic was
audited and rejected rather than copied.
