# Section B Retrieval Pipeline

## Setup

```bash
pip install -r requirements.txt
```

The corpus is expected at `data/Wikipedia Entries/`.

## Evaluation

The grader calls `main.run(queries)`. A local public self-test can be run with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/eval_public.py
```

Current clean public self-test on this branch:

```text
mean_ndcg@10=0.4501
query_phase_time=25.01s
```

## Required Artifacts

The retrieval path loads prebuilt artifacts from `artifacts/`; these should be
present in the submitted repository/LFS checkout. The active query-time sources
are:

```text
faiss.index
chunk_meta.json
page_meta.json
index_vectors.npy
index_meta.json
faiss_title190.index
chunk_meta_title190.json
bm25_*.npy/json
chunk_bm25_*.npy/json
title_lead_bm25_*.npy/json
```

`scripts/build_index.py` remains the basic offline dense-index builder, but the
submitted run is intended to load the committed artifacts rather than rebuild
them during evaluation.

## Notes

The default retrieval is a clean hybrid ranker over dense page/chunk retrieval,
BM25 page/chunk retrieval, title-lead field BM25, title dense chunks, generic
lexical overlap, and source-agreement features. It does not use query IDs,
hardcoded page IDs, public-query triggers, signature expansion, or family-aware
ranking.
