# Submission Artifacts

The final retrieval path loads committed artifacts from `artifacts/`. Grading
should not rebuild indexes.

## Dense Retrieval

| File | Purpose |
| --- | --- |
| `faiss.index` | Main dense chunk FAISS index. |
| `chunk_meta.json` | Page ids, chunk ids, sections, and build metadata for `faiss.index`. |
| `index_vectors.npy` | Dense page vectors used for page-level semantic retrieval. |
| `index_meta.json` | Page ids aligned with `index_vectors.npy`. |
| `page_meta.json` | Page title/content metadata aligned with corpus pages. |

## Title Dense Chunks

| File | Purpose |
| --- | --- |
| `faiss_title190.index` | Optional title-focused dense chunk index enabled by default. |
| `chunk_meta_title190.json` | Page ids and metadata aligned with `faiss_title190.index`. |

The title dense chunk source is general: it adds an independent dense signal
with stronger title context and falls back safely if the artifact is absent.

## Lexical Retrieval

| File group | Purpose |
| --- | --- |
| `bm25_*` | Page-level BM25 lexical index. |
| `chunk_bm25_*` | Chunk-level BM25 lexical index. |
| `title_lead_bm25_*` | Title/lead field BM25 index used for candidate generation and reranking features. |

## Compatibility Artifacts

| File | Purpose |
| --- | --- |
| `page_features.json` | Generic page-feature artifact retained for compatibility with previous clean experiments. |
| `page_signatures.json` | Historical artifact retained in the repository; the final retrieval path does not load or use it. |

## LFS

Large `.npy` and `.index` artifacts are Git LFS files. A fresh checkout should
run:

```bash
git lfs pull
```

before evaluation.
