# Submission Audit

Date: 2026-06-18

Branch: `exp/improve-reranker-rebalance-clean`

Reference score tag: `score-04501-clean`

Reference score: `mean_ndcg@10=0.4501`

## Cleanup Decisions

| File/path | Reason it looked unnecessary | Deleted? | Why |
| --- | --- | --- | --- |
| `.idea/` | Local IDE project metadata. | yes | Not used by retrieval, docs, artifacts, or fresh-clone eval. |
| `__pycache__/`, `scripts/__pycache__/` | Python bytecode cache. | yes | Regenerated automatically and not tracked. |
| `build_full.log` | Local build log. | yes | Not used by final code or documentation. |
| `Project A (1).pdf`, `Researching Retrieval Pipeline for NDCG.pdf` | Local PDFs. | yes | Not needed for the grading repository. |
| `artifacts/faiss_dense180_lead.index`, `artifacts/chunk_meta_dense180_lead.json` | Failed dense180 experiment artifacts. | yes | Ignored locally and not loaded by final retrieval. |
| `artifacts/lead_bm25_*`, `artifacts/title_bm25_*` | Unused field-BM25 experiment artifacts. | yes | Final retrieval uses `title_lead_bm25_*`. |
| `scripts/build_dense180_lead_index.py` | Local builder for rejected dense180 experiment. | yes | Not tracked and not needed for final grading. |
| `codex_4076_candidate.diff` | Stale patch artifact. | yes | Not used by final code, docs, or evaluation. |
| `docs/current_query_diagnostic_report.md` | Current-code diagnostic report. | no | Useful for video/debug explanation. |
| `scripts/build_page_features.py` | Page-feature artifact utility. | no | Kept because `page_features.json` remains part of the repository. |

## Required Artifact Check

| Artifact | Exists | Tracked | LFS | Used by final code |
| --- | --- | --- | --- | --- |
| `artifacts/faiss.index` | yes | yes | yes | yes |
| `artifacts/chunk_meta.json` | yes | yes | no | yes |
| `artifacts/index_vectors.npy` | yes | yes | yes | yes |
| `artifacts/index_meta.json` | yes | yes | no | yes |
| `artifacts/bm25_*` | yes | yes | mixed | yes |
| `artifacts/chunk_bm25_*` | yes | yes | mixed | yes |
| `artifacts/title_lead_bm25_*` | yes | yes | mixed | yes |
| `artifacts/faiss_title190.index` | yes | yes | yes | yes |
| `artifacts/chunk_meta_title190.json` | yes | yes | no | yes |
| `artifacts/page_features.json` | yes | yes | no | yes |

`mixed` means the JSON metadata files are normal Git files and the NumPy arrays
are Git LFS files.

## Safety Notes

No indexes were rebuilt during submission cleanup. The cleanup did not add
public-query-specific rules, query IDs, hardcoded page IDs, phrase trigger
lists, synthetic family logic, first-word signatures, or template-specific
rules.
