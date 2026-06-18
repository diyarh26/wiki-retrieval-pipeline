# Final Pipeline Summary

Final branch: `exp/improve-reranker-rebalance-clean`

Final retrieval tag: `score-04501-clean`

Final retrieval commit: `b66d2f3744fd9a772c6e77d2810a7fac519c623c`

Public score:

```text
mean_ndcg@10=0.4501
```

## Query-Time Flow

1. Embed all incoming queries with `sentence-transformers/all-MiniLM-L6-v2`.
2. Retrieve candidate pages from independent sources:
   - dense page vectors
   - dense chunk FAISS index
   - title-focused dense chunk FAISS index
   - page BM25
   - expanded-query page BM25
   - chunk BM25
   - title/lead field BM25
3. Merge source candidates into a 120-page rerank pool.
4. Compute normalized reranker features:
   - dense and lexical scores
   - inverse-rank evidence
   - rare-token coverage
   - phrase overlap
   - source count
   - top-20 source agreement
   - broad query-type field BM25 weighting for diplomacy queries
5. Sort by a single global weighted score and return the top 10 page ids.

## Why This Version Is Clean

The final branch keeps only general retrieval methods. It does not use query IDs,
hardcoded page IDs, public phrase trigger lists, first-24-word signatures,
synthetic family extraction, family-aware ranking, or template-specific routing.

The archived `archive/old-05809` version was used only as an idea source. Its
high score was not used for submission because the audit found suspicious
template/family/signature/trigger logic. See `docs/old_05809_audit.md`.

## Rejected Late Experiments

| Experiment | Score | Reason rejected |
| --- | ---: | --- |
| Dense180 replacing title190 | 0.4480 | Lower score; q_public_23 regression. |
| Dense180 as primary dense chunk index | 0.4453 | Lower score; q_public_18, q_public_19, q_public_23 regressions. |
| Default + title190 + dense180 | 0.4501 | Neutral score and requires an extra untracked large artifact. |
| Lead-only cross-encoder | 0.4393 | Improved mean but unstable per-query behavior. |

## Submission Safety

No rebuild is required for the submitted run. A fresh clone should install
dependencies, run `git lfs pull`, and evaluate with `scripts/eval_public.py`.
