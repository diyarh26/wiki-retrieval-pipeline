Date: 2026-06-18
Branch: exp/general-improve-from-04300
Commit: 24f628b316d07bdaf36ea72c6b25d2f4c3f522c4
Score: 0.4300
Time: 23.69s
Files changed: none for baseline verification
Artifacts changed: none
General method: safe clean generalized hybrid retrieval baseline
Why it is not overfit: established safe baseline; no public-query-specific logic added
Keep/reject: keep as starting best

Date: 2026-06-18
Branch: exp/improve-old-clean-recovery
Commit: pending in same improvement commit
Score: 0.4329
Time: 24.80s official eval (`scripts/eval_public.py`)
Files changed: `retrieve.py`, `docs/old_05809_audit.md`, `docs/score_log.md`, title-190 artifacts
Artifacts changed: add `artifacts/faiss_title190.index`, add `artifacts/chunk_meta_title190.json`
General method: enable the existing optional title dense chunk retrieval source by default and merge its page scores with the standard dense chunk source
Why it is not overfit: no query IDs, hardcoded page IDs, phrase triggers, signatures, or family logic; title chunks are a general candidate/evidence source and fall back safely if absent
Keep/reject: keep; current best clean score

Experiment 1 rejected variants:

```text
title chunks only:        0.4329, +0.0029 vs baseline, keep
field title+lead+combined: 0.4224, -0.0076 vs baseline, reject
title chunks + all fields: 0.4164, -0.0136 vs baseline, reject
```

Per-query deltas for kept title-chunk variant:

```text
q_public_4   0.6900 -> 0.6900  +0.0000
q_public_5   0.0000 -> 0.0000  +0.0000
q_public_6   0.1781 -> 0.1667  -0.0114
q_public_7   0.1900 -> 0.1900  +0.0000
q_public_10  0.0000 -> 0.0000  +0.0000
q_public_13  0.3155 -> 0.3010  -0.0144
q_public_15  1.0000 -> 1.0000  +0.0000
q_public_17  0.0600 -> 0.0634  +0.0034
q_public_18  0.1542 -> 0.1542  +0.0000
q_public_19  0.0608 -> 0.0608  +0.0000
q_public_20  0.1404 -> 0.1771  +0.0368
q_public_21  0.6934 -> 0.6934  +0.0000
q_public_22  0.0000 -> 0.0000  +0.0000
q_public_23  0.3274 -> 0.3974  +0.0700
q_public_25  0.6199 -> 0.6199  +0.0000
q_public_26  0.5701 -> 0.5701  +0.0000
q_public_27  0.0000 -> 0.0000  +0.0000
q_public_28  0.0000 -> 0.0000  +0.0000
```

Collapse check: no query lost more than 0.0144 NDCG in the kept variant; previous CE-loss queries were stable except `q_public_6` at -0.0114.
Fresh-clone status: requires title-190 artifacts to reproduce the 0.4329 score; code falls back safely if they are absent.

Date: 2026-06-18
Branch: exp/improve-candidate-pool-clean
Commit: pending in same improvement commit
Score: 0.4338
Time: 24.93s official eval (`scripts/eval_public.py`)
Files changed: `retrieve.py`, `docs/score_log.md`
Artifacts changed: none beyond title-190 artifacts inherited from `score-04329-clean`
General method: increase only the rerank/merged chunk-dense candidate pool from 100 to 120 while leaving source top-k settings and final scoring weights unchanged
Why it is not overfit: no query-specific logic; this is a small recall-oriented candidate-pool increase over generic independent retrieval sources
Keep/reject: keep; current best clean score

Experiment 2 candidate-pool variants:

```text
rerank pool 120: 0.4338, +0.0009 vs 0.4329, keep
rerank pool 125: 0.4338, +0.0009 vs 0.4329, tie but reject in favor of smaller pool
rerank pool 150: 0.4338, +0.0009 vs 0.4329, tie but reject in favor of smaller pool
rerank pool 175: 0.4332, +0.0003 vs 0.4329, reject
rerank pool 200: 0.4332, +0.0003 vs 0.4329, reject
all source pools 150: 0.4202, -0.0127 vs 0.4329, reject
all source pools 200: 0.4236, -0.0092 vs 0.4329, reject
chunk BM25 top 200 only: 0.4308, -0.0021 vs 0.4329, reject
field BM25 150/rank100: 0.4233, -0.0095 vs 0.4329, reject
field BM25 200/rank150: 0.4190, -0.0139 vs 0.4329, reject
```

Per-query deltas for kept rerank-pool variant vs `score-04329-clean`:

```text
q_public_4   0.6900 -> 0.7055  +0.0155
q_public_5   0.0000 -> 0.0000  +0.0000
q_public_6   0.1667 -> 0.1667  +0.0000
q_public_7   0.1900 -> 0.1900  +0.0000
q_public_10  0.0000 -> 0.0000  +0.0000
q_public_15  1.0000 -> 1.0000  +0.0000
q_public_18  0.1542 -> 0.1650  +0.0108
q_public_19  0.0608 -> 0.0608  +0.0000
q_public_21  0.6934 -> 0.6934  +0.0000
q_public_22  0.0000 -> 0.0000  +0.0000
q_public_23  0.3974 -> 0.3974  +0.0000
q_public_25  0.6199 -> 0.6199  +0.0000
q_public_26  0.5701 -> 0.5701  +0.0000
q_public_27  0.0000 -> 0.0000  +0.0000
q_public_28  0.0000 -> 0.0000  +0.0000
```

Collapse check: no per-query losses for the kept 120-pool variant; gains came from `q_public_4` and `q_public_18`.
Fresh-clone status: should work from a fresh clone after LFS pull; no new artifacts added in this experiment.

Date: 2026-06-18
Branch: exp/continue-clean-improvements
Commit: pending in same improvement commit
Score: 0.4394
Time: 26.59s official eval (`scripts/eval_public.py`)
Delta vs 0.4338: +0.0056
Files changed: `retrieve.py`, `docs/score_log.md`
Artifacts changed: none
Method: increase generic query/document phrase-overlap weight from 0.068 to 0.136 and disable the expanded-rare-token bonus
Why it is general / not overfit: both are global lexical reranker weights over existing generic features; no query IDs, page IDs, public phrase triggers, signatures, families, or template-specific logic
Per-query wins:

```text
q_public_4   0.7055 -> 0.7172  +0.0116
q_public_13  0.3010 -> 0.3155  +0.0144
q_public_14  0.3333 -> 0.3869  +0.0535
q_public_18  0.1650 -> 0.2487  +0.0836
```

Per-query losses:

```text
none observed in validation vs score-04338-clean
```

Keep/reject: keep; new current best clean score

Validation notes:

```text
phrase weight x1.5: 0.4351, +0.0013 vs 0.4338, below checkpoint threshold
phrase weight x1.75/x2.0/x2.25: 0.4385, +0.0047 vs 0.4338
phrase x2.0 + expanded rare weight 0.0: 0.4394, +0.0056 vs 0.4338, kept
phrase x2.0 + rare weight 0.0: 0.4390, +0.0052 vs 0.4338, rejected in favor of higher score
phrase x2.0 + field boosts: 0.4390, +0.0052 vs 0.4338, rejected in favor of higher score
phrase x2.0 + chunk dense 0.75: 0.4391, +0.0053 vs 0.4338, rejected because it introduced q_public_23 loss
```

Date: 2026-06-18
Branch: exp/improve-reranker-rebalance-clean
Commit: pending in same improvement commit
Score: 0.4501
Time: 26.18s official eval (`scripts/eval_public.py`)
Delta vs 0.4300: +0.0201
Delta vs 0.4338: +0.0163
Delta vs current best 0.4394: +0.0107
Files changed: `retrieve.py`, `docs/score_log.md`
Artifacts changed: none
LFS needed: existing retrieval artifacts only; no new artifacts added
General method: rounded global final-fusion rebalance that strengthens dense score, chunk BM25 score, expanded BM25 score, chunk-dense score, field BM25 rank, rare-token coverage, and generic phrase-overlap evidence while reducing page-level BM25 score, expanded-BM25 rank, and chunk-dense rank pressure
Why it is general / not overfit: all changes are global reranker weights over existing generic retrieval features; no query IDs, page IDs, public phrase trigger lists, signatures, family extraction, template routing, or public-query detection were added
Per-query wins vs 0.4394:

```text
q_public_12  0.6309 -> 1.0000  +0.3691
```

Per-query losses vs 0.4394:

```text
q_public_23  0.3974 -> 0.3890  -0.0083
q_public_25  0.6199 -> 0.5701  -0.0498
```

Focus-query monitor vs 0.4394:

```text
q_public_5   0.0000 -> 0.0000  +0.0000
q_public_10  0.0000 -> 0.0000  +0.0000
q_public_18  0.2487 -> 0.2487  +0.0000
q_public_19  0.0608 -> 0.0608  +0.0000
q_public_22  0.0000 -> 0.0000  +0.0000
q_public_25  0.6199 -> 0.5701  -0.0498
q_public_26  0.5701 -> 0.5701  +0.0000
q_public_27  0.0000 -> 0.0000  +0.0000
q_public_28  0.0000 -> 0.0000  +0.0000
```

Previous CE-loss monitor vs 0.4394:

```text
q_public_4   0.7172 -> 0.7172  +0.0000
q_public_6   0.1667 -> 0.1667  +0.0000
q_public_7   0.1900 -> 0.1900  +0.0000
q_public_15  1.0000 -> 1.0000  +0.0000
q_public_21  0.6934 -> 0.6934  +0.0000
q_public_23  0.3974 -> 0.3890  -0.0083
```

Collapse check: no severe good-query collapse observed; `q_public_25` has a moderate -0.0498 loss and should be monitored in the next experiment.
Fresh-clone status: should work from a fresh clone after LFS pull; no new artifacts or environment variables are required.
Keep/reject: keep; crossed the 0.45 checkpoint and paused for report

Validation notes:

```text
full random weighted mix: 0.4496, rejected as too fussy and lower than the rounded compact variant
compact without type/RRF/top50/field-score changes: 0.4501, kept as the basis
rounded compact 13-weight variant: 0.4501, matched exact compact score
rounded compact with title/source-count left at current values: 0.4501, kept simpler
rounded small core without expanded BM25/chunk-dense stabilizers: 0.4482, rejected because q_public_23 dropped to 0.3317
rounded conservative core: 0.4375, rejected
lead dense optional source: 0.4315, rejected
title+lead optional dense source: 0.4338, rejected as neutral vs previous best
```
