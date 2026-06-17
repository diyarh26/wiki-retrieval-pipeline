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
Branch: exp/improve-reranker-clean
Commit: report-only, not committed as an improvement
Score: best probe remained 0.4338
Time: baseline in probe 24.90s; warm variants about 5.66s-5.95s each
Files changed: report only
Artifacts changed: none
General method: conservative reranker probes using field-BM25 gating, source-count/consensus weights, chunk-BM25 weight, and small RRF add-ons
Why it is not overfit: all probes were generic source-agreement or evidence-weight changes with no query IDs, page IDs, phrase triggers, signatures, or family logic
Keep/reject: reject all retrieval changes; none beat current best 0.4338

Experiment 3 reranker variants:

```text
field require other source: 0.4338, +0.0000 vs 0.4338, neutral/reject
field min source count 2: 0.4338, +0.0000 vs 0.4338, neutral/reject
field require + score cap 0.6: 0.4241, -0.0097 vs 0.4338, reject
source count weight 0.08: 0.4338, +0.0000 vs 0.4338, neutral/reject
source count weight 0.10: 0.4326, -0.0012 vs 0.4338, reject
light consensus boost: 0.4338, +0.0000 vs 0.4338, neutral/reject
mid consensus boost: 0.4227, -0.0111 vs 0.4338, reject
chunk BM25 weight 0.45: 0.4215, -0.0123 vs 0.4338, reject
RRF add 0.02: 0.4338, +0.0000 vs 0.4338, neutral/reject
RRF add 0.05: 0.4329, -0.0009 vs 0.4338, reject
```

Collapse check: rejected variants caused notable losses on `q_public_13`, `q_public_25`, and sometimes `q_public_4`; neutral variants had no per-query movement.
Fresh-clone status: no code/artifact changes kept.

Additional Experiment 3 rescue simulations:

```text
multi-source bounded rescue, preserve top 7/8, max 1/2 rescues from ranks 11-30: no config matched or beat 0.4338
strict lexical+dense source-agreement rescue: best observed 0.4263, -0.0075 vs 0.4338, reject
```

Reason for rejection:

```text
The relevant pages for q_public_5, q_public_19, q_public_27, and q_public_28 are often present below top 10, but generic rescue rules also promote high-evidence nonrelevant pages on other queries. The best strict rescue damaged q_public_13 by -0.3010 and q_public_3 by -0.1144.
```

Date: 2026-06-18
Branch: exp/improve-reranker-clean
Commit: report-only, not committed as an improvement
Score: no CE score run in this pass
Time: not applicable
Files changed: report only
Artifacts changed: none
General method: considered evidence-snippet cross-encoder rescue after reading existing local CE experiment reports
Why it is not overfit: no CE code was added; skipped known public-unstable or previously failed CE variants
Keep/reject: reject/skip CE for this pass

Experiment 4 CE decision:

```text
lead-only top-20 CE from prior notes: 0.4393 but explicitly unstable and not to be repeated
evidence-snippet CE from prior notes: best 0.3600, far below baseline/current best
fresh-clone safety: not guaranteed because the CE model cache may not exist
decision: do not add CE code or artifacts
```
