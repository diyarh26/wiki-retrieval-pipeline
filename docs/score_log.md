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
