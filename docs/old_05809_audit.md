# Audit: archive/old-05809 vs safe-04300

Date: 2026-06-18
Baseline branch/tag: `score/clean-generalized` / `safe-04300`
Safe commit: `24f628b316d07bdaf36ea72c6b25d2f4c3f522c4`
Archive branch/commit: `archive/old-05809` / `d8a313cc015f76ba04aa7e486ac2b38b53b90e12`

Baseline verification before audit:

```text
public_queries=29
mean_ndcg@10=0.4300
query_phase_time=23.69s
```

Commands used for comparison:

```bash
git diff --stat safe-04300 archive/old-05809
git diff safe-04300 archive/old-05809 -- retrieve.py chunk_lexical.py scripts/build_chunk_bm25.py scripts/build_title_chunk_index.py chunk.py index.py lexical.py field_lexical.py page_features.py
```

## Diff Summary

Viewing the diff as `safe-04300 -> archive/old-05809`, the archive changes these tracked paths:

```text
M artifacts/page_features.json
D artifacts/title_lead_bm25_*.npy/json
D codex_4076_candidate.diff
M docs/experiments.md
M embed.py
D field_lexical.py
M lexical.py
M page_features.py
M retrieve.py
D scripts/build_field_bm25.py
M scripts/diagnose_retrieval.py
```

No code differences were found in the requested comparison for:

```text
chunk_lexical.py
scripts/build_chunk_bm25.py
scripts/build_title_chunk_index.py
chunk.py
index.py
```

## A. Safe / General Candidates

These ideas are general retrieval methods, but most are already present in a cleaner form in `safe-04300`.

1. Multi-source candidate generation in `retrieve.py`
   - Archive uses page dense, page BM25, expanded page BM25, chunk BM25, and dense chunk retrieval as candidate sources.
   - General value: independent retrievers increase recall before reranking.
   - Port status: already implemented more cleanly in `safe-04300`, with the additional clean field-BM25 source.

2. Chunk BM25 as a lexical passage retriever
   - Archive can load `chunk_bm25` artifacts and rank pages by aggregating high-scoring chunks.
   - General value: passage-level lexical matching can recover pages whose full-page BM25 score is diluted.
   - Port status: already present in `safe-04300`; no archive code change needed.

3. Optional title dense chunk index loading
   - Archive has optional `faiss_title190.index` / `chunk_meta_title190.json` loading behind `WIKI_USE_TITLE_CHUNKS`.
   - General value: title-focused dense chunks can help when titles carry entity evidence.
   - Port status: already present in `safe-04300`, disabled by default.

4. Normalized weighted score fusion
   - Archive combines normalized dense, BM25, chunk BM25, chunk dense, lexical overlap, title overlap, phrase overlap, and source count features.
   - General value: a linear hybrid reranker over retrieval evidence is defensible.
   - Caution: archive weights include negative evidence weights and appear heavily tuned to public behavior.
   - Port status: `safe-04300` already has a cleaner positive-evidence fusion and diagnostic rows.

5. Generic text overlap features
   - Archive computes rare-token coverage, expanded rare-token coverage, title overlap, and query n-gram phrase hits.
   - General value: lexical overlap features are normal reranking signals.
   - Caution: archive uses template query expansion and some negative weights, so only the general feature shapes are candidates.
   - Port status: already present in `safe-04300`.

6. Source agreement / consensus
   - Archive uses source count; the safe code extends this with clean source consensus and RRF-style diagnostics.
   - General value: pages retrieved by multiple independent sources are often more reliable.
   - Port status: already improved in `safe-04300`.

7. Broad page-type compatibility
   - A broad query/page type match can be general if based on generic vocabulary.
   - Caution: archive's implementation is not generic; it contains public-template phrases and synthetic-family parsing.
   - Port status: safe version keeps only generic type hints.

8. Diagnostics around candidate recall
   - Archive contains diagnostic concepts for inspecting per-source candidates.
   - General value: useful for experiment analysis, not query-time ranking.
   - Port status: `safe-04300` already has richer `debug_search_batch` and diagnostic support.

## B. Suspicious / Overfit Changes Rejected

These must not be ported.

1. `lexical.py`: `QUERY_EXPANSIONS`
   - Contains public-shaped trigger phrases and expansions such as company expansion, factory modernization, riverfront festivals, field trials, treaty routes, and basketball finals language.
   - Violates the ban on public phrase trigger lists.
   - Rejected.

2. `page_features.py`: synthetic-template regex extraction
   - `_SPORTS_RE`, `_COMPANY_RE`, `_CITY_RE`, `_RESEARCH_RE`, and `_DIPLOMACY_RE` parse highly specific generated page openings.
   - Produces synthetic family keys from template slots such as role/team/year, company/industry/executive, city/geography/population, researcher/method/year, and treaty/person/role/site.
   - Violates the bans on synthetic family extraction, template-specific rules, and synthetic query family detection.
   - Rejected.

3. `page_features.py`: `_PHRASE_SIGNALS`
   - Contains long content phrases tied to public-style answers, such as finals details, profit-sharing, spin-offs, commuter rail, riverfront festivals, field trials, patent pools, and treaty mechanics.
   - Violates the explicit ban on `_PHRASE_SIGNALS` and public phrase trigger lists.
   - Rejected.

4. `page_features.py`: `_QUERY_SIGNAL_TRIGGERS`
   - Maps query phrases to hand-authored signals for sports, company, city, research, and diplomacy templates.
   - Violates the explicit ban on `_QUERY_SIGNAL_TRIGGERS` and public phrase triggers.
   - Rejected.

5. `page_features.py`: query type weights with public phrases
   - `_QUERY_TYPE_WEIGHTS` includes many narrow phrases rather than broad topic vocabulary.
   - Violates the ban on code that detects public queries or synthetic families.
   - Rejected.

6. `page_features.py`: multi-answer triggers
   - `_MULTI_ANSWER_TRIGGERS` includes query-shape phrases such as "what links", "fit together", and specific public-style clauses.
   - Used to route into family-aware ranking.
   - Rejected.

7. `retrieve.py`: first-24-word signatures
   - `_signature_from_content` and `_load_signature_cache` group pages by the first 24 words of content.
   - Violates the explicit ban on first-24-word signatures.
   - Rejected.

8. `retrieve.py`: sibling/family candidate expansion
   - `_rerank_feature_union` expands candidates by signature siblings and `family_to_pages`.
   - Violates the bans on synthetic family extraction and family-aware ranking.
   - Rejected.

9. `retrieve.py`: family-aware ranking
   - `_family_ranked_pages`, `FAMILY_*` weights, `SIGNAL_RERANK_WEIGHT`, `FAMILY_SIGNAL_WEIGHT`, family-size weighting, and per-type `pages_per_family` rules directly rank synthetic families.
   - Violates the explicit ban on family-aware ranking.
   - Rejected.

10. `retrieve.py`: `_should_family_rank`
    - Contains special-case query phrase conditions for company topics.
    - Violates the ban on public-query-specific logic.
    - Rejected.

11. `retrieve.py`: signal-based reranking
    - `query_signals` / `signal_score` connect public-shaped query triggers to page signals.
    - Violates the bans on `_QUERY_SIGNAL_TRIGGERS`, `_PHRASE_SIGNALS`, and public phrase logic.
    - Rejected.

12. `artifacts/page_features.json`
    - Archive artifact stores version-1 synthetic type/family/signal data created by the rejected template parser.
    - Violates the artifact-side equivalent of the same synthetic family/signals logic.
    - Rejected.

13. Tuned negative weights and candidate-order bonuses
    - Archive uses negative weights for dense rank, expanded BM25 rank, chunk rank, rare tokens, title overlap, source count, signature support, and candidate order.
    - Negative weights are not inherently forbidden, but in this archive they are entangled with signatures/families/signals and appear public-score tuned.
    - Rejected for direct porting; only clean weight searches may be tested separately.

## C. Not Useful / Obsolete

1. `field_lexical.py` deletion
   - Archive lacks the later clean title/lead/title_lead BM25 implementation.
   - Safe branch already includes `field_lexical.py` and uses it as a clean source.
   - Obsolete and not useful.

2. `scripts/build_field_bm25.py` deletion and `artifacts/title_lead_bm25_*` deletion
   - Archive removes clean field-BM25 build support and artifacts.
   - This would damage fresh-clone/default behavior if copied.
   - Not useful.

3. `scripts/diagnose_retrieval.py` simplification
   - Archive removes expanded BM25, chunk BM25, field BM25, current-run diagnostics, and feature dumps that are useful for controlled experiments.
   - Not useful.

4. `embed.py` removal of `WIKI_EMBED_DEVICE`
   - Archive removes a harmless device override from the safe branch.
   - Not a retrieval-quality improvement.
   - Not useful.

5. `docs/experiments.md` removal of cleanup notes
   - Archive predates the cleanup documentation and therefore deletes the clean-reranker note.
   - Documentation history only, not a retrieval method.
   - Not useful.

6. `codex_4076_candidate.diff` deletion
   - Removes an old patch artifact.
   - Not a retrieval method.
   - Not useful.

7. Requested files with no diffs
   - `chunk_lexical.py`, chunk/index builders, `chunk.py`, and `index.py` do not provide old-branch implementation changes to port.
   - Not useful for old-branch extraction.

## Clean Ideas To Test Next

The archive does not contain a clean standalone patch that should be copied directly. The clean experiments should instead use the safe code's existing generalized mechanisms:

1. Candidate pool expansion
   - Test larger candidate limits for page dense, page BM25, chunk BM25, field BM25, and rerank candidate union.
   - General method: increase recall before the same clean reranker.

2. Chunk BM25 / field BM25 source balance
   - Test clean weights and caps for `CHUNK_BM25_*` and field BM25 evidence.
   - General method: tune source fusion, without phrase or family rules.

3. RRF only for candidate seeding or conservative consensus
   - Test RRF as a candidate source or small consensus feature, not as public-query routing.
   - General method: rank aggregation over independent retrievers.

4. Conservative rescue from ranks 11-30
   - Promote a page only when it has multiple independent source hits.
   - General method: multi-source evidence rescue with bounded top-10 churn.

5. Snippet cross-encoder only if artifacts/models are available and runtime stays safe
   - Use snippets from generic lexical windows, preserve most baseline top-10 pages, and allow at most two rescues.
   - General method: neural reranking of evidence snippets, not query-specific triggers.

## Audit Decision

Do not port archive logic wholesale. The public-score jump appears largely tied to rejected template expansions, synthetic family extraction, signature sibling expansion, and phrase/signal triggers. Safe/general value remains in the broad hybrid-retrieval ideas, which are already mostly represented in `safe-04300`; further work should be small controlled experiments on candidate recall and clean source fusion.
