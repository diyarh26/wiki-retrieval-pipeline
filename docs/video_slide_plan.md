# Section B Video Slide Plan

Target: 3:00 maximum, 8 slides, both members speak.

Team split suggestion:

- Aleen: slides 1-4, about 85-90 seconds.
- Diyar: slides 5-8, about 85-90 seconds.

Keep slides visual and concise. Do not show code blocks. Use diagrams, small
tables, and one score chart.

## Slide 1 - Problem And Goal

On slide:

- Task: retrieve top 10 Wikipedia-style page IDs for each query.
- Score: binary `mean_ndcg@10`.
- Constraint: query-time run loads submitted artifacts; no index rebuild.
- Final public score: `0.4501`.

Speaker: Aleen, 15-20 seconds

Script:

> Our Section B system is an end-to-end retrieval pipeline over Wikipedia-style
> entries. The autograder gives a batch of queries, and our `run()` function
> returns ranked page IDs. We focused on a clean pipeline that loads prebuilt
> artifacts and improves NDCG@10 without query-specific rules.

Visual idea:

- Simple input/output diagram: Query batch -> Retrieval pipeline -> Top 10 page IDs.

## Slide 2 - Pipeline Overview

On slide:

```text
Pages -> chunking -> MiniLM embeddings -> FAISS + BM25 artifacts
Queries -> dense + lexical candidate retrieval -> weighted reranker -> top 10
```

Key sources:

- Dense page retrieval
- Dense chunk retrieval
- Title-190 dense chunks
- Page BM25
- Chunk BM25
- Title+lead field BM25

Speaker: Aleen, 20-25 seconds

Script:

> The pipeline has two phases. Offline, we chunk pages, embed them with
> MiniLM, and build FAISS and BM25 artifacts. At query time, we retrieve
> candidates from several independent sources, combine them into a candidate
> pool, and rerank with one global weighted score.

Visual idea:

- Flowchart with two colors: offline artifacts and query-time retrieval.

## Slide 3 - Chunking, Embedding, And Indexing

On slide:

- Section-aware chunks from `chunk.py`.
- MiniLM model: `sentence-transformers/all-MiniLM-L6-v2`.
- FAISS dense chunk index: `artifacts/faiss.index`.
- Page dense vectors: `artifacts/index_vectors.npy`.
- Title-focused dense chunks: `artifacts/faiss_title190.index`.
- BM25 artifacts for page, chunk, and title+lead fields.

Speaker: Aleen, 25 seconds

Script:

> For dense retrieval, we use MiniLM embeddings. The main FAISS index stores
> section-aware chunks. We also keep page-level dense vectors, which help with
> whole-document semantic matching. A title-focused dense index gives extra
> evidence when the title is important. In parallel, BM25 artifacts give us
> lexical evidence at page, chunk, and title-plus-lead levels.

Visual idea:

- Artifact table with 3 columns: dense, lexical, metadata.

## Slide 4 - Candidate Generation

On slide:

- Candidate pool size: 120 pages.
- Independent sources improve recall.
- Title chunks and field BM25 added clean evidence.
- No public-query triggers or hardcoded IDs.

Speaker: Aleen, 20 seconds

Script:

> Candidate generation was one of the main improvements. Instead of trusting
> one source, we union candidates from dense page, dense chunk, BM25, chunk
> BM25, title chunks, and field BM25. We kept the final pool at 120 because it
> improved recall without adding too much noisy competition.

Visual idea:

- Venn-style source-union diagram feeding into "120 candidates".

## Slide 5 - Reranking Method

On slide:

Reranker signals:

- normalized dense scores
- BM25 and chunk BM25 scores/ranks
- title+lead field BM25
- rare-token coverage
- phrase overlap
- source count and top-20 source agreement

Speaker: Diyar, 25 seconds

Script:

> After candidate retrieval, we rerank with a single global weighted fusion.
> The ranking is dense-led, but it is stabilized by lexical evidence and source
> agreement. For example, a page that appears in multiple independent sources
> gets more confidence than a page found by only one source.

Visual idea:

- Bar/stack diagram showing signals combined into one final score.

## Slide 6 - Empirical Progress

On slide:

Use a small bar chart:

| Experiment | NDCG@10 |
| --- | ---: |
| Starter baseline | 0.3289 |
| Clean baseline | 0.3783 |
| Title+lead BM25 | 0.3883 |
| Field/candidate improvements | 0.4076 |
| Source consensus reranker | 0.4300 |
| Title chunks + candidate pool | 0.4338 |
| Final reranker rebalance | 0.4501 |

Speaker: Diyar, 25 seconds

Script:

> We made decisions empirically. The safe clean baseline was 0.4300. Adding
> title chunks and improving the candidate pool reached 0.4338. Then global
> reranker rebalance improved the final public score to 0.4501. Each kept
> change had to improve or preserve the score without query-specific logic.

Visual idea:

- Bar chart from 0.3289 to 0.4501, with final bar highlighted.

## Slide 7 - What We Rejected

On slide:

Rejected experiments:

- Old `0.5809` branch: audited but rejected as suspicious/overfit.
- Full 180/45 lead chunking: dropped to `0.4059`.
- Dense180 variants: worse or neutral with extra artifact cost.
- Cross-encoder: best `0.4393`, but unstable.
- Pure RRF: less stable than weighted fusion.

Speaker: Diyar, 25 seconds

Script:

> We also rejected several tempting ideas. The archived 0.5809 branch had a
> high public score, but the audit found template, family, signature, and
> trigger-style logic, so we did not use it. Shorter chunking and cross-encoder
> reranking also looked promising, but the measured scores were lower or caused
> regressions.

Visual idea:

- "Kept vs rejected" table with green/red labels.

## Slide 8 - Final Submission And Generalization

On slide:

- Final branch on GitHub: `main`.
- Final clean checkpoint: `score-04501-clean`.
- Fresh clone eval: `mean_ndcg@10=0.4501`.
- Runtime: about `28s`, under 60s.
- No query IDs, page IDs, phrase triggers, synthetic families, signatures, or templates.

Speaker: Diyar, 20-25 seconds

Script:

> The final repository is ready on `main`. A fresh clone with Git LFS pulled
> runs the public eval without rebuilding indexes and reproduces 0.4501 in
> about 28 seconds. The final system is clean because the improvements are
> general retrieval methods: independent candidate sources and global reranker
> features, not public-query memorization.

Visual idea:

- Final checklist with score, runtime, fresh clone, and anti-overfit checks.

## Timing Plan

| Slide | Speaker | Target time |
| --- | --- | ---: |
| 1 | Aleen | 0:20 |
| 2 | Aleen | 0:25 |
| 3 | Aleen | 0:25 |
| 4 | Aleen | 0:20 |
| 5 | Diyar | 0:25 |
| 6 | Diyar | 0:25 |
| 7 | Diyar | 0:25 |
| 8 | Diyar | 0:25 |

Total target: about 2:50. This leaves a small buffer below 3:00.

## Slide Design Notes

- Use 1 architecture diagram, 1 artifact table, 1 score chart, and 1 rejected-experiment table.
- Keep each slide to 3-5 bullets.
- Do not paste code on slides.
- Put the final GitHub branch and video link in the README after recording.
- Practice once with a timer; cut one bullet from slides 3 or 7 if over 2:55.
