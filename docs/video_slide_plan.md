# Exact 8-Slide Video Deck Instructions

Use this file to create the final PowerPoint/Google Slides deck. It follows the
assignment requirements exactly:

- Maximum 3:00 video.
- At most 10 slides; this deck uses 8.
- Both team members speak.
- Explain each pipeline stage: chunk, embed, index, retrieve.
- Show process and empirical results using charts/metrics.
- Do not scroll through code and do not paste code on slides.

## Global Design

- Style: clean technical presentation, not code-heavy.
- Background: white or very light gray.
- Accent colors:
  - Dense retrieval: blue.
  - BM25/lexical retrieval: green.
  - Reranking/fusion: purple.
  - Rejected drafts: orange/red.
- Use diagrams, tables, and one bar chart.
- Do not put Python code on slides.
- Keep each slide readable: 3-6 short bullets or one compact table.

## Speaker Split And Timing

| Slide | Topic | Speaker | Target time |
| --- | --- | --- | ---: |
| 1 | Introduction | Aleen | 0:20 |
| 2 | Overall pipeline | Aleen | 0:25 |
| 3 | Chunking | Aleen | 0:25 |
| 4 | Embedding and indexing | Aleen | 0:25 |
| 5 | Retrieval and reranking | Diyar | 0:30 |
| 6 | Development process and mid results | Diyar | 0:30 |
| 7 | Drafts and rejected experiments | Diyar | 0:25 |
| 8 | Final result and conclusion | Diyar | 0:25 |

Total target: about 2:55.

---

## Slide 1 - Introduction

Purpose: introduce the assignment task and our final result.

Layout:

- Title at top.
- Simple query-to-output diagram in the middle.
- Final score card on the right.

Title:

```text
Section B: Wikipedia Retrieval Pipeline
```

Put these bullets on the slide:

```text
Goal: return top 10 page IDs for each query
Metric: binary mean NDCG@10
Grading: run(queries) loads submitted artifacts only
Final public score: 0.4501
```

Diagram text:

```text
Query batch -> run(queries) -> ranked page IDs
```

Final score card:

```text
mean_ndcg@10 = 0.4501
fresh-clone runtime ≈ 28.76s
limit = 60s
```

Speaker notes for Aleen:

```text
Our project is an end-to-end retrieval pipeline for Section B. The input is a
batch of natural-language queries, and the output is a ranked list of page IDs
for each query. Only the top 10 are scored using mean NDCG@10. Our final public
self-test score is 0.4501, and a fresh clone runs in about 29 seconds, under the
60-second limit.
```

---

## Slide 2 - Overall Pipeline

Purpose: show the full system before going into details.

Layout:

- Two horizontal lanes:
  - Offline build lane.
  - Query-time lane.
- Use an artifacts folder icon between them.

Title:

```text
Overall Pipeline
```

Offline lane:

```text
27,074 pages
-> section-aware chunks
-> MiniLM embeddings
-> FAISS + BM25 artifacts
```

Query-time lane:

```text
queries
-> six retrieval source groups
-> 120-page candidate pool
-> weighted reranker
-> top 10 page IDs
```

Artifact callout:

```text
Artifacts are committed in artifacts/
Large .npy and .index files use Git LFS
No index rebuild during grading
```

Speaker notes for Aleen:

```text
The pipeline has an offline phase and a query-time phase. Offline, we process
27,074 page files into chunks, embed them with MiniLM, and build FAISS and BM25
artifacts. At query time, we use six retrieval source groups to create a
120-page candidate pool, then rerank those candidates with one global weighted
formula.
```

---

## Slide 3 - Chunking Stage

Purpose: explain exactly how pages are split.

Layout:

- Left: chunking method bullets.
- Right: small table with exact chunk settings and counts.
- Optional visual: page split into overlapping windows.

Title:

```text
Chunking: Section-Aware Windows
```

Put these bullets on the slide:

```text
Split pages by detected section headings
Prefix every chunk with title + section
Short pages stay as one chunk
Main dense chunks use 300-word windows
```

Exact values table:

| Setting | Value |
| --- | ---: |
| Corpus pages | 27,074 |
| Main chunk size | 300 words |
| Main chunk overlap | 60 words |
| Short-page threshold | 320 words |
| Main dense chunks | 267,809 |
| Dense chunk page score | max chunk + 0.10 * avg top 3 |

Add a small note:

```text
Rejected draft: 180/45 lead chunking dropped to 0.4059
```

Speaker notes for Aleen:

```text
For chunking, we used section-aware preprocessing. The code detects headings,
keeps the section name, and prefixes each chunk with the page title and section.
The main index uses 300-word chunks with 60-word overlap. Pages with 320 words
or fewer stay as one chunk. This created 267,809 main dense chunks. We also
tested shorter 180/45 lead chunks, but that dropped the score to 0.4059, so we
rejected it.
```

---

## Slide 4 - Embedding And Indexing Stage

Purpose: explain embeddings, FAISS indexes, and lexical indexes with exact
artifact values.

Layout:

- Left: dense indexing.
- Right: BM25 indexing.
- Bottom: artifact names.

Title:

```text
Embedding And Indexing
```

Dense indexing box:

```text
Model: sentence-transformers/all-MiniLM-L6-v2
Embedding dimension: 384
Normalized vectors
FAISS type: IndexFlatIP
```

Dense artifacts table:

| Artifact | Purpose | Count |
| --- | --- | ---: |
| `faiss.index` | main dense chunks | 267,809 vectors |
| `index_vectors.npy` | dense page vectors | 27,074 vectors |
| `faiss_title190.index` | title-focused dense chunks | 427,630 vectors |

Title-190 settings:

| Setting | Value |
| --- | ---: |
| Chunk size | 190 words |
| Overlap | 45 words |
| Short-page threshold | 220 words |
| Context | title + section prepended |

BM25 artifacts table:

| Artifact group | Scope | Exact values |
| --- | --- | --- |
| `bm25_*` | full page BM25 | 27,074 docs, 488,972 terms |
| `chunk_bm25_*` | chunk BM25 | 267,809 chunks |
| `title_lead_bm25_*` | title + first lead paragraph | 27,074 docs, 82,675 terms |

BM25 settings:

```text
k1 = 1.2, b = 0.75, title weight = 3.0
title+lead uses first 160 lead words
```

Speaker notes for Aleen:

```text
For embedding, we used the required all-MiniLM-L6-v2 model. It gives
384-dimensional normalized vectors, and we use FAISS IndexFlatIP. We store
three dense artifacts: main chunk vectors, page vectors, and a title-focused
190-word chunk index. For lexical search, we built page BM25, chunk BM25, and
title-plus-lead BM25. BM25 uses k1 1.2 and b 0.75, with title tokens weighted
by 3.
```

---

## Slide 5 - Retrieval And Reranking Stage

Purpose: explain candidate generation, source top-k values, and final features.

Layout:

- Top: six retrieval source groups feeding into candidate pool.
- Bottom: compact feature weight table for reranker.

Title:

```text
Retrieval And Reranking
```

Candidate source diagram:

```text
Dense page top 100
Dense chunk search top 2500 chunks -> page scores
Title-190 dense chunk search top 2500 chunks -> page scores
Page BM25 top 100
Expanded-query page BM25 top 100
Chunk BM25 top 150
Title+lead BM25 top 100, top 50 used as evidence
        -> union -> 120 candidate pages
```

Important wording:

```text
These are six main source groups; expanded-query BM25 reuses the page BM25 artifact.
```

Visible reranker feature table:

| Signal | Weight |
| --- | ---: |
| Dense score | 0.913 |
| Chunk BM25 score | 0.549 |
| Phrase overlap | 0.203 |
| Field BM25 score | 0.163 |
| Field BM25 rank | 0.170 |
| Source count | 0.058 |
| Rare-token coverage | 0.050 |
| Page BM25 score | 0.044 |
| Chunk dense score | 0.033 |
| Top-20 source consensus | 0.019 |

Small note:

```text
Final score = global weighted fusion, not query-specific rules
```

Speaker notes for Diyar:

```text
At query time, candidates come from dense and lexical sources. Dense page
retrieval returns 100 pages. Dense chunk retrieval searches 2,500 chunks and
aggregates them by page. The title-190 dense index does the same. Page BM25
returns 100 pages, expanded-query BM25 also returns 100 using the same page BM25
artifact, chunk BM25 returns 150, and title-plus-lead BM25 returns 100 with the
top 50 used as evidence. These are unioned into 120 candidate pages. Then the
reranker uses global weights, led by dense score, chunk BM25, phrase overlap,
field BM25, and source agreement.
```

---

## Slide 6 - Development Process And Mid Results

Purpose: show the empirical process that led to the final design.

Layout:

- Main visual: vertical bar chart.
- Right side: short "decision rule" box.

Title:

```text
Development Process: Scores Improved Step By Step
```

Bar chart data:

| Experiment checkpoint | Public NDCG@10 |
| --- | ---: |
| Starter baseline | 0.3289 |
| Clean baseline | 0.3783 |
| Title+lead BM25 | 0.3883 |
| Field/candidate improvements | 0.4076 |
| Safe source-consensus baseline | 0.4300 |
| Title chunks + candidate pool | 0.4338 |
| Phrase/rare-token rebalance | 0.4394 |
| Final reranker rebalance | 0.4501 |

Chart settings:

```text
Use vertical bars
Y-axis min = 0.30
Y-axis max = 0.47
Show labels with 4 decimals
Color final bar purple
Color safe 0.4300 bar green
Color earlier bars blue/gray
```

Decision rule box:

```text
Keep only if:
1. score improved or stayed equal
2. no severe query collapse
3. method was general, not public-query-specific
```

Speaker notes for Diyar:

```text
We followed a controlled experiment process. The initial starter baseline was
0.3289, and after cleaning the system we reached 0.3783. Field BM25 and
candidate improvements moved the score to 0.4076. The safe source-consensus
baseline was 0.4300. Then title chunks and candidate pool tuning reached
0.4338, phrase and rare-token rebalance reached 0.4394, and the final global
reranker rebalance reached 0.4501.
```

---

## Slide 7 - Drafts And Rejected Experiments

Purpose: show that final choices were justified by failed experiments, not just
chosen randomly.

Layout:

- Table with rejected draft, score, and reason.
- Use red/orange for rejected rows.

Title:

```text
Rejected Drafts: Evidence-Based Decisions
```

Table:

| Draft / experiment | Score | Decision |
| --- | ---: | --- |
| Archived old branch | 0.5809 | Rejected: suspicious template/family/signature logic |
| Full 180/45 lead chunking | 0.4059 | Rejected: shorter chunks hurt ranking |
| Dense180 replacing title190 | 0.4480 | Rejected: lower than final |
| Dense180 as primary dense chunks | 0.4453 | Rejected: query regressions |
| Default + dense180 extra artifact | 0.4501 | Rejected: neutral score, extra 733 MB artifact |
| Lead-only cross-encoder | 0.4393 | Rejected: unstable per-query regressions |
| Evidence-snippet cross-encoder | 0.3600 | Rejected: large score drop |
| Pure RRF variants | below final | Rejected: less stable than weighted fusion |

Bottom takeaway:

```text
We chose the simpler clean design because it generalized better and stayed reproducible.
```

Speaker notes for Diyar:

```text
We tested several drafts and rejected them when the evidence was not good. The
old 0.5809 branch had a high public score, but it used suspicious template and
family-style logic, so we did not use it. Shorter 180/45 chunks dropped to
0.4059. Dense180 variants were worse or neutral but added large artifact cost.
Cross-encoders had some individual wins but were unstable and below the final
score.
```

---

## Slide 8 - Final Result And Conclusion

Purpose: end with final reproducible result and why it is clean/general.

Layout:

- Left: final result card.
- Right: clean/general checklist.
- Bottom: one-sentence conclusion.

Title:

```text
Final Result And Conclusion
```

Final result card:

```text
Final branch: main
Clean checkpoint: score-04501-clean
Public mean_ndcg@10: 0.4501
Fresh-clone query time: 28.76s
```

Submission readiness checklist:

```text
Fresh clone works
Git LFS artifacts included
No rebuild needed during grading
Runtime under 60 seconds
```

Clean/generalization checklist:

```text
No query IDs
No hardcoded page IDs
No public phrase trigger lists
No synthetic family logic
No first-word signatures
No template-specific routing
```

Conclusion sentence:

```text
Final design = multiple general retrieval sources + one global weighted reranker.
```

Speaker notes for Diyar:

```text
The final repository is on main. The clean checkpoint is score-04501-clean, and
the final public score is 0.4501. A fresh clone with Git LFS pulled runs without
rebuilding indexes and finishes in about 28.76 seconds. The final design is
clean because it uses general retrieval sources and one global reranker, without
query IDs, hardcoded page IDs, public trigger phrases, synthetic families, or
template-specific routing.
```

---

## Prompt For Creating The PowerPoint

Paste this into a slide-generation chat/tool:

```text
Create an 8-slide PowerPoint for a 3-minute university project video using the
exact instructions in docs/video_slide_plan.md. The slide order must be:
1. Introduction
2. Overall pipeline
3. Chunking
4. Embedding and indexing
5. Retrieval and reranking
6. Development process and mid results
7. Drafts and rejected experiments
8. Final result and conclusion

Use the exact tables, chart values, constants, and speaker notes from the file.
Make Slide 6 a vertical bar chart with all NDCG values. Make Slide 7 a rejected
experiments table. Do not paste Python code on slides. Use diagrams and compact
tables. Aleen speaks on slides 1-4, and Diyar speaks on slides 5-8.
```

## Final Recording Checklist

- Keep total video under 3:00.
- Both team members speak.
- Explain chunk, embed, index, and retrieve stages.
- Show Slide 6 chart clearly.
- Mention final score `0.4501`.
- Mention fresh-clone runtime `28.76s`.
- Mention that old `0.5809` was rejected as suspicious/overfit.
- After uploading the video, replace `Video link: TODO` in `README.md`.
