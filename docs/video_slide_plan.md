# Section B Video PowerPoint Instructions

Use this file as the exact instruction sheet for creating the final
PowerPoint/Google Slides deck.

Requirements from the assignment:

- Maximum video length: 3:00.
- Maximum slides: 10.
- Both team members must speak.
- Explain each pipeline stage: chunk, embed, index, retrieve.
- Show the process followed and empirical results with metrics/plots.
- Do not present by scrolling through code or pasting code on slides.

This deck uses 8 slides and targets about 2:50, leaving a small buffer.

## Global Slide Style

Use a clean technical style:

- Background: white or very light gray.
- Accent colors:
  - Dense retrieval: blue.
  - Lexical/BM25 retrieval: green.
  - Reranking/fusion: purple.
  - Rejected experiments: red/orange.
- Font: Aptos, Calibri, Inter, or Arial.
- Title size: 34-40 pt.
- Body size: 20-26 pt.
- Keep each slide sparse. Use diagrams and charts, not paragraphs.
- Do not paste Python code.
- Put speaker notes below each slide, not on the slide itself.

## Speaker Split

| Speaker | Slides | Target time |
| --- | --- | ---: |
| Aleen | 1-4 | 1:25-1:30 |
| Diyar | 5-8 | 1:20-1:25 |

Total target: about 2:50.

## Slide 1 - Problem And Final Goal

Speaker: Aleen

Target time: 0:20

Layout:

- Title at top.
- Simple horizontal diagram in the middle.
- Final score callout box on the right.

Title:

```text
Section B Retrieval Pipeline
```

Slide bullets:

```text
Task: return top 10 page IDs for each query
Metric: mean NDCG@10 with binary relevance
Constraint: load submitted artifacts; no index rebuild at grading time
Final public score: 0.4501
```

Diagram:

```text
Query batch -> Retrieval pipeline -> Ranked top 10 page IDs
```

Callout box:

```text
Fresh-clone eval:
mean_ndcg@10 = 0.4501
runtime ≈ 28s
```

Speaker notes:

```text
Our Section B system is an end-to-end retrieval pipeline over Wikipedia-style
entries. The autograder gives a batch of queries, and our run function returns
ranked page IDs. We focused on a clean pipeline that loads prebuilt artifacts
and improves NDCG@10 without query-specific rules.
```

## Slide 2 - Full Pipeline Overview

Speaker: Aleen

Target time: 0:25

Layout:

- Use a two-row flowchart.
- Top row: offline build.
- Bottom row: query-time retrieval.
- Put "artifacts/" between offline and query-time.

Title:

```text
Pipeline Overview
```

Flowchart row 1, label it "Offline, untimed":

```text
Pages -> chunking -> MiniLM embeddings -> FAISS + BM25 artifacts
```

Flowchart row 2, label it "Query time, graded":

```text
Queries -> dense + lexical retrieval -> 120 candidates -> weighted reranker -> top 10
```

Small side box:

```text
Artifacts are committed in artifacts/
Git LFS stores large .npy and .index files
```

Speaker notes:

```text
The pipeline has two phases. Offline, we chunk pages, embed them with MiniLM,
and build FAISS and BM25 artifacts. At query time, we retrieve candidates from
several independent sources, combine them into a 120-page candidate pool, and
rerank with one global weighted score.
```

## Slide 3 - Chunk, Embed, And Index Stages

Speaker: Aleen

Target time: 0:25

Layout:

- Three vertical columns labeled Chunk, Embed, Index.
- Use small icons if available: document/chunk, vector/embedding, database/index.

Title:

```text
Chunk, Embed, Index
```

Column 1 title:

```text
Chunk
```

Column 1 bullets:

```text
Section-aware page chunks
Standard dense chunks for main FAISS
Title-190 chunks for stronger entity/title evidence
```

Column 2 title:

```text
Embed
```

Column 2 bullets:

```text
Model: all-MiniLM-L6-v2
Embeds page chunks
Embeds query batch at runtime
```

Column 3 title:

```text
Index
```

Column 3 bullets:

```text
FAISS chunk index
Dense page vectors
Page BM25
Chunk BM25
Title+lead BM25
```

Artifact strip at bottom:

```text
faiss.index | index_vectors.npy | faiss_title190.index | bm25_* | chunk_bm25_* | title_lead_bm25_*
```

Speaker notes:

```text
For dense retrieval, we use MiniLM embeddings. The main FAISS index stores
section-aware chunks. We also keep page-level dense vectors, which help with
whole-document semantic matching. A title-focused dense index gives extra
evidence when titles matter. In parallel, BM25 artifacts give lexical evidence
at page, chunk, and title-plus-lead levels.
```

## Slide 4 - Retrieval Candidate Sources

Speaker: Aleen

Target time: 0:20

Layout:

- Center circle: "120-page candidate pool".
- Six source boxes around it feeding arrows into the center.
- Use blue for dense sources, green for BM25 sources.

Title:

```text
Candidate Generation
```

Source boxes:

```text
Dense page retrieval
Dense chunk retrieval
Title-190 dense chunks
Page BM25
Chunk BM25
Title+lead BM25
```

Center box:

```text
Union of sources
120 candidate pages
```

Bottom note:

```text
Goal: improve recall before final reranking
```

Speaker notes:

```text
Candidate generation was one of the main improvements. Instead of trusting one
source, we union candidates from dense page, dense chunk, BM25, chunk BM25,
title chunks, and field BM25. We kept the final pool at 120 because it improved
recall without adding too much noisy competition.
```

## Slide 5 - Reranking And Final Scoring

Speaker: Diyar

Target time: 0:25

Layout:

- Left side: feature list grouped by type.
- Right side: simple stacked-score illustration.
- Do not list exact weight constants. Explain categories.

Title:

```text
Weighted Reranker
```

Feature groups:

```text
Dense evidence
- page dense score
- chunk dense score

Lexical evidence
- page BM25
- chunk BM25
- title+lead BM25
- rare-token coverage
- phrase overlap

Agreement evidence
- source count
- top-20 source consensus
```

Formula graphic:

```text
final score =
dense evidence
+ lexical evidence
+ agreement evidence
```

Bottom note:

```text
One global scoring formula, not query-specific rules
```

Speaker notes:

```text
After candidate retrieval, we rerank with a single global weighted fusion. The
ranking is dense-led, but it is stabilized by lexical evidence and source
agreement. A page that appears in multiple independent sources gets more
confidence than a page found by only one source.
```

## Slide 6 - Empirical Progress Chart

Speaker: Diyar

Target time: 0:25

Layout:

- Use a vertical bar chart.
- X-axis: experiment names.
- Y-axis: Public mean NDCG@10.
- Highlight the final bar in purple or dark blue.
- Put numeric labels above every bar.

Title:

```text
Empirical Progress
```

Chart type:

```text
Vertical bar chart
```

Chart data:

| Experiment | Score |
| --- | ---: |
| Starter baseline | 0.3289 |
| Clean baseline | 0.3783 |
| Title+lead BM25 | 0.3883 |
| Field/candidate improvements | 0.4076 |
| Source consensus reranker | 0.4300 |
| Title chunks + candidate pool | 0.4338 |
| Final reranker rebalance | 0.4501 |

Chart settings:

```text
Y-axis minimum: 0.30
Y-axis maximum: 0.47
Show data labels with 4 decimals
Final bar color: purple or dark blue
Earlier bars color: gray or light blue
```

Small callout next to final bar:

```text
+0.0201 over safe 0.4300 baseline
```

Speaker notes:

```text
We made decisions empirically. The safe clean baseline was 0.4300. Adding title
chunks and improving the candidate pool reached 0.4338. Then global reranker
rebalance improved the final public score to 0.4501. Each kept change had to
improve or preserve the score without query-specific logic.
```

## Slide 7 - Rejected Experiments And Lessons

Speaker: Diyar

Target time: 0:25

Layout:

- Use a table.
- Left column: experiment.
- Middle column: observed result.
- Right column: decision/lesson.
- Use red/orange markers for rejected rows.

Title:

```text
What We Rejected
```

Table:

| Experiment | Result | Lesson |
| --- | ---: | --- |
| Archived old branch | 0.5809 | Rejected: suspicious template/family/signature logic |
| Full 180/45 lead chunking | 0.4059 | Shorter chunks hurt final ranking |
| Dense180 variants | 0.4453-0.4501 | Worse or neutral; extra artifact cost |
| Lead-only cross-encoder | 0.4393 | Some wins, but unstable regressions |
| Evidence-snippet cross-encoder | 0.3600 | Snippets were not reliable enough |
| Pure RRF variants | below final | Less stable than weighted fusion |

Bottom note:

```text
We kept only changes that were clean, general, and empirically useful.
```

Speaker notes:

```text
We also rejected several tempting ideas. The archived 0.5809 branch had a high
public score, but the audit found template, family, signature, and trigger-style
logic, so we did not use it. Shorter chunking and cross-encoder reranking also
looked promising, but the measured scores were lower or caused regressions.
```

## Slide 8 - Final Submission And Generalization

Speaker: Diyar

Target time: 0:25

Layout:

- Use a checklist on the left.
- Use a final result card on the right.

Title:

```text
Final Clean Submission
```

Checklist:

```text
Fresh clone works with git lfs pull
No index rebuild required
Runtime ≈ 28 seconds, under 60 seconds
Final code is on main
```

Anti-overfit checklist:

```text
No query IDs
No hardcoded page IDs
No public phrase triggers
No synthetic family logic
No first-word signatures
No template-specific routing
```

Final result card:

```text
mean_ndcg@10 = 0.4501
branch = main
clean checkpoint = score-04501-clean
```

Speaker notes:

```text
The final repository is ready on main. A fresh clone with Git LFS pulled runs
the public eval without rebuilding indexes and reproduces 0.4501 in about 28
seconds. The final system is clean because the improvements are general
retrieval methods: independent candidate sources and global reranker features,
not public-query memorization.
```

## Optional Slide Generator Prompt

Paste this prompt into ChatGPT, Gemini, Canva, PowerPoint Copilot, or another
slide generator:

```text
Create an 8-slide technical presentation for a 3-minute university project
video. Use the exact slide titles, bullet text, chart data, table data, and
speaker notes from docs/video_slide_plan.md. Keep the style clean and
professional. Do not include code blocks on the slides. Use diagrams, a bar
chart, and compact tables. The presentation is about a Section B Wikipedia
retrieval pipeline with final public mean_ndcg@10 = 0.4501. Both team members
must speak: Aleen presents slides 1-4 and Diyar presents slides 5-8.
```

## Timing Checklist

| Slide | Speaker | Target |
| --- | --- | ---: |
| 1 | Aleen | 0:20 |
| 2 | Aleen | 0:25 |
| 3 | Aleen | 0:25 |
| 4 | Aleen | 0:20 |
| 5 | Diyar | 0:25 |
| 6 | Diyar | 0:25 |
| 7 | Diyar | 0:25 |
| 8 | Diyar | 0:25 |

Total target: about 2:50.

## Final Recording Checklist

- Record under 3:00.
- Both members speak clearly.
- Do not scroll through code.
- Mention the final score: `0.4501`.
- Mention fresh-clone eval works.
- Mention that the final method avoids query-specific and template-specific logic.
- After uploading the video, replace `Video link: TODO` in `README.md` with the real link.
