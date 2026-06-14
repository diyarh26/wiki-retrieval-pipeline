# VM Handoff - Section B Retrieval

This note is for continuing the project from the Azure GPU VM repo at:

```text
/home/student/wiki-retrieval-pipeline
```

## Current Goal

Improve Section B retrieval quality while preserving the required API:

```python
run(queries: list[str]) -> list[list[int]]
```

The system must load prebuilt artifacts from `artifacts/` at grading time. The autograder does not rebuild the index.

## Current Git State To Expect

Latest pushed code includes:

- Section-aware chunking in `chunk.py`
- FAISS `IndexFlatIP` artifact build/load in `index.py`
- Hybrid retrieval/rank fusion in `retrieve.py`

Important commits:

```text
e1bffe10 chore: establish runnable baseline
dd2bea8c feat: add section-aware faiss retrieval
b485248a feat: fuse chunk and page retrieval signals
```

## Known Results

Baseline page-level starter on local machine:

```text
public_queries=29
mean_ndcg@10=0.3289
query_phase_time=4.22s
```

Chunk-only FAISS result on VM before hybrid rank fusion:

```text
public_queries=29
mean_ndcg@10=0.1698
query_phase_time=2.06s
```

That score is worse than baseline, so do not treat pure chunk retrieval as final.

## VM Artifact State

The VM already built the full chunked FAISS artifacts:

```text
artifacts/faiss.index      ~393M
artifacts/chunk_meta.json  ~9.3M
artifacts/page_meta.json   ~1.9M
```

The repo also needs the page-level baseline vectors for hybrid retrieval:

```text
artifacts/index_vectors.npy  should be ~40M, not 133 bytes
artifacts/index_meta.json
```

If `index_vectors.npy` is only 133 bytes, it is an LFS pointer. Run:

```bash
git lfs install
git lfs pull
```

## Next Step

Do not rebuild first. Pull latest code and evaluate hybrid ranking using the already-built VM artifacts:

```bash
cd ~/wiki-retrieval-pipeline
git status
git pull
git lfs install
git lfs pull
ls -lh artifacts
python scripts/eval_public.py
```

Report:

- `git status`
- `ls -lh artifacts`
- `python scripts/eval_public.py` output

## Working Rules

- Do not run full builds silently.
- Before a full build, run a small/timed trial and report.
- Commit one focused phase at a time.
- Do not modify read-only files:
  - `eval.py`
  - `scripts/build_index.py`
  - `scripts/eval_public.py`
- Keep artifacts only after they are proven useful.
