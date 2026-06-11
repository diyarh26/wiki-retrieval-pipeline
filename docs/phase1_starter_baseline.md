# Phase 1 - Starter Understanding and Baseline

Date: 2026-06-11

## Repository State

- Branch: `main`
- Remote: `origin` at `https://github.com/diyarh26/wiki-retrieval-pipeline.git`
- Corpus: 27,074 JSON page files under `data/Wikipedia Entries/`
- Public queries: corrected course file, 29 unique query texts
- Artifacts: empty except `artifacts/.gitkeep`

## Assignment Boundaries

Editable files from the handout:
- `main.py`
- `chunk.py`
- `embed.py`
- `index.py`
- `retrieve.py`
- `utils.py`
- `README.md`

Read-only files from the handout:
- `eval.py`
- `scripts/build_index.py`
- `scripts/eval_public.py`

## Current Starter Pipeline

- `main.run(queries)` delegates to `retrieve.search_batch`.
- `main.build_offline_index()` delegates to `index.build_index`.
- `chunk.chunk_entry` currently creates one chunk per page using `title + content`.
- `embed.embed_texts` uses `sentence-transformers/all-MiniLM-L6-v2` and returns normalized 384-dimensional vectors.
- `index.build_index` embeds all chunks and saves:
  - `artifacts/index_vectors.npy`
  - `artifacts/index_meta.json`
- `index.load_index` reloads those two files at query time.
- `retrieve.search_batch` currently performs brute-force NumPy dot-product search and deduplicates page IDs.

## Baseline Attempt

Command attempted with the available bundled Python:

```powershell
& 'C:\Users\Diyar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\eval_public.py
```

Result: blocked before evaluation because `sentence_transformers` is not installed.

Exact failure:

```text
ModuleNotFoundError: No module named 'sentence_transformers'
```

Dependency check:

```text
numpy: installed
sentence_transformers: missing
faiss: missing
```

Also, `python` is not currently available on PATH in this shell, so local setup needs a normal Python environment or the commands must use an explicit Python executable.

## Phase 1 Conclusion

The starter is understood and the baseline is blocked only by local dependency/setup state, not by code inspection. Phase 2 can begin after dependency installation is approved, because the first working pipeline requires `sentence-transformers` and likely `faiss-cpu`.
