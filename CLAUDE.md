# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`qatool` is a codebase Q&A CLI: it indexes a repository into a local vector store and answers natural-language questions with Claude, grounded in retrieved code chunks. Python ≥3.10, packaged via `pyproject.toml` (setuptools), entry point `qatool = qatool.cli:main`. The only runtime dependency is `tree-sitter`. HTTP calls to Voyage and Anthropic go through `urllib` directly, not through vendor SDKs.

## Commands

```bash
pip install -e ".[test]"                      # install with test deps (pytest, coverage)
pytest                                        # run all tests
pytest tests/test_qa.py                       # single file
pytest tests/test_qa.py::test_claude_client_requires_api_key   # single test

# Coverage, same as CI (.github/workflows/coverage.yml, runs on PRs to main)
coverage run -m pytest && coverage combine && coverage report --fail-under=70

qatool index <repo>                           # full index -> <repo>/.qatool/index/store.json
qatool ask --repo <repo> [--top-k N] "question"
qatool reindex --repo <repo> --changed-files <git-diff-name-status-file>
docker build -t qatool .                      # image ENTRYPOINT is `qatool`
```

`.coveragerc` sets `parallel = True`, so `coverage combine` is required before `coverage report`. CI fails if coverage is below 70%.

## Architecture

Pipeline: **chunking → embeddings → vectorstore**. `indexing` (full) and `reindex` (incremental) drive it, and `qa` reads from it.

- `cli.py` handles argparse subcommands and imports modules lazily inside each `cmd_*`. `cmd_reindex` rewrites `sys.argv` and calls `reindex.main()`, so reindex's own argparse is the real interface. User-facing errors print as `[qatool] error: ...` and exit with `SystemExit(2)`.
- `indexing.py` contains `should_index()`, the single source of file-filtering rules: ignored dirs and suffixes, secret files (`.env`, `*.pem`, keys…), a binary-content heuristic, and `.qatoolignore` patterns. `reindex.py` reuses it. `index_repository()` skips files whose sha256 matches the stored hash, embeds in batches of 32 files, falls back to per-file embedding when a batch fails, and removes stale files, including files that disappear during the run.
- `reindex.py` parses `git diff --name-status` output (A/M/D/R) and deletes old paths for deletes and renames. It re-embeds only files whose hash changed. It requires an existing index and never creates one. The `.githooks/post-commit` hook calls it; enable the hook with `git config core.hooksPath .githooks`. For merge commits, the hook diffs against the first parent.
- `chunking.py` defines the `Chunk(text, metadata)` type. Metadata keys are `file`, `start_line`, `end_line`, `symbol`, `language`. The code tries tree-sitter, but `get_language()` loads `build/languages/tree-sitter-<lang>/language.so` through the old `Language(path, name)` API. Those builds are not in the repo, so in practice every file falls back to a **single whole-file chunk**.
- `embeddings.py` selects a provider through `QATOOL_EMBEDDING_PROVIDER`: the default is **`mock`**, which is deterministic and offline, and `voyage` is the other option. Related env vars: `QATOOL_EMBEDDING_MODEL`, `QATOOL_EMBEDDING_ENDPOINT`, `QATOOL_EMBEDDING_API_KEY`/`VOYAGE_API_KEY`, `QATOOL_EMBEDDING_BATCH_SIZE`, `QATOOL_EMBEDDING_MAX_RETRIES`, `QATOOL_MOCK_EMBEDDING_DIMENSION`. `embed_texts()` batches requests and retries rate-limit and transient errors.
- `vectorstore.py` implements `VectorStore`, a JSON-file store (`store.json`) holding `file_hashes` and `chunks` (with vectors). Writes are atomic (temp file + replace) and guarded by a cross-platform file lock (`fcntl`/`msvcrt`). Retrieval is brute-force cosine similarity. The store raises on vector dimension mismatch, so queries must use the same embedding provider and dimension as indexing.
- `qa.py`: `ask_question()` embeds the question, queries the store, formats the retrieved chunks with `file:Lstart-Lend` references, and calls `ClaudeModelClient` (Anthropic Messages API, needs `ANTHROPIC_API_KEY`, model override `QATOOL_MODEL`). It then appends a deduplicated `Sources:` list. The `ModelClient` protocol lets tests inject a fake client.

## Testing conventions

Tests never hit the network. They use `monkeypatch` to replace module-level names where they are used, for example `qa.embed_texts`, `qa.VectorStore`, and `qa.urllib.request.urlopen`, or they inject fakes such as `model_client=`. Stores are created under `tmp_path`.
