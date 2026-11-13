# Q&A agent

[![Coverage](https://github.com/ksastry1988/AIQATool/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/ksastry1988/AIQATool/actions/workflows/coverage.yml)

A codebase Q&A agent: index a repository and ask natural-language
questions about it ("where does rate limiting happen", "trace how a
request flows through auth"), answered by Claude grounded in
retrieved, language-aware code chunks.

## Status

Skeleton / work in progress. Some core pieces are still stubbed with `TODO`s:

- `qatool/chunking.py` — tree-sitter based, function/class-level chunking
  (implemented for Python, JavaScript, TypeScript, Go, Java, Rust, C/C++;
  other files are indexed as whole-file chunks)
- `qatool/embeddings.py` — embedding provider (Voyage AI recommended)
- `qatool/vectorstore.py` — implemented persistent local vector store
- `qatool/cli.py` — `index`, `ask`, and `reindex` commands
- `qatool/reindex.py` — incremental re-indexing (implemented), used by
  the git hook below

## Setup

```bash
pip install -e .
```

## Usage

```bash
qatool index ./path-to-repo
qatool ask --repo ./path-to-repo "where does authentication happen"
```

The ask command retrieves up to eight relevant chunks by default. Use
`--top-k` to change the number of retrieved chunks. It sends the question and
retrieved context to Claude, returns an answer with source citations, and
requires `ANTHROPIC_API_KEY`. Set `QATOOL_MODEL` to override the default Claude
model.
Prompt context is bounded by `QATOOL_MAX_CONTEXT_CHARS` (default 24000);
lower-ranked chunks are dropped or truncated to fit. `QATOOL_MAX_OUTPUT_TOKENS`
(default 1024) caps the answer length.

## Configuration

`index`, `reindex`, and `ask` all resolve settings through `qatool/config.py`,
so they behave consistently. Precedence: CLI flags > environment variables >
config file > defaults. Invalid settings fail fast with exit code 2 and a
message naming the setting; secret values are never printed.

The optional config file is JSON, read from `$QATOOL_CONFIG`, else
`<repo>/.qatool/config.json`, else `<repo>/qatool.json`. Keys are the
variable names below (API keys are rejected in the file; use the environment).

```json
{"QATOOL_TOP_K": 5, "QATOOL_INDEX_PATH": ".qatool/index", "QATOOL_EMBEDDING_PROVIDER": "mock"}
```

| Setting | Default | Description |
| --- | --- | --- |
| `QATOOL_CONFIG` | unset | Path to a config file (environment only) |
| `QATOOL_MODEL` | `claude-3-5-sonnet-latest` | Claude model for `ask` |
| `QATOOL_TOP_K` | `8` | Chunks retrieved by `ask` (`--top-k` overrides) |
| `QATOOL_INDEX_PATH` | `.qatool/index` | Vector store directory; relative paths are under the repo. Must be under `.qatool` or outside the repo (otherwise rejected) |
| `QATOOL_EMBEDDING_PROVIDER` | `mock` | `mock` or `voyage` |
| `QATOOL_EMBEDDING_MODEL` | `voyage-3-large` | Embedding model |
| `QATOOL_EMBEDDING_ENDPOINT` | Voyage endpoint | Embedding API URL |
| `QATOOL_EMBEDDING_BATCH_SIZE` | `128` | Texts per embedding request |
| `QATOOL_EMBEDDING_MAX_RETRIES` | `2` | Retries for failed requests |
| `QATOOL_EMBEDDING_RETRY_BASE_DELAY_SECONDS` | `0.5` | Retry backoff base |
| `QATOOL_EMBEDDING_TIMEOUT_SECONDS` | `30` | Request timeout |
| `QATOOL_MOCK_EMBEDDING_DIMENSION` | `32` | Mock vector size |
| `ANTHROPIC_API_KEY` | required for `ask` | Secret; environment only |
| `QATOOL_EMBEDDING_API_KEY` / `VOYAGE_API_KEY` | required for `voyage` | Secret; environment only |

## Docker

Build the image from the repository root:

```bash
docker build -t qatool .
```

The image runs `qatool --help` by default. Pass a qatool command after the
image name, and mount the repository being indexed when needed:

```bash
docker run --rm qatool --help
docker run --rm -v "$PWD:/workspace" qatool index /workspace
```

The image contains the package and its runtime dependencies. Persistent
index data should be stored in the mounted repository's `.qatool` directory.

## Incremental re-indexing on commit

Initialize the local index first, then enable the included git hook so it
stays fresh automatically:

```bash
qatool index /path/to/repo
git config core.hooksPath .githooks
```

This runs `.githooks/post-commit` after every commit/merge, which
diffs the changed files and re-embeds only those (see
`qatool/reindex.py`).

The hook invokes `qatool reindex --repo <repo> --changed-files <temp-file>`
using the current commit's `git diff --name-status` output. For merge
commits, it diffs the merge result against the first parent so only newly
introduced merge changes are reindexed. If the local index file
`.qatool/index/store.json` does not exist yet, reindexing fails with a
clear message telling you to run `qatool index` first.

## Ignoring files

Copy `.qatoolignore.example` to `.qatoolignore` in the root of any
repo you index, and adjust the patterns.

Syntax (one pattern per line, `#` starts a comment, paths relative to repo root):

- `name` matches a file or directory with that name at any depth
- `dir/` matches a directory and everything beneath it
- `*.log`, `docs/*.md` are glob patterns matched against the relative path

Secret-like files (`.env*` except `*.example`/`*.sample`/`*.template`, `*.pem`,
`*.key`, `credentials.json`, `*.tfstate`, `*.tfvars`, ...) and generated files
(lockfiles, `*.min.js`, `dist/`, `build/`, ...) are excluded by default. The
same rules apply to `qatool index` and `qatool reindex`; a file that becomes
ignored is removed from the index on its next reindex.

## Tests

```bash
pip install -e ".[test]"
pytest
```

## Code coverage

Coverage is configured in `.coveragerc`. Install the coverage tool and run the
same test suite with branch coverage enabled:

```bash
pip install coverage
coverage run -m pytest
coverage report
coverage html  # Open coverage_html/index.html for the detailed report
```

The commands also produce `coverage.xml` for CI or coverage-reporting services.
