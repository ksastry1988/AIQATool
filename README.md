# qatool

[![Coverage](https://github.com/ksastry1988/AIQATool/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/ksastry1988/AIQATool/actions/workflows/coverage.yml)

A codebase Q&A agent: index a repository and ask natural-language
questions about it ("where does rate limiting happen", "trace how a
request flows through auth"), answered by Claude grounded in
retrieved, language-aware code chunks.

## Status

Skeleton / work in progress. Core pieces are stubbed with `TODO`s:

- `qatool/chunking.py` — tree-sitter based, function/class-level chunking
- `qatool/embeddings.py` — embedding provider (Voyage AI recommended)
- `qatool/vectorstore.py` — vector DB wrapper (Chroma / LanceDB / Qdrant)
- `qatool/cli.py` — `index` and `ask` commands
- `qatool/reindex.py` — incremental re-indexing (implemented), used by
  the git hook below

## Setup

```bash
pip install -e .
```

## Usage (once implemented)

```bash
qatool index ./path-to-repo
qatool ask "where does authentication happen"
```

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
using the current commit's `git diff --name-status` output. If the local
index file `.qatool/index/store.json` does not exist yet, reindexing fails
with a clear message telling you to run `qatool index` first.

## Ignoring files

Copy `.qatoolignore.example` to `.qatoolignore` in the root of any
repo you index, and adjust the patterns.

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
