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

## Incremental re-indexing on commit

Enable the included git hook so the index stays fresh automatically:

```bash
git config core.hooksPath .githooks
```

This runs `.githooks/post-commit` after every commit/merge, which
diffs the changed files and re-embeds only those (see
`qatool/reindex.py`).

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
