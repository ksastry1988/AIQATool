# Q&A agent

[![Coverage](https://github.com/ksastry1988/AIQATool/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/ksastry1988/AIQATool/actions/workflows/coverage.yml)

A codebase Q&A agent: index a repository and ask natural-language
questions about it ("where does rate limiting happen", "trace how a
request flows through auth"), answered by Claude grounded in
retrieved, language-aware code chunks.

## Demo

- [Index](assets/QATool_Index.mov): running `qatool index` on a repository
  to chunk, embed and store it in the local index.
- [Ask](assets/QATool_Ask.mov): running `qatool ask` to answer a question
  about the indexed code, with `file:Lstart-Lend` source references.

## Quickstart

1. **Install** (Python 3.10+):

   ```bash
   git clone https://github.com/ksastry1988/AIQATool.git
   cd AIQATool
   pip install -e .
   qatool --help
   ```

   A Docker image is also available (see [Docker](#docker)).

2. **Set credentials** (see [Provider credentials](#provider-credentials)):

   ```bash
   export ANTHROPIC_API_KEY=...            # answers (required for `ask`)
   export QATOOL_EMBEDDING_PROVIDER=voyage # real embeddings
   export VOYAGE_API_KEY=...
   ```

3. **Index the repository** (full index, see [Full indexing](#full-indexing)):

   ```bash
   qatool index /path/to/repo
   echo ".qatool/" >> /path/to/repo/.gitignore   # keep the index out of commits
   ```

4. **Enable the post-commit hook** so the index stays fresh (see
   [Incremental re-indexing on commit](#incremental-re-indexing-on-commit)):

   ```bash
   cd /path/to/repo
   mkdir -p .githooks && cp /path/to/AIQATool/.githooks/post-commit .githooks/
   chmod +x .githooks/post-commit
   git config core.hooksPath .githooks
   ```

5. **Ask a question** (see [Asking questions](#asking-questions)):

   ```bash
   qatool ask --repo /path/to/repo "where does authentication happen"
   ```

## Provider credentials

qatool uses two providers. Credentials are read **only from environment
variables** (they are rejected in the config file and never printed).

| Purpose | Provider | Variables |
| --- | --- | --- |
| Answers (`ask`) | Anthropic Claude | `ANTHROPIC_API_KEY` (required); `QATOOL_MODEL` optional |
| Embeddings (`index`, `reindex`, `ask`) | `voyage` (recommended) | `QATOOL_EMBEDDING_PROVIDER=voyage` and `VOYAGE_API_KEY` (or `QATOOL_EMBEDDING_API_KEY`) |
| Embeddings (offline/testing) | `mock` (default) | none |

> **Important:** the default `mock` embedding provider produces
> deterministic placeholder vectors, not semantic ones, so retrieval quality
> is poor. Use `voyage` for real use. Use the same provider and model for
> `index`, `reindex`, and `ask`; if you change them, re-run `qatool index`.

## Full indexing

```bash
qatool index /path/to/repo
```

Walks the repository, chunks supported files, embeds them, and stores the
result in the [local index](#local-index-location). Files excluded by the
default rules or `.qatoolignore` are skipped (see
[Ignoring files](#ignoring-files)), as are binary files. Re-running `index`
is safe: unchanged files (by content hash) are skipped, changed files are
re-embedded, and files that were deleted or became ignored are removed. It
prints a summary of scanned, indexed, unchanged, and removed files, and
reports per-file errors. The path must be an existing directory.

## Asking questions

```bash
qatool ask --repo /path/to/repo "where does rate limiting happen"
qatool ask --repo /path/to/repo --top-k 12 "trace a request through auth"
```

`--repo` defaults to the current directory, which must already be indexed.
The command retrieves the most relevant chunks (up to eight by default; use
`--top-k` or `QATOOL_TOP_K`), sends the question and context to Claude, and
prints an answer with source citations (file and line range). It requires
`ANTHROPIC_API_KEY`. Set `QATOOL_MODEL` to override the default Claude model.
Prompt context is bounded by `QATOOL_MAX_CONTEXT_CHARS` (default 24000);
lower-ranked chunks are dropped or truncated to fit. `QATOOL_MAX_OUTPUT_TOKENS`
(default 8192) caps the answer length. The model's reasoning counts toward this
limit, so values much below a few thousand can cut answers off.

## Local index location

The index is stored locally in `<repo>/.qatool/index/store.json` (override
with `QATOOL_INDEX_PATH`, which must be under `.qatool` or outside the repo).
It is created by `qatool index`, never uploaded by qatool itself (chunk text
is sent to the embedding provider while indexing, and retrieved chunks are
sent to Claude when asking). Add `.qatool/` to your repository's
`.gitignore`; otherwise the index files get committed and the hook reindexes them. To start over, delete the `.qatool/index` directory and run
`qatool index` again.

## Supported languages and limitations

Function/class-level chunking (tree-sitter) is supported for these
extensions:

| Language | Extensions |
| --- | --- |
| Python | `.py`, `.pyi` |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` |
| TypeScript | `.ts`, `.mts`, `.cts`, `.tsx` |
| Go | `.go` |
| Java | `.java` |
| Rust | `.rs` |
| C / C++ | `.c`, `.h`, `.cc`, `.cpp`, `.cxx`, `.hh`, `.hpp`, `.hxx` |

Limitations:

- Other text files (Markdown, YAML, JSON, other languages, ...) are indexed
  as a single whole-file chunk, which may be less precise for large files.
- Large definitions are split into chunks of at most 80 lines.
- Binary files, secret-like files, and generated files are never indexed.
- The index is local to one checkout; there is no shared or remote index.
- Answers are only as good as the retrieved chunks and depend on Claude;
  verify cited sources for important decisions.
- The default `mock` embedding provider is not semantic (see
  [Provider credentials](#provider-credentials)).
- `reindex` fails if no index exists yet; run `qatool index` first.

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
| `QATOOL_MODEL` | `claude-opus-5-5` | Claude model for `ask` |
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
stays fresh automatically. Inside this repository the hook is already at
`.githooks/post-commit`; for another repository, copy it there first:

```bash
qatool index /path/to/repo
cd /path/to/repo
mkdir -p .githooks && cp /path/to/AIQATool/.githooks/post-commit .githooks/
chmod +x .githooks/post-commit
git config core.hooksPath .githooks
```

The `qatool` command must be on `PATH` when git runs the hook (for example
install it in the environment your git client uses). The hook is
per-clone: each collaborator runs `git config core.hooksPath .githooks`.
You can also reindex manually with
`qatool reindex --repo /path/to/repo --changed-files <file>`, where the file
holds NUL-separated `git diff --name-status -z` output.

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
