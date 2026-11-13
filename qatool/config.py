"""Centralized qatool configuration.

Precedence (highest first): CLI flags, environment variables, config file,
built-in defaults. The optional config file is JSON, located at
`$QATOOL_CONFIG`, or `<repo>/.qatool/config.json`, or `<repo>/qatool.json`.
Keys use the same names as the environment variables. Secrets (API keys)
are only accepted from the environment, never from the config file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TOP_K = 8
DEFAULT_LLM_MODEL = "claude-opus-5-5"
DEFAULT_MAX_CONTEXT_CHARS = 24000
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_INDEX_PATH = ".qatool/index"
CONFIG_ENV = "QATOOL_CONFIG"
CONFIG_FILENAMES = (".qatool/config.json", "qatool.json")

SECRET_KEYS = frozenset(
    {"QATOOL_EMBEDDING_API_KEY", "VOYAGE_API_KEY", "ANTHROPIC_API_KEY"}
)
FILE_KEYS = frozenset(
    {
        "QATOOL_MODEL",
        "QATOOL_TOP_K",
        "QATOOL_MAX_CONTEXT_CHARS",
        "QATOOL_MAX_OUTPUT_TOKENS",
        "QATOOL_INDEX_PATH",
        "QATOOL_EMBEDDING_PROVIDER",
        "QATOOL_EMBEDDING_MODEL",
        "QATOOL_EMBEDDING_ENDPOINT",
        "QATOOL_EMBEDDING_BATCH_SIZE",
        "QATOOL_EMBEDDING_MAX_RETRIES",
        "QATOOL_EMBEDDING_RETRY_BASE_DELAY_SECONDS",
        "QATOOL_EMBEDDING_TIMEOUT_SECONDS",
        "QATOOL_MOCK_EMBEDDING_DIMENSION",
    }
)

_file_values: dict[str, str] = {}


class ConfigError(ValueError):
    """Raised for invalid or unreadable configuration."""


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    llm_model: str
    top_k: int
    index_path: Path
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    config_file: Path | None = None


def lookup(name: str) -> str | None:
    """Return a setting: environment first, then the loaded config file."""
    value = os.environ.get(name)
    if value is not None:
        return value
    return _file_values.get(name)


def find_config_file(repo_root: Path) -> Path | None:
    explicit = os.environ.get(CONFIG_ENV)
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"{CONFIG_ENV} points to {path}, which is not a readable file")
        return path
    for name in CONFIG_FILENAMES:
        candidate = repo_root / name
        if candidate.is_file():
            return candidate
    return None


def read_config_file(path: Path) -> dict[str, str]:
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"config file {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a JSON object")
    values: dict[str, str] = {}
    for key, value in data.items():
        if key in SECRET_KEYS:
            raise ConfigError(
                f"config file {path}: {key} is a secret and must be set via the "
                "environment, not the config file"
            )
        if key not in FILE_KEYS:
            raise ConfigError(
                f"config file {path}: unknown setting {key!r}; "
                f"supported: {', '.join(sorted(FILE_KEYS))}"
            )
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ConfigError(f"config file {path}: {key} must be a string or number")
        values[key] = str(value)
    return values


def _parse_top_k(raw: str, source: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{source} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{source} must be a positive integer, got {value}")
    return value


def _parse_positive(name: str, default: int) -> int:
    raw = (lookup(name) or "").strip()
    return _parse_top_k(raw, name) if raw else default


def load_settings(repo_root: Path, top_k: int | None = None) -> Settings:
    """Load, validate and activate settings for a repository.

    Activating makes file-provided values visible to `lookup` so that index,
    reindex and ask resolve identical configuration.
    """
    global _file_values
    repo_root = Path(repo_root).resolve()
    config_file = find_config_file(repo_root)
    _file_values = read_config_file(config_file) if config_file else {}

    raw_top_k = lookup("QATOOL_TOP_K")
    if top_k is None:
        top_k = _parse_top_k(raw_top_k, "QATOOL_TOP_K") if raw_top_k else DEFAULT_TOP_K
    elif top_k <= 0:
        raise ConfigError(f"top_k must be positive, got {top_k}")

    max_context_chars = _parse_positive("QATOOL_MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS)
    max_output_tokens = _parse_positive("QATOOL_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)

    llm_model = (lookup("QATOOL_MODEL") or "").strip() or DEFAULT_LLM_MODEL

    raw_index = (lookup("QATOOL_INDEX_PATH") or "").strip() or DEFAULT_INDEX_PATH
    index_path = Path(raw_index).expanduser()
    if not index_path.is_absolute():
        index_path = repo_root / index_path
    index_path = index_path.resolve()
    if index_path.is_relative_to(repo_root) and not index_path.is_relative_to(repo_root / ".qatool"):
        raise ConfigError(
            f"QATOOL_INDEX_PATH {index_path} is inside the repository but outside "
            ".qatool, so its files would be indexed; use a path under .qatool or "
            "outside the repository"
        )

    # Fail early on invalid embedding settings rather than mid-index.
    from .embeddings import EmbeddingError, load_embedding_config

    try:
        embedding = load_embedding_config()
    except EmbeddingError as exc:
        raise ConfigError(str(exc)) from exc
    if embedding.provider_name not in ("mock", "voyage"):
        raise ConfigError(
            f"unsupported embedding provider {embedding.provider_name!r}; "
            "set QATOOL_EMBEDDING_PROVIDER to 'mock' or 'voyage'"
        )
    if embedding.provider_name == "voyage" and not embedding.api_key:
        raise ConfigError(
            "embedding provider 'voyage' requires an API key; set "
            "QATOOL_EMBEDDING_API_KEY or VOYAGE_API_KEY in the environment"
        )

    return Settings(
        repo_root=repo_root,
        llm_model=llm_model,
        top_k=top_k,
        index_path=index_path,
        max_context_chars=max_context_chars,
        max_output_tokens=max_output_tokens,
        config_file=config_file,
    )
