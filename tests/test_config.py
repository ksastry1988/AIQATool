import json

import pytest

from qatool import config
from qatool.config import ConfigError, load_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in list(config.FILE_KEYS) + list(config.SECRET_KEYS) + [config.CONFIG_ENV]:
        monkeypatch.delenv(name, raising=False)
    yield
    config._file_values = {}


def test_defaults(tmp_path):
    s = load_settings(tmp_path)
    assert s.top_k == 8
    assert s.index_path == tmp_path.resolve() / ".qatool" / "index"
    assert s.llm_model == config.DEFAULT_LLM_MODEL


def test_file_then_env_then_cli_precedence(tmp_path, monkeypatch):
    (tmp_path / "qatool.json").write_text(
        json.dumps({"QATOOL_TOP_K": 3, "QATOOL_MODEL": "m1", "QATOOL_INDEX_PATH": ".qatool/idx"})
    )
    s = load_settings(tmp_path)
    assert (s.top_k, s.llm_model) == (3, "m1")
    monkeypatch.setenv("QATOOL_TOP_K", "5")
    assert load_settings(tmp_path).top_k == 5
    assert load_settings(tmp_path, top_k=2).top_k == 2


def test_file_values_reach_embedding_config(tmp_path):
    from qatool.embeddings import load_embedding_config

    (tmp_path / "qatool.json").write_text(json.dumps({"QATOOL_MOCK_EMBEDDING_DIMENSION": 7}))
    load_settings(tmp_path)
    assert load_embedding_config().mock_dimension == 7


@pytest.mark.parametrize(
    "content,match",
    [
        ("{", "not valid JSON"),
        ("[]", "JSON object"),
        ('{"BOGUS": 1}', "unknown setting"),
        ('{"VOYAGE_API_KEY": "sk-secret"}', "environment"),
    ],
)
def test_invalid_file_errors_do_not_leak_values(tmp_path, content, match):
    (tmp_path / "qatool.json").write_text(content)
    with pytest.raises(ConfigError, match=match) as exc:
        load_settings(tmp_path)
    assert "sk-secret" not in str(exc.value)


def test_invalid_values(tmp_path, monkeypatch):
    monkeypatch.setenv("QATOOL_TOP_K", "abc")
    with pytest.raises(ConfigError, match="QATOOL_TOP_K"):
        load_settings(tmp_path)
    monkeypatch.setenv("QATOOL_TOP_K", "0")
    with pytest.raises(ConfigError, match="positive"):
        load_settings(tmp_path)


def test_voyage_requires_key_and_unknown_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("QATOOL_EMBEDDING_PROVIDER", "voyage")
    with pytest.raises(ConfigError, match="API key"):
        load_settings(tmp_path)
    monkeypatch.setenv("QATOOL_EMBEDDING_PROVIDER", "nope")
    with pytest.raises(ConfigError, match="unsupported embedding provider"):
        load_settings(tmp_path)


def test_index_path_inside_repo_must_be_under_dot_qatool(tmp_path, monkeypatch):
    monkeypatch.setenv("QATOOL_INDEX_PATH", "idx")
    with pytest.raises(ConfigError, match="outside .qatool"):
        load_settings(tmp_path)
    monkeypatch.setenv("QATOOL_INDEX_PATH", str(tmp_path.parent / "elsewhere"))
    assert load_settings(tmp_path).index_path == (tmp_path.parent / "elsewhere").resolve()
