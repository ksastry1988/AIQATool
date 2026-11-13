import os

import pytest

from qatool import config


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Keep tests offline and independent of the developer's shell.

    Without this, a shell with QATOOL_EMBEDDING_PROVIDER=voyage and a real
    VOYAGE_API_KEY makes tests call the live API with that key.
    """
    for name in list(os.environ):
        if name.startswith("QATOOL_") or name in {"VOYAGE_API_KEY", "ANTHROPIC_API_KEY"}:
            monkeypatch.delenv(name)
    monkeypatch.setattr(config, "_file_values", {})
