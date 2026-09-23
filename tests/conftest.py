import pytest


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Every test gets its own .cache/ so tests never touch the real one."""
    monkeypatch.setattr("llm.CACHE_DIR", tmp_path / ".cache")
    yield
