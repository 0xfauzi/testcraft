import pytest

pytest.importorskip("pydantic")

from testcraft.config.loader import ConfigLoader


def test_load_config_ignores_helper_environment(monkeypatch):
    """Ensure helper-only TESTCRAFT_E2E_* variables are ignored."""
    monkeypatch.setenv("TESTCRAFT_E2E_PROVIDER", "fake-provider")

    loader = ConfigLoader()
    config = loader.load_config(reload=True)

    # Default config should load without e2e-specific overrides leaking in
    assert config.llm.default_provider == "openai"
    assert "e2e_provider" not in config.model_dump()
