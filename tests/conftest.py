"""Global fixtures and utilities for testcraft test suite.

This module provides common fixtures used across all test modules.
Following the refactoring of test_catalog_validation.py, this includes
catalog-related fixtures and helper functions.
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from testcraft.config.model_catalog_loader import (
    load_catalog,
    ModelCatalog,
    CatalogEntry,
    LimitsModel,
    FlagsModel,
    PricingPerMillionModel,
    PricingModel,
    BetaModel,
    SourceModel,
)


# ================================================================================
# Catalog Fixtures
# ================================================================================

@pytest.fixture
def catalog_fixtures_path():
    """Return the path to catalog fixture files."""
    return Path(__file__).parent / "fixtures" / "catalogs"


@pytest.fixture
def valid_catalog_path(catalog_fixtures_path):
    """Return path to valid catalog fixture."""
    return catalog_fixtures_path / "valid.toml"


@pytest.fixture
def malformed_catalog_path(catalog_fixtures_path):
    """Return path to malformed catalog fixture."""
    return catalog_fixtures_path / "malformed.toml"


@pytest.fixture
def invalid_values_catalog_path(catalog_fixtures_path):
    """Return path to catalog with invalid values fixture."""
    return catalog_fixtures_path / "invalid_values.toml"


@pytest.fixture
def empty_catalog_path(catalog_fixtures_path):
    """Return path to empty catalog fixture."""
    return catalog_fixtures_path / "empty.toml"


@pytest.fixture
def missing_required_catalog_path(catalog_fixtures_path):
    """Return path to catalog with missing required fields fixture."""
    return catalog_fixtures_path / "missing_required.toml"


@pytest.fixture
def patched_catalog_path(tmp_path):
    """Factory fixture to patch catalog path with temporary TOML content.
    
    Usage:
        def test_something(patched_catalog_path):
            toml_content = '''
            version = "0.1"
            models = []
            '''
            with patched_catalog_path(toml_content):
                catalog = load_catalog()
                # ... test catalog
    """
    def _patch_catalog(content: str):
        temp_file = tmp_path / "test_catalog.toml"
        temp_file.write_text(content)
        
        return patch('testcraft.config.model_catalog_loader._catalog_path', 
                    return_value=temp_file)
    
    return _patch_catalog


@pytest.fixture
def catalog():
    """Return the real catalog for integration tests."""
    return load_catalog()


# ================================================================================
# Test Data Fixtures
# ================================================================================

@pytest.fixture
def pricing_cases():
    """Provide common pricing test cases."""
    return [
        {
            "name": "standard_openai",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "input_per_million": 0.15,
            "output_per_million": 0.60,
        },
        {
            "name": "anthropic_claude",
            "provider": "anthropic", 
            "model": "claude-3-7-sonnet",
            "input_per_million": 3.0,
            "output_per_million": 15.0,
        },
        {
            "name": "reasoning_model",
            "provider": "openai",
            "model": "o1-mini",
            "input_per_million": 3.0,
            "output_per_million": 12.0,
        },
    ]


@pytest.fixture
def provider_model_cases():
    """Provide common provider/model combinations for testing."""
    return [
        ("openai", "gpt-4o"),
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-sonnet"),
        ("anthropic", "claude-3-7-sonnet"),
    ]


@pytest.fixture
def token_usage_scenarios():
    """Provide common token usage scenarios for testing."""
    return {
        "minimal": {"prompt_tokens": 100, "completion_tokens": 50},
        "moderate": {"prompt_tokens": 1000, "completion_tokens": 500},
        "large": {"prompt_tokens": 5000, "completion_tokens": 2000},
        "extreme": {"prompt_tokens": 50000, "completion_tokens": 10000},
    }


# ================================================================================
# Helper Functions and Builders
# ================================================================================

def make_limits(max_context=128000, default_max_output=4096, max_thinking=0):
    """Helper to create LimitsModel with defaults."""
    return LimitsModel(
        max_context=max_context,
        default_max_output=default_max_output,
        max_thinking=max_thinking
    )


def make_catalog_entry(
    provider="test-provider",
    model_id="test-model",
    aliases=None,
    limits=None,
    flags=None,
    beta=None,
    pricing=None,
    source=None
):
    """Helper to create CatalogEntry with sensible defaults."""
    return CatalogEntry(
        provider=provider,
        model_id=model_id,
        aliases=aliases or [],
        limits=limits or make_limits(),
        flags=flags,
        beta=beta,
        pricing=pricing,
        source=source
    )


def write_temp_toml(tmp_path, content, filename="test_catalog.toml"):
    """Helper to write temporary TOML content and return path."""
    temp_file = tmp_path / filename
    temp_file.write_text(content)
    return temp_file


# ================================================================================
# Parametrization Helpers
# ================================================================================

# Invalid schema field combinations for parametrized testing
INVALID_SCHEMA_CASES = [
    pytest.param(
        {"max_context": -1000, "default_max_output": 4096},
        "negative_max_context",
        id="negative-max-context"
    ),
    pytest.param(
        {"max_context": 0, "default_max_output": 4096},
        "zero_max_context", 
        id="zero-max-context"
    ),
    pytest.param(
        {"max_context": 128000, "default_max_output": -4096},
        "negative_max_output",
        id="negative-max-output"
    ),
    pytest.param(
        {"max_context": "not_an_int", "default_max_output": 4096},
        "invalid_type_context",
        id="invalid-type-context"
    ),
    pytest.param(
        {"max_context": 128000, "default_max_output": "not_an_int"},
        "invalid_type_output",
        id="invalid-type-output"
    ),
]

# Missing required field combinations
MISSING_REQUIRED_CASES = [
    pytest.param(
        {"model_id": "test-model", "limits": make_limits()},
        "missing_provider",
        id="missing-provider"
    ),
    pytest.param(
        {"provider": "test-provider", "limits": make_limits()},
        "missing_model_id",
        id="missing-model-id"
    ),
    pytest.param(
        {"provider": "test-provider", "model_id": "test-model"},
        "missing_limits",
        id="missing-limits"
    ),
]

# Safety margin test cases
SAFETY_MARGIN_CASES = [
    pytest.param(0.5, id="50-percent-margin"),
    pytest.param(0.8, id="80-percent-margin"),
    pytest.param(0.9, id="90-percent-margin"),
    pytest.param(1.0, id="no-margin"),
]

# Use case multiplier test cases
USE_CASE_SCENARIOS = [
    pytest.param("test_generation", id="test-generation"),
    pytest.param("code_analysis", id="code-analysis"),
    pytest.param("refinement", id="refinement"),
]
