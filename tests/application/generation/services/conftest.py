"""
Shared fixtures for generation services tests.

This module contains common fixtures used across generation service test modules.
"""

from unittest.mock import MagicMock

import pytest

from testcraft.application.generation.config import GenerationConfig


@pytest.fixture
def mock_telemetry_port():
    """Create mock telemetry port with span setup."""
    telemetry_port = MagicMock()
    
    # Setup telemetry mock with span context manager
    mock_span = MagicMock()
    telemetry_port.create_child_span.return_value.__enter__.return_value = mock_span
    
    return telemetry_port, mock_span


@pytest.fixture
def mock_context_port():
    """Create mock context port with default returns."""
    mock = MagicMock()
    mock.retrieve.return_value = {"results": [], "total_found": 0}
    mock.get_related_context.return_value = {"related_files": [], "relationships": []}
    return mock


@pytest.fixture
def mock_parser_port():
    """Create mock parser port with default returns."""
    mock = MagicMock()
    mock.parse_file.return_value = {"ast": None, "source_lines": []}
    return mock


@pytest.fixture
def mock_state_port():
    """Create mock state port."""
    return MagicMock()


@pytest.fixture
def mock_file_discovery_service():
    """Create mock file discovery service."""
    return MagicMock()


@pytest.fixture
def mock_coverage_port():
    """Create mock coverage port."""
    return MagicMock()


@pytest.fixture
def default_config():
    """Get default generation configuration."""
    return GenerationConfig.get_default_config()


@pytest.fixture
def custom_config():
    """Get customizable generation configuration."""
    config = GenerationConfig.get_default_config()
    return config
