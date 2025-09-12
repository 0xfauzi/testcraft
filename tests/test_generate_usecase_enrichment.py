"""
Tests for GenerateUseCase context enrichment capabilities.

This module tests the comprehensive detection features including environment/config
detection, client boundary detection, fixture discovery, and side-effect detection.
"""

import ast
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from testcraft.application.generate_usecase import GenerateUseCase
from testcraft.application.generation.services.enrichment_detectors import EnrichmentDetectors
from testcraft.config.models import TestCraftConfig


class TestGenerateUseCaseEnrichment:
    """Test cases for GenerateUseCase enrichment capabilities."""

    @pytest.fixture
    def mock_ports(self):
        """Create mock ports for testing."""
        llm_port = MagicMock()
        writer_port = MagicMock()
        coverage_port = MagicMock()
        refine_port = MagicMock()
        context_port = MagicMock()
        parser_port = MagicMock()
        state_port = MagicMock()
        telemetry_port = MagicMock()

        # Setup telemetry port mocks
        mock_span = MagicMock()
        mock_span.get_trace_context.return_value = MagicMock(trace_id="test_trace_123")
        telemetry_port.create_span.return_value.__enter__.return_value = mock_span
        telemetry_port.create_child_span.return_value.__enter__.return_value = mock_span

        return {
            "llm_port": llm_port,
            "writer_port": writer_port,
            "coverage_port": coverage_port,
            "refine_port": refine_port,
            "context_port": context_port,
            "parser_port": parser_port,
            "state_port": state_port,
            "telemetry_port": telemetry_port,
        }

    @pytest.fixture
    def generate_usecase(self, mock_ports):
        """Create GenerateUseCase instance with mocked ports."""
        return GenerateUseCase(**mock_ports)

    def test_detect_env_config_usage_basic_patterns(self, generate_usecase):
        """Test detection of basic environment and config patterns."""
        source_text = """
import os
from dotenv import load_dotenv

# Environment variables
DATABASE_URL = os.environ["DATABASE_URL"]
api_key = os.environ.get("API_KEY")
debug_mode = os.getenv("DEBUG_MODE")

# Config usage
db_host = config.get("database_host")
app_settings = settings.DATABASE_NAME
# env_value = settings.SECRET_KEY
"""

        ast_tree = ast.parse(source_text)
        result = EnrichmentDetectors.detect_env_config_usage(source_text, ast_tree)

        # Check environment variables
        assert "DATABASE_URL" in result["env_vars"]
        assert "API_KEY" in result["env_vars"]
        assert "DEBUG_MODE" in result["env_vars"]

        # Check config keys
        assert "database_host" in result["config_keys"]
        assert "DATABASE_NAME" in result["config_keys"]

    def test_detect_env_config_usage_with_caps(self, generate_usecase):
        """Test that environment detection respects max_vars limit."""
        source_text = """
import os
var1 = os.environ["VAR1"]
var2 = os.environ["VAR2"] 
var3 = os.environ["VAR3"]
var4 = os.environ["VAR4"]
"""

        ast_tree = ast.parse(source_text)
        result = EnrichmentDetectors.detect_env_config_usage(source_text, ast_tree, max_vars=2)

        # Should be limited to max_vars
        assert len(result["env_vars"]) == 2

    def test_detect_client_boundaries_database_clients(self, generate_usecase):
        """Test detection of database client patterns."""
        source_text = """
import sqlite3
import psycopg2
from sqlalchemy import create_engine
import redis

# Database connections
conn = sqlite3.connect("database.db")
pool = psycopg2.connect(host="localhost")
engine = create_engine("postgresql://user:pass@localhost/db")
r = redis.Redis(host="localhost")
"""

        ast_tree = ast.parse(source_text)
        result = EnrichmentDetectors.detect_client_boundaries(source_text, ast_tree)

        assert "sqlite3" in result["database_clients"]
        assert "psycopg2" in result["database_clients"]
        assert "sqlalchemy" in result["database_clients"]
        assert "redis" in result["database_clients"]

    def test_detect_client_boundaries_http_clients(self, generate_usecase):
        """Test detection of HTTP client patterns."""
        source_text = """
import requests
import httpx
import aiohttp
from urllib.request import urlopen

# HTTP clients
response = requests.get("https://api.example.com")
client = httpx.AsyncClient()
session = aiohttp.ClientSession()
data = urlopen("https://example.com")
"""

        ast_tree = ast.parse(source_text)
        result = EnrichmentDetectors.detect_client_boundaries(source_text, ast_tree)

        assert "requests" in result["http_clients"]
        assert "httpx" in result["http_clients"]
        assert "aiohttp" in result["http_clients"]
        assert "urllib" in result["http_clients"]

    def test_discover_comprehensive_fixtures_builtin(self, generate_usecase):
        """Test discovery of built-in pytest fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            result = EnrichmentDetectors.discover_comprehensive_fixtures(project_root)

            # Should include built-in fixtures
            builtin_fixtures = result["builtin"]
            assert "tmp_path" in builtin_fixtures
            # Check that we have some common fixtures (the exact ones depend on the limit)
            common_fixtures = ["monkeypatch", "caplog", "request", "config"]
            assert any(fixture in builtin_fixtures for fixture in common_fixtures)

    def test_discover_comprehensive_fixtures_custom(self, generate_usecase):
        """Test discovery of custom fixtures from conftest.py."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            tests_dir = project_root / "tests"
            tests_dir.mkdir()

            # Create conftest.py with custom fixtures
            conftest_content = '''
import pytest

@pytest.fixture(scope="session")
def database_session():
    """Database session fixture."""
    pass

@pytest.fixture
def sample_data():
    """Sample data fixture."""
    return {"key": "value"}

@pytest.fixture(scope="module") 
def api_client():
    """API client fixture."""
    pass
'''
            (tests_dir / "conftest.py").write_text(conftest_content)

            result = EnrichmentDetectors.discover_comprehensive_fixtures(project_root)

            custom_fixtures = result["custom"]
            assert "database_session" in custom_fixtures
            assert custom_fixtures["database_session"] == "session"
            assert "sample_data" in custom_fixtures
            assert custom_fixtures["sample_data"] == "function"  # default
            assert "api_client" in custom_fixtures
            assert custom_fixtures["api_client"] == "module"

    def test_discover_comprehensive_fixtures_third_party(self, generate_usecase):
        """Test detection of third-party fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            tests_dir = project_root / "tests"
            tests_dir.mkdir()

            # Create test file with third-party fixture imports
            test_content = """
import pytest
from pytest_django import django_db
from pytest_asyncio import event_loop
from pytest_httpx import httpx_mock

def test_with_django_db(django_db):
    pass

def test_with_event_loop(event_loop):
    pass
"""
            (tests_dir / "test_sample.py").write_text(test_content)

            result = EnrichmentDetectors.discover_comprehensive_fixtures(project_root)

            third_party = result["third_party"]
            assert "django_db" in third_party
            assert "event_loop" in third_party

    def test_discover_comprehensive_fixtures_with_caps(self, generate_usecase):
        """Test that fixture discovery respects max_fixtures limit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Test with low fixture limit
            result = EnrichmentDetectors.discover_comprehensive_fixtures(project_root, max_fixtures=3)

            # Each category should respect the limit (max_fixtures // 3)
            assert len(result["builtin"]) <= 1  # 3 // 3 = 1 per category
            assert len(result["custom"]) <= 1
            assert len(result["third_party"]) <= 1

    def test_detect_side_effect_boundaries_filesystem(self, generate_usecase):
        """Test detection of filesystem side effects."""
        source_text = """
import pathlib
import shutil
import glob

# Filesystem operations
with open("file.txt", "r") as f:
    content = f.read()

path = pathlib.Path("data")
shutil.copy("src.txt", "dst.txt")
files = glob.glob("*.py")
"""

        ast_tree = ast.parse(source_text)
        result = EnrichmentDetectors.detect_side_effect_boundaries(source_text, ast_tree)

        filesystem_effects = result.get("filesystem", [])
        assert any("open" in effect for effect in filesystem_effects)
        assert any("pathlib" in effect for effect in filesystem_effects)
        assert any("shutil" in effect for effect in filesystem_effects)
        assert any("glob" in effect for effect in filesystem_effects)

    def test_detect_side_effect_boundaries_time_and_random(self, generate_usecase):
        """Test detection of time and random side effects."""
        source_text = """
import time
import datetime
import random
import uuid

# Time operations
time.sleep(1)
now = datetime.datetime.now()

# Random operations  
value = random.choice([1, 2, 3])
identifier = uuid.uuid4()
"""

        ast_tree = ast.parse(source_text)
        result = EnrichmentDetectors.detect_side_effect_boundaries(source_text, ast_tree)

        time_effects = result.get("time", [])
        assert any("time.sleep" in effect for effect in time_effects)
        assert any("datetime" in effect for effect in time_effects)

        random_effects = result.get("random", [])
        assert any("random" in effect for effect in random_effects)
        assert any("uuid" in effect for effect in random_effects)

    def test_detect_side_effect_boundaries_process_and_network(self, generate_usecase):
        """Test detection of process and network side effects."""
        source_text = """
import subprocess
import os
import socket
import requests

# Process operations
result = subprocess.run(["ls", "-la"])
os.system("echo hello")

# Network operations
sock = socket.socket()
response = requests.get("http://example.com")
"""

        ast_tree = ast.parse(source_text)
        result = EnrichmentDetectors.detect_side_effect_boundaries(source_text, ast_tree)

        process_effects = result.get("process", [])
        assert any("subprocess" in effect for effect in process_effects)
        assert any("os.system" in effect for effect in process_effects)

        network_effects = result.get("network", [])
        assert any("socket" in effect for effect in network_effects)
        assert any("requests" in effect for effect in network_effects)

    def test_context_enrichment_config_integration(self, mock_ports):
        """Test that context enrichment config is properly integrated."""
        config_data = {
            "context_enrichment": {
                "enable_env_detection": False,
                "enable_db_boundary_detection": True,
                "enable_http_boundary_detection": False,
                "enable_comprehensive_fixtures": True,
                "enable_side_effect_detection": False,
                "max_env_vars": 10,
                "max_fixtures": 8,
            }
        }

        usecase = GenerateUseCase(config=config_data, **mock_ports)

        # Check that config was properly mapped to context categories
        context_categories = usecase._config["context_categories"]
        assert (
            context_categories["deps_config_fixtures"] is True
        )  # Any detection enabled
        assert context_categories["side_effects"] is False  # Specifically disabled

        # Check that enrichment config is stored
        enrichment_cfg = usecase._config["context_enrichment"]
        assert enrichment_cfg["enable_env_detection"] is False
        assert enrichment_cfg["max_env_vars"] == 10

    def test_error_handling_in_detection_methods(self, generate_usecase):
        """Test that detection methods handle errors gracefully."""
        # Test with invalid source text that could cause parsing errors
        invalid_source = "invalid python syntax {"

        try:
            ast_tree = ast.parse("pass")  # Valid fallback AST

            # These should not raise exceptions even with invalid input
            env_result = EnrichmentDetectors.detect_env_config_usage(
                invalid_source, ast_tree
            )
            client_result = EnrichmentDetectors.detect_client_boundaries(
                invalid_source, ast_tree
            )
            side_effect_result = EnrichmentDetectors.detect_side_effect_boundaries(
                invalid_source, ast_tree
            )

            # Results should be empty or have default structure
            assert isinstance(env_result, dict)
            assert isinstance(client_result, dict)
            assert isinstance(side_effect_result, dict)

        except Exception as e:
            pytest.fail(
                f"Detection methods should handle errors gracefully, but got: {e}"
            )

    def test_fixture_discovery_with_nonexistent_directory(self, generate_usecase):
        """Test fixture discovery with non-existent project directory."""
        non_existent_path = Path("/definitely/does/not/exist")

        result = EnrichmentDetectors.discover_comprehensive_fixtures(non_existent_path)

        # Should still return builtin fixtures even when project directory doesn't exist
        assert "builtin" in result
        assert len(result["builtin"]) > 0
        assert "custom" in result
        assert "third_party" in result


class TestContextEnrichmentConfig:
    """Test the ContextEnrichmentConfig model specifically."""

    def test_context_enrichment_config_defaults(self):
        """Test ContextEnrichmentConfig default values."""
        config = TestCraftConfig()
        enrichment = config.context_enrichment

        assert enrichment.enable_env_detection is True
        assert enrichment.enable_db_boundary_detection is True
        assert enrichment.enable_http_boundary_detection is True
        assert enrichment.enable_comprehensive_fixtures is True
        assert enrichment.enable_side_effect_detection is True
        assert enrichment.max_env_vars == 20
        assert enrichment.max_fixtures == 15

    def test_context_enrichment_config_validation(self):
        """Test ContextEnrichmentConfig field validation."""
        # Test valid config
        config_data = {
            "context_enrichment": {
                "enable_env_detection": False,
                "max_env_vars": 50,
                "max_fixtures": 30,
            }
        }
        config = TestCraftConfig(**config_data)

        assert config.context_enrichment.enable_env_detection is False
        assert config.context_enrichment.max_env_vars == 50
        assert config.context_enrichment.max_fixtures == 30

        # Test invalid max values
        with pytest.raises(Exception):  # ValidationError from pydantic
            TestCraftConfig(context_enrichment={"max_env_vars": 0})  # Below minimum

        with pytest.raises(Exception):  # ValidationError from pydantic
            TestCraftConfig(context_enrichment={"max_fixtures": 200})  # Above maximum
