"""
Unit tests for BootstrapRunner service.

Tests the bootstrap functionality including conftest.py and PYTHONPATH strategies.
"""

from pathlib import Path
from unittest.mock import patch

from testcraft.application.generation.services.bootstrap_runner import (
    BootstrapRunner,
    BootstrapStrategy,
)
from testcraft.domain.models import ImportMap


class TestBootstrapRunner:
    """Test BootstrapRunner core functionality."""

    def test_bootstrap_runner_initialization(self):
        """Test BootstrapRunner initializes correctly."""
        runner = BootstrapRunner(prefer_conftest=True)
        assert runner._prefer_conftest is True

    def test_ensure_bootstrap_no_bootstrap_needed(self):
        """Test ensure_bootstrap returns NO_BOOTSTRAP when not needed."""
        runner = BootstrapRunner()
        import_map = ImportMap(
            target_import="from test import func",
            sys_path_roots=[],
            needs_bootstrap=False,
            bootstrap_conftest="",
        )
        tests_dir = Path("/tmp")

        strategy = runner.ensure_bootstrap(import_map, tests_dir)
        assert strategy == BootstrapStrategy.NO_BOOTSTRAP

    def test_set_pythonpath_env_single_root(self):
        """Test set_pythonpath_env with single root."""
        runner = BootstrapRunner()
        env_vars = runner.set_pythonpath_env(["/test/path"])

        assert "PYTHONPATH" in env_vars
        pythonpath = env_vars["PYTHONPATH"]
        assert "/test/path" in pythonpath

    def test_set_pythonpath_env_single_root_no_existing(self):
        """Test set_pythonpath_env with single root and no existing PYTHONPATH."""
        runner = BootstrapRunner()

        with patch.dict("os.environ", {}, clear=True):
            env_vars = runner.set_pythonpath_env(["/test/path"])

        assert "PYTHONPATH" in env_vars
        pythonpath = env_vars["PYTHONPATH"]
        assert pythonpath == "/test/path"

    def test_set_pythonpath_env_multiple_roots(self):
        """Test set_pythonpath_env with multiple roots."""
        runner = BootstrapRunner()
        env_vars = runner.set_pythonpath_env(["/path1", "/path2", "/path3"])

        assert "PYTHONPATH" in env_vars
        pythonpath = env_vars["PYTHONPATH"]
        assert "/path1" in pythonpath
        assert "/path2" in pythonpath
        assert "/path3" in pythonpath

    def test_set_pythonpath_env_removes_duplicates(self):
        """Test set_pythonpath_env removes duplicate paths."""
        runner = BootstrapRunner()
        env_vars = runner.set_pythonpath_env(["/path1", "/path2", "/path1", "/path2"])

        assert "PYTHONPATH" in env_vars
        pythonpath = env_vars["PYTHONPATH"]
        # Check that duplicates are removed
        paths = pythonpath.split(":")
        assert len([p for p in paths if p == "/path1"]) == 1
        assert len([p for p in paths if p == "/path2"]) == 1

    def test_set_pythonpath_env_filters_empty_strings(self):
        """Test set_pythonpath_env filters out empty strings."""
        runner = BootstrapRunner()
        env_vars = runner.set_pythonpath_env(["/path1", "", "/path2", "   ", "/path3"])

        assert "PYTHONPATH" in env_vars
        pythonpath = env_vars["PYTHONPATH"]
        assert "/path1" in pythonpath
        assert "/path2" in pythonpath
        assert "/path3" in pythonpath
