"""
Tests for the File Discovery Service.

This module contains comprehensive tests for the FileDiscoveryService class,
testing file discovery patterns, exclusion rules, and edge cases.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from testcraft.adapters.io.file_discovery import (FileDiscoveryError,
                                                  FileDiscoveryService)
from testcraft.config.models import TestPatternConfig


class TestFileDiscoveryService:
    """Test suite for FileDiscoveryService."""

    @pytest.fixture
    def default_config(self):
        """Create a default test pattern configuration."""
        return TestPatternConfig()

    @pytest.fixture
    def custom_config(self):
        """Create a custom test pattern configuration."""
        return TestPatternConfig(
            test_patterns=["test_*.py", "*_test.py"],
            exclude=["migrations/*", "*/deprecated/*"],
            exclude_dirs=["venv", ".git", "__pycache__", "build"],
        )

    @pytest.fixture
    def minimal_config(self):
        """Create a minimal test pattern configuration for testing."""
        return TestPatternConfig(
            test_patterns=["test_*.py", "*_test.py", "tests/**/*.py"],
            exclude=[],  # No exclusions for testing
            exclude_dirs=["__pycache__", "venv"],  # Only essential exclusions
        )

    @pytest.fixture
    def file_discovery_service(self, minimal_config):
        """Create a FileDiscoveryService with minimal config for testing."""
        return FileDiscoveryService(minimal_config)

    @pytest.fixture
    def custom_file_discovery_service(self, custom_config):
        """Create a FileDiscoveryService with custom config."""
        return FileDiscoveryService(custom_config)

    @pytest.fixture
    def sample_project(self, tmp_path):
        """Create a sample project structure for testing."""
        project_path = tmp_path / "sample_project"
        project_path.mkdir()

        # Create source files
        (project_path / "module1.py").write_text("def function1(): pass")
        (project_path / "module2.py").write_text("class Class1: pass")
        (project_path / "utils.py").write_text("def helper(): pass")

        # Create test files
        (project_path / "test_module1.py").write_text("def test_function1(): pass")
        (project_path / "module2_test.py").write_text("def test_class1(): pass")

        # Create subdirectories
        subdir = project_path / "package"
        subdir.mkdir()
        (subdir / "submodule.py").write_text("def sub_function(): pass")
        (subdir / "test_submodule.py").write_text("def test_sub_function(): pass")

        # Create tests directory
        tests_dir = project_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_integration.py").write_text("def test_integration(): pass")

        # Create excluded directories
        excluded_dir = project_path / "__pycache__"
        excluded_dir.mkdir()
        (excluded_dir / "module1.pyc").write_text("compiled")

        venv_dir = project_path / "venv"
        venv_dir.mkdir()
        (venv_dir / "lib.py").write_text("venv file")

        # Create non-Python files
        (project_path / "README.md").write_text("# Project")
        (project_path / "config.json").write_text("{}")

        return project_path

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        service = FileDiscoveryService()

        assert service.config is not None
        assert isinstance(service.config, TestPatternConfig)
        assert "test_*.py" in service.config.test_patterns

    def test_init_with_custom_config(self, custom_config):
        """Test initialization with custom configuration."""
        service = FileDiscoveryService(custom_config)

        assert service.config == custom_config
        assert "venv" in service.exclude_dirs_set
        assert "migrations/*" in service.exclude_patterns_set

    def test_discover_source_files_basic(self, minimal_config, sample_project):
        """Test basic source file discovery."""
        # Use a fresh service instance with minimal config
        file_discovery_service = FileDiscoveryService(minimal_config)

        source_files = file_discovery_service.discover_source_files(sample_project)

        # Should find source files but not test files
        source_file_names = [Path(f).name for f in source_files]

        assert "module1.py" in source_file_names
        assert "module2.py" in source_file_names
        assert "utils.py" in source_file_names
        assert "submodule.py" in source_file_names

        # Should not include test files
        assert "test_module1.py" not in source_file_names
        assert "module2_test.py" not in source_file_names
        assert "test_submodule.py" not in source_file_names
        assert "test_integration.py" not in source_file_names

    def test_discover_source_files_include_tests(self, minimal_config, sample_project):
        """Test source file discovery including test files."""
        file_discovery_service = FileDiscoveryService(minimal_config)

        source_files = file_discovery_service.discover_source_files(
            sample_project, include_test_files=True
        )

        source_file_names = [Path(f).name for f in source_files]

        # Should include both source and test files
        assert "module1.py" in source_file_names
        assert "test_module1.py" in source_file_names
        assert "module2_test.py" in source_file_names
        assert "test_integration.py" in source_file_names

    def test_discover_test_files(self, minimal_config, sample_project):
        """Test test file discovery."""
        file_discovery_service = FileDiscoveryService(minimal_config)

        test_files = file_discovery_service.discover_test_files(sample_project)

        test_file_names = [Path(f).name for f in test_files]

        assert "test_module1.py" in test_file_names
        assert "module2_test.py" in test_file_names
        assert "test_submodule.py" in test_file_names
        assert "test_integration.py" in test_file_names

        # Should not include source files
        assert "module1.py" not in test_file_names
        assert "utils.py" not in test_file_names

    def test_discover_all_python_files_separate(self, minimal_config, sample_project):
        """Test discovering all Python files with separation."""
        file_discovery_service = FileDiscoveryService(minimal_config)

        result = file_discovery_service.discover_all_python_files(
            sample_project, separate_tests=True
        )

        assert "source_files" in result
        assert "test_files" in result
        assert "total_files" in result

        source_names = [Path(f).name for f in result["source_files"]]
        test_names = [Path(f).name for f in result["test_files"]]

        assert "module1.py" in source_names
        assert "test_module1.py" in test_names
        assert result["total_files"] == len(result["source_files"]) + len(
            result["test_files"]
        )

    def test_discover_all_python_files_combined(
        self, file_discovery_service, sample_project
    ):
        """Test discovering all Python files combined."""
        result = file_discovery_service.discover_all_python_files(
            sample_project, separate_tests=False
        )

        assert "all_files" in result
        assert "total_files" in result
        assert "source_files" not in result
        assert "test_files" not in result

        all_names = [Path(f).name for f in result["all_files"]]

        assert "module1.py" in all_names
        assert "test_module1.py" in all_names
        assert result["total_files"] == len(result["all_files"])

    def test_filter_existing_files(self, file_discovery_service, sample_project):
        """Test filtering existing files."""
        file_paths = [
            sample_project / "module1.py",
            sample_project / "nonexistent.py",
            sample_project / "README.md",  # Not Python
            sample_project / "__pycache__" / "module1.pyc",  # Excluded dir
        ]

        filtered_files = file_discovery_service.filter_existing_files(file_paths)
        filtered_names = [Path(f).name for f in filtered_files]

        assert "module1.py" in filtered_names
        assert "nonexistent.py" not in filtered_names
        assert "README.md" not in filtered_names
        assert "module1.pyc" not in filtered_names

    def test_custom_file_patterns(self, file_discovery_service, sample_project):
        """Test discovery with custom file patterns."""
        # Create some .pyx files
        (sample_project / "cython_module.pyx").write_text("# cython code")

        source_files = file_discovery_service.discover_source_files(
            sample_project, file_patterns=["*.py", "*.pyx"]
        )

        source_file_names = [Path(f).name for f in source_files]
        assert "module1.py" in source_file_names
        assert "cython_module.pyx" in source_file_names

    def test_custom_test_patterns(self, file_discovery_service, sample_project):
        """Test discovery with custom test patterns."""
        # Create files with different test patterns
        (sample_project / "spec_module.py").write_text("def spec_test(): pass")

        test_files = file_discovery_service.discover_test_files(
            sample_project, test_patterns=["test_*.py", "*_test.py", "spec_*.py"]
        )

        test_file_names = [Path(f).name for f in test_files]
        assert "test_module1.py" in test_file_names
        assert "module2_test.py" in test_file_names
        assert "spec_module.py" in test_file_names

    def test_exclusion_patterns(self, custom_file_discovery_service, tmp_path):
        """Test file exclusion patterns."""
        project_path = tmp_path / "test_project"
        project_path.mkdir()

        # Create files that should be excluded
        migrations_dir = project_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "001_initial.py").write_text("# migration")

        deprecated_dir = project_path / "old" / "deprecated"
        deprecated_dir.mkdir(parents=True)
        (deprecated_dir / "old_module.py").write_text("# deprecated")

        # Create normal files
        (project_path / "normal_module.py").write_text("# normal")

        source_files = custom_file_discovery_service.discover_source_files(project_path)
        source_file_names = [Path(f).name for f in source_files]

        assert "normal_module.py" in source_file_names
        assert "001_initial.py" not in source_file_names
        assert "old_module.py" not in source_file_names

    def test_exclusion_directories(self, custom_file_discovery_service, tmp_path):
        """Test directory exclusion."""
        project_path = tmp_path / "test_project"
        project_path.mkdir()

        # Create files in excluded directories
        venv_dir = project_path / "venv"
        venv_dir.mkdir()
        (venv_dir / "lib_module.py").write_text("# venv file")

        git_dir = project_path / ".git"
        git_dir.mkdir()
        (git_dir / "hook.py").write_text("# git hook")

        # Create normal file
        (project_path / "normal_module.py").write_text("# normal")

        source_files = custom_file_discovery_service.discover_source_files(project_path)
        source_file_names = [Path(f).name for f in source_files]

        assert "normal_module.py" in source_file_names
        assert "lib_module.py" not in source_file_names
        assert "hook.py" not in source_file_names

    def test_is_test_file(self, file_discovery_service):
        """Test test file identification."""
        assert file_discovery_service._is_test_file(Path("test_module.py"))
        assert file_discovery_service._is_test_file(Path("module_test.py"))
        assert file_discovery_service._is_test_file(
            Path("/project/tests/integration.py")
        )
        assert file_discovery_service._is_test_file(Path("/project/test/unit.py"))
        assert file_discovery_service._is_test_file(Path("conftest.py"))

        assert not file_discovery_service._is_test_file(Path("module.py"))
        assert not file_discovery_service._is_test_file(Path("utils.py"))

    def test_matches_pattern(self, file_discovery_service):
        """Test pattern matching."""
        assert file_discovery_service._matches_pattern("test_module.py", "test_*")
        assert file_discovery_service._matches_pattern("module_test.py", "*_test.py")
        assert file_discovery_service._matches_pattern(
            "migrations/001.py", "migrations/*"
        )
        assert file_discovery_service._matches_pattern(
            "path/to/deprecated/file.py", "deprecated"
        )

        assert not file_discovery_service._matches_pattern("module.py", "test_*")
        assert not file_discovery_service._matches_pattern(
            "test_module.py", "*_test.py"
        )

    def test_get_discovery_stats(self, minimal_config, sample_project):
        """Test getting discovery statistics."""
        file_discovery_service = FileDiscoveryService(minimal_config)

        stats = file_discovery_service.get_discovery_stats(sample_project)

        assert "project_path" in stats
        assert "source_files_count" in stats
        assert "test_files_count" in stats
        assert "total_files" in stats
        assert "config_patterns" in stats

        assert stats["source_files_count"] > 0
        assert stats["test_files_count"] > 0
        assert (
            stats["total_files"]
            == stats["source_files_count"] + stats["test_files_count"]
        )

    def test_nonexistent_project_path(self, file_discovery_service):
        """Test handling of nonexistent project path."""
        nonexistent_path = Path("/nonexistent/project")

        with pytest.raises(FileDiscoveryError) as exc_info:
            file_discovery_service.discover_source_files(nonexistent_path)

        assert "does not exist" in str(exc_info.value)

    def test_error_handling_in_discovery(self, file_discovery_service, tmp_path):
        """Test error handling during file discovery."""
        project_path = tmp_path / "test_project"
        project_path.mkdir()

        # Create a file that will cause issues during glob
        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.side_effect = Exception("Glob failed")

            # Should handle the error gracefully
            source_files = file_discovery_service.discover_source_files(project_path)
            assert isinstance(source_files, list)  # Should return empty list

    def test_error_handling_in_stats(self, file_discovery_service):
        """Test error handling in statistics generation."""
        nonexistent_path = Path("/nonexistent")

        stats = file_discovery_service.get_discovery_stats(nonexistent_path)

        assert "error" in stats
        assert "project_path" in stats

    def test_file_discovery_error_attributes(self):
        """Test FileDiscoveryError attributes."""
        original_error = ValueError("Original error")
        discovery_error = FileDiscoveryError("Discovery failed", cause=original_error)

        assert str(discovery_error) == "Discovery failed"
        assert discovery_error.cause == original_error

    def test_caching_behavior(self, file_discovery_service):
        """Test that exclude sets are cached."""
        # First access should create the cache
        exclude_dirs_1 = file_discovery_service.exclude_dirs_set
        exclude_patterns_1 = file_discovery_service.exclude_patterns_set

        # Second access should return the same objects (cached)
        exclude_dirs_2 = file_discovery_service.exclude_dirs_set
        exclude_patterns_2 = file_discovery_service.exclude_patterns_set

        assert exclude_dirs_1 is exclude_dirs_2
        assert exclude_patterns_1 is exclude_patterns_2

    def test_empty_project_discovery(self, file_discovery_service, tmp_path):
        """Test discovery in an empty project."""
        empty_project = tmp_path / "empty_project"
        empty_project.mkdir()

        source_files = file_discovery_service.discover_source_files(empty_project)
        test_files = file_discovery_service.discover_test_files(empty_project)

        assert source_files == []
        assert test_files == []

    def test_should_include_test_file_logic(self, custom_file_discovery_service):
        """Test test file inclusion logic."""
        # Test files in excluded directories should still be excluded
        assert not custom_file_discovery_service._should_include_test_file(
            Path("venv/test_module.py")
        )

        # Regular test files should be included
        assert custom_file_discovery_service._should_include_test_file(
            Path("test_module.py")
        )

        # Non-Python files should be excluded
        assert not custom_file_discovery_service._should_include_test_file(
            Path("test_module.txt")
        )

    def test_path_resolution(self, file_discovery_service, sample_project):
        """Test that file paths are properly resolved to absolute paths."""
        source_files = file_discovery_service.discover_source_files(sample_project)

        # All returned paths should be absolute
        for file_path in source_files:
            path = Path(file_path)
            assert path.is_absolute()
            assert path.exists()

    def test_duplicate_removal(self, file_discovery_service, tmp_path):
        """Test that duplicate files are removed from results."""
        project_path = tmp_path / "test_project"
        project_path.mkdir()

        # Create a file that might be found by multiple patterns
        test_file = project_path / "test_module.py"
        test_file.write_text("def test(): pass")

        # Use patterns that might find the same file
        source_files = file_discovery_service.discover_source_files(
            project_path,
            file_patterns=["*.py", "test_*.py"],  # Overlapping patterns
            include_test_files=True,
        )

        # Should not have duplicates
        assert len(source_files) == len(set(source_files))

        # The test file should appear only once
        test_file_matches = [f for f in source_files if "test_module.py" in f]
        assert len(test_file_matches) == 1
