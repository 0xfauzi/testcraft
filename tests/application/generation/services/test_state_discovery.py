"""
Tests for state synchronization and discovery services.

This module contains unit tests for state sync discovery and directory tree builder services.
"""

import tempfile
from pathlib import Path

import pytest

from testcraft.application.generation.services.state_discovery import StateSyncDiscovery
from testcraft.application.generation.services.structure import DirectoryTreeBuilder


class TestStateSyncDiscovery:
    """Test cases for StateSyncDiscovery service."""

    @pytest.fixture
    def service(self, mock_state_port, mock_file_discovery_service, mock_telemetry_port):
        """Create StateSyncDiscovery service."""
        state_port = mock_state_port
        file_discovery_service = mock_file_discovery_service
        telemetry_port, _ = mock_telemetry_port
        return StateSyncDiscovery(state_port, file_discovery_service, telemetry_port)

    def test_sync_and_discover_with_target_files(
        self, service, mock_state_port, mock_file_discovery_service, mock_telemetry_port
    ):
        """Test sync and discover with target files."""
        state_port = mock_state_port
        file_discovery_service = mock_file_discovery_service
        telemetry_port, mock_span = mock_telemetry_port

        # Setup mocks
        state_port.get_all_state.return_value = {"previous": "state"}
        file_discovery_service.filter_existing_files.return_value = [
            "file1.py",
            "file2.py",
        ]

        project_path = Path("/test/project")
        target_files = ["file1.py", "file2.py", "nonexistent.py"]

        result = service.sync_and_discover(project_path, target_files)

        # Verify calls
        state_port.get_all_state.assert_called_once_with("generation")
        file_discovery_service.filter_existing_files.assert_called_once()

        # Verify result
        assert len(result["files"]) == 2
        assert result["files"][0] == Path("file1.py")
        assert result["files"][1] == Path("file2.py")
        assert result["previous_state"] == {"previous": "state"}
        assert result["project_path"] == project_path

        # Verify telemetry
        mock_span.set_attribute.assert_called()

    def test_sync_and_discover_without_target_files(
        self, service, mock_state_port, mock_file_discovery_service, mock_telemetry_port
    ):
        """Test sync and discover without target files (discovery mode)."""
        state_port = mock_state_port
        file_discovery_service = mock_file_discovery_service
        telemetry_port, mock_span = mock_telemetry_port

        # Setup mocks
        state_port.get_all_state.return_value = {}
        file_discovery_service.discover_source_files.return_value = [
            "src/main.py",
            "src/utils.py",
        ]

        project_path = Path("/test/project")

        result = service.sync_and_discover(project_path, None)

        # Verify calls
        state_port.get_all_state.assert_called_once_with("generation")
        file_discovery_service.discover_source_files.assert_called_once_with(
            project_path, include_test_files=False
        )

        # Verify result
        assert len(result["files"]) == 2
        assert result["files"][0] == Path("src/main.py")
        assert result["files"][1] == Path("src/utils.py")
        assert result["previous_state"] == {}


class TestDirectoryTreeBuilder:
    """Test cases for DirectoryTreeBuilder service."""

    def test_build_tree(self):
        """Test directory tree building."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some files and directories
            (temp_path / "main.py").write_text("# main file")
            (temp_path / "utils.py").write_text("# utils file")
            (temp_path / "subdir").mkdir()
            (temp_path / "subdir" / "helper.py").write_text("# helper file")
            (temp_path / ".hidden").mkdir()  # Should be ignored

            tree = DirectoryTreeBuilder.build_tree(temp_path)

            assert tree["name"] == temp_path.name
            assert tree["type"] == "directory"
            assert "children" in tree

            # Find Python files in children
            py_files = [child for child in tree["children"] if child["type"] == "file"]
            directories = [
                child for child in tree["children"] if child["type"] == "directory"
            ]

            # Should have main.py and utils.py
            py_file_names = {f["name"] for f in py_files}
            assert "main.py" in py_file_names
            assert "utils.py" in py_file_names

            # Should have subdir but not .hidden
            dir_names = {d["name"] for d in directories}
            assert "subdir" in dir_names
            assert ".hidden" not in dir_names

    def test_build_tree_error_handling(self):
        """Test tree building with error handling."""
        nonexistent = Path("/definitely/does/not/exist")
        tree = DirectoryTreeBuilder.build_tree(nonexistent)

        # Should return empty dict on error
        assert tree == {}
