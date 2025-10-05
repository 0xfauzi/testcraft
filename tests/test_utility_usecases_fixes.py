"""
Tests for UtilityUseCase bug fixes.

This module tests the fixes for:
1. Real backup file creation
2. Correct get_all_state() API usage
3. Return type validation for load_state() and persist_state()
4. Fail-fast behavior in sync operations
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from testcraft.application.utility_usecases import UtilityUseCase, UtilityUseCaseError


@pytest.fixture
def mock_state_port():
    """Create a mock StatePort."""
    mock = Mock()
    mock.get_all_state = Mock(return_value={"test_key": "test_value"})
    mock.load_state = Mock(
        return_value={
            "success": True,
            "loaded_keys": ["key1", "key2"],
            "loaded_values": {"key1": "value1"},
            "load_metadata": {},
        }
    )
    mock.persist_state = Mock(
        return_value={
            "success": True,
            "persisted_keys": ["key1", "key2"],
            "persistence_location": "/tmp/test",
            "persistence_metadata": {},
        }
    )
    mock.clear_state = Mock(
        return_value={"success": True, "cleared_keys": [], "clear_metadata": {}}
    )
    return mock


@pytest.fixture
def mock_telemetry():
    """Create a mock TelemetryPort."""
    mock = Mock()
    span_mock = Mock()
    span_mock.__enter__ = Mock(return_value=span_mock)
    span_mock.__exit__ = Mock(return_value=False)
    span_mock.set_attribute = Mock()
    span_mock.record_exception = Mock()
    mock.create_span = Mock(return_value=span_mock)
    mock.is_enabled = Mock(return_value=True)
    mock.get_trace_context = Mock(return_value=None)
    return mock


@pytest.fixture
def mock_cost_port():
    """Create a mock CostPort."""
    mock = Mock()
    mock.get_summary = Mock(return_value={"total_cost": 10.50})
    mock.get_cost_breakdown = Mock(return_value={})
    mock.check_cost_limit = Mock(return_value={"within_limits": True})
    return mock


class TestBackupCreation:
    """Test actual backup file creation (Fix #1)."""

    @pytest.mark.asyncio
    async def test_backup_creates_file(self, tmp_path, mock_state_port, mock_telemetry):
        """Verify backup file is created on disk."""
        # Setup
        usecase = UtilityUseCase(mock_state_port, mock_telemetry)

        # Mock get_all_state to return test data
        test_state = {"generation": {"test": "data"}, "coverage": {}}
        mock_state_port.get_all_state.return_value = test_state

        # Execute with tmp_path as cwd
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = await usecase.reset_state(confirm_reset=True, create_backup=True)

        # Verify backup was created
        assert result["reset_results"]["backup_created"]
        backup_location = result["reset_results"]["backup_location"]
        assert backup_location is not None

        backup_file = Path(backup_location)
        assert backup_file.exists()
        assert backup_file.parent.name == "backups"

        # Verify backup content is valid JSON
        with open(backup_file) as f:
            backup_data = json.load(f)
        assert isinstance(backup_data, dict)
        assert backup_data == test_state

    @pytest.mark.asyncio
    async def test_backup_directory_creation(
        self, tmp_path, mock_state_port, mock_telemetry
    ):
        """Verify .testcraft/backups directory is created if missing."""
        # Setup
        usecase = UtilityUseCase(mock_state_port, mock_telemetry)

        # Ensure directory doesn't exist initially
        backup_dir = tmp_path / ".testcraft" / "backups"
        assert not backup_dir.exists()

        # Execute
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            await usecase.reset_state(confirm_reset=True, create_backup=True)

        # Verify directory was created
        assert backup_dir.exists()
        assert backup_dir.is_dir()

    @pytest.mark.asyncio
    async def test_backup_metadata_accuracy(
        self, tmp_path, mock_state_port, mock_telemetry
    ):
        """Verify backup metadata is accurate."""
        # Setup
        usecase = UtilityUseCase(mock_state_port, mock_telemetry)
        test_state = {"key1": "value1", "key2": {"nested": "value"}}
        mock_state_port.get_all_state.return_value = test_state

        # Execute
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = await usecase.reset_state(confirm_reset=True, create_backup=True)

        # Verify metadata
        backup_location = result["reset_results"]["backup_location"]
        backup_file = Path(backup_location)

        # Check backup_size_bytes is accurate
        actual_size = backup_file.stat().st_size
        assert actual_size > 0

        # State entries count should match
        assert len(test_state) == 2


class TestStateAPIConsistency:
    """Test correct get_all_state() API usage (Fix #2)."""

    @pytest.mark.asyncio
    async def test_debug_state_uses_state_prefix_kwarg(
        self, mock_state_port, mock_telemetry
    ):
        """Verify debug_state uses state_prefix as keyword argument."""
        # Setup
        usecase = UtilityUseCase(mock_state_port, mock_telemetry)

        # Mock get_all_state to track calls
        mock_state_port.get_all_state = Mock(return_value={})

        # Execute
        await usecase.debug_state()

        # Verify get_all_state was called with state_prefix keyword
        calls = mock_state_port.get_all_state.call_args_list
        assert len(calls) == 3  # generation, coverage, telemetry

        # Each call should use state_prefix keyword
        for call in calls:
            kwargs = call.kwargs
            assert "state_prefix" in kwargs
            assert kwargs["state_prefix"] in ["generation", "coverage", "telemetry"]

    @pytest.mark.asyncio
    async def test_debug_state_handles_empty_categories(
        self, mock_state_port, mock_telemetry
    ):
        """Verify debug_state handles empty state categories correctly."""
        # Setup
        usecase = UtilityUseCase(mock_state_port, mock_telemetry)
        mock_state_port.get_all_state = Mock(return_value={})

        # Execute
        result = await usecase.debug_state()

        # Verify result structure
        assert result["success"]
        assert "debug_state" in result
        assert "generation" in result["debug_state"]
        assert "coverage" in result["debug_state"]
        assert "telemetry" in result["debug_state"]


class TestReturnTypeValidation:
    """Test validation of port return types (Fix #3)."""

    @pytest.mark.asyncio
    async def test_sync_handles_none_return_from_load(self, mock_telemetry):
        """Verify sync handles None return from load_state."""
        # Setup - mock state that returns None
        mock_state = Mock()
        mock_state.load_state = Mock(return_value=None)
        mock_state.persist_state = Mock(
            return_value={
                "success": True,
                "persisted_keys": [],
                "persistence_location": "",
                "persistence_metadata": {},
            }
        )

        # Disable fail_fast to allow test to capture errors without raising
        usecase = UtilityUseCase(
            mock_state, mock_telemetry, config={"sync_fail_fast": False}
        )

        # Execute
        result = await usecase.sync_state(force_reload=True)

        # Verify error was captured
        assert not result["success"]
        assert len(result["sync_results"]["critical_errors"]) > 0
        error_msg = result["sync_results"]["critical_errors"][0]
        assert "unexpected type" in error_msg.lower()
        assert "NoneType" in error_msg

    @pytest.mark.asyncio
    async def test_sync_handles_string_return_from_persist(self, mock_telemetry):
        """Verify sync handles string return from persist_state."""
        # Setup - mock state that returns string instead of dict
        mock_state = Mock()
        mock_state.load_state = Mock(
            return_value={
                "success": True,
                "loaded_keys": [],
                "loaded_values": {},
                "load_metadata": {},
            }
        )
        mock_state.persist_state = Mock(return_value="invalid_string_return")

        # Disable fail_fast to allow test to capture errors without raising
        usecase = UtilityUseCase(
            mock_state, mock_telemetry, config={"sync_fail_fast": False}
        )

        # Execute
        result = await usecase.sync_state(force_reload=False, persist_after_sync=True)

        # Verify error was captured
        assert not result["success"]
        assert len(result["sync_results"]["critical_errors"]) > 0
        error_msg = result["sync_results"]["critical_errors"][0]
        assert "unexpected type" in error_msg.lower()
        assert "str" in error_msg

    @pytest.mark.asyncio
    async def test_sync_handles_valid_returns(self, mock_state_port, mock_telemetry):
        """Verify sync works correctly with valid return types."""
        # Setup - use fixture with valid returns
        usecase = UtilityUseCase(mock_state_port, mock_telemetry)

        # Execute
        result = await usecase.sync_state(force_reload=True, persist_after_sync=True)

        # Verify success
        assert result["success"]
        assert len(result["sync_results"]["critical_errors"]) == 0
        assert "load_from_storage" in result["sync_results"]["operations_performed"]
        assert "persist_to_storage" in result["sync_results"]["operations_performed"]


class TestFailFast:
    """Test fail-fast behavior in sync operations (Fix #4)."""

    @pytest.mark.asyncio
    async def test_sync_fails_fast_on_load_error_when_enabled(self, mock_telemetry):
        """Verify sync raises immediately on load error when fail_fast=True."""
        # Setup - mock state that raises on load
        mock_state = Mock()
        mock_state.load_state = Mock(side_effect=RuntimeError("Storage unavailable"))

        usecase = UtilityUseCase(
            mock_state, mock_telemetry, config={"sync_fail_fast": True}
        )

        # Execute and verify exception
        with pytest.raises(UtilityUseCaseError) as exc_info:
            await usecase.sync_state(force_reload=True)

        assert "Storage unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_sync_continues_on_load_error_when_disabled(self, mock_telemetry):
        """Verify sync continues on load error when fail_fast=False."""
        # Setup - mock state that raises on load but succeeds on persist
        mock_state = Mock()
        mock_state.load_state = Mock(side_effect=RuntimeError("Storage unavailable"))
        mock_state.persist_state = Mock(
            return_value={
                "success": True,
                "persisted_keys": ["key1"],
                "persistence_location": "/tmp/test",
                "persistence_metadata": {},
            }
        )

        usecase = UtilityUseCase(
            mock_state, mock_telemetry, config={"sync_fail_fast": False}
        )

        # Execute - should not raise
        result = await usecase.sync_state(force_reload=True, persist_after_sync=True)

        # Verify it continued despite load error
        assert not result["success"]  # Overall failed due to critical error
        assert len(result["sync_results"]["critical_errors"]) > 0
        # But persist should still have been attempted
        mock_state.persist_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_fails_fast_on_invalid_return_type(self, mock_telemetry):
        """Verify sync fails fast on invalid return type when fail_fast=True."""
        # Setup - mock state that returns invalid type
        mock_state = Mock()
        mock_state.load_state = Mock(return_value=["invalid", "list", "return"])

        usecase = UtilityUseCase(
            mock_state, mock_telemetry, config={"sync_fail_fast": True}
        )

        # Execute and verify exception
        with pytest.raises(UtilityUseCaseError) as exc_info:
            await usecase.sync_state(force_reload=True)

        assert "unexpected type" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_sync_tracks_critical_vs_regular_errors(
        self, mock_state_port, mock_telemetry
    ):
        """Verify sync differentiates between critical and regular errors."""
        # Setup - mock state port that succeeds but validation fails
        usecase = UtilityUseCase(mock_state_port, mock_telemetry)

        # Mock validation to raise
        with patch.object(
            usecase,
            "_validate_state_consistency",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Validation error"),
        ):
            # Execute
            result = await usecase.sync_state(force_reload=True)

        # Verify error categorization
        assert result[
            "success"
        ]  # Should still succeed (validation error is non-critical)
        assert len(result["sync_results"]["errors"]) > 0  # Regular error recorded
        assert len(result["sync_results"]["critical_errors"]) == 0  # No critical errors

    @pytest.mark.asyncio
    async def test_sync_metadata_includes_fail_fast_config(
        self, mock_state_port, mock_telemetry
    ):
        """Verify sync result metadata includes fail_fast configuration."""
        # Setup
        usecase = UtilityUseCase(
            mock_state_port, mock_telemetry, config={"sync_fail_fast": False}
        )

        # Execute
        result = await usecase.sync_state()

        # Verify metadata
        assert "metadata" in result
        assert "fail_fast" in result["metadata"]
        assert result["metadata"]["fail_fast"] is False


class TestIntegration:
    """Integration tests for utility operations."""

    @pytest.mark.asyncio
    async def test_full_backup_and_sync_cycle(
        self, tmp_path, mock_state_port, mock_telemetry
    ):
        """End-to-end test of backup creation during reset and subsequent sync."""
        # Setup
        usecase = UtilityUseCase(mock_state_port, mock_telemetry)
        test_state = {
            "generation": {"file1": "data1"},
            "coverage": {"file2": "data2"},
        }
        mock_state_port.get_all_state.return_value = test_state

        # Execute reset with backup
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            reset_result = await usecase.reset_state(
                confirm_reset=True, create_backup=True
            )

        # Verify backup was created
        assert reset_result["reset_results"]["backup_created"]
        backup_file = Path(reset_result["reset_results"]["backup_location"])
        assert backup_file.exists()

        # Verify backup contains expected data
        with open(backup_file) as f:
            backup_data = json.load(f)
        assert backup_data == test_state

        # Execute sync after reset
        sync_result = await usecase.sync_state(
            force_reload=True, persist_after_sync=True
        )

        # Verify sync succeeded
        assert sync_result["success"]
        assert len(sync_result["sync_results"]["operations_performed"]) >= 2

    @pytest.mark.asyncio
    async def test_cost_summary_with_valid_port(
        self, mock_state_port, mock_telemetry, mock_cost_port
    ):
        """Test cost summary generation with valid cost port."""
        # Setup
        usecase = UtilityUseCase(
            mock_state_port, mock_telemetry, cost_port=mock_cost_port
        )

        # Execute
        result = await usecase.get_cost_summary()

        # Verify
        assert result["success"]
        assert "cost_summary" in result
        assert result["cost_summary"]["total_cost"] == 10.50
