"""
Tests for CoverageEvaluator service.

This module contains unit tests for the coverage evaluation service.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from testcraft.application.generation.services.coverage_evaluator import (
    CoverageEvaluator,
)


class TestCoverageEvaluator:
    """Test cases for CoverageEvaluator service."""

    @pytest.fixture
    def service(self, mock_coverage_port, mock_telemetry_port):
        """Create CoverageEvaluator service."""
        coverage_port = mock_coverage_port
        telemetry_port, _ = mock_telemetry_port
        return CoverageEvaluator(coverage_port, telemetry_port)

    def test_measure_initial_success(self, service, mock_coverage_port, mock_telemetry_port):
        """Test successful initial coverage measurement."""
        coverage_port = mock_coverage_port
        telemetry_port, mock_span = mock_telemetry_port

        # Setup mocks
        coverage_port.measure_coverage.return_value = {
            "file1.py": {"lines": 10, "covered": 8}
        }
        coverage_port.get_coverage_summary.return_value = {
            "overall_line_coverage": 0.8,
            "overall_branch_coverage": 0.75,
            "files_covered": 1,
            "total_lines": 10,
        }

        source_files = [Path("file1.py")]
        result = service.measure_initial(source_files)

        # Verify calls
        coverage_port.measure_coverage.assert_called_once_with(["file1.py"])
        coverage_port.get_coverage_summary.assert_called_once()

        # Verify result
        assert result["overall_line_coverage"] == 0.8
        assert result["overall_branch_coverage"] == 0.75
        assert result["files_covered"] == 1
        assert result["total_lines"] == 10

        # Verify telemetry
        mock_span.set_attribute.assert_called()

    def test_measure_initial_failure_graceful(self, service, mock_coverage_port, mock_telemetry_port):
        """Test graceful failure handling in initial coverage measurement."""
        coverage_port = mock_coverage_port
        telemetry_port, mock_span = mock_telemetry_port

        # Setup mocks to fail
        coverage_port.measure_coverage.side_effect = Exception("Coverage failed")

        source_files = [Path("file1.py")]
        result = service.measure_initial(source_files)

        # Should return empty coverage data
        assert result["overall_line_coverage"] == 0.0
        assert result["overall_branch_coverage"] == 0.0
        assert result["files_covered"] == 0
        assert result["total_lines"] == 0

    def test_calculate_delta(self, service, mock_telemetry_port):
        """Test coverage delta calculation."""
        initial = {
            "overall_line_coverage": 0.6,
            "overall_branch_coverage": 0.5,
            "total_lines": 100,
        }
        final = {
            "overall_line_coverage": 0.8,
            "overall_branch_coverage": 0.7,
            "total_lines": 120,
        }

        result = service.calculate_delta(initial, final)

        assert abs(result["line_coverage_delta"] - 0.2) < 0.001
        assert abs(result["branch_coverage_delta"] - 0.2) < 0.001
        assert result["total_lines_delta"] == 20
        assert result["initial_line_coverage"] == 0.6
        assert result["final_line_coverage"] == 0.8
        assert abs(result["improvement_percentage"] - 20.0) < 0.001

    def test_calculate_delta_error_handling(self, service, mock_telemetry_port):
        """Test delta calculation with error handling."""
        # Invalid input should be handled gracefully
        result = service.calculate_delta({}, None)

        assert "error" in result
        assert result["line_coverage_delta"] == 0.0
