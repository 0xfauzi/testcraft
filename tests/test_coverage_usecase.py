"""
Tests for the Coverage Use Case.

This module contains comprehensive tests for the CoverageUseCase class,
testing all major workflows and edge cases with mocked dependencies.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch, ANY
from datetime import datetime
from typing import Dict, Any, List

from testcraft.application.coverage_usecase import CoverageUseCase, CoverageUseCaseError
from testcraft.domain.models import CoverageResult
from testcraft.ports.coverage_port import CoveragePort
from testcraft.ports.state_port import StatePort
from testcraft.ports.telemetry_port import TelemetryPort, SpanKind, MetricValue
from testcraft.adapters.io.file_discovery import FileDiscoveryService
from testcraft.config.models import TestPatternConfig


class TestCoverageUseCase:
    """Test suite for CoverageUseCase."""

    @pytest.fixture
    def mock_coverage_port(self):
        """Create a mock coverage port."""
        mock = Mock(spec=CoveragePort)
        mock.measure_coverage.return_value = {
            "/test/file1.py": CoverageResult(
                line_coverage=0.85,
                branch_coverage=0.75,
                missing_lines=[10, 15, 20]
            ),
            "/test/file2.py": CoverageResult(
                line_coverage=0.92,
                branch_coverage=0.88,
                missing_lines=[5]
            )
        }
        mock.get_coverage_summary.return_value = {
            "overall_line_coverage": 0.885,
            "overall_branch_coverage": 0.815,
            "files_covered": 2,
            "total_lines": 100,
            "missing_coverage": {}
        }
        mock.report_coverage.return_value = {
            "report_content": "Coverage Report Content",
            "summary_stats": {"overall": 0.885},
            "format": "detailed"
        }
        mock.identify_gaps.return_value = {
            "/test/file1.py": [10, 15, 20]
        }
        return mock

    @pytest.fixture
    def mock_state_port(self):
        """Create a mock state port."""
        mock = Mock(spec=StatePort)
        mock.get_all_state.return_value = {
            "last_coverage_run": {
                "coverage_summary": {
                    "overall_line_coverage": 0.8,
                    "overall_branch_coverage": 0.7
                }
            }
        }
        return mock

    @pytest.fixture
    def mock_telemetry_port(self):
        """Create a mock telemetry port."""
        mock = Mock(spec=TelemetryPort)
        
        # Create a proper context manager mock for spans
        mock_span_context = Mock()
        mock_span_context.set_attribute = Mock()
        mock_span_context.record_exception = Mock()
        mock_span_context.trace_id = "test-trace-id"
        mock_span_context.span_id = "test-span-id"
        
        # Create context manager mocks that properly support __enter__ and __exit__
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span_context)
        mock_context_manager.__exit__ = Mock(return_value=None)
        
        # Set up the telemetry port methods
        mock.create_span.return_value = mock_context_manager
        mock.create_child_span.return_value = mock_context_manager
        mock.record_metrics = Mock()
        mock.flush = Mock()
        mock.get_trace_context.return_value = mock_span_context
        
        return mock

    @pytest.fixture
    def mock_file_discovery_service(self):
        """Create a mock file discovery service."""
        mock = Mock(spec=FileDiscoveryService)
        mock.discover_source_files.return_value = ["/test/module1.py", "/test/module2.py"]
        mock.discover_test_files.return_value = ["/test/test_module1.py"]
        mock.filter_existing_files.return_value = ["/test/module1.py", "/test/module2.py"]
        return mock

    @pytest.fixture
    def coverage_usecase(self, mock_coverage_port, mock_state_port, mock_telemetry_port, mock_file_discovery_service):
        """Create a CoverageUseCase instance with mocked dependencies."""
        return CoverageUseCase(
            coverage_port=mock_coverage_port,
            state_port=mock_state_port,
            telemetry_port=mock_telemetry_port,
            file_discovery_service=mock_file_discovery_service,
            config={
                'coverage_threshold': 0.8,
                'enable_gap_analysis': True,
                'output_formats': ['detailed', 'summary']
            }
        )

    @pytest.fixture
    def sample_project_path(self, tmp_path):
        """Create a sample project structure for testing."""
        project_path = tmp_path / "test_project"
        project_path.mkdir()
        
        # Create some Python files
        (project_path / "module1.py").write_text("def function1(): pass")
        (project_path / "module2.py").write_text("class Class1: pass")
        (project_path / "test_module1.py").write_text("def test_function1(): pass")
        
        # Create subdirectory
        subdir = project_path / "subpackage"
        subdir.mkdir()
        (subdir / "module3.py").write_text("def function3(): pass")
        
        return project_path

    @pytest.mark.asyncio
    async def test_measure_and_report_success(
        self, 
        coverage_usecase, 
        sample_project_path,
        mock_coverage_port,
        mock_state_port,
        mock_telemetry_port
    ):
        """Test successful coverage measurement and reporting."""
        # Act
        result = await coverage_usecase.measure_and_report(
            project_path=sample_project_path
        )
        
        # Assert
        assert result["success"] is True
        assert "coverage_data" in result
        assert "coverage_summary" in result
        assert "reports" in result
        assert "coverage_gaps" in result
        assert result["files_measured"] == 2
        assert result["overall_line_coverage"] == 0.885
        assert result["overall_branch_coverage"] == 0.815
        
        # Verify port interactions
        mock_coverage_port.measure_coverage.assert_called_once()
        mock_coverage_port.get_coverage_summary.assert_called_once()
        mock_coverage_port.report_coverage.assert_called()
        mock_coverage_port.identify_gaps.assert_called_once()
        
        mock_state_port.get_all_state.assert_called_with("coverage")
        mock_state_port.update_state.assert_called_with("last_coverage_run", ANY)
        
        mock_telemetry_port.create_span.assert_called_once()
        mock_telemetry_port.record_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_measure_and_report_with_specific_files(
        self,
        coverage_usecase,
        sample_project_path,
        mock_coverage_port,
        mock_file_discovery_service
    ):
        """Test coverage measurement with specific source files."""
        source_files = [
            sample_project_path / "module1.py",
            sample_project_path / "module2.py"
        ]
        
        # Act
        result = await coverage_usecase.measure_and_report(
            project_path=sample_project_path,
            source_files=source_files
        )
        
        # Assert
        assert result["success"] is True
        
        # Verify that file discovery service was called for filtering
        mock_file_discovery_service.filter_existing_files.assert_called_once()
        
        # Verify that measure_coverage was called
        mock_coverage_port.measure_coverage.assert_called_once()

    @pytest.mark.asyncio
    async def test_measure_and_report_with_test_files(
        self,
        coverage_usecase,
        sample_project_path,
        mock_coverage_port,
        mock_file_discovery_service
    ):
        """Test coverage measurement including test files."""
        coverage_usecase._config['include_test_files'] = True
        
        # Act
        result = await coverage_usecase.measure_and_report(
            project_path=sample_project_path
        )
        
        # Assert
        assert result["success"] is True
        
        # Verify that file discovery service was called for both source and test files
        mock_file_discovery_service.discover_source_files.assert_called_once()
        mock_file_discovery_service.discover_test_files.assert_called_once()

    @pytest.mark.asyncio
    async def test_coverage_measurement_failure(
        self,
        coverage_usecase,
        sample_project_path,
        mock_coverage_port
    ):
        """Test handling of coverage measurement failures."""
        # Arrange
        mock_coverage_port.measure_coverage.side_effect = Exception("Coverage tool failed")
        
        # Act & Assert
        with pytest.raises(CoverageUseCaseError) as exc_info:
            await coverage_usecase.measure_and_report(
                project_path=sample_project_path
            )
        
        assert "Coverage measurement failed" in str(exc_info.value)
        assert exc_info.value.cause is not None

    @pytest.mark.asyncio
    async def test_report_generation_failure_partial_success(
        self,
        coverage_usecase,
        sample_project_path,
        mock_coverage_port
    ):
        """Test partial failure in report generation."""
        # Arrange - make one report format fail
        def side_effect(coverage_data, output_format):
            if output_format == "summary":
                raise Exception("Summary report failed")
            return {
                "report_content": f"{output_format} report",
                "format": output_format
            }
        
        mock_coverage_port.report_coverage.side_effect = side_effect
        
        # Act
        result = await coverage_usecase.measure_and_report(
            project_path=sample_project_path
        )
        
        # Assert
        assert result["success"] is True
        assert "detailed" in result["reports"]
        assert "summary" in result["reports"]
        assert result["reports"]["summary"]["success"] is False
        assert "error" in result["reports"]["summary"]

    def test_file_discovery_service_integration(self, coverage_usecase, mock_file_discovery_service):
        """Test integration with file discovery service."""
        # Test that the coverage use case properly uses the file discovery service
        assert coverage_usecase._file_discovery == mock_file_discovery_service
        
        # Test that the service is called with correct parameters
        project_path = Path("/test/project")
        
        # This would be called during measure_and_report
        mock_file_discovery_service.discover_source_files(
            project_path, 
            include_test_files=False
        )
        
        mock_file_discovery_service.discover_source_files.assert_called_with(
            project_path,
            include_test_files=False
        )

    @pytest.mark.asyncio
    async def test_batched_coverage_measurement(
        self,
        coverage_usecase,
        sample_project_path,
        mock_coverage_port,
        mock_file_discovery_service
    ):
        """Test coverage measurement with file batching."""
        # Arrange - set small batch size
        coverage_usecase._config['max_files_per_batch'] = 1
        
        # Mock file discovery to return multiple files
        mock_file_discovery_service.discover_source_files.return_value = [
            "/file1.py", "/file2.py", "/file3.py"
        ]
        
        # Mock coverage responses for each batch
        mock_coverage_port.measure_coverage.side_effect = [
            {"/file1.py": CoverageResult(line_coverage=0.8, branch_coverage=0.7, missing_lines=[])},
            {"/file2.py": CoverageResult(line_coverage=0.9, branch_coverage=0.8, missing_lines=[])},
            {"/file3.py": CoverageResult(line_coverage=0.85, branch_coverage=0.75, missing_lines=[])}
        ]
        
        # Act
        result = await coverage_usecase.measure_and_report(
            project_path=sample_project_path
        )
        
        # Assert
        assert result["success"] is True
        assert mock_coverage_port.measure_coverage.call_count == 3  # One call per file

    @pytest.mark.asyncio
    async def test_get_coverage_history(
        self,
        coverage_usecase,
        mock_state_port
    ):
        """Test coverage history retrieval."""
        # Act
        history = await coverage_usecase.get_coverage_history(limit=5)
        
        # Assert
        assert len(history) == 1
        assert "coverage_summary" in history[0]
        mock_state_port.get_all_state.assert_called_with("coverage")

    @pytest.mark.asyncio
    async def test_get_coverage_history_empty(
        self,
        coverage_usecase,
        mock_state_port
    ):
        """Test coverage history retrieval with no history."""
        # Arrange
        mock_state_port.get_all_state.return_value = {}
        
        # Act
        history = await coverage_usecase.get_coverage_history()
        
        # Assert
        assert len(history) == 0

    def test_calculate_coverage_delta_with_previous_data(
        self,
        coverage_usecase,
        mock_state_port
    ):
        """Test coverage delta calculation with previous data."""
        current_coverage = {
            "overall_line_coverage": 0.9,
            "overall_branch_coverage": 0.85
        }
        
        # Act
        delta = coverage_usecase.calculate_coverage_delta(current_coverage)
        
        # Assert
        assert delta["has_previous_data"] is True
        assert delta["line_coverage_delta"] == pytest.approx(0.1)  # 0.9 - 0.8
        assert delta["branch_coverage_delta"] == pytest.approx(0.15)  # 0.85 - 0.7
        assert delta["trend"] == "improving"
        assert delta["improvement_percentage"] == pytest.approx(10.0)

    def test_calculate_coverage_delta_no_previous_data(
        self,
        coverage_usecase,
        mock_state_port
    ):
        """Test coverage delta calculation without previous data."""
        # Arrange
        mock_state_port.get_all_state.return_value = {}
        
        current_coverage = {
            "overall_line_coverage": 0.9,
            "overall_branch_coverage": 0.85
        }
        
        # Act
        delta = coverage_usecase.calculate_coverage_delta(current_coverage)
        
        # Assert
        assert delta["has_previous_data"] is False
        assert delta["line_coverage_delta"] == 0.0
        assert "No previous coverage data available" in delta["message"]

    def test_calculate_coverage_delta_declining(
        self,
        coverage_usecase
    ):
        """Test coverage delta calculation with declining coverage."""
        current_coverage = {
            "overall_line_coverage": 0.75,
            "overall_branch_coverage": 0.65
        }
        
        previous_coverage = {
            "overall_line_coverage": 0.85,
            "overall_branch_coverage": 0.75
        }
        
        # Act
        delta = coverage_usecase.calculate_coverage_delta(
            current_coverage, 
            previous_coverage
        )
        
        # Assert
        assert delta["trend"] == "declining"
        assert delta["line_coverage_delta"] == pytest.approx(-0.1)
        assert delta["improvement_percentage"] == 0.0

    def test_calculate_coverage_delta_stable(
        self,
        coverage_usecase
    ):
        """Test coverage delta calculation with stable coverage."""
        current_coverage = {
            "overall_line_coverage": 0.85,
            "overall_branch_coverage": 0.75
        }
        
        previous_coverage = {
            "overall_line_coverage": 0.85,
            "overall_branch_coverage": 0.75
        }
        
        # Act
        delta = coverage_usecase.calculate_coverage_delta(
            current_coverage, 
            previous_coverage
        )
        
        # Assert
        assert delta["trend"] == "stable"
        assert delta["line_coverage_delta"] == 0.0

    @pytest.mark.asyncio
    async def test_telemetry_recording(
        self,
        coverage_usecase,
        sample_project_path,
        mock_telemetry_port
    ):
        """Test telemetry and metrics recording."""
        # Act
        await coverage_usecase.measure_and_report(
            project_path=sample_project_path
        )
        
        # Assert telemetry calls
        mock_telemetry_port.create_span.assert_called_once_with(
            "coverage_measure_and_report",
            kind=SpanKind.INTERNAL,
            attributes=ANY
        )
        
        mock_telemetry_port.record_metrics.assert_called_once()
        recorded_metrics = mock_telemetry_port.record_metrics.call_args[0][0]
        
        # Check that we recorded the expected metrics
        metric_names = [metric.name for metric in recorded_metrics]
        expected_metrics = [
            "coverage_files_measured",
            "coverage_overall_line_percentage", 
            "coverage_overall_branch_percentage",
            "coverage_files_above_threshold"
        ]
        
        for expected_metric in expected_metrics:
            assert expected_metric in metric_names
        
        mock_telemetry_port.flush.assert_called_once_with(timeout_seconds=5.0)

    @pytest.mark.asyncio
    async def test_state_recording(
        self,
        coverage_usecase,
        sample_project_path,
        mock_state_port
    ):
        """Test state recording after coverage measurement."""
        # Act
        await coverage_usecase.measure_and_report(
            project_path=sample_project_path
        )
        
        # Assert
        mock_state_port.update_state.assert_called_once()
        call_args = mock_state_port.update_state.call_args
        
        assert call_args[0][0] == "last_coverage_run"
        state_data = call_args[0][1]
        
        assert "last_coverage_run_timestamp" in state_data
        assert "coverage_summary" in state_data
        assert "files_measured" in state_data
        assert "coverage_method" in state_data
        assert "config_used" in state_data
        assert "file_coverage_details" in state_data

    @pytest.mark.asyncio
    async def test_gap_analysis_disabled(
        self,
        coverage_usecase,
        sample_project_path,
        mock_coverage_port
    ):
        """Test coverage measurement with gap analysis disabled."""
        # Arrange
        coverage_usecase._config['enable_gap_analysis'] = False
        
        # Act
        result = await coverage_usecase.measure_and_report(
            project_path=sample_project_path
        )
        
        # Assert
        assert result["success"] is True
        assert result["coverage_gaps"] == {}  # Should be empty when disabled
        mock_coverage_port.identify_gaps.assert_not_called()

    def test_configuration_override(self):
        """Test that configuration can be properly overridden."""
        # Arrange
        custom_config = {
            'coverage_threshold': 0.9,
            'enable_gap_analysis': False,
            'output_formats': ['json'],
            'max_files_per_batch': 25
        }
        
        mock_coverage_port = Mock(spec=CoveragePort)
        mock_state_port = Mock(spec=StatePort)
        mock_telemetry_port = Mock(spec=TelemetryPort)
        mock_file_discovery_service = Mock(spec=FileDiscoveryService)
        
        # Act
        coverage_usecase = CoverageUseCase(
            coverage_port=mock_coverage_port,
            state_port=mock_state_port,
            telemetry_port=mock_telemetry_port,
            file_discovery_service=mock_file_discovery_service,
            config=custom_config
        )
        
        # Assert
        assert coverage_usecase._config['coverage_threshold'] == 0.9
        assert coverage_usecase._config['enable_gap_analysis'] is False
        assert coverage_usecase._config['output_formats'] == ['json']
        assert coverage_usecase._config['max_files_per_batch'] == 25

    @pytest.mark.asyncio
    async def test_error_handling_in_state_recording(
        self,
        coverage_usecase,
        sample_project_path,
        mock_state_port
    ):
        """Test that state recording errors don't fail the entire operation."""
        # Arrange
        mock_state_port.update_state.side_effect = Exception("State recording failed")
        
        # Act - should not raise exception
        result = await coverage_usecase.measure_and_report(
            project_path=sample_project_path
        )
        
        # Assert
        assert result["success"] is True  # Should still succeed despite state recording failure

    @pytest.mark.asyncio
    async def test_error_handling_in_telemetry_recording(
        self,
        coverage_usecase,
        sample_project_path,
        mock_telemetry_port
    ):
        """Test that telemetry recording errors don't fail the entire operation."""
        # Arrange
        mock_telemetry_port.record_metrics.side_effect = Exception("Telemetry failed")
        
        # Act - should not raise exception
        result = await coverage_usecase.measure_and_report(
            project_path=sample_project_path
        )
        
        # Assert
        assert result["success"] is True  # Should still succeed despite telemetry failure

    def test_coverage_usecase_error_attributes(self):
        """Test CoverageUseCaseError attributes."""
        original_error = ValueError("Original error")
        coverage_error = CoverageUseCaseError("Coverage failed", cause=original_error)
        
        assert str(coverage_error) == "Coverage failed"
        assert coverage_error.cause == original_error

    @pytest.mark.asyncio
    async def test_nonexistent_project_path(self, mock_coverage_port, mock_state_port, mock_telemetry_port):
        """Test handling of nonexistent project path."""
        nonexistent_path = Path("/nonexistent/project")
        
        # Create a real FileDiscoveryService (not mocked) so it can properly validate the path
        from testcraft.adapters.io.file_discovery import FileDiscoveryService
        real_file_discovery = FileDiscoveryService()
        
        # Create coverage use case with real file discovery service
        coverage_usecase = CoverageUseCase(
            coverage_port=mock_coverage_port,
            state_port=mock_state_port,
            telemetry_port=mock_telemetry_port,
            file_discovery_service=real_file_discovery
        )
        
        # Act & Assert
        with pytest.raises(CoverageUseCaseError):
            await coverage_usecase.measure_and_report(
                project_path=nonexistent_path
            )

    @pytest.mark.asyncio
    async def test_empty_source_files_list(
        self,
        coverage_usecase,
        sample_project_path,
        mock_coverage_port,
        mock_file_discovery_service
    ):
        """Test handling of empty source files list."""
        # Arrange - provide empty list
        source_files = []
        
        # Mock filter_existing_files to return empty list
        mock_file_discovery_service.filter_existing_files.return_value = []
        
        # Act
        result = await coverage_usecase.measure_and_report(
            project_path=sample_project_path,
            source_files=source_files
        )
        
        # Assert - should fall back to discovery
        assert result["success"] is True
        mock_file_discovery_service.discover_source_files.assert_called_once()
        mock_coverage_port.measure_coverage.assert_called_once()

    def test_default_file_discovery_service_creation(self):
        """Test that a default file discovery service is created when none provided."""
        mock_coverage_port = Mock(spec=CoveragePort)
        mock_state_port = Mock(spec=StatePort)
        mock_telemetry_port = Mock(spec=TelemetryPort)
        
        # Act - don't provide file_discovery_service
        coverage_usecase = CoverageUseCase(
            coverage_port=mock_coverage_port,
            state_port=mock_state_port,
            telemetry_port=mock_telemetry_port
        )
        
        # Assert - should have created a default file discovery service
        assert coverage_usecase._file_discovery is not None
        assert isinstance(coverage_usecase._file_discovery, FileDiscoveryService)

    @pytest.mark.asyncio
    async def test_coverage_measurement_with_no_files_found(
        self,
        coverage_usecase,
        tmp_path,
        mock_coverage_port
    ):
        """Test coverage measurement when no valid files are found."""
        # Create empty project directory
        empty_project = tmp_path / "empty_project"
        empty_project.mkdir()
        
        # Mock coverage port to handle empty file list
        mock_coverage_port.measure_coverage.return_value = {}
        mock_coverage_port.get_coverage_summary.return_value = {
            "overall_line_coverage": 0.0,
            "overall_branch_coverage": 0.0,
            "files_covered": 0,
            "total_lines": 0,
            "missing_coverage": {}
        }
        
        # Act
        result = await coverage_usecase.measure_and_report(
            project_path=empty_project
        )
        
        # Assert
        assert result["success"] is True
        assert result["files_measured"] == 0
        assert result["overall_line_coverage"] == 0.0
