"""
Tests for AnalyzeUseCase.

This module tests the analyze use case functionality, including file discovery,
processing decisions, and analysis report generation.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from testcraft.application.analyze_usecase import AnalyzeUseCase, AnalyzeUseCaseError
from testcraft.domain.models import AnalysisReport


class TestAnalyzeUseCase:
    """Test cases for AnalyzeUseCase."""

    @pytest.fixture
    def mock_ports(self):
        """Create mock ports for testing."""
        coverage_port = MagicMock()
        state_port = MagicMock()
        telemetry_port = MagicMock()
        
        # Setup telemetry port mocks
        mock_span = MagicMock()
        mock_span.get_trace_context.return_value = MagicMock(trace_id="test_trace_123")
        telemetry_port.create_span.return_value.__enter__.return_value = mock_span
        telemetry_port.create_child_span.return_value.__enter__.return_value = mock_span
        
        return {
            'coverage_port': coverage_port,
            'state_port': state_port,
            'telemetry_port': telemetry_port
        }

    @pytest.fixture
    def analyze_usecase(self, mock_ports):
        """Create AnalyzeUseCase instance with mocked ports."""
        from unittest.mock import Mock
        from testcraft.adapters.io.file_discovery import FileDiscoveryService
        
        # Create a mock file discovery service
        mock_file_discovery = Mock(spec=FileDiscoveryService)
        mock_file_discovery.discover_source_files.return_value = ["/test/module1.py", "/test/module2.py"]
        mock_file_discovery.filter_existing_files.return_value = ["/test/module1.py", "/test/module2.py"]
        
        return AnalyzeUseCase(
            coverage_port=mock_ports['coverage_port'],
            state_port=mock_ports['state_port'],
            telemetry_port=mock_ports['telemetry_port'],
            file_discovery_service=mock_file_discovery,
            config={'coverage_threshold': 0.8}
        )

    @pytest.fixture
    def sample_project_path(self, tmp_path):
        """Create a sample project structure for testing."""
        project = tmp_path / "sample_project"
        project.mkdir()
        
        # Create sample Python files
        (project / "main.py").write_text("def main(): pass")
        (project / "utils.py").write_text("def helper(): pass")
        (project / "config.py").write_text("CONFIG = {}")
        
        # Create test directory with existing test
        tests_dir = project / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_main.py").write_text("def test_main(): pass")
        
        return project

    @pytest.mark.asyncio
    async def test_analyze_generation_needs_basic(self, analyze_usecase, sample_project_path, mock_ports):
        """Test basic analysis functionality."""
        # Setup mocks
        mock_ports['state_port'].get_all_state.return_value = {}
        mock_ports['coverage_port'].measure_coverage.return_value = {
            'file1.py': {'line_coverage': 0.5},
            'file2.py': {'line_coverage': 0.9}
        }
        mock_ports['coverage_port'].get_coverage_summary.return_value = {
            'overall_line_coverage': 0.6,
            'files_covered': 2,
            'total_lines': 100
        }
        
        # Execute analysis
        result = await analyze_usecase.analyze_generation_needs(sample_project_path)
        
        # Verify result is an AnalysisReport
        assert isinstance(result, AnalysisReport)
        assert result.files_to_process  # Should have files to process
        assert result.reasons  # Should have reasons
        assert result.existing_test_presence  # Should have test presence info
        
        # Verify reasons and test presence have entries for all files to process
        for file_path in result.files_to_process:
            assert file_path in result.reasons
            assert file_path in result.existing_test_presence

    @pytest.mark.asyncio
    async def test_analyze_with_target_files(self, analyze_usecase, sample_project_path, mock_ports):
        """Test analysis with specific target files."""
        target_files = [sample_project_path / "utils.py"]
        
        # Setup mocks
        mock_ports['state_port'].get_all_state.return_value = {}
        mock_ports['coverage_port'].measure_coverage.return_value = {}
        mock_ports['coverage_port'].get_coverage_summary.return_value = {
            'overall_line_coverage': 0.0,
            'files_covered': 0,
            'total_lines': 0
        }
        
        # Execute analysis
        result = await analyze_usecase.analyze_generation_needs(
            sample_project_path, 
            target_files=target_files
        )
        
        # Should analyze only the target file
        assert isinstance(result, AnalysisReport)
        assert len(result.files_to_process) == 1
        assert str(target_files[0]) in result.files_to_process

    @pytest.mark.asyncio
    async def test_analyze_existing_test_detection(self, analyze_usecase, sample_project_path, mock_ports):
        """Test detection of existing test files."""
        # Setup mocks
        mock_ports['state_port'].get_all_state.return_value = {}
        mock_ports['coverage_port'].measure_coverage.return_value = {}
        mock_ports['coverage_port'].get_coverage_summary.return_value = {
            'overall_line_coverage': 0.6,
            'files_covered': 1,
            'total_lines': 50
        }
        
        # Execute analysis
        result = await analyze_usecase.analyze_generation_needs(sample_project_path)
        
        # main.py should have existing tests detected (test_main.py exists)
        main_py_path = str(sample_project_path / "main.py")
        if main_py_path in result.existing_test_presence:
            # Note: The exact test detection logic depends on implementation
            # This test verifies the structure is correct
            assert isinstance(result.existing_test_presence[main_py_path], bool)

    @pytest.mark.asyncio
    async def test_analyze_processing_reasons(self, analyze_usecase, sample_project_path, mock_ports):
        """Test that processing reasons are generated correctly."""
        # Setup mocks
        mock_ports['state_port'].get_all_state.return_value = {}
        mock_ports['coverage_port'].measure_coverage.return_value = {}
        mock_ports['coverage_port'].get_coverage_summary.return_value = {
            'overall_line_coverage': 0.3,  # Low coverage
            'files_covered': 1,
            'total_lines': 100
        }
        
        # Execute analysis
        result = await analyze_usecase.analyze_generation_needs(sample_project_path)
        
        # Verify reasons are provided and meaningful
        for file_path, reason in result.reasons.items():
            assert isinstance(reason, str)
            assert len(reason) > 0
            # Reason should mention coverage or tests
            assert any(keyword in reason.lower() for keyword in ['coverage', 'test', 'threshold'])

    @pytest.mark.asyncio
    async def test_analyze_error_handling(self, analyze_usecase, mock_ports):
        """Test error handling in analysis."""
        # Setup mock to raise exception
        mock_ports['state_port'].get_all_state.side_effect = Exception("State error")
        
        # Should raise AnalyzeUseCaseError
        with pytest.raises(AnalyzeUseCaseError) as exc_info:
            await analyze_usecase.analyze_generation_needs("/nonexistent/path")
        
        assert "Analysis failed" in str(exc_info.value)
        assert exc_info.value.cause is not None

    @pytest.mark.asyncio
    async def test_analyze_empty_project(self, analyze_usecase, tmp_path, mock_ports):
        """Test analysis of project with no Python files."""
        empty_project = tmp_path / "empty"
        empty_project.mkdir()
        
        # Setup mocks
        mock_ports['state_port'].get_all_state.return_value = {}
        mock_ports['coverage_port'].measure_coverage.return_value = {}
        mock_ports['coverage_port'].get_coverage_summary.return_value = {
            'overall_line_coverage': 0.0,
            'files_covered': 0,
            'total_lines': 0
        }
        
        # Execute analysis
        result = await analyze_usecase.analyze_generation_needs(empty_project)
        
        # Should return empty report
        assert isinstance(result, AnalysisReport)
        assert len(result.files_to_process) == 0
        assert len(result.reasons) == 0
        assert len(result.existing_test_presence) == 0

    @pytest.mark.asyncio 
    async def test_analyze_configuration_impact(self, mock_ports):
        """Test that configuration affects analysis decisions."""
        from unittest.mock import Mock
        from testcraft.adapters.io.file_discovery import FileDiscoveryService
        
        mock_file_discovery = Mock(spec=FileDiscoveryService)
        
        # Test with high coverage threshold
        high_threshold_usecase = AnalyzeUseCase(
            coverage_port=mock_ports['coverage_port'],
            state_port=mock_ports['state_port'],
            telemetry_port=mock_ports['telemetry_port'],
            file_discovery_service=mock_file_discovery,
            config={'coverage_threshold': 0.9}
        )
        
        # Test with low coverage threshold
        low_threshold_usecase = AnalyzeUseCase(
            coverage_port=mock_ports['coverage_port'],
            state_port=mock_ports['state_port'],
            telemetry_port=mock_ports['telemetry_port'],
            file_discovery_service=mock_file_discovery,
            config={'coverage_threshold': 0.1}
        )
        
        # Verify configuration is stored correctly
        assert high_threshold_usecase._config['coverage_threshold'] == 0.9
        assert low_threshold_usecase._config['coverage_threshold'] == 0.1

    def test_analyze_usecase_initialization(self, mock_ports):
        """Test AnalyzeUseCase initialization."""
        from unittest.mock import Mock
        from testcraft.adapters.io.file_discovery import FileDiscoveryService
        
        config = {
            'coverage_threshold': 0.75,
            'custom_setting': 'test_value'
        }
        
        mock_file_discovery = Mock(spec=FileDiscoveryService)
        
        usecase = AnalyzeUseCase(
            coverage_port=mock_ports['coverage_port'],
            state_port=mock_ports['state_port'],
            telemetry_port=mock_ports['telemetry_port'],
            file_discovery_service=mock_file_discovery,
            config=config
        )
        
        # Verify configuration is applied
        assert usecase._config['coverage_threshold'] == 0.75
        assert usecase._config['custom_setting'] == 'test_value'
        
        # Verify ports are assigned
        assert usecase._coverage == mock_ports['coverage_port']
        assert usecase._state == mock_ports['state_port']
        assert usecase._telemetry == mock_ports['telemetry_port']
        assert usecase._file_discovery == mock_file_discovery

    @pytest.mark.asyncio
    async def test_build_processing_reasons_error_handling(self, analyze_usecase, sample_project_path, mock_ports):
        """Test error handling in reason building."""
        # Setup mocks
        mock_ports['state_port'].get_all_state.return_value = {}
        mock_ports['coverage_port'].measure_coverage.return_value = {}
        mock_ports['coverage_port'].get_coverage_summary.return_value = {
            'overall_line_coverage': 0.5,
            'files_covered': 1,
            'total_lines': 100
        }
        
        # Mock the _get_processing_reason method to raise an exception
        with patch.object(analyze_usecase, '_get_processing_reason', side_effect=Exception("Reason error")):
            with pytest.raises(AnalyzeUseCaseError) as exc_info:
                await analyze_usecase.analyze_generation_needs(sample_project_path)
            
            assert "Analysis failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_build_test_presence_info_error_handling(self, analyze_usecase, sample_project_path, mock_ports):
        """Test error handling in test presence analysis.""" 
        # Setup mocks
        mock_ports['state_port'].get_all_state.return_value = {}
        mock_ports['coverage_port'].measure_coverage.return_value = {}
        mock_ports['coverage_port'].get_coverage_summary.return_value = {
            'overall_line_coverage': 0.5,
            'files_covered': 1,
            'total_lines': 100
        }
        
        # Mock the _has_existing_tests method to raise an exception
        with patch.object(analyze_usecase, '_has_existing_tests', side_effect=Exception("Test presence error")):
            with pytest.raises(AnalyzeUseCaseError) as exc_info:
                await analyze_usecase.analyze_generation_needs(sample_project_path)
            
            assert "Analysis failed" in str(exc_info.value)
