"""
Tests for PlanBuilder service.

This module contains unit tests for the plan builder service.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from testcraft.application.generation.services.plan_builder import PlanBuilder
from testcraft.domain.models import TestGenerationPlan


class TestPlanBuilder:
    """Test cases for PlanBuilder service."""

    @pytest.fixture
    def service(
        self, 
        mock_parser_port, 
        mock_file_discovery_service, 
        mock_telemetry_port
    ):
        """Create PlanBuilder service."""
        parser_port = mock_parser_port
        file_discovery_service = mock_file_discovery_service
        telemetry_port, _ = mock_telemetry_port
        
        return PlanBuilder(
            parser_port=parser_port,
            file_discovery_service=file_discovery_service,
            telemetry_port=telemetry_port,
            coverage_threshold=0.8
        )

    def test_build_plans_basic(self, service, mock_parser_port):
        """Test basic plan building."""
        # Setup mock parser to return valid AST elements
        mock_parser_port.parse_file.return_value = {
            "ast": None,
            "source_lines": ["def test_func(): pass"],
            "elements": []
        }
        
        source_paths = [Path("src/module.py")]
        
        results = service.build_plans(source_paths)
        
        # Should return list of TestGenerationPlan
        assert isinstance(results, list)

    def test_decide_files_to_process_basic(self, service):
        """Test file processing decision."""
        discovered_files = [Path("src/new_module.py")]
        coverage_data = {"overall_line_coverage": 0.5}
        
        result = service.decide_files_to_process(discovered_files, coverage_data)
        
        # Should return list of files to process
        assert isinstance(result, list)

    def test_set_project_context(self, service):
        """Test setting project context."""
        project_path = Path("/test/project")
        test_files = ["test_module.py", "test_utils.py"]
        
        # Should not raise an exception
        service.set_project_context(project_path, test_files)
        
        # Verify context was set
        assert service._current_project_path == project_path
        assert service._cached_test_files == test_files
