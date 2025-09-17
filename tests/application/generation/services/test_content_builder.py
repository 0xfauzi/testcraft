"""
Tests for ContentBuilder service.

This module contains unit tests for the content builder service.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from testcraft.application.generation.services.content_builder import ContentBuilder
from testcraft.domain.models import TestGenerationPlan


class TestContentBuilder:
    """Test cases for ContentBuilder service."""

    @pytest.fixture
    def service(self, mock_parser_port):
        """Create ContentBuilder service."""
        return ContentBuilder(mock_parser_port)

    def test_build_code_content_basic(self, service, mock_parser_port):
        """Test basic code content building."""
        # Setup mock plan
        mock_plan = MagicMock(spec=TestGenerationPlan)
        mock_plan.elements_to_test = []
        
        # Setup mock parser
        mock_parser_port.parse_file.return_value = {
            "ast": None,
            "source_lines": ["import os", "def test_func(): pass"]
        }
        
        result = service.build_code_content(mock_plan, Path("test.py"))
        
        # Should return string content
        assert isinstance(result, str)

    def test_determine_test_path(self, service):
        """Test test path determination from a generation plan."""
        from testcraft.domain.models import TestElement, TestElementType
        
        # Create a mock plan with elements
        element = TestElement(
            name="test_func",
            type=TestElementType.FUNCTION,
            line_range=(1, 5),
            docstring="Test function",
        )
        mock_plan = MagicMock(spec=TestGenerationPlan)
        mock_plan.elements_to_test = [element]
        mock_plan.source_file_path = "src/module.py"
        
        result = service.determine_test_path(mock_plan)
        
        # Should return string path
        assert isinstance(result, str)
        assert "test" in result

    def test_clear_cache(self, service):
        """Test cache clearing functionality."""
        # This should not raise an exception
        service.clear_cache()
        
        # Verify cache is cleared
        assert len(service._parse_cache) == 0
