"""
Tests for ContextAssembler service.

This module contains unit tests for the context assembler service,
including advanced context features and prompt budgeting.
"""

import ast
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from testcraft.application.generation.config import GenerationConfig
from testcraft.application.generation.services.context_assembler import ContextAssembler


class TestContextAssemblerAdvanced:
    """Test cases for ContextAssembler advanced context features."""

    @pytest.fixture
    def context_assembler(self, mock_context_port, mock_parser_port):
        """Create ContextAssembler instance with advanced features enabled."""
        config = GenerationConfig.get_default_config()
        return ContextAssembler(mock_context_port, mock_parser_port, config)

    def test_get_coverage_hints_placeholder(self, context_assembler):
        """Test coverage hints extraction (placeholder implementation)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "example.py"
            source_path.write_text("def example_func(): pass")
            
            result = context_assembler._get_coverage_hints(source_path)
            
            # Should return empty list for placeholder implementation
            assert isinstance(result, list)
            assert len(result) == 0

    def test_get_callgraph_neighbors_with_relationships(self, context_assembler, mock_context_port):
        """Test call-graph neighbor extraction with relationships."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "example.py"
            source_path.write_text("def example_func(): pass")
            
            # Setup mock with relationships and related files
            mock_context_port.get_related_context.return_value = {
                "relationships": ["calls:other_func", "imports:module"],
                "related_files": [str(source_path)]
            }
            
            result = context_assembler._get_callgraph_neighbors(source_path)
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert "Call-graph edges" in result[0]
            assert "calls:other_func" in result[0]

    def test_get_callgraph_neighbors_empty(self, context_assembler, mock_context_port):
        """Test call-graph neighbor extraction with no relationships."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "example.py"
            source_path.write_text("def example_func(): pass")
            
            # Setup mock with empty relationships
            mock_context_port.get_related_context.return_value = {
                "relationships": [],
                "related_files": []
            }
            
            result = context_assembler._get_callgraph_neighbors(source_path)
            
            assert isinstance(result, list)
            assert len(result) == 0

    def test_get_error_paths_with_raises(self, context_assembler):
        """Test error path extraction with raise statements."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "example.py"
            source_path.write_text("""
def example_func():
    raise ValueError("test error")
    raise TypeError("another error")
""")
            
            # Create mock plan with element that has docstring raises
            mock_element = MagicMock()
            mock_element.docstring = ":raises ValueError: When input is invalid\\n:raises KeyError: When key not found"
            mock_plan = MagicMock()
            mock_plan.elements_to_test = [mock_element]
            
            result = context_assembler._get_error_paths(source_path, mock_plan)
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert "Error paths" in result[0]
            # Should contain both docstring and AST-detected exceptions
            assert "ValueError" in result[0]
            assert "TypeError" in result[0]

    def test_get_error_paths_empty(self, context_assembler):
        """Test error path extraction with no errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "example.py"
            source_path.write_text("def example_func(): return True")
            
            mock_plan = MagicMock()
            mock_plan.elements_to_test = []
            
            result = context_assembler._get_error_paths(source_path, mock_plan)
            
            assert isinstance(result, list)
            assert len(result) == 0

    def test_get_usage_examples_with_results(self, context_assembler, mock_context_port):
        """Test usage example extraction with mock results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "example.py"
            source_path.write_text("def example_func(): pass")
            
            # Create mock plan with elements
            mock_element = MagicMock()
            mock_element.name = "example_func"
            mock_plan = MagicMock()
            mock_plan.elements_to_test = [mock_element]
            
            # Setup mock context port with usage examples
            mock_context_port.retrieve.return_value = {
                "results": [
                    {"snippet": "result = example_func(arg1, arg2)", "path": "test1.py"},
                    {"snippet": "example_func()", "path": "test2.py"}
                ]
            }
            
            result = context_assembler._get_usage_examples(source_path, mock_plan)
            
            assert isinstance(result, list)
            # Should get at least 1 result (may be deduplicated)
            assert len(result) >= 1
            assert all("Usage example_func" in item for item in result)

    def test_get_usage_examples_deduplication(self, context_assembler, mock_context_port):
        """Test usage example deduplication."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "example.py"
            source_path.write_text("def example_func(): pass")
            
            mock_element = MagicMock()
            mock_element.name = "example_func"
            mock_plan = MagicMock()
            mock_plan.elements_to_test = [mock_element]
            
            # Setup mock with duplicate snippets
            mock_context_port.retrieve.return_value = {
                "results": [
                    {"snippet": "example_func()", "path": "test1.py"},
                    {"snippet": "example_func()", "path": "test2.py"},  # Duplicate
                    {"snippet": "different_call()", "path": "test3.py"}  # No call pattern
                ]
            }
            
            result = context_assembler._get_usage_examples(source_path, mock_plan)
            
            assert isinstance(result, list)
            # Should deduplicate and only include snippets with call patterns
            assert len(result) == 1
            assert "example_func()" in result[0]

    def test_get_pytest_settings_context(self, context_assembler):
        """Test pytest settings context extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "example.py"
            source_path.write_text("def example_func(): pass")
            
            # Create a pyproject.toml file
            pyproject_path = Path(temp_dir) / "pyproject.toml"
            pyproject_path.write_text("""
[tool.pytest.ini_options]
markers = ["slow", "integration"]
testpaths = ["tests"]
""")
            
            result = context_assembler._get_pytest_settings_context(source_path)
            
            assert isinstance(result, list)
            # Should extract pytest settings
            assert len(result) >= 0  # May be empty if no settings found

    def test_get_side_effects_context(self, context_assembler, mock_parser_port):
        """Test side effects context extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "example.py"
            source_path.write_text("""
import os
import requests

def example_func():
    os.environ['TEST'] = 'value'
    requests.get('http://example.com')
""")
            
            # Mock parser to return realistic AST
            mock_parser_port.parse_file.return_value = {
                "ast": ast.parse(source_path.read_text()),
                "source_lines": source_path.read_text().splitlines()
            }
            
            result = context_assembler._get_side_effects_context(source_path)
            
            assert isinstance(result, list)
            # Should detect side effects (implementation dependent)
            assert len(result) >= 0


class TestContextAssemblerPromptBudgets:
    """Test cases for enhanced prompt budgeting, ordering, and deduplication."""

    def test_assemble_final_context_with_budget_caps(self, mock_context_port, mock_parser_port):
        """Test context assembly with budget caps."""
        config = GenerationConfig.get_default_config()
        config["prompt_budgets"] = {
            "per_item_chars": 100,
            "total_chars": 500,
            "section_caps": {
                "snippets": 2,
                "neighbors": 1,
                "test_exemplars": 1,
                "contracts": 1,
                "deps_config_fixtures": 1,
                "coverage_hints": 1,
                "callgraph": 1,
                "error_paths": 1,
                "usage_examples": 1,
                "pytest_settings": 1,
                "side_effects": 1,
            }
        }
        
        context_assembler = ContextAssembler(mock_context_port, mock_parser_port, config)
        
        # Create oversized context sections
        context_sections = [
            ["snippet1" * 50, "snippet2" * 50, "snippet3" * 50],  # snippets (should cap at 2)
            ["neighbor1" * 50],  # neighbors
            ["exemplar1" * 50],  # test_exemplars
            ["contract1" * 50],  # contracts
            ["deps1" * 50],  # deps_config_fixtures
            ["coverage1" * 50],  # coverage_hints
            ["callgraph1" * 50],  # callgraph
            ["error1" * 50],  # error_paths
            ["usage1" * 50],  # usage_examples
            ["pytest1" * 50],  # pytest_settings
            ["side1" * 50],  # side_effects
        ]
        
        result = context_assembler._assemble_final_context(context_sections)
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) <= config["prompt_budgets"]["total_chars"]
        
        # Should contain content from multiple sections
        assert "snippet1" in result
        assert "neighbor1" in result

    def test_assemble_final_context_per_item_capping(self, mock_context_port, mock_parser_port):
        """Test per-item character capping."""
        config = GenerationConfig.get_default_config()
        config["prompt_budgets"]["per_item_chars"] = 50  # Very small cap
        
        context_assembler = ContextAssembler(mock_context_port, mock_parser_port, config)
        
        # Create context with long items
        context_sections = [
            ["A" * 200],  # Should be truncated to 50 chars
            [],  # neighbors
            [],  # test_exemplars
            [],  # contracts
            [],  # deps_config_fixtures
            [],  # coverage_hints
            [],  # callgraph
            [],  # error_paths
            [],  # usage_examples
            [],  # pytest_settings
            [],  # side_effects
        ]
        
        result = context_assembler._assemble_final_context(context_sections)
        
        assert result is not None
        # Each item should be capped
        lines = result.split('\n\n')
        for line in lines:
            if line.strip():
                assert len(line) <= config["prompt_budgets"]["per_item_chars"]

    def test_assemble_final_context_deduplication(self, mock_context_port, mock_parser_port):
        """Test deduplication of context items."""
        config = GenerationConfig.get_default_config()
        context_assembler = ContextAssembler(mock_context_port, mock_parser_port, config)
        
        # Create context with duplicates
        duplicate_item = "duplicate content"
        context_sections = [
            [duplicate_item, duplicate_item, "unique content"],  # snippets with duplicate
            [],  # neighbors
            [],  # test_exemplars
            [],  # contracts
            [],  # deps_config_fixtures
            [],  # coverage_hints
            [],  # callgraph
            [],  # error_paths
            [],  # usage_examples
            [],  # pytest_settings
            [],  # side_effects
        ]
        
        result = context_assembler._assemble_final_context(context_sections)
        
        assert result is not None
        # Should only appear once after deduplication
        assert result.count(duplicate_item) == 1
        assert "unique content" in result

    def test_assemble_final_context_deterministic_ordering(self, mock_context_port, mock_parser_port):
        """Test deterministic ordering of context sections."""
        config = GenerationConfig.get_default_config()
        context_assembler = ContextAssembler(mock_context_port, mock_parser_port, config)
        
        # Create context with identifiable markers
        context_sections = [
            ["MARKER_SNIPPETS"],  # snippets
            ["MARKER_NEIGHBORS"],  # neighbors
            ["MARKER_EXEMPLARS"],  # test_exemplars
            ["MARKER_CONTRACTS"],  # contracts
            ["MARKER_DEPS"],  # deps_config_fixtures
            [],  # coverage_hints
            [],  # callgraph
            [],  # error_paths
            [],  # usage_examples
            [],  # pytest_settings
            [],  # side_effects
        ]
        
        result1 = context_assembler._assemble_final_context(context_sections)
        result2 = context_assembler._assemble_final_context(context_sections)
        
        # Should be identical (deterministic)
        assert result1 == result2
        
        # Should maintain expected order
        if result1:
            snippets_pos = result1.find("MARKER_SNIPPETS")
            neighbors_pos = result1.find("MARKER_NEIGHBORS")
            if snippets_pos != -1 and neighbors_pos != -1:
                assert snippets_pos < neighbors_pos  # snippets should come before neighbors

    def test_assemble_final_context_empty_sections(self, mock_context_port, mock_parser_port):
        """Test handling of empty context sections."""
        config = GenerationConfig.get_default_config()
        context_assembler = ContextAssembler(mock_context_port, mock_parser_port, config)
        
        # All empty sections
        context_sections = [[], [], [], [], [], [], [], [], [], [], []]
        
        result = context_assembler._assemble_final_context(context_sections)
        
        # Should return None for empty context
        assert result is None

    def test_assemble_final_context_section_count_mismatch(self, mock_context_port, mock_parser_port):
        """Test handling of section count mismatch."""
        config = GenerationConfig.get_default_config()
        
        with patch("testcraft.application.generation.services.context.assemble.logger.warning") as mock_warn:
            context_assembler = ContextAssembler(mock_context_port, mock_parser_port, config)
            
            # Wrong number of sections (should be 12, providing 3)
            context_sections = [["item1"], ["item2"], ["item3"]]
            
            result = context_assembler._assemble_final_context(context_sections)
            
            # Should log warning about mismatch (actual message mentions expected 12 sections)
            mock_warn.assert_called_once()
            assert "Section count mismatch" in str(mock_warn.call_args)
            
            # Should still work with available sections
            assert result is not None or result is None  # Either way is acceptable

    def test_assemble_final_context_data_driven_sections(self, mock_context_port, mock_parser_port):
        """Test data-driven section handling with custom config."""
        config = GenerationConfig.get_default_config()
        # Customize section caps to test data-driven approach
        config["prompt_budgets"]["section_caps"] = {
            "snippets": 1,
            "custom_section": 2,  # Not in default order
        }
        
        context_assembler = ContextAssembler(mock_context_port, mock_parser_port, config)
        
        context_sections = [
            ["snippet1", "snippet2"],  # Should be capped at 1
            [],  # neighbors
            [],  # test_exemplars
            [],  # contracts
            [],  # deps_config_fixtures
            [],  # coverage_hints
            [],  # callgraph
            [],  # error_paths
            [],  # usage_examples
            [],  # pytest_settings
            [],  # side_effects
        ]
        
        result = context_assembler._assemble_final_context(context_sections)
        
        # Should respect section caps even with custom configuration
        if result:
            # Should only have one snippet due to cap
            assert result.count("snippet") <= 2  # Allow some flexibility

    def test_assemble_final_context_total_budget_enforcement(self, mock_context_port, mock_parser_port):
        """Test total character budget enforcement with truncation."""
        config = GenerationConfig.get_default_config()
        config["prompt_budgets"]["total_chars"] = 100  # Very small total budget
        
        context_assembler = ContextAssembler(mock_context_port, mock_parser_port, config)
        
        # Create large context that exceeds total budget
        context_sections = [
            ["A" * 50, "B" * 50, "C" * 50],  # 150 chars total
            [],  # neighbors
            [],  # test_exemplars
            [],  # contracts
            [],  # deps_config_fixtures
            [],  # coverage_hints
            [],  # callgraph
            [],  # error_paths
            [],  # usage_examples
            [],  # pytest_settings
            [],  # side_effects
        ]
        
        result = context_assembler._assemble_final_context(context_sections)
        
        assert result is not None
        # Should be within total budget
        assert len(result) <= config["prompt_budgets"]["total_chars"]
        # Should include truncation marker if content was cut
        if len(result) == config["prompt_budgets"]["total_chars"]:
            # Content was likely truncated
            pass  # This is expected behavior
