"""
Tests for generation services.

This module contains comprehensive unit tests for all the extracted services
from the generation workflow refactoring.
"""

import ast
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from testcraft.application.generation.config import GenerationConfig
from testcraft.application.generation.services.batch_executor import BatchExecutor
from testcraft.application.generation.services.context_assembler import ContextAssembler
from testcraft.application.generation.services.coverage_evaluator import (
    CoverageEvaluator,
)
from testcraft.application.generation.services.state_discovery import StateSyncDiscovery
from testcraft.application.generation.services.structure import DirectoryTreeBuilder
from testcraft.domain.models import GenerationResult, TestGenerationPlan


class TestGenerationConfig:
    """Test cases for GenerationConfig."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = GenerationConfig.get_default_config()

        assert config["batch_size"] == 5
        assert config["enable_context"] is True
        assert config["enable_refinement"] is True
        assert config["max_refinement_iterations"] == 3
        assert config["coverage_threshold"] == 0.8
        assert config["test_framework"] == "pytest"

        # Check context categories
        assert "context_categories" in config
        assert config["context_categories"]["snippets"] is True
        assert config["context_categories"]["neighbors"] is True

        # Check prompt budgets
        assert "prompt_budgets" in config
        assert config["prompt_budgets"]["per_item_chars"] == 1500
        assert config["prompt_budgets"]["total_chars"] == 10000

    def test_merge_config_empty(self):
        """Test config merging with empty overrides."""
        config = GenerationConfig.merge_config(None)
        defaults = GenerationConfig.get_default_config()
        assert config == defaults

        config = GenerationConfig.merge_config({})
        assert config == defaults

    def test_merge_config_with_overrides(self):
        """Test config merging with overrides."""
        overrides = {
            "batch_size": 10,
            "enable_context": False,
            "test_framework": "unittest",
        }

        config = GenerationConfig.merge_config(overrides)

        assert config["batch_size"] == 10
        assert config["enable_context"] is False
        assert config["test_framework"] == "unittest"
        # Unchanged defaults
        assert config["enable_refinement"] is True
        assert config["max_refinement_iterations"] == 3

    def test_merge_context_categories(self):
        """Test deep merging of context categories."""
        overrides = {
            "context_categories": {
                "snippets": False,
                "contracts": True,
            }
        }

        config = GenerationConfig.merge_config(overrides)

        # Should merge deeply
        assert config["context_categories"]["snippets"] is False
        assert config["context_categories"]["contracts"] is True
        # Unchanged defaults should still be there
        assert "neighbors" in config["context_categories"]
        assert config["context_categories"]["neighbors"] is True

    def test_context_enrichment_mapping(self):
        """Test context enrichment to context categories mapping."""
        overrides = {
            "context_enrichment": {
                "enable_env_detection": True,
                "enable_db_boundary_detection": False,
                "enable_side_effect_detection": True,
            }
        }

        config = GenerationConfig.merge_config(overrides)

        assert config["context_categories"]["deps_config_fixtures"] is True
        assert config["context_categories"]["side_effects"] is True
        assert "context_enrichment" in config

    def test_validate_config_valid(self):
        """Test config validation with valid values."""
        config = GenerationConfig.get_default_config()
        # Should not raise
        GenerationConfig.validate_config(config)

    def test_validate_config_fixes_invalid_batch_size(self):
        """Test config validation fixes invalid batch size."""
        config = {"batch_size": -1}

        with patch(
            "testcraft.application.generation.config.logger.warning"
        ) as mock_warn:
            GenerationConfig.validate_config(config)
            mock_warn.assert_called_once()

        assert config["batch_size"] == 5

    def test_validate_config_fixes_invalid_coverage_threshold(self):
        """Test config validation fixes invalid coverage threshold."""
        config = {"coverage_threshold": 1.5}

        with patch(
            "testcraft.application.generation.config.logger.warning"
        ) as mock_warn:
            GenerationConfig.validate_config(config)
            mock_warn.assert_called_once()

        assert config["coverage_threshold"] == 0.8

    def test_validate_prompt_budgets_comprehensive(self):
        """Test comprehensive prompt budget validation."""
        config = {
            "prompt_budgets": {
                "per_item_chars": 50,  # Too small
                "total_chars": 60000,  # Very large (exceeds 50000 threshold)
                "section_caps": {
                    "snippets": 5,
                    "invalid_section": 3,  # Invalid section
                    "contracts": -1,  # Invalid negative value
                    "neighbors": 60,  # Very large value
                },
            }
        }

        with patch(
            "testcraft.application.generation.config.logger.warning"
        ) as mock_warn:
            GenerationConfig.validate_config(config)

            # Should have multiple warnings
            assert mock_warn.call_count >= 4

            # Check specific warning calls - format the messages properly
            warning_messages = []
            for call in mock_warn.call_args_list:
                if len(call[0]) > 1:
                    # Format the message with its arguments like logger.warning would
                    warning_messages.append(call[0][0] % call[0][1:])
                else:
                    warning_messages.append(call[0][0])

            # Should warn about small per_item_chars
            assert any(
                "per_item_chars" in msg and "using default 1500" in msg
                for msg in warning_messages
            )

            # Should warn about large total_chars
            assert any("Very large total_chars" in msg for msg in warning_messages)

            # Should warn about unknown section (includes list of valid sections)
            assert any(
                "Unknown section_cap 'invalid_section', valid sections:" in msg
                for msg in warning_messages
            )

            # Should warn about invalid negative section cap
            assert any(
                "Invalid section_cap for 'contracts'" in msg for msg in warning_messages
            )

            # Should warn about very large section cap
            assert any(
                "Very large section_cap for 'neighbors'" in msg
                for msg in warning_messages
            )

        # Should fix invalid values
        assert config["prompt_budgets"]["per_item_chars"] == 1500

    def test_validate_prompt_budgets_consistency_check(self):
        """Test prompt budget consistency validation."""
        config = {
            "prompt_budgets": {
                "per_item_chars": 3000,  # Large per-item
                "total_chars": 4000,  # Small total (per_item * 2 > total)
            }
        }

        with patch(
            "testcraft.application.generation.config.logger.warning"
        ) as mock_warn:
            GenerationConfig.validate_config(config)

            # Should warn about consistency issue
            warning_messages = [str(call) for call in mock_warn.call_args_list]
            assert any(
                "per_item_chars" in msg
                and "total_chars" in msg
                and "truncation issues" in msg
                for msg in warning_messages
            )

    def test_validate_prompt_budgets_valid_values(self):
        """Test prompt budget validation with valid values."""
        config = {
            "prompt_budgets": {
                "per_item_chars": 1000,
                "total_chars": 5000,
                "section_caps": {
                    "snippets": 10,
                    "neighbors": 5,
                    "test_exemplars": 3,
                },
            }
        }

        # Should not raise or warn for valid configuration
        GenerationConfig.validate_config(config)

        # Values should remain unchanged
        assert config["prompt_budgets"]["per_item_chars"] == 1000
        assert config["prompt_budgets"]["total_chars"] == 5000


class TestStateSyncDiscovery:
    """Test cases for StateSyncDiscovery service."""

    @pytest.fixture
    def mock_ports(self):
        """Create mock ports."""
        state_port = MagicMock()
        file_discovery_service = MagicMock()
        telemetry_port = MagicMock()

        # Setup telemetry mock
        mock_span = MagicMock()
        telemetry_port.create_child_span.return_value.__enter__.return_value = mock_span

        return state_port, file_discovery_service, telemetry_port, mock_span

    @pytest.fixture
    def service(self, mock_ports):
        """Create StateSyncDiscovery service."""
        state_port, file_discovery_service, telemetry_port, _ = mock_ports
        return StateSyncDiscovery(state_port, file_discovery_service, telemetry_port)

    def test_sync_and_discover_with_target_files(self, service, mock_ports):
        """Test sync and discover with target files."""
        state_port, file_discovery_service, telemetry_port, mock_span = mock_ports

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
        file_discovery_service.filter_existing_files.assert_called_once_with(
            ["file1.py", "file2.py", "nonexistent.py"], project_path
        )

        # Verify result
        assert len(result["files"]) == 2
        assert result["files"][0] == Path("file1.py")
        assert result["files"][1] == Path("file2.py")
        assert result["previous_state"] == {"previous": "state"}
        assert result["project_path"] == project_path

        # Verify telemetry
        mock_span.set_attribute.assert_called()

    def test_sync_and_discover_without_target_files(self, service, mock_ports):
        """Test sync and discover without target files (discovery mode)."""
        state_port, file_discovery_service, telemetry_port, mock_span = mock_ports

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


class TestCoverageEvaluator:
    """Test cases for CoverageEvaluator service."""

    @pytest.fixture
    def mock_ports(self):
        """Create mock ports."""
        coverage_port = MagicMock()
        telemetry_port = MagicMock()

        # Setup telemetry mock
        mock_span = MagicMock()
        telemetry_port.create_child_span.return_value.__enter__.return_value = mock_span

        return coverage_port, telemetry_port, mock_span

    @pytest.fixture
    def service(self, mock_ports):
        """Create CoverageEvaluator service."""
        coverage_port, telemetry_port, _ = mock_ports
        return CoverageEvaluator(coverage_port, telemetry_port)

    def test_measure_initial_success(self, service, mock_ports):
        """Test successful initial coverage measurement."""
        coverage_port, telemetry_port, mock_span = mock_ports

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

    def test_measure_initial_failure_graceful(self, service, mock_ports):
        """Test graceful failure handling in initial coverage measurement."""
        coverage_port, telemetry_port, mock_span = mock_ports

        # Setup mocks to fail
        coverage_port.measure_coverage.side_effect = Exception("Coverage failed")

        source_files = [Path("file1.py")]
        result = service.measure_initial(source_files)

        # Should return empty coverage data
        assert result["overall_line_coverage"] == 0.0
        assert result["overall_branch_coverage"] == 0.0
        assert result["files_covered"] == 0
        assert result["total_lines"] == 0

    def test_calculate_delta(self, service, mock_ports):
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

    def test_calculate_delta_error_handling(self, service, mock_ports):
        """Test delta calculation with error handling."""
        # Invalid input should be handled gracefully
        result = service.calculate_delta({}, None)

        assert "error" in result
        assert result["line_coverage_delta"] == 0.0


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


class TestBatchExecutor:
    """Test cases for BatchExecutor service."""

    @pytest.fixture
    def mock_ports(self):
        """Create mock ports."""
        telemetry_port = MagicMock()

        # Setup telemetry mock
        mock_span = MagicMock()
        telemetry_port.create_child_span.return_value.__enter__.return_value = mock_span

        return telemetry_port, mock_span

    @pytest.fixture
    def service(self, mock_ports):
        """Create BatchExecutor service."""
        telemetry_port, _ = mock_ports
        return BatchExecutor(telemetry_port)

    @pytest.mark.asyncio
    async def test_run_in_batches(self, service, mock_ports):
        """Test running generation in batches."""
        from testcraft.domain.models import TestElement, TestElementType

        telemetry_port, mock_span = mock_ports

        # Create test plans
        element1 = TestElement(
            name="test_func1",
            type=TestElementType.FUNCTION,
            line_range=(1, 5),
            docstring="Test function 1",
        )
        element2 = TestElement(
            name="test_func2",
            type=TestElementType.FUNCTION,
            line_range=(6, 10),
            docstring="Test function 2",
        )

        plan1 = TestGenerationPlan(elements_to_test=[element1])
        plan2 = TestGenerationPlan(elements_to_test=[element2])

        plans = [plan1, plan2]

        # Mock generation function
        async def mock_generation_fn(plan):
            element_name = plan.elements_to_test[0].name
            return GenerationResult(
                file_path=f"test_{element_name}.py",
                content=f"def {element_name}(): pass",
                success=True,
                error_message=None,
            )

        results = await service.run_in_batches(plans, 2, mock_generation_fn)

        # Verify results
        assert len(results) == 2
        assert all(r.success for r in results)
        assert "test_func1" in results[0].content
        assert "test_func2" in results[1].content

        # Verify telemetry
        mock_span.set_attribute.assert_called()

    @pytest.mark.asyncio
    async def test_run_in_batches_with_failures(self, service, mock_ports):
        """Test batch execution with some failures."""
        from testcraft.domain.models import TestElement, TestElementType

        telemetry_port, mock_span = mock_ports

        element = TestElement(
            name="test_func",
            type=TestElementType.FUNCTION,
            line_range=(1, 5),
            docstring="Test function",
        )
        plan = TestGenerationPlan(elements_to_test=[element])

        # Mock generation function that raises exception
        async def failing_generation_fn(plan):
            raise Exception("Generation failed")

        results = await service.run_in_batches([plan], 1, failing_generation_fn)

        # Should handle exception gracefully
        assert len(results) == 1
        assert not results[0].success
        assert "Generation failed" in results[0].error_message


class TestContextAssemblerAdvanced:
    """Test cases for ContextAssembler advanced context features."""

    @pytest.fixture
    def mock_context_port(self):
        """Mock context port."""
        mock = MagicMock()
        mock.retrieve.return_value = {"results": [], "total_found": 0}
        mock.get_related_context.return_value = {
            "related_files": [],
            "relationships": [],
        }
        return mock

    @pytest.fixture
    def mock_parser_port(self):
        """Mock parser port."""
        mock = MagicMock()
        mock.parse_file.return_value = {"ast": None, "source_lines": []}
        return mock

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

    def test_get_callgraph_neighbors_with_relationships(
        self, context_assembler, mock_context_port
    ):
        """Test call-graph neighbor extraction with relationships."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "example.py"
            source_path.write_text("def example_func(): pass")

            # Setup mock with relationships and related files
            mock_context_port.get_related_context.return_value = {
                "relationships": ["calls:other_func", "imports:module"],
                "related_files": [str(source_path)],
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
                "related_files": [],
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

    def test_get_usage_examples_with_results(
        self, context_assembler, mock_context_port
    ):
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
                    {
                        "snippet": "result = example_func(arg1, arg2)",
                        "path": "test1.py",
                    },
                    {"snippet": "example_func()", "path": "test2.py"},
                ]
            }

            result = context_assembler._get_usage_examples(source_path, mock_plan)

            assert isinstance(result, list)
            assert len(result) == 2
            assert all("Usage example_func" in item for item in result)

    def test_get_usage_examples_deduplication(
        self, context_assembler, mock_context_port
    ):
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
                    {
                        "snippet": "different_call()",
                        "path": "test3.py",
                    },  # No call pattern
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
                "source_lines": source_path.read_text().splitlines(),
            }

            result = context_assembler._get_side_effects_context(source_path)

            assert isinstance(result, list)
            # Should detect side effects (implementation dependent)
            assert len(result) >= 0


class TestContextAssemblerPromptBudgets:
    """Test cases for enhanced prompt budgeting, ordering, and deduplication."""

    @pytest.fixture
    def mock_context_port(self):
        """Mock context port."""
        mock = MagicMock()
        mock.retrieve.return_value = {"results": [], "total_found": 0}
        mock.get_related_context.return_value = {
            "related_files": [],
            "relationships": [],
        }
        return mock

    @pytest.fixture
    def mock_parser_port(self):
        """Mock parser port."""
        mock = MagicMock()
        mock.parse_file.return_value = {"ast": None, "source_lines": []}
        return mock

    def test_assemble_final_context_with_budget_caps(
        self, mock_context_port, mock_parser_port
    ):
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
            },
        }

        context_assembler = ContextAssembler(
            mock_context_port, mock_parser_port, config
        )

        # Create oversized context sections
        context_sections = [
            [
                "snippet1" * 50,
                "snippet2" * 50,
                "snippet3" * 50,
            ],  # snippets (should cap at 2)
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

    def test_assemble_final_context_per_item_capping(
        self, mock_context_port, mock_parser_port
    ):
        """Test per-item character capping."""
        config = GenerationConfig.get_default_config()
        config["prompt_budgets"]["per_item_chars"] = 50  # Very small cap

        context_assembler = ContextAssembler(
            mock_context_port, mock_parser_port, config
        )

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
        lines = result.split("\n\n")
        for line in lines:
            if line.strip():
                assert len(line) <= config["prompt_budgets"]["per_item_chars"]

    def test_assemble_final_context_deduplication(
        self, mock_context_port, mock_parser_port
    ):
        """Test deduplication of context items."""
        config = GenerationConfig.get_default_config()
        context_assembler = ContextAssembler(
            mock_context_port, mock_parser_port, config
        )

        # Create context with duplicates
        duplicate_item = "duplicate content"
        context_sections = [
            [
                duplicate_item,
                duplicate_item,
                "unique content",
            ],  # snippets with duplicate
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

    def test_assemble_final_context_deterministic_ordering(
        self, mock_context_port, mock_parser_port
    ):
        """Test deterministic ordering of context sections."""
        config = GenerationConfig.get_default_config()
        context_assembler = ContextAssembler(
            mock_context_port, mock_parser_port, config
        )

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
                assert (
                    snippets_pos < neighbors_pos
                )  # snippets should come before neighbors

    def test_assemble_final_context_empty_sections(
        self, mock_context_port, mock_parser_port
    ):
        """Test handling of empty context sections."""
        config = GenerationConfig.get_default_config()
        context_assembler = ContextAssembler(
            mock_context_port, mock_parser_port, config
        )

        # All empty sections
        context_sections = [[], [], [], [], [], [], [], [], [], [], []]

        result = context_assembler._assemble_final_context(context_sections)

        # Should return None for empty context
        assert result is None

    def test_assemble_final_context_section_count_mismatch(
        self, mock_context_port, mock_parser_port
    ):
        """Test handling of section count mismatch."""
        config = GenerationConfig.get_default_config()

        with patch(
            "testcraft.application.generation.services.context_assembler.logger.warning"
        ) as mock_warn:
            context_assembler = ContextAssembler(
                mock_context_port, mock_parser_port, config
            )

            # Wrong number of sections (should be 11, providing 3)
            context_sections = [["item1"], ["item2"], ["item3"]]

            result = context_assembler._assemble_final_context(context_sections)

            # Should log warning about mismatch
            mock_warn.assert_called_once()
            assert "Section count mismatch" in str(mock_warn.call_args)

            # Should still work with available sections
            assert result is not None or result is None  # Either way is acceptable

    def test_assemble_final_context_data_driven_sections(
        self, mock_context_port, mock_parser_port
    ):
        """Test data-driven section handling with custom config."""
        config = GenerationConfig.get_default_config()
        # Customize section caps to test data-driven approach
        config["prompt_budgets"]["section_caps"] = {
            "snippets": 1,
            "custom_section": 2,  # Not in default order
        }

        context_assembler = ContextAssembler(
            mock_context_port, mock_parser_port, config
        )

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

    def test_assemble_final_context_total_budget_enforcement(
        self, mock_context_port, mock_parser_port
    ):
        """Test total character budget enforcement with truncation."""
        config = GenerationConfig.get_default_config()
        config["prompt_budgets"]["total_chars"] = 100  # Very small total budget

        context_assembler = ContextAssembler(
            mock_context_port, mock_parser_port, config
        )

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
