"""
Tests for ImportResolver integration with ContextAssembler.

Verifies that ContextAssembler properly integrates ImportResolver to:
- Call ImportResolver to generate import_map
- Include import_map in returned context data
- Include canonical import lines in enriched context string
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from testcraft.application.generation.services.context_assembler import ContextAssembler
from testcraft.application.generation.services.import_resolver import (
    ImportMap,
    ImportResolver,
)
from testcraft.domain.models import TestElement, TestElementType, TestGenerationPlan


class TestContextAssemblerImportIntegration:
    """Tests for ImportResolver integration with ContextAssembler."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = None
        self.mock_context_port = MagicMock()
        self.mock_parser_port = MagicMock()
        self.mock_import_resolver = Mock(spec=ImportResolver)

        # Default config
        self.config = {
            "enable_context": True,
            "context_enrichment": {
                "enable_usage_examples": False,  # Simplified for testing
                "enable_env_detection": False,
                "enable_comprehensive_fixtures": False,
            },
            "prompt_budgets": {
                "per_item_chars": 600,
                "total_chars": 4000,
                "section_caps": {},
            },
        }

        self.context_assembler = ContextAssembler(
            context_port=self.mock_context_port,
            parser_port=self.mock_parser_port,
            config=self.config,
            import_resolver=self.mock_import_resolver,
        )

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir:
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_temp_file(self, content: str = "") -> Path:
        """Create a temporary file for testing."""
        if not self.temp_dir:
            self.temp_dir = Path(tempfile.mkdtemp())

        test_file = self.temp_dir / "test_module.py"
        test_file.write_text(
            content or "def example_function(): pass", encoding="utf-8"
        )
        return test_file

    def test_context_for_generation_includes_import_map(self):
        """Test that context_for_generation includes import_map in response."""
        # Arrange
        source_path = self.create_temp_file()
        expected_import_map = ImportMap(
            target_import="import test_module as _under_test",
            sys_path_roots=[str(self.temp_dir)],
            needs_bootstrap=False,
            bootstrap_conftest="",
        )

        self.mock_import_resolver.resolve.return_value = expected_import_map
        self.mock_context_port.retrieve.return_value = {"results": []}

        plan = TestGenerationPlan(
            file_path=str(source_path),
            elements_to_test=[
                TestElement(
                    name="example_function",
                    type=TestElementType.FUNCTION,
                    line_range=(1, 1),
                )
            ],
        )

        # Act
        result = self.context_assembler.context_for_generation(plan, source_path)

        # Assert
        assert result is not None, "Should return context result"
        assert isinstance(result, dict), "Should return dictionary"
        assert "import_map" in result, "Should include import_map in result"
        assert result["import_map"] == expected_import_map, (
            "Should include correct import_map"
        )

        # Verify ImportResolver was called
        self.mock_import_resolver.resolve.assert_called_once_with(source_path)

    def test_context_for_generation_includes_canonical_import_in_context_string(self):
        """Test that canonical import is included in the enriched context string."""
        # Arrange
        source_path = self.create_temp_file()
        expected_import_map = ImportMap(
            target_import="import test_module as _under_test",
            sys_path_roots=[str(self.temp_dir)],
            needs_bootstrap=True,
            bootstrap_conftest="import sys\nsys.path.insert(0, '.')",
        )

        self.mock_import_resolver.resolve.return_value = expected_import_map
        self.mock_context_port.retrieve.return_value = {"results": []}

        # Mock enhanced context builder to return a simple context
        def mock_format_for_llm(enriched_context):
            # Simulate format_for_llm processing the enriched context
            if enriched_context and "context" in enriched_context:
                return enriched_context["context"]
            return "# Base context"

        with patch.object(
            self.context_assembler._enhanced_context_builder,
            "build_enriched_context",
            return_value={"context": "# Base context"},
        ):
            with patch.object(
                self.context_assembler._enhanced_context_builder,
                "format_for_llm",
                side_effect=mock_format_for_llm,
            ):
                plan = TestGenerationPlan(
                    file_path=str(source_path),
                    elements_to_test=[
                        TestElement(
                            name="example_function",
                            type=TestElementType.FUNCTION,
                            line_range=(1, 1),
                        )
                    ],
                )

                # Act
                result = self.context_assembler.context_for_generation(
                    plan, source_path
                )

                # Assert
                assert result is not None, "Should return context result"
                assert "context" in result, "Should include context string"

                context_string = result["context"]
                assert (
                    "# Canonical import: import test_module as _under_test"
                    in context_string
                )
                assert f"# Sys.path roots: ['{self.temp_dir}']" in context_string
                assert "# Bootstrap: conftest.py setup required" in context_string
                assert "# Bootstrap content available" in context_string

    def test_context_for_refinement_includes_import_map(self):
        """Test that context_for_refinement includes import_map in response."""
        # Arrange
        test_file = self.create_temp_file("def test_example(): pass")
        expected_import_map = ImportMap(
            target_import="import test_module as _under_test",
            sys_path_roots=[str(self.temp_dir)],
            needs_bootstrap=False,
            bootstrap_conftest="",
        )

        self.mock_import_resolver.resolve.return_value = expected_import_map
        self.mock_parser_port.analyze_dependencies.return_value = {
            "imports": [],
            "internal_deps": [],
        }

        # Act
        result = self.context_assembler.context_for_refinement(
            test_file, "def test_example(): pass"
        )

        # Assert
        assert result is not None, "Should return context result"
        assert isinstance(result, dict), "Should return dictionary"
        assert "import_map" in result, "Should include import_map in result"
        assert result["import_map"] == expected_import_map, (
            "Should include correct import_map"
        )

        # Verify ImportResolver was called
        self.mock_import_resolver.resolve.assert_called_once_with(test_file)

    def test_import_resolver_failure_handled_gracefully(self):
        """Test that ImportResolver failures are handled gracefully."""
        # Arrange
        source_path = self.create_temp_file()

        self.mock_import_resolver.resolve.side_effect = Exception(
            "Import resolution failed"
        )
        self.mock_context_port.retrieve.return_value = {"results": []}

        plan = TestGenerationPlan(
            file_path=str(source_path),
            elements_to_test=[
                TestElement(
                    name="example_function",
                    type=TestElementType.FUNCTION,
                    line_range=(1, 1),
                )
            ],
        )

        # Act
        result = self.context_assembler.context_for_generation(plan, source_path)

        # Assert
        # Should still return result, but without import_map
        if result:  # May be None if no other context is available
            assert "import_map" not in result or result["import_map"] is None

        # Verify ImportResolver was called
        self.mock_import_resolver.resolve.assert_called_once_with(source_path)

    def test_no_source_path_skips_import_resolution(self):
        """Test that missing source_path skips import resolution."""
        # Arrange
        plan = TestGenerationPlan(
            file_path="unknown",
            elements_to_test=[
                TestElement(
                    name="example_function",
                    type=TestElementType.FUNCTION,
                    line_range=(1, 1),
                )
            ],
        )

        # Act
        result = self.context_assembler.context_for_generation(plan, source_path=None)

        # Assert
        # Should not call ImportResolver
        self.mock_import_resolver.resolve.assert_not_called()

        # Result should not contain import_map or it should be None
        if result:
            assert "import_map" not in result or result["import_map"] is None

    def test_backward_compatibility_with_default_import_resolver(self):
        """Test that ContextAssembler works with default ImportResolver when none provided."""
        # Arrange - create ContextAssembler without mock ImportResolver
        context_assembler = ContextAssembler(
            context_port=self.mock_context_port,
            parser_port=self.mock_parser_port,
            config=self.config,
            # import_resolver=None  # Should use default
        )

        source_path = self.create_temp_file()
        self.mock_context_port.retrieve.return_value = {"results": []}

        plan = TestGenerationPlan(
            file_path=str(source_path),
            elements_to_test=[
                TestElement(
                    name="example_function",
                    type=TestElementType.FUNCTION,
                    line_range=(1, 1),
                )
            ],
        )

        # Act - should not raise exception
        result = context_assembler.context_for_generation(plan, source_path)

        # Assert - should return result (may contain import_map or not, depending on ImportResolver success)
        # The key point is that it doesn't crash
        assert result is None or isinstance(result, dict), (
            "Should return dict or None without crashing"
        )
