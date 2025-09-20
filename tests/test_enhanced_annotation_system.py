"""
Test suite for enhanced annotation system functionality.

Verifies that the new specific test identification and actionable
fix instruction generation works correctly.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from testcraft.application.generation.services.pytest_refiner import PytestRefiner
from testcraft.config.models import RefineConfig


@pytest.fixture
def refiner_with_enhanced_annotations():
    """Create a PytestRefiner with enhanced annotations enabled."""
    config = RefineConfig(
        enable=True,
        annotate_failed_tests=True,
        annotation_placement="top",
        annotation_include_failure_excerpt=True,
        annotation_max_failure_chars=600,
        annotation_style="docstring",
        include_llm_fix_instructions=True,
    )

    mock_refine_port = MagicMock()
    mock_telemetry_port = MagicMock()
    mock_writer_port = MagicMock()

    return PytestRefiner(
        refine_port=mock_refine_port,
        telemetry_port=mock_telemetry_port,
        executor=None,
        max_concurrent_refines=1,
        backoff_sec=1.0,
        writer_port=mock_writer_port,
    )


class TestFailingTestExtraction:
    """Test extraction of specific failing test names from pytest output."""

    def test_extract_failing_tests_from_pytest_output(
        self, refiner_with_enhanced_annotations
    ):
        """Test extraction of specific failing test names."""
        pytest_output = """
FAILED tests/test_weatherscheduler.py::test_collect_weather_data_happy_path - AttributeError: 'MagicMock' object has no attribute 'temperature'
FAILED tests/test_weatherscheduler.py::test_collect_weather_data_city_creation - ImportError: No module named 'weather_collector.models'
FAILED tests/test_weatherscheduler.py::test_collect_weather_data_partial_failure - AssertionError: assert call('Failed')

_______ test_collect_weather_data_happy_path _______
AttributeError: 'MagicMock' object has no attribute 'temperature'
        """

        failing_tests = (
            refiner_with_enhanced_annotations._extract_failing_tests_from_output(
                pytest_output
            )
        )

        assert len(failing_tests) >= 3
        assert (
            "tests/test_weatherscheduler.py::test_collect_weather_data_happy_path"
            in failing_tests
        )
        assert (
            "tests/test_weatherscheduler.py::test_collect_weather_data_city_creation"
            in failing_tests
        )
        assert (
            "tests/test_weatherscheduler.py::test_collect_weather_data_partial_failure"
            in failing_tests
        )

    def test_extract_failing_tests_handles_empty_output(
        self, refiner_with_enhanced_annotations
    ):
        """Test handling of empty or None output."""
        assert (
            refiner_with_enhanced_annotations._extract_failing_tests_from_output("")
            == []
        )
        assert (
            refiner_with_enhanced_annotations._extract_failing_tests_from_output(None)
            == []
        )

    def test_extract_failing_tests_handles_no_failures(
        self, refiner_with_enhanced_annotations
    ):
        """Test handling of output with no failures."""
        pytest_output = """
=============================== test session starts ===============================
collected 5 items

tests/test_example.py::test_success PASSED                                    [100%]

=============================== 1 passed in 0.02s ===============================
        """

        failing_tests = (
            refiner_with_enhanced_annotations._extract_failing_tests_from_output(
                pytest_output
            )
        )
        assert failing_tests == []


class TestFailureContextExtraction:
    """Test extraction of failure context from pytest output."""

    def test_extract_failure_context_identifies_error_types(
        self, refiner_with_enhanced_annotations
    ):
        """Test identification of different error types."""
        pytest_output = """
AttributeError: 'MagicMock' object has no attribute 'temperature'
ImportError: No module named 'weather_collector.models'
AssertionError: assert call('Failed') not in mock_console.print.call_args_list
SyntaxError: invalid syntax (test_file.py, line 42)
        """

        context = refiner_with_enhanced_annotations._extract_failure_context(
            pytest_output
        )

        assert len(context["error_types"]) == 4
        assert any("AttributeError" in error for error in context["error_types"])
        assert any("ImportError" in error for error in context["error_types"])
        assert any("AssertionError" in error for error in context["error_types"])
        assert any("SyntaxError" in error for error in context["error_types"])

    def test_extract_failure_context_captures_import_errors(
        self, refiner_with_enhanced_annotations
    ):
        """Test capture of specific import error details."""
        pytest_output = """
ImportError: No module named 'weather_collector.models'
ModuleNotFoundError: No module named 'missing_dependency'
        """

        context = refiner_with_enhanced_annotations._extract_failure_context(
            pytest_output
        )

        assert len(context["import_errors"]) == 2
        assert "No module named 'weather_collector.models'" in context["import_errors"]
        assert "No module named 'missing_dependency'" in context["import_errors"]

    def test_extract_failure_context_handles_assertion_details(
        self, refiner_with_enhanced_annotations
    ):
        """Test extraction of assertion failure details."""
        pytest_output = """
    assert any("Failed to collect data for TestCity" in str(call_args[0][0]) for call_args in mock_console.print.call_args_list)
AssertionError: assert False
        """

        context = refiner_with_enhanced_annotations._extract_failure_context(
            pytest_output
        )

        assert "assert any(" in context["assertion_details"]
        assert len(context["traceback_highlights"]) > 0


class TestEnhancedFixInstructions:
    """Test generation of enhanced, specific fix instructions."""

    def test_generate_enhanced_fix_instructions_with_failing_tests(
        self, refiner_with_enhanced_annotations
    ):
        """Test generation of instructions with specific failing tests."""
        failing_tests = [
            "tests/test_weatherscheduler.py::test_collect_weather_data_happy_path",
            "tests/test_weatherscheduler.py::test_collect_weather_data_city_creation",
        ]
        failure_context = {
            "error_types": [
                "AttributeError: 'MagicMock' object has no attribute 'temperature'"
            ],
            "import_errors": ["No module named 'weather_collector.models'"],
            "assertion_details": "",
            "key_messages": [],
            "traceback_highlights": [],
        }
        original_instructions = "Review mock configurations"

        instructions = (
            refiner_with_enhanced_annotations._generate_enhanced_fix_instructions(
                failing_tests, failure_context, original_instructions
            )
        )

        assert "🔍 SPECIFIC FAILING TESTS:" in instructions
        assert "test_collect_weather_data_happy_path" in instructions
        assert "test_collect_weather_data_city_creation" in instructions
        assert "❌ ERROR ANALYSIS:" in instructions
        assert "AttributeError" in instructions
        assert "📦 IMPORT FIXES NEEDED:" in instructions
        assert "weather_collector.models" in instructions
        assert "📋 STEP-BY-STEP FIX PLAN:" in instructions

    def test_generate_enhanced_fix_instructions_with_import_errors(
        self, refiner_with_enhanced_annotations
    ):
        """Test generation of specific import fix instructions."""
        failing_tests = ["tests/test_example.py::test_import_issue"]
        failure_context = {
            "error_types": ["ImportError: No module named 'missing_module'"],
            "import_errors": ["No module named 'missing_module'"],
            "assertion_details": "",
            "key_messages": [],
            "traceback_highlights": [],
        }

        instructions = (
            refiner_with_enhanced_annotations._generate_enhanced_fix_instructions(
                failing_tests, failure_context, ""
            )
        )

        assert "📦 IMPORT FIXES NEEDED:" in instructions
        assert "Missing module 'missing_module'" in instructions
        assert "with patch('missing_module')" in instructions
        assert "Fix import issues FIRST" in instructions

    def test_generate_enhanced_fix_instructions_with_assertion_errors(
        self, refiner_with_enhanced_annotations
    ):
        """Test generation of assertion-specific guidance."""
        failing_tests = ["tests/test_example.py::test_assertion_issue"]
        failure_context = {
            "error_types": ["AssertionError: assert False"],
            "import_errors": [],
            "assertion_details": "assert call('Expected') in mock_object.call_args_list",
            "key_messages": [],
            "traceback_highlights": [
                "mock_object.some_method('Actual')",
                "assert call('Expected')",
            ],
        }

        instructions = (
            refiner_with_enhanced_annotations._generate_enhanced_fix_instructions(
                failing_tests, failure_context, ""
            )
        )

        assert "⚖️  ASSERTION ANALYSIS:" in instructions
        assert "assert call('Expected')" in instructions
        assert "Review expected vs actual values" in instructions
        assert "Debug assertion" in instructions

    def test_generate_enhanced_fix_instructions_handles_empty_context(
        self, refiner_with_enhanced_annotations
    ):
        """Test handling of empty failure context."""
        instructions = (
            refiner_with_enhanced_annotations._generate_enhanced_fix_instructions(
                [], {}, ""
            )
        )

        # Should still generate some basic guidance
        assert "📋 STEP-BY-STEP FIX PLAN:" in instructions
        assert len(instructions) > 100  # Should be substantial


class TestIntegratedAnnotationSystem:
    """Test the complete integrated annotation system."""

    @pytest.mark.asyncio
    async def test_annotate_failed_test_uses_enhanced_instructions(
        self, refiner_with_enhanced_annotations
    ):
        """Test that the annotation system uses enhanced instructions."""

        # Create a temporary test file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write("""
def test_example():
    assert True
""")
            tmp_file_path = Path(tmp_file.name)

        try:
            # Mock pytest output with specific failures
            failure_output = """
FAILED tests/test_example.py::test_example - AttributeError: 'MagicMock' object has no attribute 'value'
ImportError: No module named 'test_module'

_______ test_example _______
AttributeError: 'MagicMock' object has no attribute 'value'
    test_example.py, line 15, in test_example
        result = mock_object.value
            """

            fix_instructions = "Configure mock attributes properly"

            # Mock the writer port
            mock_writer = MagicMock()
            refiner_with_enhanced_annotations._writer = mock_writer

            # Call the annotation method
            await refiner_with_enhanced_annotations._annotate_failed_test(
                test_file=tmp_file_path,
                reason_status="max_iterations_exceeded",
                iterations=3,
                failure_output=failure_output,
                fix_instructions=fix_instructions,
                extra={"active_import_path": "test.module"},
            )

            # Verify writer was called
            mock_writer.write_file.assert_called_once()

            # Get the written content
            written_content = mock_writer.write_file.call_args[1]["content"]

            # Verify enhanced features are present
            assert "🔍 SPECIFIC FAILING TESTS:" in written_content
            assert "tests/test_example.py::test_example" in written_content
            assert "❌ ERROR ANALYSIS:" in written_content
            assert "AttributeError" in written_content
            assert "📦 IMPORT FIXES NEEDED:" in written_content
            assert "test_module" in written_content
            assert "📋 STEP-BY-STEP FIX PLAN:" in written_content
            assert "ENHANCED FIX GUIDE:" in written_content

        finally:
            # Clean up
            if tmp_file_path.exists():
                tmp_file_path.unlink()

    @pytest.mark.asyncio
    async def test_annotation_includes_specific_pytest_command(
        self, refiner_with_enhanced_annotations
    ):
        """Test that annotation includes specific pytest command for first failing test."""

        # Create temporary test file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write("def test_example(): pass")
            tmp_file_path = Path(tmp_file.name)

        try:
            failure_output = """
FAILED tests/specific_test.py::test_specific_function - ImportError: No module named 'xyz'
            """

            mock_writer = MagicMock()
            refiner_with_enhanced_annotations._writer = mock_writer

            await refiner_with_enhanced_annotations._annotate_failed_test(
                test_file=tmp_file_path,
                reason_status="unrefinable",
                iterations=1,
                failure_output=failure_output,
                fix_instructions="",
                extra={},
            )

            written_content = mock_writer.write_file.call_args[1]["content"]

            # Should include specific test command
            assert (
                "pytest tests/specific_test.py::test_specific_function -vv --tb=short"
                in written_content
            )

        finally:
            if tmp_file_path.exists():
                tmp_file_path.unlink()
