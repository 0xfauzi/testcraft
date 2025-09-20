"""
Tests for failed refinement annotation behavior.

Validates the feature that annotates test files with fix instructions
when refinement fails, ensuring proper configuration handling and annotation content.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from testcraft.application.generation.services.pytest_refiner import PytestRefiner
from testcraft.config.models import RefineConfig
from testcraft.ports.refine_port import RefinePort
from testcraft.ports.telemetry_port import TelemetryPort
from testcraft.ports.writer_port import WriterPort


@pytest.fixture
def temp_test_file():
    """Create a temporary test file for annotation tests."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
def test_example():
    assert 1 == 2  # Failing test
""")
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_refine_port():
    """Mock RefinePort for testing."""
    return Mock(spec=RefinePort)


@pytest.fixture
def mock_telemetry_port():
    """Mock TelemetryPort for testing."""
    mock_port = Mock(spec=TelemetryPort)
    mock_span = Mock()
    mock_span.set_attribute = Mock()
    mock_port.create_child_span.return_value.__enter__.return_value = mock_span
    return mock_port


@pytest.fixture
def mock_writer_port():
    """Mock WriterPort for testing."""
    return Mock(spec=WriterPort)


@pytest.fixture
def refiner_with_annotation_enabled(
    mock_refine_port, mock_telemetry_port, mock_writer_port
):
    """PytestRefiner with annotation enabled."""
    config = RefineConfig(
        annotate_failed_tests=True,
        annotation_style="docstring",
        annotation_placement="top",
        annotation_include_failure_excerpt=True,
        annotation_max_failure_chars=600,
        include_llm_fix_instructions=True,
    )

    return PytestRefiner(
        refine_port=mock_refine_port,
        telemetry_port=mock_telemetry_port,
        executor=Mock(),
        config=config,
        writer_port=mock_writer_port,
    )


@pytest.fixture
def refiner_with_annotation_disabled(
    mock_refine_port, mock_telemetry_port, mock_writer_port
):
    """PytestRefiner with annotation disabled."""
    config = RefineConfig(annotate_failed_tests=False)

    return PytestRefiner(
        refine_port=mock_refine_port,
        telemetry_port=mock_telemetry_port,
        executor=Mock(),
        config=config,
        writer_port=mock_writer_port,
    )


class TestUnrefinableFailureAnnotation:
    """Test annotation behavior for unrefinable failures."""

    @pytest.mark.asyncio
    async def test_annotates_on_unrefinable_failure(
        self, temp_test_file, refiner_with_annotation_enabled
    ):
        """Test that unrefinable failures get annotated."""

        # Mock pytest to return unrefinable failure
        with patch.object(refiner_with_annotation_enabled, "run_pytest") as mock_pytest:
            mock_pytest.return_value = {
                "returncode": 1,
                "stdout": "",
                "stderr": "ModuleNotFoundError: No module named 'missing_module'",
            }

            # Mock _is_unrefinable to return True
            with patch.object(
                refiner_with_annotation_enabled, "_is_unrefinable"
            ) as mock_unrefinable:
                mock_unrefinable.return_value = {
                    "unrefinable": True,
                    "failure_category": "import_error",
                }

                result = await refiner_with_annotation_enabled.refine_until_pass(
                    test_path=str(temp_test_file),
                    max_iterations=1,
                    build_source_context_fn=AsyncMock(return_value={}),
                )

        # Verify failure result
        assert result["success"] is False
        assert "import_error" in result["final_status"]

        # Verify annotation was created
        content = temp_test_file.read_text()
        assert "TEST REFINEMENT FAILED — MANUAL FIX REQUIRED" in content
        assert "unrefinable_import_error" in content
        assert "TESTCRAFT_FAILED_REFINEMENT_GUIDE" in content


class TestNoChangeFailureAnnotation:
    """Test annotation behavior for no-change failures."""

    @pytest.mark.asyncio
    async def test_annotates_on_no_change_early_stop(
        self, temp_test_file, refiner_with_annotation_enabled, mock_refine_port
    ):
        """Test that LLM no-change early stops get annotated."""

        # Mock pytest to return failure
        with patch.object(refiner_with_annotation_enabled, "run_pytest") as mock_pytest:
            mock_pytest.return_value = {
                "returncode": 1,
                "stdout": "",
                "stderr": "AssertionError: Test failed",
            }

            # Mock _is_unrefinable to return False (not unrefinable)
            with patch.object(
                refiner_with_annotation_enabled, "_is_unrefinable"
            ) as mock_unrefinable:
                mock_unrefinable.return_value = {
                    "unrefinable": False,
                    "failure_category": "assertion_error",
                }

                # Mock refine_from_failures to return no-change with fix instructions
                mock_refine_port.refine_from_failures.return_value = {
                    "success": False,
                    "final_status": "llm_no_change",
                    "fix_instructions": "Update import to runtime path 'my_module.submodule'\nMonkeypatch time.sleep to avoid timing issues",
                    "active_import_path": "my_module.submodule",
                    "preflight_suggestions": "Fix None/True/False casing",
                    "llm_confidence": 0.8,
                    "improvement_areas": ["imports", "timing"],
                    "iteration": 1,
                }

                # Mock build source context
                build_source_context_fn = AsyncMock(return_value={})

                result = await refiner_with_annotation_enabled.refine_until_pass(
                    test_path=str(temp_test_file),
                    max_iterations=1,
                    build_source_context_fn=build_source_context_fn,
                )

        # Verify failure result
        assert result["success"] is False
        assert result["final_status"] == "no_change_detected"

        # Verify annotation was created with TODO items
        content = temp_test_file.read_text()
        assert "TEST REFINEMENT FAILED — MANUAL FIX REQUIRED" in content
        assert "no_change_detected" in content
        assert "LLM FIX GUIDE (last iteration):" in content
        assert "- [ ] Update import to runtime path 'my_module.submodule'" in content
        assert "- [ ] Monkeypatch time.sleep to avoid timing issues" in content
        assert "Active import path: my_module.submodule" in content


class TestSyntaxErrorAnnotation:
    """Test annotation behavior for syntax errors."""

    @pytest.mark.asyncio
    async def test_annotates_on_syntax_error(
        self, temp_test_file, refiner_with_annotation_enabled, mock_refine_port
    ):
        """Test that syntax errors get annotated."""

        # Mock pytest to return failure
        with patch.object(refiner_with_annotation_enabled, "run_pytest") as mock_pytest:
            mock_pytest.return_value = {
                "returncode": 1,
                "stdout": "",
                "stderr": "SyntaxError: invalid syntax",
            }

            # Mock _is_unrefinable to return False
            with patch.object(
                refiner_with_annotation_enabled, "_is_unrefinable"
            ) as mock_unrefinable:
                mock_unrefinable.return_value = {
                    "unrefinable": False,
                    "failure_category": "syntax_error",
                }

                # Mock refine_from_failures to return syntax error
                mock_refine_port.refine_from_failures.return_value = {
                    "success": False,
                    "final_status": "syntax_error",
                    "error": "LLM returned invalid Python syntax",
                    "fix_instructions": "Check parentheses and indentation\nEnsure proper Python syntax",
                    "active_import_path": "",
                    "preflight_suggestions": "",
                    "iteration": 1,
                }

                # Mock build source context
                build_source_context_fn = AsyncMock(return_value={})

                result = await refiner_with_annotation_enabled.refine_until_pass(
                    test_path=str(temp_test_file),
                    max_iterations=1,
                    build_source_context_fn=build_source_context_fn,
                )

        # Verify failure result
        assert result["success"] is False
        assert result["final_status"] == "syntax_error"

        # Verify annotation was created
        content = temp_test_file.read_text()
        assert "TEST REFINEMENT FAILED — MANUAL FIX REQUIRED" in content
        assert "syntax_error" in content
        assert "- [ ] Check parentheses and indentation" in content


class TestMaxIterationsAnnotation:
    """Test annotation behavior for max iterations exhausted."""

    @pytest.mark.asyncio
    async def test_annotates_on_max_iterations_exceeded(
        self, temp_test_file, refiner_with_annotation_enabled, mock_refine_port
    ):
        """Test that max iterations exhausted gets annotated."""

        # Mock pytest to always return failure
        with patch.object(refiner_with_annotation_enabled, "run_pytest") as mock_pytest:
            mock_pytest.return_value = {
                "returncode": 1,
                "stdout": "",
                "stderr": "AssertionError: Test still failing",
            }

            # Mock _is_unrefinable to return False
            with patch.object(
                refiner_with_annotation_enabled, "_is_unrefinable"
            ) as mock_unrefinable:
                mock_unrefinable.return_value = {
                    "unrefinable": False,
                    "failure_category": "assertion_error",
                }

                # Mock refine_from_failures to return successful refinement each iteration
                mock_refine_port.refine_from_failures.return_value = {
                    "success": True,
                    "refined_content": "def test_example():\n    assert 1 == 1  # Modified test\n",
                    "fix_instructions": "Update assertion to use correct values\nCheck test logic",
                    "active_import_path": "test_module",
                    "preflight_suggestions": "",
                    "iteration": 2,
                }

                # Mock build source context
                build_source_context_fn = AsyncMock(return_value={})

                result = await refiner_with_annotation_enabled.refine_until_pass(
                    test_path=str(temp_test_file),
                    max_iterations=2,  # Set low max to trigger exhaustion
                    build_source_context_fn=build_source_context_fn,
                )

        # Verify failure result
        assert result["success"] is False
        assert result["iterations"] == 2
        assert "Maximum refinement iterations" in result["error"]

        # Verify annotation was created
        content = temp_test_file.read_text()
        assert "TEST REFINEMENT FAILED — MANUAL FIX REQUIRED" in content
        assert "max_iterations_exceeded" in content
        assert "Iterations: 2" in content


class TestAnnotationConfiguration:
    """Test various annotation configuration options."""

    @pytest.mark.asyncio
    async def test_annotation_disabled(
        self, temp_test_file, refiner_with_annotation_disabled, mock_refine_port
    ):
        """Test that annotation can be disabled."""

        # Mock pytest to return unrefinable failure
        with patch.object(
            refiner_with_annotation_disabled, "run_pytest"
        ) as mock_pytest:
            mock_pytest.return_value = {
                "returncode": 1,
                "stdout": "",
                "stderr": "ModuleNotFoundError: No module named 'missing_module'",
            }

            # Mock _is_unrefinable to return True
            with patch.object(
                refiner_with_annotation_disabled, "_is_unrefinable"
            ) as mock_unrefinable:
                mock_unrefinable.return_value = {
                    "unrefinable": True,
                    "failure_category": "import_error",
                }

                # Store original content
                original_content = temp_test_file.read_text()

                result = await refiner_with_annotation_disabled.refine_until_pass(
                    test_path=str(temp_test_file),
                    max_iterations=1,
                    build_source_context_fn=AsyncMock(return_value={}),
                )

        # Verify failure result
        assert result["success"] is False

        # Verify no annotation was added
        content = temp_test_file.read_text()
        assert content == original_content
        assert "TEST REFINEMENT FAILED" not in content

    @pytest.mark.asyncio
    async def test_annotation_style_hash(
        self, temp_test_file, mock_refine_port, mock_telemetry_port, mock_writer_port
    ):
        """Test hash-style annotations."""

        config = RefineConfig(
            annotate_failed_tests=True,
            annotation_style="hash",
            annotation_placement="top",
            annotation_include_failure_excerpt=False,
            include_llm_fix_instructions=True,
        )

        refiner = PytestRefiner(
            refine_port=mock_refine_port,
            telemetry_port=mock_telemetry_port,
            executor=Mock(),
            config=config,
            writer_port=mock_writer_port,
        )

        # Mock pytest to return unrefinable failure
        with patch.object(refiner, "run_pytest") as mock_pytest:
            mock_pytest.return_value = {
                "returncode": 1,
                "stdout": "",
                "stderr": "ImportError: No module named 'test'",
            }

            # Mock _is_unrefinable to return True
            with patch.object(refiner, "_is_unrefinable") as mock_unrefinable:
                mock_unrefinable.return_value = {
                    "unrefinable": True,
                    "failure_category": "import_error",
                }

                await refiner.refine_until_pass(
                    test_path=str(temp_test_file),
                    max_iterations=1,
                    build_source_context_fn=AsyncMock(return_value={}),
                )

        # Verify hash-style annotation
        content = temp_test_file.read_text()
        lines = content.split("\n")

        # Should start with hash comments, not docstring
        assert any("# TEST REFINEMENT FAILED" in line for line in lines)
        assert not content.startswith('"""')

    @pytest.mark.asyncio
    async def test_annotation_placement_bottom(
        self, temp_test_file, mock_refine_port, mock_telemetry_port, mock_writer_port
    ):
        """Test bottom placement of annotations."""

        config = RefineConfig(
            annotate_failed_tests=True,
            annotation_style="docstring",
            annotation_placement="bottom",
            annotation_include_failure_excerpt=True,
            include_llm_fix_instructions=True,
        )

        refiner = PytestRefiner(
            refine_port=mock_refine_port,
            telemetry_port=mock_telemetry_port,
            executor=Mock(),
            config=config,
            writer_port=mock_writer_port,
        )

        # Store original content
        original_content = temp_test_file.read_text()

        # Mock pytest to return unrefinable failure
        with patch.object(refiner, "run_pytest") as mock_pytest:
            mock_pytest.return_value = {
                "returncode": 1,
                "stdout": "",
                "stderr": "ImportError: No module named 'test'",
            }

            # Mock _is_unrefinable to return True
            with patch.object(refiner, "_is_unrefinable") as mock_unrefinable:
                mock_unrefinable.return_value = {
                    "unrefinable": True,
                    "failure_category": "import_error",
                }

                await refiner.refine_until_pass(
                    test_path=str(temp_test_file),
                    max_iterations=1,
                    build_source_context_fn=AsyncMock(return_value={}),
                )

        # Verify bottom placement
        content = temp_test_file.read_text()

        # Should start with original content
        assert content.startswith(original_content)
        # Should end with annotation
        assert "TEST REFINEMENT FAILED" in content
        assert content.count('"""') == 2  # Opening and closing docstring


class TestAnnotationIdempotency:
    """Test that annotations are idempotent (don't duplicate)."""

    @pytest.mark.asyncio
    async def test_no_duplicate_annotation_on_repeated_runs(
        self, temp_test_file, refiner_with_annotation_enabled
    ):
        """Test that running refinement twice doesn't create duplicate annotations."""

        async def run_failed_refinement():
            # Mock pytest to return unrefinable failure
            with patch.object(
                refiner_with_annotation_enabled, "run_pytest"
            ) as mock_pytest:
                mock_pytest.return_value = {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "ImportError: No module named 'test'",
                }

                # Mock _is_unrefinable to return True
                with patch.object(
                    refiner_with_annotation_enabled, "_is_unrefinable"
                ) as mock_unrefinable:
                    mock_unrefinable.return_value = {
                        "unrefinable": True,
                        "failure_category": "import_error",
                    }

                    return await refiner_with_annotation_enabled.refine_until_pass(
                        test_path=str(temp_test_file),
                        max_iterations=1,
                        build_source_context_fn=AsyncMock(return_value={}),
                    )

        # Run refinement twice
        result1 = await run_failed_refinement()
        result2 = await run_failed_refinement()

        # Both should fail
        assert result1["success"] is False
        assert result2["success"] is False

        # Should only have one annotation marker
        content = temp_test_file.read_text()
        marker_count = content.count("TESTCRAFT_FAILED_REFINEMENT_GUIDE")
        assert marker_count == 1

        # Should only have one annotation block
        annotation_count = content.count("TEST REFINEMENT FAILED")
        assert annotation_count == 1


class TestAnnotationHelpers:
    """Test annotation helper methods."""

    def test_format_fix_instructions_as_todos(self, refiner_with_annotation_enabled):
        """Test conversion of fix instructions to TODO format."""

        # Test bullet points
        instructions = "- Fix import statement\n- Update assertion\n- Add mock"
        result = refiner_with_annotation_enabled._format_fix_instructions_as_todos(
            instructions
        )

        expected_lines = [
            "- [ ] Fix import statement",
            "- [ ] Update assertion",
            "- [ ] Add mock",
        ]
        assert result == "\n".join(expected_lines)

        # Test numbered list
        instructions = "1. Fix import statement\n2. Update assertion\n3. Add mock"
        result = refiner_with_annotation_enabled._format_fix_instructions_as_todos(
            instructions
        )

        expected_lines = [
            "- [ ] Fix import statement",
            "- [ ] Update assertion",
            "- [ ] Add mock",
        ]
        assert result == "\n".join(expected_lines)

        # Test plain text
        instructions = "Fix import statement\nUpdate assertion\nAdd mock"
        result = refiner_with_annotation_enabled._format_fix_instructions_as_todos(
            instructions
        )

        expected_lines = [
            "- [ ] Fix import statement",
            "- [ ] Update assertion",
            "- [ ] Add mock",
        ]
        assert result == "\n".join(expected_lines)

        # Test empty instructions
        result = refiner_with_annotation_enabled._format_fix_instructions_as_todos("")
        assert result == "No specific instructions available."


class TestTelemetryIntegration:
    """Test telemetry integration for annotations."""

    @pytest.mark.asyncio
    async def test_telemetry_records_annotation_added(
        self, temp_test_file, refiner_with_annotation_enabled, mock_telemetry_port
    ):
        """Test that telemetry correctly records when annotations are added."""

        # Mock pytest to return unrefinable failure
        with patch.object(refiner_with_annotation_enabled, "run_pytest") as mock_pytest:
            mock_pytest.return_value = {
                "returncode": 1,
                "stdout": "",
                "stderr": "ImportError: No module named 'test'",
            }

            # Mock _is_unrefinable to return True
            with patch.object(
                refiner_with_annotation_enabled, "_is_unrefinable"
            ) as mock_unrefinable:
                mock_unrefinable.return_value = {
                    "unrefinable": True,
                    "failure_category": "import_error",
                }

                await refiner_with_annotation_enabled.refine_until_pass(
                    test_path=str(temp_test_file),
                    max_iterations=1,
                    build_source_context_fn=AsyncMock(return_value={}),
                )

        # Verify telemetry was called
        mock_span = (
            mock_telemetry_port.create_child_span.return_value.__enter__.return_value
        )

        # Check that telemetry attributes were set
        attribute_calls = [call.args for call in mock_span.set_attribute.call_args_list]
        attribute_dict = dict(attribute_calls)

        # Should record annotation was added
        assert attribute_dict.get("failed_annotation_added") is True
        assert "annotation_style" in attribute_dict
        assert "annotation_placement" in attribute_dict
        assert "annotation_size" in attribute_dict
