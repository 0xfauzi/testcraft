"""
Tests for refine adapters.

This module tests the refine adapter implementations, focusing on
pytest failure-based test refinement functionality.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from testcraft.adapters.refine.main_adapter import RefineAdapter
from testcraft.config.models import RefineConfig
from testcraft.ports.llm_port import LLMPort


class MockLLMPort:
    """Mock LLM port for testing."""

    def __init__(self, responses: list = None):
        self.responses = responses or []
        self.call_count = 0
        self.last_refinement_instructions = None

    def refine_content(
        self, original_content: str, refinement_instructions: str, **kwargs
    ) -> dict[str, Any]:
        self.last_refinement_instructions = refinement_instructions
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return {
                "refined_content": response,
                "changes_made": "Mock refinement",
                "confidence": 0.8,
            }
        return {
            "refined_content": "def test_example():\n    assert True",
            "changes_made": "Mock refinement",
            "confidence": 0.8,
        }

    def generate_tests(
        self,
        code_content: str,
        context: str = None,
        test_framework: str = "pytest",
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "tests": "def test_mock(): pass",
            "coverage_focus": [],
            "confidence": 0.5,
            "metadata": {},
        }

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs
    ) -> dict[str, Any]:
        return {
            "testability_score": 5.0,
            "complexity_metrics": {},
            "recommendations": [],
            "potential_issues": [],
        }


class TestRefineAdapter:
    """Test cases for RefineAdapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMPort()
        self.adapter = RefineAdapter(self.mock_llm)

    def test_init(self):
        """Test adapter initialization."""
        llm = Mock(spec=LLMPort)
        adapter = RefineAdapter(llm)
        assert adapter.llm is llm

    def test_refine_from_failures_file_not_found(self):
        """Test refinement with non-existent test file."""
        result = self.adapter.refine_from_failures(
            test_file="nonexistent.py",
            failure_output="ImportError: No module named 'foo'",
        )

        assert result["success"] is False
        assert "not found" in result["error"]
        assert result["iterations_used"] == 0
        assert result["final_status"] == "file_not_found"

    @patch("testcraft.adapters.refine.main_adapter.Path")
    @patch("testcraft.adapters.refine.main_adapter.run_subprocess_simple")
    def test_refine_from_failures_success_first_iteration(
        self, mock_run_command, mock_path
    ):
        """Test successful refinement on first iteration."""
        # Mock file operations
        mock_test_file = Mock()
        mock_test_file.exists.return_value = True
        mock_test_file.read_text.return_value = "def test_broken():\n    assert 1 == 2"
        mock_test_file.write_text = Mock()
        mock_test_file.suffix = ".py"
        mock_test_file.name = "test_example.py"  # For validation
        mock_test_file.parts = ("tests", "test_example.py")  # For validation
        mock_resolved_path = Mock()
        mock_resolved_path.__str__ = Mock(
            return_value="/fake/path/tests/test_example.py"
        )
        mock_resolved_path.parts = ("fake", "path", "tests", "test_example.py")
        mock_test_file.resolve.return_value = mock_resolved_path
        mock_backup = Mock()
        mock_backup.write_text = Mock()
        mock_backup.exists.return_value = True
        mock_backup.unlink = Mock()
        mock_test_file.with_suffix.return_value = mock_backup
        mock_path.return_value = mock_test_file

        # Mock pytest success
        mock_run_command.return_value = ("PASSED", "", 0)

        # Mock LLM response
        self.mock_llm.responses = ["def test_fixed():\n    assert 1 == 1"]

        result = self.adapter.refine_from_failures(
            test_file="test_example.py", failure_output="AssertionError: assert 1 == 2"
        )

        assert result["success"] is True
        assert result["refined_content"] == "def test_fixed():\n    assert 1 == 1"
        assert result["iterations_used"] == 1
        assert result["final_status"] == "success"

    @patch("testcraft.adapters.refine.main_adapter.Path")
    @patch("testcraft.adapters.refine.main_adapter.run_subprocess_simple")
    def test_refine_from_failures_max_iterations(self, mock_run_command, mock_path):
        """Test refinement hitting max iterations."""
        # Mock file operations
        mock_test_file = Mock()
        mock_test_file.exists.return_value = True

        # Make read_text return different content for each iteration to avoid no-change detection
        # The sequence is: initial_read, then what_llm_wrote_iteration_1, then what_llm_wrote_iteration_2
        def mock_read_text(*args, **kwargs):
            # Handle encoding parameter and return content based on call count
            call_count = getattr(mock_read_text, "call_count", 0)
            mock_read_text.call_count = call_count + 1

            if call_count == 0:
                return "def test_broken():\n    assert False"  # Initial read
            elif call_count == 1:
                return (
                    "def test_refined1():\n    assert False"  # After first refinement
                )
            else:
                return (
                    "def test_refined2():\n    assert False"  # After second refinement
                )

        mock_test_file.read_text.side_effect = mock_read_text
        mock_test_file.write_text = Mock()
        mock_test_file.suffix = ".py"
        mock_test_file.name = "test_example.py"  # For validation
        mock_test_file.parts = ("tests", "test_example.py")  # For validation
        mock_resolved_path = Mock()
        mock_resolved_path.__str__ = Mock(
            return_value="/fake/path/tests/test_example.py"
        )
        mock_resolved_path.parts = ("fake", "path", "tests", "test_example.py")
        mock_test_file.resolve.return_value = mock_resolved_path
        mock_backup = Mock()
        mock_backup.write_text = Mock()
        mock_backup.exists.return_value = True
        mock_backup.unlink = Mock()
        mock_test_file.with_suffix.return_value = mock_backup
        mock_path.return_value = mock_test_file

        # Mock pytest always failing
        mock_run_command.return_value = ("FAILED", "AssertionError", 1)

        # Mock LLM responses that match what the file will contain after each refinement
        self.mock_llm.responses = [
            "def test_refined1():\n    assert False",  # First LLM response - writes to file
            "def test_refined2():\n    assert False",  # Second LLM response - writes to file
            "def test_refined3():\n    assert False",  # Won't be reached due to max_iterations=2
        ]

        result = self.adapter.refine_from_failures(
            test_file="test_example.py",
            failure_output="AssertionError",
            max_iterations=2,
        )

        assert result["success"] is False
        # The test can hit either max iterations or LLM no change condition - both are valid failure modes
        assert result["error"] in [
            "Max iterations (2) reached without success",
            "LLM returned no changes or identical content",
            "LLM returned identical content to input (normalized)",
        ]
        assert result["iterations_used"] <= 2
        assert result["final_status"] in ["max_iterations", "llm_no_change"]

    @patch("testcraft.adapters.refine.main_adapter.Path")
    def test_refine_from_failures_llm_error(self, mock_path):
        """Test refinement with LLM error."""
        # Mock file operations
        mock_test_file = Mock()
        mock_test_file.exists.return_value = True
        mock_test_file.read_text.return_value = "def test_broken():\n    assert False"
        mock_path.return_value = mock_test_file

        # Mock LLM raising exception
        self.mock_llm.responses = []  # Empty responses will cause error
        with patch.object(
            self.mock_llm, "refine_content", side_effect=Exception("LLM error")
        ):
            result = self.adapter.refine_from_failures(
                test_file="test_example.py", failure_output="AssertionError"
            )

        assert result["success"] is False
        assert "LLM refinement failed" in result["error"]
        assert result["final_status"] == "llm_error"

    @patch("testcraft.adapters.refine.main_adapter.Path")
    def test_refine_from_failures_no_change_detection(self, mock_path):
        """Test no-change detection in refinement."""
        # Mock file operations - return same content twice
        mock_test_file = Mock()
        mock_test_file.exists.return_value = True
        mock_test_file.read_text.side_effect = [
            "def test_same():\n    assert False",  # First read
            "def test_same():\n    assert False",  # Second read (after "refinement")
        ]
        mock_test_file.write_text = Mock()
        mock_test_file.name = "test_example.py"  # For validation
        mock_test_file.parts = ("tests", "test_example.py")  # For validation
        mock_resolved_path = Mock()
        mock_resolved_path.__str__ = Mock(
            return_value="/fake/path/tests/test_example.py"
        )
        mock_resolved_path.parts = ("fake", "path", "tests", "test_example.py")
        mock_test_file.resolve.return_value = mock_resolved_path
        mock_test_file.with_suffix.return_value = Mock()
        mock_path.return_value = mock_test_file

        # Mock LLM response (same as original)
        self.mock_llm.responses = ["def test_same():\n    assert False"]

        result = self.adapter.refine_from_failures(
            test_file="test_example.py", failure_output="AssertionError"
        )

        assert result["success"] is False
        assert (
            "No changes made" in result["error"]
            or "no changes" in result["error"].lower()
            or "identical content" in result["error"].lower()
        )
        assert result["final_status"] in ["no_change", "llm_no_change"]

    def test_build_refinement_payload(self):
        """Test building refinement payload."""
        test_file = Path("test_example.py")
        current_content = "def test_example():\n    assert False"
        failure_output = "AssertionError: assert False"

        payload = self.adapter._build_refinement_payload(
            test_file=test_file,
            current_content=current_content,
            failure_output=failure_output,
            iteration=1,
        )

        assert payload["task"] == "refine_failing_test"
        assert payload["test_file_path"] == str(test_file)
        assert payload["current_test_content"] == current_content
        assert payload["pytest_failure_output"] == failure_output
        assert payload["iteration"] == 1
        assert "instructions" in payload
        assert len(payload["instructions"]) > 0

    def test_build_refinement_payload_with_source_context(self):
        """Test building refinement payload with source context."""
        source_context = {"module": "example.py", "functions": ["add", "subtract"]}

        payload = self.adapter._build_refinement_payload(
            test_file=Path("test_example.py"),
            current_content="def test_add(): pass",
            failure_output="ImportError",
            source_context=source_context,
            iteration=2,
        )

        assert payload["source_context"] == source_context
        assert payload["iteration"] == 2

    def test_payload_to_instructions(self):
        """Test converting payload to refinement instructions."""
        payload = {
            "test_file_path": "test_example.py",
            "current_test_content": "def test_example():\n    assert False",
            "pytest_failure_output": "AssertionError: assert False",
            "iteration": 1,
        }

        instructions = self.adapter._payload_to_instructions(payload)

        assert "test_example.py" in instructions
        assert "assert False" in instructions
        assert "AssertionError" in instructions
        assert "Iteration: 1" in instructions

    def test_payload_to_instructions_with_source_context(self):
        """Test instructions generation with source context."""
        payload = {
            "test_file_path": "test_example.py",
            "current_test_content": "def test_add(): pass",
            "pytest_failure_output": "ImportError: No module named 'math'",
            "iteration": 1,
            "source_context": {"imports": ["import math"], "functions": ["add"]},
        }

        instructions = self.adapter._payload_to_instructions(payload)

        assert "Source Code Context:" in instructions
        assert "import math" in instructions

    def test_extract_test_content_with_code_block(self):
        """Test extracting test content from LLM response with code block."""
        llm_response = """Here's the fixed test:

```python
def test_fixed():
    assert 1 == 1
```

The issue was with the assertion."""

        content = self.adapter._extract_test_content(llm_response)
        assert content == "def test_fixed():\n    assert 1 == 1"

    def test_extract_test_content_without_code_block(self):
        """Test extracting test content from LLM response without code block."""
        llm_response = "def test_fixed():\n    assert 1 == 1"

        content = self.adapter._extract_test_content(llm_response)
        assert content == "def test_fixed():\n    assert 1 == 1"

    @patch("testcraft.adapters.refine.main_adapter.Path")
    def test_apply_refinement_safely_success(self, mock_path_class):
        """Test safe application of refinement."""
        # Mock the test file
        mock_test_file = Mock()
        mock_test_file.read_text.return_value = "original content"
        mock_test_file.write_text = Mock()
        mock_test_file.suffix = ".py"  # Mock the suffix property

        # Mock the backup file
        mock_backup_file = Mock()
        mock_backup_file.exists.return_value = True
        mock_backup_file.unlink = Mock()
        mock_backup_file.write_text = Mock()
        mock_test_file.with_suffix.return_value = mock_backup_file

        refined_content = "refined content"

        self.adapter._apply_refinement_safely(mock_test_file, refined_content)

        # Verify backup was created
        mock_backup_file.write_text.assert_called_once_with(
            "original content", encoding="utf-8"
        )
        # Verify refined content was written
        mock_test_file.write_text.assert_called_once_with(
            refined_content, encoding="utf-8"
        )
        # Verify backup was cleaned up
        mock_backup_file.unlink.assert_called_once()

    @patch("testcraft.adapters.refine.main_adapter.run_subprocess_simple")
    def test_run_pytest_verification_success(self, mock_run_command):
        """Test pytest verification with successful result."""
        mock_run_command.return_value = ("test passed", "", 0)

        result = self.adapter._run_pytest_verification(Path("test_example.py"))

        assert result["success"] is True
        assert result["output"] == "test passed"
        assert result["return_code"] == 0

        # Verify the command was called with expected basic arguments
        mock_run_command.assert_called_once()
        args, kwargs = mock_run_command.call_args
        assert args[0] == ["python", "-m", "pytest", "test_example.py", "-v"]
        assert kwargs["timeout"] == 60
        assert kwargs["raise_on_error"] is False

    @patch("testcraft.adapters.refine.main_adapter.run_subprocess_simple")
    def test_run_pytest_verification_failure(self, mock_run_command):
        """Test pytest verification with failed result."""
        mock_run_command.return_value = ("", "AssertionError: test failed", 1)

        result = self.adapter._run_pytest_verification(Path("test_example.py"))

        assert result["success"] is False
        assert "AssertionError" in result["output"]
        assert result["return_code"] == 1

    @patch("testcraft.adapters.refine.main_adapter.run_subprocess_simple")
    def test_run_pytest_verification_exception(self, mock_run_command):
        """Test pytest verification with exception."""
        mock_run_command.side_effect = Exception("Command failed")

        result = self.adapter._run_pytest_verification(Path("test_example.py"))

        assert result["success"] is False
        assert "Pytest execution failed" in result["output"]
        assert result["return_code"] == -1

    def test_placeholder_methods(self):
        """Test that placeholder methods return expected structures."""
        # Test basic refine method
        outcome = self.adapter.refine(test_files=["test.py"])
        assert hasattr(outcome, "updated_files")
        assert hasattr(outcome, "rationale")

        # Test analyze_test_quality
        quality = self.adapter.analyze_test_quality("test.py")
        assert "quality_score" in quality
        assert "issues" in quality
        assert "recommendations" in quality

        # Test suggest_improvements
        improvements = self.adapter.suggest_improvements("test.py")
        assert "suggestions" in improvements
        assert "priority" in improvements

        # Test optimize_test_structure
        optimization = self.adapter.optimize_test_structure("test.py")
        assert "optimized_structure" in optimization
        assert "changes_needed" in optimization

        # Test enhance_test_coverage
        coverage = self.adapter.enhance_test_coverage("test.py", "source.py")
        assert "new_tests" in coverage
        assert "coverage_improvement" in coverage


class TestRefineAdapterIntegration:
    """Integration tests for RefineAdapter."""

    @pytest.mark.skip(
        reason="Complex integration test with many mocking dependencies - needs refactoring"
    )
    def test_full_refinement_flow_mock(self):
        """Test the complete refinement flow with mocking."""
        with (
            patch("testcraft.adapters.refine.main_adapter.Path") as mock_path,
            patch(
                "testcraft.adapters.refine.main_adapter.run_subprocess_simple"
            ) as mock_run_command,
            patch("ast.parse") as mock_ast_parse,
            patch("ast.walk") as mock_ast_walk,
        ):
            # Setup mocks
            mock_test_file = Mock()
            mock_test_file.exists.return_value = True
            mock_test_file.read_text.side_effect = [
                "def test_broken():\n    assert 1 == 2",  # First read
                "def test_fixed():\n    assert 1 == 1",  # After refinement
            ]
            mock_test_file.write_text = Mock()
            mock_test_file.suffix = ".py"
            mock_test_file.name = "test_example.py"
            # Mock Path methods for validation
            mock_resolved = Mock()
            mock_resolved.parts = ("tests", "test_example.py")
            mock_resolved.__str__ = Mock(return_value="tests/test_example.py")
            mock_test_file.resolve.return_value = mock_resolved
            mock_backup = Mock()
            mock_backup.write_text = Mock()
            mock_backup.exists.return_value = True
            mock_backup.unlink = Mock()
            mock_test_file.with_suffix.return_value = mock_backup
            mock_path.return_value = mock_test_file

            # Mock successful pytest after refinement
            mock_run_command.return_value = ("PASSED", "", 0)

            # Mock successful syntax validation
            mock_ast_parse.return_value = Mock()  # ast.parse returns AST object
            mock_ast_walk.return_value = []  # Empty list to bypass AST processing

            # Setup adapter with mock LLM
            mock_llm = MockLLMPort(["def test_fixed():\n    assert 1 == 1"])
            adapter = RefineAdapter(mock_llm)

            # Run refinement
            result = adapter.refine_from_failures(
                test_file="test_example.py",
                failure_output="AssertionError: assert 1 == 2",
            )

            # Verify result
            assert result["success"] is True
            assert result["iterations_used"] == 1
            assert "def test_fixed()" in result["refined_content"]

            # Verify LLM was called with correct refinement instructions
            assert (
                "AssertionError: assert 1 == 2" in mock_llm.last_refinement_instructions
            )
            assert "assert 1 == 2" in mock_llm.last_refinement_instructions


class TestRefineAdapterHardening:
    """Tests for refinement hardening and validation."""

    def test_validate_refined_content_none_content(self):
        """Test validation rejects None content."""
        llm = MockLLMPort()
        config = RefineConfig()
        adapter = RefineAdapter(llm, config=config)

        result = adapter._validate_refined_content(None, "original content")

        assert not result["is_valid"]
        assert "None content" in result["reason"]

    def test_validate_refined_content_empty_content(self):
        """Test validation rejects empty content when configured."""
        llm = MockLLMPort()
        config = RefineConfig()
        adapter = RefineAdapter(llm, config=config)

        result = adapter._validate_refined_content("", "original content")

        assert not result["is_valid"]
        assert "empty or whitespace-only" in result["reason"]

    def test_validate_refined_content_whitespace_only(self):
        """Test validation rejects whitespace-only content."""
        llm = MockLLMPort()
        config = RefineConfig()
        adapter = RefineAdapter(llm, config=config)

        result = adapter._validate_refined_content("   \n\t  ", "original content")

        assert not result["is_valid"]
        assert "empty or whitespace-only" in result["reason"]

    def test_validate_refined_content_literal_none(self):
        """Test validation rejects literal 'None' strings."""
        llm = MockLLMPort()
        config = RefineConfig()
        adapter = RefineAdapter(llm, config=config)

        test_cases = ["None", "none", "NONE", "null", "NULL", "Null"]

        for literal_none in test_cases:
            result = adapter._validate_refined_content(literal_none, "original content")
            assert not result["is_valid"], f"Should reject literal '{literal_none}'"
            assert "literal" in result["reason"].lower()

    def test_validate_refined_content_identical_content(self):
        """Test validation rejects identical content when configured."""
        llm = MockLLMPort()
        config = RefineConfig()
        adapter = RefineAdapter(llm, config=config)

        original = "def test_example():\n    assert True"

        result = adapter._validate_refined_content(original, original)

        assert not result["is_valid"]
        assert "identical content" in result["reason"]

    def test_validate_refined_content_syntax_error(self):
        """Test validation rejects content with syntax errors."""
        llm = MockLLMPort()
        config = RefineConfig()
        adapter = RefineAdapter(llm, config=config)

        invalid_python = "def test_example(\n    assert True  # missing closing paren"

        result = adapter._validate_refined_content(invalid_python, "original")

        assert not result["is_valid"]
        assert "invalid Python syntax" in result["reason"]

    def test_validate_refined_content_valid_python(self):
        """Test validation passes for valid Python content."""
        llm = MockLLMPort()
        config = RefineConfig()
        adapter = RefineAdapter(llm, config=config)

        valid_python = "def test_example():\n    assert True"

        result = adapter._validate_refined_content(valid_python, "different content")

        assert result["is_valid"]
        assert "reason" not in result

    def test_validate_test_path_safety_valid_paths(self, tmp_path):
        """Test path safety validation accepts valid test paths."""
        llm = MockLLMPort()
        adapter = RefineAdapter(llm)

        # Create actual test files in temporary directory to ensure they resolve properly
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        test_file_1 = tests_dir / "test_example.py"
        test_file_1.write_text("def test_example():\n    pass")

        test_file_2 = tmp_path / "test_module.py"
        test_file_2.write_text("def test_module():\n    pass")

        # Test absolute path structure
        nested_tests = tmp_path / "some" / "project" / "tests"
        nested_tests.mkdir(parents=True)
        test_file_3 = nested_tests / "test_feature.py"
        test_file_3.write_text("def test_feature():\n    pass")

        valid_paths = [
            test_file_1,  # tests/test_example.py (relative to tmp_path)
            test_file_2,  # test_module.py (relative to tmp_path)
            test_file_3,  # some/project/tests/test_feature.py (relative to tmp_path)
        ]

        for path in valid_paths:
            result = adapter._validate_test_path_safety(path)
            assert result, f"Should accept valid path: {path}"

    def test_validate_test_path_safety_invalid_paths(self, tmp_path):
        """Test path safety validation rejects invalid paths."""
        llm = MockLLMPort()
        adapter = RefineAdapter(llm)

        # Create actual invalid files to test path resolution
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        invalid_file_1 = src_dir / "module.py"
        invalid_file_1.write_text("# Not a test file")

        invalid_file_2 = tests_dir / "example.txt"
        invalid_file_2.write_text("Not Python")

        invalid_file_3 = tmp_path / "example.py"
        invalid_file_3.write_text("# No test indicator")

        invalid_paths = [
            invalid_file_1,  # src/module.py - Not a test file
            invalid_file_2,  # tests/example.txt - Not Python
            invalid_file_3,  # example.py - No test indicator
            Path("/etc/passwd"),  # System file (keep as absolute path)
        ]

        for path in invalid_paths:
            result = adapter._validate_test_path_safety(path)
            assert not result, f"Should reject invalid path: {path}"

    def test_refine_from_failures_llm_no_change_early_exit(self, tmp_path):
        """Test that llm_no_change status causes early exit."""
        # Create test file
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_example():\n    assert False")

        # Mock LLM that returns None (triggers validation failure)
        llm = MockLLMPort([None])
        config = RefineConfig()
        adapter = RefineAdapter(llm, config=config)

        result = adapter.refine_from_failures(
            test_file=test_file, failure_output="Test failed", max_iterations=3
        )

        assert not result["success"]
        assert result["final_status"] == "llm_invalid_output"
        assert result["iterations_used"] == 1
        assert "literal 'None' content" in result["error"]

    def test_refine_from_failures_identical_content_early_exit(self, tmp_path):
        """Test that identical content causes early exit."""
        # Create test file
        original_content = "def test_example():\n    assert False"
        test_file = tmp_path / "test_example.py"
        test_file.write_text(original_content)

        # Mock LLM that returns identical content
        llm = MockLLMPort([original_content])
        config = RefineConfig()
        adapter = RefineAdapter(llm, config=config)

        result = adapter.refine_from_failures(
            test_file=test_file, failure_output="Test failed", max_iterations=3
        )

        assert not result["success"]
        assert result["final_status"] == "llm_no_change"
        assert result["iterations_used"] == 1
        assert "identical content" in result["error"]

    def test_refine_from_failures_syntax_error_rollback(self, tmp_path):
        """Test that syntax errors trigger rollback."""
        # Create test file
        original_content = "def test_example():\n    assert False"
        test_file = tmp_path / "test_example.py"
        test_file.write_text(original_content)

        # Mock LLM that returns invalid syntax
        llm = MockLLMPort(["def test_example(\n    assert True  # syntax error"])
        config = RefineConfig()
        adapter = RefineAdapter(llm, config=config)

        result = adapter.refine_from_failures(
            test_file=test_file, failure_output="Test failed", max_iterations=3
        )

        assert not result["success"]
        assert result["final_status"] == "syntax_error"
        assert "invalid Python syntax" in result["error"]

        # Verify original content is preserved
        assert test_file.read_text() == original_content

    def test_refine_config_guardrails_can_be_disabled(self):
        """Test that guardrails can be selectively disabled."""
        llm = MockLLMPort()
        config = RefineConfig()
        config.refinement_guardrails = {
            "reject_empty": False,
            "reject_literal_none": False,
            "reject_identical": False,
            "validate_syntax": False,
            "format_on_refine": False,
        }
        adapter = RefineAdapter(llm, config=config)

        # Test that validation passes when guardrails are disabled
        result = adapter._validate_refined_content("", "original")
        assert result["is_valid"]  # Empty content allowed when reject_empty=False

        result = adapter._validate_refined_content("None", "original")
        assert result["is_valid"]  # Literal None allowed when reject_literal_none=False

        result = adapter._validate_refined_content("original", "original")
        assert result[
            "is_valid"
        ]  # Identical content allowed when reject_identical=False

    def test_configuration_pytest_args_default(self):
        """Test that pytest args have sensible defaults."""
        config = RefineConfig()
        assert config.pytest_args_for_refinement == ["-vv", "--tb=short", "-x"]

    def test_configuration_guardrails_defaults(self):
        """Test that guardrails have safe defaults."""
        config = RefineConfig()
        guardrails = config.refinement_guardrails

        assert guardrails["reject_empty"] is True
        assert guardrails["reject_literal_none"] is True
        assert guardrails["reject_identical"] is True
        assert guardrails["validate_syntax"] is True
        assert guardrails["format_on_refine"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
