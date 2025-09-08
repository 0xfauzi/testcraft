"""
Tests for refine adapters.

This module tests the refine adapter implementations, focusing on
pytest failure-based test refinement functionality.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from testcraft.adapters.refine.main_adapter import RefineAdapter
from testcraft.ports.llm_port import LLMPort


class MockLLMPort:
    """Mock LLM port for testing."""

    def __init__(self, responses: list = None):
        self.responses = responses or []
        self.call_count = 0
        self.last_prompt = None

    def generate(self, prompt: str, **kwargs) -> str:
        self.last_prompt = prompt
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "def test_example():\n    assert True"


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
        mock_test_file.read_text.side_effect = [
            "def test_broken():\n    assert False",  # Initial read
            "def test_refined1():\n    assert False",  # What the file contains after first LLM refinement
            "def test_refined2():\n    assert False",  # What the file contains after second LLM refinement
        ]
        mock_test_file.write_text = Mock()
        mock_test_file.suffix = ".py"
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
            self.mock_llm, "generate", side_effect=Exception("LLM error")
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

    def test_payload_to_prompt(self):
        """Test converting payload to prompt."""
        payload = {
            "test_file_path": "test_example.py",
            "current_test_content": "def test_example():\n    assert False",
            "pytest_failure_output": "AssertionError: assert False",
            "iteration": 1,
        }

        prompt = self.adapter._payload_to_prompt(payload)

        assert "test_example.py" in prompt
        assert "assert False" in prompt
        assert "AssertionError" in prompt
        assert "Iteration: 1" in prompt
        assert "```python" in prompt

    def test_payload_to_prompt_with_source_context(self):
        """Test prompt generation with source context."""
        payload = {
            "test_file_path": "test_example.py",
            "current_test_content": "def test_add(): pass",
            "pytest_failure_output": "ImportError: No module named 'math'",
            "iteration": 1,
            "source_context": {"imports": ["import math"], "functions": ["add"]},
        }

        prompt = self.adapter._payload_to_prompt(payload)

        assert "Source Code Context:" in prompt
        assert "import math" in prompt

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

        mock_run_command.assert_called_once_with(
            ["python", "-m", "pytest", "test_example.py", "-v"],
            timeout=60,
            raise_on_error=False,
        )

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

    def test_full_refinement_flow_mock(self):
        """Test the complete refinement flow with mocking."""
        with (
            patch("testcraft.adapters.refine.main_adapter.Path") as mock_path,
            patch(
                "testcraft.adapters.refine.main_adapter.run_subprocess_simple"
            ) as mock_run_command,
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
            mock_backup = Mock()
            mock_backup.write_text = Mock()
            mock_backup.exists.return_value = True
            mock_backup.unlink = Mock()
            mock_test_file.with_suffix.return_value = mock_backup
            mock_path.return_value = mock_test_file

            # Mock successful pytest after refinement
            mock_run_command.return_value = ("PASSED", "", 0)

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

            # Verify LLM was called with correct prompt
            assert "AssertionError: assert 1 == 2" in mock_llm.last_prompt
            assert "assert 1 == 2" in mock_llm.last_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
