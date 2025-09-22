"""
Integration tests for PytestRefiner service.

This module tests the complete pytest refinement workflow including
passing tests scenarios, no-op detection, and configuration integration.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import Mock, patch

import pytest

from testcraft.application.generation.services.pytest_refiner import PytestRefiner
from testcraft.config.models import RefineConfig
from testcraft.domain.models import RefineOutcome


class MockRefinePort:
    """Mock RefinePort for testing."""

    def __init__(self, responses: list[dict] = None):
        """Initialize mock with predefined responses."""
        self.responses = responses or []
        self.call_count = 0
        self.last_call_args = None

    def refine_from_failures(
        self, test_file, failure_output, source_context=None, max_iterations=1, **kwargs
    ) -> dict[str, Any]:
        """Mock refine_from_failures method."""
        self.last_call_args = {
            "test_file": test_file,
            "failure_output": failure_output,
            "source_context": source_context,
            "max_iterations": max_iterations,
            **kwargs,
        }

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response

        # Default successful response
        return {
            "success": True,
            "refined_content": "def test_example():\n    assert True",
            "iterations_used": 1,
            "final_status": "success",
        }

    def refine(self, test_files, source_files=None, refinement_goals=None, **kwargs):
        """Mock refine method."""
        return RefineOutcome(
            updated_files=[str(f) for f in test_files],
            rationale="Mock refinement",
            plan="Mock plan",
        )


class MockTelemetryPort:
    """Mock TelemetryPort for testing."""

    def __init__(self):
        self.spans = []
        self.current_span = None

    def create_child_span(self, name: str):
        """Create a mock span context manager."""
        span = Mock()
        span.set_attribute = Mock()
        span.__enter__ = Mock(return_value=span)
        span.__exit__ = Mock(return_value=None)
        self.spans.append((name, span))
        return span


class TestPytestRefinerIntegration:
    """Integration tests for PytestRefiner."""

    def setup_method(self):
        """Set up test fixtures."""
        self.refine_port = MockRefinePort()
        self.telemetry_port = MockTelemetryPort()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.config = RefineConfig()

    def teardown_method(self):
        """Clean up test fixtures."""
        self.executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_refine_until_pass_with_passing_tests(self, tmp_path):
        """Test that passing tests skip refinement immediately."""
        # Create test file
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_example():\n    assert True")

        refiner = PytestRefiner(
            self.refine_port,
            self.telemetry_port,
            self.executor,
            config=self.config,
        )

        # Mock successful pytest run
        with patch.object(refiner, "run_pytest") as mock_pytest:
            mock_pytest.return_value = {
                "stdout": "PASSED",
                "stderr": "",
                "returncode": 0,
                "command": "python -m pytest test_example.py",
            }

            async def mock_context_fn(test_path, content):
                return {"related_files": ["src/module.py"]}

            result = await refiner.refine_until_pass(
                test_path=str(test_file),
                max_iterations=3,
                build_source_context_fn=mock_context_fn,
            )

            # Verify test passes immediately without refinement
            assert result["success"] is True
            assert result["iterations"] == 1  # Only first iteration needed
            assert result["final_status"] == "passed"
            assert "passing after 1 iteration" in result["refinement_details"]

            # Verify refine port was never called
            assert self.refine_port.call_count == 0

    @pytest.mark.asyncio
    async def test_refine_until_pass_with_llm_no_change_early_exit(self, tmp_path):
        """Test early exit when LLM returns llm_no_change."""
        # Create failing test file
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_example():\n    assert False")

        # Configure refine port to return llm_no_change
        self.refine_port.responses = [
            {
                "success": False,
                "error": "LLM returned identical content to input",
                "iterations_used": 1,
                "final_status": "llm_no_change",
            }
        ]

        refiner = PytestRefiner(
            self.refine_port,
            self.telemetry_port,
            self.executor,
            config=self.config,
        )

        # Mock failing pytest run
        with patch.object(refiner, "run_pytest") as mock_pytest:
            mock_pytest.return_value = {
                "stdout": "FAILED",
                "stderr": "AssertionError: assert False",
                "returncode": 1,
                "command": "python -m pytest test_example.py",
            }

            async def mock_context_fn(test_path, content):
                return {"related_files": ["src/module.py"]}

            result = await refiner.refine_until_pass(
                test_path=str(test_file),
                max_iterations=3,
                build_source_context_fn=mock_context_fn,
            )

            # Verify early exit on llm_no_change
            assert result["success"] is False
            assert result["iterations"] == 1
            assert result["final_status"] == "no_change_detected"
            assert "LLM explicitly returned no changes" in result["error"]

            # Verify refine port was called once
            assert self.refine_port.call_count == 1

    @pytest.mark.asyncio
    async def test_refine_until_pass_configurable_pytest_args(self, tmp_path):
        """Test that configurable pytest args are used."""
        # Create test file
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_example():\n    assert False")

        # Configure custom pytest args
        custom_config = RefineConfig()
        custom_config.pytest_args_for_refinement = [
            "--verbose",
            "--tb=long",
            "--maxfail=1",
        ]

        refiner = PytestRefiner(
            self.refine_port,
            self.telemetry_port,
            self.executor,
            config=custom_config,
        )

        # Mock pytest run to capture args
        with patch.object(refiner, "run_pytest") as mock_pytest:
            mock_pytest.return_value = {
                "stdout": "PASSED",
                "stderr": "",
                "returncode": 0,
                "command": "python -m pytest",
            }

            async def mock_context_fn(test_path, content):
                return {}

            await refiner.refine_until_pass(
                test_path=str(test_file),
                max_iterations=1,
                build_source_context_fn=mock_context_fn,
            )

            # Verify pytest was called with custom args
            mock_pytest.assert_called_once_with(str(test_file))

            # Check that the refiner has the custom args configured
            assert refiner._pytest_args == ["--verbose", "--tb=long", "--maxfail=1"]

    @pytest.mark.asyncio
    async def test_refine_until_pass_successful_refinement(self, tmp_path):
        """Test successful refinement flow."""
        # Create failing test file
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_example():\n    assert False")

        # Configure refine port to succeed on first try
        self.refine_port.responses = [
            {
                "success": True,
                "refined_content": "def test_example():\n    assert True",
                "iterations_used": 1,
                "final_status": "success",
            }
        ]

        refiner = PytestRefiner(
            self.refine_port,
            self.telemetry_port,
            self.executor,
            config=self.config,
        )

        # Mock pytest run: fail first, pass after refinement
        pytest_calls = [
            {
                "stdout": "FAILED",
                "stderr": "AssertionError: assert False",
                "returncode": 1,
                "command": "python -m pytest test_example.py",
            },
            {
                "stdout": "PASSED",
                "stderr": "",
                "returncode": 0,
                "command": "python -m pytest test_example.py",
            },
        ]

        with patch.object(refiner, "run_pytest") as mock_pytest:
            mock_pytest.side_effect = pytest_calls

            async def mock_context_fn(test_path, content):
                return {"related_files": ["src/module.py"]}

            result = await refiner.refine_until_pass(
                test_path=str(test_file),
                max_iterations=3,
                build_source_context_fn=mock_context_fn,
            )

            # Verify successful refinement
            assert result["success"] is True
            assert (
                result["iterations"] == 2
            )  # One failed run + one successful run = 2 iterations
            assert result["final_status"] == "passed"

            # Verify refine port was called with correct parameters
            assert self.refine_port.call_count == 1
            assert self.refine_port.last_call_args["test_file"] == test_file
            assert "AssertionError" in self.refine_port.last_call_args["failure_output"]

    @pytest.mark.asyncio
    async def test_refine_until_pass_max_iterations_exceeded(self, tmp_path):
        """Test behavior when max iterations are exceeded."""
        # Create failing test file
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_example():\n    assert False")

        # Configure refine port to keep failing
        self.refine_port.responses = [
            {
                "success": True,
                "refined_content": "def test_example():\n    assert False  # still broken",
                "iterations_used": 1,
                "final_status": "success",
            },
            {
                "success": True,
                "refined_content": "def test_example():\n    assert False  # still broken v2",
                "iterations_used": 1,
                "final_status": "success",
            },
            {
                "success": True,
                "refined_content": "def test_example():\n    assert False  # still broken v3",
                "iterations_used": 1,
                "final_status": "success",
            },
        ]

        refiner = PytestRefiner(
            self.refine_port,
            self.telemetry_port,
            self.executor,
            config=self.config,
        )

        # Mock pytest to keep failing
        with patch.object(refiner, "run_pytest") as mock_pytest:
            mock_pytest.return_value = {
                "stdout": "FAILED",
                "stderr": "AssertionError: assert False",
                "returncode": 1,
                "command": "python -m pytest test_example.py",
            }

            async def mock_context_fn(test_path, content):
                return {"related_files": ["src/module.py"]}

            result = await refiner.refine_until_pass(
                test_path=str(test_file),
                max_iterations=2,
                build_source_context_fn=mock_context_fn,
            )

            # Verify max iterations behavior
            assert result["success"] is False
            assert result["iterations"] == 2  # Hit max iterations
            assert result["final_status"] == "failed"  # Still failed at end
            assert "Maximum refinement iterations" in result["error"]

            # Verify refine port was called for each iteration
            assert self.refine_port.call_count == 2

    @pytest.mark.asyncio
    async def test_refine_until_pass_exponential_backoff(self, tmp_path):
        """Test exponential backoff between successful refinement attempts."""
        # Create failing test file
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_example():\n    assert False")

        # Configure refine port to succeed multiple times
        self.refine_port.responses = [
            {
                "success": True,
                "refined_content": "def test_example():\n    assert False  # iteration 1",
                "iterations_used": 1,
                "final_status": "success",
            },
            {
                "success": True,
                "refined_content": "def test_example():\n    assert True  # iteration 2",
                "iterations_used": 1,
                "final_status": "success",
            },
        ]

        # Set short backoff for testing
        refiner = PytestRefiner(
            self.refine_port,
            self.telemetry_port,
            self.executor,
            config=self.config,
            backoff_sec=0.1,  # Short backoff for test speed
        )

        # Mock pytest: fail twice, then pass
        pytest_calls = [
            {
                "stdout": "FAILED",
                "stderr": "AssertionError",
                "returncode": 1,
                "command": "pytest",
            },
            {
                "stdout": "FAILED",
                "stderr": "AssertionError",
                "returncode": 1,
                "command": "pytest",
            },
            {"stdout": "PASSED", "stderr": "", "returncode": 0, "command": "pytest"},
        ]

        with patch.object(refiner, "run_pytest") as mock_pytest:
            mock_pytest.side_effect = pytest_calls

            async def mock_context_fn(test_path, content):
                return {"related_files": ["src/module.py"]}

            # Measure time to verify backoff
            import time

            start_time = time.time()

            result = await refiner.refine_until_pass(
                test_path=str(test_file),
                max_iterations=3,
                build_source_context_fn=mock_context_fn,
            )

            elapsed_time = time.time() - start_time

            # Verify successful completion
            assert result["success"] is True
            assert (
                result["iterations"] == 3
            )  # Three pytest runs (2 failures + 1 success = 3 iterations)
            assert result["final_status"] == "passed"

            # Verify backoff occurred (should be at least base backoff time)
            # Backoff progression: 0.1 * (2^0) = 0.1s for first iteration
            assert elapsed_time >= 0.1

    @pytest.mark.asyncio
    async def test_refine_until_pass_telemetry_attributes(self, tmp_path):
        """Test that telemetry attributes are set correctly."""
        # Create test file
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_example():\n    assert False")

        refiner = PytestRefiner(
            self.refine_port,
            self.telemetry_port,
            self.executor,
            config=self.config,
        )

        # Mock passing pytest (immediate success)
        with patch.object(refiner, "run_pytest") as mock_pytest:
            mock_pytest.return_value = {
                "stdout": "PASSED",
                "stderr": "",
                "returncode": 0,
                "command": "python -m pytest test_example.py",
            }

            async def mock_context_fn(test_path, content):
                return {}

            await refiner.refine_until_pass(
                test_path=str(test_file),
                max_iterations=3,
                build_source_context_fn=mock_context_fn,
            )

            # Verify telemetry span was created
            assert len(self.telemetry_port.spans) >= 1
            span_name, span_mock = self.telemetry_port.spans[0]
            assert span_name == "refine_test_file"

            # Verify span attributes were set
            span_mock.set_attribute.assert_any_call("test_file", str(test_file))
            span_mock.set_attribute.assert_any_call("max_iterations", 3)
            span_mock.set_attribute.assert_any_call("refinement_successful", True)

    @pytest.mark.asyncio
    async def test_format_pytest_failure_output(self):
        """Test pytest failure output formatting."""
        refiner = PytestRefiner(
            self.refine_port,
            self.telemetry_port,
            self.executor,
        )

        pytest_result = {
            "command": "python -m pytest test_example.py -v",
            "returncode": 1,
            "stdout": "test_example.py::test_function FAILED",
            "stderr": "AssertionError: assert False",
            "error": None,
        }

        formatted_output = refiner.format_pytest_failure_output(pytest_result)

        # Verify all components are included
        assert "Command: python -m pytest test_example.py -v" in formatted_output
        assert "Exit Code: 1" in formatted_output
        assert "Test Output:" in formatted_output
        assert "test_example.py::test_function FAILED" in formatted_output
        assert "Error Output:" in formatted_output
        assert "AssertionError: assert False" in formatted_output

    @pytest.mark.asyncio
    async def test_run_pytest_custom_args(self, tmp_path):
        """Test that run_pytest uses configured pytest args."""
        # Create test file
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_example():\n    assert True")

        # Configure custom pytest args
        custom_config = RefineConfig()
        custom_config.pytest_args_for_refinement = ["--tb=no", "-q"]

        refiner = PytestRefiner(
            self.refine_port,
            self.telemetry_port,
            self.executor,
            config=custom_config,
        )

        # Mock the underlying async runner
        with patch(
            "testcraft.adapters.io.async_runner.run_python_module_async_with_executor"
        ) as mock_runner:
            mock_runner.return_value = ("PASSED", "", 0)

            await refiner.run_pytest(str(test_file))

            # Verify custom args were passed
            mock_runner.assert_called_once()
            args = mock_runner.call_args[1]["args"]
            assert str(test_file) in args
            assert "--tb=no" in args
            assert "-q" in args


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
