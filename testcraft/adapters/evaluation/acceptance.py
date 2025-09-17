"""
Acceptance check functions for test evaluation.

This module provides automated acceptance checks including syntax validation,
import checking, pytest execution, and coverage improvement measurement.
"""

import ast
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ...ports.coverage_port import CoveragePort
from ...ports.evaluation_port import AcceptanceResult
from ..io.safety import SafetyPolicies

logger = logging.getLogger(__name__)


class AcceptanceChecker:
    """Handles automated acceptance checks for generated tests."""

    def __init__(
        self,
        coverage_adapter: CoveragePort,
        project_root: Path,
        safety_enabled: bool = True,
    ):
        """
        Initialize acceptance checker.

        Args:
            coverage_adapter: Coverage measurement adapter
            project_root: Project root for safety validation
            safety_enabled: Whether to enforce safety policies
        """
        self.coverage_adapter = coverage_adapter
        self.project_root = project_root
        self.safety_enabled = safety_enabled

    def run_acceptance_checks(
        self,
        test_content: str,
        source_file: str,
        baseline_coverage: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AcceptanceResult:
        """
        Run comprehensive automated acceptance checks on test content.

        This performs syntactic validation, import checking, pytest execution,
        and optional coverage improvement measurement.
        """
        logger.debug(f"Running acceptance checks for {source_file}")

        try:
            error_messages = []

            # 1. Syntactic validation
            syntactically_valid = self._check_syntax(test_content, error_messages)

            # 2. Import validation (requires syntactic validity)
            imports_successfully = False
            if syntactically_valid:
                imports_successfully = self._check_imports(
                    test_content, source_file, error_messages
                )

            # 3. Pytest execution (requires import success)
            pytest_passes = False
            if imports_successfully:
                pytest_passes = self._run_pytest(
                    test_content, source_file, error_messages
                )

            # 4. Optional coverage improvement measurement
            coverage_improvement = None
            if pytest_passes and baseline_coverage:
                coverage_improvement = self._measure_coverage_improvement(
                    test_content, source_file, baseline_coverage
                )

            result = AcceptanceResult(
                syntactically_valid=syntactically_valid,
                imports_successfully=imports_successfully,
                pytest_passes=pytest_passes,
                coverage_improvement=coverage_improvement,
                error_messages=error_messages,
            )

            logger.debug(f"Acceptance checks completed: {result.all_checks_pass}")
            return result

        except Exception as e:
            logger.error(f"Acceptance checks failed: {e}")
            raise

    def _check_syntax(self, test_content: str, error_messages: list[str]) -> bool:
        """Check if test content is syntactically valid Python."""
        try:
            ast.parse(test_content)
            return True
        except SyntaxError as e:
            error_messages.append(f"Syntax error: {e}")
            return False
        except Exception as e:
            error_messages.append(f"Parse error: {e}")
            return False

    def _check_imports(
        self, test_content: str, source_file: str, error_messages: list[str]
    ) -> bool:
        """Check if test content imports successfully."""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(test_content)
                temp_file = Path(f.name)

            # Validate file path safety if enabled
            if self.safety_enabled:
                SafetyPolicies.validate_file_path(temp_file, self.project_root)

            # Try to compile the test file
            result = subprocess.run(
                ["python", "-m", "py_compile", str(temp_file)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            temp_file.unlink()  # Clean up

            if result.returncode == 0:
                return True
            else:
                error_messages.append(f"Import error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            error_messages.append("Import check timed out")
            return False
        except Exception as e:
            error_messages.append(f"Import check failed: {e}")
            return False

    def _run_pytest(
        self, test_content: str, source_file: str, error_messages: list[str]
    ) -> bool:
        """Run pytest on the test content."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                test_file = temp_path / "test_generated.py"
                test_file.write_text(test_content, encoding="utf-8")

                # Run pytest with appropriate flags
                result = subprocess.run(
                    [
                        "python",
                        "-m",
                        "pytest",
                        str(test_file),
                        "-v",
                        "--tb=short",
                        "--no-header",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=self.project_root,
                )

                if result.returncode == 0:
                    return True
                else:
                    error_messages.append(f"Pytest failed: {result.stdout}")
                    if result.stderr:
                        error_messages.append(f"Pytest stderr: {result.stderr}")
                    return False

        except subprocess.TimeoutExpired:
            error_messages.append("Pytest execution timed out")
            return False
        except Exception as e:
            error_messages.append(f"Pytest execution failed: {e}")
            return False

    def _measure_coverage_improvement(
        self, test_content: str, source_file: str, baseline_coverage: dict[str, Any]
    ) -> float | None:
        """Measure coverage improvement from baseline."""
        try:
            # This is a simplified implementation
            # In practice, you'd run coverage with the new test and compare
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                test_file = temp_path / "test_generated.py"
                test_file.write_text(test_content, encoding="utf-8")

                # Measure coverage with new test
                new_coverage = self.coverage_adapter.measure_coverage(
                    [source_file], [str(test_file)]
                )

                source_file_key = str(Path(source_file).resolve())
                if source_file_key in new_coverage:
                    new_line_coverage = new_coverage[source_file_key].line_coverage
                    baseline_line_coverage = baseline_coverage.get("line_coverage", 0.0)

                    improvement = new_line_coverage - baseline_line_coverage
                    return max(0.0, improvement)  # Don't report negative improvements

                return None

        except Exception as e:
            logger.warning(f"Could not measure coverage improvement: {e}")
            return None
