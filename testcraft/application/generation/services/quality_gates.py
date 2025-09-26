"""
Quality Gates Service for test generation validation.

Implements the quality gates specified in the Context Assembly Specification:
1. Import gate: canonical import as first non-comment import
2. Bootstrap gate: ensure conftest.py exists if needed
3. Compile gate: pytest imports succeed
4. Determinism gate: pytest runs twice with same seed → identical results
5. Coverage gate: coverage delta measured and gaps fed to REFINE
6. Mutation gate: mutation sampling (optional)

This service orchestrates existing components like GeneratorGuardrails,
CoverageEvaluator, and mutation testing infrastructure.
"""

from __future__ import annotations

import ast
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from testcraft.config.models import DeterminismConfig, QualityConfig
from testcraft.domain.models import GenerationResult, ImportMap

from .bootstrap_runner import BootstrapRunner, BootstrapStrategy
from .coverage_evaluator import CoverageEvaluator

logger = logging.getLogger(__name__)


class QualityGateResult:
    """Result of a quality gate check."""

    def __init__(
        self,
        gate_name: str,
        passed: bool,
        message: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.gate_name = gate_name
        self.passed = passed
        self.message = message
        self.metadata = metadata or {}


class QualityGatesService:
    """
    Orchestrates quality gates for test generation.

    Integrates with existing services to provide comprehensive validation
    of generated tests according to the Context Assembly Specification.
    """

    def __init__(
        self,
        enriched_context: dict[str, Any],
        generation_result: GenerationResult,
        import_map: ImportMap,
        coverage_evaluator: CoverageEvaluator,
        bootstrap_runner: BootstrapRunner,
        quality_config: QualityConfig,
        enable_mutation_sampling: bool | None = None,
        determinism_config: DeterminismConfig | None = None,
    ):
        """
        Initialize the quality gates service.

        Args:
            enriched_context: Context from EnrichedContextBuilder
            generation_result: Result of test generation
            import_map: Import map for the target module
            coverage_evaluator: Service for coverage measurement
            bootstrap_runner: Service for bootstrap management
            quality_config: Quality configuration settings
            enable_mutation_sampling: Whether to run mutation sampling (overrides config if provided)
            determinism_config: Configuration for determinism testing
        """
        self.enriched_context = enriched_context
        self.generation_result = generation_result
        self.import_map = import_map
        self.coverage_evaluator = coverage_evaluator
        self.bootstrap_runner = bootstrap_runner
        self.quality_config = quality_config
        self.enable_mutation_sampling = (
            enable_mutation_sampling
            if enable_mutation_sampling is not None
            else quality_config.enable_mutation_gate
        )
        self.determinism_config = determinism_config or DeterminismConfig()
        self.test_content = generation_result.content
        self.test_file_path = generation_result.file_path

    def run_all_gates(self) -> tuple[bool, list[QualityGateResult]]:
        """
        Run all quality gates and return results.

        Returns:
            Tuple of (overall_success, list_of_gate_results)
        """
        results = []

        # Import gate
        if self.quality_config.enable_import_gate:
            results.append(self._import_gate())
        else:
            results.append(
                QualityGateResult(
                    "import_gate",
                    passed=True,
                    message="Import gate disabled",
                    metadata={"skipped": True},
                )
            )

        # Bootstrap gate
        if self.quality_config.enable_bootstrap_gate:
            results.append(self._bootstrap_gate())
        else:
            results.append(
                QualityGateResult(
                    "bootstrap_gate",
                    passed=True,
                    message="Bootstrap gate disabled",
                    metadata={"skipped": True},
                )
            )

        # Compile gate
        if self.quality_config.enable_compile_gate:
            results.append(self._compile_gate())
        else:
            results.append(
                QualityGateResult(
                    "compile_gate",
                    passed=True,
                    message="Compile gate disabled",
                    metadata={"skipped": True},
                )
            )

        # Determinism gate
        if self.quality_config.enable_determinism_gate:
            results.append(self._determinism_gate())
        else:
            results.append(
                QualityGateResult(
                    "determinism_gate",
                    passed=True,
                    message="Determinism gate disabled",
                    metadata={"skipped": True},
                )
            )

        # Coverage gate
        if self.quality_config.enable_coverage_gate:
            results.append(self._coverage_gate())
        else:
            results.append(
                QualityGateResult(
                    "coverage_gate",
                    passed=True,
                    message="Coverage gate disabled",
                    metadata={"skipped": True},
                )
            )

        # Mutation gate (optional)
        if self.enable_mutation_sampling:
            results.append(self._mutation_gate())
        else:
            results.append(
                QualityGateResult(
                    "mutation_gate",
                    passed=True,
                    message="Mutation sampling disabled",
                    metadata={"skipped": True},
                )
            )

        # Overall success
        overall_success = all(result.passed for result in results)

        return overall_success, results

    def _import_gate(self) -> QualityGateResult:
        """Validate that canonical import is first non-comment import."""
        try:
            tree = ast.parse(self.test_content)

            # Find first non-comment import
            first_import = None
            for node in ast.walk(tree):
                if isinstance(node, ast.Import | ast.ImportFrom) and node.module:
                    # Check if it's a comment (AST doesn't preserve comments)
                    # We check if the line starts with # by looking at the source
                    lines = self.test_content.split("\n")
                    line_num = getattr(node, "lineno", 1) - 1  # 0-based index
                    if line_num < len(lines):
                        line = lines[line_num].strip()
                        if not line.startswith("#"):
                            first_import = node
                            break

            if first_import is None:
                return QualityGateResult(
                    "import_gate",
                    passed=False,
                    message="No import statements found in generated test",
                )

            # Check if first import matches canonical import
            canonical_import = self.import_map.target_import
            if isinstance(first_import, ast.Import):
                actual_import = first_import.names[0].name
            else:  # ast.ImportFrom
                actual_import = first_import.module or ""

            if actual_import != canonical_import:
                return QualityGateResult(
                    "import_gate",
                    passed=False,
                    message=f"First import '{actual_import}' != canonical import '{canonical_import}'",
                    metadata={"expected": canonical_import, "actual": actual_import},
                )

            return QualityGateResult(
                "import_gate",
                passed=True,
                message=f"Canonical import '{canonical_import}' correctly placed first",
            )

        except SyntaxError as e:
            return QualityGateResult(
                "import_gate",
                passed=False,
                message=f"Syntax error in generated test: {e}",
                metadata={"error": str(e)},
            )

    def _bootstrap_gate(self) -> QualityGateResult:
        """Ensure bootstrap requirements are met."""
        tests_dir = Path(self.test_file_path).parent

        if self.import_map.needs_bootstrap:
            strategy = self.bootstrap_runner.ensure_bootstrap(
                self.import_map, tests_dir
            )

            if strategy == BootstrapStrategy.NO_BOOTSTRAP:
                return QualityGateResult(
                    "bootstrap_gate",
                    passed=False,
                    message="Bootstrap needed but not applied",
                )

            return QualityGateResult(
                "bootstrap_gate",
                passed=True,
                message=f"Bootstrap applied using {strategy.value}",
                metadata={"strategy": strategy.value},
            )

        return QualityGateResult(
            "bootstrap_gate",
            passed=True,
            message="No bootstrap needed",
            metadata={"needs_bootstrap": False},
        )

    def _compile_gate(self) -> QualityGateResult:
        """Ensure pytest can import the generated test without errors."""
        try:
            # Write test to temporary file for testing
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as temp_file:
                temp_file.write(self.test_content)
                temp_file_path = temp_file.name

            # Try to import the test file
            result = subprocess.run(
                ["python", "-c", f"import {Path(temp_file_path).stem}"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Clean up
            Path(temp_file_path).unlink()

            if result.returncode != 0:
                return QualityGateResult(
                    "compile_gate",
                    passed=False,
                    message=f"Cannot import generated test: {result.stderr}",
                    metadata={"stderr": result.stderr, "stdout": result.stdout},
                )

            return QualityGateResult(
                "compile_gate",
                passed=True,
                message="Generated test imports successfully",
            )

        except subprocess.TimeoutExpired:
            return QualityGateResult(
                "compile_gate",
                passed=False,
                message="Import test timed out",
                metadata={"timeout": True},
            )
        except Exception as e:
            return QualityGateResult(
                "compile_gate",
                passed=False,
                message=f"Error testing import: {e}",
                metadata={"error": str(e)},
            )

    def _determinism_gate(self) -> QualityGateResult:
        """Run pytest twice with same seed and compare results."""
        try:
            # Write test to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as temp_file:
                temp_file.write(self.test_content)
                temp_file_path = temp_file.name

            # Build pytest command with determinism settings
            pytest_cmd = ["pytest", temp_file_path, "-v", "--tb=short"]

            # Add determinism flags based on configuration
            if self.determinism_config.freeze_time:
                pytest_cmd.extend(["--freeze-time"])

            # Add random seed for reproducibility
            pytest_cmd.extend(["--randomly-seed", str(self.determinism_config.seed)])

            # Run pytest twice with same configuration
            logger.debug(
                f"Running determinism test with command: {' '.join(pytest_cmd)}"
            )

            run1_result = subprocess.run(
                pytest_cmd,
                capture_output=True,
                text=True,
                timeout=self.quality_config.determinism_timeout_seconds,
                env={**dict(os.environ), "TZ": self.determinism_config.tz},
            )

            run2_result = subprocess.run(
                pytest_cmd,
                capture_output=True,
                text=True,
                timeout=self.quality_config.determinism_timeout_seconds,
                env={**dict(os.environ), "TZ": self.determinism_config.tz},
            )

            # Clean up
            Path(temp_file_path).unlink()

            # Compare results comprehensively
            run1_success = run1_result.returncode == 0
            run2_success = run2_result.returncode == 0
            returncode_same = run1_result.returncode == run2_result.returncode
            stdout_same = run1_result.stdout == run2_result.stdout
            stderr_same = run1_result.stderr == run2_result.stderr

            # Tests are deterministic if all aspects match
            is_deterministic = (
                run1_success
                and run2_success
                and returncode_same
                and stdout_same
                and stderr_same
            )

            if not is_deterministic:
                # Determine which aspects differ
                differences = []
                if not run1_success or not run2_success:
                    differences.append("test_success")
                if not returncode_same:
                    differences.append("return_code")
                if not stdout_same:
                    differences.append("stdout")
                if not stderr_same:
                    differences.append("stderr")

                return QualityGateResult(
                    "determinism_gate",
                    passed=False,
                    message=f"Tests are non-deterministic: {', '.join(differences)} differ",
                    metadata={
                        "run1_success": run1_success,
                        "run2_success": run2_success,
                        "returncode_same": returncode_same,
                        "stdout_same": stdout_same,
                        "stderr_same": stderr_same,
                        "differences": differences,
                        "run1_returncode": run1_result.returncode,
                        "run2_returncode": run2_result.returncode,
                        "run1_stdout_length": len(run1_result.stdout),
                        "run2_stdout_length": len(run2_result.stdout),
                        "run1_stderr_length": len(run1_result.stderr),
                        "run2_stderr_length": len(run2_result.stderr),
                    },
                )

            return QualityGateResult(
                "determinism_gate",
                passed=True,
                message="Tests are deterministic",
                metadata={
                    "run1_success": run1_success,
                    "run2_success": run2_success,
                    "returncode_same": returncode_same,
                    "stdout_same": stdout_same,
                    "stderr_same": stderr_same,
                    "seed": self.determinism_config.seed,
                    "timezone": self.determinism_config.tz,
                    "freeze_time": self.determinism_config.freeze_time,
                },
            )

        except subprocess.TimeoutExpired:
            return QualityGateResult(
                "determinism_gate",
                passed=False,
                message="Determinism test timed out",
                metadata={"timeout": True},
            )
        except Exception as e:
            return QualityGateResult(
                "determinism_gate",
                passed=False,
                message=f"Error in determinism test: {e}",
                metadata={"error": str(e), "error_type": type(e).__name__},
            )

    def _coverage_gate(self) -> QualityGateResult:
        """Measure coverage delta and validate improvement."""
        try:
            # Get coverage data from the service
            # This should be called after the tests have been run
            # For now, we'll get it from the enriched context or calculate it
            coverage_delta = self.enriched_context.get("coverage_delta", {})

            # If no delta in context, try to calculate it using the coverage evaluator
            if not coverage_delta:
                try:
                    # This would need access to initial and final coverage data
                    # For now, return a warning that coverage analysis is not available
                    return QualityGateResult(
                        "coverage_gate",
                        passed=False,
                        message="Coverage delta calculation not available - need initial/final coverage data",
                        metadata={
                            "coverage_available": False,
                            "note": "Coverage gate requires integration with CoverageEvaluator to measure before/after coverage",
                        },
                    )
                except Exception as calc_error:
                    return QualityGateResult(
                        "coverage_gate",
                        passed=False,
                        message=f"Error calculating coverage delta: {calc_error}",
                        metadata={"error": str(calc_error), "calculation_error": True},
                    )

            line_delta = coverage_delta.get("line_coverage_delta", 0)
            branch_delta = coverage_delta.get("branch_coverage_delta", 0)

            # Check if coverage improved
            if line_delta <= 0 and branch_delta <= 0:
                return QualityGateResult(
                    "coverage_gate",
                    passed=False,
                    message="Coverage did not improve",
                    metadata={
                        "line_delta": line_delta,
                        "branch_delta": branch_delta,
                        "improvement": False,
                    },
                )

            return QualityGateResult(
                "coverage_gate",
                passed=True,
                message=f"Coverage improved: +{line_delta:.1%} lines, +{branch_delta:.1%} branches",
                metadata={
                    "line_delta": line_delta,
                    "branch_delta": branch_delta,
                    "improvement": True,
                    "gaps_identified": coverage_delta.get("gaps_identified", []),
                    "uncovered_branches": coverage_delta.get("uncovered_branches", []),
                },
            )

        except Exception as e:
            return QualityGateResult(
                "coverage_gate",
                passed=False,
                message=f"Error in coverage analysis: {e}",
                metadata={"error": str(e), "error_type": type(e).__name__},
            )

    def _mutation_gate(self) -> QualityGateResult:
        """Run mutation sampling on generated tests."""
        try:
            # This would integrate with the existing mutation testing system
            # For now, stub the implementation
            return QualityGateResult(
                "mutation_gate",
                passed=True,
                message="Mutation sampling stubbed (not yet implemented)",
                metadata={"stubbed": True},
            )

        except Exception as e:
            return QualityGateResult(
                "mutation_gate",
                passed=False,
                message=f"Error in mutation testing: {e}",
                metadata={"error": str(e)},
            )
