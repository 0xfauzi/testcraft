"""
PytestRefiner facade maintaining the original API.

This module provides the PytestRefiner class that delegates to the modular components
while preserving the exact same public interface for backward compatibility.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .....adapters.io.file_status_tracker import FileStatus, FileStatusTracker
from .....config.models import RefineConfig
from .....ports.refine_port import RefinePort
from .....ports.telemetry_port import TelemetryPort
from .....ports.writer_port import WriterPort
from .annotators import TestAnnotator
from .failure_parsing import FailureParser
from .llm_pipeline import LLMRefinementPipeline
from .runner import PytestRunner
from .writer import RefinementWriter

logger = logging.getLogger(__name__)


class PytestRefiner:
    """
    Service for pytest execution and test refinement.

    Provides pytest execution, failure formatting, and iterative refinement
    of test files until they pass or maximum iterations are reached.

    This is a facade that delegates to modular components while maintaining
    the exact same public API for backward compatibility.
    """

    def __init__(
        self,
        refine_port: RefinePort,
        telemetry_port: TelemetryPort,
        executor: ThreadPoolExecutor,
        config: RefineConfig | None = None,
        max_concurrent_refines: int = 2,
        backoff_sec: float = 0.2,
        status_tracker: Optional[FileStatusTracker] = None,
        writer_port: WriterPort | None = None,
    ):
        """
        Initialize the pytest refiner.

        Args:
            refine_port: Port for test refinement operations
            telemetry_port: Port for telemetry operations
            executor: Thread pool executor for async operations
            config: Refinement configuration for pytest args and settings
            max_concurrent_refines: Maximum concurrent pytest/refine operations
            backoff_sec: Backoff time between refinement iterations
            status_tracker: Optional file status tracker for live updates
            writer_port: Optional writer port for safe file operations
        """
        self._refine = refine_port
        self._telemetry = telemetry_port
        self._executor = executor
        self._config = config or RefineConfig()
        self._backoff_sec = backoff_sec
        self._status_tracker = status_tracker
        self._writer = writer_port
        
        # Create semaphore to limit concurrent pytest operations
        self._refine_semaphore = asyncio.Semaphore(max_concurrent_refines)

        # Initialize modular components
        self._runner = PytestRunner(executor, self._config)
        self._failure_parser = FailureParser()
        self._writer_helper = RefinementWriter(writer_port)
        self._annotator = TestAnnotator(self._config, telemetry_port, writer_port)
        self._pipeline = LLMRefinementPipeline(
            refine_port=refine_port,
            telemetry_port=telemetry_port,
            runner=self._runner,
            annotator=self._annotator,
            config=self._config,
            backoff_sec=backoff_sec,
            status_tracker=status_tracker,
        )

    async def run_pytest(self, test_path: str) -> dict[str, Any]:
        """
        Run pytest on a specific test file and return results.

        Uses the async_runner abstraction which wraps subprocess patterns
        for async workflows.

        Args:
            test_path: Path to the test file to run

        Returns:
            Dictionary with pytest execution results including stdout, stderr, returncode
        """
        return await self._runner.run_pytest(test_path)

    @staticmethod
    def extract_import_path_from_failure(failure_output: str) -> str:
        """
        Extract the active import path from pytest failure traceback.
        
        This method looks for import paths in traceback lines to determine the 
        actual module path used at runtime, which is more reliable than source 
        tree aliases when mocking/patching.
        
        Args:
            failure_output: Pytest failure output containing traceback
            
        Returns:
            Detected import path or empty string if not found
        """
        return FailureParser.extract_import_path_from_failure(failure_output)

    def format_pytest_failure_output(self, pytest_result: dict[str, Any]) -> str:
        """
        Format pytest execution results into a clean failure output string.

        Args:
            pytest_result: Results from run_pytest

        Returns:
            Formatted failure output suitable for LLM refinement
        """
        return self._failure_parser.format_pytest_failure_output(pytest_result)

    def detect_xfail_in_output(self, pytest_output: str) -> bool:
        """
        Detect if pytest output contains XFAIL markers.
        
        Args:
            pytest_output: Combined stdout/stderr from pytest
            
        Returns:
            True if XFAIL markers are detected
        """
        return self._runner.detect_xfail_in_output(pytest_output)

    def detect_suspected_prod_bug(self, failure_output: str) -> dict[str, Any] | None:
        """
        DEPRECATED: Basic pattern matching for production bugs.
        This is only used as a fallback when LLM analysis is not available.
        The LLM should be the primary source of bug detection and description.
        
        Args:
            failure_output: Formatted pytest failure output
            
        Returns:
            Dictionary with bug detection info or None
        """
        return self._failure_parser.detect_suspected_prod_bug(failure_output)

    async def refine_until_pass(
        self,
        test_path: str,
        max_iterations: int,
        build_source_context_fn: Callable[[Path, str], Awaitable[dict[str, Any] | None]],
    ) -> dict[str, Any]:
        """
        Refine a test file through pytest execution and LLM refinement.

        This method implements the complete refinement workflow:
        1. Run pytest to get failure output
        2. If tests pass, return success
        3. If tests fail, use refine port to fix failures
        4. Repeat until max iterations or tests pass
        5. Detect no-change scenarios to avoid infinite loops

        Args:
            test_path: Path to the test file to refine
            max_iterations: Maximum number of refinement iterations
            build_source_context_fn: Function to build source context for refinement

        Returns:
            Dictionary with refinement results including success status,
            iterations used, final pytest status, and any errors
        """
        return await self._pipeline.refine_until_pass(
            test_path=test_path,
            max_iterations=max_iterations,
            build_source_context_fn=build_source_context_fn,
            refine_semaphore=self._refine_semaphore,
        )

    # Private methods that were used internally - delegate to components

    def _classify_pytest_result(self, output: str, returncode: int) -> dict[str, Any]:
        """
        Classify pytest outcome to decide if LLM refinement should proceed.

        We skip refinement and do NOT send failures to the LLM when pytest
        didn't actually run tests (collection/usage/internal errors), because
        the LLM cannot fix environment or import problems reliably.
        """
        return self._runner.classify_pytest_result(output, returncode)

    async def _mark_test_with_bug_info(
        self, test_file: Path, bug_detection: dict[str, Any], failure_output: str
    ) -> None:
        """
        Mark the test file with production bug information.
        
        Args:
            test_file: Path to the test file
            bug_detection: Bug detection information (should contain 'description' from LLM)
            failure_output: The pytest failure output
        """
        await self._annotator.mark_test_with_bug_info(test_file, bug_detection, failure_output)

    async def _annotate_failed_test(
        self,
        test_file: Path,
        failure_output: str,
        reason_status: str,
        iterations: int,
        fix_instructions: str | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Build and insert a high-visibility annotation into the test file when refinement fails.

        Respects RefineConfig:
        - annotate_failed_tests
        - annotation_placement ("top" | "bottom")
        - annotation_include_failure_excerpt
        - annotation_max_failure_chars
        - annotation_style ("docstring" | "hash")
        - include_llm_fix_instructions
        """
        await self._annotator.annotate_failed_test(
            test_file=test_file,
            failure_output=failure_output,
            reason_status=reason_status,
            iterations=iterations,
            fix_instructions=fix_instructions,
            extra=extra,
        )

    async def _create_bug_report(
        self, test_path: str, bug_detection: dict[str, Any], 
        failure_output: str, iteration: int
    ) -> None:
        """
        Create a detailed bug report file.
        
        Args:
            test_path: Path to the test file
            bug_detection: Bug detection information (should contain 'description' from LLM)
            failure_output: The pytest failure output
            iteration: Current refinement iteration
        """
        await self._annotator.create_bug_report(test_path, bug_detection, failure_output, iteration)

    def _format_fix_instructions_as_todos(self, instructions: str) -> str:
        """Convert fix instructions to TODO checklist format."""
        return self._annotator._format_fix_instructions_as_todos(instructions)

    def _extract_failing_tests_from_output(self, failure_output: str) -> list[str]:
        """Extract specific failing test names from pytest output."""
        return self._failure_parser.extract_failing_tests_from_output(failure_output)

    def _extract_failure_context(self, failure_output: str) -> dict[str, Any]:
        """Extract targeted failure context from pytest output."""
        return self._failure_parser.extract_failure_context(failure_output)

    def _generate_enhanced_fix_instructions(self, 
                                           failing_tests: list[str], 
                                           failure_context: dict[str, Any], 
                                           original_instructions: str) -> str:
        """Generate enhanced, specific fix instructions."""
        return self._annotator._generate_enhanced_fix_instructions(
            failing_tests, failure_context, original_instructions
        )

    async def _write_with_writer_port(self, test_file: Path, content: str) -> None:
        """Write using the writer port."""
        await self._writer_helper.write_with_writer_port(test_file, content)

    async def _write_with_fallback(self, test_file: Path, content: str) -> None:
        """Fallback writing method."""
        await self._writer_helper.write_with_fallback(test_file, content)

    def _indent_text(self, text: str, prefix: str) -> str:
        """Indent text with a prefix on each line."""
        return self._annotator._indent_text(text, prefix)
