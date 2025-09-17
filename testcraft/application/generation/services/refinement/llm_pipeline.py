"""
LLM integration and refinement orchestration.

Handles the core refinement loop with LLM integration, status tracking, and error handling.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

from .....adapters.io.file_status_tracker import FileStatus, FileStatusTracker
from .....config.models import RefineConfig
from .....ports.refine_port import RefinePort
from .....ports.telemetry_port import TelemetryPort
from .annotators import TestAnnotator
from .failure_parsing import FailureParser
from .runner import PytestRunner

logger = logging.getLogger(__name__)


class LLMRefinementPipeline:
    """Orchestrates the LLM-based refinement pipeline."""

    def __init__(
        self,
        refine_port: RefinePort,
        telemetry_port: TelemetryPort,
        runner: PytestRunner,
        annotator: TestAnnotator,
        config: RefineConfig,
        backoff_sec: float = 0.2,
        status_tracker: FileStatusTracker | None = None,
    ):
        """
        Initialize the LLM refinement pipeline.

        Args:
            refine_port: Port for test refinement operations
            telemetry_port: Port for telemetry operations
            runner: Pytest runner instance
            annotator: Test annotator instance
            config: Refinement configuration
            backoff_sec: Backoff time between refinement iterations
            status_tracker: Optional file status tracker for live updates
        """
        self._refine = refine_port
        self._telemetry = telemetry_port
        self._runner = runner
        self._annotator = annotator
        self._config = config
        self._backoff_sec = backoff_sec
        self._status_tracker = status_tracker
        self._failure_parser = FailureParser()

    async def refine_until_pass(
        self,
        test_path: str,
        max_iterations: int,
        build_source_context_fn: Callable[[Path, str], Awaitable[dict[str, Any] | None]],
        refine_semaphore: asyncio.Semaphore,
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
            refine_semaphore: Semaphore to limit concurrent refinement operations

        Returns:
            Dictionary with refinement results including success status,
            iterations used, final pytest status, and any errors
        """
        test_file = Path(test_path)

        # Track content between iterations to detect no-change scenarios
        previous_content = None
        if test_file.exists():
            try:
                previous_content = test_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to read test file %s: %s", test_path, e)

        # Use semaphore to limit concurrent refinement operations
        async with refine_semaphore:
            with self._telemetry.create_child_span("refine_test_file") as span:
                span.set_attribute("test_file", test_path)
                span.set_attribute("max_iterations", max_iterations)
                span.set_attribute("backoff_sec", self._backoff_sec)
                
                # Update status to testing
                if self._status_tracker:
                    self._status_tracker.update_file_status(
                        test_path,
                        FileStatus.TESTING,
                        operation="Initial Testing",
                        step="Running pytest to check current status",
                        progress=80.0
                    )

                # Track last fix metadata for annotation purposes
                last_fix_instructions = None
                last_active_import_path = ""
                last_preflight = ""

                for iteration in range(max_iterations):
                    span.set_attribute(f"iteration_{iteration}_started", True)
                    
                    # Update status for each iteration
                    if self._status_tracker:
                        if iteration == 0:
                            self._status_tracker.update_file_status(
                                test_path,
                                FileStatus.TESTING,
                                operation="Testing",
                                step=f"Running initial pytest check",
                                progress=80.0 + (iteration * 5)
                            )
                        else:
                            self._status_tracker.update_file_status(
                                test_path,
                                FileStatus.REFINING,
                                operation=f"Refinement {iteration}",
                                step=f"Running pytest (iteration {iteration + 1})",
                                progress=80.0 + (iteration * 5)
                            )

                    try:
                        # Step 1: Run pytest to get current test status
                        pytest_result = await self._runner.run_pytest(test_path)
                        span.set_attribute(
                            f"iteration_{iteration}_pytest_returncode",
                            pytest_result["returncode"],
                        )

                        # Step 2: Check if tests are now passing
                        if pytest_result["returncode"] == 0:
                            # Check for XFAIL markers if configured to fail on them
                            combined_output = pytest_result.get("stdout", "") + pytest_result.get("stderr", "")
                            
                            if self._config.fail_on_xfail_markers and self._runner.detect_xfail_in_output(combined_output):
                                logger.warning(
                                    "Tests marked with XFAIL detected for %s, treating as failure (fail_on_xfail_markers=True)",
                                    test_path
                                )
                                span.set_attribute("xfail_detected", True)
                                
                                # Create a bug detection dict for XFAIL
                                xfail_bug_detection = {
                                    "suspected": True,
                                    "type": "xfail_marker",
                                    "pattern": "XFAIL/xfailed markers",
                                    "confidence": 0.9,
                                    "description": "Test contains XFAIL markers which may mask production bugs"
                                }
                                
                                # Mark the test and create report
                                await self._annotator.mark_test_with_bug_info(
                                    test_file,
                                    xfail_bug_detection,
                                    combined_output
                                )
                                
                                await self._annotator.create_bug_report(
                                    test_path,
                                    xfail_bug_detection,
                                    combined_output,
                                    iteration + 1
                                )
                                
                                # Update status to failed due to XFAIL
                                if self._status_tracker:
                                    self._status_tracker.update_file_status(
                                        test_path,
                                        FileStatus.FAILED,
                                        operation="XFAIL Detected",
                                        step="Tests contain xfail markers (potential production bug)",
                                        progress=0.0
                                    )
                                
                                return {
                                    "test_file": test_path,
                                    "success": False,
                                    "iterations": iteration + 1,
                                    "final_status": "xfail_detected",
                                    "error": "Tests contain xfail markers which may mask production bugs",
                                    "last_failure": combined_output,
                                    "suspected_prod_bug": "XFAIL markers detected - tests may be masking real bugs",
                                    "bug_report_created": True,
                                }
                            
                            # Tests pass! Refinement successful
                            span.set_attribute("refinement_successful", True)
                            span.set_attribute("final_iteration", iteration)
                            
                            # Update status to completed
                            if self._status_tracker:
                                self._status_tracker.update_file_status(
                                    test_path,
                                    FileStatus.COMPLETED,
                                    operation="Tests Passing",
                                    step=f"All tests pass after {iteration + 1} iteration(s)",
                                    progress=100.0
                                )
                                self._status_tracker.update_refinement_result(
                                    test_path, iteration + 1, True
                                )

                            return {
                                "test_file": test_path,
                                "success": True,
                                "iterations": iteration + 1,
                                "final_status": "passed",
                                "final_output": pytest_result.get("stdout", ""),
                                "refinement_details": f"Tests passing after {iteration + 1} iteration(s)",
                            }

                        # Step 3: Tests are failing. If failure is unrefinable, stop early.
                        if pytest_result.get("unrefinable", False):
                            category = pytest_result.get("failure_category", "unknown")
                            logger.warning(
                                "Skipping LLM refinement for %s due to unrefinable pytest failure: %s",
                                test_path,
                                category,
                            )
                            span.set_attribute("unrefinable_failure", category)
                            if self._status_tracker:
                                self._status_tracker.update_file_status(
                                    test_path,
                                    FileStatus.FAILED,
                                    operation="Pytest Failure (Unrefinable)",
                                    step=f"{category} — fix environment/imports before refinement",
                                    progress=0.0,
                                )
                            # Annotate failed test with unrefinable failure info
                            if self._config.annotate_failed_tests:
                                await self._annotator.annotate_failed_test(
                                    test_file=Path(test_path),
                                    failure_output=self._failure_parser.format_pytest_failure_output(pytest_result),
                                    reason_status=f"unrefinable_{category}",
                                    iterations=iteration + 1,
                                    fix_instructions=None,  # No LLM instructions for unrefinable
                                    extra={}
                                )
                            
                            return {
                                "test_file": test_path,
                                "success": False,
                                "iterations": iteration + 1,
                                "final_status": category,
                                "error": "Unrefinable pytest failure; fix environment/imports or test discovery",
                                "last_failure": self._failure_parser.format_pytest_failure_output(pytest_result),
                            }

                        # Otherwise attempt refinement
                        failure_output = self._failure_parser.format_pytest_failure_output(pytest_result)
                        
                        # Update status for refinement
                        if self._status_tracker:
                            self._status_tracker.update_file_status(
                                test_path,
                                FileStatus.REFINING,
                                operation=f"Refinement {iteration + 1}",
                                step="Using LLM to fix test failures",
                                progress=85.0 + (iteration * 3)
                            )

                        # Build source context for better refinement
                        # Read current test file content for context
                        try:
                            test_content = test_file.read_text(encoding='utf-8') if test_file.exists() else ""
                        except Exception as e:
                            logger.warning("Failed to read test file %s: %s", test_file, e)
                            test_content = ""
                        
                        source_context = await build_source_context_fn(test_file, test_content)

                        # Use refine port to fix the failures
                        refine_result = self._refine.refine_from_failures(
                            test_file=test_file,
                            failure_output=failure_output,
                            source_context=source_context,
                            max_iterations=self._config.max_retries + 1,  # Use config max_retries
                            timeout_seconds=int(self._config.max_total_minutes * 60),  # Convert minutes to seconds
                        )

                        # Update tracking metadata from refine result
                        last_fix_instructions = refine_result.get("fix_instructions")
                        last_active_import_path = refine_result.get("active_import_path", "")
                        last_preflight = refine_result.get("preflight_suggestions", "")

                        span.set_attribute(
                            f"iteration_{iteration}_refine_success",
                            refine_result.get("success", False),
                        )
                        span.set_attribute(
                            f"iteration_{iteration}_refine_final_status",
                            refine_result.get("final_status", "unknown"),
                        )

                        # Check for early stopping conditions
                        final_status = refine_result.get("final_status")
                        
                        # Handle production bug detection from LLM
                        if final_status == "prod_bug_suspected":
                            suspected_bug_desc = refine_result.get("suspected_prod_bug", "Production bug detected by LLM")
                            logger.warning(
                                "LLM detected production bug on iteration %d for %s: %s",
                                iteration + 1,
                                test_path,
                                suspected_bug_desc
                            )
                            span.set_attribute("prod_bug_suspected", True)
                            span.set_attribute("bug_description", suspected_bug_desc)
                            
                            # Create bug detection dict with LLM's analysis
                            bug_detection = {
                                "suspected": True,
                                "type": "llm_detected",
                                "pattern": "LLM analysis",
                                "confidence": 0.9,  # High confidence since LLM analyzed it
                                "description": suspected_bug_desc
                            }
                            
                            # Mark the test with production bug information
                            await self._annotator.mark_test_with_bug_info(
                                test_file,
                                bug_detection,
                                failure_output
                            )
                            
                            # Create a bug report file
                            await self._annotator.create_bug_report(
                                test_path,
                                bug_detection,
                                failure_output,
                                iteration + 1
                            )
                            
                            # Update status
                            if self._status_tracker:
                                self._status_tracker.update_file_status(
                                    test_path,
                                    FileStatus.FAILED,
                                    operation="Production Bug Detected by LLM",
                                    step=suspected_bug_desc[:100],  # Truncate for display
                                    progress=0.0
                                )
                            
                            return {
                                "test_file": test_path,
                                "success": False,
                                "iterations": iteration + 1,
                                "final_status": "prod_bug_suspected",
                                "error": f"Production bug detected: {suspected_bug_desc}",
                                "last_failure": failure_output,
                                "suspected_prod_bug": suspected_bug_desc,
                                "bug_report_created": True,
                            }
                        
                        # Handle schema-invalid output - allow single retry per iteration
                        if final_status == "llm_invalid_output":
                            logger.warning(
                                "LLM returned invalid output on iteration %d for %s: %s",
                                iteration + 1,
                                test_path,
                                refine_result.get("error", "Unknown error")
                            )
                            span.set_attribute("invalid_output_iteration", iteration + 1)
                            
                            # Check if schema repair was attempted (from the refine adapter)
                            schema_repaired = refine_result.get("repaired", False)
                            if schema_repaired:
                                logger.info(
                                    "Schema repair was attempted for %s on iteration %d - continuing to next iteration",
                                    test_path, iteration + 1
                                )
                                # Continue to next iteration after repair attempt
                                continue
                            else:
                                # Schema repair wasn't available or failed - this shouldn't happen with new code
                                logger.error(
                                    "Schema validation failed without repair attempt for %s on iteration %d",
                                    test_path, iteration + 1
                                )
                                continue
                        
                        # Handle layered validation statuses that indicate "no real change"
                        no_change_statuses = {
                            "content_identical", 
                            "content_cosmetic_noop", 
                            "content_semantically_identical",
                            "llm_no_change"
                        }
                        
                        if final_status in no_change_statuses:
                            # Check if we should stop on no change (default is True)
                            stop_on_no_change = getattr(self._config, 'stop_on_no_change', True) if hasattr(self, '_config') else True
                            
                            # Also check cosmetic handling config
                            treat_cosmetic_as_no_change = getattr(self._config, 'treat_cosmetic_as_no_change', True) if hasattr(self, '_config') else True
                            
                            # For cosmetic changes, respect the cosmetic config setting
                            if final_status == "content_cosmetic_noop" and not treat_cosmetic_as_no_change:
                                logger.info(
                                    "LLM made cosmetic changes on iteration %d for %s, continuing (treat_cosmetic_as_no_change=False)",
                                    iteration + 1,
                                    test_path,
                                )
                                # Continue to next iteration since cosmetic changes are allowed
                                continue
                            
                            if stop_on_no_change:
                                # Provide a more descriptive message based on the specific status
                                status_messages = {
                                    "content_identical": "LLM returned identical content",
                                    "content_cosmetic_noop": "LLM made only cosmetic changes (whitespace/formatting)",
                                    "content_semantically_identical": "LLM made changes that are semantically equivalent",
                                    "llm_no_change": "LLM explicitly returned no changes"
                                }
                                
                                reason = status_messages.get(final_status, "LLM refinement made no meaningful changes")
                                
                                logger.info(
                                    "%s on iteration %d for %s, stopping early (stop_on_no_change=True)",
                                    reason, iteration + 1, test_path,
                                )
                                span.set_attribute("stopped_reason", final_status)
                                span.set_attribute("no_change_detected", True)
                                span.set_attribute("specific_no_change_reason", reason)
                                
                                # Annotate failed test with no-change failure info
                                if self._config.annotate_failed_tests:
                                    await self._annotator.annotate_failed_test(
                                        test_file=Path(test_path),
                                        failure_output=failure_output,
                                        reason_status="no_change_detected",
                                        iterations=iteration + 1,
                                        fix_instructions=last_fix_instructions,
                                        extra={
                                            "active_import_path": last_active_import_path,
                                            "preflight_suggestions": last_preflight,
                                        }
                                    )
                                
                                return {
                                    "test_file": test_path,
                                    "success": False,
                                    "iterations": iteration + 1,
                                    "final_status": "no_change_detected",
                                    "error": reason,
                                    "last_failure": failure_output,
                                    "no_change_details": {
                                        "validation_status": final_status,
                                        "reason": reason,
                                        "diff_snippet": refine_result.get("diff_snippet", ""),
                                    }
                                }
                            else:
                                logger.info(
                                    "LLM returned no meaningful changes on iteration %d for %s, continuing (stop_on_no_change=False)",
                                    iteration + 1,
                                    test_path,
                                )
                                # Continue to next iteration
                                continue
                        
                        # Handle syntax errors - should stop
                        if final_status == "syntax_error":
                            logger.error(
                                "LLM returned syntactically invalid Python on iteration %d for %s",
                                iteration + 1,
                                test_path,
                            )
                            span.set_attribute("stopped_reason", "syntax_error")
                            
                            # Annotate failed test with syntax error failure info
                            if self._config.annotate_failed_tests:
                                await self._annotator.annotate_failed_test(
                                    test_file=Path(test_path),
                                    failure_output=failure_output,
                                    reason_status="syntax_error",
                                    iterations=iteration + 1,
                                    fix_instructions=last_fix_instructions,
                                    extra={
                                        "active_import_path": last_active_import_path,
                                        "preflight_suggestions": last_preflight,
                                    }
                                )
                            
                            return {
                                "test_file": test_path,
                                "success": False,
                                "iterations": iteration + 1,
                                "final_status": "syntax_error",
                                "error": refine_result.get("error", "Syntax error in refined content"),
                                "last_failure": failure_output,
                            }

                        if not refine_result.get("success"):
                            logger.warning(
                                "Refinement failed on iteration %d for %s: %s",
                                iteration + 1,
                                test_path,
                                refine_result.get("error", "Unknown error"),
                            )
                            continue

                        # Step 4: Check if content actually changed (no-change detection)
                        refined_content = refine_result.get("refined_content")
                        if (
                            refined_content
                            and refined_content.strip() == (previous_content or "").strip()
                        ):
                            # Content didn't change, avoid infinite loop
                            logger.warning(
                                "No content changes detected in iteration %d for %s, stopping refinement",
                                iteration + 1,
                                test_path,
                            )
                            span.set_attribute("stopped_reason", "no_content_change")

                            # Annotate failed test with no content change failure info
                            if self._config.annotate_failed_tests:
                                await self._annotator.annotate_failed_test(
                                    test_file=Path(test_path),
                                    failure_output=failure_output,
                                    reason_status="no_content_change_detected",
                                    iterations=iteration + 1,
                                    fix_instructions=last_fix_instructions,
                                    extra={
                                        "active_import_path": last_active_import_path,
                                        "preflight_suggestions": last_preflight,
                                    }
                                )

                            return {
                                "test_file": test_path,
                                "success": False,
                                "iterations": iteration + 1,
                                "final_status": "no_change_detected",
                                "error": "Refinement made no changes to test content",
                                "last_failure": failure_output,
                            }

                        # Step 5: Content changed, update for next iteration
                        # Note: The refine_port is responsible for writing the content to the file
                        # We just track the content for change detection
                        if refined_content:
                            previous_content = refined_content

                        # Step 6: Add exponential backoff only after successful write attempts
                        if iteration < max_iterations - 1 and self._backoff_sec > 0:
                            # Exponential backoff: base * (2 ^ iteration) but cap at reasonable limit
                            backoff_time = min(self._backoff_sec * (2 ** iteration), 5.0)
                            logger.debug(
                                "Applying backoff of %.2fs after successful refinement iteration %d",
                                backoff_time, iteration + 1
                            )
                            await asyncio.sleep(backoff_time)

                    except Exception as e:
                        logger.warning(
                            "Refinement iteration %d failed for %s: %s",
                            iteration + 1,
                            test_path,
                            e,
                        )
                        span.set_attribute(f"iteration_{iteration}_error", str(e))
                        continue

                # Step 6: All refinement attempts exhausted
                span.set_attribute("refinement_successful", False)
                span.set_attribute("stopped_reason", "max_iterations_exceeded")

                # Run final pytest to get latest status
                try:
                    final_pytest = await self._runner.run_pytest(test_path)
                    final_status = "passed" if final_pytest["returncode"] == 0 else "failed"
                    final_output = self._failure_parser.format_pytest_failure_output(final_pytest)
                except Exception:
                    final_status = "unknown"
                    final_output = "Could not determine final test status"

                # Annotate failed test with max iterations exhausted failure info
                if self._config.annotate_failed_tests:
                    # One last call for targeted manual suggestions from LLM
                    manual_suggestions_payload = {}
                    try:
                        # Build minimal source context again
                        try:
                            test_content = test_file.read_text(encoding='utf-8') if test_file.exists() else ""
                        except Exception:
                            test_content = ""
                        source_context_last = await build_source_context_fn(test_file, test_content)

                        manual_suggestions_payload = self._refine.manual_fix_suggestions(  # type: ignore[attr-defined]
                            test_file=test_path,
                            failure_output=final_output,
                            source_context=source_context_last,
                        ) if hasattr(self._refine, 'manual_fix_suggestions') else {}
                    except Exception as e:
                        logger.debug("Manual fix suggestions request failed: %s", e)

                    # Enhance annotation with manual suggestions and root cause when available
                    extra_info = {
                        "active_import_path": manual_suggestions_payload.get("active_import_path", last_active_import_path),
                        "preflight_suggestions": manual_suggestions_payload.get("preflight_suggestions", last_preflight),
                    }
                    manual_text = manual_suggestions_payload.get("manual_suggestions")
                    root_cause = manual_suggestions_payload.get("root_cause")
                    llm_fix_text = last_fix_instructions or ""
                    if manual_text:
                        # Prefer the explicit manual suggestions over earlier fix instructions
                        llm_fix_text = manual_text
                    if root_cause:
                        # Prepend root cause to the fix text for visibility
                        llm_fix_text = (f"ROOT CAUSE: {root_cause}\n\n" + (llm_fix_text or "")).strip()

                    await self._annotator.annotate_failed_test(
                        test_file=Path(test_path),
                        failure_output=final_output,
                        reason_status="max_iterations_exceeded",
                        iterations=max_iterations,
                        fix_instructions=llm_fix_text,
                        extra=extra_info,
                    )

                return {
                    "test_file": test_path,
                    "success": False,
                    "iterations": max_iterations,
                    "final_status": final_status,
                    "error": f"Maximum refinement iterations ({max_iterations}) exceeded",
                    "last_failure": final_output,
                    "manual_fix_suggestions": manual_suggestions_payload.get("manual_suggestions") if isinstance(manual_suggestions_payload, dict) else None,
                    "root_cause": manual_suggestions_payload.get("root_cause") if isinstance(manual_suggestions_payload, dict) else None,
                }
