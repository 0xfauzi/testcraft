"""
Batch executor service for concurrent test generation.

Handles batching and concurrent execution of test generation operations
with proper error handling and result aggregation. Includes live status
tracking for real-time progress updates.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from ....adapters.io.file_status_tracker import FileStatus, FileStatusTracker
from ....domain.models import GenerationResult, TestGenerationPlan
from ....ports.telemetry_port import TelemetryPort

logger = logging.getLogger(__name__)


class BatchExecutor:
    """
    Service for executing test generation operations in batches.

    Provides concurrent execution of test generation plans with batching,
    error handling, and result aggregation.
    """

    def __init__(
        self,
        telemetry_port: TelemetryPort,
        status_tracker: FileStatusTracker | None = None,
    ):
        """
        Initialize the batch executor.

        Args:
            telemetry_port: Port for telemetry operations
            status_tracker: Optional file status tracker for live updates
        """
        self._telemetry = telemetry_port
        self._status_tracker = status_tracker
        self._lock = asyncio.Lock()

    async def run_in_batches(
        self,
        plans: list[TestGenerationPlan],
        batch_size: int,
        generation_fn: Callable[[TestGenerationPlan], Awaitable[GenerationResult]],
    ) -> list[GenerationResult]:
        """
        Execute test generation operations in batches with concurrency.

        Args:
            plans: List of test generation plans to process
            batch_size: Number of plans to process concurrently per batch
            generation_fn: Async function to generate tests for a single plan

        Returns:
            List of generation results from all batches
        """
        with self._telemetry.create_child_span("execute_test_generation") as span:
            try:
                generation_results = []

                # Initialize all files in status tracker
                if self._status_tracker:
                    file_paths = [plan.file_path for plan in plans]
                    for file_path in file_paths:
                        async with self._lock:
                            self._status_tracker.update_file_status(
                                file_path,
                                FileStatus.WAITING,
                                operation="Queued for generation",
                                step="Waiting in batch queue",
                            )

                # Process plans in batches
                for i in range(0, len(plans), batch_size):
                    batch = plans[i : i + batch_size]
                    span.set_attribute(f"batch_{i // batch_size}_size", len(batch))

                    # Update status for batch start
                    if self._status_tracker:
                        for plan in batch:
                            async with self._lock:
                                self._status_tracker.update_file_status(
                                    plan.file_path,
                                    FileStatus.ANALYZING,
                                    operation="Starting generation",
                                    step="Preparing for LLM generation",
                                    progress=10.0,
                                )

                    # Process batch concurrently
                    batch_results = await self._process_generation_batch(
                        batch, generation_fn
                    )
                    generation_results.extend(batch_results)

                    # Update status for batch completion
                    if self._status_tracker:
                        for j, result in enumerate(batch_results):
                            plan = batch[j]
                            if hasattr(result, "success") and result.success:
                                self._status_tracker.update_generation_result(
                                    plan.file_path,
                                    success=True,
                                    tests_generated=getattr(
                                        result, "tests_generated", 0
                                    ),
                                    test_file_path=getattr(
                                        result, "test_file_path", None
                                    ),
                                )
                            else:
                                error_msg = getattr(
                                    result, "error_message", "Generation failed"
                                )
                                self._status_tracker.update_generation_result(
                                    plan.file_path, success=False, error=error_msg
                                )

                span.set_attribute("total_generated", len(generation_results))
                # Be defensive: some generation functions may return dicts
                # instead of GenerationResult instances. Count success robustly.
                span.set_attribute(
                    "successful_generations",
                    sum(
                        1
                        for r in generation_results
                        if (
                            (getattr(r, "success", False))
                            if not isinstance(r, dict)
                            else bool(r.get("success", False))
                        )
                    ),
                )

                return generation_results

            except Exception as e:
                logger.exception("Test generation execution failed: %s", e)
                span.record_exception(e)

                # Update status for any remaining files
                if self._status_tracker:
                    for plan in plans:
                        async with self._lock:
                            self._status_tracker.update_file_status(
                                plan.file_path,
                                FileStatus.FAILED,
                                operation="Generation failed",
                                step=f"Batch execution error: {str(e)}",
                            )

                raise

    async def _process_generation_batch(
        self,
        batch: list[TestGenerationPlan],
        generation_fn: Callable[[TestGenerationPlan], Awaitable[GenerationResult]],
    ) -> list[GenerationResult]:
        """
        Process a batch of generation plans concurrently.

        Args:
            batch: Batch of test generation plans
            generation_fn: Function to generate tests for a single plan

        Returns:
            List of generation results for the batch
        """
        tasks = []

        for plan in batch:
            # Update status to generating
            if self._status_tracker:
                async with self._lock:
                    self._status_tracker.update_file_status(
                        plan.file_path,
                        FileStatus.GENERATING,
                        operation="LLM Generation",
                        step="Sending request to language model",
                        progress=25.0,
                    )

            # Create wrapped task with status updates
            task = self._track_generation_task(plan, generation_fn)
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions (and non-GenerationResult values) to GenerationResult objects
        generation_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Generation failed for plan %d: %s", i, result)
                generation_results.append(
                    GenerationResult(
                        file_path=batch[i].file_path,
                        content=None,
                        success=False,
                        error_message=str(result),
                    )
                )
            else:
                # If a dict-like result is returned, coerce it into a GenerationResult
                if isinstance(result, dict):
                    try:
                        success_flag = bool(result.get("success", False))
                        error_message = result.get("error_message")
                        if not success_flag and not error_message:
                            # Normalize common alt key
                            error_message = result.get("error") or "Unknown error"

                        generation_results.append(
                            GenerationResult(
                                file_path=batch[i].file_path,
                                content=result.get("content"),
                                success=success_flag,
                                error_message=error_message,
                            )
                        )
                    except Exception as coerce_error:
                        logger.warning(
                            "Invalid generation result for plan %d, coercing to failure: %s",
                            i,
                            coerce_error,
                        )
                        generation_results.append(
                            GenerationResult(
                                file_path=batch[i].file_path,
                                content=None,
                                success=False,
                                error_message="Invalid generation result structure",
                            )
                        )
                else:
                    generation_results.append(result)

        return generation_results

    async def _track_generation_task(
        self,
        plan: TestGenerationPlan,
        generation_fn: Callable[[TestGenerationPlan], Awaitable[GenerationResult]],
    ) -> GenerationResult:
        """
        Wrapper for generation function that provides live status updates.

        Args:
            plan: Test generation plan
            generation_fn: Original generation function

        Returns:
            Generation result
        """
        try:
            if self._status_tracker:
                async with self._lock:
                    self._status_tracker.update_file_status(
                        plan.file_path,
                        FileStatus.GENERATING,
                        operation="LLM Processing",
                        step="Generating test code with AI",
                        progress=50.0,
                    )

            # Call original generation function
            result = await generation_fn(plan)

            if self._status_tracker:
                async with self._lock:
                    if hasattr(result, "success") and result.success:
                        self._status_tracker.update_file_status(
                            plan.file_path,
                            FileStatus.WRITING,
                            operation="Writing Tests",
                            step="Saving generated test file",
                            progress=75.0,
                        )
                    else:
                        self._status_tracker.update_file_status(
                            plan.file_path,
                            FileStatus.FAILED,
                            operation="Generation Failed",
                            step=getattr(result, "error_message", "Unknown error"),
                            progress=0.0,
                        )

            return result

        except Exception as e:
            if self._status_tracker:
                async with self._lock:
                    self._status_tracker.update_file_status(
                        plan.file_path,
                        FileStatus.FAILED,
                        operation="Generation Error",
                        step=f"Exception: {str(e)}",
                        progress=0.0,
                    )
            raise
