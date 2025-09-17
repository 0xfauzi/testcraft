"""
File writing and persistence operations for GenerateUseCase.

This module handles all file writing operations, including immediate mode
processing, validation, formatting, and rollback capabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ....domain.models import GenerationResult
from ....ports.writer_port import WriterPort
from ....ports.telemetry_port import TelemetryPort

logger = logging.getLogger(__name__)


class WritingService:
    """
    Service for handling file writing and persistence operations.
    
    Provides both batch and immediate writing modes with validation,
    formatting, and rollback capabilities.
    """
    
    def __init__(
        self,
        writer_port: WriterPort,
        telemetry_port: TelemetryPort,
        config: dict[str, Any],
    ):
        """Initialize writing service with required ports."""
        self._writer = writer_port
        self._telemetry = telemetry_port
        self._config = config
    
    async def write_test_files_batch(
        self, generation_results: list[GenerationResult]
    ) -> list[dict[str, Any]]:
        """Write generated test files using batch strategy."""
        with self._telemetry.create_child_span("write_test_files_batch") as span:
            write_results = []
            
            try:
                for result in generation_results:
                    if not result.success or not result.content:
                        # Skip failed generations
                        continue
                    
                    try:
                        # Write the test file
                        write_result = self._writer.write_test_file(
                            test_path=result.file_path,
                            test_content=result.content,
                            overwrite=True,  # Configuration could control this
                            disable_ruff_format=bool(self._config.get("disable_ruff_format", False)),
                        )
                        
                        write_results.append({
                            "source_result": result,
                            "write_result": write_result,
                            "success": write_result.get("success", False),
                        })
                        
                    except Exception as e:
                        logger.warning("Failed to write test file %s: %s", result.file_path, e)
                        write_results.append({
                            "source_result": result,
                            "write_result": {"success": False, "error": str(e)},
                            "success": False,
                        })
                
                span.set_attribute(
                    "files_written", sum(1 for r in write_results if r["success"])
                )
                
                return write_results
                
            except Exception as e:
                logger.exception("Test file writing failed: %s", e)
                from .state_discovery import GenerateUseCaseError
                raise GenerateUseCaseError(f"Test file writing failed: {e}", cause=e)
    
    async def write_test_file_immediate(
        self, generation_result: GenerationResult
    ) -> dict[str, Any]:
        """
        Write a single test file with immediate validation and formatting.
        
        Implements pre-validation, syntax checking, and rollback capability.
        
        Args:
            generation_result: The generation result to write
            
        Returns:
            Write result dictionary with success status and details
        """
        if not generation_result.success or not generation_result.content:
            return {
                "success": False,
                "error": "Invalid generation result",
                "file_path": generation_result.file_path,
            }
        
        try:
            # Pre-validate content before writing (syntax check happens in writer)
            write_result = self._writer.write_test_file(
                test_path=generation_result.file_path,
                test_content=generation_result.content,
                overwrite=True,  # Immediate mode allows overwrite
                disable_ruff_format=bool(self._config.get("disable_ruff_format", False)),
            )
            
            # Convert to consistent format
            return {
                "success": write_result.get("success", False),
                "file_path": generation_result.file_path,
                "bytes_written": write_result.get("bytes_written", 0),
                "formatted": write_result.get("formatted", False),
                "error": write_result.get("error") if not write_result.get("success") else None,
                "source_result": generation_result,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": generation_result.file_path,
                "source_result": generation_result,
            }
    
    async def rollback_failed_write(self, file_path: str) -> None:
        """
        Rollback a failed write by removing incomplete test files.
        
        Args:
            file_path: Path to the file to rollback
        """
        try:
            path = Path(file_path)
            if path.exists():
                # Only remove if it appears to be incomplete (small file)
                # This is a safety check to avoid removing valid existing files
                try:
                    content = path.read_text(encoding="utf-8")
                    if len(content.strip()) < 50:  # Very small files are likely incomplete
                        path.unlink()
                        logger.info("Rolled back incomplete test file: %s", file_path)
                except Exception:
                    # If we can't read it, it's likely corrupted, so remove it
                    path.unlink()
                    logger.info("Rolled back corrupted test file: %s", file_path)
                    
        except Exception as e:
            logger.warning("Failed to rollback file %s: %s", file_path, e)


class ImmediateModeProcessor:
    """
    Processor for immediate mode workflow: generate → write → refine per file.
    
    Handles the complete immediate processing pipeline with status tracking
    and error handling.
    """
    
    def __init__(
        self,
        writing_service: WritingService,
        telemetry_port: TelemetryPort,
        config: dict[str, Any],
    ):
        """Initialize immediate mode processor."""
        self._writing_service = writing_service
        self._telemetry = telemetry_port
        self._config = config
        self._current_status_tracker = None
    
    def set_status_tracker(self, status_tracker) -> None:
        """Set the status tracker for live updates during processing."""
        self._current_status_tracker = status_tracker
    
    async def process_plan_immediate(
        self, 
        plan, 
        generation_result: GenerationResult,
        refine_callback=None,
    ) -> dict[str, Any]:
        """
        Process a single plan with immediate write-and-refine workflow.
        
        Implements: generate → write → pytest → refine (iteratively) → return results
        
        Args:
            plan: The test generation plan
            generation_result: The generation result from pipeline
            refine_callback: Optional callback for refinement step
            
        Returns:
            Rich result dict containing generation, write, and refinement results
        """
        result = {
            "plan": plan,
            "generation_result": generation_result,
            "write_result": None,
            "refinement_result": None,
            "success": False,
            "errors": [],
        }
        
        try:
            # Check if generation was successful
            if not generation_result.success or not generation_result.content:
                result["errors"].append("Test generation failed")
                if self._current_status_tracker:
                    self._update_status_failed(
                        plan.file_path,
                        "Generation Failed",
                        generation_result.error_message or "Test generation failed"
                    )
                return result
            
            # Update status for writing
            if self._current_status_tracker:
                self._update_status_writing(plan.file_path)
            
            # Step 2: Pre-validate and write test file
            try:
                write_result = await self._writing_service.write_test_file_immediate(generation_result)
                result["write_result"] = write_result
                
                if not write_result.get("success", False):
                    result["errors"].append(f"Write failed: {write_result.get('error', 'Unknown')}")
                    
                    if self._current_status_tracker:
                        self._update_status_failed(
                            plan.file_path,
                            "Write Failed",
                            write_result.get('error', 'Unknown write error')
                        )
                    
                    # Rollback if configured to do so
                    if not self._config["keep_failed_writes"]:
                        await self._writing_service.rollback_failed_write(generation_result.file_path)
                    
                    return result
                
            except Exception as e:
                result["errors"].append(f"Write error: {e}")
                if self._current_status_tracker:
                    self._update_status_failed(
                        plan.file_path,
                        "Write Error",
                        f"Exception during write: {str(e)}"
                    )
                if not self._config["keep_failed_writes"]:
                    await self._writing_service.rollback_failed_write(generation_result.file_path)
                return result
            
            # Step 3: Immediate refinement if enabled
            if self._config["enable_refinement"] and refine_callback:
                try:
                    # Update status for testing start
                    if self._current_status_tracker:
                        self._update_status_testing(plan.file_path)
                    
                    # Use provided refinement callback
                    refinement_result = await refine_callback(generation_result.file_path)
                    result["refinement_result"] = refinement_result
                    
                    if refinement_result.get("success", False):
                        result["success"] = True
                        if self._current_status_tracker:
                            self._update_status_completed(
                                plan.file_path,
                                f"All tests pass after {refinement_result.get('iterations', 1)} iteration(s)"
                            )
                    else:
                        result["errors"].append(f"Refinement failed: {refinement_result.get('error', 'Unknown')}")
                        if self._current_status_tracker:
                            self._update_status_failed(
                                plan.file_path,
                                "Refinement Failed",
                                refinement_result.get('error', 'Maximum iterations reached')
                            )
                        
                except Exception as e:
                    result["errors"].append(f"Refinement error: {e}")
                    if self._current_status_tracker:
                        self._update_status_failed(
                            plan.file_path,
                            "Refinement Error",
                            f"Exception during refinement: {str(e)}"
                        )
            else:
                # No refinement enabled, mark as success if write succeeded
                result["success"] = True
                if self._current_status_tracker:
                    self._update_status_completed(
                        plan.file_path,
                        "Test file saved (refinement disabled)"
                    )
            
            return result

        except Exception as e:
            logger.warning("Immediate plan processing failed: %s", e)
            result["errors"].append(f"Processing error: {e}")
            return result
    
    def _update_status_writing(self, file_path: str) -> None:
        """Update status for writing phase."""
        if self._current_status_tracker:
            # Import here to avoid circular imports
            try:
                from ....adapters.textual.file_status import FileStatus
                self._current_status_tracker.update_file_status(
                    file_path,
                    FileStatus.WRITING,
                    operation="Writing Tests",
                    step="Saving generated test file to disk",
                    progress=60.0
                )
            except ImportError:
                # Fallback if FileStatus is not available
                pass
    
    def _update_status_testing(self, file_path: str) -> None:
        """Update status for testing phase."""
        if self._current_status_tracker:
            try:
                from ....adapters.textual.file_status import FileStatus
                self._current_status_tracker.update_file_status(
                    file_path,
                    FileStatus.TESTING,
                    operation="Initial Testing",
                    step="Running pytest on generated tests",
                    progress=80.0
                )
            except ImportError:
                pass
    
    def _update_status_completed(self, file_path: str, step: str) -> None:
        """Update status for completion."""
        if self._current_status_tracker:
            try:
                from ....adapters.textual.file_status import FileStatus
                self._current_status_tracker.update_file_status(
                    file_path,
                    FileStatus.COMPLETED,
                    operation="Tests Passing",
                    step=step,
                    progress=100.0
                )
            except ImportError:
                pass
    
    def _update_status_failed(self, file_path: str, operation: str, step: str) -> None:
        """Update status for failure."""
        if self._current_status_tracker:
            try:
                from ....adapters.textual.file_status import FileStatus
                self._current_status_tracker.update_file_status(
                    file_path,
                    FileStatus.FAILED,
                    operation=operation,
                    step=step,
                    progress=0.0
                )
            except ImportError:
                pass
