"""
State sync and status reporting for GenerateUseCase.

This module handles state persistence, incremental logging, and status
reporting for the test generation process.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from ....domain.models import GenerationResult
from ....ports.state_port import StatePort

logger = logging.getLogger(__name__)


class StateReportingService:
    """
    Service for handling state persistence and status reporting.
    
    Provides centralized state management including incremental logging,
    final state recording, and status persistence.
    """
    
    def __init__(
        self,
        state_port: StatePort,
        executor: ThreadPoolExecutor,
        config: dict[str, Any],
    ):
        """Initialize state reporting service with required ports."""
        self._state = state_port
        self._executor = executor
        self._config = config
    
    async def record_per_file_state(self, file_result: dict[str, Any]) -> None:
        """
        Record incremental state and telemetry for a single file's processing.
        
        Args:
            file_result: Result dictionary from processing a single plan
        """
        try:
            # Extract file path for logging
            file_path = "unknown"
            if file_result.get("generation_result"):
                file_path = file_result["generation_result"].file_path
            
            # Build per-file state entry
            file_state = {
                "file_path": file_path,
                "timestamp": datetime.now().isoformat(),
                "success": file_result.get("success", False),
                "errors": file_result.get("errors", []),
                "stages": {},
            }
            
            # Record generation stage
            if file_result.get("generation_result"):
                gen_result = file_result["generation_result"]
                file_state["stages"]["generation"] = {
                    "success": gen_result.success,
                    "content_length": len(gen_result.content) if gen_result.content else 0,
                    "error": gen_result.error_message,
                }
            
            # Record write stage  
            if file_result.get("write_result"):
                write_result = file_result["write_result"]
                file_state["stages"]["write"] = {
                    "success": write_result.get("success", False),
                    "bytes_written": write_result.get("bytes_written", 0),
                    "formatted": write_result.get("formatted", False),
                    "error": write_result.get("error"),
                }
            
            # Record refinement stage
            if file_result.get("refinement_result"):
                refine_result = file_result["refinement_result"]
                file_state["stages"]["refinement"] = {
                    "success": refine_result.get("success", False),
                    "iterations": refine_result.get("iterations", 0),
                    "final_status": refine_result.get("final_status"),
                    "error": refine_result.get("error"),
                }
            
            # Store in state using append strategy for incremental logging
            self._state.update_state(
                "immediate_generation_log", 
                file_state, 
                merge_strategy="append"
            )
            
        except Exception as e:
            logger.warning("Failed to record per-file state for %s: %s", file_path, e)
    
    async def record_final_state(
        self,
        generation_results: list[GenerationResult],
        refinement_results: list[dict[str, Any]],
        coverage_delta: dict[str, Any],
    ) -> None:
        """Record final state information for future runs and analysis."""
        try:
            # Compile state data
            state_data = {
                "last_run_timestamp": asyncio.get_event_loop().time(),
                "generation_summary": {
                    "total_files_processed": len(generation_results),
                    "successful_generations": self._count_successful_generations(generation_results),
                    "failed_generations": self._count_failed_generations(generation_results),
                },
                "refinement_summary": {
                    "total_files_refined": len(refinement_results),
                    "successful_refinements": sum(
                        1 for r in refinement_results if r.get("success", False)
                    ),
                },
                "coverage_improvement": coverage_delta,
                "config_used": self._config.copy(),
            }
            
            # Record state
            self._state.update_state("last_generation_run", state_data)
            
            # Persist state for future runs
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._state.persist_state
            )
            
        except Exception as e:
            logger.warning("Failed to record final state: %s", e)
            # Don't fail the entire operation for state recording issues
    
    def _count_successful_generations(self, generation_results: list[GenerationResult]) -> int:
        """Count successful generation results."""
        return sum(
            1
            for r in generation_results
            if (
                (getattr(r, "success", False))
                if not isinstance(r, dict)
                else bool(r.get("success", False))
            )
        )
    
    def _count_failed_generations(self, generation_results: list[GenerationResult]) -> int:
        """Count failed generation results."""
        return sum(
            1
            for r in generation_results
            if not (
                (getattr(r, "success", False))
                if not isinstance(r, dict)
                else bool(r.get("success", False))
            )
        )


class StatusTracker:
    """
    Status tracker for live updates during generation.
    
    Provides a centralized interface for updating file processing status
    during the generation workflow.
    """
    
    def __init__(self):
        """Initialize status tracker."""
        self._file_statuses = {}
        self._callbacks = []
    
    def add_callback(self, callback) -> None:
        """Add a callback for status updates."""
        self._callbacks.append(callback)
    
    def update_file_status(
        self,
        file_path: str,
        status,
        operation: str = "",
        step: str = "",
        progress: float = 0.0,
    ) -> None:
        """
        Update the status of a file being processed.
        
        Args:
            file_path: Path to the file
            status: Current status (enum or string)
            operation: Current operation being performed
            step: Detailed step description
            progress: Progress percentage (0.0 to 100.0)
        """
        self._file_statuses[file_path] = {
            "status": status,
            "operation": operation,
            "step": step,
            "progress": progress,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(file_path, status, operation, step, progress)
            except Exception as e:
                logger.warning("Status callback failed: %s", e)
    
    def get_file_status(self, file_path: str) -> dict[str, Any] | None:
        """Get the current status of a file."""
        return self._file_statuses.get(file_path)
    
    def get_all_statuses(self) -> dict[str, dict[str, Any]]:
        """Get all file statuses."""
        return self._file_statuses.copy()
    
    def clear_status(self, file_path: str) -> None:
        """Clear the status for a specific file."""
        self._file_statuses.pop(file_path, None)
    
    def clear_all_statuses(self) -> None:
        """Clear all file statuses."""
        self._file_statuses.clear()


class GenerationProgressReporter:
    """
    Progress reporter for generation operations.
    
    Provides high-level progress reporting and summary statistics
    for the entire generation process.
    """
    
    def __init__(self):
        """Initialize progress reporter."""
        self._start_time = None
        self._total_files = 0
        self._completed_files = 0
        self._failed_files = 0
        self._current_phase = "initializing"
    
    def start_generation(self, total_files: int) -> None:
        """Start tracking generation progress."""
        self._start_time = datetime.now()
        self._total_files = total_files
        self._completed_files = 0
        self._failed_files = 0
        self._current_phase = "generating"
    
    def file_completed(self, success: bool = True) -> None:
        """Mark a file as completed."""
        if success:
            self._completed_files += 1
        else:
            self._failed_files += 1
    
    def set_phase(self, phase: str) -> None:
        """Set the current processing phase."""
        self._current_phase = phase
    
    def get_progress_summary(self) -> dict[str, Any]:
        """Get current progress summary."""
        elapsed_time = None
        if self._start_time:
            elapsed_time = (datetime.now() - self._start_time).total_seconds()
        
        processed_files = self._completed_files + self._failed_files
        progress_percentage = (processed_files / self._total_files * 100) if self._total_files > 0 else 0
        
        return {
            "phase": self._current_phase,
            "total_files": self._total_files,
            "completed_files": self._completed_files,
            "failed_files": self._failed_files,
            "processed_files": processed_files,
            "progress_percentage": progress_percentage,
            "elapsed_time_seconds": elapsed_time,
            "start_time": self._start_time.isoformat() if self._start_time else None,
        }
    
    def is_complete(self) -> bool:
        """Check if generation is complete."""
        return (self._completed_files + self._failed_files) >= self._total_files
