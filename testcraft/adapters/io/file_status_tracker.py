"""
File Status Tracker for Live Generation and Refinement Updates

This module provides real-time tracking of file processing status during
test generation and refinement operations. It integrates with the enhanced
UI system to provide live status updates with granular details.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    TaskID,
)
from rich.table import Table

from .enhanced_ui import EnhancedUIAdapter
from .ui_rich import UIStyle


class FileStatus(Enum):
    """Status values for file processing."""

    WAITING = "waiting"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    WRITING = "writing"
    TESTING = "testing"
    REFINING = "refining"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# State transition validation
_VALID_TRANSITIONS = {
    FileStatus.WAITING: {FileStatus.ANALYZING, FileStatus.SKIPPED},
    FileStatus.ANALYZING: {FileStatus.GENERATING, FileStatus.FAILED},
    FileStatus.GENERATING: {FileStatus.WRITING, FileStatus.FAILED},
    FileStatus.WRITING: {FileStatus.TESTING, FileStatus.FAILED},
    FileStatus.TESTING: {FileStatus.REFINING, FileStatus.COMPLETED, FileStatus.FAILED},
    FileStatus.REFINING: {FileStatus.TESTING, FileStatus.COMPLETED, FileStatus.FAILED},
    # Terminal states - no transitions allowed
    FileStatus.COMPLETED: set(),
    FileStatus.FAILED: set(),
    FileStatus.SKIPPED: set(),
}


@dataclass
class FileProcessingState:
    """Detailed state information for a single file."""

    file_path: str
    status: FileStatus = FileStatus.WAITING
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    # Generation details
    generation_start: float | None = None
    generation_end: float | None = None
    generation_success: bool = False
    generation_error: str | None = None

    # Writing details
    write_start: float | None = None
    write_end: float | None = None
    write_success: bool = False
    write_error: str | None = None
    test_file_path: str | None = None

    # Testing/refinement details
    test_start: float | None = None
    test_end: float | None = None
    pytest_runs: int = 0
    refinement_iterations: int = 0
    final_test_success: bool = False
    test_errors: list[str] = field(default_factory=list)

    # Metrics
    tests_generated: int = 0
    lines_of_code: int = 0
    coverage_improvement: float = 0.0

    # Current operation details
    current_operation: str = ""
    current_step: str = ""
    progress_percentage: float = 0.0

    def get_duration(self) -> float:
        """Get total processing duration."""
        return time.time() - self.start_time

    def get_generation_duration(self) -> float:
        """Get generation phase duration."""
        if self.generation_start and self.generation_end:
            return self.generation_end - self.generation_start
        return 0.0

    def get_refinement_duration(self) -> float:
        """Get refinement phase duration."""
        if self.test_start and self.test_end:
            return self.test_end - self.test_start
        return 0.0

    def is_complete(self) -> bool:
        """Check if file processing is complete."""
        return self.status in [
            FileStatus.COMPLETED,
            FileStatus.FAILED,
            FileStatus.SKIPPED,
        ]


class FileStatusTracker:
    """
    Live tracking system for file processing during generation and refinement.

    Provides real-time updates on the status of each file as it moves through
    the generation pipeline, with detailed information about each phase.
    """

    def __init__(self, ui_adapter: EnhancedUIAdapter) -> None:
        """Initialize the file status tracker."""
        self.ui = ui_adapter
        self.console = ui_adapter.console
        self._lock = Lock()

        # Detect minimal mode from UI adapter
        self.minimal_mode = ui_adapter.ui_style == UIStyle.MINIMAL

        # File tracking
        self._files: dict[str, FileProcessingState] = {}
        self._file_order: list[str] = []

        # Live display
        self._live_display: Live | None = None
        self._layout: Layout | None = None
        self._is_running = False
        self._display_task: asyncio.Task | None = None

        # Progress tracking
        self._overall_progress: Progress | None = None
        self._overall_task: TaskID | None = None

        # Statistics
        self._start_time = time.time()
        self._completed_count = 0
        self._failed_count = 0

        # Initialization state
        self._initialized = False

        # Logging
        self._logger = logging.getLogger(__name__)

        # Error tracking
        self._consecutive_errors = 0
        self._last_error_time = 0

    def initialize_files(self, file_paths: list[str]) -> None:
        """Initialize tracking for a list of files."""
        with self._lock:
            self._files.clear()
            self._file_order = file_paths.copy()

            for file_path in file_paths:
                self._files[file_path] = FileProcessingState(file_path=file_path)

            self._initialized = True
            self._completed_count = 0
            self._failed_count = 0
            self._consecutive_errors = 0

    def _validate_transition(self, current: FileStatus, new: FileStatus) -> bool:
        """Validate if a state transition is allowed."""
        valid_next_states = _VALID_TRANSITIONS.get(current, set())
        return new in valid_next_states

    def _start_generation_phase(self, file_state: FileProcessingState) -> None:
        """Start the generation phase for a file."""
        if not file_state.generation_start:
            file_state.generation_start = time.time()

    def _end_generation_phase(
        self, file_state: FileProcessingState, success: bool = True
    ) -> None:
        """End the generation phase for a file."""
        if file_state.generation_start and not file_state.generation_end:
            file_state.generation_end = time.time()
            file_state.generation_success = success

    def _start_writing_phase(self, file_state: FileProcessingState) -> None:
        """Start the writing phase for a file."""
        if not file_state.write_start:
            file_state.write_start = time.time()

    def _end_writing_phase(
        self, file_state: FileProcessingState, success: bool = True
    ) -> None:
        """End the writing phase for a file."""
        if file_state.write_start and not file_state.write_end:
            file_state.write_end = time.time()
            file_state.write_success = success

    def _start_testing_phase(self, file_state: FileProcessingState) -> None:
        """Start the testing phase for a file."""
        if not file_state.test_start:
            file_state.test_start = time.time()

    def _end_testing_phase(
        self, file_state: FileProcessingState, success: bool = True
    ) -> None:
        """End the testing phase for a file."""
        if file_state.test_start and not file_state.test_end:
            file_state.test_end = time.time()
            file_state.final_test_success = success

    def _check_timeout(self, file_state: FileProcessingState) -> FileStatus | None:
        """Check if file has exceeded reasonable timeout limits."""
        current_time = time.time()

        # Check generation timeout (5 minutes)
        if (
            file_state.status == FileStatus.GENERATING
            and file_state.generation_start
            and current_time - file_state.generation_start > 300
        ):
            self._logger.warning(
                f"File {file_state.file_path} timed out in GENERATING phase"
            )
            return FileStatus.FAILED

        # Check writing timeout (2 minutes)
        if (
            file_state.status == FileStatus.WRITING
            and file_state.write_start
            and current_time - file_state.write_start > 120
        ):
            self._logger.warning(
                f"File {file_state.file_path} timed out in WRITING phase"
            )
            return FileStatus.FAILED

        # Check testing timeout (10 minutes)
        if (
            file_state.status in [FileStatus.TESTING, FileStatus.REFINING]
            and file_state.test_start
            and current_time - file_state.test_start > 600
        ):
            self._logger.warning(
                f"File {file_state.file_path} timed out in TESTING/REFINING phase"
            )
            return FileStatus.FAILED

        return None

    def reset_file_status(self, file_path: str) -> None:
        """Reset a file's status for manual recovery."""
        with self._lock:
            if file_path in self._files:
                old_state = self._files[file_path]
                self._files[file_path] = FileProcessingState(file_path=file_path)
                self._logger.info(
                    f"Reset file {file_path} from {old_state.status} to WAITING"
                )

    def start_live_tracking(self, title: str = "File Processing Status") -> None:
        """Start the live status display."""
        if self._is_running:
            return

        self._is_running = True
        self._start_time = time.time()

        # Create layout based on mode
        self._layout = Layout()

        if self.minimal_mode:
            # Single column layout: header (size 1-2), main (single files list only), footer (size 2-3)
            self._layout.split_column(
                Layout(name="header", size=2),
                Layout(name="main", ratio=1),
                Layout(name="footer", size=3),
            )
            # No split for main - just files list
            self._layout["main"].update(Layout(name="files"))
        else:
            # Classic two-column layout
            self._layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main", ratio=1),
                Layout(name="footer", size=8),
            )

            self._layout["main"].split_row(
                Layout(name="files", ratio=3),
                Layout(name="stats", ratio=1),
            )

        # Initialize clean header
        self._layout["header"].update(
            Panel(f"[header]{title}[/]", style="border", padding=(0, 1))
        )

        # Start live display with appropriate refresh rate
        refresh_rate = 2 if self.minimal_mode else 3
        # In minimal mode, prefer transient live so nothing persists
        self._live_display = Live(
            self._layout,
            console=self.console,
            refresh_per_second=refresh_rate,
            transient=True if self.minimal_mode else False,
        )
        self._live_display.start()

        # Start update loop and store task reference
        self._display_task = asyncio.create_task(self._update_display_loop())

    async def stop_live_tracking(self) -> None:
        """Stop the live status display."""
        self._is_running = False

        # Cancel the display task if it exists
        if self._display_task and not self._display_task.done():
            self._display_task.cancel()
            try:
                await self._display_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling

        if self._live_display:
            self._live_display.stop()
            self._live_display = None

    def update_file_status(
        self,
        file_path: str,
        status: FileStatus,
        operation: str = "",
        step: str = "",
        progress: float = 0.0,
        **kwargs,
    ) -> None:
        """Update the status of a specific file."""
        # Check initialization guard
        if not self._initialized:
            self._logger.warning(
                f"Attempted to update file status before initialization: {file_path}"
            )
            return

        with self._lock:
            if file_path not in self._files:
                self._files[file_path] = FileProcessingState(file_path=file_path)

            file_state = self._files[file_path]

            # Validate state transition
            if not self._validate_transition(file_state.status, status):
                self._logger.warning(
                    f"Invalid state transition for {file_path}: "
                    f"{file_state.status} -> {status}"
                )
                return

            # Check for timeout
            timeout_status = self._check_timeout(file_state)
            if timeout_status:
                status = timeout_status

            # Update basic state
            old_status = file_state.status
            file_state.status = status
            file_state.last_update = time.time()
            file_state.current_operation = operation
            file_state.current_step = step
            file_state.progress_percentage = progress

            # Update phase-specific information using new methods
            if status == FileStatus.GENERATING:
                self._start_generation_phase(file_state)
            elif status == FileStatus.WRITING:
                self._start_writing_phase(file_state)
                # End generation phase if it was active
                if old_status == FileStatus.GENERATING:
                    self._end_generation_phase(file_state, True)
            elif status == FileStatus.TESTING:
                self._start_testing_phase(file_state)
                # End writing phase if it was active
                if old_status == FileStatus.WRITING:
                    self._end_writing_phase(file_state, True)
            elif status == FileStatus.REFINING:
                if not file_state.test_start:
                    self._start_testing_phase(file_state)
                file_state.pytest_runs += 1
            elif status in [FileStatus.COMPLETED, FileStatus.FAILED]:
                # End testing phase if it was active
                if old_status in [FileStatus.TESTING, FileStatus.REFINING]:
                    self._end_testing_phase(file_state, status == FileStatus.COMPLETED)

                if status == FileStatus.COMPLETED:
                    self._completed_count += 1
                elif status == FileStatus.FAILED:
                    self._failed_count += 1

            # Update specific metrics from kwargs
            for key, value in kwargs.items():
                if hasattr(file_state, key):
                    setattr(file_state, key, value)

    def update_generation_result(
        self,
        file_path: str,
        success: bool,
        tests_generated: int = 0,
        error: str | None = None,
        test_file_path: str | None = None,
    ) -> None:
        """Update generation results for a file."""
        with self._lock:
            if file_path in self._files:
                file_state = self._files[file_path]
                file_state.generation_success = success
                file_state.tests_generated = tests_generated
                file_state.test_file_path = test_file_path
                if error:
                    file_state.generation_error = error

    def update_refinement_result(
        self,
        file_path: str,
        iteration: int,
        success: bool,
        errors: list[str] | None = None,
    ) -> None:
        """Update refinement results for a file."""
        with self._lock:
            if file_path in self._files:
                file_state = self._files[file_path]
                file_state.refinement_iterations = iteration
                file_state.final_test_success = success
                if errors:
                    file_state.test_errors.extend(errors)

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for all files."""
        with self._lock:
            total_files = len(self._files)
            completed = sum(
                1 for f in self._files.values() if f.status == FileStatus.COMPLETED
            )
            failed = sum(
                1 for f in self._files.values() if f.status == FileStatus.FAILED
            )
            in_progress = sum(1 for f in self._files.values() if not f.is_complete())

            total_duration = time.time() - self._start_time
            avg_duration = total_duration / max(completed + failed, 1)

            total_tests = sum(f.tests_generated for f in self._files.values())
            total_pytest_runs = sum(f.pytest_runs for f in self._files.values())

            return {
                "total_files": total_files,
                "completed": completed,
                "failed": failed,
                "in_progress": in_progress,
                "success_rate": completed / max(total_files, 1),
                "total_duration": total_duration,
                "avg_duration": avg_duration,
                "total_tests_generated": total_tests,
                "total_pytest_runs": total_pytest_runs,
                "files_per_minute": (completed + failed) / max(total_duration / 60, 1),
            }

    def _get_display_snapshot(self) -> dict[str, Any]:
        """Get a snapshot of current state for display purposes."""
        with self._lock:
            return {
                "files": self._files.copy(),
                "file_order": self._file_order.copy(),
                "completed_count": self._completed_count,
                "failed_count": self._failed_count,
                "start_time": self._start_time,
            }

    async def _update_display_loop(self) -> None:
        """Continuous update loop for the live display."""
        sleep_time = (
            0.5 if self.minimal_mode else 0.33
        )  # 2 Hz for minimal, 3 Hz for classic
        while self._is_running:
            try:
                self._update_display()
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                self._logger.info("Display update loop cancelled")
                break
            except Exception as e:
                # Log error with exponential backoff to prevent spam
                current_time = time.time()
                if current_time - self._last_error_time > min(
                    30, 2**self._consecutive_errors
                ):
                    self._logger.warning(f"Display update failed: {e}")
                    self._consecutive_errors += 1
                    self._last_error_time = int(current_time)
                continue

    def _update_display(self) -> None:
        """Update the live display with current file status."""
        if not self._layout or not self._is_running:
            return

        # Get snapshot under lock to prevent race conditions
        try:
            snapshot = self._get_display_snapshot()
        except Exception as e:
            self._logger.warning(f"Failed to get display snapshot: {e}")
            return

        try:
            # Update files table using snapshot data
            files_table = self._create_files_table_from_snapshot(snapshot)
            if self.minimal_mode:
                # In minimal mode, files table is directly in main
                self._layout["main"].update(files_table)
            else:
                # In classic mode, files table is in files section
                self._layout["files"].update(files_table)

            # Update statistics panel (skip in minimal mode)
            if not self.minimal_mode:
                stats_panel = self._create_stats_panel_from_snapshot(snapshot)
                self._layout["stats"].update(stats_panel)

            # Update footer with overall progress
            footer_content = self._create_footer_content_from_snapshot(snapshot)
            self._layout["footer"].update(footer_content)

            # Reset error counter on successful update
            self._consecutive_errors = 0

        except Exception as e:
            # Handle display-specific errors (Rich rendering issues)
            self._logger.warning(f"Display update failed: {e}")
            self._consecutive_errors += 1

    def _create_files_table_from_snapshot(self, snapshot: dict[str, Any]) -> Table:
        """Create the clean, minimal files status table from snapshot data."""
        table = Table(
            show_header=True,
            header_style="header",
            border_style="border",
            show_lines=False,
            expand=True,
            box=None,  # Remove box for cleaner look
        )

        # Use lowercase headers for consistency (especially in minimal mode)
        table.add_column("file", style="primary", width=30)
        table.add_column("status", justify="center", width=12)
        table.add_column("progress", justify="center", width=15)
        table.add_column("tests", justify="center", width=8)
        table.add_column("time", justify="center", width=8)

        # Sort files by status and name
        files_to_show = []
        for file_path in snapshot["file_order"]:
            if file_path in snapshot["files"]:
                files_to_show.append(snapshot["files"][file_path])

        # Show recent/active files first
        files_to_show.sort(key=lambda f: (f.is_complete(), f.last_update), reverse=True)

        # Show fewer files in minimal mode (top 10), more in classic (12)
        max_files = 10 if self.minimal_mode else 12
        for file_state in files_to_show[:max_files]:
            file_name = Path(file_state.file_path).name

            # Minimal status display
            status_text, status_color = self._get_minimal_status_display(
                file_state.status
            )

            # Clean progress indicator
            if file_state.progress_percentage > 0:
                progress_dots = "●" * int(
                    file_state.progress_percentage / 25
                )  # 4 dots max
                progress_empty = "○" * (4 - len(progress_dots))
                if file_state.status == FileStatus.COMPLETED:
                    progress_display = "[success]●●●●[/]"
                elif file_state.status == FileStatus.FAILED:
                    progress_display = "[error]○○○○[/]"
                else:
                    progress_display = (
                        f"[accent]{progress_dots}[/][muted]{progress_empty}[/]"
                    )
            else:
                progress_display = "[muted]○○○○[/]"

            # Minimal tests display
            tests_display = (
                str(file_state.tests_generated)
                if file_state.tests_generated > 0
                else "—"
            )

            # Clean duration
            duration = file_state.get_duration()
            if duration < 60:
                duration_display = f"{duration:.1f}s" if duration > 0 else "—"
            else:
                mins, secs = divmod(duration, 60)
                duration_display = f"{int(mins)}m{secs:02.0f}s"

            table.add_row(
                file_name,
                f"[{status_color}]{status_text}[/]",
                progress_display,
                tests_display,
                f"[muted]{duration_display}[/]",
            )

        return table

    def _create_files_table(self) -> Table:
        """Create the clean, minimal files status table (deprecated - use snapshot version)."""
        # Fallback to snapshot method for backward compatibility
        try:
            snapshot = self._get_display_snapshot()
            return self._create_files_table_from_snapshot(snapshot)
        except Exception:
            # Return empty table on error
            table = Table(show_header=True, header_style="header")
            table.add_column("file", style="primary")
            return table

    def _create_stats_panel_from_snapshot(self, snapshot: dict[str, Any]) -> Panel:
        """Create the clean, minimal statistics panel from snapshot data."""
        total_files = len(snapshot["files"])
        completed = snapshot["completed_count"]
        failed = snapshot["failed_count"]
        in_progress = sum(1 for f in snapshot["files"].values() if not f.is_complete())

        total_tests = sum(f.tests_generated for f in snapshot["files"].values())
        success_rate = completed / max(total_files, 1)

        content_lines = [
            f"[success]done[/] {completed}",
            f"[error]failed[/] {failed}",
            f"[status_working]active[/] {in_progress}",
            "",
            f"tests {total_tests}",
            f"rate {success_rate:.0%}",
        ]

        return Panel(
            "\n".join(content_lines),
            title="stats",
            border_style="border",
            padding=(1, 1),
        )

    def _create_stats_panel(self) -> Panel:
        """Create the clean, minimal statistics panel (deprecated - use snapshot version)."""
        try:
            snapshot = self._get_display_snapshot()
            return self._create_stats_panel_from_snapshot(snapshot)
        except Exception:
            # Return minimal panel on error
            return Panel("stats unavailable", title="stats", border_style="border")

    def _create_footer_content_from_snapshot(self, snapshot: dict[str, Any]) -> Panel:
        """Create clean footer with minimal overall progress from snapshot data."""
        total_processed = snapshot["completed_count"] + snapshot["failed_count"]
        total_files = len(snapshot["files"])

        if total_files > 0:
            overall_progress = total_processed / total_files
            # Simple progress dots - use 10 dots for minimal, 10 for classic too
            dot_count = 10
            progress_dots = "●" * int(overall_progress * dot_count)
            progress_empty = "○" * (dot_count - len(progress_dots))
            progress_text = f"[accent]{progress_dots}[/][muted]{progress_empty}[/] {total_processed}/{total_files}"
        else:
            progress_text = "[muted]starting...[/]"

        elapsed = time.time() - snapshot["start_time"]
        if elapsed < 60:
            time_text = f"{elapsed:.0f}s"
        else:
            mins, secs = divmod(elapsed, 60)
            time_text = f"{int(mins)}m{secs:02.0f}s"

        footer_text = f"progress {progress_text}  •  {time_text}"

        return Panel(footer_text, border_style="border", padding=(0, 1), title=None)

    def _create_footer_content(self) -> Panel:
        """Create clean footer with minimal overall progress (deprecated - use snapshot version)."""
        try:
            snapshot = self._get_display_snapshot()
            return self._create_footer_content_from_snapshot(snapshot)
        except Exception:
            # Return minimal footer on error
            return Panel("progress unavailable", border_style="border", padding=(0, 1))

    def _get_minimal_status_display(self, status: FileStatus) -> tuple[str, str]:
        """Get minimal display text and color for a status."""
        status_map = {
            FileStatus.WAITING: ("waiting", "muted"),
            FileStatus.ANALYZING: ("analyzing", "status_working"),
            FileStatus.GENERATING: ("generating", "status_working"),
            FileStatus.WRITING: ("writing", "status_working"),
            FileStatus.TESTING: ("testing", "status_working"),
            FileStatus.REFINING: ("refining", "status_working"),
            FileStatus.COMPLETED: ("done", "status_pass"),
            FileStatus.FAILED: ("failed", "status_fail"),
            FileStatus.SKIPPED: ("skipped", "muted"),
        }

        return status_map.get(status, ("unknown", "muted"))


# Context manager for easy usage
class LiveFileTracking:
    """Context manager for live file status tracking."""

    def __init__(
        self, ui_adapter: EnhancedUIAdapter, title: str = "File Processing"
    ) -> None:
        self.tracker = FileStatusTracker(ui_adapter)
        self.title = title
        self._initialized = False

    def __enter__(self) -> "LiveFileTracking":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Handle async cleanup - we need to get the event loop
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If we're in an async context, we need to schedule the cleanup
                asyncio.create_task(self.tracker.stop_live_tracking())
            else:
                # If no loop is running, just call the cleanup synchronously
                loop.run_until_complete(self.tracker.stop_live_tracking())
        except RuntimeError:
            # No event loop, just call cleanup synchronously (will be a no-op)
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.tracker.stop_live_tracking())
            except Exception:
                # If all else fails, just set the running flag to False
                self.tracker._is_running = False

    def initialize_and_start(self, file_paths: list[str]) -> FileStatusTracker:
        """Initialize files and start tracking."""
        self.tracker.initialize_files(file_paths)
        self.tracker.start_live_tracking(self.title)
        self._initialized = True
        return self.tracker
