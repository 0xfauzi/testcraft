"""
File Status Tracker for Live Generation and Refinement Updates

This module provides real-time tracking of file processing status during
test generation and refinement operations. It integrates with the enhanced
UI system to provide live status updates with granular details.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from threading import Lock
import asyncio

from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.text import Text

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


@dataclass
class FileProcessingState:
    """Detailed state information for a single file."""
    file_path: str
    status: FileStatus = FileStatus.WAITING
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    # Generation details
    generation_start: Optional[float] = None
    generation_end: Optional[float] = None
    generation_success: bool = False
    generation_error: Optional[str] = None
    
    # Writing details  
    write_start: Optional[float] = None
    write_end: Optional[float] = None
    write_success: bool = False
    write_error: Optional[str] = None
    test_file_path: Optional[str] = None
    
    # Testing/refinement details
    test_start: Optional[float] = None
    test_end: Optional[float] = None
    pytest_runs: int = 0
    refinement_iterations: int = 0
    final_test_success: bool = False
    test_errors: List[str] = field(default_factory=list)
    
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
        return self.status in [FileStatus.COMPLETED, FileStatus.FAILED, FileStatus.SKIPPED]


class FileStatusTracker:
    """
    Live tracking system for file processing during generation and refinement.
    
    Provides real-time updates on the status of each file as it moves through
    the generation pipeline, with detailed information about each phase.
    """
    
    def __init__(self, ui_adapter: EnhancedUIAdapter):
        """Initialize the file status tracker."""
        self.ui = ui_adapter
        self.console = ui_adapter.console
        self._lock = Lock()
        
        # Detect minimal mode from UI adapter
        self.minimal_mode = ui_adapter.ui_style == UIStyle.MINIMAL
        
        # File tracking
        self._files: Dict[str, FileProcessingState] = {}
        self._file_order: List[str] = []
        
        # Live display
        self._live_display: Optional[Live] = None
        self._layout: Optional[Layout] = None
        self._is_running = False
        
        # Progress tracking
        self._overall_progress: Optional[Progress] = None
        self._overall_task: Optional[TaskID] = None
        
        # Statistics
        self._start_time = time.time()
        self._completed_count = 0
        self._failed_count = 0
        
    def initialize_files(self, file_paths: List[str]) -> None:
        """Initialize tracking for a list of files."""
        with self._lock:
            self._files.clear()
            self._file_order = file_paths.copy()
            
            for file_path in file_paths:
                self._files[file_path] = FileProcessingState(file_path=file_path)
    
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
            Panel(
                f"[header]{title}[/]",
                style="border",
                padding=(0, 1)
            )
        )
        
        # Start live display with appropriate refresh rate
        refresh_rate = 2 if self.minimal_mode else 3
        # In minimal mode, prefer transient live so nothing persists
        self._live_display = Live(
            self._layout,
            console=self.console,
            refresh_per_second=refresh_rate,
            transient=True if self.minimal_mode else False
        )
        self._live_display.start()
        
        # Start update loop
        asyncio.create_task(self._update_display_loop())
    
    def stop_live_tracking(self) -> None:
        """Stop the live status display."""
        self._is_running = False
        if self._live_display:
            self._live_display.stop()
            self._live_display = None
    
    def update_file_status(self, file_path: str, status: FileStatus, 
                          operation: str = "", step: str = "", 
                          progress: float = 0.0, **kwargs) -> None:
        """Update the status of a specific file."""
        with self._lock:
            if file_path not in self._files:
                self._files[file_path] = FileProcessingState(file_path=file_path)
            
            file_state = self._files[file_path]
            file_state.status = status
            file_state.last_update = time.time()
            file_state.current_operation = operation
            file_state.current_step = step
            file_state.progress_percentage = progress
            
            # Update phase-specific information
            if status == FileStatus.GENERATING and not file_state.generation_start:
                file_state.generation_start = time.time()
            elif status in [FileStatus.WRITING, FileStatus.TESTING] and file_state.generation_start and not file_state.generation_end:
                file_state.generation_end = time.time()
                file_state.generation_success = True
            elif status == FileStatus.WRITING and not file_state.write_start:
                file_state.write_start = time.time()
            elif status == FileStatus.TESTING and file_state.write_start and not file_state.write_end:
                file_state.write_end = time.time() 
                file_state.write_success = True
            elif status == FileStatus.TESTING and not file_state.test_start:
                file_state.test_start = time.time()
            elif status == FileStatus.REFINING:
                if not file_state.test_start:
                    file_state.test_start = time.time()
                file_state.pytest_runs += 1
            elif status in [FileStatus.COMPLETED, FileStatus.FAILED]:
                if file_state.test_start and not file_state.test_end:
                    file_state.test_end = time.time()
                if status == FileStatus.COMPLETED:
                    file_state.final_test_success = True
                    self._completed_count += 1
                elif status == FileStatus.FAILED:
                    self._failed_count += 1
                    
            # Update specific metrics from kwargs
            for key, value in kwargs.items():
                if hasattr(file_state, key):
                    setattr(file_state, key, value)
    
    def update_generation_result(self, file_path: str, success: bool, 
                               tests_generated: int = 0, error: Optional[str] = None,
                               test_file_path: Optional[str] = None) -> None:
        """Update generation results for a file."""
        with self._lock:
            if file_path in self._files:
                file_state = self._files[file_path]
                file_state.generation_success = success
                file_state.tests_generated = tests_generated
                file_state.test_file_path = test_file_path
                if error:
                    file_state.generation_error = error
                    
    def update_refinement_result(self, file_path: str, iteration: int,
                               success: bool, errors: List[str] = None) -> None:
        """Update refinement results for a file."""
        with self._lock:
            if file_path in self._files:
                file_state = self._files[file_path]
                file_state.refinement_iterations = iteration
                file_state.final_test_success = success
                if errors:
                    file_state.test_errors.extend(errors)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all files."""
        with self._lock:
            total_files = len(self._files)
            completed = sum(1 for f in self._files.values() if f.status == FileStatus.COMPLETED)
            failed = sum(1 for f in self._files.values() if f.status == FileStatus.FAILED)
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
                "files_per_minute": (completed + failed) / max(total_duration / 60, 1)
            }
    
    async def _update_display_loop(self) -> None:
        """Continuous update loop for the live display."""
        sleep_time = 0.5 if self.minimal_mode else 0.33  # 2 Hz for minimal, 3 Hz for classic
        while self._is_running:
            try:
                self._update_display()
                await asyncio.sleep(sleep_time)
            except Exception as e:
                # Don't crash the display loop
                continue
    
    def _update_display(self) -> None:
        """Update the live display with current file status."""
        if not self._layout or not self._is_running:
            return
            
        with self._lock:
            # Update files table
            files_table = self._create_files_table()
            if self.minimal_mode:
                # In minimal mode, files table is directly in main
                self._layout["main"].update(files_table)
            else:
                # In classic mode, files table is in files section
                self._layout["files"].update(files_table)
            
            # Update statistics panel (skip in minimal mode)
            if not self.minimal_mode:
                stats_panel = self._create_stats_panel()
                self._layout["stats"].update(stats_panel)
            
            # Update footer with overall progress
            footer_content = self._create_footer_content()
            self._layout["footer"].update(footer_content)
    
    def _create_files_table(self) -> Table:
        """Create the clean, minimal files status table."""
        table = Table(
            show_header=True,
            header_style="header",
            border_style="border",
            show_lines=False,
            expand=True,
            box=None  # Remove box for cleaner look
        )
        
        # Use lowercase headers for consistency (especially in minimal mode)
        table.add_column("file", style="primary", width=30)
        table.add_column("status", justify="center", width=12)
        table.add_column("progress", justify="center", width=15)
        table.add_column("tests", justify="center", width=8)
        table.add_column("time", justify="center", width=8)
        
        # Sort files by status and name
        files_to_show = []
        for file_path in self._file_order:
            if file_path in self._files:
                files_to_show.append(self._files[file_path])
        
        # Show recent/active files first
        files_to_show.sort(key=lambda f: (f.is_complete(), f.last_update), reverse=True)
        
        # Show fewer files in minimal mode (top 10), more in classic (12)
        max_files = 10 if self.minimal_mode else 12
        for file_state in files_to_show[:max_files]:
            file_name = Path(file_state.file_path).name
            
            # Minimal status display
            status_text, status_color = self._get_minimal_status_display(file_state.status)
            
            # Clean progress indicator
            if file_state.progress_percentage > 0:
                progress_dots = "●" * int(file_state.progress_percentage / 25)  # 4 dots max
                progress_empty = "○" * (4 - len(progress_dots))
                if file_state.status == FileStatus.COMPLETED:
                    progress_display = f"[success]●●●●[/]"
                elif file_state.status == FileStatus.FAILED:
                    progress_display = f"[error]○○○○[/]"
                else:
                    progress_display = f"[accent]{progress_dots}[/][muted]{progress_empty}[/]"
            else:
                progress_display = "[muted]○○○○[/]"
            
            # Minimal tests display
            tests_display = str(file_state.tests_generated) if file_state.tests_generated > 0 else "—"
            
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
                f"[muted]{duration_display}[/]"
            )
        
        return table
    
    def _create_stats_panel(self) -> Panel:
        """Create the clean, minimal statistics panel."""
        stats = self.get_summary_stats()
        
        content_lines = [
            f"[success]done[/] {stats['completed']}",
            f"[error]failed[/] {stats['failed']}",
            f"[status_working]active[/] {stats['in_progress']}",
            "",
            f"tests {stats['total_tests_generated']}",
            f"rate {stats['success_rate']:.0%}",
        ]
        
        return Panel(
            "\n".join(content_lines),
            title="stats", 
            border_style="border",
            padding=(1, 1)
        )
    
    def _create_footer_content(self) -> Panel:
        """Create clean footer with minimal overall progress."""
        stats = self.get_summary_stats()
        total_processed = stats['completed'] + stats['failed']
        total_files = stats['total_files']
        
        if total_files > 0:
            overall_progress = total_processed / total_files
            # Simple progress dots - use 10 dots for minimal, 10 for classic too
            dot_count = 10
            progress_dots = "●" * int(overall_progress * dot_count)
            progress_empty = "○" * (dot_count - len(progress_dots))
            progress_text = f"[accent]{progress_dots}[/][muted]{progress_empty}[/] {total_processed}/{total_files}"
        else:
            progress_text = "[muted]starting...[/]"
        
        elapsed = stats['total_duration']
        if elapsed < 60:
            time_text = f"{elapsed:.0f}s"
        else:
            mins, secs = divmod(elapsed, 60)
            time_text = f"{int(mins)}m{secs:02.0f}s"
        
        footer_text = f"progress {progress_text}  •  {time_text}"
        
        return Panel(
            footer_text,
            border_style="border",
            padding=(0, 1),
            title=None
        )
    
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
    
    def __init__(self, ui_adapter: EnhancedUIAdapter, title: str = "File Processing"):
        self.tracker = FileStatusTracker(ui_adapter)
        self.title = title
        self._initialized = False
    
    def __enter__(self) -> 'LiveFileTracking':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.stop_live_tracking()
        
    def initialize_and_start(self, file_paths: List[str]) -> FileStatusTracker:
        """Initialize files and start tracking."""
        self.tracker.initialize_files(file_paths)
        self.tracker.start_live_tracking(self.title)
        self._initialized = True
        return self.tracker
