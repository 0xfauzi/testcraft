"""
Enhanced UI adapter with sophisticated Rich components and logging integration.

This module provides a comprehensive UI system with:
- Advanced progress tracking with multiple stages
- Rich tables for all structured data
- Beautiful panels and sections
- Animated progress indicators
- Integrated logging display
- Real-time status dashboards
"""

import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.status import Status
from rich.table import Table

from .ui_rich import RichUIAdapter, UIStyle


class EnhancedUIAdapter(RichUIAdapter):
    """
    Enhanced UI adapter with sophisticated Rich components.

    Extends the basic RichUIAdapter with:
    - Multi-stage progress tracking
    - Advanced table displays
    - Real-time dashboards
    - Animated status indicators
    - Rich logging integration
    - Performance metrics display
    """

    # Class-level lock for thread-safe logging setup
    _logging_setup_lock = threading.Lock()
    _logging_handler_ref: RichHandler | None = None

    def __init__(
        self,
        console: Console | None = None,
        enable_rich_logging: bool = True,
        ui_style: UIStyle = UIStyle.CLASSIC,
    ):
        """Initialize enhanced UI with rich logging support."""
        super().__init__(console)

        # Store UI style for rendering decisions
        self.ui_style = ui_style

        # Enhanced progress tracking
        self._progress_stages: dict[str, TaskID] = {}
        self._active_live_display: Live | None = None
        self._current_dashboard: Layout | None = None

        # Rich logging setup
        if enable_rich_logging:
            self._setup_rich_logging()

        # Performance tracking
        self._operation_start_times: dict[str, float] = {}
        self._metrics: dict[str, Any] = {}

    def _setup_rich_logging(self) -> None:
        """Set up beautiful Rich logging integration with thread safety."""
        with self._logging_setup_lock:
            # Check if we already set up logging for this instance
            if hasattr(self, "logger") and getattr(self, "logger", None):
                return

            # Check if root logger already has a RichHandler - if so, do nothing
            root_logger = logging.getLogger()
            has_rich_handler = any(
                isinstance(h, RichHandler) for h in root_logger.handlers
            )

            if has_rich_handler and self._logging_handler_ref:
                # Root already configured, just create our logger
                self.logger = logging.getLogger("testcraft.enhanced_ui")
                self.logger.handlers = []  # Remove any existing handlers
                self.logger.propagate = True  # Ensure it uses root handler
                return

            try:
                # If we reach here, we're likely in library usage mode
                # Set up minimal RichHandler on root (this should rarely happen in CLI mode)
                rich_handler = RichHandler(
                    console=self.console,
                    show_time=True,
                    show_path=False,
                    markup=False,  # Avoid rich markup leaking in minimal logs
                    rich_tracebacks=True,
                )

                # Store reference to prevent duplicate checks
                self._logging_handler_ref = rich_handler

                # Minimal formatter
                rich_handler.setFormatter(logging.Formatter(fmt="%(message)s"))

                # Only add if no RichHandler exists
                if not any(isinstance(h, RichHandler) for h in root_logger.handlers):
                    root_logger.addHandler(rich_handler)
                    root_logger.setLevel(logging.INFO)

                # Create enhanced logger for this class with no handlers, using propagation
                self.logger = logging.getLogger("testcraft.enhanced_ui")
                self.logger.handlers = []  # Ensure no per-logger handlers
                self.logger.propagate = True

            except Exception:
                # Fallback: create a simple logger without Rich if setup fails
                self.logger = logging.getLogger("testcraft.enhanced_ui")
                self.logger.handlers = []
                self.logger.propagate = True

    def _cleanup_operation(self, operation_name: str) -> None:
        """Clean up operation tracking data safely."""
        try:
            if operation_name in self._operation_start_times:
                del self._operation_start_times[operation_name]
        except KeyError:
            pass  # Already cleaned up

    @contextmanager
    def create_operation_tracker(self, operation_name: str, total_steps: int = 1):
        """Context manager for tracking multi-step operations with beautiful progress."""
        self._operation_start_times[operation_name] = time.time()

        # Minimal mode: use ephemeral status only; Classic: use persistent progress
        if self.ui_style == UIStyle.MINIMAL:
            status = Status(
                f"{operation_name}...", console=self.console, spinner="dots"
            )
        else:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=False,  # Keep progress visible after completion
            )

        if self.ui_style != UIStyle.MINIMAL:
            main_task = progress.add_task(
                f"[cyan]{operation_name}[/]", total=total_steps
            )

        class OperationTracker:
            def __init__(self, progress_instance, task_id, ui_adapter) -> None:
                self.progress = progress_instance
                self.task_id = task_id
                self.ui = ui_adapter
                self.current_step = 0

            def advance_step(self, description: str = "", increment: int = 1):
                """Advance to the next step with optional description."""
                self.current_step += increment
                if self.ui.ui_style == UIStyle.MINIMAL:
                    # Update ephemeral status text only; no persisted progress line
                    if description:
                        self.ui.console.log(
                            f"{description}"
                        ) if False else None  # no-op placeholder
                else:
                    if description:
                        self.progress.update(
                            self.task_id, description=f"[cyan]{description}[/]"
                        )
                    self.progress.advance(self.task_id, increment)

            def update_description(self, description: str):
                """Update the current step description."""
                if self.ui.ui_style != UIStyle.MINIMAL:
                    self.progress.update(
                        self.task_id, description=f"[cyan]{description}[/]"
                    )

            def log_progress(self, message: str, level: str = "info"):
                """Log progress with rich formatting."""
                if hasattr(self.ui, "logger"):
                    log_func = getattr(self.ui.logger, level, self.ui.logger.info)
                    log_func(f"[bold cyan]{operation_name}:[/] {message}")

        try:
            if self.ui_style == UIStyle.MINIMAL:
                with status:
                    tracker = OperationTracker(None, None, self)
                    yield tracker
                    # Status is transient; nothing persisted
                    duration = time.time() - self._operation_start_times[operation_name]
                    self._metrics[operation_name] = {
                        "duration": duration,
                        "steps": total_steps,
                        "avg_step_time": duration / max(total_steps, 1),
                    }
            else:
                with progress:
                    tracker = OperationTracker(progress, main_task, self)
                    yield tracker

                    # Mark as complete
                    progress.update(
                        main_task, description=f"[green]✓ {operation_name} Complete[/]"
                    )

                    # Calculate performance metrics
                    duration = time.time() - self._operation_start_times[operation_name]
                    self._metrics[operation_name] = {
                        "duration": duration,
                        "steps": total_steps,
                        "avg_step_time": duration / max(total_steps, 1),
                    }
        except Exception:
            # Ensure cleanup happens even on exceptions
            duration = time.time() - self._operation_start_times[operation_name]
            self._metrics[operation_name] = {
                "duration": duration,
                "steps": total_steps,
                "avg_step_time": duration / max(total_steps, 1),
                "error": True,
            }
            raise
        finally:
            # Always clean up operation tracking data
            self._cleanup_operation(operation_name)

    def create_real_time_dashboard(
        self, title: str = "TestCraft Dashboard"
    ) -> "DashboardManager":
        """Create a real-time updating dashboard for complex operations."""
        return DashboardManager(self, title)

    def display_file_progress_table(
        self, files_data: list[dict[str, Any]], title: str = "File Processing Status"
    ):
        """Display a clean, minimal table showing file processing progress."""

        def _create_and_display_table():
            # Input validation
            if not isinstance(files_data, list):
                self.display_error_with_suggestions(
                    f"Invalid files_data type: expected list, got {type(files_data)}",
                    ["Ensure files_data is a list of dictionaries"],
                )
                return

            if not files_data:
                return  # Nothing to display

            table = Table(
                title=f"[title]{title}[/]",
                show_header=True,
                header_style="header",
                border_style="border",
                title_style="title",
                show_lines=False,
                expand=True,
                box=None,  # Remove box for cleaner look
            )

            # Minimal columns
            table.add_column("File", style="primary", width=35)
            table.add_column("Status", justify="center", width=12)
            table.add_column("Progress", justify="center", width=15)
            table.add_column("Tests", justify="center", width=8)
            table.add_column("Time", justify="center", width=8)

            for file_data in files_data:
                # Defensive dict access with validation
                if not isinstance(file_data, dict):
                    continue  # Skip invalid entries

                file_path = str(file_data.get("file_path", ""))
                file_name = Path(file_path).name if file_path else "Unknown"

                # Clean minimal status
                status = file_data.get("status", "pending")
                if status == "completed":
                    status_display = "[success]done[/]"
                elif status in [
                    "processing",
                    "generating",
                    "writing",
                    "testing",
                    "refining",
                ]:
                    status_display = "[status_working]active[/]"
                elif status == "failed":
                    status_display = "[error]failed[/]"
                else:
                    status_display = "[muted]waiting[/]"

                # Simple progress dots
                progress_val = file_data.get("progress", 0.0)
                if status == "completed":
                    progress_display = "[success]●●●●[/]"
                elif status == "failed":
                    progress_display = "[error]○○○○[/]"
                elif progress_val > 0:
                    dots = min(
                        4, max(0, int(progress_val * 4))
                    )  # 4 dots max, ensure 0-4 range
                    filled = "●" * dots
                    empty = "○" * (4 - dots)
                    progress_display = f"[accent]{filled}[/][muted]{empty}[/]"
                else:
                    progress_display = "[muted]○○○○[/]"

                # Minimal tests display
                tests_count = file_data.get("tests_generated", 0)
                tests_display = str(tests_count) if tests_count > 0 else "—"

                # Clean duration
                duration = file_data.get("duration", 0)
                if duration > 0:
                    if duration < 60:
                        duration_display = f"{duration:.1f}s"
                    else:
                        mins, secs = divmod(duration, 60)
                        duration_display = f"{int(mins)}m{secs:.0f}s"
                else:
                    duration_display = "—"

                table.add_row(
                    file_name,
                    status_display,
                    progress_display,
                    tests_display,
                    f"[muted]{duration_display}[/]",
                )

            self.console.print(table)

        # Wrap the entire operation in safe_execute
        self._safe_execute("display_file_progress_table", _create_and_display_table)

    def display_metrics_panel(
        self, metrics: dict[str, Any], title: str = "Performance Metrics"
    ):
        """Display performance metrics in a clean, minimal panel."""

        def _create_and_display_metrics():
            # Input validation
            if not isinstance(metrics, dict):
                return

            metrics_content = []

            for operation, data in metrics.items():
                if not isinstance(data, dict):
                    continue

                duration = data.get("duration", 0)
                success_rate = data.get("success_rate", 0)
                items = data.get("items_processed", 0)

                metrics_content.append(f"[primary]{operation}[/]")
                metrics_content.append(
                    f"  {duration:.1f}s • {items} items • {success_rate:.0%} success"
                )
                metrics_content.append("")

            if metrics_content:
                panel = Panel(
                    "\n".join(metrics_content[:-1]),  # Remove last empty line
                    title=title.lower(),
                    border_style="border",
                    padding=(1, 1),
                )
                self.console.print(panel)

        self._safe_execute("display_metrics_panel", _create_and_display_metrics)

    def display_error_with_suggestions(
        self, error_message: str, suggestions: list[str], title: str = "Error"
    ):
        """Display error with helpful suggestions in minimal style."""
        error_content = [f"[error]{error_message}[/]"]

        if suggestions:
            error_content.append("")
            error_content.append("[warning]suggestions:[/]")
            for suggestion in suggestions:
                error_content.append(f"  {suggestion}")

        panel = Panel(
            "\n".join(error_content),
            title=f"[error]{title.lower()}[/]",
            border_style="border",
            padding=(1, 1),
        )
        self.console.print(panel)

    def _safe_execute(self, operation_name: str, operation_func, *args, **kwargs):
        """Safely execute UI operations with error handling."""
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            # Log error but don't crash the UI
            if hasattr(self, "logger") and self.logger:
                self.logger.warning(f"UI operation '{operation_name}' failed: {e}")
            # Return None or empty result to allow graceful degradation
            return None

    def display_success_summary(self, summary_data: dict[str, Any]):
        """Display a clean success summary with essential metrics."""

        def _create_and_display_summary():
            # Input validation
            if not isinstance(summary_data, dict):
                return

            # Simple success message
            main_message = summary_data.get(
                "message", "Operation completed successfully!"
            )
            self.display_success(main_message, "success")

            # Minimal metrics display
            if "metrics" in summary_data:
                metrics = summary_data.get("metrics")
                if isinstance(metrics, dict):
                    self.display_metrics_panel(metrics, "metrics")

            # Clean file results if available
            if "files_processed" in summary_data:
                files_processed = summary_data.get("files_processed")
                if isinstance(files_processed, list):
                    self.display_file_progress_table(files_processed, "results")

        self._safe_execute("display_success_summary", _create_and_display_summary)

    def get_renderer(self):
        """Get the appropriate renderer based on UI style."""
        if self.ui_style == UIStyle.MINIMAL:
            return MinimalRenderer()
        else:
            # Return self for classic rendering using existing methods
            return self


class MinimalRenderer:
    """Minimal renderer for single-pass, compact UI output."""

    def render_generation_results(self, results: dict, console: Console):
        """Render generation results with minimal single-line summary + compact table."""

        # Extract key metrics
        files_written = results.get("files_written", 0)
        files_processed = results.get("files_processed", 0)
        tests_generated = results.get("tests_generated", 0)
        total_duration = results.get("total_duration", 0)

        # Coverage delta (only if > 0)
        coverage_delta = results.get("coverage_delta", {})
        line_coverage_delta = coverage_delta.get("line_coverage_delta", 0)

        # One-line summary
        summary_parts = [
            f"done {files_written}/{files_processed}",
            f"tests {tests_generated}",
        ]

        if line_coverage_delta > 0:
            summary_parts.append(f"Δcov {line_coverage_delta:+.1%}")

        summary_parts.append(f"time {total_duration:.1f}s")

        summary_line = " • ".join(summary_parts)
        console.print(summary_line)

        # Compact table - reuse existing display_file_progress_table logic but without title
        generation_results = results.get("generation_results", [])
        refinement_results = results.get("refinement_results", [])

        # Create lookup for refinement results
        refinement_by_file = {}
        for refine_result in refinement_results:
            file_path = refine_result.get("test_file", "")
            refinement_by_file[file_path] = refine_result

        # Build file data for table
        files_data = []
        for gen_result in generation_results:
            if hasattr(gen_result, "file_path"):
                file_path = gen_result.file_path
                success = gen_result.success
            else:
                file_path = gen_result.get("file_path", "unknown")
                success = gen_result.get("success", False)

            # Get refinement data
            refine_result = refinement_by_file.get(file_path)
            refine_success = (
                refine_result.get("success", False) if refine_result else True
            )

            # Determine final status
            final_success = success and refine_success

            file_data = {
                "file_path": file_path,
                "status": "completed" if final_success else "failed",
                "progress": 1.0 if final_success else 0.5 if success else 0.0,
                "tests_generated": refine_result.get("tests_generated", 0)
                if refine_result
                else (5 if success else 0),
                "duration": refine_result.get("duration", 0) if refine_result else 0,
            }
            files_data.append(file_data)

        # Compact table with no title, lowercase headers, minimal styling
        # Only show table for multi-file runs; single-file stays summary-only
        if files_data and len(files_data) > 1:
            self._render_compact_table(files_data, console)

    def _render_compact_table(self, files_data: list[dict[str, Any]], console: Console):
        """Render a compact table with minimal styling."""
        try:
            table = Table(
                show_header=True,
                header_style="header",
                border_style="border",
                show_lines=False,
                expand=True,
                box=None,  # Remove box for cleaner look
                padding=(0, 1),  # Minimal padding
            )
        except Exception:
            # For table creation, just return early on error since it's not critical
            return

        # Lowercase headers
        table.add_column("file", style="primary", width=35)
        table.add_column("status", justify="center", width=12)
        table.add_column("progress", justify="center", width=15)
        table.add_column("tests", justify="center", width=8)
        table.add_column("time", justify="center", width=8)

        for file_data in files_data:
            # Defensive dict access with validation
            if not isinstance(file_data, dict):
                continue

            file_path = str(file_data.get("file_path", ""))
            file_name = Path(file_path).name if file_path else "Unknown"

            # Clean minimal status
            status = file_data.get("status", "pending")
            if status == "completed":
                status_display = "[success]done[/]"
            elif status in [
                "processing",
                "generating",
                "writing",
                "testing",
                "refining",
            ]:
                status_display = "[status_working]active[/]"
            elif status == "failed":
                status_display = "[error]failed[/]"
            else:
                status_display = "[muted]waiting[/]"

            # Simple progress dots (4 dots max)
            progress_val = file_data.get("progress", 0.0)
            if status == "completed":
                progress_display = "[success]●●●●[/]"
            elif status == "failed":
                progress_display = "[error]○○○○[/]"
            elif progress_val > 0:
                dots = min(
                    4, max(0, int(progress_val * 4))
                )  # 4 dots max, ensure 0-4 range
                filled = "●" * dots
                empty = "○" * (4 - dots)
                progress_display = f"[accent]{filled}[/][muted]{empty}[/]"
            else:
                progress_display = "[muted]○○○○[/]"

            # Minimal tests display
            tests_count = file_data.get("tests_generated", 0)
            tests_display = str(tests_count) if tests_count > 0 else "—"

            # Clean duration
            duration = file_data.get("duration", 0)
            if duration > 0:
                if duration < 60:
                    duration_display = f"{duration:.1f}s"
                else:
                    mins, secs = divmod(duration, 60)
                    duration_display = f"{int(mins)}m{secs:.0f}s"
            else:
                duration_display = "—"

                table.add_row(
                    file_name,
                    status_display,
                    progress_display,
                    tests_display,
                    f"[muted]{duration_display}[/]",
                )

            console.print(table)


class DashboardManager:
    """Manager for real-time updating dashboards."""

    def __init__(self, ui_adapter: EnhancedUIAdapter, title: str) -> None:
        self.ui = ui_adapter
        self.title = title
        self.layout = Layout()
        self.live_display: Live | None = None
        self._data: dict[str, Any] = {}

        # Setup layout structure
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )

        # Initialize with header
        self.layout["header"].update(
            Panel(
                Align.center(f"[bold bright_magenta]{title}[/]"), style="bright_magenta"
            )
        )

    def __enter__(self):
        """Start the live dashboard."""
        try:
            self.live_display = Live(
                self.layout,
                console=self.ui.console,
                refresh_per_second=2,
                transient=False,
            )
            self.live_display.start()
            return self
        except Exception:
            # If start fails, set to None to prevent cleanup errors
            self.live_display = None
            raise

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the live dashboard."""
        if self.live_display is not None:
            try:
                self.live_display.stop()
            except Exception:
                pass  # Ignore cleanup errors

    def update_main_content(self, content: Any) -> Any:
        """Update the main content area."""
        self.layout["left"].update(content)

    def update_sidebar(self, content: Any) -> Any:
        """Update the sidebar content."""
        self.layout["right"].update(content)

    def update_footer(self, content: Any) -> Any:
        """Update the footer content."""
        self.layout["footer"].update(content)

    def set_data(self, key: str, value: Any):
        """Set data for the dashboard."""
        self._data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data from the dashboard."""
        return self._data.get(key, default)
