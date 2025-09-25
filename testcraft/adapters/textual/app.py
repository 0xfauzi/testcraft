"""
Main TestCraft Textual Application.

This module provides the core TestCraftTextualApp that manages screens,
routing, global theme, and coordinates the overall TUI experience.
"""

import logging
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.driver import Driver
from textual.widgets import Footer, Header

from .events import (
    ErrorOccurred,
    FileStatusChanged,
    LogMessage,
    OperationCompleted,
    OperationStarted,
    ProgressUpdated,
    StatsUpdated,
)
from .screens.analyze_screen import AnalyzeScreen
from .screens.coverage_screen import CoverageScreen
from .screens.generate_screen import GenerateScreen
from .screens.status_screen import StatusScreen
from .screens.wizard_screen import WizardScreen


class TestCraftTextualApp(App[None]):
    """
    Main TestCraft Textual Application.

    Provides a rich TUI for TestCraft operations including Generate, Analyze,
    Coverage, and Status screens. Manages global state, theme, and routing.
    """

    TITLE = "TestCraft"
    SUB_TITLE = "AI-Powered Test Generation"
    CSS_PATH = Path(__file__).parent / "theme.tcss"

    # Keybindings for global navigation
    BINDINGS = [
        ("ctrl+c,q", "quit", "Quit"),
        ("g", "show_generate", "Generate"),
        ("a", "show_analyze", "Analyze"),
        ("c", "show_coverage", "Coverage"),
        ("s", "show_status", "Status"),
        ("w", "show_wizard", "Wizard"),
        ("/", "search", "Search"),
        ("?", "help", "Help"),
        ("d", "toggle_dark", "Toggle Dark Mode"),
        ("r", "refresh", "Refresh"),
        ("l", "show_logs", "Show Logs"),
    ]

    # Screen registry - static screen definitions
    SCREENS = {
        "generate": GenerateScreen,
        "analyze": AnalyzeScreen,
        "coverage": CoverageScreen,
        "status": StatusScreen,
        "wizard": WizardScreen,
    }

    def __init__(
        self,
        driver_class: type[Driver] | None = None,
        css_path: str | None = None,
        watch_css: bool = False,
    ):
        super().__init__(
            driver_class=driver_class,
            css_path=css_path,
            watch_css=watch_css,
        )

        # Global application state
        self.current_operation: str | None = None
        self.operation_stats: dict[str, Any] = {}
        self.file_states: dict[str, dict[str, Any]] = {}

        # Setup logging integration
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging integration with the TUI."""

        # Create a custom handler that sends logs to the TUI
        class TUILogHandler(logging.Handler):
            def __init__(self, app: "TestCraftTextualApp") -> None:
                super().__init__()
                self.app = app

            def emit(self, record: logging.LogRecord) -> None:
                try:
                    message = self.format(record)
                    self.app.post_message(
                        LogMessage(
                            level=record.levelname,
                            message=message,
                            timestamp=self.formatTime(record),
                        )
                    )
                except Exception:
                    # Don't let logging errors break the app
                    pass

        # Add our handler to the testcraft logger
        logger = logging.getLogger("testcraft")
        handler = TUILogHandler(self)
        handler.setFormatter(
            logging.Formatter("%(levelname)s - %(name)s - %(message)s")
        )
        logger.addHandler(handler)

    def on_mount(self) -> None:
        """Initialize the application on startup."""
        self.title = "TestCraft - AI-Powered Test Generation"

        # Start with the generate screen
        self.push_screen("generate")

    def compose(self) -> ComposeResult:
        """Compose the basic app structure."""
        yield Header()
        yield Footer()

    # Action handlers for global navigation
    def action_show_generate(self) -> None:
        """Switch to the Generate screen."""
        self.switch_screen("generate")

    def action_show_analyze(self) -> None:
        """Switch to the Analyze screen."""
        self.switch_screen("analyze")

    def action_show_coverage(self) -> None:
        """Switch to the Coverage screen."""
        self.switch_screen("coverage")

    def action_show_status(self) -> None:
        """Switch to the Status screen."""
        self.switch_screen("status")

    def action_show_wizard(self) -> None:
        """Switch to the Wizard screen."""
        self.switch_screen("wizard")

    def action_show_logs(self) -> None:
        """Show or toggle the logs display."""
        # This could switch to a logs screen or toggle a logs panel
        self.log("Logs display toggle requested")

    def action_search(self) -> None:
        """Open search/filter interface."""
        # Could open a search modal or command palette
        self.log("Search requested")

    def action_help(self) -> None:
        """Show help/keybindings."""
        # Could open a help modal or screen
        self.log("Help requested")

    def action_refresh(self) -> None:
        """Refresh current screen data."""
        current_screen = self.screen
        if hasattr(current_screen, "refresh"):
            current_screen.refresh()
        else:
            self.log("Refreshing current screen")

    def action_toggle_dark(self) -> None:
        """Toggle dark/light mode."""
        self.dark = not self.dark

    # Global event handlers for cross-screen communication
    def on_progress_updated(self, event: ProgressUpdated) -> None:
        """Handle progress updates from any screen."""
        self.log(f"Progress: {event.current}/{event.total} - {event.message}")
        # Update global progress state or forward to appropriate screen

    def on_file_status_changed(self, event: FileStatusChanged) -> None:
        """Handle file status changes."""
        self.file_states[event.file_path] = {
            "status": event.status,
            "progress": event.progress,
            "tests_generated": event.tests_generated,
            "duration": event.duration,
            "error": event.error,
        }

        # Forward to current screen if it handles file updates
        current_screen = self.screen
        if hasattr(current_screen, "on_file_status_changed"):
            current_screen.on_file_status_changed(event)

    def on_stats_updated(self, event: StatsUpdated) -> None:
        """Handle statistics updates."""
        self.operation_stats.update(event.stats)

        # Forward to current screen if it handles stats
        try:
            current_screen = self.screen
            if hasattr(current_screen, "on_stats_updated"):
                current_screen.on_stats_updated(event)
        except Exception:
            # No screens on stack yet, or other screen access error
            # This can happen during app initialization
            pass

    def on_error_occurred(self, event: ErrorOccurred) -> None:
        """Handle error notifications."""
        self.log(f"Error: {event.error}")
        if event.details:
            self.log(f"Details: {event.details}")

        # Could show error toast or modal
        self.notify(event.error, severity="error")

    def on_operation_started(self, event: OperationStarted) -> None:
        """Handle operation start notifications."""
        self.current_operation = event.operation
        self.log(f"Operation started: {event.operation}")
        if event.message:
            self.notify(event.message, title=f"Starting {event.operation}")

    def on_operation_completed(self, event: OperationCompleted) -> None:
        """Handle operation completion notifications."""
        self.current_operation = None
        status = "completed" if event.success else "failed"
        self.log(f"Operation {status}: {event.operation}")

        severity = "information" if event.success else "error"
        message = event.message or f"{event.operation} {status}"
        self.notify(message, severity=severity)

    def on_log_message(self, event: LogMessage) -> None:
        """Handle log messages from the application."""
        # Forward to logs widget/screen if available
        # For now, just use the built-in log
        self.log(f"[{event.level}] {event.message}")

    def start_operation(self, operation: str, message: str = "") -> None:
        """Convenience method to start an operation."""
        self.post_message(OperationStarted(operation, message))

    def complete_operation(
        self,
        operation: str,
        success: bool = True,
        message: str = "",
        results: dict[str, Any] | None = None,
    ) -> None:
        """Convenience method to complete an operation."""
        self.post_message(OperationCompleted(operation, success, message, results))

    def update_file_status(
        self,
        file_path: str,
        status: str,
        progress: float = 0.0,
        tests_generated: int = 0,
        duration: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Convenience method to update file status."""
        # Update internal state directly (for testing and immediate access)
        self.file_states[file_path] = {
            "status": status,
            "progress": progress,
            "tests_generated": tests_generated,
            "duration": duration,
            "error": error,
        }

        # Also post message for reactive updates
        self.post_message(
            FileStatusChanged(
                file_path, status, progress, tests_generated, duration, error
            )
        )

    def update_stats(self, stats: dict[str, Any]) -> None:
        """Convenience method to update statistics."""
        self.post_message(StatsUpdated(stats))

    def report_error(self, error: str, details: str | None = None) -> None:
        """Convenience method to report an error."""
        self.post_message(ErrorOccurred(error, details))
