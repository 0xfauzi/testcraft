"""
Logs widget for displaying application logs.

Wraps RichLog with TestCraft-specific functionality including
log filtering, level indicators, and automatic scrolling.
"""

import logging
from datetime import datetime
from typing import Any

from rich.text import Text
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import RichLog, Static

from ..events import LogMessage


class Logs(Vertical):
    """
    Enhanced logs widget with filtering and controls.

    Provides a RichLog with additional controls for filtering
    by log level and searching through log messages.
    """

    # Reactive attributes for filtering
    filter_level: reactive[str] = reactive("INFO")
    auto_scroll: reactive[bool] = reactive(True)

    # Log levels with styling
    LEVEL_STYLES = {
        "DEBUG": "dim cyan",
        "INFO": "blue",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold red",
    }

    LEVEL_PRIORITY = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_class("logs-widget")

        # Log storage for filtering
        self._log_entries: list[dict[str, Any]] = []
        self._rich_log: RichLog | None = None

    def compose(self):
        """Compose the logs widget."""
        # Controls (simple version for now)
        yield Static("Logs", id="logs-header")

        # Main log display
        self._rich_log = RichLog(
            auto_scroll=True,
            max_lines=1000,
            wrap=True,
            markup=True,
            highlight=True,
        )
        yield self._rich_log

    def on_mount(self) -> None:
        """Initialize the logs widget."""
        # Welcome message
        self.write_log(
            "INFO", "TestCraft TUI started", datetime.now().strftime("%H:%M:%S")
        )

    def write_log(self, level: str, message: str, timestamp: str | None = None) -> None:
        """Write a log message."""
        if not timestamp:
            timestamp = datetime.now().strftime("%H:%M:%S")

        # Store the log entry
        log_entry = {
            "level": level.upper(),
            "message": message,
            "timestamp": timestamp,
        }
        self._log_entries.append(log_entry)

        # Apply filtering
        if self._should_show_log(log_entry):
            self._display_log_entry(log_entry)

        # Trim old entries if we have too many
        if len(self._log_entries) > 1000:
            self._log_entries = self._log_entries[-500:]  # Keep last 500

    def _should_show_log(self, log_entry: dict[str, Any]) -> bool:
        """Check if a log entry should be displayed based on current filter."""
        entry_level = log_entry["level"]
        filter_priority = self.LEVEL_PRIORITY.get(self.filter_level, 20)
        entry_priority = self.LEVEL_PRIORITY.get(entry_level, 20)

        return entry_priority >= filter_priority

    def _display_log_entry(self, log_entry: dict[str, Any]) -> None:
        """Display a single log entry in the RichLog."""
        if not self._rich_log:
            return

        level = log_entry["level"]
        message = log_entry["message"]
        timestamp = log_entry["timestamp"]

        # Create formatted log line
        log_line = Text()

        # Add timestamp
        log_line.append(f"[{timestamp}] ", style="dim")

        # Add level with styling
        level_style = self.LEVEL_STYLES.get(level, "white")
        log_line.append(f"{level:8}", style=level_style)

        # Add separator
        log_line.append(" | ", style="dim")

        # Add message
        log_line.append(message)

        # Write to RichLog
        self._rich_log.write(log_line)

    def set_filter_level(self, level: str) -> None:
        """Set the minimum log level to display."""
        old_level = self.filter_level
        self.filter_level = level.upper()

        # If filter changed, refresh the display
        if old_level != self.filter_level:
            self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh the log display with current filter settings."""
        if not self._rich_log:
            return

        # Clear current display
        self._rich_log.clear()

        # Re-display filtered logs
        for log_entry in self._log_entries:
            if self._should_show_log(log_entry):
                self._display_log_entry(log_entry)

    def clear_logs(self) -> None:
        """Clear all log entries."""
        self._log_entries.clear()
        if self._rich_log:
            self._rich_log.clear()

    def get_log_count_by_level(self) -> dict[str, int]:
        """Get count of log entries by level."""
        counts = {}
        for entry in self._log_entries:
            level = entry["level"]
            counts[level] = counts.get(level, 0) + 1
        return counts

    def export_logs(self) -> str:
        """Export all logs as formatted text."""
        lines = []
        for entry in self._log_entries:
            timestamp = entry["timestamp"]
            level = entry["level"]
            message = entry["message"]
            lines.append(f"[{timestamp}] {level:8} | {message}")
        return "\n".join(lines)

    def on_log_message(self, event: LogMessage) -> None:
        """Handle log message events."""
        self.write_log(event.level, event.message, event.timestamp)


class SimpleLogHandler(logging.Handler):
    """
    A logging handler that sends logs to a Logs widget.

    Can be attached to Python's logging system to capture
    application logs and display them in the TUI.
    """

    def __init__(self, logs_widget: Logs) -> None:
        super().__init__()
        self.logs_widget = logs_widget

        # Set up formatting
        formatter = logging.Formatter("%(name)s - %(message)s")
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the logs widget."""
        try:
            message = self.format(record)
            timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            self.logs_widget.write_log(record.levelname, message, timestamp)
        except Exception:
            # Don't let logging errors break the app
            pass
