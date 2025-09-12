"""
Custom event/message classes for TestCraft Textual UI.

These events bridge between the domain layer (use cases) and the UI layer,
allowing clean separation of concerns while enabling reactive updates.
"""

from typing import Any, Dict, List, Optional
from textual.message import Message


class ProgressUpdated(Message):
    """Sent when overall progress is updated."""
    
    def __init__(self, current: int, total: int, message: str = "") -> None:
        self.current = current
        self.total = total
        self.message = message
        super().__init__()


class FileStatusChanged(Message):
    """Sent when a file's processing status changes."""
    
    def __init__(
        self,
        file_path: str,
        status: str,
        progress: float = 0.0,
        tests_generated: int = 0,
        duration: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        self.file_path = file_path
        self.status = status
        self.progress = progress
        self.tests_generated = tests_generated
        self.duration = duration
        self.error = error
        super().__init__()


class ResultsReady(Message):
    """Sent when processing results are available."""
    
    def __init__(self, results: Dict[str, Any]) -> None:
        self.results = results
        super().__init__()


class StatsUpdated(Message):
    """Sent when aggregate statistics are updated."""
    
    def __init__(self, stats: Dict[str, Any]) -> None:
        self.stats = stats
        super().__init__()


class ErrorOccurred(Message):
    """Sent when an error occurs that should be displayed to the user."""
    
    def __init__(self, error: str, details: Optional[str] = None) -> None:
        self.error = error
        self.details = details
        super().__init__()


class OperationStarted(Message):
    """Sent when a long-running operation begins."""
    
    def __init__(self, operation: str, message: str = "") -> None:
        self.operation = operation
        self.message = message
        super().__init__()


class OperationCompleted(Message):
    """Sent when a long-running operation completes."""
    
    def __init__(
        self,
        operation: str,
        success: bool = True,
        message: str = "",
        results: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.operation = operation
        self.success = success
        self.message = message
        self.results = results or {}
        super().__init__()


class LogMessage(Message):
    """Sent when a log message should be displayed."""
    
    def __init__(self, level: str, message: str, timestamp: str) -> None:
        self.level = level
        self.message = message
        self.timestamp = timestamp
        super().__init__()


class ConfigurationChanged(Message):
    """Sent when configuration is updated."""
    
    def __init__(self, config_section: str, changes: Dict[str, Any]) -> None:
        self.config_section = config_section
        self.changes = changes
        super().__init__()
