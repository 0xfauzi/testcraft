"""
Enhanced logging system with Rich integration and structured messages.

This module provides:
- Rich-formatted log messages with beautiful colors
- Structured logging with context
- Operation-specific loggers
- Performance tracking
- Error analysis and suggestions
- Progress-aware logging
"""

import csv
import json
import logging
import os
import queue
import re
import threading
import time
import unicodedata
import weakref
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape

from .rich_cli import TESTCRAFT_THEME


class LogMode(str, Enum):
    """Logging output modes."""

    CLASSIC = "classic"
    MINIMAL = "minimal"


class OutputFormat(str, Enum):
    """Structured output formats for machine parsing."""

    CONSOLE = "console"
    JSON = "json"
    CSV = "csv"


class StructuredLogger:
    """Enhanced logger with rich formatting and structured messages."""

    def __init__(self, name: str, console: Console | None = None) -> None:
        """Initialize structured logger with Rich formatting."""
        self.name = name
        self.console = console or Console(theme=TESTCRAFT_THEME)
        self.logger = logging.getLogger(name)
        self._setup_rich_handler()

        # Context tracking
        self._context_stack: list[dict[str, Any]] = []
        self._operation_context: dict[str, Any] = {}
        self._max_context_depth = 10  # Prevent stack overflow

    def _setup_rich_handler(self):
        """Set up rich handler for beautiful log output."""
        # Check if root logger has RichHandler - if so, rely on root
        root_logger = logging.getLogger()
        has_root_rich_handler = any(
            isinstance(h, RichHandler) for h in root_logger.handlers
        )

        if has_root_rich_handler:
            # Root is configured, remove per-logger handlers and use propagation
            self.logger.handlers = []
            self.logger.propagate = True
            return

        # Root not configured (library usage), set up global logging once and then rely on it
        if not LoggerManager._console:
            LoggerManager.setup_global_logging(self.console)

        # Ensure this logger uses root propagation
        self.logger.handlers = []
        self.logger.propagate = True

    @contextmanager
    def operation_context(self, operation: str, **context):
        """Context manager for operation-specific logging with proper stack management."""
        # Input validation
        if not operation or not isinstance(operation, str):
            raise ValueError("Operation name must be a non-empty string")

        # Check context depth to prevent stack overflow
        if len(self._context_stack) >= self._max_context_depth:
            raise RuntimeError(
                f"Context stack depth limit ({self._max_context_depth}) exceeded"
            )

        start_time = time.time()
        operation_id = f"{operation}_{int(start_time)}"

        # Create isolated context copy to prevent reference sharing
        context_copy = dict(context) if context else {}

        # Push context with isolation
        full_context = {
            "operation": operation,
            "operation_id": operation_id,
            "start_time": start_time,
            **context_copy,
        }

        # Validate context before pushing
        self._validate_context(full_context)

        self._context_stack.append(full_context)
        self._operation_context = full_context.copy()  # Ensure isolation

        # Make variables available to both try and except blocks
        exception_info = None

        # Log operation start
        if LoggerManager.log_mode == LogMode.CLASSIC:
            self.info(
                f"Starting {operation}",
                extra={"operation_start": True, **context_copy},
            )
        else:
            # Minimal one-liner (will likely be suppressed unless verbose)
            self.info(
                f"starting {operation}", extra={"operation_start": True, **context_copy}
            )

        try:
            yield self
        except Exception as e:
            # Store exception info for logging
            exception_info = e
            # Log operation failure
            duration = time.time() - start_time
            error_msg = str(e) if e else "Unknown error"
            self.error(
                (
                    f"{operation} failed after {duration:.2f}s: {escape(error_msg)}"
                    if LoggerManager.log_mode == LogMode.MINIMAL
                    else f"{operation} failed after {duration:.2f}s: {escape(error_msg)}"
                ),
                extra={
                    "operation_failed": True,
                    "duration": duration,
                    "error": error_msg,
                },
            )
            raise
        finally:
            # Pop context with proper stack discipline
            try:
                if self._context_stack:
                    self._context_stack.pop()

                # Update current context safely
                if self._context_stack:
                    self._operation_context = self._context_stack[-1].copy()
                else:
                    self._operation_context = {}

            except (IndexError, AttributeError) as e:
                # Log stack corruption but don't crash
                self.warning(f"Context stack corruption detected: {e}")
                self._context_stack = []
                self._operation_context = {}

            # Only log completion if no exception occurred
            if exception_info is None:
                duration = time.time() - start_time
                if LoggerManager.log_mode == LogMode.CLASSIC:
                    self.info(
                        f"{operation} completed in {duration:.1f}s",
                        extra={"operation_complete": True, "duration": duration},
                    )
                else:
                    self.info(
                        f"{operation} completed in {duration:.1f}s",
                        extra={"operation_complete": True, "duration": duration},
                    )

    def _validate_context(self, context: dict[str, Any]) -> None:
        """Validate context dictionary for consistency and safety."""
        if not isinstance(context, dict):
            raise ValueError("Context must be a dictionary")

        # Check for reasonable size limits
        if len(context) > 50:
            raise ValueError("Context dictionary too large (>50 keys)")

        # Validate key types and values
        for key, value in context.items():
            if not isinstance(key, str):
                raise ValueError(f"Context keys must be strings, got {type(key)}")

            # Validate value types - prevent complex objects that might cause issues
            if value is not None:
                value_type = type(value)
                if not (
                    value_type in (str, int, float, bool)
                    or (hasattr(value, "__dict__") and len(vars(value)) == 0)
                ):  # Empty objects
                    # For complex objects, ensure they're serializable
                    try:
                        str(value)
                    except Exception as e:
                        raise ValueError(
                            f"Context value for key '{key}' is not safely serializable"
                        ) from e

    def file_operation_start(self, file_path: str | Path, operation: str):
        """Log start of file operation with minimal formatting."""
        file_name = Path(file_path).name
        msg = (
            f"{operation} {file_name}"
            if LoggerManager.log_mode == LogMode.CLASSIC
            else f"{operation} {file_name}"
        )
        self.info(
            msg,
            extra={
                "file_operation": True,
                "file_path": str(file_path),
                "operation": operation,
                "phase": "start",
            },
        )

    def file_operation_complete(
        self,
        file_path: str | Path,
        operation: str,
        duration: float,
        success: bool = True,
        **metrics,
    ):
        """Log completion of file operation with clean formatting."""
        file_name = Path(file_path).name
        if LoggerManager.log_mode == LogMode.CLASSIC:
            message = f"{operation} {file_name} {duration:.1f}s"
        else:
            message = f"{operation} {file_name} {duration:.1f}s"

        # Add essential metrics only
        if metrics:
            if "tests_generated" in metrics and metrics["tests_generated"] > 0:
                message += f" ({metrics['tests_generated']} tests)"

        log_level = self.info if success else self.warning
        log_level(
            message,
            extra={
                "file_operation": True,
                "file_path": str(file_path),
                "operation": operation,
                "phase": "complete",
                "duration": duration,
                "success": success,
                **metrics,
            },
        )

    def batch_operation_start(
        self, operation: str, total_items: int, batch_size: int = 1
    ):
        """Log start of batch operation."""
        self.info(
            f"{operation} processing {total_items} items",
            extra={
                "batch_operation": True,
                "operation": operation,
                "total_items": total_items,
                "batch_size": batch_size,
                "phase": "start",
            },
        )

    def batch_progress(
        self,
        operation: str,
        completed: int,
        total: int,
        current_item: str | None = None,
    ):
        """Log batch operation progress with minimal display."""
        percentage = (completed / total * 100) if total > 0 else 0

        message = f"{operation} {completed}/{total} ({percentage:.0f}%)"

        if current_item:
            current_name = (
                Path(current_item).name
                if isinstance(current_item, str | Path)
                else str(current_item)
            )
            message += f" {current_name}"

        self.info(
            message,
            extra={
                "batch_progress": True,
                "operation": operation,
                "completed": completed,
                "total": total,
                "percentage": percentage,
                "current_item": str(current_item) if current_item else None,
            },
        )

    def error_with_context(
        self,
        message: str,
        error: Exception,
        suggestions: list[str] | None = None,
        **context,
    ):
        """Log error with rich context and suggestions."""
        # Use minimal template: error component: message • k1=v1 • k2=v2
        error_message = (
            f"ERROR: {message}: {escape(str(error))}"
            if LoggerManager.log_mode == LogMode.CLASSIC
            else f"{message}: {escape(str(error))}"
        )

        if suggestions:
            error_message += "\nSuggestions:"
            for suggestion in suggestions:
                error_message += f"\n  {escape(suggestion)}"

        self.error(
            error_message,
            extra={
                "error_with_context": True,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "suggestions": suggestions or [],
                **context,
            },
            exc_info=True,
        )

    def performance_summary(self, operation: str, metrics: dict[str, Any]):
        """Log performance summary with metrics."""
        # Use minimal template: operation: summary • duration=Xs • rate=Y • success=Z%
        summary_parts = [
            f"{operation}: performance"
            if LoggerManager.log_mode == LogMode.CLASSIC
            else f"{operation}: performance"
        ]

        # Format key metrics in minimal style
        metric_parts = []
        if "duration" in metrics:
            metric_parts.append(f"duration={metrics['duration']:.1f}s")

        if "items_processed" in metrics:
            items = metrics["items_processed"]
            duration = metrics.get("duration", 1)
            rate = items / duration
            metric_parts.append(f"rate={rate:.1f}/s")

        if "success_rate" in metrics:
            rate = metrics["success_rate"]
            metric_parts.append(f"success={rate:.0%}")

        if "memory_usage" in metrics:
            memory_mb = metrics["memory_usage"] / 1024 / 1024
            metric_parts.append(f"memory={memory_mb:.1f}MB")

        if metric_parts:
            summary_parts[0] += " • " + " • ".join(metric_parts)

        self.info(
            summary_parts[0],
            extra={"performance_summary": True, "operation": operation, **metrics},
        )

    def debug_context(self, message: str, **context):
        """Log debug message with context."""
        if self.logger.isEnabledFor(logging.DEBUG):
            # Use minimal template: debug component: message • k1=v1 • k2=v2
            context_str = " • ".join(f"{k}={v}" for k, v in context.items())
            if LoggerManager.log_mode == LogMode.CLASSIC:
                if context_str:
                    message_text = f"DEBUG: {message} • {context_str}"
                else:
                    message_text = f"DEBUG: {message}"
            else:
                message_text = (
                    f"{message}{(' • ' + context_str) if context_str else ''}"
                )

            self.debug(message_text, extra={"debug_context": True, **context})

    # Standard logging methods with rich formatting
    def info(self, message: str, **kwargs):
        """Log info message with rich formatting."""
        self.logger.info(LoggerManager._prepare_message(message), **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message with rich formatting."""
        self.logger.debug(LoggerManager._prepare_message(message), **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with rich formatting."""
        self.logger.warning(LoggerManager._prepare_message(message), **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with rich formatting."""
        self.logger.error(LoggerManager._prepare_message(message), **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message with rich formatting."""
        self.logger.critical(LoggerManager._prepare_message(message), **kwargs)


class LoggerManager:
    """Manager for creating and configuring structured loggers."""

    # Use a regular dict to avoid WeakValueDictionary evictions under test runners
    _loggers: dict[str, StructuredLogger] = {}
    _console: Console | None = None
    log_mode: LogMode = LogMode.CLASSIC
    output_format: OutputFormat = OutputFormat.CONSOLE
    _explicit_level: int | None = None
    _setup_complete: bool = False
    _setup_lock: threading.Lock = threading.Lock()

    # Async logging support
    _async_enabled: bool = False
    _log_queue: queue.Queue | None = None
    _background_thread: threading.Thread | None = None
    _shutdown_event: threading.Event | None = None

    # CSV logging support
    _csv_file: Any | None = None
    _csv_writer: Any | None = None

    @classmethod
    def setup_global_logging(
        cls, console: Console | None = None, level: int = logging.INFO
    ):
        """Set up global logging configuration with thread safety."""
        with cls._setup_lock:
            # Atomic check-and-set pattern
            if cls._setup_complete:
                return

            cls._console = console or Console(theme=TESTCRAFT_THEME)

            # Configure root logger
            root_logger = logging.getLogger()

            # Guard: if root already has our RichHandler, return early to be idempotent
            existing_rich_handler = None
            for handler in root_logger.handlers:
                if (
                    isinstance(handler, RichHandler)
                    and getattr(handler, "console", None) is cls._console
                ):
                    existing_rich_handler = handler
                    break

            if existing_rich_handler:
                # Already configured with our console, just ensure correct level
                root_logger.setLevel(level)
                cls._explicit_level = level
                cls._setup_complete = True
                return

            # Remove any existing RichHandlers that aren't ours, keep other handlers
            handlers_to_remove = []
            for handler in root_logger.handlers:
                if isinstance(handler, RichHandler):
                    handlers_to_remove.append(handler)

            for handler in handlers_to_remove:
                root_logger.removeHandler(handler)

            # Add our rich handler
            rich_handler = RichHandler(
                console=cls._console,
                show_time=True,
                show_path=False,
                markup=False,  # We sanitize ourselves for minimal; classic logs avoid inline markup too
                rich_tracebacks=True,
            )

            rich_handler.setFormatter(logging.Formatter(fmt="%(message)s"))

            root_logger.addHandler(rich_handler)
            root_logger.setLevel(level)
            cls._explicit_level = level
            cls._setup_complete = True

    @classmethod
    def enable_async_logging(cls, queue_size: int = 1000) -> None:
        """Enable asynchronous logging with background thread processing."""
        if cls._async_enabled:
            return  # Already enabled

        cls._async_enabled = True
        cls._log_queue = queue.Queue(maxsize=queue_size)
        cls._shutdown_event = threading.Event()

        # Start background thread
        cls._background_thread = threading.Thread(
            target=cls._async_log_worker, daemon=True, name="AsyncLogger"
        )
        cls._background_thread.start()

    @classmethod
    def disable_async_logging(cls) -> None:
        """Disable asynchronous logging and clean up resources."""
        if not cls._async_enabled:
            return

        cls._async_enabled = False

        # Signal shutdown
        if cls._shutdown_event:
            cls._shutdown_event.set()

        # Wait for background thread to finish
        if cls._background_thread and cls._background_thread.is_alive():
            cls._background_thread.join(timeout=5.0)

        # Clean up
        cls._log_queue = None
        cls._background_thread = None
        cls._shutdown_event = None

    @classmethod
    def _async_log_worker(cls) -> None:
        """Background worker thread for async logging."""
        while not cls._shutdown_event or not cls._shutdown_event.is_set():
            try:
                # Get log record with timeout
                record = cls._log_queue.get(timeout=1.0) if cls._log_queue else None
                if not record:
                    continue
                cls._process_structured_log(record)
                if cls._log_queue:
                    cls._log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Log errors but don't crash the worker
                print(f"Async logging error: {e}")

    @classmethod
    def _process_structured_log(cls, record: logging.LogRecord) -> None:
        """Process a log record in structured format."""
        if cls.output_format == OutputFormat.JSON:
            cls._write_json_log(record)
        elif cls.output_format == OutputFormat.CSV:
            cls._write_csv_log(record)
        # CONSOLE format handled by normal logging

    @classmethod
    def _write_json_log(cls, record: logging.LogRecord) -> None:
        """Write log record in JSON format."""
        try:
            log_data = {
                "timestamp": time.time(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

            # Add extra fields if present
            if hasattr(record, "__dict__"):
                for key, value in record.__dict__.items():
                    if key not in (
                        "name",
                        "msg",
                        "args",
                        "levelname",
                        "levelno",
                        "pathname",
                        "filename",
                        "module",
                        "funcName",
                        "lineno",
                        "created",
                        "msecs",
                        "relativeCreated",
                        "thread",
                        "threadName",
                        "processName",
                        "process",
                        "getMessage",
                        "exc_info",
                        "exc_text",
                        "stack_info",
                    ):
                        log_data[key] = value

            print(json.dumps(log_data, default=str))

        except Exception as e:
            # Fallback to console if JSON serialization fails
            print(f"JSON logging failed: {e}")

    @classmethod
    def _write_csv_log(cls, record: logging.LogRecord) -> None:
        """Write log record in CSV format."""
        if not cls._csv_writer:
            cls._setup_csv_logging()

        try:
            # Prepare CSV row
            row = [
                str(time.time()),
                record.levelname,
                record.name,
                record.getMessage(),
                record.module or "",
                record.funcName or "",
                str(record.lineno or ""),
            ]

            if cls._csv_writer:
                cls._csv_writer.writerow(row)
            if cls._csv_file:
                cls._csv_file.flush()

        except Exception as e:
            # Fallback to console if CSV writing fails
            print(f"CSV logging failed: {e}")

    @classmethod
    def _setup_csv_logging(cls) -> None:
        """Set up CSV logging with headers."""
        try:
            cls._csv_file = open("testcraft.log.csv", "a", newline="", encoding="utf-8")
            cls._csv_writer = csv.writer(cls._csv_file)
            # Write header if file is empty
            if cls._csv_file.tell() == 0:
                cls._csv_writer.writerow(
                    [
                        "timestamp",
                        "level",
                        "logger",
                        "message",
                        "module",
                        "function",
                        "line",
                    ]
                )
        except Exception as e:
            print(f"CSV setup failed: {e}")

    @classmethod
    def set_output_format(cls, output_format: OutputFormat) -> None:
        """Set the structured output format."""
        cls.output_format = output_format

        if output_format == OutputFormat.CSV:
            cls._setup_csv_logging()

    @classmethod
    def set_log_mode(
        cls, mode: LogMode, verbose: bool = False, quiet: bool = False
    ) -> None:
        """Configure log mode and root level appropriately."""
        cls.log_mode = mode
        # Determine level precedence: quiet > verbose > default
        if quiet or os.getenv("TESTCRAFT_QUIET", "").lower() in {"1", "true", "yes"}:
            level = logging.WARNING
        elif verbose:
            level = logging.DEBUG
        else:
            # Minimal defaults to WARNING, classic to INFO
            level = logging.WARNING if mode == LogMode.MINIMAL else logging.INFO

        # Apply to root if not explicitly overridden later
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        cls._explicit_level = level

    @staticmethod
    def _strip_rich_tags(message: str) -> str:
        """Remove Rich markup tags like [primary]...[/] from a message."""
        if not message or not isinstance(message, str):
            return str(message) if message is not None else ""

        try:
            # Remove [tag] and [/tag] patterns conservatively
            return re.sub(r"\[/?.*?\]", "", message)
        except (re.error, TypeError) as e:
            # Log the error and return original message if regex fails
            logger = logging.getLogger("testcraft.sanitization")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Rich tag stripping failed: {e}, returning original message"
                )
            return str(message)

    @staticmethod
    def _strip_emojis(message: str) -> str:
        """Remove emojis and symbols from a message for minimal logs using proper Unicode categories."""
        if not message:
            return message

        try:
            # Define specific emoji and symbol ranges to remove
            emoji_ranges = [
                # Emoticons
                (0x1F600, 0x1F64F),  # Emoticons
                (0x1F300, 0x1F5FF),  # Miscellaneous Symbols and Pictographs
                (0x1F680, 0x1F6FF),  # Transport and Map Symbols
                (0x1F1E0, 0x1F1FF),  # Regional Indicator Symbols
                (0x2600, 0x26FF),  # Miscellaneous Symbols
                (0x2700, 0x27BF),  # Dingbats
                (0x1F926, 0x1F937),  # Gestures
                (0x10000, 0x10FFFF),  # Supplementary Private Use Area (for some emoji)
            ]

            result = []
            for char in message:
                code_point = ord(char)

                # Check if character is in emoji/symbol ranges
                should_remove = False
                for start, end in emoji_ranges:
                    if start <= code_point <= end:
                        should_remove = True
                        break

                # Also remove characters in symbol categories
                if not should_remove:
                    try:
                        category = unicodedata.category(char)
                        # Remove symbols, marks, and other special characters
                        if category in (
                            "So",
                            "Sm",
                            "Sc",
                        ):  # Symbol, Math Symbol, Currency Symbol
                            should_remove = True
                    except (TypeError, ValueError):
                        # If unicodedata fails, fall back to checking code point
                        if 0x2000 <= code_point <= 0x206F:  # General Punctuation
                            should_remove = True

                if not should_remove:
                    result.append(char)

            return "".join(result)

        except Exception:
            # Fallback: remove known problematic characters
            problematic_chars = [
                "✅",
                "⚠️",
                "💥",
                "🔍",
                "🎯",
                "🎊",
                "🧪",
                "📊",
                "❌",
                "🏆",
                "🟢",
                "🟡",
                "🟠",
                "🔴",
                "⏭️",
                "❓",
                "🔥",
                "💯",
                "✨",
                "🎉",
            ]
            for char in problematic_chars:
                message = message.replace(char, "")
            return message

    @classmethod
    def _prepare_message(cls, message: str) -> str:
        """Prepare message for logging based on current log mode (sanitize)."""
        if message is None:
            return ""

        # Ensure we have a string
        if not isinstance(message, str):
            try:
                message = str(message)
            except Exception:
                # If conversion fails, return a safe default
                return "[unprintable message]"

        if cls.log_mode == LogMode.MINIMAL:
            try:
                # Strip tags and emojis; collapse whitespace
                msg = cls._strip_rich_tags(message)
                msg = cls._strip_emojis(msg)
                msg = re.sub(r"\s+", " ", msg).strip()
                return msg
            except Exception as e:
                # Log error and return original message if processing fails
                logger = logging.getLogger("testcraft.sanitization")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Message preparation failed: {e}, returning original message"
                    )
                return message
        return message

    @classmethod
    def get_logger(cls, name: str) -> StructuredLogger:
        """Get or create a structured logger (robust against WeakValueDict races)."""
        # Avoid WeakValueDictionary race by using setdefault pattern
        logger_ref = cls._loggers.setdefault(name, StructuredLogger(name, cls._console))
        try:
            structured_logger = logger_ref
        except KeyError:
            structured_logger = cls._loggers.setdefault(name, StructuredLogger(name, cls._console))

        # Enforce no per-logger handlers and propagation to root
        structured_logger.logger.handlers = []
        structured_logger.logger.propagate = True

        return structured_logger

    @classmethod
    def get_file_logger(cls, file_path: str | Path) -> StructuredLogger:
        """Get logger named after file path."""
        if isinstance(file_path, Path):
            name = f"testcraft.{file_path.stem}"
        else:
            name = f"testcraft.{Path(file_path).stem}"
        return cls.get_logger(name)

    @classmethod
    def get_operation_logger(cls, operation: str) -> StructuredLogger:
        """Get logger for specific operation."""
        return cls.get_logger(f"testcraft.{operation}")

    @classmethod
    def cleanup_unused_loggers(cls) -> int:
        """Clean up unused loggers to prevent memory leaks. Returns count of removed loggers."""
        removed_count = 0
        keys_to_remove: list[str] = []

        # Find loggers that are no longer referenced
        for key, logger_ref in cls._loggers.items():
            if logger_ref is None:  # Weak reference is dead
                keys_to_remove.append(key)
                removed_count += 1

        # Remove dead references
        for key in keys_to_remove:
            cls._loggers.pop(key, None)

        return removed_count

    @classmethod
    def cleanup_resources(cls) -> None:
        """Clean up all logging resources and background threads."""
        # Clean up async logging
        cls.disable_async_logging()

        # Close CSV file if open
        if cls._csv_file:
            try:
                cls._csv_file.close()
            except Exception:
                pass
            cls._csv_file = None
            cls._csv_writer = None

        # Clean up unused loggers
        cls.cleanup_unused_loggers()

        # Reset state
        cls._async_enabled = False
        cls.output_format = OutputFormat.CONSOLE


# Convenience functions for common use cases
def setup_enhanced_logging(console: Console | None = None, level: int = logging.INFO):
    """Set up enhanced logging system."""
    LoggerManager.setup_global_logging(console, level)
    # Ensure primary logger exists even under Click testing environments
    logging.getLogger("testcraft.main").propagate = True
    # Use setdefault to avoid WeakValueDict KeyError
    return LoggerManager.get_logger("testcraft.main")


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger by name."""
    return LoggerManager.get_logger(name)


def get_file_logger(file_path: str | Path) -> StructuredLogger:
    """Get logger for file operations."""
    return LoggerManager.get_file_logger(file_path)


def get_operation_logger(operation: str) -> StructuredLogger:
    """Get logger for specific operations."""
    return LoggerManager.get_operation_logger(operation)
