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

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
from enum import Enum
import os
import re

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.text import Text

from .rich_cli import TESTCRAFT_THEME


class LogMode(str, Enum):
    """Logging output modes."""
    CLASSIC = "classic"
    MINIMAL = "minimal"


class StructuredLogger:
    """Enhanced logger with rich formatting and structured messages."""
    
    def __init__(self, name: str, console: Optional[Console] = None):
        """Initialize structured logger with Rich formatting."""
        self.name = name
        self.console = console or Console(theme=TESTCRAFT_THEME)
        self.logger = logging.getLogger(name)
        self._setup_rich_handler()
        
        # Context tracking
        self._context_stack: List[Dict[str, Any]] = []
        self._operation_context: Dict[str, Any] = {}
        
    def _setup_rich_handler(self):
        """Set up rich handler for beautiful log output."""
        # Check if root logger has RichHandler - if so, rely on root
        root_logger = logging.getLogger()
        has_root_rich_handler = any(isinstance(h, RichHandler) for h in root_logger.handlers)
        
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
        """Context manager for operation-specific logging."""
        start_time = time.time()
        operation_id = f"{operation}_{int(start_time)}"
        
        # Push context
        full_context = {
            "operation": operation,
            "operation_id": operation_id,
            "start_time": start_time,
            **context
        }
        self._context_stack.append(full_context)
        self._operation_context = full_context
        
        # Log operation start
        if LoggerManager.log_mode == LogMode.CLASSIC:
            self.info(
                f"[primary]starting[/] [accent]{operation}[/]",
                extra={"operation_start": True, **context}
            )
        else:
            # Minimal one-liner (will likely be suppressed unless verbose)
            self.info(
                f"starting {operation}",
                extra={"operation_start": True, **context}
            )
        
        try:
            yield self
        except Exception as e:
            # Log operation failure
            duration = time.time() - start_time
            self.error(
                (
                    f"{operation} failed after {duration:.2f}s: {escape(str(e))}"
                    if LoggerManager.log_mode == LogMode.MINIMAL
                    else f"[error]{operation}[/] failed after {duration:.2f}s: {escape(str(e))}"
                ),
                extra={"operation_failed": True, "duration": duration, "error": str(e)}
            )
            raise
        finally:
            # Pop context and log completion
            if self._context_stack:
                self._context_stack.pop()
            self._operation_context = self._context_stack[-1] if self._context_stack else {}
            
            duration = time.time() - start_time
            if LoggerManager.log_mode == LogMode.CLASSIC:
                self.info(
                    f"[success]{operation}[/] completed in {duration:.1f}s",
                    extra={"operation_complete": True, "duration": duration}
                )
            else:
                self.info(
                    f"{operation} completed in {duration:.1f}s",
                    extra={"operation_complete": True, "duration": duration}
                )
    
    def file_operation_start(self, file_path: Union[str, Path], operation: str):
        """Log start of file operation with minimal formatting."""
        file_name = Path(file_path).name
        msg = (
            f"[accent]{operation}[/] [primary]{file_name}[/]"
            if LoggerManager.log_mode == LogMode.CLASSIC
            else f"{operation} {file_name}"
        )
        self.info(
            msg,
            extra={
                "file_operation": True,
                "file_path": str(file_path),
                "operation": operation,
                "phase": "start"
            }
        )
    
    def file_operation_complete(self, file_path: Union[str, Path], operation: str, 
                             duration: float, success: bool = True, **metrics):
        """Log completion of file operation with clean formatting."""
        file_name = Path(file_path).name
        status_color = "success" if success else "error"
        if LoggerManager.log_mode == LogMode.CLASSIC:
            message = f"[{status_color}]{operation}[/] [primary]{file_name}[/] {duration:.1f}s"
        else:
            message = f"{operation} {file_name} {duration:.1f}s"
        
        # Add essential metrics only
        if metrics:
            if "tests_generated" in metrics and metrics["tests_generated"] > 0:
                message += f" [muted]({metrics['tests_generated']} tests)[/]"
        
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
                **metrics
            }
        )
    
    def batch_operation_start(self, operation: str, total_items: int, batch_size: int = 1):
        """Log start of batch operation."""
        self.info(
            f"[primary]{operation}[/] processing {total_items} items",
            extra={
                "batch_operation": True,
                "operation": operation,
                "total_items": total_items,
                "batch_size": batch_size,
                "phase": "start"
            }
        )
    
    def batch_progress(self, operation: str, completed: int, total: int, 
                      current_item: Optional[str] = None):
        """Log batch operation progress with minimal display."""
        percentage = (completed / total * 100) if total > 0 else 0
        
        message = f"[accent]{operation}[/] {completed}/{total} ({percentage:.0f}%)"
        
        if current_item:
            current_name = Path(current_item).name if isinstance(current_item, (str, Path)) else str(current_item)
            message += f" [muted]{current_name}[/]"
        
        self.info(
            message,
            extra={
                "batch_progress": True,
                "operation": operation,
                "completed": completed,
                "total": total,
                "percentage": percentage,
                "current_item": str(current_item) if current_item else None
            }
        )
    
    def error_with_context(self, message: str, error: Exception, 
                          suggestions: Optional[List[str]] = None,
                          **context):
        """Log error with rich context and suggestions."""
        # Use minimal template: [error] component: message • k1=v1 • k2=v2
        error_message = (
            f"[error] {message}: {escape(str(error))}[/]"
            if LoggerManager.log_mode == LogMode.CLASSIC
            else f"{message}: {escape(str(error))}"
        )
        
        if suggestions:
            error_message += "\n[warning]suggestions:[/]"
            for suggestion in suggestions:
                error_message += f"\n  {escape(suggestion)}"
        
        self.error(
            error_message,
            extra={
                "error_with_context": True,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "suggestions": suggestions or [],
                **context
            },
            exc_info=True
        )
    
    def performance_summary(self, operation: str, metrics: Dict[str, Any]):
        """Log performance summary with metrics."""
        # Use minimal template: [info] operation: summary • duration=Xs • rate=Y • success=Z%
        summary_parts = [
            f"[info] {operation}: performance"
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
            extra={
                "performance_summary": True,
                "operation": operation,
                **metrics
            }
        )
    
    def debug_context(self, message: str, **context):
        """Log debug message with context."""
        if self.logger.isEnabledFor(logging.DEBUG):
            # Use minimal template: [debug] component: message • k1=v1 • k2=v2
            context_str = " • ".join(f"{k}={v}" for k, v in context.items())
            if LoggerManager.log_mode == LogMode.CLASSIC:
                if context_str:
                    message_text = f"[debug] {message} • {context_str}"
                else:
                    message_text = f"[debug] {message}"
            else:
                message_text = f"{message}{(' • ' + context_str) if context_str else ''}"
            
            self.debug(
                message_text,
                extra={"debug_context": True, **context}
            )
    
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
    
    _loggers: Dict[str, StructuredLogger] = {}
    _console: Optional[Console] = None
    log_mode: LogMode = LogMode.CLASSIC
    _explicit_level: Optional[int] = None
    
    @classmethod
    def setup_global_logging(cls, console: Optional[Console] = None, level: int = logging.INFO):
        """Set up global logging configuration."""
        cls._console = console or Console(theme=TESTCRAFT_THEME)
        
        # Configure root logger
        root_logger = logging.getLogger()
        
        # Guard: if root already has our RichHandler, return early to be idempotent
        existing_rich_handler = None
        for handler in root_logger.handlers:
            if isinstance(handler, RichHandler) and getattr(handler, 'console', None) is cls._console:
                existing_rich_handler = handler
                break
        
        if existing_rich_handler:
            # Already configured with our console, just ensure correct level
            root_logger.setLevel(level)
            cls._explicit_level = level
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

    @classmethod
    def set_log_mode(cls, mode: LogMode, verbose: bool = False, quiet: bool = False) -> None:
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
        # Remove [tag] and [/tag] patterns conservatively
        return re.sub(r"\[/?.*?\]", "", message)

    @staticmethod
    def _strip_emojis(message: str) -> str:
        """Remove common emojis from a message for minimal logs."""
        # Basic removal of non-ASCII symbol-ish chars and a small allowlist
        # Keep punctuation and basic unicode, drop surrogate-pair emojis
        try:
            # Remove characters in a common emoji range
            return re.sub(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]", "", message)
        except re.error:
            # Narrow fallback for environments without wide Unicode
            for ch in ("✅","⚠️","💥","🔍","🎯","🎊","🧪","📊","❌","🏆","🟢","🟡","🟠","🔴","⏭️","❓"):
                message = message.replace(ch, "")
            return message

    @classmethod
    def _prepare_message(cls, message: str) -> str:
        """Prepare message for logging based on current log mode (sanitize)."""
        if cls.log_mode == LogMode.MINIMAL:
            # Strip tags and emojis; collapse whitespace
            msg = cls._strip_rich_tags(str(message))
            msg = cls._strip_emojis(msg)
            msg = re.sub(r"\s+", " ", msg).strip()
            return msg
        return str(message)
    
    @classmethod
    def get_logger(cls, name: str) -> StructuredLogger:
        """Get or create a structured logger."""
        if name not in cls._loggers:
            cls._loggers[name] = StructuredLogger(name, cls._console)
        
        # Enforce no per-logger handlers and propagation to root
        structured_logger = cls._loggers[name]
        structured_logger.logger.handlers = []
        structured_logger.logger.propagate = True
        
        return structured_logger
    
    @classmethod
    def get_file_logger(cls, file_path: Union[str, Path]) -> StructuredLogger:
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


# Convenience functions for common use cases
def setup_enhanced_logging(console: Optional[Console] = None, level: int = logging.INFO):
    """Set up enhanced logging system."""
    LoggerManager.setup_global_logging(console, level)
    return LoggerManager.get_logger("testcraft.main")


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger by name."""
    return LoggerManager.get_logger(name)


def get_file_logger(file_path: Union[str, Path]) -> StructuredLogger:
    """Get logger for file operations."""
    return LoggerManager.get_file_logger(file_path)


def get_operation_logger(operation: str) -> StructuredLogger:
    """Get logger for specific operations."""
    return LoggerManager.get_operation_logger(operation)
