"""
Textual UI adapter implementing UIPort interface.

Provides a bridge between TestCraft's UI port interface and Textual
components, enabling both transient screen usage for CLI commands
and full TUI integration.
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich.console import Console
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, Switch

from ...adapters.io.ui_rich import UIError
from .app import TestCraftTextualApp
from .events import (
    ErrorOccurred,
    ProgressUpdated,
    StatsUpdated,
)
from .widgets import FileTable, Notifications, StatsPanel


class TextualUIAdapter:
    """
    Textual UI adapter implementing UIPort interface.

    Provides both transient screen capabilities for CLI commands
    and integration with the main TestCraftTextualApp for full TUI mode.
    """

    def __init__(
        self,
        console: Console | None = None,
        app: TestCraftTextualApp | None = None,
        transient: bool = True,
    ):
        """
        Initialize the Textual UI adapter.

        Args:
            console: Rich console instance for fallback operations
            app: Existing TestCraftTextualApp instance (for full TUI mode)
            transient: If True, creates ephemeral screens for operations
        """
        self.console = console or Console()
        self._app = app
        self._transient = transient
        self._current_screen: ModalScreen | None = None

        # Operation tracking
        self._operation_start_times: dict[str, float] = {}
        self._metrics: dict[str, Any] = {}

        # Logger
        self.logger = logging.getLogger("testcraft.textual_ui")

    # ========================================================================
    # UIPort Interface Implementation
    # ========================================================================

    def display_progress(
        self,
        progress_data: dict[str, Any],
        progress_type: str = "general",
        **kwargs: Any,
    ) -> None:
        """Display progress information using Textual components."""
        try:
            current = progress_data.get("current", 0)
            total = progress_data.get("total", 100)
            message = progress_data.get("message", "Processing...")

            if self._app and not self._transient:
                # Update existing app progress
                self._app.post_message(ProgressUpdated(current, total, message))
            else:
                # Use console fallback for transient operations
                percentage = (current / total) * 100 if total > 0 else 0
                self.console.print(f"[blue]{message}[/] {percentage:.1f}%")

        except Exception as e:
            raise UIError(f"Failed to display progress: {e}") from e

    def display_results(
        self, results: dict[str, Any], result_type: str = "general", **kwargs: Any
    ) -> None:
        """Display results using Textual components."""
        try:
            if self._app and not self._transient:
                # Send results to the app
                self._app.post_message(
                    StatsUpdated(
                        {
                            "operation": result_type,
                            "results": results,
                        }
                    )
                )
            else:
                # Console fallback
                summary = results.get("summary", "Operation completed")
                success = results.get("success", True)

                if success:
                    self.console.print(f"[green]✓[/] {summary}")
                else:
                    self.console.print(f"[red]✗[/] {summary}")

        except Exception as e:
            raise UIError(f"Failed to display results: {e}") from e

    def display_error(
        self, error_message: str, error_type: str = "general", **kwargs: Any
    ) -> None:
        """Display error information."""
        try:
            if self._app and not self._transient:
                self._app.post_message(
                    ErrorOccurred(error_message, kwargs.get("details"))
                )
            else:
                self.console.print(f"[red]Error:[/] {error_message}")

        except Exception as e:
            raise UIError(f"Failed to display error: {e}") from e

    def display_warning(
        self, warning_message: str, warning_type: str = "general", **kwargs: Any
    ) -> None:
        """Display warning information."""
        try:
            if self._app and not self._transient:
                # Use notifications system
                if hasattr(self._app.screen, "query_one"):
                    try:
                        notifications = self._app.screen.query_one(Notifications)
                        notifications.warning(warning_message)
                    except Exception:
                        pass

            # Always show in console as fallback
            self.console.print(f"[yellow]Warning:[/] {warning_message}")

        except Exception as e:
            raise UIError(f"Failed to display warning: {e}") from e

    def display_info(
        self, info_message: str, info_type: str = "general", **kwargs: Any
    ) -> None:
        """Display informational message."""
        try:
            if self._app and not self._transient:
                # Use notifications system
                if hasattr(self._app.screen, "query_one"):
                    try:
                        notifications = self._app.screen.query_one(Notifications)
                        notifications.info(info_message)
                    except Exception:
                        pass

            # Console fallback
            self.console.print(f"[blue]Info:[/] {info_message}")

        except Exception as e:
            raise UIError(f"Failed to display info: {e}") from e

    def get_user_input(
        self, prompt: str, input_type: str = "string", **kwargs: Any
    ) -> Any:
        """Get input from the user."""
        try:
            if self._app and not self._transient:
                # Create modal input dialog
                result = self._show_input_dialog(prompt, input_type, **kwargs)
                return result
            else:
                # Console fallback
                from rich.prompt import Prompt

                if input_type == "boolean":
                    from rich.prompt import Confirm

                    return Confirm.ask(prompt, default=kwargs.get("default", False))
                else:
                    return Prompt.ask(prompt, default=kwargs.get("default", ""))

        except Exception as e:
            raise UIError(f"Failed to get user input: {e}") from e

    def confirm_action(
        self, message: str, default: bool = False, **kwargs: Any
    ) -> bool:
        """Get confirmation from the user."""
        try:
            if self._app and not self._transient:
                # Create confirmation dialog
                return self._show_confirmation_dialog(message, default, **kwargs)
            else:
                # Console fallback
                from rich.prompt import Confirm

                return Confirm.ask(message, default=default)

        except Exception as e:
            raise UIError(f"Failed to get confirmation: {e}") from e

    # ========================================================================
    # Enhanced API Methods (Parity with EnhancedUIAdapter)
    # ========================================================================

    @contextmanager
    def create_operation_tracker(self, operation_name: str, total_steps: int = 1):
        """Context manager for tracking multi-step operations."""
        self._operation_start_times[operation_name] = time.time()

        if self._app and not self._transient:
            # Use app's built-in operation tracking
            self._app.start_operation(operation_name, f"Starting {operation_name}...")

        class OperationTracker:
            def __init__(self, ui_adapter):
                self.ui = ui_adapter
                self.current_step = 0
                self.total_steps = total_steps
                self.operation_name = operation_name

            def advance_step(self, description: str = "", increment: int = 1):
                """Advance to the next step with optional description."""
                self.current_step += increment

                if self.ui._app and not self.ui._transient:
                    progress = (self.current_step / self.total_steps) * 100
                    self.ui._app.post_message(
                        ProgressUpdated(
                            self.current_step,
                            self.total_steps,
                            description or f"Step {self.current_step}",
                        )
                    )
                else:
                    # Console fallback
                    progress = (self.current_step / self.total_steps) * 100
                    self.ui.console.print(f"[blue]{description}[/] ({progress:.0f}%)")

            def update_description(self, description: str):
                """Update the current step description."""
                self.advance_step(description, 0)  # No increment, just update

            def log_progress(self, message: str, level: str = "info"):
                """Log progress message."""
                log_func = getattr(self.ui.logger, level, self.ui.logger.info)
                log_func(f"{self.operation_name}: {message}")

        tracker = OperationTracker(self)

        try:
            yield tracker

            # Complete operation
            if self._app and not self._transient:
                duration = time.time() - self._operation_start_times[operation_name]
                self._app.complete_operation(
                    operation_name,
                    success=True,
                    message=f"Completed in {duration:.1f}s",
                )

            # Store metrics
            duration = time.time() - self._operation_start_times[operation_name]
            self._metrics[operation_name] = {
                "duration": duration,
                "steps": total_steps,
                "avg_step_time": duration / max(total_steps, 1),
            }

        except Exception as e:
            # Mark operation as failed
            if self._app and not self._transient:
                self._app.complete_operation(
                    operation_name, success=False, message=f"Failed: {e}"
                )
            raise
        finally:
            # Clean up
            if operation_name in self._operation_start_times:
                del self._operation_start_times[operation_name]

    def create_status_spinner(self, message: str):
        """Create a status spinner for operations."""

        @contextmanager
        def spinner_context():
            if self._app and not self._transient:
                self._app.start_operation("status", message)
                try:
                    yield
                finally:
                    self._app.complete_operation("status")
            else:
                # Console fallback
                from rich.status import Status

                with Status(message, console=self.console):
                    yield

        return spinner_context()

    def display_file_progress_table(
        self, files_data: list[dict[str, Any]], title: str = "File Processing Status"
    ):
        """Display file processing progress table."""
        try:
            if self._app and not self._transient:
                # Update file table in the current screen
                try:
                    current_screen = self._app.screen
                    file_table = current_screen.query_one(FileTable)

                    # Update each file's status
                    for file_data in files_data:
                        file_table.update_file_status(
                            file_data.get("file_path", ""),
                            file_data.get("status", "pending"),
                            file_data.get("progress", 0.0),
                            file_data.get("tests_generated", 0),
                            file_data.get("duration", 0.0),
                            file_data.get("error"),
                        )
                except Exception:
                    # Fallback to console if no file table available
                    self._display_console_file_table(files_data, title)
            else:
                # Console mode
                self._display_console_file_table(files_data, title)

        except Exception as e:
            self.logger.error(f"Failed to display file progress table: {e}")

    def _display_console_file_table(self, files_data: list[dict[str, Any]], title: str):
        """Display file table in console as fallback."""
        from rich.table import Table

        table = Table(title=title, show_header=True, header_style="bold blue")
        table.add_column("File", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Progress", justify="center")
        table.add_column("Tests", justify="center")
        table.add_column("Duration", justify="center")

        for file_data in files_data:
            file_path = str(file_data.get("file_path", ""))
            file_name = Path(file_path).name if file_path else "Unknown"

            status = file_data.get("status", "pending")
            progress = file_data.get("progress", 0.0)
            tests = file_data.get("tests_generated", 0)
            duration = file_data.get("duration", 0.0)

            # Format status with color
            status_colors = {
                "completed": "green",
                "done": "green",
                "failed": "red",
                "running": "yellow",
                "pending": "dim",
            }
            status_color = status_colors.get(status, "white")
            status_display = f"[{status_color}]{status}[/]"

            # Format progress
            progress_display = f"{progress:.1f}%" if progress > 0 else "—"

            # Format other columns
            tests_display = str(tests) if tests > 0 else "—"
            duration_display = f"{duration:.1f}s" if duration > 0 else "—"

            table.add_row(
                file_name,
                status_display,
                progress_display,
                tests_display,
                duration_display,
            )

        self.console.print(table)

    def display_metrics_panel(
        self, metrics: dict[str, Any], title: str = "Performance Metrics"
    ):
        """Display performance metrics panel."""
        try:
            if self._app and not self._transient:
                # Update stats panel if available
                try:
                    current_screen = self._app.screen
                    stats_panel = current_screen.query_one(StatsPanel)
                    stats_panel.update_stats(metrics)
                except Exception:
                    # Fallback to console
                    self._display_console_metrics(metrics, title)
            else:
                self._display_console_metrics(metrics, title)

        except Exception as e:
            self.logger.error(f"Failed to display metrics panel: {e}")

    def _display_console_metrics(self, metrics: dict[str, Any], title: str):
        """Display metrics in console as fallback."""
        from rich.panel import Panel

        content = []
        for operation, data in metrics.items():
            if isinstance(data, dict):
                duration = data.get("duration", 0)
                success_rate = data.get("success_rate", 0)
                items = data.get("items_processed", 0)

                content.append(f"[bold cyan]{operation}[/]")
                content.append(f"  Duration: {duration:.1f}s")
                if items > 0:
                    content.append(f"  Items: {items}")
                if success_rate > 0:
                    content.append(f"  Success: {success_rate:.0%}")
                content.append("")

        if content:
            panel = Panel(
                "\n".join(content[:-1]),  # Remove last empty line
                title=title,
                border_style="blue",
            )
            self.console.print(panel)

    def display_error_with_suggestions(
        self, error_message: str, suggestions: list[str], title: str = "Error"
    ):
        """Display error with suggestions."""
        try:
            full_message = f"{error_message}\n\nSuggestions:\n"
            for i, suggestion in enumerate(suggestions, 1):
                full_message += f"{i}. {suggestion}\n"

            if self._app and not self._transient:
                self._app.post_message(ErrorOccurred(error_message, full_message))
            else:
                from rich.panel import Panel

                panel = Panel(
                    full_message.strip(), title=f"[red]{title}[/]", border_style="red"
                )
                self.console.print(panel)

        except Exception as e:
            self.logger.error(f"Failed to display error with suggestions: {e}")

    def display_success_summary(self, summary_data: dict[str, Any]):
        """Display success summary."""
        try:
            message = summary_data.get("message", "Operation completed successfully")

            if self._app and not self._transient:
                # Use notifications
                try:
                    notifications = self._app.screen.query_one(Notifications)
                    notifications.success(message)
                except Exception:
                    pass

            # Console display
            self.console.print(f"[green]✓ {message}[/]")

        except Exception as e:
            self.logger.error(f"Failed to display success summary: {e}")

    # ========================================================================
    # Textual-specific Methods
    # ========================================================================

    def _show_input_dialog(
        self, prompt: str, input_type: str = "string", **kwargs: Any
    ) -> Any:
        """Show input dialog in Textual app."""
        # This would create a modal dialog for input
        # For now, fallback to console
        from rich.prompt import Prompt

        if input_type == "boolean":
            from rich.prompt import Confirm

            return Confirm.ask(prompt, default=kwargs.get("default", False))
        else:
            return Prompt.ask(prompt, default=kwargs.get("default", ""))

    def _show_confirmation_dialog(
        self, message: str, default: bool = False, **kwargs: Any
    ) -> bool:
        """Show confirmation dialog in Textual app."""
        # This would create a modal confirmation dialog
        # For now, fallback to console
        from rich.prompt import Confirm

        return Confirm.ask(message, default=default)

    def set_app(self, app: TestCraftTextualApp) -> None:
        """Set the Textual app instance."""
        self._app = app
        self._transient = False

    def set_transient(self, transient: bool = True) -> None:
        """Set transient mode (for CLI operations vs full TUI)."""
        self._transient = transient

    # ========================================================================
    # Status tracker integration (for generate command compatibility)
    # ========================================================================

    def update_file_status(
        self,
        file_path: str,
        status: str,
        progress: float = 0.0,
        tests_generated: int = 0,
        duration: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Update file status (for integration with file status tracker)."""
        if self._app and not self._transient:
            self._app.update_file_status(
                file_path, status, progress, tests_generated, duration, error
            )

    def notify(self, message: str, severity: str = "info") -> None:
        """Send a notification."""
        if self._app:
            self._app.notify(message, severity=severity)
        else:
            # Console fallback
            colors = {
                "info": "blue",
                "success": "green",
                "warning": "yellow",
                "error": "red",
            }
            color = colors.get(severity, "white")
            self.console.print(f"[{color}]{message}[/]")


class TextualInputDialog(ModalScreen):
    """Modal dialog for user input."""

    def __init__(self, prompt: str, input_type: str = "string", default: Any = None):
        super().__init__()
        self.prompt = prompt
        self.input_type = input_type
        self.default = default
        self.result = None

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.prompt, id="prompt")

            if self.input_type == "boolean":
                yield Switch(value=bool(self.default), id="input")
            else:
                yield Input(
                    value=str(self.default) if self.default else "",
                    placeholder="Enter value...",
                    id="input",
                )

            with Horizontal(id="buttons"):
                yield Button("OK", id="ok", variant="success")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            if self.input_type == "boolean":
                self.result = self.query_one("#input", Switch).value
            else:
                value = self.query_one("#input", Input).value

                # Type conversion
                if self.input_type == "int":
                    try:
                        self.result = int(value)
                    except ValueError:
                        self.result = 0
                elif self.input_type == "float":
                    try:
                        self.result = float(value)
                    except ValueError:
                        self.result = 0.0
                else:
                    self.result = value

            self.dismiss(self.result)
        else:
            self.dismiss(None)


class TextualConfirmDialog(ModalScreen):
    """Modal dialog for confirmation."""

    def __init__(self, message: str, default: bool = False):
        super().__init__()
        self.message = message
        self.default = default
        self.result = default

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.message, id="message")

            with Horizontal(id="buttons"):
                yield Button("Yes", id="yes", variant="success")
                yield Button("No", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "yes":
            self.result = True
        else:
            self.result = False

        self.dismiss(self.result)
