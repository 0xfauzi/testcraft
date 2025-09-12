"""
Error dialog for TestCraft Textual UI.

Provides a dialog for displaying errors with detailed information
and recovery suggestions.
"""

from typing import List, Optional
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Button, Static, Collapsible

from .base import BaseModal


class ErrorDialog(BaseModal):
    """
    Modal dialog for displaying errors.
    
    Provides a comprehensive error display with message, details,
    and recovery suggestions.
    """
    
    DEFAULT_CSS = """
    ErrorDialog .modal-container {
        border: thick $error;
        max-width: 80;
    }
    
    ErrorDialog .error-icon {
        text-align: center;
        font-size: 200%;
        color: $error;
        margin-bottom: 1;
    }
    
    ErrorDialog .error-message {
        color: $error;
        text-style: bold;
        margin: 1 0;
        text-align: center;
    }
    
    ErrorDialog .error-details {
        background: $surface;
        border: solid $border;
        padding: 1;
        margin: 1 0;
        max-height: 15;
        overflow-y: auto;
        color: $text-muted;
    }
    
    ErrorDialog .error-code {
        font-family: monospace;
        background: $background;
        padding: 0.5 1;
        margin: 0.5 0;
        border: solid $border;
    }
    
    ErrorDialog .suggestions-title {
        color: $warning;
        text-style: bold;
        margin-top: 1;
    }
    
    ErrorDialog .suggestion-item {
        margin-left: 2;
        margin-top: 0.5;
        color: $text;
    }
    
    ErrorDialog .suggestion-item::before {
        content: "• ";
        color: $primary;
    }
    
    ErrorDialog Collapsible {
        margin: 1 0;
    }
    """
    
    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
        Binding("enter", "close", "Close", show=False),
        Binding("c", "copy", "Copy Error", show=True),
    ]
    
    def __init__(
        self,
        error: Exception,
        title: str = "Error",
        message: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        show_traceback: bool = True,
        **kwargs
    ):
        """
        Initialize the error dialog.
        
        Args:
            error: The exception to display
            title: Title for the dialog
            message: Optional custom error message
            suggestions: Optional list of recovery suggestions
            show_traceback: Whether to show the full traceback
            **kwargs: Additional arguments passed to BaseModal
        """
        super().__init__(title=title, dismissable=True, **kwargs)
        self.error = error
        self.message = message or str(error)
        self.suggestions = suggestions or []
        self.show_traceback = show_traceback
        
        # Generate recovery suggestions if not provided
        if not self.suggestions:
            self.suggestions = self._generate_suggestions()
    
    def compose_content(self) -> ComposeResult:
        """Compose the error dialog content."""
        # Error icon
        yield Static("❌", classes="error-icon")
        
        # Error message
        yield Static(self.message, classes="error-message")
        
        # Error details
        if self.show_traceback and hasattr(self.error, '__traceback__'):
            with Collapsible(title="Error Details", collapsed=False):
                with ScrollableContainer(classes="error-details"):
                    # Exception type and value
                    yield Static(
                        f"Exception Type: {type(self.error).__name__}",
                        classes="error-info"
                    )
                    yield Static(
                        f"Exception Value: {str(self.error)}",
                        classes="error-info"
                    )
                    
                    # Traceback
                    if self.show_traceback:
                        import traceback
                        tb_lines = traceback.format_exception(
                            type(self.error),
                            self.error,
                            self.error.__traceback__
                        )
                        tb_text = "".join(tb_lines)
                        yield Static(tb_text, classes="error-code")
        
        # Recovery suggestions
        if self.suggestions:
            with Vertical():
                yield Static("💡 Suggestions:", classes="suggestions-title")
                for suggestion in self.suggestions:
                    yield Static(suggestion, classes="suggestion-item")
    
    def compose_footer(self) -> ComposeResult:
        """Compose the dialog footer with buttons."""
        with Horizontal(classes="modal-buttons"):
            yield Button("Close", id="close", variant="primary")
            yield Button("Copy Error", id="copy", variant="default")
            
            # Add report button if applicable
            if self._should_show_report_button():
                yield Button("Report Issue", id="report", variant="warning")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "close":
            self.action_close()
        elif event.button.id == "copy":
            self.action_copy()
        elif event.button.id == "report":
            self.action_report()
    
    def action_close(self) -> None:
        """Handle the close action."""
        self.dismiss(None)
    
    def action_copy(self) -> None:
        """Copy error details to clipboard."""
        import traceback
        
        # Format error for copying
        error_text = f"""
Error: {self.message}
Type: {type(self.error).__name__}
Value: {str(self.error)}

Traceback:
{"".join(traceback.format_exception(type(self.error), self.error, self.error.__traceback__))}

Suggestions:
{chr(10).join(f"- {s}" for s in self.suggestions)}
        """.strip()
        
        # Copy to clipboard (this would need platform-specific implementation)
        # For now, just log it
        print(f"Error copied to clipboard:\n{error_text}")
        
        # Show confirmation
        self.app.bell()
    
    def action_report(self) -> None:
        """Handle the report issue action."""
        # This would open a bug report form or link
        # For now, just dismiss with a special result
        self.dismiss("report")
    
    def _generate_suggestions(self) -> List[str]:
        """
        Generate recovery suggestions based on the error type.
        
        Returns:
            List of suggestion strings
        """
        suggestions = []
        error_type = type(self.error).__name__
        error_str = str(self.error).lower()
        
        # Common error patterns and suggestions
        if error_type == "FileNotFoundError":
            suggestions.append("Check that the file path is correct")
            suggestions.append("Ensure you have the necessary permissions")
            suggestions.append("Verify the file hasn't been moved or deleted")
        
        elif error_type == "PermissionError":
            suggestions.append("Check file/directory permissions")
            suggestions.append("Try running with appropriate privileges")
            suggestions.append("Ensure the file is not locked by another process")
        
        elif error_type == "ConnectionError" or "connection" in error_str:
            suggestions.append("Check your internet connection")
            suggestions.append("Verify the server is accessible")
            suggestions.append("Check firewall settings")
        
        elif error_type == "ValidationError" or "validation" in error_str:
            suggestions.append("Review the input data format")
            suggestions.append("Check for required fields")
            suggestions.append("Ensure values meet validation criteria")
        
        elif error_type == "ImportError" or error_type == "ModuleNotFoundError":
            suggestions.append("Install missing dependencies")
            suggestions.append("Check your Python environment")
            suggestions.append("Verify the module name is correct")
        
        elif "timeout" in error_str:
            suggestions.append("Try the operation again")
            suggestions.append("Check for network issues")
            suggestions.append("Consider increasing timeout values")
        
        elif "memory" in error_str:
            suggestions.append("Free up system memory")
            suggestions.append("Process smaller batches of data")
            suggestions.append("Check for memory leaks")
        
        else:
            # Generic suggestions
            suggestions.append("Try the operation again")
            suggestions.append("Check the application logs for details")
            suggestions.append("Contact support if the issue persists")
        
        return suggestions
    
    def _should_show_report_button(self) -> bool:
        """
        Determine if the report button should be shown.
        
        Returns:
            True if report button should be shown
        """
        # Show report button for unexpected errors
        expected_errors = [
            "FileNotFoundError",
            "PermissionError",
            "ValidationError",
            "ValueError",
            "KeyError",
        ]
        return type(self.error).__name__ not in expected_errors
    
    @classmethod
    def show_error(
        cls,
        app,
        error: Exception,
        title: str = "Error",
        message: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        callback: callable = None
    ) -> "ErrorDialog":
        """
        Convenience method to show an error dialog.
        
        Args:
            app: The Textual app instance
            error: The exception to display
            title: Dialog title
            message: Optional custom message
            suggestions: Optional recovery suggestions
            callback: Optional callback function
            
        Returns:
            The dialog instance
        """
        dialog = cls(
            error=error,
            title=title,
            message=message,
            suggestions=suggestions
        )
        
        if callback:
            dialog.set_callback(callback)
        
        app.push_screen(dialog)
        return dialog
