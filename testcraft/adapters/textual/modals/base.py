"""
Base modal class for TestCraft Textual UI dialogs.

Provides a foundation for all modal dialogs with consistent styling
and behavior.
"""

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


class BaseModal(ModalScreen):
    """
    Base class for all modal dialogs.

    Provides consistent styling, keyboard handling, and result management
    for modal dialogs in the TestCraft UI.
    """

    DEFAULT_CSS = """
    BaseModal {
        align: center middle;
        background: $overlay 60%;
    }
    
    .modal-container {
        width: 60%;
        min-width: 40;
        max-width: 80;
        height: auto;
        min-height: 11;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    
    .modal-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        text-align: center;
    }
    
    .modal-content {
        margin: 1 0;
        height: auto;
    }
    
    .modal-footer {
        margin-top: 1;
        height: 3;
        align: center middle;
    }
    
    .modal-buttons {
        layout: horizontal;
        align: center middle;
        height: 3;
    }
    
    .modal-buttons Button {
        margin: 0 1;
        min-width: 10;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, title: str = "", dismissable: bool = True, **kwargs):
        """
        Initialize the base modal.

        Args:
            title: Title for the modal dialog
            dismissable: Whether the modal can be dismissed with escape
            **kwargs: Additional arguments passed to ModalScreen
        """
        super().__init__(**kwargs)
        self.title = title
        self.dismissable = dismissable
        self.result: Any = None
        self._callback: callable | None = None

    def compose(self) -> ComposeResult:
        """Compose the base modal structure."""
        with Container(classes="modal-container"):
            if self.title:
                yield Static(self.title, classes="modal-title")

            with Vertical(classes="modal-content"):
                # Subclasses should override compose_content
                yield from self.compose_content()

            # Subclasses can override compose_footer for custom buttons
            with Container(classes="modal-footer"):
                yield from self.compose_footer()

    def compose_content(self) -> ComposeResult:
        """
        Compose the modal content.

        Should be overridden by subclasses to provide specific content.
        """
        yield Static("Modal content goes here")

    def compose_footer(self) -> ComposeResult:
        """
        Compose the modal footer.

        Should be overridden by subclasses to provide specific buttons.
        """
        yield Container(classes="modal-buttons")

    def action_cancel(self) -> None:
        """Handle the cancel action."""
        if self.dismissable:
            self.dismiss(None)

    def dismiss(self, result: Any = None) -> None:
        """
        Dismiss the modal with an optional result.

        Args:
            result: The result to return from the modal
        """
        self.result = result

        # Call callback if provided
        if self._callback:
            self._callback(result)

        # Dismiss the modal screen
        self.app.pop_screen()

    def set_callback(self, callback: callable) -> None:
        """
        Set a callback to be called when the modal is dismissed.

        Args:
            callback: Function to call with the modal result
        """
        self._callback = callback

    def on_mount(self) -> None:
        """Called when the modal is mounted."""
        # Focus the first focusable widget
        for widget in self.query("*").results():
            if widget.can_focus:
                widget.focus()
                break

    @classmethod
    def show(cls, app, **kwargs) -> "BaseModal":
        """
        Convenience method to show a modal.

        Args:
            app: The Textual app instance
            **kwargs: Arguments to pass to the modal constructor

        Returns:
            The modal instance
        """
        modal = cls(**kwargs)
        app.push_screen(modal)
        return modal
