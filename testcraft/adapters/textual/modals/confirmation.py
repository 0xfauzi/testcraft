"""
Confirmation dialog for TestCraft Textual UI.

Provides a simple yes/no confirmation dialog with customizable
messages and button text.
"""

from collections.abc import Callable
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, Static

from .base import BaseModal


class ConfirmationDialog(BaseModal):
    """
    Modal dialog for yes/no confirmations.

    Provides a customizable confirmation dialog with optional
    dangerous action styling.
    """

    DEFAULT_CSS = """
    ConfirmationDialog .modal-message {
        margin: 1 0;
        text-align: center;
        padding: 1;
    }

    ConfirmationDialog .modal-message.warning {
        color: $warning;
    }

    ConfirmationDialog .modal-message.error {
        color: $error;
    }

    ConfirmationDialog .modal-icon {
        text-align: center;
        font-size: 200%;
        margin-bottom: 1;
    }

    ConfirmationDialog .modal-icon.warning {
        color: $warning;
    }

    ConfirmationDialog .modal-icon.danger {
        color: $error;
    }
    """

    BINDINGS = [
        Binding("y", "confirm", "Yes", show=False),
        Binding("n", "cancel", "No", show=False),
        Binding("enter", "confirm", "Confirm", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        message: str,
        title: str = "Confirm",
        yes_text: str = "Yes",
        no_text: str = "No",
        dangerous: bool = False,
        show_icon: bool = True,
        **kwargs,
    ):
        """
        Initialize the confirmation dialog.

        Args:
            message: The confirmation message to display
            title: Title for the dialog
            yes_text: Text for the confirmation button
            no_text: Text for the cancel button
            dangerous: Whether this is a dangerous action (changes styling)
            show_icon: Whether to show an icon
            **kwargs: Additional arguments passed to BaseModal
        """
        super().__init__(title=title, **kwargs)
        self.message = message
        self.yes_text = yes_text
        self.no_text = no_text
        self.dangerous = dangerous
        self.show_icon = show_icon

    def compose_content(self) -> ComposeResult:
        """Compose the confirmation dialog content."""
        if self.show_icon:
            icon = "⚠️" if self.dangerous else "❓"
            icon_class = "danger" if self.dangerous else "warning"
            yield Static(icon, classes=f"modal-icon {icon_class}")

        message_class = "error" if self.dangerous else ""
        yield Static(self.message, classes=f"modal-message {message_class}")

    def compose_footer(self) -> ComposeResult:
        """Compose the dialog footer with buttons."""
        with Horizontal(classes="modal-buttons"):
            # Yes button - use error variant for dangerous actions
            variant = "error" if self.dangerous else "primary"
            yield Button(self.yes_text, id="yes", variant=variant)

            # No button
            yield Button(self.no_text, id="no", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "yes":
            self.action_confirm()
        elif event.button.id == "no":
            self.action_cancel()

    def action_confirm(self) -> None:
        """Handle the confirm action."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Handle the cancel action."""
        self.dismiss(False)

    @classmethod
    def ask(
        cls,
        app,
        message: str,
        title: str = "Confirm",
        yes_text: str = "Yes",
        no_text: str = "No",
        dangerous: bool = False,
        callback: Callable[..., Any] = None,
    ) -> "ConfirmationDialog":
        """
        Convenience method to show a confirmation dialog.

        Args:
            app: The Textual app instance
            message: The confirmation message
            title: Dialog title
            yes_text: Text for yes button
            no_text: Text for no button
            dangerous: Whether this is a dangerous action
            callback: Optional callback function to call with result

        Returns:
            The dialog instance
        """
        dialog = cls(
            message=message,
            title=title,
            yes_text=yes_text,
            no_text=no_text,
            dangerous=dangerous,
        )

        if callback:
            dialog.set_callback(callback)

        app.push_screen(dialog)
        return dialog
