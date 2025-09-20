"""
Progress dialog for TestCraft Textual UI.

Provides a dialog for displaying progress of long-running operations
with cancellation support.
"""

from collections.abc import Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.timer import Timer
from textual.widgets import Button, ProgressBar, Static

from .base import BaseModal


class ProgressDialog(BaseModal):
    """
    Modal dialog for displaying operation progress.

    Provides a progress bar with status messages and optional
    cancellation support.
    """

    DEFAULT_CSS = """
    ProgressDialog .modal-container {
        width: 60%;
        min-width: 50;
        max-width: 80;
    }

    ProgressDialog .progress-icon {
        text-align: center;
        font-size: 200%;
        margin-bottom: 1;
    }

    ProgressDialog .progress-message {
        text-align: center;
        margin: 1 0;
        color: $text;
    }

    ProgressDialog .progress-status {
        text-align: center;
        margin: 0.5 0;
        color: $text-muted;
        font-size: 90%;
    }

    ProgressDialog .progress-bar-container {
        margin: 1 0;
        padding: 0 2;
    }

    ProgressDialog ProgressBar {
        width: 100%;
    }

    ProgressDialog .progress-percentage {
        text-align: center;
        margin: 0.5 0;
        color: $primary;
        text-style: bold;
    }

    ProgressDialog .progress-eta {
        text-align: center;
        margin: 0.5 0;
        color: $text-muted;
        font-size: 90%;
    }

    ProgressDialog .progress-details {
        background: $surface;
        border: solid $border;
        padding: 1;
        margin: 1 0;
        max-height: 10;
        overflow-y: auto;
    }

    ProgressDialog.indeterminate .progress-icon {
        animation: spin 2s linear infinite;
    }

    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        title: str = "Processing",
        message: str = "Please wait...",
        total: int | None = None,
        show_percentage: bool = True,
        show_eta: bool = True,
        show_cancel: bool = True,
        auto_close: bool = True,
        **kwargs,
    ):
        """
        Initialize the progress dialog.

        Args:
            title: Title for the dialog
            message: Main message to display
            total: Total number of steps (None for indeterminate)
            show_percentage: Whether to show percentage complete
            show_eta: Whether to show estimated time remaining
            show_cancel: Whether to show cancel button
            auto_close: Whether to auto-close on completion
            **kwargs: Additional arguments passed to BaseModal
        """
        super().__init__(title=title, dismissable=False, **kwargs)
        self.message = message
        self.total = total
        self.show_percentage = show_percentage
        self.show_eta = show_eta
        self.show_cancel = show_cancel
        self.auto_close = auto_close

        self.current = 0
        self.status_text = ""
        self.is_cancelled = False
        self.is_complete = False

        self._progress_bar: ProgressBar | None = None
        self._message_label: Static | None = None
        self._status_label: Static | None = None
        self._percentage_label: Static | None = None
        self._eta_label: Static | None = None
        self._icon_label: Static | None = None
        self._cancel_callback: Callable | None = None
        self._update_timer: Timer | None = None

    def compose_content(self) -> ComposeResult:
        """Compose the progress dialog content."""
        # Progress icon
        icon = "⏳" if self.total is None else "📊"
        self._icon_label = Static(icon, classes="progress-icon")
        if self.total is None:
            self._icon_label.add_class("indeterminate")
        yield self._icon_label

        # Main message
        self._message_label = Static(self.message, classes="progress-message")
        yield self._message_label

        # Status text
        self._status_label = Static(self.status_text, classes="progress-status")
        yield self._status_label

        # Progress bar
        with Vertical(classes="progress-bar-container"):
            if self.total is not None:
                self._progress_bar = ProgressBar(total=self.total, show_eta=False)
            else:
                # Indeterminate progress
                self._progress_bar = ProgressBar(total=100, show_eta=False)
                self._progress_bar.advance(50)  # Set to middle for indeterminate
            yield self._progress_bar

        # Percentage
        if self.show_percentage and self.total is not None:
            self._percentage_label = Static("0%", classes="progress-percentage")
            yield self._percentage_label

        # ETA
        if self.show_eta and self.total is not None:
            self._eta_label = Static("", classes="progress-eta")
            yield self._eta_label

    def compose_footer(self) -> ComposeResult:
        """Compose the dialog footer with buttons."""
        with Horizontal(classes="modal-buttons"):
            if self.show_cancel:
                yield Button("Cancel", id="cancel", variant="default")
            else:
                # Show close button when complete
                yield Button("Close", id="close", variant="primary", disabled=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "cancel":
            self.action_cancel()
        elif event.button.id == "close":
            self.action_close()

    def update_progress(
        self,
        current: int | None = None,
        message: str | None = None,
        status: str | None = None,
    ) -> None:
        """
        Update the progress dialog.

        Args:
            current: Current progress value
            message: New main message
            status: New status text
        """
        if self.is_cancelled:
            return

        # Update current progress
        if current is not None and self.total is not None:
            self.current = min(current, self.total)
            if self._progress_bar:
                self._progress_bar.update(progress=self.current)

            # Update percentage
            if self._percentage_label:
                percentage = (self.current / self.total) * 100
                self._percentage_label.update(f"{percentage:.0f}%")

            # Check for completion
            if self.current >= self.total:
                self.mark_complete()

        # Update message
        if message and self._message_label:
            self.message = message
            self._message_label.update(message)

        # Update status
        if status and self._status_label:
            self.status_text = status
            self._status_label.update(status)

    def advance(self, amount: int = 1) -> None:
        """
        Advance the progress by a given amount.

        Args:
            amount: Amount to advance by
        """
        if self.total is not None:
            self.update_progress(self.current + amount)

    def mark_complete(self) -> None:
        """Mark the operation as complete."""
        self.is_complete = True

        # Update UI
        if self._icon_label:
            self._icon_label.update("✅")
            self._icon_label.remove_class("indeterminate")

        if self._message_label:
            self._message_label.update("Complete!")

        if self._percentage_label:
            self._percentage_label.update("100%")

        # Enable close button
        close_button = self.query_one("#close", Button)
        if close_button:
            close_button.disabled = False

        # Auto close if configured
        if self.auto_close:
            self.set_timer(1.0, self.action_close)

    def mark_error(self, error_message: str = "An error occurred") -> None:
        """
        Mark the operation as failed.

        Args:
            error_message: Error message to display
        """
        self.is_complete = True

        # Update UI
        if self._icon_label:
            self._icon_label.update("❌")
            self._icon_label.remove_class("indeterminate")

        if self._message_label:
            self._message_label.update(error_message)
            self._message_label.add_class("error")

        # Enable close button
        close_button = self.query_one("#close", Button)
        if close_button:
            close_button.disabled = False
            close_button.variant = "error"

    def set_indeterminate(self, indeterminate: bool = True) -> None:
        """
        Set whether the progress is indeterminate.

        Args:
            indeterminate: Whether progress is indeterminate
        """
        if indeterminate:
            self.total = None
            if self._icon_label:
                self._icon_label.add_class("indeterminate")
            if self._percentage_label:
                self._percentage_label.display = False
            if self._eta_label:
                self._eta_label.display = False
        else:
            if self._icon_label:
                self._icon_label.remove_class("indeterminate")

    def set_cancel_callback(self, callback: Callable) -> None:
        """
        Set a callback to be called when cancel is pressed.

        Args:
            callback: Function to call on cancel
        """
        self._cancel_callback = callback

    def action_cancel(self) -> None:
        """Handle the cancel action."""
        self.is_cancelled = True

        # Call cancel callback if set
        if self._cancel_callback:
            self._cancel_callback()

        # Update UI
        if self._message_label:
            self._message_label.update("Cancelling...")

        # Dismiss after a short delay
        self.set_timer(0.5, lambda: self.dismiss("cancelled"))

    def action_close(self) -> None:
        """Handle the close action."""
        self.dismiss("complete" if self.is_complete else None)

    @classmethod
    def show_progress(
        cls,
        app,
        title: str = "Processing",
        message: str = "Please wait...",
        total: int | None = None,
        callback: callable = None,
    ) -> "ProgressDialog":
        """
        Convenience method to show a progress dialog.

        Args:
            app: The Textual app instance
            title: Dialog title
            message: Progress message
            total: Total steps (None for indeterminate)
            callback: Optional callback for completion

        Returns:
            The dialog instance
        """
        dialog = cls(title=title, message=message, total=total)

        if callback:
            dialog.set_callback(callback)

        app.push_screen(dialog)
        return dialog
