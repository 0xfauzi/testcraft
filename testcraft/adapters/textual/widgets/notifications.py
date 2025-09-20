"""
Notifications widget for displaying toast-like messages.

Provides temporary notification messages with different severity levels
and automatic dismissal or user interaction.
"""

import time
from collections.abc import Callable
from enum import Enum

from rich.panel import Panel
from rich.text import Text
from textual.containers import Vertical
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Static


class NotificationSeverity(Enum):
    """Notification severity levels."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class Notification:
    """A single notification message."""

    def __init__(
        self,
        message: str,
        severity: NotificationSeverity = NotificationSeverity.INFO,
        title: str | None = None,
        duration: float = 5.0,
        dismissible: bool = True,
        action: Callable | None = None,
        action_label: str = "Action",
    ):
        self.message = message
        self.severity = severity
        self.title = title
        self.duration = duration
        self.dismissible = dismissible
        self.action = action
        self.action_label = action_label
        self.timestamp = time.time()
        self.id = f"notification_{self.timestamp}"


class NotificationWidget(Static):
    """A single notification display widget."""

    SEVERITY_STYLES = {
        NotificationSeverity.INFO: ("blue", "ℹ"),
        NotificationSeverity.SUCCESS: ("green", "✓"),
        NotificationSeverity.WARNING: ("yellow", "⚠"),
        NotificationSeverity.ERROR: ("red", "✗"),
    }

    def __init__(self, notification: Notification, **kwargs):
        super().__init__(**kwargs)
        self.notification = notification
        self.add_class("notification")
        self.add_class(f"notification-{notification.severity.value}")

        self._render_notification()

    def _render_notification(self) -> None:
        """Render the notification content."""
        notif = self.notification

        # Get style information
        color, icon = self.SEVERITY_STYLES.get(notif.severity, ("white", "•"))

        # Create notification content
        content = Text()

        # Add icon
        content.append(f"{icon} ", style=f"bold {color}")

        # Add title if present
        if notif.title:
            content.append(notif.title, style=f"bold {color}")
            content.append("\n")

        # Add message
        content.append(notif.message, style="white")

        # Add action hint if present
        if notif.action:
            content.append(f"\n[Enter] {notif.action_label}", style="dim")

        # Add dismiss hint if dismissible
        if notif.dismissible:
            content.append("  [Esc] Dismiss", style="dim")

        # Wrap in panel
        panel = Panel(
            content,
            border_style=color,
            padding=(0, 1),
        )

        self.update(panel)

    def on_key(self, event) -> None:
        """Handle key presses."""
        if event.key == "enter" and self.notification.action:
            self.notification.action()
            self.remove()
        elif event.key == "escape" and self.notification.dismissible:
            self.remove()


class Notifications(Vertical):
    """
    Container for managing multiple notification widgets.

    Handles automatic dismissal, stacking, and user interaction
    with notification messages.
    """

    # Maximum number of notifications to show
    max_notifications: reactive[int] = reactive(5)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_class("notifications-container")

        # Active notifications tracking
        self._notifications: list[NotificationWidget] = []
        self._timers: list[Timer] = []

    def show_notification(
        self,
        message: str,
        severity: NotificationSeverity = NotificationSeverity.INFO,
        title: str | None = None,
        duration: float = 5.0,
        dismissible: bool = True,
        action: Callable | None = None,
        action_label: str = "Action",
    ) -> None:
        """Show a new notification."""
        notification = Notification(
            message=message,
            severity=severity,
            title=title,
            duration=duration,
            dismissible=dismissible,
            action=action,
            action_label=action_label,
        )

        self._add_notification(notification)

    def _add_notification(self, notification: Notification) -> None:
        """Add a notification to the display."""
        # Create widget
        widget = NotificationWidget(notification)

        # Remove oldest if at max capacity
        if len(self._notifications) >= self.max_notifications:
            self._remove_oldest()

        # Add to display
        self.mount(widget)
        self._notifications.append(widget)

        # Set up auto-dismissal timer if duration > 0
        if notification.duration > 0:
            timer = self.set_timer(
                notification.duration, lambda w=widget: self._dismiss_notification(w)
            )
            self._timers.append(timer)

    def _remove_oldest(self) -> None:
        """Remove the oldest notification."""
        if self._notifications:
            oldest = self._notifications[0]
            self._dismiss_notification(oldest)

    def _dismiss_notification(self, widget: NotificationWidget) -> None:
        """Dismiss a specific notification widget."""
        if widget in self._notifications:
            self._notifications.remove(widget)
            widget.remove()

        # Clean up any associated timers
        self._cleanup_timers()

    def _cleanup_timers(self) -> None:
        """Clean up expired timers."""
        active_timers = []
        for timer in self._timers:
            if not timer.finished:
                active_timers.append(timer)
        self._timers = active_timers

    def clear_all(self) -> None:
        """Clear all notifications."""
        for widget in self._notifications[:]:
            self._dismiss_notification(widget)

    def clear_by_severity(self, severity: NotificationSeverity) -> None:
        """Clear notifications of a specific severity."""
        to_remove = [
            widget
            for widget in self._notifications
            if widget.notification.severity == severity
        ]

        for widget in to_remove:
            self._dismiss_notification(widget)

    # Convenience methods for different severity levels
    def info(self, message: str, title: str | None = None, **kwargs) -> None:
        """Show an info notification."""
        self.show_notification(message, NotificationSeverity.INFO, title, **kwargs)

    def success(self, message: str, title: str | None = None, **kwargs) -> None:
        """Show a success notification."""
        self.show_notification(message, NotificationSeverity.SUCCESS, title, **kwargs)

    def warning(self, message: str, title: str | None = None, **kwargs) -> None:
        """Show a warning notification."""
        self.show_notification(message, NotificationSeverity.WARNING, title, **kwargs)

    def error(self, message: str, title: str | None = None, **kwargs) -> None:
        """Show an error notification."""
        self.show_notification(message, NotificationSeverity.ERROR, title, **kwargs)


class ToastNotifications(Static):
    """
    A simpler toast-style notifications widget.

    Shows notifications that fade in/out automatically
    without requiring user interaction.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_class("toast-notifications")
        self._current_toast: str | None = None

    def show_toast(
        self,
        message: str,
        duration: float = 3.0,
        severity: NotificationSeverity = NotificationSeverity.INFO,
    ) -> None:
        """Show a toast message."""
        self._current_toast = message

        # Style based on severity
        color, icon = NotificationWidget.SEVERITY_STYLES.get(severity, ("white", "•"))

        # Display the toast
        toast_text = Text()
        toast_text.append(f"{icon} ", style=f"bold {color}")
        toast_text.append(message, style="white")

        self.update(toast_text)

        # Auto-dismiss after duration
        if duration > 0:
            self.set_timer(duration, self._clear_toast)

    def _clear_toast(self) -> None:
        """Clear the current toast."""
        self._current_toast = None
        self.update("")
