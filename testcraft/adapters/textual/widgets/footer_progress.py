"""
FooterProgress widget for showing overall operation progress.

Displays a progress bar with dots, elapsed time, current status,
and keyboard hints in the footer area.
"""

import time

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import ProgressBar, Static

from ..events import ProgressUpdated


class FooterProgress(Static):
    """
    Footer progress widget showing operation status and progress.

    Displays:
    - 10-dot progress indicator
    - Elapsed time
    - Current operation status
    - Keyboard hints
    """

    # Reactive attributes
    current_progress: reactive[int] = reactive(0)
    total_progress: reactive[int] = reactive(100)
    status_message: reactive[str] = reactive("Ready")
    operation_active: reactive[bool] = reactive(False)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_class("footer-progress")

        # Time tracking
        self._start_time: float | None = None
        self._last_update_time: float | None = None

        # Update timer for elapsed time
        self._timer = None

    def on_mount(self) -> None:
        """Initialize the footer progress on mount."""
        self._update_display()
        # Update display every second when active
        self._timer = self.set_interval(1.0, self._update_display_if_active)

    def on_unmount(self) -> None:
        """Clean up timer on unmount."""
        if self._timer:
            self._timer.stop()

    def watch_current_progress(self, new_progress: int) -> None:
        """React to progress changes."""
        self._update_display()

    def watch_total_progress(self, new_total: int) -> None:
        """React to total progress changes."""
        self._update_display()

    def watch_status_message(self, new_message: str) -> None:
        """React to status message changes."""
        self._update_display()

    def watch_operation_active(self, is_active: bool) -> None:
        """React to operation status changes."""
        if is_active and not self._start_time:
            self._start_time = time.time()
        elif not is_active:
            self._start_time = None

        self._update_display()

    def _update_display_if_active(self) -> None:
        """Update display only if operation is active."""
        if self.operation_active:
            self._update_display()

    def _update_display(self) -> None:
        """Update the footer progress display."""
        # Build the footer content
        footer_text = Text()

        # Left side: Progress dots and status
        progress_dots = self._create_progress_dots()
        footer_text.append(progress_dots)
        footer_text.append("  ")
        footer_text.append(self.status_message, style="white")

        # Center: Elapsed time (if active)
        if self.operation_active and self._start_time:
            elapsed = time.time() - self._start_time
            elapsed_text = self._format_elapsed_time(elapsed)
            footer_text.append(f"  [{elapsed_text}]", style="dim")

        # Right side: Keyboard hints (we'll add some padding to push right)
        # For now, keep it simple - in a real implementation you might want
        # to use a more complex layout with Horizontal containers
        footer_text.append("    ")
        hints = self._get_keyboard_hints()
        footer_text.append(hints, style="dim")

        self.update(footer_text)

    def _create_progress_dots(self) -> Text:
        """Create the 10-dot progress indicator."""
        if self.total_progress <= 0:
            return Text("●●●●●●●●●●", style="dim")

        # Calculate how many dots should be filled
        progress_ratio = min(self.current_progress / self.total_progress, 1.0)
        filled_dots = int(progress_ratio * 10)

        dots_text = Text()

        # Add filled dots
        if filled_dots > 0:
            dots_text.append("●" * filled_dots, style="green")

        # Add empty dots
        if filled_dots < 10:
            dots_text.append("○" * (10 - filled_dots), style="dim")

        return dots_text

    def _format_elapsed_time(self, elapsed_seconds: float) -> str:
        """Format elapsed time in a compact format."""
        if elapsed_seconds < 60:
            return f"{elapsed_seconds:.0f}s"
        elif elapsed_seconds < 3600:
            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)
            return f"{minutes}m{seconds:02d}s"
        else:
            hours = int(elapsed_seconds // 3600)
            minutes = int((elapsed_seconds % 3600) // 60)
            return f"{hours}h{minutes:02d}m"

    def _get_keyboard_hints(self) -> str:
        """Get relevant keyboard hints based on current state."""
        if self.operation_active:
            return "p=pause r=refresh l=logs q=quit"
        else:
            return "g=generate a=analyze c=coverage s=status q=quit"

    def start_operation(self, total: int = 100, message: str = "Starting...") -> None:
        """Start a new operation."""
        self.current_progress = 0
        self.total_progress = total
        self.status_message = message
        self.operation_active = True

    def update_progress(
        self, current: int, total: int | None = None, message: str | None = None
    ) -> None:
        """Update progress values."""
        self.current_progress = current
        if total is not None:
            self.total_progress = total
        if message is not None:
            self.status_message = message

        self._last_update_time = time.time()

    def complete_operation(self, message: str = "Completed") -> None:
        """Complete the current operation."""
        self.current_progress = self.total_progress
        self.status_message = message
        self.operation_active = False

    def fail_operation(self, message: str = "Failed") -> None:
        """Mark operation as failed."""
        self.status_message = message
        self.operation_active = False

    def set_idle(self, message: str = "Ready") -> None:
        """Set to idle state."""
        self.current_progress = 0
        self.total_progress = 100
        self.status_message = message
        self.operation_active = False

    def on_progress_updated(self, event: ProgressUpdated) -> None:
        """Handle progress update events."""
        self.update_progress(event.current, event.total, event.message)


class SimpleProgressBar(ProgressBar):
    """
    A simplified progress bar for use in tight spaces.

    Shows just the progress bar without additional text.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_class("simple-progress")

    def set_progress(self, current: float, total: float = 100.0) -> None:
        """Set the progress value."""
        if total > 0:
            progress = min(current / total, 1.0) * 100
            self.progress = progress
        else:
            self.progress = 0
