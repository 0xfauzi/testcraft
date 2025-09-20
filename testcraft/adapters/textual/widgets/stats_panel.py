"""
StatsPanel widget for displaying aggregate statistics.

Shows real-time statistics about the current operation including
file counts, success rates, test generation metrics, and performance data.
"""

from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from ..events import StatsUpdated


class StatsPanel(Static):
    """
    A panel displaying aggregate operation statistics.

    Shows file counts, success rates, test generation metrics,
    performance data, and operation progress.
    """

    # Reactive stats data
    stats_data: reactive[dict[str, Any]] = reactive({})

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_class("stats-panel")

        # Default stats structure
        self._default_stats = {
            "files_total": 0,
            "files_done": 0,
            "files_failed": 0,
            "files_pending": 0,
            "files_running": 0,
            "tests_generated": 0,
            "success_rate": 0.0,
            "avg_duration": 0.0,
            "total_duration": 0.0,
            "operation": "Ready",
            "start_time": None,
            "estimated_completion": None,
        }

        self.stats_data = self._default_stats.copy()

    def watch_stats_data(self, new_stats: dict[str, Any]) -> None:
        """React to stats data changes."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the displayed statistics."""
        stats = self.stats_data

        # Create the statistics table
        table = Table.grid(padding=(0, 2))
        table.add_column("Label", style="bold")
        table.add_column("Value", justify="right")

        # File processing stats
        table.add_row("Files Total:", str(stats.get("files_total", 0)))
        table.add_row("Files Done:", str(stats.get("files_done", 0)))
        table.add_row("Files Failed:", str(stats.get("files_failed", 0)))
        table.add_row("Files Running:", str(stats.get("files_running", 0)))
        table.add_row("Files Pending:", str(stats.get("files_pending", 0)))

        table.add_row("", "")  # Spacer

        # Generation stats
        table.add_row("Tests Generated:", str(stats.get("tests_generated", 0)))

        # Success rate
        success_rate = stats.get("success_rate", 0.0)
        success_color = self._get_success_rate_color(success_rate)
        success_text = Text(f"{success_rate:.1f}%", style=success_color)
        table.add_row("Success Rate:", success_text)

        table.add_row("", "")  # Spacer

        # Performance stats
        avg_duration = stats.get("avg_duration", 0.0)
        total_duration = stats.get("total_duration", 0.0)

        table.add_row("Avg Duration:", self._format_duration(avg_duration))
        table.add_row("Total Duration:", self._format_duration(total_duration))

        # Operation status
        operation = stats.get("operation", "Ready")
        table.add_row("Status:", operation)

        # Estimated completion (if available)
        estimated = stats.get("estimated_completion")
        if estimated:
            table.add_row("ETA:", estimated)

        # Create the panel
        panel = Panel(
            table,
            title="Statistics",
            title_align="left",
            border_style="blue",
        )

        # Update the widget content
        self.update(panel)

    def _get_success_rate_color(self, rate: float) -> str:
        """Get color for success rate based on value."""
        if rate >= 90:
            return "green"
        elif rate >= 70:
            return "yellow"
        else:
            return "red"

    def _format_duration(self, duration: float) -> str:
        """Format duration in seconds to human readable format."""
        if duration == 0:
            return "-"
        elif duration < 1:
            return f"{duration * 1000:.0f}ms"
        elif duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            minutes = int(duration // 60)
            seconds = duration % 60
            return f"{minutes}m{seconds:.0f}s"
        else:
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            return f"{hours}h{minutes}m"

    def update_stats(self, new_stats: dict[str, Any]) -> None:
        """Update statistics with new data."""
        # Merge with existing stats
        updated_stats = self.stats_data.copy()
        updated_stats.update(new_stats)

        # Calculate derived stats
        updated_stats = self._calculate_derived_stats(updated_stats)

        self.stats_data = updated_stats

    def _calculate_derived_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Calculate derived statistics from base data."""
        # Calculate success rate
        total = stats.get("files_total", 0)
        done = stats.get("files_done", 0)
        failed = stats.get("files_failed", 0)

        if total > 0 and (done + failed) > 0:
            stats["success_rate"] = (done / (done + failed)) * 100
        else:
            stats["success_rate"] = 0.0

        # Calculate average duration
        if done > 0 and "total_duration" in stats:
            stats["avg_duration"] = stats["total_duration"] / done
        else:
            stats["avg_duration"] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset all statistics to defaults."""
        self.stats_data = self._default_stats.copy()

    def on_stats_updated(self, event: StatsUpdated) -> None:
        """Handle stats update events."""
        self.update_stats(event.stats)


class CompactStatsPanel(Static):
    """
    A compact version of the stats panel for smaller spaces.

    Shows only the most important metrics in a horizontal layout.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_class("stats-panel", "compact")
        self._stats = {
            "done": 0,
            "failed": 0,
            "total": 0,
            "tests": 0,
        }

    def update_stats(self, stats: dict[str, Any]) -> None:
        """Update with new statistics."""
        self._stats.update(
            {
                "done": stats.get("files_done", 0),
                "failed": stats.get("files_failed", 0),
                "total": stats.get("files_total", 0),
                "tests": stats.get("tests_generated", 0),
            }
        )
        self._update_display()

    def _update_display(self) -> None:
        """Update the compact display."""
        done = self._stats["done"]
        failed = self._stats["failed"]
        total = self._stats["total"]
        tests = self._stats["tests"]

        # Calculate completion percentage
        if total > 0:
            complete_pct = ((done + failed) / total) * 100
        else:
            complete_pct = 0.0

        # Create compact display text
        status_text = Text()
        status_text.append("Files: ", style="bold")
        status_text.append(f"{done}", style="green")
        status_text.append("/")
        status_text.append(f"{failed}", style="red")
        status_text.append("/")
        status_text.append(f"{total}", style="white")
        status_text.append(f" ({complete_pct:.0f}%)", style="dim")

        status_text.append("  Tests: ", style="bold")
        status_text.append(f"{tests}", style="cyan")

        self.update(status_text)
