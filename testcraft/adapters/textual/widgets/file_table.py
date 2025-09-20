"""
FileTable widget for displaying file processing status.

Provides a DataTable-based widget for showing real-time file processing
status, including progress indicators, test counts, and duration.
"""

from typing import Any

from textual.reactive import reactive
from textual.widgets import DataTable

from ..events import FileStatusChanged


class FileTable(DataTable):
    """
    A DataTable for displaying file processing status.

    Shows files being processed with their current status, progress,
    tests generated, and processing time.
    """

    # Reactive attributes for filtering and sorting
    filter_status: reactive[str] = reactive("all")
    sort_column: reactive[str] = reactive("file")
    sort_reverse: reactive[bool] = reactive(False)

    # Column definitions
    COLUMNS = [
        ("File", "file"),
        ("Status", "status"),
        ("Progress", "progress"),
        ("Tests", "tests"),
        ("Duration", "duration"),
        ("Error", "error"),
    ]

    # Status styling mappings
    STATUS_STYLES = {
        "pending": "status-pending",
        "running": "status-working",
        "done": "status-success",
        "failed": "status-error",
        "skipped": "status-warning",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_class("file-table")
        self._file_data: dict[str, dict[str, Any]] = {}
        self._setup_columns()

    def _setup_columns(self) -> None:
        """Initialize the table columns."""
        for column_label, _ in self.COLUMNS:
            self.add_column(column_label, key=column_label.lower())

    def on_mount(self) -> None:
        """Initialize the table on mount."""
        self.cursor_type = "row"
        self.zebra_stripes = True
        self.show_header = True

    def update_file_status(
        self,
        file_path: str,
        status: str,
        progress: float = 0.0,
        tests_generated: int = 0,
        duration: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Update the status of a file in the table."""
        self._file_data[file_path] = {
            "file": file_path,
            "status": status,
            "progress": progress,
            "tests": tests_generated,
            "duration": duration,
            "error": error or "",
        }
        self._refresh_table()

    def _refresh_table(self) -> None:
        """Refresh the table display with current data."""
        # Clear existing rows
        self.clear()

        # Filter and sort data
        filtered_data = self._get_filtered_data()
        sorted_data = self._get_sorted_data(filtered_data)

        # Add rows to table
        for file_path, file_data in sorted_data:
            row_cells = self._format_row_cells(file_data)
            row_key = file_path

            # Add the row with styling based on status
            self.add_row(*row_cells, key=row_key)

            # Apply status-specific styling to the row
            status = file_data.get("status", "pending")
            if status in self.STATUS_STYLES:
                # Note: Textual styling of specific rows requires custom approach
                pass

    def _get_filtered_data(self) -> list[tuple[str, dict[str, Any]]]:
        """Get data filtered by current filter settings."""
        if self.filter_status == "all":
            return list(self._file_data.items())

        return [
            (path, data)
            for path, data in self._file_data.items()
            if data.get("status", "").lower() == self.filter_status.lower()
        ]

    def _get_sorted_data(
        self, data: list[tuple[str, dict[str, Any]]]
    ) -> list[tuple[str, dict[str, Any]]]:
        """Sort data by current sort settings."""
        sort_key = self.sort_column

        # Define sort key function
        def get_sort_value(item: tuple[str, dict[str, Any]]) -> Any:
            _, file_data = item
            value = file_data.get(sort_key, "")

            # Handle numeric values
            if sort_key in ["progress", "tests", "duration"]:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return 0.0

            # Handle string values
            return str(value).lower()

        return sorted(data, key=get_sort_value, reverse=self.sort_reverse)

    def _format_row_cells(self, file_data: dict[str, Any]) -> list[str]:
        """Format row cells for display."""
        file_path = file_data.get("file", "")
        status = file_data.get("status", "pending")
        progress = file_data.get("progress", 0.0)
        tests = file_data.get("tests", 0)
        duration = file_data.get("duration", 0.0)
        error = file_data.get("error", "")

        # Format file path (show just filename for space)
        from pathlib import Path

        file_display = Path(file_path).name if file_path else ""

        # Format progress as percentage
        progress_display = f"{progress:.1f}%" if progress > 0 else "-"

        # Format duration
        if duration > 0:
            if duration < 60:
                duration_display = f"{duration:.1f}s"
            else:
                minutes = int(duration // 60)
                seconds = duration % 60
                duration_display = f"{minutes}m{seconds:.1f}s"
        else:
            duration_display = "-"

        # Format tests count
        tests_display = str(tests) if tests > 0 else "-"

        # Format error (truncate if too long)
        error_display = error[:20] + "..." if len(error) > 20 else error

        return [
            file_display,
            status.title(),
            progress_display,
            tests_display,
            duration_display,
            error_display,
        ]

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Handle header click for sorting."""
        column_key = event.column_key

        # Map display column names to data keys
        column_mapping = {
            col_label.lower(): col_key for col_label, col_key in self.COLUMNS
        }
        sort_key = column_mapping.get(column_key, column_key)

        # Toggle sort direction if same column, otherwise use ascending
        if self.sort_column == sort_key:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = sort_key
            self.sort_reverse = False

        self._refresh_table()

    def set_filter_status(self, status: str) -> None:
        """Set the status filter."""
        self.filter_status = status.lower()
        self._refresh_table()

    def clear_files(self) -> None:
        """Clear all file data."""
        self._file_data.clear()
        self.clear()

    def get_file_count_by_status(self) -> dict[str, int]:
        """Get count of files by status."""
        counts = {}
        for file_data in self._file_data.values():
            status = file_data.get("status", "pending")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def get_selected_file(self) -> str | None:
        """Get the currently selected file path."""
        try:
            cursor_row = self.cursor_row
            if cursor_row >= 0:
                # Get the file path from our data based on row index
                filtered_data = self._get_filtered_data()
                sorted_data = self._get_sorted_data(filtered_data)
                if cursor_row < len(sorted_data):
                    return sorted_data[cursor_row][0]
        except (IndexError, AttributeError):
            pass
        return None

    def on_file_status_changed(self, event: FileStatusChanged) -> None:
        """Handle file status change events."""
        self.update_file_status(
            event.file_path,
            event.status,
            event.progress,
            event.tests_generated,
            event.duration,
            event.error,
        )
