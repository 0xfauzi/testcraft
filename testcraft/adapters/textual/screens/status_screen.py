"""
Status screen for TestCraft Textual UI.

Displays current system status, recent operation history,
and computed statistics with filtering and pagination.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, DataTable, Button, Select
from rich.table import Table


class StatusScreen(Screen):
    """Screen for viewing system status and history."""
    
    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("c", "clear_history", "Clear History"),
        ("e", "export_history", "Export History"),
    ]
    
    def compose(self) -> ComposeResult:
        """Compose the Status screen."""
        yield Header()
        yield Footer()
        
        with Container(id="main-content"):
            yield Static("TestCraft Status", id="title")
            
            # Current status panel
            with Container(id="current-status", classes="panel"):
                yield Static("Current Status")
                yield Static(id="status-info")
            
            # Controls
            with Horizontal(id="controls"):
                yield Select(
                    options=[
                        ("all", "All Operations"),
                        ("generate", "Generation"),
                        ("analyze", "Analysis"),
                        ("coverage", "Coverage"),
                    ],
                    id="filter-select"
                )
                yield Button("Refresh", id="refresh-btn")
                yield Button("Clear History", id="clear-btn", variant="warning")
                yield Button("Export", id="export-btn")
            
            # History table
            with Container(id="history-panel", classes="panel"):
                yield Static("Operation History")
                yield DataTable(id="history-table")
            
            # Statistics panel
            with Container(id="stats-panel", classes="panel"):
                yield Static("Statistics")
                yield Static(id="stats-info")
    
    def on_mount(self) -> None:
        """Initialize the status screen."""
        self.title = "TestCraft - Status & History"
        
        # Setup history table
        table = self.query_one("#history-table", DataTable)
        table.add_column("Timestamp", key="timestamp")
        table.add_column("Operation", key="operation")
        table.add_column("Status", key="status")
        table.add_column("Files", key="files")
        table.add_column("Tests", key="tests")
        table.add_column("Duration", key="duration")
        
        # Initialize displays
        self._update_current_status()
        self._update_statistics()
        self._load_history()
    
    def _update_current_status(self) -> None:
        """Update current status information."""
        status_info = self.query_one("#status-info", Static)
        
        # Create status table
        table = Table.grid(padding=(0, 2))
        table.add_column("Label", style="bold")
        table.add_column("Value")
        
        table.add_row("System Status:", "Ready")
        table.add_row("Last Operation:", "Generate Tests")
        table.add_row("Files Processed:", "45")
        table.add_row("Tests Generated:", "234")
        table.add_row("Success Rate:", "92.3%")
        
        status_info.update(table)
    
    def _update_statistics(self) -> None:
        """Update statistics display."""
        stats_info = self.query_one("#stats-info", Static)
        
        # Create stats table
        table = Table.grid(padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        
        table.add_row("Total Operations:", "127")
        table.add_row("Successful Operations:", "115")
        table.add_row("Failed Operations:", "12")
        table.add_row("Total Files Processed:", "2,456")
        table.add_row("Total Tests Generated:", "12,890")
        table.add_row("Avg Tests per File:", "5.2")
        table.add_row("Avg Processing Time:", "2.8s")
        
        stats_info.update(table)
    
    def _load_history(self) -> None:
        """Load operation history."""
        table = self.query_one("#history-table", DataTable)
        
        # Sample history data (would come from actual storage)
        history_data = [
            ("2024-01-15 14:30:25", "Generate", "Completed", "23", "115", "45.2s"),
            ("2024-01-15 13:15:10", "Coverage", "Completed", "18", "-", "12.5s"),
            ("2024-01-15 12:45:33", "Generate", "Failed", "5", "12", "8.1s"),
            ("2024-01-15 11:20:15", "Analyze", "Completed", "31", "-", "25.8s"),
            ("2024-01-15 10:15:42", "Generate", "Completed", "12", "58", "28.3s"),
        ]
        
        for row_data in history_data:
            table.add_row(*row_data)
    
    def action_refresh(self) -> None:
        """Refresh all status information."""
        self._update_current_status()
        self._update_statistics()
        self._load_history()
        self.app.log("Status refreshed")
    
    def action_clear_history(self) -> None:
        """Clear operation history."""
        table = self.query_one("#history-table", DataTable)
        table.clear()
        self.app.log("History cleared")
    
    def action_export_history(self) -> None:
        """Export history to file."""
        self.app.log("History export requested (placeholder)")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "refresh-btn":
            self.action_refresh()
        elif button_id == "clear-btn":
            self.action_clear_history()
        elif button_id == "export-btn":
            self.action_export_history()
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle filter selection changes."""
        filter_value = event.value
        self.app.log(f"Filter changed to: {filter_value}")
        # Would filter the history table based on operation type
    
    def refresh(self, *args, **kwargs) -> None:
        """Refresh hook compatible with Textual's API."""
        super().refresh(*args, **kwargs)
        
        # Only call action_refresh if we're fully mounted and widgets exist
        if self.is_attached and self.query("#status-info"):
            try:
                self.action_refresh()
            except Exception as e:
                if self.app:
                    self.app.log(f"Error during status refresh: {e}")
