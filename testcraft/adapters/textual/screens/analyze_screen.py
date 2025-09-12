"""
Analyze screen for TestCraft Textual UI.

Provides interface for running code analysis and showing
categorized recommendations for test improvements.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Vertical
from textual.widgets import Header, Footer, Static, Button, Input, DataTable


class AnalyzeScreen(Screen):
    """Screen for code analysis operations."""
    
    BINDINGS = [
        ("ctrl+s", "start_analysis", "Start Analysis"),
        ("r", "refresh", "Refresh"),
    ]
    
    def compose(self) -> ComposeResult:
        """Compose the Analyze screen."""
        yield Header()
        yield Footer()
        
        with Container(id="main-content"):
            yield Static("Code Analysis", id="title")
            
            # Configuration
            with Container(id="config-panel", classes="panel"):
                yield Static("Analysis Configuration")
                yield Input(placeholder="Project path...", id="project-path")
                yield Button("Start Analysis", id="start-btn", variant="success")
            
            # Results
            with Container(id="results-panel", classes="panel"):
                yield Static("Analysis Results")
                yield DataTable(id="results-table")
    
    def on_mount(self) -> None:
        """Initialize the analyze screen."""
        self.title = "TestCraft - Analyze Code"
        
        # Setup results table
        table = self.query_one("#results-table", DataTable)
        table.add_column("File", key="file")
        table.add_column("Category", key="category")
        table.add_column("Recommendation", key="recommendation")
        table.add_column("Priority", key="priority")
    
    def action_start_analysis(self) -> None:
        """Start code analysis."""
        self.app.log("Analysis started (placeholder)")
    
    def action_refresh(self) -> None:
        """Refresh analysis results."""
        self.app.log("Analysis refreshed")
    
    def refresh(self, *args, **kwargs) -> None:
        """Refresh hook compatible with Textual's API."""
        super().refresh(*args, **kwargs)
        
        # Only call action_refresh if we're fully mounted
        if self.is_attached:
            try:
                self.action_refresh()
            except Exception as e:
                if self.app:
                    self.app.log(f"Error during analyze screen refresh: {e}")
