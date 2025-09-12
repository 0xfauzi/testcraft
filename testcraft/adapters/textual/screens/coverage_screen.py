"""
Coverage screen for TestCraft Textual UI.

Provides interface for running coverage analysis and displaying
coverage metrics with detailed file-level breakdowns.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal
from textual.widgets import Header, Footer, Static, Button, Input, DataTable, ProgressBar


class CoverageScreen(Screen):
    """Screen for test coverage analysis."""
    
    BINDINGS = [
        ("ctrl+s", "start_coverage", "Start Coverage"),
        ("r", "refresh", "Refresh"),
    ]
    
    def compose(self) -> ComposeResult:
        """Compose the Coverage screen."""
        yield Header()
        yield Footer()
        
        with Container(id="main-content"):
            yield Static("Test Coverage Analysis", id="title")
            
            # Configuration
            with Container(id="config-panel", classes="panel"):
                yield Static("Coverage Configuration")
                with Horizontal():
                    yield Input(placeholder="Project path...", id="project-path")
                    yield Input(placeholder="Include patterns...", id="include-patterns")
                    yield Input(placeholder="Exclude patterns...", id="exclude-patterns")
                yield Button("Run Coverage", id="start-btn", variant="success")
            
            # Summary stats
            with Container(id="summary-panel", classes="panel"):
                yield Static("Coverage Summary")
                with Horizontal():
                    with Container(classes="stat-item"):
                        yield Static("Line Coverage:", classes="stat-label")
                        yield ProgressBar(id="line-coverage")
                    with Container(classes="stat-item"):
                        yield Static("Branch Coverage:", classes="stat-label")
                        yield ProgressBar(id="branch-coverage")
            
            # File-level results
            with Container(id="results-panel", classes="panel"):
                yield Static("File Coverage Details")
                yield DataTable(id="coverage-table")
    
    def on_mount(self) -> None:
        """Initialize the coverage screen."""
        self.title = "TestCraft - Coverage Analysis"
        
        # Setup coverage table
        table = self.query_one("#coverage-table", DataTable)
        table.add_column("File", key="file")
        table.add_column("Lines", key="lines")
        table.add_column("Covered", key="covered")
        table.add_column("Coverage %", key="coverage")
        table.add_column("Missing Lines", key="missing")
        
        # Initialize progress bars
        self.query_one("#line-coverage", ProgressBar).progress = 0
        self.query_one("#branch-coverage", ProgressBar).progress = 0
    
    def action_start_coverage(self) -> None:
        """Start coverage analysis."""
        self.app.log("Coverage analysis started (placeholder)")
        
        # Simulate some coverage data
        import random
        line_coverage = random.uniform(70, 95)
        branch_coverage = random.uniform(60, 90)
        
        self.query_one("#line-coverage", ProgressBar).progress = line_coverage
        self.query_one("#branch-coverage", ProgressBar).progress = branch_coverage
    
    def action_refresh(self) -> None:
        """Refresh coverage results."""
        self.app.log("Coverage refreshed")
    
    def refresh(self, *args, **kwargs) -> None:
        """Refresh hook compatible with Textual's API."""
        super().refresh(*args, **kwargs)
        
        # Only call action_refresh if we're fully mounted
        if self.is_attached:
            try:
                self.action_refresh()
            except Exception as e:
                if self.app:
                    self.app.log(f"Error during coverage screen refresh: {e}")
