"""
Generate screen for TestCraft Textual UI.

Provides the main interface for configuring and running test generation,
including parameter forms, live file tracking, statistics, and progress.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, Button, Input, Switch, Select, Static, 
    ProgressBar, TabbedContent, TabPane
)
from textual.reactive import reactive
from textual.worker import Worker  # Import Worker for type hints
from textual import events

from ..widgets import FileTable, StatsPanel, FooterProgress, Toolbar, Notifications
from ..events import (
    OperationStarted, OperationCompleted, FileStatusChanged, 
    StatsUpdated, ErrorOccurred
)


class GenerateScreen(Screen):
    """
    Main screen for test generation operations.
    
    Layout:
    - Header: Title and navigation
    - Top: Parameter configuration form
    - Main: File table (left) + Stats panel (right)
    - Footer: Progress bar and hints
    """
    
    BINDINGS = [
        ("ctrl+s", "start_generation", "Start Generation"),
        ("ctrl+x", "stop_generation", "Stop Generation"),
        ("r", "refresh", "Refresh"),
        ("f", "toggle_filter", "Filter Files"),
        ("enter", "view_selected_file", "View File Details"),
    ]
    
    # Reactive state
    operation_active: reactive[bool] = reactive(False)
    files_processed: reactive[int] = reactive(0)
    total_files: reactive[int] = reactive(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Configuration state
        self.config = {
            "project_path": str(Path.cwd()),
            "batch_size": 5,
            "immediate": True,
            "include_patterns": "**/*.py",
            "exclude_patterns": "**/test_*.py,**/*_test.py",
            "max_tests_per_file": 10,
            "enable_refinement": True,
        }
        
        # Operation state
        self._current_worker: Optional[Worker] = None
        self._file_data: Dict[str, Dict[str, Any]] = {}
    
    def compose(self) -> ComposeResult:
        """Compose the Generate screen."""
        yield Header()
        yield Footer()
        
        # Main content container
        with Container(id="main-content"):
            # Configuration panel (top)
            with Container(id="config-panel", classes="panel"):
                yield Static("Test Generation Configuration", id="config-title")
                yield from self._create_config_form()
            
            # Main working area
            with Horizontal(id="main-area"):
                # Left: File processing table
                with Container(id="file-area"):
                    yield Static("File Processing", id="file-title")
                    yield FileTable(id="file-table")
                
                # Right: Statistics and controls
                with Container(id="sidebar", classes="stats-panel"):
                    yield StatsPanel(id="stats-panel")
                    yield Toolbar(id="toolbar")
            
            # Bottom: Notifications
            yield Notifications(id="notifications")
    
    def _create_config_form(self) -> List:
        """Create the configuration form widgets."""
        form_widgets = []
        
        # Project path
        form_widgets.extend([
            Static("Project Path:", classes="form-label"),
            Input(
                value=self.config["project_path"],
                placeholder="Enter project path...",
                id="project-path"
            ),
        ])
        
        # Batch size and immediate mode in a horizontal layout
        with Horizontal(classes="form-row"):
            form_widgets.extend([
                Static("Batch Size:", classes="form-label"),
                Input(
                    value=str(self.config["batch_size"]),
                    placeholder="5",
                    id="batch-size"
                ),
                Static("Immediate Mode:", classes="form-label"),
                Switch(value=self.config["immediate"], id="immediate-mode"),
            ])
        
        # File patterns
        form_widgets.extend([
            Static("Include Patterns:", classes="form-label"),
            Input(
                value=self.config["include_patterns"],
                placeholder="**/*.py",
                id="include-patterns"
            ),
            Static("Exclude Patterns:", classes="form-label"),
            Input(
                value=self.config["exclude_patterns"],
                placeholder="**/test_*.py,**/*_test.py",
                id="exclude-patterns"
            ),
        ])
        
        # Advanced options in a collapsible section
        with Horizontal(classes="form-row"):
            form_widgets.extend([
                Static("Max Tests/File:", classes="form-label"),
                Input(
                    value=str(self.config["max_tests_per_file"]),
                    placeholder="10",
                    id="max-tests"
                ),
                Static("Enable Refinement:", classes="form-label"),
                Switch(value=self.config["enable_refinement"], id="enable-refinement"),
            ])
        
        # Action buttons
        with Horizontal(classes="form-buttons"):
            form_widgets.extend([
                Button("Start Generation", id="start-btn", variant="success"),
                Button("Stop", id="stop-btn", variant="error", disabled=True),
                Button("Clear", id="clear-btn"),
                Button("Load Config", id="load-config-btn"),
                Button("Save Config", id="save-config-btn"),
            ])
        
        return form_widgets
    
    def on_mount(self) -> None:
        """Initialize the screen when mounted."""
        self.title = "TestCraft - Generate Tests"
        self.sub_title = "Configure and run AI-powered test generation"
        
        # Initialize widgets with current config
        self._update_form_from_config()
        
        # Set up initial stats
        self._update_stats()
    
    def _update_form_from_config(self) -> None:
        """Update form inputs from current config."""
        try:
            self.query_one("#project-path", Input).value = self.config["project_path"]
            self.query_one("#batch-size", Input).value = str(self.config["batch_size"])
            self.query_one("#immediate-mode", Switch).value = self.config["immediate"]
            self.query_one("#include-patterns", Input).value = self.config["include_patterns"]
            self.query_one("#exclude-patterns", Input).value = self.config["exclude_patterns"]
            self.query_one("#max-tests", Input).value = str(self.config["max_tests_per_file"])
            self.query_one("#enable-refinement", Switch).value = self.config["enable_refinement"]
        except Exception as e:
            self.app.log(f"Error updating form: {e}")
    
    def _update_config_from_form(self) -> None:
        """Update config from current form values."""
        try:
            self.config.update({
                "project_path": self.query_one("#project-path", Input).value,
                "batch_size": int(self.query_one("#batch-size", Input).value or "5"),
                "immediate": self.query_one("#immediate-mode", Switch).value,
                "include_patterns": self.query_one("#include-patterns", Input).value,
                "exclude_patterns": self.query_one("#exclude-patterns", Input).value,
                "max_tests_per_file": int(self.query_one("#max-tests", Input).value or "10"),
                "enable_refinement": self.query_one("#enable-refinement", Switch).value,
            })
        except (ValueError, AttributeError) as e:
            self.app.log(f"Error reading form values: {e}")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "start-btn":
            self.action_start_generation()
        elif button_id == "stop-btn":
            self.action_stop_generation()
        elif button_id == "clear-btn":
            self._clear_results()
        elif button_id == "load-config-btn":
            self._load_config()
        elif button_id == "save-config-btn":
            self._save_config()
    
    def action_start_generation(self) -> None:
        """Start test generation."""
        if self.operation_active:
            return
        
        # Update config from form
        self._update_config_from_form()
        
        # Validate configuration
        if not self._validate_config():
            return
        
        # Start the operation
        self.operation_active = True
        self._enable_form_controls(False)
        
        # Start background worker
        self._current_worker = self.run_worker(self._start_generation_worker())
        
        # Update UI
        self.app.post_message(
            OperationStarted("test_generation", "Starting test generation...")
        )
    
    def action_stop_generation(self) -> None:
        """Stop test generation."""
        if not self.operation_active:
            return
        
        # Cancel the worker
        if self._current_worker:
            self._current_worker.cancel()
            self._current_worker = None
        
        # Update state
        self.operation_active = False
        self._enable_form_controls(True)
        
        # Update UI
        self.app.post_message(
            OperationCompleted("test_generation", False, "Generation stopped by user")
        )
    
    async def _start_generation_worker(self) -> None:
        """Background worker for test generation."""
        try:
            # This would integrate with the actual TestCraft generation use case
            # For now, we'll simulate the process
            await self._simulate_generation()
            
        except Exception as e:
            self.app.post_message(
                ErrorOccurred(f"Generation failed: {e}")
            )
        finally:
            # Always clean up
            self.operation_active = False
            self._enable_form_controls(True)
            self._current_worker = None
    
    async def _simulate_generation(self) -> None:
        """Simulate test generation process."""
        import asyncio
        import random
        from pathlib import Path
        
        project_path = Path(self.config["project_path"])
        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
        
        # Find Python files to process
        include_patterns = self.config["include_patterns"].split(",")
        exclude_patterns = self.config["exclude_patterns"].split(",")
        
        files_to_process = []
        for pattern in include_patterns:
            for file_path in project_path.glob(pattern.strip()):
                if file_path.is_file():
                    # Check if excluded
                    excluded = False
                    for exclude_pattern in exclude_patterns:
                        if file_path.match(exclude_pattern.strip()):
                            excluded = True
                            break
                    
                    if not excluded:
                        files_to_process.append(str(file_path))
        
        total_files = len(files_to_process)
        self.total_files = total_files
        
        if total_files == 0:
            self.app.post_message(
                ErrorOccurred("No files found matching the specified patterns")
            )
            return
        
        # Process files in batches
        batch_size = self.config["batch_size"]
        processed = 0
        
        for i in range(0, len(files_to_process), batch_size):
            if self._current_worker and self._current_worker.is_cancelled:
                break
            
            batch = files_to_process[i:i+batch_size]
            
            for file_path in batch:
                if self._current_worker and self._current_worker.is_cancelled:
                    break
                
                # Update file status
                self.app.post_message(
                    FileStatusChanged(file_path, "running", 0.0)
                )
                
                # Simulate processing time
                processing_time = random.uniform(1.0, 3.0)
                import time
                start_time = time.time()
                
                # Simulate progress updates
                for progress in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    if self._current_worker and self._current_worker.is_cancelled:
                        break
                    
                    await asyncio.sleep(processing_time / 5)
                    
                    self.app.post_message(
                        FileStatusChanged(
                            file_path, 
                            "running", 
                            progress * 100,
                            int(progress * random.randint(3, 8)),
                            time.time() - start_time
                        )
                    )
                
                # Complete file processing
                if not (self._current_worker and self._current_worker.is_cancelled):
                    success = random.random() > 0.1  # 90% success rate
                    status = "done" if success else "failed"
                    error = None if success else "Simulated processing error"
                    tests_generated = random.randint(3, 8) if success else 0
                    
                    self.app.post_message(
                        FileStatusChanged(
                            file_path,
                            status,
                            100.0,
                            tests_generated,
                            time.time() - start_time,
                            error
                        )
                    )
                    
                    processed += 1
                    self.files_processed = processed
                    
                    # Update overall stats
                    self._update_stats()
        
        # Complete operation
        if not (self._current_worker and self._current_worker.is_cancelled):
            self.app.post_message(
                OperationCompleted(
                    "test_generation", 
                    True, 
                    f"Generated tests for {processed} files"
                )
            )
    
    def _validate_config(self) -> bool:
        """Validate the current configuration."""
        try:
            project_path = Path(self.config["project_path"])
            if not project_path.exists():
                self.app.post_message(
                    ErrorOccurred(f"Project path does not exist: {project_path}")
                )
                return False
            
            if self.config["batch_size"] <= 0:
                self.app.post_message(
                    ErrorOccurred("Batch size must be greater than 0")
                )
                return False
            
            return True
            
        except Exception as e:
            self.app.post_message(
                ErrorOccurred(f"Configuration error: {e}")
            )
            return False
    
    def _enable_form_controls(self, enabled: bool) -> None:
        """Enable or disable form controls."""
        try:
            # Toggle main action buttons
            self.query_one("#start-btn", Button).disabled = not enabled
            self.query_one("#stop-btn", Button).disabled = enabled
            
            # Toggle form inputs
            for input_widget in self.query(Input):
                input_widget.disabled = not enabled
            
            for switch_widget in self.query(Switch):
                switch_widget.disabled = not enabled
        
        except Exception as e:
            self.app.log(f"Error toggling form controls: {e}")
    
    def _clear_results(self) -> None:
        """Clear all results and reset the display."""
        try:
            file_table = self.query_one("#file-table", FileTable)
            file_table.clear_files()
            
            self._file_data.clear()
            self.files_processed = 0
            self.total_files = 0
            
            self._update_stats()
            
        except Exception as e:
            self.app.log(f"Error clearing results: {e}")
    
    def _update_stats(self) -> None:
        """Update the statistics display."""
        try:
            # Calculate stats from file data
            stats = {
                "files_total": self.total_files,
                "files_done": len([f for f in self._file_data.values() if f.get("status") == "done"]),
                "files_failed": len([f for f in self._file_data.values() if f.get("status") == "failed"]),
                "files_running": len([f for f in self._file_data.values() if f.get("status") == "running"]),
                "files_pending": self.total_files - self.files_processed,
                "tests_generated": sum(f.get("tests", 0) for f in self._file_data.values()),
                "total_duration": sum(f.get("duration", 0) for f in self._file_data.values()),
                "operation": "Generating" if self.operation_active else "Ready",
            }
            
            self.app.post_message(StatsUpdated(stats))
            
        except Exception as e:
            self.app.log(f"Error updating stats: {e}")
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        # This would integrate with TestCraft's config system
        self.app.post_message(
            ErrorOccurred("Config loading not implemented yet", "This feature will be added in a future update")
        )
    
    def _save_config(self) -> None:
        """Save configuration to file."""
        # This would integrate with TestCraft's config system
        self.app.post_message(
            ErrorOccurred("Config saving not implemented yet", "This feature will be added in a future update")
        )
    
    def action_refresh(self) -> None:
        """Refresh the screen data."""
        self._update_stats()
        self.app.log("Screen refreshed")
    
    def action_toggle_filter(self) -> None:
        """Toggle file filtering."""
        # This would show/hide filtering options
        self.app.log("File filtering toggle requested")
    
    def action_view_selected_file(self) -> None:
        """View details for the selected file."""
        try:
            file_table = self.query_one("#file-table", FileTable)
            selected_file = file_table.get_selected_file()
            
            if selected_file and selected_file in self._file_data:
                # This would show a detailed view of the file processing
                self.app.log(f"Viewing details for: {selected_file}")
            else:
                self.app.log("No file selected")
                
        except Exception as e:
            self.app.log(f"Error viewing file details: {e}")
    
    def on_file_status_changed(self, event: FileStatusChanged) -> None:
        """Handle file status updates."""
        # Store file data
        self._file_data[event.file_path] = {
            "status": event.status,
            "progress": event.progress,
            "tests": event.tests_generated,
            "duration": event.duration,
            "error": event.error,
        }
        
        # Forward to file table
        try:
            file_table = self.query_one("#file-table", FileTable)
            file_table.on_file_status_changed(event)
        except Exception as e:
            self.app.log(f"Error updating file table: {e}")
        
        # Update stats
        self._update_stats()
    
    def on_stats_updated(self, event: StatsUpdated) -> None:
        """Handle stats updates."""
        try:
            stats_panel = self.query_one("#stats-panel", StatsPanel)
            stats_panel.on_stats_updated(event)
        except Exception as e:
            self.app.log(f"Error updating stats panel: {e}")
    
    # Screen lifecycle methods
    def start_operation(self) -> None:
        """External method to start operation."""
        self.action_start_generation()
    
    def stop_operation(self) -> None:
        """External method to stop operation."""
        self.action_stop_generation()
    
    def refresh(self, *args, **kwargs) -> None:
        """Refresh hook compatible with Textual's API."""
        super().refresh(*args, **kwargs)
        
        # Only call action_refresh if we're fully mounted
        if self.is_attached:
            try:
                self.action_refresh()
            except Exception as e:
                if self.app:
                    self.app.log(f"Error during generate screen refresh: {e}")
