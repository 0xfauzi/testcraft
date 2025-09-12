"""
Toolbar widget for common actions and navigation.

Provides a horizontal toolbar with buttons for common TestCraft
operations and quick navigation between screens.
"""

from typing import Optional
from textual.app import ComposeResult
from textual.widgets import Button
from textual.containers import Horizontal
from textual.reactive import reactive


class Toolbar(Horizontal):
    """
    A horizontal toolbar with buttons for TestCraft navigation and actions.
    
    Uses Textual's standard compose pattern for predictable widget management.
    """
    
    # Reactive state for updating button states
    current_screen: reactive[str] = reactive("generate")
    operation_active: reactive[bool] = reactive(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_class("toolbar")
    
    def compose(self) -> ComposeResult:
        """Compose the toolbar with static buttons."""
        yield Button("Generate", id="toolbar_generate", variant="primary")
        yield Button("Analyze", id="toolbar_analyze")
        yield Button("Coverage", id="toolbar_coverage")
        yield Button("Status", id="toolbar_status")
        yield Button("Start", id="toolbar_start", variant="success")
        yield Button("Stop", id="toolbar_stop", variant="error", disabled=True)
        yield Button("Refresh", id="toolbar_refresh")
        yield Button("Settings", id="toolbar_settings")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id
        
        # Navigation buttons
        if button_id == "toolbar_generate":
            self._navigate_to("generate")
        elif button_id == "toolbar_analyze":
            self._navigate_to("analyze")
        elif button_id == "toolbar_coverage":
            self._navigate_to("coverage")
        elif button_id == "toolbar_status":
            self._navigate_to("status")
        elif button_id == "toolbar_settings":
            self._navigate_to("wizard")
        
        # Operation buttons
        elif button_id == "toolbar_start":
            self._start_operation()
        elif button_id == "toolbar_stop":
            self._stop_operation()
        elif button_id == "toolbar_refresh":
            self._refresh()
    
    def watch_current_screen(self, screen_name: str) -> None:
        """React to screen changes by updating button styles."""
        if not self.is_attached:
            return
            
        try:
            # Reset all navigation buttons to default
            for button_name in ["generate", "analyze", "coverage", "status"]:
                button = self.query_one(f"#toolbar_{button_name}", Button)
                button.variant = "default"
            
            # Highlight current screen button
            current_button = self.query_one(f"#toolbar_{screen_name.lower()}", Button)
            current_button.variant = "primary"
            
        except Exception as e:
            if self.app:
                self.app.log(f"Error updating toolbar for screen {screen_name}: {e}")
    
    def watch_operation_active(self, is_active: bool) -> None:
        """React to operation state changes."""
        if not self.is_attached:
            return
            
        try:
            # Update operation buttons
            start_button = self.query_one("#toolbar_start", Button)
            stop_button = self.query_one("#toolbar_stop", Button)
            
            start_button.disabled = is_active
            stop_button.disabled = not is_active
            
            # Disable navigation during operations
            for button_name in ["generate", "analyze", "coverage", "status", "settings"]:
                button = self.query_one(f"#toolbar_{button_name}", Button)
                button.disabled = is_active
                
        except Exception as e:
            if self.app:
                self.app.log(f"Error updating toolbar operation state: {e}")
    
    def _navigate_to(self, screen_name: str) -> None:
        """Navigate to a specific screen."""
        if self.app:
            try:
                self.app.switch_screen(screen_name)
                self.current_screen = screen_name
            except Exception as e:
                self.app.log(f"Error navigating to {screen_name}: {e}")
    
    def _start_operation(self) -> None:
        """Start the current screen's operation."""
        if self.app and hasattr(self.app.screen, 'start_operation'):
            try:
                self.app.screen.start_operation()
                self.operation_active = True
            except Exception as e:
                self.app.log(f"Error starting operation: {e}")
    
    def _stop_operation(self) -> None:
        """Stop the current operation."""
        if self.app and hasattr(self.app.screen, 'stop_operation'):
            try:
                self.app.screen.stop_operation()
                self.operation_active = False
            except Exception as e:
                self.app.log(f"Error stopping operation: {e}")
    
    def _refresh(self) -> None:
        """Refresh the current screen."""
        if self.app and hasattr(self.app.screen, 'refresh'):
            try:
                self.app.screen.refresh()
            except Exception as e:
                self.app.log(f"Error refreshing screen: {e}")


class SimpleToolbar(Horizontal):
    """
    A simpler toolbar with just essential buttons.
    
    Used in compact layouts where space is limited.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_class("simple-toolbar")
    
    def compose(self) -> ComposeResult:
        """Compose the simple toolbar."""
        yield Button("Start", id="start", variant="success")
        yield Button("Stop", id="stop", variant="error", disabled=True)
        yield Button("Refresh", id="refresh")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle simple toolbar button presses."""
        button_id = event.button.id
        
        if button_id == "start":
            self._toggle_operation(True)
        elif button_id == "stop":
            self._toggle_operation(False)
        elif button_id == "refresh":
            self._refresh()
    
    def _toggle_operation(self, start: bool) -> None:
        """Toggle operation state."""
        start_button = self.query_one("#start", Button)
        stop_button = self.query_one("#stop", Button)
        
        start_button.disabled = start
        stop_button.disabled = not start
        
        # Notify parent or app
        if self.app:
            if start and hasattr(self.app.screen, 'start_operation'):
                self.app.screen.start_operation()
            elif not start and hasattr(self.app.screen, 'stop_operation'):
                self.app.screen.stop_operation()
    
    def _refresh(self) -> None:
        """Refresh current screen."""
        if self.app and hasattr(self.app.screen, 'refresh'):
            self.app.screen.refresh()