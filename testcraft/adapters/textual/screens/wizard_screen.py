"""
Configuration wizard screen for TestCraft Textual UI.

Provides a multi-step wizard for configuring TestCraft settings,
including project setup, model configuration, and preferences.
"""

from typing import Dict, Any, List
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header, Footer, Static, Button, Input, Switch, Select, 
    TabbedContent, TabPane, ProgressBar
)


class WizardScreen(Screen):
    """Configuration wizard screen."""
    
    BINDINGS = [
        ("ctrl+n", "next_step", "Next Step"),
        ("ctrl+p", "previous_step", "Previous Step"),
        ("ctrl+s", "save_config", "Save Configuration"),
        ("escape", "cancel_wizard", "Cancel"),
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_step = 0
        self.config_data = {}
        
        # Define wizard steps
        self.steps = [
            ("project", "Project Setup"),
            ("models", "AI Model Configuration"),
            ("generation", "Generation Settings"), 
            ("coverage", "Coverage Settings"),
            ("advanced", "Advanced Options"),
            ("summary", "Review & Save"),
        ]
    
    def compose(self) -> ComposeResult:
        """Compose the wizard screen."""
        yield Header()
        yield Footer()
        
        with Container(id="main-content"):
            yield Static("TestCraft Configuration Wizard", id="title")
            
            # Progress indicator
            with Container(id="progress-panel", classes="panel"):
                yield Static("Setup Progress")
                yield ProgressBar(id="wizard-progress")
                yield Static(id="step-info")
            
            # Wizard content using tabs
            with TabbedContent(id="wizard-tabs"):
                yield TabPane("Project", self._create_project_tab(), id="tab-project")
                yield TabPane("Models", self._create_models_tab(), id="tab-models")
                yield TabPane("Generation", self._create_generation_tab(), id="tab-generation")
                yield TabPane("Coverage", self._create_coverage_tab(), id="tab-coverage")
                yield TabPane("Advanced", self._create_advanced_tab(), id="tab-advanced")
                yield TabPane("Summary", self._create_summary_tab(), id="tab-summary")
            
            # Navigation buttons
            with Horizontal(id="nav-buttons"):
                yield Button("Previous", id="prev-btn", disabled=True)
                yield Button("Next", id="next-btn", variant="primary")
                yield Button("Save & Exit", id="save-btn", variant="success")
                yield Button("Cancel", id="cancel-btn")
    
    def _create_project_tab(self) -> Container:
        """Create project setup tab content."""
        return Container(
            Static("Project Configuration", classes="tab-title"),
            Static("Configure your TestCraft project settings:"),
            
            Static("Project Name:", classes="form-label"),
            Input(placeholder="My TestCraft Project", id="project-name"),
            
            Static("Project Path:", classes="form-label"),
            Input(placeholder="/path/to/project", id="project-path"),
            
            Static("Default Include Patterns:", classes="form-label"),
            Input(value="**/*.py", id="include-patterns"),
            
            Static("Default Exclude Patterns:", classes="form-label"),
            Input(value="**/test_*.py,**/*_test.py", id="exclude-patterns"),
            
            classes="wizard-step"
        )
    
    def _create_models_tab(self) -> Container:
        """Create AI models configuration tab content."""
        return Container(
            Static("AI Model Configuration", classes="tab-title"),
            Static("Configure the AI models for test generation:"),
            
            Static("Primary Model Provider:", classes="form-label"),
            Select(
                options=[
                    ("openai", "OpenAI"),
                    ("anthropic", "Anthropic"),
                    ("local", "Local Model"),
                ],
                id="model-provider"
            ),
            
            Static("Model Name:", classes="form-label"),
            Select(
                options=[
                    ("gpt-4", "GPT-4"),
                    ("gpt-3.5-turbo", "GPT-3.5 Turbo"),
                    ("claude-3", "Claude 3"),
                ],
                id="model-name"
            ),
            
            Static("API Key:", classes="form-label"),
            Input(placeholder="Enter your API key...", password=True, id="api-key"),
            
            Static("Temperature:", classes="form-label"),
            Input(value="0.7", id="temperature"),
            
            Static("Max Tokens:", classes="form-label"),
            Input(value="2000", id="max-tokens"),
            
            classes="wizard-step"
        )
    
    def _create_generation_tab(self) -> Container:
        """Create generation settings tab content."""
        return Container(
            Static("Test Generation Settings", classes="tab-title"),
            Static("Configure default test generation behavior:"),
            
            Static("Default Batch Size:", classes="form-label"),
            Input(value="5", id="batch-size"),
            
            Static("Max Tests per File:", classes="form-label"),
            Input(value="10", id="max-tests-per-file"),
            
            Horizontal(
                Static("Enable Immediate Mode:", classes="form-label"),
                Switch(value=True, id="immediate-mode"),
                classes="form-row"
            ),
            
            Horizontal(
                Static("Enable Refinement:", classes="form-label"),
                Switch(value=True, id="enable-refinement"),
                classes="form-row"
            ),
            
            Horizontal(
                Static("Auto-save Generated Tests:", classes="form-label"),
                Switch(value=True, id="auto-save"),
                classes="form-row"
            ),
            
            Static("Test Output Directory:", classes="form-label"),
            Input(value="tests/", id="test-output-dir"),
            
            classes="wizard-step"
        )
    
    def _create_coverage_tab(self) -> Container:
        """Create coverage settings tab content."""
        return Container(
            Static("Coverage Analysis Settings", classes="tab-title"),
            Static("Configure test coverage analysis:"),
            
            Static("Coverage Tool:", classes="form-label"),
            Select(
                options=[
                    ("coverage", "Coverage.py"),
                    ("pytest-cov", "Pytest-cov"),
                ],
                value="coverage",
                id="coverage-tool"
            ),
            
            Static("Minimum Coverage Threshold (%):", classes="form-label"),
            Input(value="80", id="coverage-threshold"),
            
            Horizontal(
                Static("Include Branch Coverage:", classes="form-label"),
                Switch(value=True, id="branch-coverage"),
                classes="form-row"
            ),
            
            Horizontal(
                Static("Generate Coverage Reports:", classes="form-label"),
                Switch(value=True, id="coverage-reports"),
                classes="form-row"
            ),
            
            Static("Coverage Report Format:", classes="form-label"),
            Select(
                options=[
                    ("html", "HTML"),
                    ("xml", "XML"),
                    ("json", "JSON"),
                    ("term", "Terminal"),
                ],
                value="html",
                id="coverage-format"
            ),
            
            classes="wizard-step"
        )
    
    def _create_advanced_tab(self) -> Container:
        """Create advanced options tab content."""
        return Container(
            Static("Advanced Options", classes="tab-title"),
            Static("Configure advanced TestCraft features:"),
            
            Horizontal(
                Static("Enable Verbose Logging:", classes="form-label"),
                Switch(value=False, id="verbose-logging"),
                classes="form-row"
            ),
            
            Horizontal(
                Static("Enable Telemetry:", classes="form-label"),
                Switch(value=True, id="enable-telemetry"),
                classes="form-row"
            ),
            
            Static("Custom Prompt Templates Directory:", classes="form-label"),
            Input(placeholder="Leave empty for defaults", id="prompt-templates-dir"),
            
            Static("Plugin Directory:", classes="form-label"),
            Input(placeholder="Leave empty for defaults", id="plugin-dir"),
            
            Static("Cache Directory:", classes="form-label"),
            Input(value="~/.testcraft/cache", id="cache-dir"),
            
            Static("Log Level:", classes="form-label"),
            Select(
                options=[
                    ("DEBUG", "Debug"),
                    ("INFO", "Info"),
                    ("WARNING", "Warning"),
                    ("ERROR", "Error"),
                ],
                value="INFO",
                id="log-level"
            ),
            
            classes="wizard-step"
        )
    
    def _create_summary_tab(self) -> Container:
        """Create configuration summary tab content."""
        return Container(
            Static("Configuration Summary", classes="tab-title"),
            Static("Review your configuration before saving:"),
            
            Static(id="config-summary"),
            
            Horizontal(
                Button("Save Configuration", id="save-config-btn", variant="success"),
                Button("Reset to Defaults", id="reset-btn", variant="warning"),
                classes="form-buttons"
            ),
            
            classes="wizard-step"
        )
    
    def on_mount(self) -> None:
        """Initialize the wizard screen."""
        self.title = "TestCraft - Configuration Wizard"
        self._update_progress()
        self._update_navigation()
    
    def _update_progress(self) -> None:
        """Update the progress indicator."""
        progress = ((self.current_step + 1) / len(self.steps)) * 100
        self.query_one("#wizard-progress", ProgressBar).progress = progress
        
        step_name = self.steps[self.current_step][1]
        step_info = f"Step {self.current_step + 1} of {len(self.steps)}: {step_name}"
        self.query_one("#step-info", Static).update(step_info)
    
    def _update_navigation(self) -> None:
        """Update navigation button states."""
        prev_btn = self.query_one("#prev-btn", Button)
        next_btn = self.query_one("#next-btn", Button)
        
        prev_btn.disabled = self.current_step == 0
        
        if self.current_step == len(self.steps) - 1:
            next_btn.label = "Finish"
        else:
            next_btn.label = "Next"
    
    def action_next_step(self) -> None:
        """Move to next step."""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self._update_progress()
            self._update_navigation()
            
            # Switch to the appropriate tab
            tabs = self.query_one("#wizard-tabs", TabbedContent)
            tab_id = f"tab-{self.steps[self.current_step][0]}"
            tabs.active = tab_id
    
    def action_previous_step(self) -> None:
        """Move to previous step."""
        if self.current_step > 0:
            self.current_step -= 1
            self._update_progress()
            self._update_navigation()
            
            # Switch to the appropriate tab
            tabs = self.query_one("#wizard-tabs", TabbedContent)
            tab_id = f"tab-{self.steps[self.current_step][0]}"
            tabs.active = tab_id
    
    def action_save_config(self) -> None:
        """Save configuration and exit."""
        self._collect_config_data()
        self.app.log("Configuration saved (placeholder)")
        self.app.pop_screen()
    
    def action_cancel_wizard(self) -> None:
        """Cancel wizard and return to previous screen."""
        self.app.pop_screen()
    
    def _collect_config_data(self) -> None:
        """Collect configuration data from all form fields."""
        # This would collect all form values into self.config_data
        # and integrate with TestCraft's actual configuration system
        pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "next-btn":
            if self.current_step == len(self.steps) - 1:
                self.action_save_config()
            else:
                self.action_next_step()
        elif button_id == "prev-btn":
            self.action_previous_step()
        elif button_id == "save-btn":
            self.action_save_config()
        elif button_id == "cancel-btn":
            self.action_cancel_wizard()
        elif button_id == "save-config-btn":
            self.action_save_config()
        elif button_id == "reset-btn":
            self._reset_to_defaults()
    
    def _reset_to_defaults(self) -> None:
        """Reset all configuration to defaults."""
        self.app.log("Configuration reset to defaults")
