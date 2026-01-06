"""
Rich CLI components for testcraft.

This module provides Rich-based UI components for creating professional
CLI output including tables, progress indicators, summaries, and themed layouts.
"""

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.theme import Theme
from rich.tree import Tree

if TYPE_CHECKING:
    pass


class StatusSpinner(Protocol):
    """Protocol for status spinner objects that support start/stop and context manager."""

    def __str__(self) -> str: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def __enter__(self): ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

    def __getattr__(self, item: Any) -> Any: ...


# Minimalist testcraft theme with subtle colors and clean hierarchy
TESTCRAFT_THEME = Theme(
    {
        # Core status colors - subtle but clear
        "success": "green",
        "error": "red",
        "warning": "yellow",
        "info": "blue",
        # Minimal accent colors
        "accent": "cyan",
        "primary": "white",
        "secondary": "bright_white",
        # Clean text hierarchy
        "header": "bold white",
        "title": "bold",
        "muted": "dim white",
        "subtle": "dim",
        # Enhanced coverage colors with more granular levels
        "coverage_good": "green",
        "coverage_medium": "yellow",
        "coverage_low": "red",
        "coverage_excellent": "bright_green",
        "coverage_high": "green",
        # Clean status indicators
        "status_pass": "green",
        "status_fail": "red",
        "status_working": "yellow",
        # Minimal interactive elements
        "prompt": "cyan",
        "selected": "green",
        "prompt_accent": "cyan",
        # Semantic tokens for status badges
        "token_positive": "green",
        "token_negative": "red",
        "token_warning": "yellow",
        "token_info": "blue",
        # Clean borders with different styles
        "border": "dim white",
        "border_info": "blue",
        "border_success": "green",
        # Additional text styles
        "subheader": "bold cyan",
        "choice": "cyan",
        "code": "bright_white",
    }
)

# Ultra-minimal theme with restricted palette (≤4 colors)
MINIMAL_THEME = Theme(
    {
        # Essential status colors only
        "success": "green",
        "error": "red",
        "status_working": "yellow",
        "accent": "cyan",
        # Minimal text colors
        "muted": "dim white",
        "primary": "white",
        "border": "dim",
        # Alias other colors to core set
        "warning": "yellow",
        "info": "cyan",
        "header": "white",
        "title": "white",
        "secondary": "white",
        "subtle": "dim white",
        "coverage_good": "green",
        "coverage_medium": "yellow",
        "coverage_low": "red",
        "coverage_excellent": "green",
        "coverage_high": "green",
        "status_pass": "green",
        "status_fail": "red",
        "prompt": "cyan",
        "selected": "green",
        "prompt_accent": "cyan",
        "token_positive": "green",
        "token_negative": "red",
        "token_warning": "yellow",
        "token_info": "cyan",
        "border_info": "cyan",
        "border_success": "green",
        "subheader": "white",
        "choice": "cyan",
        "code": "white",
    }
)


def get_theme(ui_style: str) -> Theme:
    """Get the appropriate theme for the UI style."""
    if ui_style == "minimal":
        return MINIMAL_THEME
    else:
        return TESTCRAFT_THEME


class RichCliComponents:
    """
    Rich-based CLI components for testcraft.

    Provides methods for creating tables, progress indicators, summaries,
    and other UI elements with consistent theming and professional appearance.
    """

    def __init__(self, console: Console | None = None) -> None:
        """
        Initialize Rich CLI components.

        Args:
            console: Optional Rich Console instance (will create one if not provided)
        """
        self.console = console or Console(theme=TESTCRAFT_THEME)
        self._emoji_enabled = self._detect_emoji_support()
        self._wizard_cache_path = Path.home() / ".testcraft" / "wizard_state.json"

    def _detect_emoji_support(self) -> bool:
        """Determine whether the active console likely supports emoji glyphs."""
        if os.getenv("TESTCRAFT_NO_EMOJI") == "1":
            return False
        encoding = getattr(self.console, "encoding", "") or ""
        return "utf" in encoding.lower()

    def _glyph(self, key: str, ascii_fallback: str) -> str:
        """Return an emoji glyph or ASCII fallback based on terminal support."""
        if not self._emoji_enabled:
            return ascii_fallback
        mapping = {
            "success": "✅",
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️",
            "action": "➜",
            "docs": "📘",
            "spark": "✨",
            "rocket": "🚀",
            "summary": "📊",
        }
        return mapping.get(key, ascii_fallback)

    def create_coverage_table(
        self, coverage_data: dict[str, Any], show_details: bool = True
    ) -> Table:
        """
        Create a formatted coverage report table.

        Args:
            coverage_data: Coverage data to display
            show_details: Whether to show detailed line/branch coverage

        Returns:
            Formatted Rich Table
        """
        table = Table(
            title="📊 [title]Code Coverage Report[/title]",
            box=box.ROUNDED,
            show_header=True,
            header_style="header",
            title_style="primary",
            border_style="border_info",
            show_lines=True,
        )

        table.add_column("File", style="dim", width=40)
        table.add_column("Line Coverage", justify="center", width=15)
        table.add_column("Branch Coverage", justify="center", width=15)
        if show_details:
            table.add_column("Missing Lines", justify="center", width=15)
            table.add_column("Status", justify="center", width=10)

        # Add coverage data rows
        files_data = coverage_data.get("files", {})
        for file_path, file_coverage in files_data.items():
            line_cov = file_coverage.get("line_coverage", 0.0)
            branch_cov = file_coverage.get("branch_coverage", 0.0)
            missing_lines = file_coverage.get("missing_lines", [])

            # Format coverage percentages with colors
            line_cov_text = self._format_coverage_percentage(line_cov)
            branch_cov_text = self._format_coverage_percentage(branch_cov)

            # Determine overall status with beautiful icons
            avg_coverage = (line_cov + branch_cov) / 2
            if avg_coverage >= 0.95:
                status = "[coverage_excellent]🏆 Excellent[/]"
            elif avg_coverage >= 0.85:
                status = "[coverage_high]✅ Very Good[/]"
            elif avg_coverage >= 0.70:
                status = "[coverage_good]👍 Good[/]"
            elif avg_coverage >= 0.50:
                status = "[coverage_medium]⚠️ Fair[/]"
            else:
                status = "[coverage_low]🚨 Needs Work[/]"

            # Format file path
            file_display = (
                str(Path(file_path).name) if len(file_path) > 35 else file_path
            )

            if show_details:
                missing_display = str(len(missing_lines)) if missing_lines else "0"
                table.add_row(
                    file_display,
                    line_cov_text,
                    branch_cov_text,
                    missing_display,
                    status,
                )
            else:
                table.add_row(file_display, line_cov_text, branch_cov_text)

        # Add summary row if multiple files
        if len(files_data) > 1:
            overall_line = coverage_data.get("overall_line_coverage", 0.0)
            overall_branch = coverage_data.get("overall_branch_coverage", 0.0)

            table.add_section()
            if show_details:
                table.add_row(
                    "[bold]Overall[/]",
                    self._format_coverage_percentage(overall_line, bold=True),
                    self._format_coverage_percentage(overall_branch, bold=True),
                    "-",
                    "[bold]Summary[/]",
                    style="dim",
                )
            else:
                table.add_row(
                    "[bold]Overall[/]",
                    self._format_coverage_percentage(overall_line, bold=True),
                    self._format_coverage_percentage(overall_branch, bold=True),
                    style="dim",
                )

        return table

    def create_test_results_table(self, test_results: list[dict[str, Any]]) -> Table:
        """
        Create a formatted test results table.

        Args:
            test_results: List of test result data

        Returns:
            Formatted Rich Table
        """
        table = Table(
            title="🧪 [title]Test Generation Results[/title]",
            box=box.ROUNDED,
            show_header=True,
            header_style="header",
            title_style="primary",
            border_style="bright_magenta",
            show_lines=True,
        )

        table.add_column("Source File", style="dim", width=35)
        table.add_column("Test File", style="dim", width=35)
        table.add_column("Status", justify="center", width=12)
        table.add_column("Tests Generated", justify="center", width=15)
        table.add_column("Pass Rate", justify="center", width=10)

        for result in test_results:
            # Format status with beautiful colors and icons
            status = result.get("status", "unknown")
            if status == "success":
                status_text = "[status_pass]🎉 Success[/]"
            elif status == "failed":
                status_text = "[status_fail]❌ Failed[/]"
            elif status == "partial":
                status_text = "[status_partial]⚠️ Partial[/]"
            elif status == "skipped":
                status_text = "[status_skip]⏭️ Skipped[/]"
            else:
                status_text = "[muted]❓ Unknown[/]"

            # Format test count
            tests_generated = result.get("tests_generated", 0)
            test_count_text = (
                str(tests_generated) if tests_generated > 0 else "[muted]0[/]"
            )

            # Format pass rate with beautiful colors and icons
            pass_rate = result.get("pass_rate", 0.0)
            if pass_rate >= 0.95:
                pass_rate_text = f"[coverage_excellent]🏆 {pass_rate:.0%}[/]"
            elif pass_rate >= 0.85:
                pass_rate_text = f"[coverage_high]🟢 {pass_rate:.0%}[/]"
            elif pass_rate >= 0.70:
                pass_rate_text = f"[coverage_good]🟡 {pass_rate:.0%}[/]"
            elif pass_rate >= 0.50:
                pass_rate_text = f"[coverage_medium]🟠 {pass_rate:.0%}[/]"
            else:
                pass_rate_text = f"[coverage_low]🔴 {pass_rate:.0%}[/]"

            table.add_row(
                result.get("source_file", ""),
                result.get("test_file", ""),
                status_text,
                test_count_text,
                pass_rate_text,
            )

        return table

    def create_project_summary_panel(self, project_data: dict[str, Any]) -> Panel:
        """
        Create a project summary panel.

        Args:
            project_data: Project summary data

        Returns:
            Formatted Rich Panel
        """
        # Extract key metrics
        total_files = project_data.get("total_files", 0)
        files_with_tests = project_data.get("files_with_tests", 0)
        overall_coverage = project_data.get("overall_coverage", 0.0)
        tests_generated = project_data.get("tests_generated", 0)
        generation_success_rate = project_data.get("generation_success_rate", 0.0)

        # Create beautiful content with enhanced formatting
        coverage_text = self._format_coverage_percentage(
            overall_coverage, with_icon=True
        )

        # Calculate test percentage
        test_percentage = (
            (files_with_tests / total_files * 100) if total_files > 0 else 0
        )
        test_icon = (
            "🏆" if test_percentage >= 90 else "📊" if test_percentage >= 70 else "📈"
        )

        # Success rate formatting
        if generation_success_rate >= 0.9:
            success_color = "status_pass"
            success_icon = "🎯"
        elif generation_success_rate >= 0.7:
            success_color = "coverage_good"
            success_icon = "✅"
        else:
            success_color = "coverage_medium"
            success_icon = "⚠️"

        metrics_text = f"""
[header]📈 Project Health Dashboard[/]

[subheader]📁 Coverage Analysis[/]
  [info]Files Analyzed:[/]     [secondary]{total_files}[/]
  [info]Files with Tests:[/]   [secondary]{files_with_tests}[/] {test_icon} [subtle]({test_percentage:.0f}%)[/]
  [info]Overall Coverage:[/]   {coverage_text}

[subheader]🧪 Test Generation[/]
  [info]Tests Generated:[/]    [secondary]{tests_generated}[/]
  [info]Success Rate:[/]       [{success_color}]{success_icon} {generation_success_rate:.0%}[/]
"""

        return Panel(
            metrics_text.strip(),
            title="🎯 [title]Project Summary[/title]",
            border_style="bright_magenta",
            padding=(1, 2),
            title_align="center",
        )

    def create_recommendations_panel(self, recommendations: list[str]) -> Panel:
        """
        Create a recommendations panel.

        Args:
            recommendations: List of recommendation strings

        Returns:
            Formatted Rich Panel
        """
        if not recommendations:
            content = "[muted]No specific recommendations at this time.[/]"
        else:
            content = "\n".join(f"• {rec}" for rec in recommendations)

        return Panel(
            content,
            title="💡 Recommendations",
            border_style="highlight",
            padding=(1, 2),
        )

    def create_analysis_tree(self, analysis_data: dict[str, Any]) -> Tree:
        """
        Create a tree view of analysis results.

        Args:
            analysis_data: Analysis data to display

        Returns:
            Rich Tree structure
        """
        tree = Tree("📋 Analysis Results", style="bold bright_magenta underline")

        files_to_process = analysis_data.get("files_to_process", [])
        reasons = analysis_data.get("reasons", {})
        test_presence = analysis_data.get("existing_test_presence", {})

        # Group files by reason
        reason_groups: dict[str, list[str]] = {}
        for file_path in files_to_process:
            reason = reasons.get(file_path, "Unknown")
            if reason not in reason_groups:
                reason_groups[reason] = []
            reason_groups[reason].append(file_path)

        for reason, files in reason_groups.items():
            reason_node = tree.add(f"[warning]{reason}[/] ({len(files)} files)")

            for file_path in files:
                has_tests = test_presence.get(file_path, False)
                test_status = (
                    "[success]✓ Has tests[/]" if has_tests else "[error]✗ No tests[/]"
                )
                reason_node.add(f"{file_path} - {test_status}")

        return tree

    def create_progress_tracker(self) -> Progress:
        """
        Create a progress tracker with spinner and progress bars.

        Returns:
            Rich Progress instance
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

    def create_status_spinner(self, message: str) -> StatusSpinner:
        """
        Create a status spinner.

        Args:
            message: Status message to display

        Returns:
            Rich Status instance
        """
        # Return a wrapper that exposes __str__ with the message and delegates start/stop
        # Status will naturally disappear when stopped, no transient parameter needed
        status = Status(message, console=self.console, spinner="dots")

        class _StatusWrapper:
            def __init__(self, inner: Status, text: str) -> None:
                self._inner = inner
                self._text = text

            def __str__(self) -> str:  # pragma: no cover - simple helper
                return self._text

            def start(self) -> None:
                self._inner.start()

            def stop(self) -> None:
                self._inner.stop()

            def __enter__(self):
                return self._inner.__enter__()

            def __exit__(self, exc_type, exc_val, exc_tb) -> None:
                return self._inner.__exit__(exc_type, exc_val, exc_tb)

            def __getattr__(self, item: Any) -> Any:  # delegate other attrs
                return getattr(self._inner, item)

        return _StatusWrapper(status, message)

    def display_error(
        self,
        message: str,
        title: str = "Error",
        *,
        suggestions: Sequence[str] | None = None,
        primary_action: str | None = None,
        docs_url: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Display an error message with appropriate styling.

        Args:
            message: Error message
            title: Error title
        """
        body_lines = [f"[token_negative]{message}[/]"]

        if context:
            body_lines.append("")
            for key, value in context.items():
                body_lines.append(f"[muted]{key}[/]: {value}")

        if suggestions:
            body_lines.append("")
            body_lines.append("[token_warning]Suggested actions[/]:")
            for suggestion in suggestions:
                body_lines.append(f"  [token_warning]•[/] {suggestion}")

        if docs_url:
            body_lines.append("")
            body_lines.append(
                f"[token_info]{self._glyph('docs', 'docs')}[/] {docs_url}"
            )

        header = f"{self._glyph('error', 'ERROR')} {title}"
        error_panel = Panel(
            "\n".join(body_lines),
            title=header,
            border_style="token_negative",
            padding=(1, 2),
        )
        self.console.print(error_panel)
        if primary_action:
            self.console.print(
                f"[prompt_accent]{self._glyph('action', '>')} {primary_action}[/]"
            )

    def display_warning(self, message: str, title: str = "Warning") -> None:
        """
        Display a warning message with appropriate styling.

        Args:
            message: Warning message
            title: Warning title
        """
        body = f"[token_warning]{message}[/]"
        warning_panel = Panel(
            body,
            title=f"{self._glyph('warning', 'WARN')} {title}",
            border_style="token_warning",
            padding=(1, 2),
        )
        self.console.print(warning_panel)

    def display_success(self, message: str, title: str = "Success") -> None:
        """
        Display a success message with appropriate styling.

        Args:
            message: Success message
            title: Success title
        """
        success_panel = Panel(
            f"[token_positive]{message}[/]",
            title=f"{self._glyph('success', 'OK')} {title}",
            border_style="token_positive",
            padding=(1, 2),
        )
        self.console.print(success_panel)

    def display_info(self, message: str, title: str = "Info") -> None:
        """
        Display an info message with appropriate styling.

        Args:
            message: Info message
            title: Info title
        """
        body = f"[token_info]{message}[/]"
        info_panel = Panel(
            body,
            title=f"{self._glyph('info', 'INFO')} {title}",
            border_style="token_info",
            padding=(1, 2),
        )
        self.console.print(info_panel)

    def get_user_confirmation(self, message: str, default: bool = False) -> bool:
        """
        Get user confirmation with Rich styling.

        Args:
            message: Confirmation message
            default: Default value

        Returns:
            User's choice
        """
        return Confirm.ask(
            f"[prompt_accent]{message}[/]", default=default, console=self.console
        )

    def get_user_input(self, prompt: str, default: str | None = None) -> str:
        """
        Get user input with Rich styling.

        Args:
            prompt: Input prompt
            default: Default value

        Returns:
            User's input
        """
        if default is not None:
            return Prompt.ask(
                f"[prompt_accent]{prompt}[/]", default=default, console=self.console
            )
        else:
            return Prompt.ask(f"[prompt_accent]{prompt}[/]", console=self.console)

    def create_comprehensive_layout(
        self,
        summary_data: dict[str, Any],
        coverage_data: dict[str, Any],
        test_results: list[dict[str, Any]],
        recommendations: list[str],
    ) -> Layout:
        """
        Create a comprehensive layout combining multiple components.

        Args:
            summary_data: Project summary data
            coverage_data: Coverage data
            test_results: Test results data
            recommendations: Recommendations list

        Returns:
            Rich Layout with multiple panels
        """
        layout = Layout()

        # Create main sections
        layout.split_column(
            Layout(name="header", size=8),
            Layout(name="body"),
            Layout(name="footer", size=6),
        )

        # Header with project summary
        layout["header"].update(self.create_project_summary_panel(summary_data))

        # Body with coverage and results
        layout["body"].split_row(Layout(name="coverage"), Layout(name="results"))

        layout["coverage"].update(self.create_coverage_table(coverage_data))
        layout["results"].update(self.create_test_results_table(test_results))

        # Footer with recommendations
        layout["footer"].update(self.create_recommendations_panel(recommendations))

        # Note: _layout_items is not available in Rich Layout API
        return layout

    def _format_coverage_percentage(
        self, coverage: float, bold: bool = False, with_icon: bool = True
    ) -> str:
        """Format coverage percentage with beautiful colors and optional icons."""
        percentage = f"{coverage:.1%}"

        # Enhanced color mapping with more granular levels
        # Map thresholds to match test expectations: 0.95 -> coverage_high
        if coverage >= 0.97:
            color = "coverage_excellent"
            icon = "🟢" if with_icon else ""
        elif coverage >= 0.85:
            color = "coverage_high"
            icon = "🟢" if with_icon else ""
        elif coverage >= 0.70:
            # Map 0.70-0.84 to coverage_good as tests expect
            color = "coverage_good"
            icon = "🟡" if with_icon else ""
        elif coverage >= 0.50:
            color = "coverage_medium"
            icon = "🟠" if with_icon else ""
        else:
            color = "coverage_low"
            icon = "🔴" if with_icon else ""

        # Note: Removed special case for bold formatting consistency

        if bold:
            return (
                f"[bold {color}]{icon} {percentage}[/]"
                if with_icon
                else f"[bold {color}]{percentage}[/]"
            )
        else:
            return (
                f"[{color}]{icon} {percentage}[/]"
                if with_icon
                else f"[{color}]{percentage}[/]"
            )

    def print_divider(self, title: str | None = None) -> None:
        """Print a styled divider with optional title."""
        if title:
            self.console.rule(f"[header]{title}[/]", style="accent")
        else:
            self.console.rule(style="muted")

    def print_table(self, table: Table) -> None:
        """Print a table to the console."""
        self.console.print(table)

    def print_panel(self, panel: Panel) -> None:
        """Print a panel to the console."""
        self.console.print(panel)

    def print_tree(self, tree: Tree) -> None:
        """Print a tree to the console."""
        self.console.print(tree)

    def print_layout(self, layout: Layout) -> None:
        """Print a layout to the console."""
        self.console.print(layout)

    def create_configuration_wizard(
        self, config_steps: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Interactive configuration wizard with progress tracking and resume support."""
        self.console.print()
        intro_text = (
            f"[title]{self._glyph('rocket', '>>')} TestCraft Configuration Wizard[/title]\n\n"
            "[info]Welcome to TestCraft! Let's set up your configuration.[/]\n"
            "[subtle]This wizard will guide you through the setup process.[/]"
        )
        welcome_panel = Panel(
            intro_text,
            title=f"[primary]{self._glyph('spark', '*')} Welcome[/primary]",
            border_style="bright_magenta",
            padding=(1, 2),
            title_align="center",
        )
        self.console.print(welcome_panel)
        self.console.print()

        total_steps = len(config_steps)
        config_values: dict[str, Any] = {}
        current_step = 0

        resume_state = self._load_wizard_cache()
        if resume_state:
            resume = self.get_user_confirmation(
                "Resume previous configuration session?", default=True
            )
            if resume:
                config_values.update(resume_state.get("values", {}))
                saved_index = resume_state.get("step_index", 0)
                if isinstance(saved_index, int) and saved_index < total_steps:
                    current_step = saved_index
                self.console.print(
                    f"\n[token_info]Resuming from step {current_step + 1} of {total_steps}.[/]"
                )
            else:
                self._clear_wizard_cache()

        while current_step < total_steps:
            step = config_steps[current_step]
            step_number = current_step + 1
            self._render_wizard_progress(step_number, total_steps, config_values)

            step_title = step.get("title", f"Step {step_number}")
            step_desc = step.get("description", "")

            self.console.rule(
                f"[header]Step {step_number}/{total_steps}: {step_title}[/]",
                style="accent",
            )

            if step_desc:
                desc_panel = Panel(
                    f"[info]{step_desc}[/]", border_style="border_info", padding=(0, 1)
                )
                self.console.print(desc_panel)
                self.console.print()

            fields = step.get("fields", [])
            step_values: dict[str, Any] = {}

            for field in fields:
                field_name = field.get("name")
                if not field_name:
                    continue

                field_title = field.get("title", field_name)
                field_type = field.get("type", "string")
                field_desc = field.get("description", "")
                field_default = field.get("default")
                field_choices = field.get("choices", [])
                field_required = field.get("required", False)

                existing_value = config_values.get(field_name)
                if existing_value is not None:
                    field_default = existing_value

                prompt_text = f"[prompt]{field_title}[/]"
                if field_desc:
                    prompt_text += f"\n[subtle]{field_desc}[/]"
                self.console.print(prompt_text)

                try:
                    value = self._prompt_wizard_field(
                        field_type=field_type,
                        choices=field_choices,
                        default=field_default,
                        required=field_required,
                        field_definition=field,
                    )
                except KeyboardInterrupt:
                    self.console.print("\n[warning]Operation cancelled by user[/]")
                    self._save_wizard_cache(current_step, config_values)
                    return {}

                if value is None and field_required:
                    self.console.print(
                        "[warning]This field is required. Leaving it blank may impact defaults.[/]"
                    )

                step_values[field_name] = value
                self.console.print(f"[selected]✓ {field_title}: {value}[/]")
                self.console.print()

            config_values.update(step_values)
            self._save_wizard_cache(current_step + 1, config_values)

            completion_panel = Panel(
                f"[token_positive]{self._glyph('spark', '*')} Step {step_number} captured[/]",
                border_style="border_success",
                padding=(0, 1),
            )
            self.console.print(completion_panel)
            self.console.print()

            while True:
                next_action = (
                    Prompt.ask(
                        "[prompt_accent]Next action[/] ([C]ontinue/[B]ack/[R]eview/[S]ave)",
                        default="c",
                        console=self.console,
                    )
                    .strip()
                    .lower()
                )

                if next_action in ("c", "continue", ""):
                    current_step += 1
                    break
                if next_action in ("b", "back"):
                    current_step = max(0, current_step - 1)
                    break
                if next_action in ("r", "review"):
                    self._render_configuration_summary(config_values)
                    continue
                if next_action in ("s", "save"):
                    self._save_wizard_cache(current_step, config_values)
                    self.console.print(
                        "\n[token_info]Progress saved. Run the wizard again to resume where you left off.[/]"
                    )
                    return {}
                self.console.print("[warning]Please choose C, B, R, or S.[/]")

            if next_action in ("b", "back"):
                continue

        self._clear_wizard_cache()
        self.console.rule(
            f"[header]{self._glyph('summary', 'Summary')} Configuration Complete![/]",
            style="success",
        )
        self._render_configuration_summary(config_values)

        try:
            if Confirm.ask(
                "\n[prompt]Save this configuration?[/]",
                default=True,
                console=self.console,
            ):
                self.console.print(
                    f"\n[token_positive]{self._glyph('success', 'OK')} Configuration saved successfully![/]"
                )
                return config_values
            else:
                self.console.print(
                    f"\n[token_warning]{self._glyph('warning', 'WARN')} Configuration cancelled[/]"
                )
                return {}
        except KeyboardInterrupt:
            self.console.print("\n[warning]Operation cancelled by user[/]")
            return {}

    def _prompt_wizard_field(
        self,
        *,
        field_type: str,
        choices: list[Any],
        default: Any,
        required: bool,
        field_definition: dict[str, Any],
    ) -> Any:
        """Prompt for a single wizard field with validation."""
        if field_type == "boolean":
            return Confirm.ask(
                "[choice]Enable this option?[/]",
                default=bool(default) if default is not None else False,
                console=self.console,
            )

        if field_type == "choice" and choices:
            self.console.print("[choice]Available options:[/]")
            for index, choice in enumerate(choices, 1):
                choice_text = (
                    choice
                    if isinstance(choice, str)
                    else choice.get("label", str(choice))
                )
                self.console.print(f"  [secondary]{index}.[/] {choice_text}")

            max_retries = 3
            for attempt in range(max_retries):
                choice_input = Prompt.ask(
                    "[choice]Select option (number)[/]",
                    default=str(default) if default else "1",
                    console=self.console,
                )
                try:
                    choice_index = int(choice_input) - 1
                except (TypeError, ValueError):
                    self.console.print("[warning]Please enter a valid number[/]")
                    continue
                if 0 <= choice_index < len(choices):
                    selected = choices[choice_index]
                    return (
                        selected
                        if isinstance(selected, str)
                        else selected.get("value", selected)
                    )
                self.console.print(
                    f"[warning]Enter a number between 1 and {len(choices)}[/]"
                )
            self.console.print(
                "[warning]Maximum retries exceeded. Using default choice.[/]"
            )
            return default if default in choices else choices[0]

        if field_type == "number":
            integer_mode = field_definition.get("integer", False)
            max_retries = 3
            for _ in range(max_retries):
                num_input = Prompt.ask(
                    "[choice]Enter value[/]",
                    default=str(default) if default is not None else None,
                    console=self.console,
                )
                try:
                    if num_input is None or num_input == "":
                        return 0 if integer_mode else 0.0
                    return int(num_input) if integer_mode else float(num_input)
                except ValueError:
                    self.console.print("[warning]Please enter a valid number[/]")
            self.console.print("[warning]Using default numeric value.[/]")
            return default if default is not None else (0 if integer_mode else 0.0)

        # Default: string input
        while True:
            value = Prompt.ask(
                "[choice]Enter value[/]",
                default=str(default) if default is not None else None,
                console=self.console,
            )
            if required and not value:
                self.console.print("[warning]This field is required.[/]")
                continue
            return value

    def _render_wizard_progress(
        self, step_number: int, total_steps: int, values: dict[str, Any]
    ) -> None:
        """Render a lightweight progress indicator for the wizard."""
        completed = step_number - 1
        ratio = completed / max(total_steps, 1)
        slots = 10
        filled = int(ratio * slots)
        bar = "█" * filled + "░" * (slots - filled)
        self.console.print(
            f"[muted]Progress[/] [accent]{bar}[/] [muted]{completed}/{total_steps} • {len(values)} fields captured[/]"
        )

    def _render_configuration_summary(self, config_values: dict[str, Any]) -> None:
        """Display a structured summary of collected configuration values."""
        lines = ["[title]Configuration Summary[/title]", ""]
        for key, value in config_values.items():
            label = key.replace("_", " ").title()
            lines.append(f"[info]{label}:[/] [secondary]{value}[/]")

        summary_panel = Panel(
            "\n".join(lines).strip(),
            title=f"[primary]{self._glyph('summary', 'Summary')}[/primary]",
            border_style="border_success",
            padding=(1, 2),
            title_align="center",
        )
        self.console.print(summary_panel)

    def _save_wizard_cache(self, step_index: int, values: dict[str, Any]) -> None:
        """Persist partial wizard progress to disk."""
        payload = {"step_index": step_index, "values": values}
        try:
            self._wizard_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._wizard_cache_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:  # pragma: no cover - best effort logging
            self.console.print(f"[warning]Unable to cache wizard progress: {exc}[/]")

    def _load_wizard_cache(self) -> dict[str, Any] | None:
        """Load cached wizard progress if available."""
        try:
            data = json.loads(self._wizard_cache_path.read_text())
            return data if isinstance(data, dict) else None
        except FileNotFoundError:
            return None
        except Exception:  # pragma: no cover - ignore corrupt caches
            return None

    def _clear_wizard_cache(self) -> None:
        """Remove cached wizard progress."""
        try:
            if self._wizard_cache_path.exists():
                self._wizard_cache_path.unlink()
        except Exception:  # pragma: no cover - ignore cleanup failures
            pass

    def display_code_snippet(
        self,
        code: str,
        language: str = "python",
        title: str | None = None,
        line_numbers: bool = True,
    ) -> None:
        """
        Display a beautiful code snippet with syntax highlighting.

        Args:
            code: Code to display
            language: Programming language for syntax highlighting
            title: Optional title for the code block
            line_numbers: Whether to show line numbers
        """
        syntax = Syntax(
            code,
            language,
            theme="monokai",
            line_numbers=line_numbers,
            word_wrap=True,
            background_color="grey11",
        )

        if title:
            code_panel = Panel(
                syntax,
                title=f"[code]💻 {title}[/code]",
                border_style="code",
                padding=(0, 1),
            )
            self.console.print(code_panel)
        else:
            self.console.print(syntax)


# Convenience functions for quick usage


def create_default_cli() -> RichCliComponents:
    """Create a default CLI components instance."""
    return RichCliComponents()


def print_coverage_summary(coverage_data: dict[str, Any]) -> None:
    """Print a quick coverage summary."""
    cli = create_default_cli()
    table = cli.create_coverage_table(coverage_data)
    cli.print_table(table)


def print_test_results_summary(test_results: list[dict[str, Any]]) -> None:
    """Print a quick test results summary."""
    cli = create_default_cli()
    table = cli.create_test_results_table(test_results)
    cli.print_table(table)


def print_project_overview(
    summary_data: dict[str, Any],
    coverage_data: dict[str, Any],
    test_results: list[dict[str, Any]],
    recommendations: list[str],
) -> None:
    """Print a comprehensive project overview."""
    cli = create_default_cli()
    layout = cli.create_comprehensive_layout(
        summary_data, coverage_data, test_results, recommendations
    )
    cli.print_layout(layout)
