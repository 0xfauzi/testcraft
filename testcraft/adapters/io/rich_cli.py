"""
Rich CLI components for testcraft.

This module provides Rich-based UI components for creating professional
CLI output including tables, progress indicators, summaries, and themed layouts.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.text import Text
from rich.tree import Tree
from rich.status import Status
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.columns import Columns
from rich.align import Align
from rich import box
from rich.theme import Theme
from rich.markup import escape
from rich.rule import Rule


# Define enhanced testcraft theme with beautiful colors and visual hierarchy
TESTCRAFT_THEME = Theme({
    # Core status colors
    "success": "bold bright_green",
    "error": "bold bright_red", 
    "warning": "bold gold1",
    "info": "bold deep_sky_blue1",
    
    # Accent and highlight colors
    "highlight": "bold bright_cyan",
    "accent": "bright_blue",
    "primary": "bold bright_magenta",
    "secondary": "bright_white",
    
    # Text hierarchy
    "header": "bold bright_magenta underline",
    "subheader": "bold bright_blue",
    "title": "bold white",
    "muted": "dim bright_black",
    "subtle": "grey70",
    
    # Coverage-specific colors with better gradients
    "coverage_excellent": "bold green1", # 95%+
    "coverage_high": "bold green3",      # 85-94%
    "coverage_good": "bold yellow1",     # 70-84%
    "coverage_medium": "bold orange1",   # 50-69%
    "coverage_low": "bold red1",         # < 50%
    
    # Status indicators
    "status_pass": "bold green1",
    "status_fail": "bold red1", 
    "status_skip": "bold yellow1",
    "status_partial": "bold orange1",
    
    # Interactive elements
    "prompt": "bold bright_cyan",
    "choice": "bright_white",
    "selected": "bold bright_green",
    
    # Code syntax highlighting base
    "code": "bright_white on grey11",
    "code_keyword": "bold bright_magenta",
    "code_string": "bright_green",
    "code_number": "bright_cyan",
    "code_comment": "dim green",
    
    # Progress and activity
    "progress_bar": "bright_green",
    "progress_complete": "bold bright_green", 
    "activity": "bold bright_yellow",
    
    # Special elements
    "border_success": "green1",
    "border_error": "red1",
    "border_warning": "yellow1", 
    "border_info": "blue1",
    "border_accent": "bright_magenta"
})


class RichCliComponents:
    """
    Rich-based CLI components for testcraft.
    
    Provides methods for creating tables, progress indicators, summaries,
    and other UI elements with consistent theming and professional appearance.
    """
    
    def __init__(self, console: Optional[Console] = None) -> None:
        """
        Initialize Rich CLI components.
        
        Args:
            console: Optional Rich Console instance (will create one if not provided)
        """
        self.console = console or Console(theme=TESTCRAFT_THEME)
    
    def create_coverage_table(
        self, 
        coverage_data: Dict[str, Any],
        show_details: bool = True
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
            show_lines=True
        )
        
        table.add_column("File", style="dim", width=40)
        table.add_column("Line Coverage", justify="center", width=15)
        table.add_column("Branch Coverage", justify="center", width=15)
        if show_details:
            table.add_column("Missing Lines", justify="center", width=15)
            table.add_column("Status", justify="center", width=10)
        
        # Add coverage data rows
        files_data = coverage_data.get('files', {})
        for file_path, file_coverage in files_data.items():
            line_cov = file_coverage.get('line_coverage', 0.0)
            branch_cov = file_coverage.get('branch_coverage', 0.0)
            missing_lines = file_coverage.get('missing_lines', [])
            
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
            file_display = str(Path(file_path).name) if len(file_path) > 35 else file_path
            
            if show_details:
                missing_display = str(len(missing_lines)) if missing_lines else "0"
                table.add_row(file_display, line_cov_text, branch_cov_text, missing_display, status)
            else:
                table.add_row(file_display, line_cov_text, branch_cov_text)
        
        # Add summary row if multiple files
        if len(files_data) > 1:
            overall_line = coverage_data.get('overall_line_coverage', 0.0)
            overall_branch = coverage_data.get('overall_branch_coverage', 0.0)
            
            table.add_section()
            if show_details:
                table.add_row(
                    "[bold]Overall[/]",
                    self._format_coverage_percentage(overall_line, bold=True),
                    self._format_coverage_percentage(overall_branch, bold=True),
                    "-",
                    "[bold]Summary[/]",
                    style="dim"
                )
            else:
                table.add_row(
                    "[bold]Overall[/]",
                    self._format_coverage_percentage(overall_line, bold=True),
                    self._format_coverage_percentage(overall_branch, bold=True),
                    style="dim"
                )
        
        return table
    
    def create_test_results_table(self, test_results: List[Dict[str, Any]]) -> Table:
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
            border_style="border_accent",
            show_lines=True
        )
        
        table.add_column("Source File", style="dim", width=35)
        table.add_column("Test File", style="dim", width=35)
        table.add_column("Status", justify="center", width=12)
        table.add_column("Tests Generated", justify="center", width=15)
        table.add_column("Pass Rate", justify="center", width=10)
        
        for result in test_results:
            # Format status with beautiful colors and icons
            status = result.get('status', 'unknown')
            if status == 'success':
                status_text = "[status_pass]🎉 Success[/]"
            elif status == 'failed':
                status_text = "[status_fail]❌ Failed[/]"
            elif status == 'partial':
                status_text = "[status_partial]⚠️ Partial[/]"
            elif status == 'skipped':
                status_text = "[status_skip]⏭️ Skipped[/]"
            else:
                status_text = "[muted]❓ Unknown[/]"
            
            # Format test count
            tests_generated = result.get('tests_generated', 0)
            test_count_text = str(tests_generated) if tests_generated > 0 else "[muted]0[/]"
            
            # Format pass rate with beautiful colors and icons
            pass_rate = result.get('pass_rate', 0.0)
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
                result.get('source_file', ''),
                result.get('test_file', ''),
                status_text,
                test_count_text,
                pass_rate_text
            )
        
        return table
    
    def create_project_summary_panel(self, project_data: Dict[str, Any]) -> Panel:
        """
        Create a project summary panel.
        
        Args:
            project_data: Project summary data
            
        Returns:
            Formatted Rich Panel
        """
        # Extract key metrics
        total_files = project_data.get('total_files', 0)
        files_with_tests = project_data.get('files_with_tests', 0)
        overall_coverage = project_data.get('overall_coverage', 0.0)
        tests_generated = project_data.get('tests_generated', 0)
        generation_success_rate = project_data.get('generation_success_rate', 0.0)
        
        # Create beautiful content with enhanced formatting
        coverage_text = self._format_coverage_percentage(overall_coverage, with_icon=True)
        
        # Calculate test percentage
        test_percentage = (files_with_tests / total_files * 100) if total_files > 0 else 0
        test_icon = "🏆" if test_percentage >= 90 else "📊" if test_percentage >= 70 else "📈"
        
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
            border_style="border_accent",
            padding=(1, 2),
            title_align="center"
        )
    
    def create_recommendations_panel(self, recommendations: List[str]) -> Panel:
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
            padding=(1, 2)
        )
    
    def create_analysis_tree(self, analysis_data: Dict[str, Any]) -> Tree:
        """
        Create a tree view of analysis results.
        
        Args:
            analysis_data: Analysis data to display
            
        Returns:
            Rich Tree structure
        """
        tree = Tree("📋 Analysis Results", style="header")
        
        files_to_process = analysis_data.get('files_to_process', [])
        reasons = analysis_data.get('reasons', {})
        test_presence = analysis_data.get('existing_test_presence', {})
        
        # Group files by reason
        reason_groups: Dict[str, List[str]] = {}
        for file_path in files_to_process:
            reason = reasons.get(file_path, 'Unknown')
            if reason not in reason_groups:
                reason_groups[reason] = []
            reason_groups[reason].append(file_path)
        
        for reason, files in reason_groups.items():
            reason_node = tree.add(f"[warning]{reason}[/] ({len(files)} files)")
            
            for file_path in files:
                has_tests = test_presence.get(file_path, False)
                test_status = "[success]✓ Has tests[/]" if has_tests else "[error]✗ No tests[/]"
                file_node = reason_node.add(f"{Path(file_path).name} - {test_status}")
        
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
            console=self.console
        )
    
    def create_status_spinner(self, message: str) -> Status:
        """
        Create a status spinner.
        
        Args:
            message: Status message to display
            
        Returns:
            Rich Status instance
        """
        return Status(message, console=self.console, spinner="dots")
    
    def display_error(self, message: str, title: str = "Error") -> None:
        """
        Display an error message with appropriate styling.
        
        Args:
            message: Error message
            title: Error title
        """
        error_panel = Panel(
            f"[error]{message}[/]",
            title=f"❌ {title}",
            border_style="red",
            padding=(1, 2)
        )
        self.console.print(error_panel)
    
    def display_warning(self, message: str, title: str = "Warning") -> None:
        """
        Display a warning message with appropriate styling.
        
        Args:
            message: Warning message
            title: Warning title
        """
        warning_panel = Panel(
            f"[warning]{message}[/]",
            title=f"⚠️  {title}",
            border_style="yellow",
            padding=(1, 2)
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
            f"[success]{message}[/]",
            title=f"✅ {title}",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(success_panel)
    
    def display_info(self, message: str, title: str = "Info") -> None:
        """
        Display an info message with appropriate styling.
        
        Args:
            message: Info message
            title: Info title
        """
        info_panel = Panel(
            f"[info]{message}[/]",
            title=f"ℹ️  {title}",
            border_style="blue",
            padding=(1, 2)
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
        return Confirm.ask(f"[highlight]{message}[/]", default=default, console=self.console)
    
    def get_user_input(self, prompt: str, default: Optional[str] = None) -> str:
        """
        Get user input with Rich styling.
        
        Args:
            prompt: Input prompt
            default: Default value
            
        Returns:
            User's input
        """
        return Prompt.ask(f"[highlight]{prompt}[/]", default=default, console=self.console)
    
    def create_comprehensive_layout(
        self, 
        summary_data: Dict[str, Any],
        coverage_data: Dict[str, Any],
        test_results: List[Dict[str, Any]],
        recommendations: List[str]
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
            Layout(name="footer", size=6)
        )
        
        # Header with project summary
        layout["header"].update(self.create_project_summary_panel(summary_data))
        
        # Body with coverage and results
        layout["body"].split_row(
            Layout(name="coverage"),
            Layout(name="results")
        )
        
        layout["coverage"].update(self.create_coverage_table(coverage_data))
        layout["results"].update(self.create_test_results_table(test_results))
        
        # Footer with recommendations
        layout["footer"].update(self.create_recommendations_panel(recommendations))
        
        return layout
    
    def _format_coverage_percentage(self, coverage: float, bold: bool = False, with_icon: bool = True) -> str:
        """Format coverage percentage with beautiful colors and optional icons."""
        percentage = f"{coverage:.1%}"
        
        # Enhanced color mapping with more granular levels
        if coverage >= 0.95:
            color = "coverage_excellent"
            icon = "🟢" if with_icon else ""
        elif coverage >= 0.85:
            color = "coverage_high" 
            icon = "🟢" if with_icon else ""
        elif coverage >= 0.70:
            color = "coverage_good"
            icon = "🟡" if with_icon else ""
        elif coverage >= 0.50:
            color = "coverage_medium"
            icon = "🟠" if with_icon else ""
        else:
            color = "coverage_low"
            icon = "🔴" if with_icon else ""
        
        if bold:
            return f"[bold {color}]{icon} {percentage}[/]" if with_icon else f"[bold {color}]{percentage}[/]"
        else:
            return f"[{color}]{icon} {percentage}[/]" if with_icon else f"[{color}]{percentage}[/]"
    
    def print_divider(self, title: Optional[str] = None) -> None:
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
    
    def create_configuration_wizard(self, config_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a beautiful interactive configuration wizard.
        
        Args:
            config_steps: List of configuration steps, each containing:
                - title: Step title
                - description: Step description
                - fields: List of field definitions
                - required: Whether step is required
                
        Returns:
            Dictionary of configuration values
        """
        self.console.print()
        welcome_panel = Panel(
            "[title]🚀 TestCraft Configuration Wizard[/title]\n\n"
            "[info]Welcome to TestCraft! Let's set up your configuration.[/]\n"
            "[subtle]This wizard will guide you through the setup process.[/]",
            title="[primary]✨ Welcome[/primary]",
            border_style="border_accent",
            padding=(1, 2),
            title_align="center"
        )
        self.console.print(welcome_panel)
        self.console.print()
        
        config_values = {}
        
        for i, step in enumerate(config_steps, 1):
            # Step header
            step_title = step.get('title', f'Step {i}')
            step_desc = step.get('description', '')
            
            self.console.rule(f"[header]Step {i}: {step_title}[/]", style="accent")
            
            if step_desc:
                desc_panel = Panel(
                    f"[info]{step_desc}[/]",
                    border_style="border_info",
                    padding=(0, 1)
                )
                self.console.print(desc_panel)
            
            self.console.print()
            
            # Process fields
            fields = step.get('fields', [])
            step_values = {}
            
            for field in fields:
                field_name = field.get('name')
                field_title = field.get('title', field_name)
                field_type = field.get('type', 'string')
                field_desc = field.get('description', '')
                field_default = field.get('default')
                field_choices = field.get('choices', [])
                field_required = field.get('required', False)
                
                # Display field info
                prompt_text = f"[prompt]{field_title}[/]"
                if field_desc:
                    prompt_text += f"\n[subtle]{field_desc}[/]"
                
                self.console.print(prompt_text)
                
                # Get user input based on field type
                if field_type == 'boolean':
                    value = Confirm.ask(
                        "[choice]Enable this option?[/]",
                        default=field_default or False,
                        console=self.console
                    )
                
                elif field_type == 'choice' and field_choices:
                    self.console.print("[choice]Available options:[/]")
                    for j, choice in enumerate(field_choices, 1):
                        choice_text = choice if isinstance(choice, str) else choice.get('label', str(choice))
                        self.console.print(f"  [secondary]{j}.[/] {choice_text}")
                    
                    while True:
                        try:
                            choice_input = Prompt.ask(
                                "[choice]Select option (number)[/]",
                                default=str(field_default) if field_default else "1",
                                console=self.console
                            )
                            choice_index = int(choice_input) - 1
                            if 0 <= choice_index < len(field_choices):
                                choice_item = field_choices[choice_index]
                                value = choice_item if isinstance(choice_item, str) else choice_item.get('value', choice_item)
                                break
                            else:
                                self.console.print(f"[error]Please enter a number between 1 and {len(field_choices)}[/]")
                        except ValueError:
                            self.console.print("[error]Please enter a valid number[/]")
                
                elif field_type == 'number':
                    while True:
                        try:
                            num_input = Prompt.ask(
                                "[choice]Enter value[/]",
                                default=str(field_default) if field_default is not None else None,
                                console=self.console
                            )
                            value = int(num_input) if field.get('integer', False) else float(num_input)
                            break
                        except ValueError:
                            self.console.print("[error]Please enter a valid number[/]")
                
                else:  # string type
                    value = Prompt.ask(
                        "[choice]Enter value[/]",
                        default=str(field_default) if field_default is not None else None,
                        console=self.console
                    )
                    
                    if field_required and not value:
                        self.console.print("[error]This field is required[/]")
                        continue
                
                step_values[field_name] = value
                self.console.print(f"[selected]✓ {field_title}: {value}[/]")
                self.console.print()
            
            config_values.update(step_values)
            
            # Step completion
            completion_panel = Panel(
                f"[success]✅ Step {i} completed![/]",
                border_style="border_success",
                padding=(0, 1)
            )
            self.console.print(completion_panel)
            self.console.print()
        
        # Final summary
        self.console.rule("[header]🎉 Configuration Complete![/]", style="success")
        
        summary_text = "[title]Configuration Summary[/title]\n\n"
        for key, value in config_values.items():
            summary_text += f"[info]{key.replace('_', ' ').title()}:[/] [secondary]{value}[/]\n"
        
        summary_panel = Panel(
            summary_text.strip(),
            title="[primary]📋 Summary[/primary]",
            border_style="border_success",
            padding=(1, 2),
            title_align="center"
        )
        self.console.print(summary_panel)
        
        # Confirmation
        if Confirm.ask("\n[prompt]Save this configuration?[/]", default=True, console=self.console):
            self.console.print("\n[success]🎊 Configuration saved successfully![/]")
            return config_values
        else:
            self.console.print("\n[warning]⚠️ Configuration cancelled[/]")
            return {}
    
    def display_code_snippet(
        self, 
        code: str, 
        language: str = "python", 
        title: Optional[str] = None,
        line_numbers: bool = True
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
            background_color="grey11"
        )
        
        if title:
            code_panel = Panel(
                syntax,
                title=f"[code]💻 {title}[/code]",
                border_style="code",
                padding=(0, 1)
            )
            self.console.print(code_panel)
        else:
            self.console.print(syntax)


# Convenience functions for quick usage

def create_default_cli() -> RichCliComponents:
    """Create a default CLI components instance."""
    return RichCliComponents()


def print_coverage_summary(coverage_data: Dict[str, Any]) -> None:
    """Print a quick coverage summary."""
    cli = create_default_cli()
    table = cli.create_coverage_table(coverage_data)
    cli.print_table(table)


def print_test_results_summary(test_results: List[Dict[str, Any]]) -> None:
    """Print a quick test results summary."""
    cli = create_default_cli()
    table = cli.create_test_results_table(test_results)
    cli.print_table(table)


def print_project_overview(
    summary_data: Dict[str, Any],
    coverage_data: Dict[str, Any],
    test_results: List[Dict[str, Any]],
    recommendations: List[str]
) -> None:
    """Print a comprehensive project overview."""
    cli = create_default_cli()
    layout = cli.create_comprehensive_layout(
        summary_data, coverage_data, test_results, recommendations
    )
    cli.print_layout(layout)
