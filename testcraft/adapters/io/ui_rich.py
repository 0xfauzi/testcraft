"""
Rich UI adapter implementing the UIPort interface.

This module provides a UIPort implementation using Rich components
for professional CLI output with tables, progress indicators, panels,
and interactive elements.
"""

from enum import Enum
from typing import Any

from rich.console import Console
from rich.progress import Progress
from rich.status import Status

from ...ports.ui_port import UIPort
from .rich_cli import TESTCRAFT_THEME, RichCliComponents


class UIStyle(str, Enum):
    """UI style options for controlling visual complexity and theming."""

    MINIMAL = "minimal"
    CLASSIC = "classic"


class UIError(Exception):
    """Exception raised when UI operations fail."""

    pass


class RichUIAdapter(UIPort):
    """
    Rich UI adapter implementing the UIPort interface.

    This adapter provides a rich CLI experience using the Rich library
    for displaying progress, results, errors, and interactive elements
    with professional styling and theming.
    """

    def __init__(self, console: Console | None = None) -> None:
        """
        Initialize the Rich UI adapter.

        Args:
            console: Optional Rich Console instance (will create one if not provided)
        """
        self._console = console or Console(theme=TESTCRAFT_THEME)
        self.rich_cli = RichCliComponents(self._console)
        self._active_progress: Progress | None = None
        self._active_status: Status | None = None

    def display_progress(
        self,
        progress_data: dict[str, Any],
        progress_type: str = "general",
        **kwargs: Any,
    ) -> None:
        """
        Display progress information to the user.

        Args:
            progress_data: Dictionary containing progress information
            progress_type: Type of progress to display
            **kwargs: Additional display parameters
        """
        try:
            current = progress_data.get("current", 0)
            total = progress_data.get("total", 100)
            message = progress_data.get("message", "Processing...")
            progress_data.get("percentage")

            if progress_type == "spinner":
                # Use spinner for indeterminate progress
                if self._active_status:
                    self._active_status.stop()

                # Minimal mode: transient spinner that does not persist
                if hasattr(self.console, "quiet") and getattr(
                    self.console, "quiet", False
                ):
                    return
                self._active_status = self.rich_cli.create_status_spinner(message)  # type: ignore[assignment]
                try:
                    # If minimal mode is in use, ensure transient=True is respected by rich_cli wrapper
                    if self._active_status:
                        self._active_status.start()
                except Exception:
                    pass

            elif progress_type == "bar" or progress_type == "general":
                # Use progress bar for determinate progress
                if hasattr(self.console, "quiet") and getattr(
                    self.console, "quiet", False
                ):
                    return
                if not self._active_progress:
                    self._active_progress = self.rich_cli.create_progress_tracker()
                    try:
                        self._active_progress.start()
                    except Exception:
                        pass

                # Create or update progress task
                task_id = kwargs.get("task_id")
                if task_id is None:
                    task_id = self._active_progress.add_task(message, total=total)
                    kwargs["task_id"] = task_id
                else:
                    self._active_progress.update(
                        task_id, completed=current, description=message
                    )

            elif progress_type == "coverage":
                # Special handling for coverage progress
                self._display_coverage_progress(progress_data, **kwargs)

            elif progress_type == "test_generation":
                # Special handling for test generation progress
                self._display_test_generation_progress(progress_data, **kwargs)

        except Exception as e:
            raise UIError(f"Failed to display progress: {str(e)}") from e

    def display_results(
        self, results: dict[str, Any], result_type: str = "general", **kwargs: Any
    ) -> None:
        """
        Display results to the user.

        Args:
            results: Dictionary containing results to display
            result_type: Type of results to display
            **kwargs: Additional display parameters
        """
        try:
            # Stop any active progress indicators
            self._stop_progress_indicators()

            success = results.get("success", True)
            summary = results.get("summary", "")
            details = results.get("details", {})
            results.get("metadata", {})

            if result_type == "coverage":
                self._display_coverage_results(results, **kwargs)

            elif result_type == "test_generation":
                self._display_test_generation_results(results, **kwargs)

            elif result_type == "analysis":
                self._display_analysis_results(results, **kwargs)

            elif result_type == "summary":
                self._display_summary_results(results, **kwargs)

            else:
                # Generic results display
                if success:
                    if summary:
                        self.rich_cli.display_success(summary, "Results")
                    if details and kwargs.get("show_details", True):
                        for key, value in details.items():
                            self.console.print(f"[info]{key}:[/] {value}")
                else:
                    error_msg = results.get("error_message", "Operation failed")
                    self.rich_cli.display_error(error_msg, "Operation Failed")

        except Exception as e:
            raise UIError(f"Failed to display results: {str(e)}") from e

    def display_error(
        self, error_message: str, error_type: str = "general", **kwargs: Any
    ) -> None:
        """
        Display error information to the user.

        Args:
            error_message: Error message to display
            error_type: Type of error
            **kwargs: Additional error display parameters
        """
        try:
            # Stop any active progress indicators
            self._stop_progress_indicators()

            # Prefer explicit title, otherwise use provided error_type if meaningful
            provided_title = kwargs.get("title")
            title = provided_title or (
                error_type if error_type and error_type != "general" else "Error"
            )
            details = kwargs.get("details")

            self.rich_cli.display_error(error_message, title)

            if details:
                if isinstance(details, dict):
                    for key, value in details.items():
                        self.console.print(f"[muted]{key}:[/] {value}")
                elif isinstance(details, list):
                    for item in details:
                        self.console.print(f"[muted]• {item}[/]")
                else:
                    self.console.print(f"[muted]{details}[/]")

        except Exception as e:
            # Wrap errors in UIError for consistent test expectations
            raise UIError(f"Failed to display error: {str(e)}") from e

    def display_warning(
        self, warning_message: str, warning_type: str = "general", **kwargs: Any
    ) -> None:
        """
        Display warning information to the user.

        Args:
            warning_message: Warning message to display
            warning_type: Type of warning
            **kwargs: Additional warning display parameters
        """
        try:
            provided_title = kwargs.get("title")
            title = provided_title or (
                warning_type
                if warning_type and warning_type != "general"
                else "Warning"
            )
            self.rich_cli.display_warning(warning_message, title)
        except Exception as e:
            raise UIError(f"Failed to display warning: {str(e)}") from e

    def display_info(
        self, info_message: str, info_type: str = "general", **kwargs: Any
    ) -> None:
        """
        Display informational message to the user.

        Args:
            info_message: Information message to display
            info_type: Type of information
            **kwargs: Additional info display parameters
        """
        try:
            provided_title = kwargs.get("title")
            title = provided_title or (
                info_type if info_type and info_type != "general" else "Info"
            )
            self.rich_cli.display_info(info_message, title)
        except Exception as e:
            raise UIError(f"Failed to display info: {str(e)}") from e

    def get_user_input(
        self, prompt: str, input_type: str = "string", **kwargs: Any
    ) -> Any:
        """
        Get input from the user.

        Args:
            prompt: Prompt to display to the user
            input_type: Type of input expected (string, number, boolean, etc.)
            **kwargs: Additional input parameters

        Returns:
            User input value of the specified type
        """
        try:
            default = kwargs.get("default")

            if input_type == "boolean":
                return self.rich_cli.get_user_confirmation(prompt, default or False)

            elif input_type == "number":
                while True:
                    try:
                        response = self.rich_cli.get_user_input(
                            prompt, str(default) if default else None
                        )
                        if input_type == "int":
                            return int(response)
                        else:
                            return float(response)
                    except ValueError:
                        self.rich_cli.display_error("Please enter a valid number.")

            elif input_type == "choice":
                choices = kwargs.get("choices", [])
                if not choices:
                    raise UIError("Choices must be provided for choice input type")

                # Display choices
                self.console.print(f"[highlight]{prompt}[/]")
                for i, choice in enumerate(choices, 1):
                    self.console.print(f"  {i}. {choice}")

                while True:
                    try:
                        response = self.rich_cli.get_user_input(
                            "Enter choice number", "1"
                        )
                        choice_index = int(response) - 1
                        if 0 <= choice_index < len(choices):
                            return choices[choice_index]
                        else:
                            self.rich_cli.display_error(
                                f"Please enter a number between 1 and {len(choices)}"
                            )
                    except ValueError:
                        self.rich_cli.display_error("Please enter a valid number")

            else:  # string or default
                return self.rich_cli.get_user_input(
                    prompt, str(default) if default else None
                )

        except Exception as e:
            raise UIError(f"Failed to get user input: {str(e)}") from e

    def confirm_action(
        self, message: str, default: bool = False, **kwargs: Any
    ) -> bool:
        """
        Get confirmation from the user for an action.

        Args:
            message: Message to display for confirmation
            default: Default value if user doesn't respond
            **kwargs: Additional confirmation parameters

        Returns:
            True if user confirms, False otherwise
        """
        try:
            return self.rich_cli.get_user_confirmation(message, default)
        except Exception as e:
            raise UIError(f"Failed to get user confirmation: {str(e)}") from e

    def get_user_confirmation(
        self, message: str, default: bool = False, **kwargs: Any
    ) -> bool:
        """
        Get user confirmation (alias for confirm_action).

        Args:
            message: Message to display for confirmation
            default: Default value if user doesn't respond
            **kwargs: Additional confirmation parameters

        Returns:
            True if user confirms, False otherwise
        """
        return self.confirm_action(message, default, **kwargs)

    def display_success(
        self, message: str, title: str = "Success", **kwargs: Any
    ) -> None:
        """
        Display success message to the user.

        Args:
            message: Success message to display
            title: Title for the success message
            **kwargs: Additional display parameters

        Raises:
            UIError: If success display fails
        """
        try:
            self.rich_cli.display_success(message, title)
        except Exception as e:
            raise UIError(f"Failed to display success message: {str(e)}") from e

    def print_divider(self, title: str | None = None) -> None:
        """
        Print a divider line with optional title.

        Args:
            title: Optional title for the divider
        """
        self.rich_cli.print_divider(title)

    @property
    def console(self) -> Any:
        """
        Provide access to the underlying Rich console.

        Returns:
            The Rich console instance
        """
        return self._console

    def _display_coverage_progress(
        self, progress_data: dict[str, Any], **kwargs: Any
    ) -> None:
        """Display coverage-specific progress."""
        files_processed = progress_data.get("files_processed", 0)
        total_files = progress_data.get("total_files", 1)
        current_file = progress_data.get("current_file", "")

        message = (
            f"Analyzing coverage: {current_file}"
            if current_file
            else "Analyzing coverage"
        )

        if not self._active_progress:
            self._active_progress = self.rich_cli.create_progress_tracker()
            self._active_progress.start()

        task_id = kwargs.get("task_id")
        if task_id is None:
            task_id = self._active_progress.add_task(message, total=total_files)
            kwargs["task_id"] = task_id
        else:
            self._active_progress.update(
                task_id, completed=files_processed, description=message
            )

    def _display_test_generation_progress(
        self, progress_data: dict[str, Any], **kwargs: Any
    ) -> None:
        """Display test generation specific progress."""
        tests_generated = progress_data.get("tests_generated", 0)
        total_tests = progress_data.get("total_tests", 1)
        current_test = progress_data.get("current_test", "")

        message = (
            f"Generating tests: {current_test}" if current_test else "Generating tests"
        )

        if not self._active_progress:
            self._active_progress = self.rich_cli.create_progress_tracker()
            self._active_progress.start()

        task_id = kwargs.get("task_id")
        if task_id is None:
            task_id = self._active_progress.add_task(message, total=total_tests)
            kwargs["task_id"] = task_id
        else:
            self._active_progress.update(
                task_id, completed=tests_generated, description=message
            )

    def _display_coverage_results(self, results: dict[str, Any], **kwargs: Any) -> None:
        """Display coverage-specific results."""
        coverage_data = results.get("details", {})
        summary = results.get("summary", "")

        if summary:
            self.rich_cli.display_success(summary, "Coverage Analysis Complete")

        if coverage_data:
            table = self.rich_cli.create_coverage_table(coverage_data)
            self.rich_cli.print_table(table)

    def _display_test_generation_results(
        self, results: dict[str, Any], **kwargs: Any
    ) -> None:
        """Display test generation specific results."""
        test_results = results.get("details", {}).get("test_results", [])
        summary = results.get("summary", "")

        if summary:
            self.rich_cli.display_success(summary, "Test Generation Complete")

        if test_results:
            table = self.rich_cli.create_test_results_table(test_results)
            self.rich_cli.print_table(table)

    def _display_analysis_results(self, results: dict[str, Any], **kwargs: Any) -> None:
        """Display analysis-specific results."""
        analysis_data = results.get("details", {})
        summary = results.get("summary", "")

        if summary:
            self.rich_cli.display_success(summary, "Analysis Complete")

        if analysis_data:
            tree = self.rich_cli.create_analysis_tree(analysis_data)
            self.rich_cli.print_tree(tree)

    def _display_summary_results(self, results: dict[str, Any], **kwargs: Any) -> None:
        """Display project summary results."""
        summary_data = results.get("details", {}).get("summary_data", {})
        coverage_data = results.get("details", {}).get("coverage_data", {})
        test_results = results.get("details", {}).get("test_results", [])
        recommendations = results.get("details", {}).get("recommendations", [])

        if kwargs.get("layout_mode", False):
            # Use comprehensive layout
            layout = self.rich_cli.create_comprehensive_layout(
                summary_data, coverage_data, test_results, recommendations
            )
            self.rich_cli.print_layout(layout)
        else:
            # Display components separately
            if summary_data:
                panel = self.rich_cli.create_project_summary_panel(summary_data)
                self.rich_cli.print_panel(panel)

            if coverage_data:
                table = self.rich_cli.create_coverage_table(coverage_data)
                self.rich_cli.print_table(table)

            if test_results:
                table = self.rich_cli.create_test_results_table(test_results)
                self.rich_cli.print_table(table)

            if recommendations:
                panel = self.rich_cli.create_recommendations_panel(recommendations)
                self.rich_cli.print_panel(panel)

    def _stop_progress_indicators(self) -> None:
        """Stop any active progress indicators."""
        if self._active_progress:
            self._active_progress.stop()
            self._active_progress = None

        if self._active_status:
            self._active_status.stop()
            self._active_status = None

    def finalize(self) -> None:
        """Finalize the UI and clean up resources."""
        self._stop_progress_indicators()

    def set_quiet_mode(self, quiet: bool) -> None:
        """Set quiet mode for minimal output."""
        if quiet:
            self.console.quiet = True
        else:
            self.console.quiet = False

    def run_configuration_wizard(
        self, config_steps: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Run the interactive configuration wizard.

        Args:
            config_steps: List of configuration steps

        Returns:
            Dictionary of configuration values
        """
        try:
            return self.rich_cli.create_configuration_wizard(config_steps)
        except Exception as e:
            raise UIError(f"Configuration wizard failed: {str(e)}") from e

    def display_code_snippet(
        self,
        code: str,
        language: str = "python",
        title: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Display a code snippet with beautiful syntax highlighting.

        Args:
            code: Code to display
            language: Programming language for syntax highlighting
            title: Optional title for the code block
            **kwargs: Additional display parameters
        """
        try:
            line_numbers = kwargs.get("line_numbers", True)
            self.rich_cli.display_code_snippet(code, language, title, line_numbers)
        except Exception as e:
            raise UIError(f"Failed to display code snippet: {str(e)}") from e

    def create_beautiful_summary(
        self, data: dict[str, Any], layout_style: str = "comprehensive"
    ) -> None:
        """
        Create a beautiful comprehensive summary display.

        Args:
            data: Data to display in summary
            layout_style: Style of layout ('comprehensive', 'simple', 'minimal')
        """
        try:
            summary_data = data.get("summary_data", {})
            coverage_data = data.get("coverage_data", {})
            test_results = data.get("test_results", [])
            recommendations = data.get("recommendations", [])

            if layout_style == "comprehensive":
                layout = self.rich_cli.create_comprehensive_layout(
                    summary_data, coverage_data, test_results, recommendations
                )
                self.rich_cli.print_layout(layout)
            else:
                # Display components separately with beautiful styling
                if summary_data:
                    panel = self.rich_cli.create_project_summary_panel(summary_data)
                    self.rich_cli.print_panel(panel)

                if coverage_data:
                    table = self.rich_cli.create_coverage_table(coverage_data)
                    self.rich_cli.print_table(table)

                if test_results:
                    table = self.rich_cli.create_test_results_table(test_results)
                    self.rich_cli.print_table(table)

                if recommendations:
                    panel = self.rich_cli.create_recommendations_panel(recommendations)
                    self.rich_cli.print_panel(panel)

        except Exception as e:
            raise UIError(f"Failed to create summary display: {str(e)}") from e

    def create_status_spinner(self, message: str) -> Status:
        """
        Create a status spinner for displaying ongoing operations.

        Args:
            message: Message to display with the spinner

        Returns:
            Rich Status instance
        """
        return self.rich_cli.create_status_spinner(message)  # type: ignore[return-value]
