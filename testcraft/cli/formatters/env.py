"""Environment and debug information formatters."""

from typing import Any

from rich.console import Console


class EnvironmentFormatter:
    """Formatter for environment and debug information."""

    def __init__(self, console: Console):
        """Initialize with console instance."""
        self.console = console

    def format_environment_info(self, results: dict[str, Any]) -> None:
        """Display environment information."""
        env_info = results.get("environment_info", {})

        for section_name, section_data in env_info.items():
            if isinstance(section_data, dict) and section_data:
                self.console.print(f"\n[bold]{section_name.title()}:[/]")
                for key, value in section_data.items():
                    if key != "variables":  # Skip large env var dumps
                        self.console.print(f"  {key}: {value}")

    def format_cost_summary(self, results: dict[str, Any]) -> None:
        """Display cost summary information."""
        from ...adapters.io.ui_rich import RichUIAdapter
        
        ui = RichUIAdapter(self.console)
        cost_summary = results.get("cost_summary", {})
        projections = results.get("projections", {})
        limit_status = results.get("limit_status", {})

        if cost_summary:
            ui.display_info(
                f"Total cost: ${cost_summary.get('total_cost', 0):.2f}", "Cost Summary"
            )

        if projections:
            ui.display_info(
                f"Projected monthly: ${projections.get('projected_monthly', 0):.2f}",
                "Cost Projections",
            )

        if not limit_status.get("within_limits", True):
            ui.display_warning(
                "Cost limits exceeded - consider adjusting usage", "Cost Warning"
            )

    def format_debug_state(self, results: dict[str, Any], output_format: str) -> None:
        """Display debug state information."""
        if output_format in ["json", "yaml"]:
            formatted_output = results.get("formatted_output", "")
            if formatted_output:
                self.console.print(formatted_output)
        else:
            # Text format - show key sections
            debug_state = results.get("debug_state", {})
            for section, data in debug_state.items():
                if isinstance(data, dict):
                    self.console.print(f"\n[bold]{section.title()}:[/]")
                    for key, value in list(data.items())[:10]:  # Limit output
                        self.console.print(f"  {key}: {value}")
                    if len(data) > 10:
                        self.console.print(f"  ... and {len(data) - 10} more items")
