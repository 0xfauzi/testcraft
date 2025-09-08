"""Utility commands for the TestCraft CLI."""

import asyncio
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from ..adapters.io.ui_rich import RichUIAdapter
from .config_init import ConfigInitializer

# Initialize Rich console and UI components
console = Console()
ui = RichUIAdapter(console)


def add_utility_commands(app: click.Group) -> None:
    """Add utility commands to the main CLI app."""

    @app.command()
    @click.option(
        "--format",
        "-f",
        "output_format",
        type=click.Choice(["toml", "yaml", "json"], case_sensitive=False),
        default="toml",
        help="Configuration format",
    )
    @click.option("--minimal", is_flag=True, help="Generate minimal configuration")
    @click.option(
        "--output", "-o", type=click.Path(path_type=Path), help="Output file path"
    )
    @click.pass_context
    def init_config(
        ctx: click.Context, output_format: str, minimal: bool, output: Path | None
    ) -> None:
        """Initialize configuration file with guided setup."""
        try:
            initializer = ConfigInitializer(ui)

            # Create configuration file
            config_file = initializer.create_config_file(
                format_type=output_format, minimal=minimal, output_path=output
            )

            ui.display_success(
                f"Configuration file created: {config_file}",
                "Configuration Initialized",
            )

            # Offer to run guided setup
            if not minimal and ui.get_user_confirmation(
                "Would you like to run guided configuration setup?", default=True
            ):
                initializer.run_guided_setup(config_file)

        except Exception as e:
            ui.display_error(
                f"Configuration initialization failed: {e}", "Init Config Error"
            )
            if ctx.obj.verbose:
                import traceback

                ui.display_info(traceback.format_exc(), "Debug Information")
            sys.exit(1)

    @app.command()
    @click.option("--system", is_flag=True, help="Include system information")
    @click.option("--python", is_flag=True, help="Include Python environment info")
    @click.option("--dependencies", is_flag=True, help="Include dependency information")
    @click.pass_context
    def env(ctx: click.Context, system: bool, python: bool, dependencies: bool) -> None:
        """Show environment information and diagnostics."""
        try:
            # Get use case from container
            utility_usecase = ctx.obj.container["utility_usecase"]

            # If no specific flags, show all info
            if not any([system, python, dependencies]):
                system = python = dependencies = True

            with ui.create_status_spinner("Collecting environment information..."):
                results = asyncio.run(
                    utility_usecase.get_environment_info(
                        include_system_info=system,
                        include_python_info=python,
                        include_dependency_info=dependencies,
                    )
                )

            display_environment_info(results)

        except Exception as e:
            ui.display_error(
                f"Environment info collection failed: {e}", "Environment Error"
            )
            if ctx.obj.verbose:
                import traceback

                ui.display_info(traceback.format_exc(), "Debug Information")
            sys.exit(1)

    @app.command()
    @click.option(
        "--period",
        type=click.Choice(["daily", "weekly", "monthly"]),
        default="monthly",
        help="Time period for cost summary",
    )
    @click.option("--projections", is_flag=True, help="Include cost projections")
    @click.option("--breakdown", is_flag=True, help="Break down costs by service")
    @click.pass_context
    def cost(
        ctx: click.Context, period: str, projections: bool, breakdown: bool
    ) -> None:
        """Show cost summary and projections."""
        try:
            # Get use case from container
            utility_usecase = ctx.obj.container["utility_usecase"]

            with ui.create_status_spinner("Calculating cost summary..."):
                results = asyncio.run(
                    utility_usecase.get_cost_summary(
                        time_period=period,
                        include_projections=projections,
                        breakdown_by_service=breakdown,
                    )
                )

            if results.get("success"):
                display_cost_summary(results)
            else:
                ui.display_warning(
                    results.get("error", "Cost information unavailable"),
                    "Cost Information",
                )

        except Exception as e:
            ui.display_error(f"Cost summary failed: {e}", "Cost Error")
            if ctx.obj.verbose:
                import traceback

                ui.display_info(traceback.format_exc(), "Debug Information")
            sys.exit(1)

    @app.command()
    @click.option("--telemetry", is_flag=True, help="Include telemetry information")
    @click.option("--config", is_flag=True, help="Include configuration")
    @click.option(
        "--format",
        "output_format",
        type=click.Choice(["json", "yaml", "text"]),
        default="text",
        help="Output format",
    )
    @click.pass_context
    def debug_state(
        ctx: click.Context, telemetry: bool, config: bool, output_format: str
    ) -> None:
        """Dump internal state for debugging."""
        try:
            # Get use case from container
            utility_usecase = ctx.obj.container["utility_usecase"]

            ui.display_warning(
                "This command dumps internal state information for debugging purposes.",
                "Debug Information",
            )

            with ui.create_status_spinner("Collecting debug information..."):
                results = asyncio.run(
                    utility_usecase.debug_state(
                        include_telemetry=telemetry,
                        include_config=config,
                        output_format=output_format,
                    )
                )

            display_debug_state(results, output_format)

        except Exception as e:
            ui.display_error(f"Debug state dump failed: {e}", "Debug Error")
            if ctx.obj.verbose:
                import traceback

                ui.display_info(traceback.format_exc(), "Debug Information")
            sys.exit(1)

    @app.command()
    @click.option("--reload", is_flag=True, help="Force reload from storage")
    @click.option(
        "--persist", is_flag=True, help="Persist state after sync", default=True
    )
    @click.pass_context
    def sync_state(ctx: click.Context, reload: bool, persist: bool) -> None:
        """Synchronize state between memory and storage."""
        try:
            # Get use case from container
            utility_usecase = ctx.obj.container["utility_usecase"]

            with ui.create_status_spinner("Synchronizing state..."):
                results = asyncio.run(
                    utility_usecase.sync_state(
                        force_reload=reload, persist_after_sync=persist
                    )
                )

            if results.get("success"):
                ui.display_success(
                    f"State synchronization completed. Operations: {results.get('total_operations', 0)}",
                    "Sync Complete",
                )

                # Show any errors that occurred
                errors = results.get("sync_results", {}).get("errors", [])
                if errors:
                    for error in errors:
                        ui.display_warning(error, "Sync Warning")
            else:
                ui.display_error("State synchronization failed", "Sync Failed")

        except Exception as e:
            ui.display_error(f"State synchronization failed: {e}", "Sync Error")
            if ctx.obj.verbose:
                import traceback

                ui.display_info(traceback.format_exc(), "Debug Information")
            sys.exit(1)

    @app.command()
    @click.option(
        "--categories", multiple=True, help="Specific state categories to reset"
    )
    @click.option(
        "--backup/--no-backup", default=True, help="Create backup before reset"
    )
    @click.option(
        "--confirm",
        is_flag=True,
        required=True,
        help="Required confirmation flag for destructive operation",
    )
    @click.pass_context
    def reset_state(
        ctx: click.Context, categories: tuple[str, ...], backup: bool, confirm: bool
    ) -> None:
        """Clear state and start fresh (DESTRUCTIVE)."""
        try:
            if not confirm:
                ui.display_error(
                    "This command requires --confirm flag as it permanently deletes state data",
                    "Confirmation Required",
                )
                sys.exit(1)

            # Final confirmation
            if not ui.get_user_confirmation(
                "This will permanently delete state data. Are you absolutely sure?",
                default=False,
            ):
                ui.display_info("Reset operation cancelled by user", "Cancelled")
                return

            # Get use case from container
            utility_usecase = ctx.obj.container["utility_usecase"]

            with ui.create_status_spinner("Resetting state..."):
                results = asyncio.run(
                    utility_usecase.reset_state(
                        state_categories=list(categories) if categories else None,
                        create_backup=backup,
                        confirm_reset=True,
                    )
                )

            if results.get("success"):
                reset_results = results.get("reset_results", {})
                ui.display_success(
                    f"State reset completed. Categories: {len(reset_results.get('categories_reset', []))}",
                    "Reset Complete",
                )

                if reset_results.get("backup_created"):
                    ui.display_info(
                        f"Backup created: {reset_results.get('backup_location')}",
                        "Backup Information",
                    )
            else:
                ui.display_error("State reset failed", "Reset Failed")

        except Exception as e:
            ui.display_error(f"State reset failed: {e}", "Reset Error")
            if ctx.obj.verbose:
                import traceback

                ui.display_info(traceback.format_exc(), "Debug Information")
            sys.exit(1)

    @app.command()
    def version() -> None:
        """Show version information."""
        try:
            # Try to get version from package metadata
            try:
                import importlib.metadata

                version = importlib.metadata.version("testcraft")
            except Exception:
                version = "development"

            ui.display_info(f"TestCraft version {version}", "Version Information")

            # Show additional build information if available
            build_info = {
                "Python Version": sys.version.split()[0],
                "Platform": sys.platform,
            }

            for key, value in build_info.items():
                console.print(f"[dim]{key}:[/] {value}")

        except Exception as e:
            ui.display_error(f"Version information failed: {e}", "Version Error")


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================


def display_environment_info(results: dict[str, Any]) -> None:
    """Display environment information."""
    env_info = results.get("environment_info", {})

    for section_name, section_data in env_info.items():
        if isinstance(section_data, dict) and section_data:
            console.print(f"\n[bold]{section_name.title()}:[/]")
            for key, value in section_data.items():
                if key != "variables":  # Skip large env var dumps
                    console.print(f"  {key}: {value}")


def display_cost_summary(results: dict[str, Any]) -> None:
    """Display cost summary information."""
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


def display_debug_state(results: dict[str, Any], output_format: str) -> None:
    """Display debug state information."""
    if output_format in ["json", "yaml"]:
        formatted_output = results.get("formatted_output", "")
        if formatted_output:
            console.print(formatted_output)
    else:
        # Text format - show key sections
        debug_state = results.get("debug_state", {})
        for section, data in debug_state.items():
            if isinstance(data, dict):
                console.print(f"\n[bold]{section.title()}:[/]")
                for key, value in list(data.items())[:10]:  # Limit output
                    console.print(f"  {key}: {value}")
                if len(data) > 10:
                    console.print(f"  ... and {len(data) - 10} more items")
