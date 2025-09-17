"""Environment command and related utilities."""

import asyncio
import sys

import click
from rich.console import Console

from ...adapters.io.ui_rich import RichUIAdapter
from ..formatters.env import EnvironmentFormatter

# Initialize Rich console and UI components
console = Console()
ui = RichUIAdapter(console)


@click.command()
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

        formatter = EnvironmentFormatter(console)
        formatter.format_environment_info(results)

    except Exception as e:
        ui.display_error(
            f"Environment info collection failed: {e}", "Environment Error"
        )
        if hasattr(ctx.obj, 'verbose') and ctx.obj.verbose:
            import traceback
            ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


@click.command()
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

        formatter = EnvironmentFormatter(console)
        if results.get("success"):
            formatter.format_cost_summary(results)
        else:
            ui.display_warning(
                results.get("error", "Cost information unavailable"),
                "Cost Information",
            )

    except Exception as e:
        ui.display_error(f"Cost summary failed: {e}", "Cost Error")
        if hasattr(ctx.obj, 'verbose') and ctx.obj.verbose:
            import traceback
            ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


@click.command()
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

        formatter = EnvironmentFormatter(console)
        formatter.format_debug_state(results, output_format)

    except Exception as e:
        ui.display_error(f"Debug state dump failed: {e}", "Debug Error")
        if hasattr(ctx.obj, 'verbose') and ctx.obj.verbose:
            import traceback
            ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)
