"""Models command group and subcommands."""

import sys
from datetime import datetime, timedelta

import click
from rich.console import Console

from ...adapters.io.ui_rich import RichUIAdapter
from ..formatters.models import ModelCatalogFormatter
from ..services.model_catalog_service import ModelCatalogService

# Initialize Rich console and UI components
console = Console()
ui = RichUIAdapter(console)


@click.group()
@click.pass_context
def models(ctx: click.Context) -> None:
    """Manage model catalog - show, verify, and diff model metadata."""
    pass


@models.command(name="show")
@click.option(
    "--provider", "-p", 
    type=str, 
    help="Filter by provider (e.g., 'openai', 'anthropic')"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["table", "json", "yaml"], case_sensitive=False),
    default="table",
    help="Output format"
)
@click.option(
    "--include-aliases",
    is_flag=True,
    help="Include model aliases in output"
)
@click.pass_context
def models_show(ctx: click.Context, provider: str | None, format: str, include_aliases: bool) -> None:
    """Display model catalog with limits, pricing, and provenance."""
    try:
        service = ModelCatalogService()
        formatter = ModelCatalogFormatter(console)
        
        catalog_data = service.get_catalog_data(provider, include_aliases)
        
        if format == "table":
            formatter.format_table(catalog_data)
        elif format == "json":
            output = formatter.format_json(catalog_data)
            console.print(output)
        elif format == "yaml":
            output = formatter.format_yaml(catalog_data)
            console.print(output)
            
    except Exception as e:
        ui.display_error(f"Failed to load model catalog: {e}", "Catalog Error")
        if ctx.obj and hasattr(ctx.obj, 'verbose') and ctx.obj.verbose:
            import traceback
            ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


@models.command(name="verify")
@click.option(
    "--check-usage",
    is_flag=True,
    help="Check that code usage doesn't exceed catalog caps"
)
@click.option(
    "--provider", "-p",
    type=str,
    help="Verify specific provider only"
)
@click.pass_context
def models_verify(ctx: click.Context, check_usage: bool, provider: str | None) -> None:
    """Verify model catalog integrity and usage compliance."""
    try:
        service = ModelCatalogService()
        formatter = ModelCatalogFormatter(console)
        
        verification_results = service.verify_catalog_integrity(provider, check_usage)
        formatter.format_verification_results(verification_results)
        
        if not verification_results["passed"]:
            sys.exit(1)
            
    except Exception as e:
        ui.display_error(f"Verification failed: {e}", "Verification Error")
        if ctx.obj and hasattr(ctx.obj, 'verbose') and ctx.obj.verbose:
            import traceback
            ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


@models.command(name="diff")
@click.option(
    "--since",
    type=str,
    help="Show changes since date (YYYY-MM-DD format)"
)
@click.option(
    "--provider", "-p",
    type=str,
    help="Show changes for specific provider only"
)
@click.pass_context
def models_diff(ctx: click.Context, since: str | None, provider: str | None) -> None:
    """Show changes in model catalog since specified date."""
    try:
        if since:
            try:
                since_date = datetime.strptime(since, "%Y-%m-%d")
            except ValueError:
                ui.display_error(
                    "Invalid date format. Use YYYY-MM-DD (e.g., 2025-01-15)",
                    "Date Format Error"
                )
                sys.exit(1)
        else:
            # Default to 30 days ago
            since_date = datetime.now() - timedelta(days=30)
        
        service = ModelCatalogService()
        formatter = ModelCatalogFormatter(console)
        
        diff_results = service.generate_catalog_diff(since_date, provider)
        formatter.format_catalog_diff(diff_results)
        
    except Exception as e:
        ui.display_error(f"Diff generation failed: {e}", "Diff Error")
        if ctx.obj and hasattr(ctx.obj, 'verbose') and ctx.obj.verbose:
            import traceback
            ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)
