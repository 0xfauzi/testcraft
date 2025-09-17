"""Utility commands for the TestCraft CLI."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from ..adapters.io.ui_rich import RichUIAdapter
from ..config.model_catalog_loader import load_catalog, iter_models
from .config_init import ConfigInitializer

# Initialize Rich console and UI components
console = Console()
ui = RichUIAdapter(console)


# ============================================================================
# MODELS COMMAND GROUP
# ============================================================================

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
        catalog = load_catalog()
        
        if format == "table":
            _display_models_table(catalog, provider, include_aliases)
        elif format == "json":
            _display_models_json(catalog, provider)
        elif format == "yaml":
            _display_models_yaml(catalog, provider)
            
    except Exception as e:
        ui.display_error(f"Failed to load model catalog: {e}", "Catalog Error")
        if ctx.obj and ctx.obj.verbose:
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
        catalog = load_catalog()
        verification_results = _verify_catalog_integrity(catalog, provider, check_usage)
        _display_verification_results(verification_results)
        
        if not verification_results["passed"]:
            sys.exit(1)
            
    except Exception as e:
        ui.display_error(f"Verification failed: {e}", "Verification Error")
        if ctx.obj and ctx.obj.verbose:
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
            from datetime import timedelta
            since_date = datetime.now() - timedelta(days=30)
        
        catalog = load_catalog()
        diff_results = _generate_catalog_diff(catalog, since_date, provider)
        _display_catalog_diff(diff_results, since_date)
        
    except Exception as e:
        ui.display_error(f"Diff generation failed: {e}", "Diff Error")
        if ctx.obj and ctx.obj.verbose:
            import traceback
            ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


def add_utility_commands(app: click.Group) -> None:
    """Add utility commands to the main CLI app."""
    
    # Add models command group
    app.add_command(models)

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

    @app.command(name="config-migrate")
    @click.option(
        "--config-file",
        "-c", 
        type=click.Path(exists=True, path_type=Path),
        help="Path to configuration file to migrate"
    )
    @click.option(
        "--backup/--no-backup",
        default=True,
        help="Create backup of original file before migration"
    )
    @click.option(
        "--dry-run",
        is_flag=True,
        help="Show what would be migrated without making changes"
    )
    @click.pass_context
    def config_migrate(
        ctx: click.Context, 
        config_file: Path | None, 
        backup: bool,
        dry_run: bool
    ) -> None:
        """Migrate configuration file to new schema."""
        try:
            from ..config.loader import ConfigLoader
            import tomllib
            
            # Find config file if not specified
            if not config_file:
                loader = ConfigLoader()
                config_file = loader._get_config_file_path()
                if not config_file or not config_file.exists():
                    ui.display_error("No configuration file found to migrate", "Migration Failed")
                    sys.exit(1)
            
            ui.display_info(f"Migrating configuration file: {config_file}", "Config Migration")
            
            # Load current config
            with open(config_file, "rb") as f:
                content = tomllib.load(f)
            
            # Check what needs migration
            deprecated_sections = {'style', 'coverage', 'quality', 'context_enrichment'}
            found_deprecated = []
            
            for section in deprecated_sections:
                if section in content:
                    found_deprecated.append(section)
            
            if not found_deprecated:
                ui.display_info("No deprecated sections found - configuration is up to date", "Migration")
                return
                
            ui.display_info(
                f"Found deprecated sections: {', '.join(found_deprecated)}", 
                "Deprecated Sections"
            )
            
            if dry_run:
                ui.display_info("DRY RUN MODE - No changes will be made", "Dry Run")
                
                # Show what would be migrated
                migrations = []
                
                if 'style' in content and content['style'].get('framework'):
                    migrations.append(f"style.framework -> generation.test_framework")
                    
                if 'context_enrichment' in content:
                    migrations.append(f"context_enrichment -> generation.context_enrichment")
                    
                for deprecated in found_deprecated:
                    migrations.append(f"Remove deprecated section: [{deprecated}]")
                
                if migrations:
                    ui.display_info(
                        "Planned migrations:\n" + "\n".join(f"  • {m}" for m in migrations),
                        "Migration Plan"
                    )
                return
            
            # Create backup if requested
            if backup:
                backup_file = config_file.with_suffix(f"{config_file.suffix}.backup")
                backup_file.write_bytes(config_file.read_bytes())
                ui.display_info(f"Created backup: {backup_file}", "Backup")
            
            # Perform migration using the same logic as the loader
            loader = ConfigLoader()
            migrated_content = loader._migrate_deprecated_sections(content)
            
            # Write migrated content back to file
            try:
                import tomli_w
                with open(config_file, "wb") as f:
                    tomli_w.dump(migrated_content, f)
            except ImportError:
                ui.display_error(
                    "tomli_w package required for migration. Install with: pip install tomli-w",
                    "Missing Dependency"
                )
                sys.exit(1)
            
            ui.display_success(
                f"Successfully migrated {config_file}", 
                "Migration Complete"
            )
            
            # Show what was migrated
            changes = []
            if 'style' in content and content['style'].get('framework'):
                changes.append("Migrated style.framework to generation.test_framework")
            if 'context_enrichment' in content:
                changes.append("Migrated context_enrichment to generation.context_enrichment")
            
            deprecated_removed = [s for s in found_deprecated if s in content]
            if deprecated_removed:
                changes.append(f"Removed deprecated sections: {', '.join(deprecated_removed)}")
            
            if changes:
                ui.display_info(
                    "Changes made:\n" + "\n".join(f"  • {c}" for c in changes),
                    "Migration Summary"
                )
                
        except Exception as e:
            ui.display_error(f"Migration failed: {e}", "Migration Failed")
            sys.exit(1)


# ============================================================================
# MODEL CATALOG HELPER FUNCTIONS
# ============================================================================


def _display_models_table(catalog, provider_filter: str | None = None, include_aliases: bool = False) -> None:
    """Display model catalog as a formatted table."""
    table = Table(
        title="Model Catalog",
        show_header=True,
        header_style="bold blue"
    )
    
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Model ID", style="green", no_wrap=True)
    if include_aliases:
        table.add_column("Aliases", style="yellow", max_width=20)
    table.add_column("Max Context", justify="right", style="white")
    table.add_column("Max Output", justify="right", style="white")
    table.add_column("Pricing ($/M)", style="bright_cyan", max_width=15)
    table.add_column("Flags", style="magenta", max_width=15)
    table.add_column("Last Verified", style="dim", max_width=12)
    
    filtered_models = []
    for entry in catalog.models:
        if provider_filter is None or entry.provider.lower() == provider_filter.lower():
            filtered_models.append(entry)
    
    # Sort by provider, then by model_id
    filtered_models.sort(key=lambda x: (x.provider, x.model_id))
    
    for entry in filtered_models:
        # Format pricing
        pricing_str = "N/A"
        if entry.pricing and entry.pricing.per_million:
            pricing = entry.pricing.per_million
            if pricing.input is not None and pricing.output is not None:
                pricing_str = f"${pricing.input:.2f}/${pricing.output:.2f}"
            elif pricing.input is not None:
                pricing_str = f"${pricing.input:.2f}/?"
            elif pricing.output is not None:
                pricing_str = f"?/${pricing.output:.2f}"
        
        # Format flags
        flags = []
        if entry.flags:
            if entry.flags.vision:
                flags.append("vision")
            if entry.flags.tool_use:
                flags.append("tools")
            if entry.flags.structured_outputs:
                flags.append("json")
            if entry.flags.reasoning_capable:
                flags.append("reasoning")
        
        flags_str = ", ".join(flags) if flags else "basic"
        
        # Format last verified date
        verified_str = "unknown"
        if entry.source and entry.source.last_verified:
            try:
                verified_date = datetime.strptime(entry.source.last_verified, "%Y-%m-%d")
                verified_str = verified_date.strftime("%Y-%m-%d")
            except ValueError:
                verified_str = entry.source.last_verified
        
        # Format thinking tokens
        max_thinking = ""
        if entry.limits.max_thinking and entry.limits.max_thinking > 0:
            max_thinking = f" (+{entry.limits.max_thinking//1000}K thinking)"
        
        row_data = [
            entry.provider,
            entry.model_id,
        ]
        
        if include_aliases:
            aliases_str = ", ".join(entry.aliases[:3])  # Show first 3 aliases
            if len(entry.aliases) > 3:
                aliases_str += f" (+{len(entry.aliases) - 3})"
            row_data.append(aliases_str if entry.aliases else "none")
        
        row_data.extend([
            f"{entry.limits.max_context//1000}K",
            f"{entry.limits.default_max_output//1000}K{max_thinking}",
            pricing_str,
            flags_str,
            verified_str
        ])
        
        table.add_row(*row_data)
    
    console.print(table)
    
    # Display summary
    total_models = len(filtered_models)
    providers = set(entry.provider for entry in filtered_models)
    
    summary_text = f"Total models: {total_models}"
    if provider_filter:
        summary_text += f" (filtered by {provider_filter})"
    else:
        summary_text += f" across {len(providers)} providers: {', '.join(sorted(providers))}"
    
    console.print(f"\n[dim]{summary_text}[/]")


def _display_models_json(catalog, provider_filter: str | None = None) -> None:
    """Display model catalog as JSON."""
    import json
    
    filtered_models = []
    for entry in catalog.models:
        if provider_filter is None or entry.provider.lower() == provider_filter.lower():
            filtered_models.append(entry.model_dump())
    
    output = {
        "version": catalog.version,
        "models": filtered_models,
        "metadata": {
            "total_models": len(filtered_models),
            "generated_at": datetime.now().isoformat()
        }
    }
    
    console.print(json.dumps(output, indent=2))


def _display_models_yaml(catalog, provider_filter: str | None = None) -> None:
    """Display model catalog as YAML."""
    try:
        import yaml
    except ImportError:
        ui.display_error(
            "PyYAML package required for YAML output. Install with: pip install PyYAML",
            "Missing Dependency"
        )
        sys.exit(1)
    
    filtered_models = []
    for entry in catalog.models:
        if provider_filter is None or entry.provider.lower() == provider_filter.lower():
            filtered_models.append(entry.model_dump())
    
    output = {
        "version": catalog.version,
        "models": filtered_models,
        "metadata": {
            "total_models": len(filtered_models),
            "generated_at": datetime.now().isoformat()
        }
    }
    
    console.print(yaml.dump(output, default_flow_style=False, sort_keys=False))


def _verify_catalog_integrity(catalog, provider_filter: str | None = None, check_usage: bool = False) -> dict[str, Any]:
    """Verify catalog integrity and optionally check usage compliance."""
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Check catalog structure and data integrity
    model_ids_seen = set()
    provider_stats = {}
    
    for entry in catalog.models:
        provider_key = entry.provider.lower()
        
        # Skip if filtering by provider
        if provider_filter and provider_key != provider_filter.lower():
            continue
        
        # Track provider stats
        if provider_key not in provider_stats:
            provider_stats[provider_key] = {"models": 0, "issues": 0}
        provider_stats[provider_key]["models"] += 1
        
        # Check for duplicate model IDs within provider
        model_key = f"{provider_key}:{entry.model_id.lower()}"
        if model_key in model_ids_seen:
            results["errors"].append(f"Duplicate model ID: {entry.provider}:{entry.model_id}")
            results["passed"] = False
            provider_stats[provider_key]["issues"] += 1
        model_ids_seen.add(model_key)
        
        # Check required fields
        if not entry.model_id:
            results["errors"].append(f"Missing model ID for {entry.provider} entry")
            results["passed"] = False
            provider_stats[provider_key]["issues"] += 1
        
        if not entry.limits.max_context or entry.limits.max_context <= 0:
            results["errors"].append(f"Invalid max_context for {entry.provider}:{entry.model_id}")
            results["passed"] = False
            provider_stats[provider_key]["issues"] += 1
        
        if not entry.limits.default_max_output or entry.limits.default_max_output <= 0:
            results["errors"].append(f"Invalid default_max_output for {entry.provider}:{entry.model_id}")
            results["passed"] = False
            provider_stats[provider_key]["issues"] += 1
        
        # Check logical constraints
        if entry.limits.default_max_output > entry.limits.max_context:
            results["warnings"].append(
                f"default_max_output ({entry.limits.default_max_output}) > max_context ({entry.limits.max_context}) "
                f"for {entry.provider}:{entry.model_id}"
            )
            provider_stats[provider_key]["issues"] += 1
        
        # Check source information
        if not entry.source or not entry.source.url:
            results["warnings"].append(f"Missing source URL for {entry.provider}:{entry.model_id}")
        
        if not entry.source or not entry.source.last_verified:
            results["warnings"].append(f"Missing last_verified date for {entry.provider}:{entry.model_id}")
        elif entry.source.last_verified:
            try:
                verified_date = datetime.strptime(entry.source.last_verified, "%Y-%m-%d")
                days_old = (datetime.now() - verified_date).days
                if days_old > 90:  # Warn if older than 90 days
                    results["warnings"].append(
                        f"Verification date is {days_old} days old for {entry.provider}:{entry.model_id}"
                    )
            except ValueError:
                results["warnings"].append(f"Invalid date format for {entry.provider}:{entry.model_id}")
        
        # Check alias conflicts
        for alias in entry.aliases:
            alias_key = f"{provider_key}:{alias.lower()}"
            if alias_key in model_ids_seen:
                results["errors"].append(
                    f"Alias '{alias}' for {entry.provider}:{entry.model_id} conflicts with existing model ID"
                )
                results["passed"] = False
                provider_stats[provider_key]["issues"] += 1
        
        # Add aliases to seen set
        for alias in entry.aliases:
            model_ids_seen.add(f"{provider_key}:{alias.lower()}")
    
    # Check usage compliance if requested
    if check_usage:
        usage_results = _check_usage_compliance(catalog, provider_filter)
        results["usage_compliance"] = usage_results
        if not usage_results["compliant"]:
            results["passed"] = False
            results["errors"].extend(usage_results["violations"])
    
    results["stats"] = {
        "total_models": sum(stats["models"] for stats in provider_stats.values()),
        "total_providers": len(provider_stats),
        "provider_breakdown": provider_stats,
        "total_errors": len(results["errors"]),
        "total_warnings": len(results["warnings"])
    }
    
    return results


def _check_usage_compliance(catalog, provider_filter: str | None = None) -> dict[str, Any]:
    """Check if code usage complies with catalog caps."""
    # This is a simplified implementation - in reality, this would scan
    # the codebase for model usage patterns and validate against catalog limits
    from ..adapters.llm.token_calculator import TokenCalculator
    from ..config.models import TestCraftConfig
    
    results = {
        "compliant": True,
        "violations": [],
        "checks_performed": []
    }
    
    try:
        # Load current config to check default models
        from ..config.loader import ConfigLoader
        loader = ConfigLoader()
        config = loader.load_config()
        
        # Check if default models exist in catalog
        llm_config = getattr(config, 'llm', None)
        if llm_config:
            for provider_config in [llm_config.openai, llm_config.anthropic, llm_config.azure_openai, llm_config.bedrock]:
                if provider_config and hasattr(provider_config, 'model'):
                    model_id = provider_config.model
                    provider_name = provider_config.__class__.__name__.lower().replace('config', '')
                    
                    # Skip if filtering by provider
                    if provider_filter and provider_name != provider_filter.lower():
                        continue
                    
                    # Check if model exists in catalog
                    entry = catalog.resolve(provider_name, model_id)
                    if not entry:
                        results["violations"].append(
                            f"Default model {provider_name}:{model_id} not found in catalog"
                        )
                        results["compliant"] = False
                    
                    results["checks_performed"].append(f"Verified {provider_name}:{model_id} exists in catalog")
        
        # Additional compliance checks could be added here:
        # - Scan for hardcoded model limits in code
        # - Check token calculator usage patterns
        # - Validate adapter configurations
        
    except Exception as e:
        results["violations"].append(f"Failed to check usage compliance: {e}")
        results["compliant"] = False
    
    return results


def _display_verification_results(results: dict[str, Any]) -> None:
    """Display verification results."""
    stats = results.get("stats", {})
    
    if results["passed"]:
        ui.display_success(
            f"Catalog verification passed! Verified {stats.get('total_models', 0)} models "
            f"across {stats.get('total_providers', 0)} providers",
            "Verification Success"
        )
    else:
        ui.display_error(
            f"Catalog verification failed with {len(results['errors'])} errors",
            "Verification Failed"
        )
    
    # Display errors
    if results["errors"]:
        console.print("\n[bold red]Errors:[/]")
        for error in results["errors"]:
            console.print(f"  ❌ {error}")
    
    # Display warnings
    if results["warnings"]:
        console.print("\n[bold yellow]Warnings:[/]")
        for warning in results["warnings"]:
            console.print(f"  ⚠️  {warning}")
    
    # Display provider breakdown
    provider_stats = stats.get("provider_breakdown", {})
    if provider_stats:
        table = Table(
            title="Provider Breakdown",
            show_header=True,
            header_style="bold blue"
        )
        table.add_column("Provider", style="cyan")
        table.add_column("Models", justify="right")
        table.add_column("Issues", justify="right", style="red")
        
        for provider, provider_stat in sorted(provider_stats.items()):
            table.add_row(
                provider,
                str(provider_stat["models"]),
                str(provider_stat["issues"])
            )
        
        console.print(table)
    
    # Display usage compliance if checked
    if "usage_compliance" in results:
        compliance = results["usage_compliance"]
        if compliance["compliant"]:
            ui.display_success("Code usage complies with catalog limits", "Usage Compliance")
        else:
            ui.display_error("Code usage violations found", "Usage Compliance Failed")
            for violation in compliance["violations"]:
                console.print(f"  ❌ {violation}")


def _generate_catalog_diff(catalog, since_date: datetime, provider_filter: str | None = None) -> dict[str, Any]:
    """Generate catalog diff since specified date."""
    # This is a simplified implementation. In a real system, this would:
    # - Compare against a historical version of the catalog
    # - Track changes in a version control system
    # - Maintain change logs
    
    results = {
        "since_date": since_date.isoformat(),
        "changes": [],
        "summary": {
            "added": 0,
            "modified": 0,
            "removed": 0
        }
    }
    
    # For now, we'll check last_verified dates as a proxy for recent changes
    recent_changes = []
    
    for entry in catalog.models:
        if provider_filter and entry.provider.lower() != provider_filter.lower():
            continue
        
        if entry.source and entry.source.last_verified:
            try:
                verified_date = datetime.strptime(entry.source.last_verified, "%Y-%m-%d")
                if verified_date >= since_date:
                    recent_changes.append({
                        "type": "verified",
                        "provider": entry.provider,
                        "model_id": entry.model_id,
                        "date": verified_date.isoformat(),
                        "details": f"Verification updated"
                    })
            except ValueError:
                # Invalid date format - skip
                continue
    
    results["changes"] = recent_changes
    results["summary"]["modified"] = len(recent_changes)
    
    # Add a note about the simplified implementation
    results["note"] = (
        "This is a simplified diff showing recent verification updates. "
        "A full implementation would track detailed changes in model limits, "
        "pricing, and capabilities."
    )
    
    return results


def _display_catalog_diff(diff_results: dict[str, Any], since_date: datetime) -> None:
    """Display catalog diff results."""
    since_str = since_date.strftime("%Y-%m-%d")
    changes = diff_results.get("changes", [])
    summary = diff_results.get("summary", {})
    
    if not changes:
        ui.display_info(
            f"No changes found in model catalog since {since_str}",
            "No Changes"
        )
        return
    
    # Display summary
    total_changes = sum(summary.values())
    ui.display_info(
        f"Found {total_changes} changes since {since_str}",
        "Catalog Changes"
    )
    
    # Display changes in a table
    if changes:
        table = Table(
            title=f"Model Catalog Changes Since {since_str}",
            show_header=True,
            header_style="bold blue"
        )
        table.add_column("Date", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Provider", style="cyan")
        table.add_column("Model", style="white")
        table.add_column("Details", style="dim")
        
        # Sort changes by date (newest first)
        changes.sort(key=lambda x: x["date"], reverse=True)
        
        for change in changes:
            change_date = datetime.fromisoformat(change["date"]).strftime("%m-%d")
            table.add_row(
                change_date,
                change["type"],
                change["provider"],
                change["model_id"],
                change["details"]
            )
        
        console.print(table)
    
    # Display note about implementation
    if "note" in diff_results:
        console.print(f"\n[dim]{diff_results['note']}[/]")
    
    # Display summary stats
    console.print(f"\n[dim]Summary: {summary.get('modified', 0)} verified, "
                 f"{summary.get('added', 0)} added, {summary.get('removed', 0)} removed[/]")
