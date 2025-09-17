"""Utility commands for the TestCraft CLI - refactored version."""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from ..adapters.io.ui_rich import RichUIAdapter
from .commands import cost, debug_state, env, models
from .config_init import ConfigInitializer

# Initialize Rich console and UI components
console = Console()
ui = RichUIAdapter(console)


def add_utility_commands(app: click.Group) -> None:
    """Add utility commands to the main CLI app."""
    
    # Add models command group (refactored)
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
            if hasattr(ctx.obj, 'verbose') and ctx.obj.verbose:
                import traceback
                ui.display_info(traceback.format_exc(), "Debug Information")
            sys.exit(1)

    # Add environment and utility commands (refactored)
    app.add_command(env)
    app.add_command(cost)
    app.add_command(debug_state)

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
            if hasattr(ctx.obj, 'verbose') and ctx.obj.verbose:
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
            if hasattr(ctx.obj, 'verbose') and ctx.obj.verbose:
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
