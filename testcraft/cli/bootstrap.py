"""CLI bootstrap and initialization utilities."""

import logging
import sys
from typing import Any

import click
from rich.console import Console

from ..adapters.io.enhanced_logging import (
    setup_enhanced_logging,
    LoggerManager,
    LogMode,
)
from ..adapters.io.enhanced_ui import EnhancedUIAdapter
from ..adapters.io.rich_cli import RichCliComponents, get_theme
from ..adapters.io.ui_rich import UIStyle
from ..config.loader import ConfigLoader, ConfigurationError
from .context import ClickContext
from .dependency_injection import DependencyError, create_dependency_container
from .ui_detect import detect_ui_style


def initialize_cli_context(
    ctx: click.Context,
    config_path: str | None = None,
    verbose: bool = False,
    quiet: bool = False,
    dry_run: bool = False,
    ui_flag: str | None = None,
    compact: bool = False,
) -> None:
    """Initialize CLI context with UI, logging, and configuration."""
    # Initialize context
    ctx.ensure_object(ClickContext)
    ctx.obj.verbose = verbose
    ctx.obj.quiet = quiet
    ctx.obj.dry_run = dry_run
    ctx.obj.ui_flag_explicit = bool(ui_flag) or bool(compact)
    
    # Detect and set UI style
    ui_flag = "minimal" if compact and not ui_flag else ui_flag
    ctx.obj.ui_style = detect_ui_style(ui_flag)
    
    # Initialize UI components with selected theme
    console = Console(theme=get_theme(ctx.obj.ui_style))
    
    # Set up enhanced logging system first (configure root once)
    logger = setup_enhanced_logging(console)
    # Configure logging mode & level
    LoggerManager.set_log_mode(
        LogMode.MINIMAL if ctx.obj.ui_style == UIStyle.MINIMAL else LogMode.CLASSIC,
        verbose=verbose,
        quiet=quiet,
    )
    
    # Quiet external libraries in non-verbose mode
    if not verbose:
        LoggerManager.quiet_external_libs(["asyncio", "httpx", "openai", "urllib3", "textual"])
    
    # Create UI without reconfiguring logging (logging already set up above)
    ctx.obj.ui = EnhancedUIAdapter(console, enable_rich_logging=False, ui_style=ctx.obj.ui_style)
    ctx.obj.rich_cli = RichCliComponents(console)
    # Quiet mode for minimal or explicit --quiet
    if ctx.obj.ui_style == UIStyle.MINIMAL or quiet:
        try:
            ctx.obj.ui.set_quiet_mode(True)
        except Exception:
            pass

    # Enhanced logging is already set up globally
    if verbose and not quiet:
        # Only change level on root, do not add handlers - keep propagation
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("🔍 [cyan]Debug mode enabled[/] - verbose logging active")


def load_configuration_and_dependencies(
    ctx: click.Context,
    config_path: str | None = None,
    skip_config_commands: set[str] | None = None,
) -> None:
    """Load configuration and create dependency container."""
    if skip_config_commands is None:
        skip_config_commands = {"init-config"}
    
    # Allow running certain commands without a valid config
    invoked = None
    try:
        # Best-effort detection of invoked subcommand from argv
        for arg in sys.argv[1:]:
            if arg in skip_config_commands:
                invoked = arg
                break

        if invoked in skip_config_commands:
            # Skip configuration loading for commands that don't require it
            ctx.obj.config = None
            ctx.obj.container = None
            return

        # Load configuration
        loader = ConfigLoader(config_path)
        ctx.obj.config = loader.load_config()

        # Create dependency container
        ctx.obj.container = create_dependency_container(ctx.obj.config)

    except ConfigurationError as e:
        suggestions = [
            "Check if the configuration file exists and is readable",
            "Verify the configuration file format (TOML, YAML, or JSON)",
            "Run 'testcraft init-config' to create a new configuration file"
        ]
        ctx.obj.ui.display_error_with_suggestions(
            f"Configuration error: {e}",
            suggestions,
            "Configuration Failed"
        )
        logger = setup_enhanced_logging(ctx.obj.ui.console)
        logger.error(f"💥 Configuration initialization failed: {e}")
        sys.exit(1)
    except DependencyError as e:
        suggestions = [
            "Check if all required dependencies are installed",
            "Verify your Python environment and virtual environment",
            "Try reinstalling TestCraft with 'pip install --force-reinstall testcraft'"
        ]
        ctx.obj.ui.display_error_with_suggestions(
            f"Dependency injection error: {e}",
            suggestions,
            "Initialization Failed"
        )
        logger = setup_enhanced_logging(ctx.obj.ui.console)
        logger.error(f"💥 Dependency injection failed: {e}")
        sys.exit(1)
    except Exception as e:
        suggestions = [
            "Try running with --verbose flag for more information",
            "Check your Python version (requires 3.11+)",
            "Verify file permissions and disk space"
        ]
        ctx.obj.ui.display_error_with_suggestions(
            f"Unexpected error during initialization: {e}",
            suggestions,
            "Initialization Failed"
        )
        logger = setup_enhanced_logging(ctx.obj.ui.console)
        logger.error_with_context("Unexpected initialization error", e, suggestions, fatal=True)
        sys.exit(1)
