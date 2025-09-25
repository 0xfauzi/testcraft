"""Main CLI entry point for TestCraft."""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from ..adapters.io.enhanced_logging import (
    LoggerManager,
    LogMode,
    get_operation_logger,
    setup_enhanced_logging,
)
from ..adapters.io.enhanced_ui import EnhancedUIAdapter
from ..adapters.io.rich_cli import RichCliComponents, get_theme
from ..adapters.io.ui_rich import UIStyle
from ..application.environment.preflight import EnvironmentValidator
from ..config.loader import ConfigLoader, ConfigurationError
from ..config.models import TestCraftConfig
from .commands.models import add_model_commands
from .dependency_injection import DependencyError, create_dependency_container
from .evaluation_commands import add_evaluation_commands
from .utility_commands import add_utility_commands


def detect_ui_style(ui_flag: str | None) -> UIStyle:
    """Detect appropriate UI style based on flag, environment, and TTY status."""
    # Priority 1: Explicit --ui flag
    if ui_flag:
        if ui_flag.lower() == "minimal":
            return UIStyle.MINIMAL
        elif ui_flag.lower() == "classic":
            return UIStyle.CLASSIC

    # Priority 2: Environment variable
    env_ui = os.getenv("TESTCRAFT_UI")
    if env_ui:
        if env_ui.lower() == "minimal":
            return UIStyle.MINIMAL
        elif env_ui.lower() == "classic":
            return UIStyle.CLASSIC

    # Priority 3: Auto-detect based on environment
    if os.getenv("CI") == "true" or not sys.stdout.isatty():
        return UIStyle.MINIMAL

    # Default to classic for interactive terminals
    return UIStyle.CLASSIC


# Global UI components will be initialized after CLI argument parsing


class ClickContext:
    """Context object for Click commands."""

    def __init__(self) -> None:
        self.config: TestCraftConfig | None = None
        self.container: dict[str, Any] | None = None
        self.ui: EnhancedUIAdapter | None = None  # Will be initialized in app()
        self.rich_cli: RichCliComponents | None = None  # Will be initialized in app()
        self.ui_style: UIStyle = UIStyle.CLASSIC  # Will be set in app()
        self.verbose: bool = False
        self.quiet: bool = False
        self.ui_flag_explicit: bool = False
        self.dry_run: bool = False


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Reduce output: set log level to WARNING and hide INFO",
)
@click.option(
    "--dry-run", "--dry", is_flag=True, help="Preview operations without executing them"
)
@click.option(
    "--ui",
    type=click.Choice(["minimal", "classic"], case_sensitive=False),
    help="UI style: 'minimal' for CI/non-TTY, 'classic' for interactive (auto-detected by default)",
)
@click.option(
    "--compact",
    is_flag=True,
    help="Alias for --ui minimal (compact output)",
)
@click.pass_context
def app(
    ctx: click.Context,
    config: Path | None,
    verbose: bool,
    quiet: bool,
    dry_run: bool,
    ui: str | None,
    compact: bool,
) -> None:
    """TestCraft - AI-powered test generation tool for Python projects."""
    # Initialize context
    ctx.ensure_object(ClickContext)
    ctx.obj.verbose = verbose
    ctx.obj.quiet = quiet
    ctx.obj.dry_run = dry_run
    ctx.obj.ui_flag_explicit = bool(ui) or bool(compact)

    # Detect and set UI style
    ui_flag = "minimal" if compact and not ui else ui
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

    # Create UI without reconfiguring logging (logging already set up above)
    ctx.obj.ui = EnhancedUIAdapter(
        console, enable_rich_logging=False, ui_style=ctx.obj.ui_style
    )
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

    # Allow running certain commands without a valid config
    skip_config_commands = {"init-config", "models"}
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
        loader = ConfigLoader(config)
        ctx.obj.config = loader.load_config()

        # Create dependency container
        ctx.obj.container = create_dependency_container(ctx.obj.config)

    except ConfigurationError as e:
        suggestions = [
            "Check if the configuration file exists and is readable",
            "Verify the configuration file format (TOML, YAML, or JSON)",
            "Run 'testcraft init-config' to create a new configuration file",
        ]
        ctx.obj.ui.display_error_with_suggestions(
            f"Configuration error: {e}", suggestions, "Configuration Failed"
        )
        logger.error(f"💥 Configuration initialization failed: {e}")
        sys.exit(1)
    except DependencyError as e:
        suggestions = [
            "Check if all required dependencies are installed",
            "Verify your Python environment and virtual environment",
            "Try reinstalling TestCraft with 'pip install --force-reinstall testcraft'",
        ]
        ctx.obj.ui.display_error_with_suggestions(
            f"Dependency injection error: {e}", suggestions, "Initialization Failed"
        )
        logger.error(f"💥 Dependency injection failed: {e}")
        sys.exit(1)
    except Exception as e:
        suggestions = [
            "Try running with --verbose flag for more information",
            "Check your Python version (requires 3.11+)",
            "Verify file permissions and disk space",
        ]
        ctx.obj.ui.display_error_with_suggestions(
            f"Unexpected error during initialization: {e}",
            suggestions,
            "Initialization Failed",
        )
        logger.error(f"💥 Unexpected initialization error: {e}", exc_info=verbose)
        sys.exit(1)


# ============================================================================
# MAIN COMMANDS
# ============================================================================


@app.command()
@click.argument(
    "project_path", type=click.Path(exists=True, path_type=Path), default="."
)
@click.option(
    "--target-files",
    "-f",
    multiple=True,
    type=click.Path(path_type=Path),
    help="Specific files to generate tests for",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=5,
    help="Number of files to process in parallel",
)
@click.option("--streaming", is_flag=True, help="Enable streaming LLM responses")
@click.option("--force", is_flag=True, help="Overwrite existing test files")
@click.option(
    "--immediate/--no-immediate",
    default=True,
    help="Enable immediate write-and-refine per file (default: enabled)",
)
@click.option(
    "--max-refine-workers",
    type=int,
    default=2,
    help="Maximum concurrent pytest/refine workers",
)
@click.option(
    "--keep-failed-writes",
    is_flag=True,
    help="Keep test files that fail to write or have syntax errors",
)
@click.option(
    "--disable-ruff",
    is_flag=True,
    help="Disable Ruff formatting (use Black+isort instead) to avoid timeouts",
)
@click.option(
    "--enable-symbol-resolution/--disable-symbol-resolution",
    default=True,
    help="Enable/disable missing_symbols resolution loop (default: enabled)",
)
@click.option(
    "--max-plan-retries",
    type=int,
    default=2,
    help="Maximum retries for PLAN stage with symbol resolution (default: 2)",
)
@click.option(
    "--max-refine-retries",
    type=int,
    default=3,
    help="Maximum retries for REFINE stage with symbol resolution (default: 3)",
)
@click.pass_context
def generate(
    ctx: click.Context,
    project_path: Path,
    target_files: tuple[Path, ...],
    batch_size: int,
    streaming: bool,
    force: bool,
    immediate: bool,
    max_refine_workers: int,
    keep_failed_writes: bool,
    disable_ruff: bool,
    enable_symbol_resolution: bool,
    max_plan_retries: int,
    max_refine_retries: int,
) -> None:
    """Generate tests for Python source files."""
    operation_logger = get_operation_logger("generate")

    try:
        with operation_logger.operation_context(
            "test_generation",
            project_path=str(project_path),
            target_files_count=len(target_files),
            batch_size=batch_size,
            immediate_mode=immediate,
        ):
            if ctx.obj.dry_run:
                ctx.obj.ui.display_info(
                    "DRY RUN: No tests will actually be generated", "Dry Run Mode"
                )
                operation_logger.info(
                    "🔍 [yellow]Dry run mode activated[/] - no files will be modified"
                )

            # Preflight environment validation before doing anything expensive
            # Determine if coverage tools should be present: enable when the project is not using the placeholder adapter
            require_coverage_tools = True  # prefer coverage stack for accurate results
            preflight = EnvironmentValidator.validate_for_generate(
                ctx.obj.config,
                require_refinement=True,
                require_coverage_tools=require_coverage_tools,
            )
            if not preflight.get("ok", False):
                # Show a concise error with actionable suggestions and abort
                suggestions = preflight.get("suggestions", []) or [
                    "Ensure pytest is installed and available",
                    "Set your LLM API key in the environment",
                ]
                ctx.obj.ui.display_error_with_suggestions(
                    preflight.get("message", "Environment preflight failed"),
                    suggestions,
                    "Preflight Failed",
                )
                operation_logger.error(
                    "💥 Environment preflight failed; aborting generate"
                )
                sys.exit(1)

            # Get use case from container (after preflight passes)
            generate_usecase = ctx.obj.container["generate_usecase"]

            # Configure generation parameters
            config_overrides = {
                "batch_size": batch_size,
                "enable_streaming": streaming,
                "force_overwrite": force,
                "immediate_refinement": immediate,
                "max_refine_workers": max_refine_workers,
                "keep_failed_writes": keep_failed_writes,
                "disable_ruff_format": disable_ruff,
                # Symbol resolution configuration (task 34.5)
                "enable_symbol_resolution": enable_symbol_resolution,
                "max_plan_retries": max_plan_retries,
                "max_refine_retries": max_refine_retries,
            }

            operation_logger.info(
                f"[primary]config:[/] batch={batch_size}, immediate={immediate}, workers={max_refine_workers}"
            )

            # Use enhanced live file tracking for real-time status updates
            from ..adapters.io.file_status_tracker import LiveFileTracking

            # Get file paths for tracking
            file_paths = []
            if target_files:
                file_paths = [str(f) for f in target_files]
            else:
                # Discover files in project (simplified for demo)
                try:
                    from pathlib import Path

                    project = Path(project_path)
                    file_paths = [
                        str(f)
                        for f in project.rglob("*.py")
                        if not str(f).startswith(str(project / "tests"))
                        and not str(f).name.startswith("test_")
                    ][:10]  # Limit for demo
                except Exception:
                    file_paths = []

            # Auto-minimal for immediate mode with small file count unless UI explicitly classic
            if immediate and len(file_paths) <= 3 and not ctx.obj.ui_flag_explicit:
                ctx.obj.ui_style = UIStyle.MINIMAL
                ctx.obj.ui = EnhancedUIAdapter(
                    ctx.obj.ui.console,
                    enable_rich_logging=False,
                    ui_style=ctx.obj.ui_style,
                )
                LoggerManager.set_log_mode(
                    LogMode.MINIMAL, verbose=ctx.obj.verbose, quiet=ctx.obj.quiet
                )

            # Skip live tracker entirely for minimal mode when file count <= 3
            if (
                file_paths
                and len(file_paths) > 1
                and not (ctx.obj.ui_style == UIStyle.MINIMAL and len(file_paths) <= 3)
            ):
                # Use live file tracking for multiple files
                with LiveFileTracking(
                    ctx.obj.ui, "TestCraft Test Generation"
                ) as live_file_tracker:
                    file_status_tracker = live_file_tracker.initialize_and_start(
                        file_paths
                    )

                    # Inject the status tracker into the generation pipeline
                    generate_usecase.set_status_tracker(file_status_tracker)

                    operation_logger.info(
                        f"[accent]live tracking enabled[/] for {len(file_paths)} files"
                    )

                    # Run the actual generation with integrated live tracking
                    results = asyncio.run(
                        generate_usecase.generate_tests(
                            project_path=project_path,
                            target_files=list(target_files) if target_files else None,
                            **config_overrides,
                        )
                    )

                    # Show final file status summary
                    final_stats = file_status_tracker.get_summary_stats()
                    operation_logger.info(
                        f"[primary]results:[/] {final_stats['completed']} completed, "
                        f"{final_stats['failed']} failed, {final_stats['success_rate']:.0%} success"
                    )
            else:
                # Fall back to basic progress tracking for single files
                with ctx.obj.ui.create_operation_tracker(
                    "Test Generation", total_steps=4
                ) as tracker:
                    tracker.advance_step("Initializing generation pipeline", 1)

                    # Run generation asynchronously
                    results = asyncio.run(
                        generate_usecase.generate_tests(
                            project_path=project_path,
                            target_files=list(target_files) if target_files else None,
                            **config_overrides,
                        )
                    )

                    tracker.advance_step("Processing results", 1)

            # Display results using enhanced UI components
            if results.get("success"):
                _display_generation_results(results, ctx.obj.ui)
                operation_logger.performance_summary(
                    "test_generation",
                    {
                        "files_processed": results.get("files_processed", 0),
                        "tests_generated": results.get("tests_generated", 0),
                        "success_rate": results.get("files_written", 0)
                        / max(results.get("files_processed", 1), 1),
                    },
                )
            else:
                error_msg = results.get("error_message", "Unknown error occurred")
                suggestions = [
                    "Check if the project path contains valid Python files",
                    "Verify your LLM API keys are configured correctly",
                    "Try reducing batch size or disabling immediate mode",
                ]
                ctx.obj.ui.display_error_with_suggestions(
                    error_msg, suggestions, "Generation Failed"
                )
                operation_logger.error(f"💥 Generation failed: {error_msg}")
                sys.exit(1)

    except Exception as e:
        suggestions = [
            "Check if the project directory exists and is readable",
            "Verify your configuration file is valid",
            "Try running with --verbose for more details",
        ]
        ctx.obj.ui.display_error_with_suggestions(
            f"Test generation failed: {e}", suggestions, "Generation Error"
        )
        operation_logger.error_with_context("Test generation failed", e, suggestions)
        sys.exit(1)


@app.command()
@click.argument(
    "project_path", type=click.Path(exists=True, path_type=Path), default="."
)
@click.option(
    "--target-files",
    "-f",
    multiple=True,
    type=click.Path(path_type=Path),
    help="Specific files to analyze",
)
@click.pass_context
def analyze(
    ctx: click.Context, project_path: Path, target_files: tuple[Path, ...]
) -> None:
    """Analyze what tests would be generated and why."""
    try:
        # Get use case from container
        analyze_usecase = ctx.obj.container["analyze_usecase"]

        with ctx.obj.ui.create_status_spinner("Analyzing project files..."):
            # Run analysis asynchronously
            results = asyncio.run(
                analyze_usecase.analyze_generation_needs(
                    project_path=project_path,
                    target_files=list(target_files) if target_files else None,
                )
            )

        # Display results using Rich components
        _display_analysis_results(results, ctx.obj.ui, ctx.obj.rich_cli)

    except Exception as e:
        ctx.obj.ui.display_error(f"Analysis failed: {e}", "Analysis Error")
        if ctx.obj.verbose:
            import traceback

            ctx.obj.ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


@app.command()
@click.argument(
    "project_path", type=click.Path(exists=True, path_type=Path), default="."
)
@click.option(
    "--source-files",
    "-s",
    multiple=True,
    type=click.Path(path_type=Path),
    help="Specific source files to measure",
)
@click.option(
    "--test-files",
    "-t",
    multiple=True,
    type=click.Path(path_type=Path),
    help="Specific test files to include",
)
@click.option(
    "--format",
    "-o",
    "output_format",
    type=click.Choice(["detailed", "summary", "json"], case_sensitive=False),
    multiple=True,
    default=["detailed"],
    help="Output format",
)
@click.pass_context
def coverage(
    ctx: click.Context,
    project_path: Path,
    source_files: tuple[Path, ...],
    test_files: tuple[Path, ...],
    output_format: tuple[str, ...],
) -> None:
    """Measure and report code coverage."""
    try:
        # Get use case from container
        coverage_usecase = ctx.obj.container["coverage_usecase"]

        # Configure parameters
        config_overrides = {"output_formats": list(output_format)}

        with ctx.obj.ui.create_status_spinner("Measuring code coverage..."):
            # Run coverage measurement asynchronously
            results = asyncio.run(
                coverage_usecase.measure_and_report(
                    project_path=project_path,
                    source_files=list(source_files) if source_files else None,
                    test_files=list(test_files) if test_files else None,
                    **config_overrides,
                )
            )

        # Display results using Rich components
        if results.get("success"):
            _display_coverage_results(results, ctx.obj.ui, ctx.obj.rich_cli)
        else:
            ctx.obj.ui.display_error(
                results.get("error_message", "Unknown error occurred"),
                "Coverage Failed",
            )
            sys.exit(1)

    except Exception as e:
        ctx.obj.ui.display_error(f"Coverage measurement failed: {e}", "Coverage Error")
        if ctx.obj.verbose:
            import traceback

            ctx.obj.ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


@app.command()
@click.argument(
    "project_path",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    default=None,
)
@click.option("--history", is_flag=True, help="Include generation history")
@click.option("--statistics", is_flag=True, help="Include summary statistics")
@click.option("--limit", "-l", type=int, default=10, help="Limit history entries")
@click.pass_context
def status(
    ctx: click.Context,
    project_path: Path | None,
    history: bool,
    statistics: bool,
    limit: int,
) -> None:
    """Show generation status, history, and statistics."""
    try:
        # Get use case from container
        status_usecase = ctx.obj.container["status_usecase"]

        with ctx.obj.ui.create_status_spinner("Retrieving status information..."):
            # Run status retrieval asynchronously
            results = asyncio.run(
                status_usecase.get_generation_status(
                    project_path=project_path,
                    include_history=history,
                    include_statistics=statistics,
                )
            )

        # Display results using Rich components
        _display_status_results(results, limit, ctx.obj.ui, ctx.obj.rich_cli)

    except Exception as e:
        ctx.obj.ui.display_error(f"Status retrieval failed: {e}", "Status Error")
        if ctx.obj.verbose:
            import traceback

            ctx.obj.ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


@app.command()
@click.option("--web", is_flag=True, help="Launch TUI in web browser mode")
@click.option(
    "--port", type=int, default=8080, help="Port for web mode (default: 8080)"
)
@click.pass_context
def tui(ctx: click.Context, web: bool, port: int) -> None:
    """Launch TestCraft's interactive Terminal User Interface (TUI)."""
    try:
        from ..adapters.textual.app import TestCraftTextualApp

        if web:
            # Web mode using textual-web
            try:
                import importlib.util

                if importlib.util.find_spec("textual_web") is None:
                    raise ImportError("textual_web not available")

                ctx.obj.ui.display_info(
                    f"Launching TestCraft TUI in web browser on port {port}...",
                    "Web Mode",
                )

                # Create the app
                app = TestCraftTextualApp()

                # Launch in web mode
                # Note: textual-web integration would go here
                # For now, fall back to regular terminal mode
                ctx.obj.ui.display_warning(
                    "Web mode not fully implemented yet, launching in terminal mode",
                    "Fallback",
                )
                app.run()

            except ImportError:
                ctx.obj.ui.display_error(
                    "textual-web not available. Install with: pip install textual-web",
                    "Missing Dependency",
                )
                sys.exit(1)
        else:
            # Terminal mode
            ctx.obj.ui.display_info("Launching TestCraft TUI...", "Terminal Mode")

            # Create and run the Textual app
            app = TestCraftTextualApp()
            app.run()

    except KeyboardInterrupt:
        ctx.obj.ui.display_info("TUI session ended by user", "Goodbye")
    except Exception as e:
        ctx.obj.ui.display_error(f"TUI launch failed: {e}", "TUI Error")
        if ctx.obj.verbose:
            import traceback

            ctx.obj.ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


# ============================================================================
# RESULT DISPLAY FUNCTIONS
# ============================================================================


def _display_generation_results(
    results: dict[str, Any], ui_adapter: EnhancedUIAdapter
) -> None:
    """Display test generation results using enhanced Rich components."""
    # Route to minimal renderer if minimal UI style is selected
    if ui_adapter.ui_style == UIStyle.MINIMAL:
        renderer = ui_adapter.get_renderer()
        renderer.render_generation_results(results, ui_adapter.console)
        return

    # Classic UI path: check if immediate mode was used based on metadata
    immediate_mode = (
        results.get("metadata", {})
        .get("config_used", {})
        .get("immediate_refinement", False)
    )

    if immediate_mode:
        _display_immediate_mode_results(results, ui_adapter)
    else:
        _display_legacy_mode_results(results, ui_adapter)


def _display_immediate_mode_results(
    results: dict[str, Any], ui_adapter: EnhancedUIAdapter
) -> None:
    """Display results for immediate mode with per-file detail."""
    # Prepare file processing data for enhanced display
    generation_results = results.get("generation_results", [])
    refinement_results = results.get("refinement_results", [])

    # Create lookup for refinement results
    refinement_by_file = {}
    for refine_result in refinement_results:
        file_path = refine_result.get("test_file", "")
        refinement_by_file[file_path] = refine_result

    # Build enhanced file data
    files_data = []
    for gen_result in generation_results:
        if hasattr(gen_result, "file_path"):
            file_path = gen_result.file_path
            success = gen_result.success
        else:
            file_path = gen_result.get("file_path", "unknown")
            success = gen_result.get("success", False)

        # Get refinement data
        refine_result = refinement_by_file.get(file_path)
        refine_success = refine_result.get("success", False) if refine_result else True

        # Determine final status
        final_success = success and refine_success

        file_data = {
            "file_path": file_path,
            "status": "completed" if final_success else "failed",
            "progress": 1.0 if final_success else 0.5 if success else 0.0,
            "tests_generated": refine_result.get("tests_generated", 0)
            if refine_result
            else (5 if success else 0),
            "coverage": refine_result.get("final_coverage", 0.8)
            if refine_result
            else (0.7 if success else 0.0),
            "duration": refine_result.get("duration", 0) if refine_result else 0,
        }
        files_data.append(file_data)

    # Use enhanced UI components
    # Display either the table (above) OR the summary panel, not both, to avoid duplication.
    # We already printed the table; skip the summary panel to prevent duplicate content.

    # Display coverage improvement if available
    coverage_delta = results.get("coverage_delta", {})
    if coverage_delta.get("line_coverage_delta", 0) > 0:
        ui_adapter.display_success(
            f"Coverage improved by {coverage_delta['line_coverage_delta']:.1%}",
            "Coverage Improvement",
        )


def _display_legacy_mode_results(
    results: dict[str, Any], ui_adapter: EnhancedUIAdapter
) -> None:
    """Display results for legacy mode using enhanced UI components."""
    # Create summary data
    summary_data = {
        "message": f"Successfully processed {results.get('files_processed', 0)} files",
        "metrics": {
            "test_generation": {
                "duration": results.get("total_duration", 0),
                "items_processed": results.get("files_processed", 0),
                "success_rate": results.get("files_written", 0)
                / max(results.get("files_processed", 1), 1),
            }
        },
    }

    # Display project summary panel using existing rich CLI
    project_summary_data = {
        "total_files": results.get("files_discovered", 0),
        "files_with_tests": results.get("files_written", 0),
        "overall_coverage": results.get("final_coverage", {}).get(
            "overall_line_coverage", 0
        ),
        "tests_generated": results.get("tests_generated", 0),
        "generation_success_rate": results.get("files_written", 0)
        / max(results.get("files_processed", 1), 1),
    }

    panel = ui_adapter.rich_cli.create_project_summary_panel(project_summary_data)
    ui_adapter.rich_cli.print_panel(panel)

    # Display metrics
    ui_adapter.display_metrics_panel(summary_data["metrics"])

    # Show coverage improvement
    coverage_delta = results.get("coverage_delta", {})
    if coverage_delta.get("line_coverage_delta", 0) > 0:
        ui_adapter.display_success(
            f"Coverage improved by {coverage_delta['line_coverage_delta']:.1%}",
            "Coverage Improvement",
        )


def _display_analysis_results(
    results, ui_adapter: EnhancedUIAdapter, rich_cli: RichCliComponents
) -> None:
    """Display analysis results using enhanced Rich components."""
    if hasattr(results, "files_to_process"):
        # Results is an AnalysisReport object
        files_to_process = results.files_to_process
        reasons = results.reasons
        test_presence = results.existing_test_presence
    else:
        # Results is a dictionary
        files_to_process = results.get("files_to_process", [])
        reasons = results.get("reasons", {})
        test_presence = results.get("existing_test_presence", {})

    analysis_data = {
        "files_to_process": files_to_process,
        "reasons": reasons,
        "existing_test_presence": test_presence,
    }

    # Use rich CLI components for analysis tree
    tree = rich_cli.create_analysis_tree(analysis_data)
    rich_cli.print_tree(tree)

    # Display summary statistics
    total_files = len(files_to_process)
    files_with_tests = sum(1 for f in files_to_process if test_presence.get(f, False))
    files_without_tests = total_files - files_with_tests

    ui_adapter.display_info(
        f"📊 Analysis Summary: {total_files} files need attention "
        f"({files_without_tests} without tests, {files_with_tests} need improvements)",
        "Analysis Complete",
    )


def _display_coverage_results(
    results: dict[str, Any], ui_adapter: EnhancedUIAdapter, rich_cli: RichCliComponents
) -> None:
    """Display coverage results using Rich components."""
    coverage_data = results.get("coverage_data", {})
    if coverage_data:
        table = rich_cli.create_coverage_table(coverage_data)
        rich_cli.print_table(table)

    # Show summary information
    summary = results.get("coverage_summary", {})
    if summary:
        ui_adapter.display_info(
            f"Overall coverage: {summary.get('overall_line_coverage', 0):.1%} line, "
            f"{summary.get('overall_branch_coverage', 0):.1%} branch",
            "Coverage Summary",
        )


def _display_status_results(
    results: dict[str, Any],
    limit: int,
    ui_adapter: EnhancedUIAdapter,
    rich_cli: RichCliComponents,
) -> None:
    """Display status results using Rich components."""
    current_state = results.get("current_state", {})
    history = results.get("generation_history", [])
    statistics = results.get("summary_statistics", {})

    # Display current state
    if current_state:
        state_info = []
        if current_state.get("last_generation_timestamp"):
            from datetime import datetime

            timestamp = datetime.fromtimestamp(
                current_state["last_generation_timestamp"]
            )
            state_info.append(
                f"Last generation: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        if current_state.get("generation_summary"):
            gen_summary = current_state["generation_summary"]
            state_info.append(
                f"Files processed: {gen_summary.get('total_files_processed', 0)}"
            )

        if state_info:
            ui_adapter.display_info("\n".join(state_info), "Current State")

    # Display history if available
    if history:
        ui_adapter.console.print("\n[bold]Recent Activity:[/]")
        for entry in history[:limit]:
            timestamp = entry.get("timestamp", 0)
            if timestamp:
                from datetime import datetime

                dt = datetime.fromtimestamp(timestamp)
                ui_adapter.console.print(
                    f"  • {dt.strftime('%Y-%m-%d %H:%M')} - "
                    f"{entry.get('entry_type', 'unknown')}: "
                    f"{entry.get('status', 'unknown')}"
                )

    # Display statistics if available
    if statistics:
        stats_info = []
        if statistics.get("total_runs"):
            stats_info.append(f"Total runs: {statistics['total_runs']}")
        if statistics.get("success_rate"):
            stats_info.append(f"Success rate: {statistics['success_rate']:.1%}")
        if statistics.get("average_coverage_percentage"):
            stats_info.append(
                f"Average coverage: {statistics['average_coverage_percentage']:.1%}"
            )

        if stats_info:
            ui_adapter.display_info("\n".join(stats_info), "Summary Statistics")


# Add evaluation commands
add_evaluation_commands(app)

# Add utility commands
add_utility_commands(app)

# Add model catalog commands
add_model_commands(app)


if __name__ == "__main__":
    app()
