"""Main CLI entry point for TestCraft."""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sys
from collections.abc import Iterable
from datetime import datetime
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
from ..adapters.io.file_discovery import FileDiscoveryError
from ..adapters.io.rich_cli import RichCliComponents, get_theme
from ..adapters.io.ui_rich import UIStyle
from ..application.environment.preflight import EnvironmentValidator
from ..application.generation.services.context_assembler import ContextAssembler
from ..application.generation.services.context_pack import ContextPackBuilder
from ..application.generation.services.llm_orchestrator import LLMOrchestrator
from ..config.loader import ConfigLoader, ConfigurationError
from ..config.models import TestCraftConfig
from .commands.models import add_model_commands
from .dependency_injection import DependencyError, create_dependency_container
from .evaluation_commands import add_evaluation_commands
from .utility_commands import add_utility_commands

logger = logging.getLogger(__name__)


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
    console = Console(theme=get_theme(ctx.obj.ui_style.value))

    # Set up enhanced logging system first (configure root once)
    try:
        logger = setup_enhanced_logging(console)
    except KeyError:
        # Fallback: initialize logging then use standard logger to avoid test KeyError
        setup_enhanced_logging(console)
        logger = logging.getLogger("testcraft.main")
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
    "--manual-fix-on-fail",
    is_flag=True,
    help="Trigger manual-fix guidance automatically when refinement fails",
)
@click.option(
    "--auto-accept-fixes",
    is_flag=True,
    help="Auto-accept manual-fix recommendation without prompt",
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
@click.option(
    "--plan-first",
    is_flag=True,
    help="Generate and review plan before test generation",
)
@click.option(
    "--auto-accept-plan",
    is_flag=True,
    help="Auto-accept plan without interactive review (overrides config)",
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
    manual_fix_on_fail: bool,
    auto_accept_fixes: bool,
    keep_failed_writes: bool,
    disable_ruff: bool,
    enable_symbol_resolution: bool,
    max_plan_retries: int,
    max_refine_retries: int,
    plan_first: bool,
    auto_accept_plan: bool,
) -> None:
    """Generate tests for Python source files."""
    operation_logger = get_operation_logger("generate")
    run_start: datetime | None = None

    try:
        project_path = project_path.resolve()
    except OSError:
        project_path = project_path.absolute()

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
                # In dry-run mode, skip environment preflight and any generation work
                # to allow offline planning without requiring LLM credentials.
                return

            run_start = datetime.now()

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

            writer_adapter = ctx.obj.container.get("writer_adapter")
            if writer_adapter and hasattr(writer_adapter, "set_project_root"):
                writer_adapter.set_project_root(project_path)
            if hasattr(generate_usecase, "set_project_root"):
                generate_usecase.set_project_root(project_path)
            llm_adapter = ctx.obj.container.get("llm_adapter")

            # Surface when the router is operating in offline/no-op mode
            if llm_adapter and hasattr(llm_adapter, "ensure_adapter"):
                try:
                    llm_adapter.ensure_adapter()
                except Exception as exc:  # pragma: no cover - defensive logging
                    operation_logger.debug(
                        "LLM adapter initialization check failed: %s", exc
                    )

            if llm_adapter and hasattr(llm_adapter, "provider_status"):
                try:
                    provider_status = llm_adapter.provider_status()  # type: ignore[attr-defined]
                except Exception as exc:  # pragma: no cover - defensive logging
                    operation_logger.debug("LLM provider status unavailable: %s", exc)
                    provider_status = None

                if provider_status and provider_status.get("status") == "noop":
                    provider_name = provider_status.get("provider", "unknown")
                    reason = provider_status.get("reason") or "provider unavailable"
                    ctx.obj.ui.display_warning(
                        f"LLM provider '{provider_name}' unavailable ({reason}); "
                        "running in offline stub mode. No real tests will be generated.",
                        "LLM Offline",
                    )
                    operation_logger.warning(
                        "LLM provider '%s' unavailable (%s); using offline mode",
                        provider_name,
                        reason,
                    )

            planning_cfg: dict[str, Any] = {}
            if ctx.obj.config:
                if isinstance(ctx.obj.config, dict):
                    planning_cfg = ctx.obj.config.get("planning", {}) or {}
                else:
                    try:
                        planning_model = getattr(ctx.obj.config, "planning", None)
                        if planning_model is not None:
                            planning_cfg = planning_model.model_dump()
                    except AttributeError:
                        planning_cfg = {}

            planning_enabled = planning_cfg.get("enabled", True)
            should_run_planning = planning_enabled or plan_first

            # Execute planning workflow when enabled (default) or explicitly requested
            if should_run_planning:
                operation_logger.info(
                    "🔍 [cyan]Planning workflow starting[/] - plan must be reviewed before generation"
                )

                try:
                    # Get planning use case from container
                    plan_usecase = ctx.obj.container.get("plan_usecase")

                    if not plan_usecase:
                        ctx.obj.ui.display_warning(
                            "Planning use case not available in container, skipping planning",
                            "Planning Warning",
                        )
                    else:
                        # Build planning request from target files
                        from ..application.generation.services.context_pack import (
                            ContextPackBuilder,
                        )
                        from ..domain.models import PlanningRequest

                        planning_target: Path | None = None

                        discovery_service = ctx.obj.container.get("file_discovery")

                        def _select_planning_candidate(
                            paths: Iterable[pathlib.Path],
                        ) -> pathlib.Path | None:
                            for raw_path in paths:
                                candidate_path = (
                                    raw_path
                                    if isinstance(raw_path, pathlib.Path)
                                    else pathlib.Path(raw_path)
                                )

                                if discovery_service is not None:
                                    try:
                                        filtered = (
                                            discovery_service.filter_existing_files(
                                                [candidate_path], project_path
                                            )
                                        )
                                    except FileDiscoveryError as filter_error:
                                        operation_logger.debug(
                                            "Planning candidate %s rejected by discovery filter: %s",
                                            candidate_path,
                                            filter_error,
                                        )
                                        continue

                                    if not filtered:
                                        operation_logger.debug(
                                            "Planning candidate %s excluded by discovery configuration",
                                            candidate_path,
                                        )
                                        continue

                                    candidate_path = pathlib.Path(filtered[0])
                                else:
                                    # Best-effort exclusion when discovery container is unavailable
                                    path_parts = set(candidate_path.parts)
                                    if {".venv", "venv", "site-packages"}.intersection(
                                        path_parts
                                    ):
                                        operation_logger.debug(
                                            "Planning candidate %s skipped due to virtual environment heuristic",
                                            candidate_path,
                                        )
                                        continue

                                try:
                                    return candidate_path.resolve()
                                except OSError:
                                    return candidate_path.absolute()

                            return None

                        if target_files:
                            preliminary = target_files[0]
                            planning_target = _select_planning_candidate([preliminary])
                        else:
                            planning_target = None
                            try:
                                if discovery_service is not None:
                                    discovered = (
                                        discovery_service.discover_source_files(
                                            project_path
                                        )
                                    )
                                    planning_target = _select_planning_candidate(
                                        pathlib.Path(path) for path in discovered
                                    )
                            except FileDiscoveryError as discovery_error:
                                operation_logger.debug(
                                    "Automatic planning target discovery failed: %s",
                                    discovery_error,
                                )
                            except Exception as discovery_error:  # pragma: no cover - unexpected errors
                                operation_logger.debug(
                                    "Automatic planning target discovery failed unexpectedly: %s",
                                    discovery_error,
                                )

                            if planning_target is None:
                                try:
                                    planning_target = _select_planning_candidate(
                                        project_path.rglob("*.py")
                                    )
                                except Exception as glob_error:
                                    operation_logger.debug(
                                        "Fallback planning target search failed: %s",
                                        glob_error,
                                    )

                        if not planning_target or not planning_target.is_file():
                            if ctx.obj.ui:
                                ctx.obj.ui.display_warning(
                                    "No Python source file found for planning; skipping planning step",
                                    "Planning Skipped",
                                )
                            operation_logger.warning(
                                "Skipping planning workflow: no valid source file available (project_path=%s)",
                                project_path,
                            )
                        else:
                            # Create planning request
                            planning_request = PlanningRequest(
                                target_file=planning_target,
                                target_object="module",  # For now, plan at module level
                                project_root=project_path,
                                prompt_customization=None,
                            )

                            # Build context pack for planning (prefer DI-wired builder)
                            context_pack_builder = ctx.obj.container.get(
                                "context_pack_builder"
                            )
                            if not context_pack_builder:
                                context_pack_builder = ContextPackBuilder(
                                    file_discovery_service=discovery_service
                                )
                            try:
                                context_pack = context_pack_builder.build_context_pack(
                                    target_file=planning_target,
                                    target_object="module",
                                    project_root=project_path,
                                )
                            except ValueError as build_error:
                                ctx.obj.ui.display_error(
                                    f"Planning failed: {build_error}", "Planning Failed"
                                )
                                operation_logger.error(
                                    "Context pack building failed: %s", build_error
                                )
                                sys.exit(1)

                            if not context_pack:
                                ctx.obj.ui.display_error(
                                    "Failed to build context pack for planning",
                                    "Planning Failed",
                                )
                                operation_logger.error(
                                    "💥 Context pack building failed"
                                )
                                sys.exit(1)

                            # Override auto_accept config if flag is set
                            if auto_accept_plan:
                                plan_usecase._config["planning"] = (
                                    plan_usecase._config.get("planning", {})
                                )
                                plan_usecase._config["planning"]["auto_accept"] = True
                                operation_logger.info(
                                    "⚡ [yellow]Auto-accept plan enabled[/] - skipping interactive review"
                                )
                            elif not planning_cfg.get("auto_accept", False):
                                plan_usecase._config["planning"] = (
                                    plan_usecase._config.get("planning", {})
                                )
                                plan_usecase._config["planning"]["auto_accept"] = False

                            # Execute planning workflow
                            operation_logger.info(
                                "🧠 [cyan]Generating test plan via orchestrator[/]"
                            )
                            planning_result = asyncio.run(
                                plan_usecase.execute_planning_workflow(
                                    planning_request, context_pack
                                )
                            )

                            # Check if plan was accepted
                            if not planning_result.accepted:
                                ctx.obj.ui.display_info(
                                    "Plan rejected, aborting test generation",
                                    "Planning Complete",
                                )
                                operation_logger.info(
                                    "⚠️ [yellow]Plan rejected by user[/]"
                                )
                                return

                            ctx.obj.ui.display_success(
                                f"Plan accepted (ID: {planning_result.plan_option.plan_id[:12]}...)",
                                "Planning Complete",
                            )
                            operation_logger.info(
                                f"✓ [green]Plan accepted[/] - proceeding with generation (plan_id={planning_result.plan_option.plan_id[:8]})"
                            )

                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    ctx.obj.ui.display_error(
                        f"Planning workflow failed: {e}", "Planning Error"
                    )
                    operation_logger.error(f"💥 Planning workflow failed: {e}")
                    if ctx.obj.verbose:
                        import traceback

                        ctx.obj.ui.display_info(
                            traceback.format_exc(), "Debug Information"
                        )
                    sys.exit(1)

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
                # Manual-fix integration
                "manual_fix": {
                    "on_fail": manual_fix_on_fail,
                    "auto_accept": auto_accept_fixes,
                },
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

            cost_summary = _collect_cost_summary(
                ctx.obj.container, run_start, operation_logger
            )
            if cost_summary is not None:
                results["cost_summary"] = cost_summary

            # Display results using enhanced UI components
            if results.get("success"):
                _display_generation_results(results, ctx.obj.ui)
                _display_cost_summary(cost_summary, ctx.obj.ui)
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
                _display_cost_summary(cost_summary, ctx.obj.ui)
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
        if run_start:
            cost_summary = _collect_cost_summary(
                ctx.obj.container, run_start, operation_logger
            )
            _display_cost_summary(cost_summary, ctx.obj.ui)
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
    type=click.Choice(["detailed", "summary", "json", "xml"], case_sensitive=False),
    multiple=True,
    default=["detailed"],
    help="Output format",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path(".artifacts/coverage"),
    help="Output directory for coverage artifacts (XML)",
)
@click.option(
    "--include",
    "-I",
    "include_patterns",
    multiple=True,
    help="Include specific source directories (can be used multiple times)",
)
@click.option(
    "--omit",
    "-O",
    "omit_patterns",
    multiple=True,
    help="Omit patterns from coverage (can be used multiple times)",
)
@click.pass_context
def coverage(
    ctx: click.Context,
    project_path: Path,
    source_files: tuple[Path, ...],
    test_files: tuple[Path, ...],
    output_format: tuple[str, ...],
    output_dir: Path,
    include_patterns: tuple[str, ...],
    omit_patterns: tuple[str, ...],
) -> None:
    """Measure and report code coverage."""
    try:
        # Get use case from container
        coverage_usecase = ctx.obj.container["coverage_usecase"]

        # Configure parameters
        config_overrides = {
            "output_formats": list(output_format),
            "data_dir": str(output_dir),
            # xml_output will be derived inside adapter; we pass directory
        }

        # Add include/omit if provided
        if include_patterns:
            config_overrides["include"] = list(include_patterns)
        if omit_patterns:
            config_overrides["omit"] = list(omit_patterns)

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
    "project_path", type=click.Path(exists=True, path_type=Path), default="."
)
@click.option(
    "--target-file",
    "target_file",
    type=click.Path(path_type=Path),
    required=True,
    help="Single source file to plan for (Class.method or function via --target-object)",
)
@click.option(
    "--target-object",
    "target_object",
    type=str,
    required=True,
    help="Target object (e.g., 'Class.method' or 'function')",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=Path(".artifacts/plan.json"),
    help="Output file path for the plan JSON",
)
@click.option(
    "--max-plan-retries",
    type=int,
    default=2,
    help="Maximum retries for PLAN stage (overrides config)",
)
@click.option(
    "--disable-gates/--enable-gates",
    "disable_gates",
    default=None,
    help="Disable or enable quality gates for the session (None = leave as config)",
)
@click.pass_context
def plan(
    ctx: click.Context,
    project_path: Path,
    target_file: Path,
    target_object: str,
    output_path: Path,
    max_plan_retries: int,
    disable_gates: bool | None,
) -> None:
    """Run PLAN stage only and emit structured plan JSON."""
    run_start: datetime | None = None
    try:
        if ctx.obj.dry_run:
            ctx.obj.ui.display_info("DRY RUN: Skipping PLAN execution", "Dry Run Mode")
            return

        run_start = datetime.now()

        # Resolve dependencies from container
        generate_usecase = ctx.obj.container["generate_usecase"]
        llm_adapter = ctx.obj.container.get("llm_adapter")

        if llm_adapter and hasattr(llm_adapter, "ensure_adapter"):
            try:
                llm_adapter.ensure_adapter()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("LLM adapter initialization check failed: %s", exc)

        if llm_adapter and hasattr(llm_adapter, "provider_status"):
            try:
                provider_status = llm_adapter.provider_status()  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("LLM provider status unavailable: %s", exc)
                provider_status = None

            if provider_status and provider_status.get("status") == "noop":
                provider_name = provider_status.get("provider", "unknown")
                reason = provider_status.get("reason") or "provider unavailable"
                ctx.obj.ui.display_warning(
                    f"LLM provider '{provider_name}' unavailable ({reason}); "
                    "planning will use offline stubs.",
                    "LLM Offline",
                )
                logger.warning(
                    "LLM provider '%s' unavailable (%s); planning running in offline mode",
                    provider_name,
                    reason,
                )

        # Build ContextPack for target
        context_assembler = ContextAssembler(
            context_port=ctx.obj.container["context_adapter"],
            parser_port=ctx.obj.container["parser_adapter"],
            config=ctx.obj.config.model_dump(),
        )
        context_pack_builder = ContextPackBuilder(
            context_assembler=context_assembler,
            file_discovery_service=ctx.obj.container.get("file_discovery"),
        )

        try:
            context_pack = context_pack_builder.build_context_pack(
                target_file=target_file,
                target_object=target_object,
                project_root=project_path,
            )
        except ValueError as build_error:
            ctx.obj.ui.display_error(f"Planning failed: {build_error}", "PLAN Failed")
            sys.exit(1)

        if context_pack is None:
            ctx.obj.ui.display_error(
                "Failed to build ContextPack for planning", "PLAN Failed"
            )
            sys.exit(1)

        # Create orchestrator with overrides
        orchestrator: LLMOrchestrator = (
            generate_usecase._llm_orchestrator
        )  # reuse wiring
        # Apply retry override if provided
        try:
            orchestrator._max_plan_retries = max(0, int(max_plan_retries))
        except Exception:
            pass

        # Optionally override quality gates toggle in usecase config
        if disable_gates is not None:
            try:
                generate_usecase._config["enable_quality_gates"] = not disable_gates
            except Exception:
                pass

        with ctx.obj.ui.create_status_spinner("Running PLAN stage..."):
            plan_result = orchestrator.plan_stage(
                context_pack=context_pack, project_root=project_path
            )

        cost_summary = _collect_cost_summary(ctx.obj.container, run_start, logger)
        _display_cost_summary(cost_summary, ctx.obj.ui)

        # Ensure output directory
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Write JSON plan
        import json as _json

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(_json.dumps(plan_result, ensure_ascii=False, indent=2))

        ctx.obj.ui.display_success(f"PLAN written to {output_path}", "Plan Complete")
        if ctx.obj.verbose:
            ctx.obj.ui.console.print(
                _json.dumps(plan_result, ensure_ascii=False, indent=2)
            )

    except Exception as e:
        ctx.obj.ui.display_error(f"PLAN failed: {e}", "Plan Error")
        if ctx.obj.verbose:
            import traceback

            ctx.obj.ui.display_info(traceback.format_exc(), "Debug Information")
        if run_start:
            cost_summary = _collect_cost_summary(ctx.obj.container, run_start, logger)
            _display_cost_summary(cost_summary, ctx.obj.ui)
        sys.exit(1)


@app.command("manual-fix")
@click.argument(
    "project_path", type=click.Path(exists=True, path_type=Path), default="."
)
@click.option(
    "--target-file",
    "target_file",
    type=click.Path(path_type=Path),
    required=True,
    help="Source file containing the target under test",
)
@click.option(
    "--target-object",
    "target_object",
    type=str,
    required=True,
    help="Target object (e.g., 'Class.method' or 'function')",
)
@click.option(
    "--trace-excerpt",
    "trace_excerpt",
    type=str,
    required=False,
    default="",
    help="Optional traceback excerpt indicating the suspected product bug",
)
@click.option(
    "--notes",
    type=str,
    required=False,
    default="",
    help="Additional notes for the manual fix prompt",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=Path(".artifacts/manual_fix.json"),
    help="Output file path for the manual-fix response JSON",
)
@click.option(
    "--auto-accept-fixes",
    is_flag=True,
    help="Auto-accept manual-fix recommendation without prompt",
)
@click.pass_context
def manual_fix(
    ctx: click.Context,
    project_path: Path,
    target_file: Path,
    target_object: str,
    trace_excerpt: str,
    notes: str,
    output_path: Path,
    auto_accept_fixes: bool,
) -> None:
    """Run MANUAL FIX stage to produce failing test + bug note for real bug scenarios."""
    run_start: datetime | None = None
    try:
        if ctx.obj.dry_run:
            ctx.obj.ui.display_info(
                "DRY RUN: Skipping MANUAL FIX execution", "Dry Run Mode"
            )
            return

        run_start = datetime.now()

        # Build ContextPack for target
        context_assembler = ContextAssembler(
            context_port=ctx.obj.container["context_adapter"],
            parser_port=ctx.obj.container["parser_adapter"],
            config=ctx.obj.config.model_dump(),
        )
        context_pack_builder = ContextPackBuilder(
            context_assembler=context_assembler,
            file_discovery_service=ctx.obj.container.get("file_discovery"),
        )
        try:
            context_pack = context_pack_builder.build_context_pack(
                target_file=target_file,
                target_object=target_object,
                project_root=project_path,
            )
        except ValueError as build_error:
            ctx.obj.ui.display_error(
                f"Manual-fix setup failed: {build_error}", "Manual Fix Failed"
            )
            sys.exit(1)
        if context_pack is None:
            ctx.obj.ui.display_error(
                "Failed to build ContextPack for manual-fix", "Manual Fix Failed"
            )
            sys.exit(1)

        # Use dedicated use case for manual-fix
        try:
            from ..application.manual_fix_usecase import ManualFixGuidanceUseCase
            from ..domain.manual_fix import ManualFixRequest
        except Exception as e:
            ctx.obj.ui.display_error(
                f"Missing manual-fix components: {e}", "Manual Fix Error"
            )
            sys.exit(1)

        # Build use case from container wiring
        orchestrator: LLMOrchestrator = ctx.obj.container[
            "generate_usecase"
        ]._llm_orchestrator
        context_assembler = ContextAssembler(
            context_port=ctx.obj.container["context_adapter"],
            parser_port=ctx.obj.container["parser_adapter"],
            config=ctx.obj.config.model_dump(),
        )
        context_pack_builder = ContextPackBuilder(
            context_assembler=context_assembler,
            file_discovery_service=ctx.obj.container.get("file_discovery"),
        )

        # Simple CLI presenter using UI adapter
        class _CliPresenter:
            def __init__(self, ui):
                self.ui = ui

            def present(self, recommendation, *, dry_run: bool = False) -> bool:
                self.ui.display_info(
                    "A manual-fix recommendation is ready (failing test + bug note).",
                    "Manual Fix Guidance",
                )
                self.ui.console.print(f"Hash: {recommendation.recommendation_hash}")
                self.ui.console.print(
                    "Preview: writing Markdown artifact on acceptance..."
                )
                if dry_run:
                    return False
                return click.confirm(
                    "Accept and persist manual-fix artifact?", default=False
                )

        presenter = _CliPresenter(ctx.obj.ui)

        # Override auto-accept from flag if provided
        cfg = ctx.obj.config.model_dump().copy()
        try:
            cfg.setdefault("manual_fix", {})
            if auto_accept_fixes:
                cfg["manual_fix"]["auto_accept"] = True
        except Exception:
            pass

        usecase = ManualFixGuidanceUseCase(
            llm_orchestrator=orchestrator,
            parser_port=ctx.obj.container["parser_adapter"],
            context_assembler=context_assembler,
            context_pack_builder=context_pack_builder,
            telemetry_port=ctx.obj.container["telemetry_adapter"],
            presenter=presenter,
            config=cfg,
        )

        req = ManualFixRequest(
            project_root=project_path,
            target_file=target_file,
            target_object=target_object,
            trace_excerpt=trace_excerpt,
            notes=notes,
        )

        with ctx.obj.ui.create_status_spinner("Running MANUAL FIX stage..."):
            import asyncio as _asyncio

            mf_result = _asyncio.run(usecase.run(req))

        cost_summary = _collect_cost_summary(ctx.obj.container, run_start, logger)

        if not mf_result.accepted:
            ctx.obj.ui.display_info(
                "Manual-fix not accepted or no recommendation.", "Manual Fix"
            )
            _display_cost_summary(cost_summary, ctx.obj.ui)
            sys.exit(0)

        # Optionally write JSON sidecar if --output provided
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        import json as _json

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                _json.dumps(
                    {
                        "accepted": mf_result.accepted,
                        "artifact_path": str(mf_result.artifact_path)
                        if mf_result.artifact_path
                        else None,
                        "hash": mf_result.recommendation.recommendation_hash
                        if mf_result.recommendation
                        else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

        ctx.obj.ui.display_success(
            f"Manual fix artifact: {mf_result.artifact_path}", "Manual Fix Complete"
        )
        _display_cost_summary(cost_summary, ctx.obj.ui)

    except Exception as e:
        ctx.obj.ui.display_error(f"Manual fix failed: {e}", "Manual Fix Error")
        if ctx.obj.verbose:
            import traceback

            ctx.obj.ui.display_info(traceback.format_exc(), "Debug Information")
        if run_start:
            cost_summary = _collect_cost_summary(ctx.obj.container, run_start, logger)
            _display_cost_summary(cost_summary, ctx.obj.ui)
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
@click.option("--web", is_flag=True, help="(deprecated) web mode placeholder")
@click.option("--port", type=int, default=8080, help="(deprecated) port for web mode")
@click.pass_context
def tui(ctx: click.Context, web: bool, port: int) -> None:
    """Inform users that the legacy Textual TUI has been retired."""
    ctx.obj.ui.display_warning(
        "The interactive Textual TUI is no longer available. Use the standard CLI commands instead.",
        "TUI Unavailable",
    )
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


def _collect_cost_summary(
    container: dict[str, Any] | None,
    run_start: datetime | None,
    logger_obj: Any | None = None,
) -> dict[str, Any] | None:
    """Collect cost summary from the configured cost adapter."""
    if container is None or run_start is None:
        return None

    cost_adapter = container.get("cost_adapter")
    if not cost_adapter:
        return None

    try:
        summary = cost_adapter.get_summary(
            start_time=run_start, end_time=datetime.now()
        )
    except TypeError:
        summary = cost_adapter.get_summary()
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger_obj:
            logger_obj.debug("Cost summary unavailable: %s", exc)
        return None

    if not isinstance(summary, dict):
        return None

    usage_stats = summary.get("usage_stats", {}) or {}

    total_cost_raw = summary.get("total_cost", 0) or 0
    try:
        total_cost = float(total_cost_raw)
    except (TypeError, ValueError):
        total_cost = 0.0

    def _to_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    total_tokens = _to_int(usage_stats.get("total_tokens"))
    total_calls = _to_int(usage_stats.get("total_api_calls"))

    return {
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "total_api_calls": total_calls,
    }


def _display_cost_summary(
    cost_summary: dict[str, Any] | None, ui_adapter: EnhancedUIAdapter | None
) -> None:
    """Display cost usage summary via the appropriate UI."""
    if not ui_adapter:
        return

    if not cost_summary:
        message = "Cost: unavailable • Tokens: n/a • API calls: n/a"
        if ui_adapter.ui_style == UIStyle.MINIMAL:
            ui_adapter.console.print(message)
        else:
            ui_adapter.display_warning(message, "Usage Summary")
        return

    total_cost = float(cost_summary.get("total_cost", 0.0) or 0.0)
    total_tokens = int(cost_summary.get("total_tokens", 0) or 0)
    total_calls = int(cost_summary.get("total_api_calls", 0) or 0)

    message = (
        f"Cost: ${total_cost:.4f} • Tokens: {total_tokens} • API calls: {total_calls}"
    )

    if ui_adapter.ui_style == UIStyle.MINIMAL:
        ui_adapter.console.print(message)
    else:
        ui_adapter.display_info(message, "Usage Summary")


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

    # Display report file paths if generated
    reports = results.get("reports", {})
    for format_name, report_data in reports.items():
        if format_name == "xml":
            report_path = report_data.get("report_content", "")
            if report_path and Path(report_path).exists():
                ui_adapter.display_success(
                    f"XML report: {report_path}",
                    "Report Generated",
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
        ui_adapter.console.print("\n[bold]Generation history:[/]")
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
    else:
        ui_adapter.display_info("No generation history available", "Generation History")

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
        else:
            ui_adapter.display_info("No statistics available", "Summary Statistics")
    else:
        ui_adapter.display_info("No statistics available", "Summary Statistics")


# Add evaluation commands
add_evaluation_commands(app)

# Add utility commands
add_utility_commands(app)

# Add model catalog commands
add_model_commands(app)


if __name__ == "__main__":
    app()
