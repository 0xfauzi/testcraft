"""Generate command implementation."""

import asyncio
import sys
from pathlib import Path
from typing import Any

import click

from ...adapters.io.enhanced_logging import get_operation_logger
from ...adapters.io.enhanced_ui import EnhancedUIAdapter
from ...adapters.io.ui_rich import UIStyle
from ...application.environment.preflight import EnvironmentValidator
from ..project_root_utils import derive_project_root


@click.command()
@click.argument(
    "project_path", type=click.Path(path_type=Path), default=".", required=False
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
    help="Enable immediate write-and-refine per file (default: enabled)"
)
@click.option(
    "--max-refine-workers", 
    type=int, 
    default=2, 
    help="Maximum concurrent pytest/refine workers"
)
@click.option(
    "--keep-failed-writes", 
    is_flag=True, 
    help="Keep test files that fail to write or have syntax errors"
)
@click.option(
    "--disable-ruff", 
    is_flag=True, 
    help="Disable Ruff formatting (use Black+isort instead) to avoid timeouts"
)
@click.option(
    "--from-planning-session",
    type=str,
    help="Use stored planning session by ID (use 'latest' for most recent session)"
)
@click.option(
    "--selected-only",
    is_flag=True,
    help="Only generate tests for elements selected in the planning session"
)
@click.option(
    "--skip-planning",
    is_flag=True,
    help="Skip the automatic planning phase and generate tests directly"
)
@click.pass_context
def generate(
    ctx: click.Context,
    project_path: Path | None,
    target_files: tuple[Path, ...],
    batch_size: int,
    streaming: bool,
    force: bool,
    immediate: bool,
    max_refine_workers: int,
    keep_failed_writes: bool,
    disable_ruff: bool,
    from_planning_session: str | None,
    selected_only: bool,
    skip_planning: bool,
) -> None:
    """Generate tests for Python source files. Can use stored planning sessions."""
    operation_logger = get_operation_logger("generate")
    
    try:
        # Derive actual project root considering target files
        actual_project_path = derive_project_root(
            project_path=project_path,
            target_files=list(target_files) if target_files else None
        )
        
        operation_logger.info(f"Using project root: {actual_project_path}")
        
        # AUTO-PLANNING PHASE (unless skipped or using existing session)
        planning_session_id = None
        if not skip_planning and not from_planning_session:
            try:
                operation_logger.info("Starting automatic planning phase...")
                ctx.obj.ui.display_info("Analyzing code and generating test plans before generation...", "Auto-Planning")
                
                # Get planning use case
                planning_use_case = ctx.obj.container["planning_use_case"]
                
                # Generate planning session
                planning_session = planning_use_case.generate_planning_session(
                    project_path=str(actual_project_path),
                    target_files=[str(f) for f in target_files] if target_files else None
                )
                
                if not planning_session.items:
                    ctx.obj.ui.display_info("No eligible test elements found for planning", "Planning Complete")
                    return
                
                # Display enhanced planning results
                from ..display_helpers import display_enhanced_planning_table
                display_enhanced_planning_table(planning_session, ctx.obj.ui, ctx.obj.rich_cli)
                
                # Get user selection for generation
                if sys.stdout.isatty():
                    if ctx.obj.dry_run:
                        # In dry-run mode, still show the selection interface but don't actually generate
                        ctx.obj.ui.display_info("DRY RUN: Showing planning selection interface for demonstration", "Demo Mode")
                    
                    from ..display_helpers import interactive_generation_planning_selection
                    selected_keys = interactive_generation_planning_selection(planning_session, ctx.obj.ui, ctx.obj.dry_run)
                    
                    if not selected_keys:
                        ctx.obj.ui.display_info("No elements selected for generation", "Generation Cancelled")
                        return
                    
                    # Update session with selections (even in dry-run for demo)
                    if not ctx.obj.dry_run:
                        updated_session = planning_use_case.update_session_selections(
                            planning_session.session_id, selected_keys
                        )
                        if updated_session:
                            planning_session = updated_session
                    
                    # Use this session for generation
                    planning_session_id = planning_session.session_id
                    from_planning_session = planning_session_id
                    selected_only = True
                    
                    if ctx.obj.dry_run:
                        ctx.obj.ui.display_success(
                            f"DRY RUN: Would proceed with generation for {len(selected_keys)} selected elements",
                            "Planning Demo Complete"
                        )
                        return  # Exit here for dry-run mode
                    else:
                        ctx.obj.ui.display_success(
                            f"Proceeding with generation for {len(selected_keys)} selected elements",
                            "Planning Complete"
                        )
                else:
                    # Non-interactive mode - show plans but don't require selection
                    ctx.obj.ui.display_info(
                        f"Non-interactive mode: Generated {len(planning_session.items)} plans. "
                        "Use --skip-planning or run interactively to make selections.",
                        "Planning Info"
                    )
                    # Continue with all plans
                    
            except Exception as e:
                operation_logger.warning(f"Auto-planning failed: {e}")
                ctx.obj.ui.display_warning(
                    f"Planning phase failed ({e}), proceeding with direct generation",
                    "Planning Warning"
                )
        
        with operation_logger.operation_context(
            "test_generation",
            project_path=str(actual_project_path),
            target_files_count=len(target_files),
            batch_size=batch_size,
            immediate_mode=immediate,
            planning_enabled=not skip_planning
        ):
            if ctx.obj.dry_run:
                ctx.obj.ui.display_info(
                    "DRY RUN: No tests will actually be generated", "Dry Run Mode"
                )
                operation_logger.info("🔍 [yellow]Dry run mode activated[/] - no files will be modified")

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
                operation_logger.error("💥 Environment preflight failed; aborting generate")
                sys.exit(1)

            # Get use case from container (after preflight passes)
            generate_usecase = ctx.obj.container["generate_usecase"]

            # Configure generation parameters
            # First get TOML config overrides, then merge CLI flags
            from ...application.generation.config import GenerationConfig
            toml_overrides = GenerationConfig.map_testcraft_config_to_overrides(ctx.obj.config)
            
            cli_overrides = {
                "batch_size": batch_size,
                "enable_streaming": streaming,
                "force_overwrite": force,
                "immediate_refinement": immediate,
                "max_refine_workers": max_refine_workers,
                "keep_failed_writes": keep_failed_writes,
                "disable_ruff_format": disable_ruff,
            }
            
            # Merge TOML config with CLI overrides (CLI takes precedence)
            config_overrides = {**toml_overrides, **cli_overrides}

            operation_logger.info(f"[primary]config:[/] batch={batch_size}, immediate={immediate}, workers={max_refine_workers}")

            # Use enhanced live file tracking for real-time status updates
            from ...adapters.io.file_status_tracker import LiveFileTracking, FileStatus
            
            # Get file paths for tracking
            file_paths = []
            if target_files:
                file_paths = [str(f) for f in target_files]
            else:
                # Discover files in project (simplified for demo)
                try:
                    project = actual_project_path
                    file_paths = [str(f) for f in project.rglob("*.py") 
                                if not str(f).startswith(str(project / "tests")) 
                                and not str(f).name.startswith("test_")][:10]  # Limit for demo
                except Exception:
                    file_paths = []
            
            # Auto-minimal for immediate mode with small file count unless UI explicitly classic
            if immediate and len(file_paths) <= 3 and not ctx.obj.ui_flag_explicit:
                ctx.obj.ui_style = UIStyle.MINIMAL
                ctx.obj.ui = EnhancedUIAdapter(ctx.obj.ui.console, enable_rich_logging=False, ui_style=ctx.obj.ui_style)
                from ...adapters.io.enhanced_logging import LoggerManager, LogMode
                LoggerManager.set_log_mode(LogMode.MINIMAL, verbose=ctx.obj.verbose, quiet=ctx.obj.quiet)

            # Skip live tracker entirely for minimal mode when file count <= 3
            if file_paths and len(file_paths) > 1 and not (ctx.obj.ui_style == UIStyle.MINIMAL and len(file_paths) <= 3):
                # Use live file tracking for multiple files
                with LiveFileTracking(ctx.obj.ui, "TestCraft Test Generation") as live_file_tracker:
                    file_status_tracker = live_file_tracker.initialize_and_start(file_paths)
                    
                    # Inject the status tracker into the generation pipeline
                    generate_usecase.set_status_tracker(file_status_tracker)
                    
                    operation_logger.info(f"[accent]live tracking enabled[/] for {len(file_paths)} files")
                    
                    # Handle planning session parameters
                    planning_kwargs = {}
                    if from_planning_session:
                        session_id = "latest" if from_planning_session == "latest" else from_planning_session
                        # Get the actual session ID if "latest" was specified
                        if session_id == "latest":
                            planning_use_case = ctx.obj.container["planning_use_case"]
                            latest_session = planning_use_case.get_planning_session()
                            if latest_session:
                                session_id = latest_session.session_id
                                operation_logger.info(f"Using latest planning session: {session_id}")
                            else:
                                ctx.obj.ui.display_error("No planning session found. Run 'testcraft plan' first.", "Planning Required")
                                sys.exit(1)
                        
                        planning_kwargs["from_planning_session_id"] = session_id
                        planning_kwargs["selected_only"] = selected_only
                        operation_logger.info(f"Generating with planning session {session_id}, selected_only={selected_only}")
                    
                    # Run the actual generation with integrated live tracking
                    results = asyncio.run(
                        generate_usecase.generate_tests(
                            project_path=actual_project_path,
                            target_files=list(target_files) if target_files else None,
                            **config_overrides,
                            **planning_kwargs,
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
                with ctx.obj.ui.create_operation_tracker("Test Generation", total_steps=4) as tracker:
                    tracker.advance_step("Initializing generation pipeline", 1)
                    
                    # Handle planning session parameters
                    planning_kwargs = {}
                    if from_planning_session:
                        session_id = "latest" if from_planning_session == "latest" else from_planning_session
                        # Get the actual session ID if "latest" was specified
                        if session_id == "latest":
                            planning_use_case = ctx.obj.container["planning_use_case"]
                            latest_session = planning_use_case.get_planning_session()
                            if latest_session:
                                session_id = latest_session.session_id
                                operation_logger.info(f"Using latest planning session: {session_id}")
                            else:
                                ctx.obj.ui.display_error("No planning session found. Run 'testcraft plan' first.", "Planning Required")
                                sys.exit(1)
                        
                        planning_kwargs["from_planning_session_id"] = session_id
                        planning_kwargs["selected_only"] = selected_only
                        operation_logger.info(f"Generating with planning session {session_id}, selected_only={selected_only}")
                    
                    # Run generation asynchronously
                    results = asyncio.run(
                        generate_usecase.generate_tests(
                            project_path=actual_project_path,
                            target_files=list(target_files) if target_files else None,
                            **config_overrides,
                            **planning_kwargs,
                        )
                    )
                    
                    tracker.advance_step("Processing results", 1)

            # Display results with improved severity handling
            from ...adapters.io.enhanced_logging import LoggerManager
            severity = LoggerManager.map_result_to_severity(results)
            
            if results.get("success"):
                from ..display_helpers import display_generation_results
                display_generation_results(results, ctx.obj.ui, severity=severity)
                
                # Performance summary with severity-aware metrics
                metrics = {
                    "files_processed": results.get("files_processed", 0),
                    "tests_generated": results.get("tests_generated", 0),
                    "success_rate": results.get("files_written", 0) / max(results.get("files_processed", 1), 1),
                    "success": True,
                    "failed_generations": results.get("failed_generations", 0),
                    "refine_exhausted_count": results.get("refine_exhausted_count", 0),
                }
                operation_logger.performance_summary("test_generation", metrics)
                
                # Display next actions
                next_actions = []
                if results.get("files_written", 0) > 0:
                    next_actions.append("pytest -q")
                if results.get("files_processed", 0) > 0:
                    next_actions.append("testcraft status")
                report_path = results.get("report_export_path")
                if report_path:
                    next_actions.append(f"report: {report_path}")
                else:
                    next_actions.append("report: artifacts/run.json")
                
                ctx.obj.ui.display_next_actions(next_actions)
                
            else:
                error_msg = results.get("error_message", "Unknown error occurred")
                suggestions = [
                    "Check if the project path contains valid Python files",
                    "Verify your LLM API keys are configured correctly",
                    "Try reducing batch size or disabling immediate mode"
                ]
                ctx.obj.ui.display_error_with_suggestions(error_msg, suggestions, "Generation Failed")
                operation_logger.error(f"💥 Generation failed: {error_msg}")
                sys.exit(1)

    except Exception as e:
        suggestions = [
            "Check if the project directory exists and is readable",
            "Verify your configuration file is valid",
            "Try running with --verbose for more details"
        ]
        ctx.obj.ui.display_error_with_suggestions(f"Test generation failed: {e}", suggestions, "Generation Error")
        operation_logger.error_with_context("Test generation failed", e, suggestions, fatal=True)
        sys.exit(1)
