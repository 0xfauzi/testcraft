"""Main CLI entry point for TestCraft."""

import asyncio
import sys
from pathlib import Path

import click

from ..adapters.io.enhanced_logging import get_operation_logger
from .bootstrap import initialize_cli_context, load_configuration_and_dependencies
from .commands.generate import generate
from .context import ClickContext
from .display_helpers import (
    display_analysis_results,
    display_coverage_results,
    display_status_results,
)
from .evaluation_commands import add_evaluation_commands
from .project_root_utils import derive_project_root
from .utility_commands import add_utility_commands


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Reduce output: set log level to WARNING and hide INFO")
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
def app(ctx: click.Context, config: Path | None, verbose: bool, quiet: bool, dry_run: bool, ui: str | None, compact: bool) -> None:
    """TestCraft - AI-powered test generation tool for Python projects."""
    # Initialize CLI context with UI, logging, and configuration
    ui_flag = "minimal" if compact and not ui else ui
    initialize_cli_context(ctx, str(config) if config else None, verbose, quiet, dry_run, ui_flag, compact)
    
    # Load configuration and create dependency container
    load_configuration_and_dependencies(ctx, str(config) if config else None)


# ============================================================================
# MAIN COMMANDS
# ============================================================================


@app.command()
@click.argument(
    "project_path", type=click.Path(path_type=Path), default=".", required=False
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
    ctx: click.Context, project_path: Path | None, target_files: tuple[Path, ...]
) -> None:
    """Analyze what tests would be generated and why."""
    try:
        # Derive actual project root considering target files
        actual_project_path = derive_project_root(
            project_path=project_path,
            target_files=list(target_files) if target_files else None
        )
        
        # Get use case from container
        analyze_usecase = ctx.obj.container["analyze_usecase"]

        with ctx.obj.ui.create_status_spinner("Analyzing project files..."):
            # Run analysis asynchronously
            results = asyncio.run(
                analyze_usecase.analyze_generation_needs(
                    project_path=actual_project_path,
                    target_files=list(target_files) if target_files else None,
                )
            )

        # Display results using Rich components
        display_analysis_results(results, ctx.obj.ui, ctx.obj.rich_cli)

    except Exception as e:
        ctx.obj.ui.display_error(f"Analysis failed: {e}", "Analysis Error")
        if ctx.obj.verbose:
            import traceback

            ctx.obj.ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


@app.command()
@click.argument(
    "project_path", type=click.Path(path_type=Path), default=".", required=False
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
    project_path: Path | None,
    source_files: tuple[Path, ...],
    test_files: tuple[Path, ...],
    output_format: tuple[str, ...],
) -> None:
    """Measure and report code coverage."""
    try:
        # Derive actual project root considering source and test files
        all_target_files = list(source_files) + list(test_files)
        actual_project_path = derive_project_root(
            project_path=project_path,
            target_files=all_target_files if all_target_files else None
        )
        
        # Get use case from container
        coverage_usecase = ctx.obj.container["coverage_usecase"]

        # Configure parameters
        config_overrides = {"output_formats": list(output_format)}

        with ctx.obj.ui.create_status_spinner("Measuring code coverage..."):
            # Run coverage measurement asynchronously
            results = asyncio.run(
                coverage_usecase.measure_and_report(
                    project_path=actual_project_path,
                    source_files=list(source_files) if source_files else None,
                    test_files=list(test_files) if test_files else None,
                    **config_overrides,
                )
            )

        # Display results using Rich components
        if results.get("success"):
            display_coverage_results(results, ctx.obj.ui, ctx.obj.rich_cli)
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
        display_status_results(results, limit, ctx.obj.ui, ctx.obj.rich_cli)

    except Exception as e:
        ctx.obj.ui.display_error(f"Status retrieval failed: {e}", "Status Error")
        if ctx.obj.verbose:
            import traceback

            ctx.obj.ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


@app.command()
@click.option("--web", is_flag=True, help="Launch TUI in web browser mode")
@click.option("--port", type=int, default=8080, help="Port for web mode (default: 8080)")
@click.pass_context
def tui(ctx: click.Context, web: bool, port: int) -> None:
    """Launch TestCraft's interactive Terminal User Interface (TUI)."""
    try:
        from ..adapters.textual.app import TestCraftTextualApp
        
        if web:
            # Web mode using textual-web
            try:
                import textual_web
                ctx.obj.ui.display_info(
                    f"Launching TestCraft TUI in web browser on port {port}...",
                    "Web Mode"
                )
                
                # Create the app
                app = TestCraftTextualApp()
                
                # Launch in web mode
                # Note: textual-web integration would go here
                # For now, fall back to regular terminal mode
                ctx.obj.ui.display_warning(
                    "Web mode not fully implemented yet, launching in terminal mode",
                    "Fallback"
                )
                app.run()
                
            except ImportError:
                ctx.obj.ui.display_error(
                    "textual-web not available. Install with: pip install textual-web",
                    "Missing Dependency"
                )
                sys.exit(1)
        else:
            # Terminal mode
            ctx.obj.ui.display_info(
                "Launching TestCraft TUI...",
                "Terminal Mode"
            )
            
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


@app.command()
@click.argument(
    "project_path", type=click.Path(path_type=Path), default=".", required=False
)
@click.option(
    "--target-files",
    "-f",
    multiple=True,
    type=click.Path(path_type=Path),
    help="Specific files to plan tests for",
)
@click.option(
    "--auto-accept",
    is_flag=True,
    help="Automatically accept all generated plans without user interaction",
)
@click.option(
    "--accept",
    type=str,
    help="Accept specific plans using patterns (comma-separated include/exclude patterns)",
)
@click.option(
    "--json-out",
    type=click.Path(path_type=Path),
    help="Write planning session results to JSON file",
)
@click.option(
    "--generate",
    is_flag=True,
    help="Immediately proceed to generation using selected plans",
)
@click.pass_context
def plan(
    ctx: click.Context,
    project_path: Path | None,
    target_files: tuple[Path, ...],
    auto_accept: bool,
    accept: str | None,
    json_out: Path | None,
    generate: bool,
) -> None:
    """Generate test plans for eligible code elements."""
    operation_logger = get_operation_logger("plan")
    
    try:
        # Derive actual project root considering target files
        actual_project_path = derive_project_root(
            project_path=project_path,
            target_files=list(target_files) if target_files else None
        )
        
        operation_logger.info(f"Using project root: {actual_project_path}")
        
        with operation_logger.operation_context(
            "test_planning",
            project_path=str(actual_project_path),
            target_files_count=len(target_files),
            auto_accept=auto_accept
        ):
            if ctx.obj.dry_run:
                ctx.obj.ui.display_info(
                    "DRY RUN: Planning session will be generated but not executed", "Dry Run Mode"
                )
                operation_logger.info("🔍 [yellow]Dry run mode activated[/] - planning only, no execution")

            # Get the planning use case from container
            planning_use_case = ctx.obj.container["planning_use_case"]
            
            # Generate planning session
            operation_logger.info("🎯 Generating test plans for eligible elements...")
            ctx.obj.ui.display_info("Analyzing code and generating test plans...", "Planning Session")
            
            session = planning_use_case.generate_planning_session(
                project_path=str(actual_project_path),
                target_files=[str(f) for f in target_files] if target_files else None
            )
            
            if not session.items:
                ctx.obj.ui.display_info("No eligible test elements found for planning", "Planning Complete")
                return

            # Display planning results - check if enhanced method exists, fallback to basic display
            if hasattr(ctx.obj.ui, 'render_planning_results'):
                ctx.obj.ui.render_planning_results(session)
            else:
                # Fallback to basic display
                from .display_helpers import display_planning_results_fallback
                display_planning_results_fallback(session, ctx.obj.ui, ctx.obj.rich_cli)
            
            # Handle user selection
            selected_keys = []
            
            if auto_accept:
                # Accept all plans
                selected_keys = []
                for item in session.items:
                    element_type = item.element.type.value if hasattr(item.element.type, 'value') else str(item.element.type)
                    source_file = ""
                    for tag in item.tags:
                        if tag.startswith("source_file:"):
                            source_file = tag.replace("source_file:", "")
                            break
                    key = f"{source_file}::{element_type}::{item.element.name}::{item.element.line_range[0]}-{item.element.line_range[1]}"
                    selected_keys.append(key)
                operation_logger.info(f"Auto-accepted all {len(selected_keys)} plans")
                
            elif accept:
                # Process accept patterns
                from .plan_selection import process_accept_patterns
                selected_keys = process_accept_patterns(session, accept)
                operation_logger.info(f"Pattern-selected {len(selected_keys)} plans")
                
            else:
                # Interactive selection (if TTY)
                if sys.stdout.isatty() and not ctx.obj.dry_run:
                    from .plan_selection import interactive_plan_selection
                    selected_keys = interactive_plan_selection(session, ctx.obj.ui)
                else:
                    ctx.obj.ui.display_info(
                        "Non-interactive mode: Use --auto-accept or --accept patterns for selection",
                        "Selection Required"
                    )
                    return
            
            # Update session with selections
            if selected_keys:
                updated_session = planning_use_case.update_session_selections(
                    session.session_id, selected_keys
                )
                if updated_session:
                    session = updated_session
                
                ctx.obj.ui.display_success(
                    f"Selected {len(selected_keys)} out of {len(session.items)} planned elements",
                    "Selection Complete"
                )
            
            # Save to JSON if requested
            if json_out:
                from .display_helpers import save_planning_session_json
                save_planning_session_json(session, json_out)
                ctx.obj.ui.display_success(f"Planning session saved to {json_out}", "Export Complete")
            
            # Proceed to generation if requested
            if generate and selected_keys and not ctx.obj.dry_run:
                operation_logger.info("🚀 Proceeding to test generation with selected plans...")
                ctx.obj.ui.display_info("Starting test generation for selected elements...", "Generation")
                
                # Get the generate use case and run with planning session
                generate_use_case = ctx.obj.container["generate_use_case"]
                
                try:
                    # Run generation with planning session context
                    generation_results = asyncio.run(generate_use_case.generate_tests(
                        project_path=actual_project_path,
                        target_files=[str(f) for f in target_files] if target_files else None,
                        from_planning_session_id=session.session_id,
                        selected_only=True,
                    ))
                    
                    # Display generation results
                    from .display_helpers import display_generation_results
                    display_generation_results(generation_results, ctx.obj.ui)
                    
                    ctx.obj.ui.display_success(
                        f"Generated tests for {len(selected_keys)} planned elements",
                        "Generation Complete"
                    )
                    
                except Exception as e:
                    operation_logger.error(f"Generation with planning session failed: {e}")
                    ctx.obj.ui.display_error(f"Generation failed: {e}", "Generation Error")
                
    except Exception as e:
        operation_logger.error(f"Planning failed: {e}")
        ctx.obj.ui.display_error(f"Planning failed: {e}", "Planning Error")
        sys.exit(1)


@app.command(name="plan-sessions")
@click.argument(
    "project_path", type=click.Path(path_type=Path), default=".", required=False
)
@click.pass_context
def plan_sessions(ctx: click.Context, project_path: Path | None) -> None:
    """List available planning sessions."""
    operation_logger = get_operation_logger("plan-sessions")
    
    try:
        # Derive actual project root
        actual_project_path = derive_project_root(project_path=project_path)
        operation_logger.info(f"Using project root: {actual_project_path}")
        
        # Get planning use case
        planning_use_case = ctx.obj.container["planning_use_case"]
        
        # Get latest session
        latest_session = planning_use_case.get_planning_session()
        
        if latest_session:
            from rich.table import Table
            
            table = Table(title="Available Planning Sessions", show_header=True, header_style="bold magenta")
            table.add_column("Session ID", style="cyan")
            table.add_column("Created", style="green")
            table.add_column("Project", style="yellow")
            table.add_column("Elements", justify="center")
            table.add_column("Selected", justify="center")
            
            # Format creation time
            from datetime import datetime
            created_time = datetime.fromtimestamp(latest_session.created_at).strftime("%Y-%m-%d %H:%M:%S")
            
            table.add_row(
                latest_session.session_id[:12] + "...",  # Truncate long ID
                created_time,
                Path(latest_session.project_path).name,
                str(len(latest_session.items)),
                str(len(latest_session.selected_keys))
            )
            
            ctx.obj.rich_cli.print_table(table)
            
            # Show usage instructions
            ctx.obj.ui.display_info(
                f"Use 'testcraft generate --from-planning-session latest --selected-only' to generate tests from the latest session",
                "Usage"
            )
            
            # Show where sessions are stored
            state_file = actual_project_path / ".testcraft_state.json"
            ctx.obj.ui.display_info(
                f"Planning sessions stored in: {state_file}",
                "Storage Location"
            )
        else:
            ctx.obj.ui.display_info("No planning sessions found. Run 'testcraft plan' to create one.", "No Sessions")
            
    except Exception as e:
        operation_logger.error(f"Failed to list planning sessions: {e}")
        ctx.obj.ui.display_error(f"Failed to list sessions: {e}", "Sessions Error")
        sys.exit(1)


# Add generate command
app.add_command(generate)

# Add evaluation commands
add_evaluation_commands(app)

# Add utility commands
add_utility_commands(app)


if __name__ == "__main__":
    app()
