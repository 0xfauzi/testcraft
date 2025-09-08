"""Main CLI entry point for TestCraft."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from rich.console import Console

from ..config.loader import ConfigLoader, ConfigurationError
from ..config.models import TestCraftConfig
from ..adapters.io.rich_cli import RichCliComponents
from ..adapters.io.ui_rich import RichUIAdapter
from .dependency_injection import create_dependency_container, DependencyError


# Initialize Rich console and UI components
console = Console()
ui = RichCliComponents(console)


class ClickContext:
    """Context object for Click commands."""
    
    def __init__(self):
        self.config: Optional[TestCraftConfig] = None
        self.container: Optional[Dict[str, Any]] = None
        self.ui: RichUIAdapter = RichUIAdapter(console)
        self.verbose: bool = False
        self.dry_run: bool = False


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path),
              help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--dry-run', '--dry', is_flag=True, help='Preview operations without executing them')
@click.pass_context
def app(ctx: click.Context, config: Optional[Path], verbose: bool, dry_run: bool) -> None:
    """TestCraft - AI-powered test generation tool for Python projects."""
    # Initialize context
    ctx.ensure_object(ClickContext)
    ctx.obj.verbose = verbose
    ctx.obj.dry_run = dry_run
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration
        loader = ConfigLoader(config)
        ctx.obj.config = loader.load_config()
        
        # Create dependency container
        ctx.obj.container = create_dependency_container(ctx.obj.config)
        
    except ConfigurationError as e:
        ui.display_error(f"Configuration error: {e}", "Configuration Failed")
        sys.exit(1)
    except DependencyError as e:
        ui.display_error(f"Dependency injection error: {e}", "Initialization Failed") 
        sys.exit(1)
    except Exception as e:
        ui.display_error(f"Unexpected error during initialization: {e}", "Initialization Failed")
        if verbose:
            import traceback
            ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


# ============================================================================
# MAIN COMMANDS
# ============================================================================

@app.command()
@click.argument('project_path', type=click.Path(exists=True, path_type=Path), default='.')
@click.option('--target-files', '-f', multiple=True, type=click.Path(path_type=Path),
              help='Specific files to generate tests for')
@click.option('--batch-size', '-b', type=int, default=5,
              help='Number of files to process in parallel')
@click.option('--streaming', is_flag=True, help='Enable streaming LLM responses')
@click.option('--force', is_flag=True, help='Overwrite existing test files')
@click.pass_context
def generate(
    ctx: click.Context,
    project_path: Path,
    target_files: Tuple[Path, ...],
    batch_size: int,
    streaming: bool,
    force: bool
) -> None:
    """Generate tests for Python source files."""
    try:
        if ctx.obj.dry_run:
            ui.display_info("DRY RUN: No tests will actually be generated", "Dry Run Mode")
        
        # Get use case from container
        generate_usecase = ctx.obj.container['generate_usecase']
        
        # Configure generation parameters
        config_overrides = {
            'batch_size': batch_size,
            'enable_streaming': streaming,
            'force_overwrite': force
        }
        
        with ui.create_status_spinner("Initializing test generation..."):
            # Run generation asynchronously
            results = asyncio.run(generate_usecase.generate_tests(
                project_path=project_path,
                target_files=list(target_files) if target_files else None,
                **config_overrides
            ))
        
        # Display results using Rich components
        if results.get('success'):
            _display_generation_results(results)
        else:
            ui.display_error(
                results.get('error_message', 'Unknown error occurred'),
                "Generation Failed"
            )
            sys.exit(1)
    
    except Exception as e:
        ui.display_error(f"Test generation failed: {e}", "Generation Error")
        if ctx.obj.verbose:
            import traceback
            ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


@app.command()
@click.argument('project_path', type=click.Path(exists=True, path_type=Path), default='.')
@click.option('--target-files', '-f', multiple=True, type=click.Path(path_type=Path),
              help='Specific files to analyze')
@click.pass_context
def analyze(
    ctx: click.Context,
    project_path: Path,
    target_files: Tuple[Path, ...]
) -> None:
    """Analyze what tests would be generated and why."""
    try:
        # Get use case from container
        analyze_usecase = ctx.obj.container['analyze_usecase']
        
        with ui.create_status_spinner("Analyzing project files..."):
            # Run analysis asynchronously
            results = asyncio.run(analyze_usecase.analyze_generation_needs(
                project_path=project_path,
                target_files=list(target_files) if target_files else None
            ))
        
        # Display results using Rich components
        _display_analysis_results(results)
    
    except Exception as e:
        ui.display_error(f"Analysis failed: {e}", "Analysis Error")
        if ctx.obj.verbose:
            import traceback
            ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


@app.command()
@click.argument('project_path', type=click.Path(exists=True, path_type=Path), default='.')
@click.option('--source-files', '-s', multiple=True, type=click.Path(path_type=Path),
              help='Specific source files to measure')
@click.option('--test-files', '-t', multiple=True, type=click.Path(path_type=Path),
              help='Specific test files to include')
@click.option('--format', '-o', 'output_format', 
              type=click.Choice(['detailed', 'summary', 'json'], case_sensitive=False),
              multiple=True, default=['detailed'], help='Output format')
@click.pass_context
def coverage(
    ctx: click.Context,
    project_path: Path,
    source_files: Tuple[Path, ...],
    test_files: Tuple[Path, ...],
    output_format: Tuple[str, ...]
) -> None:
    """Measure and report code coverage."""
    try:
        # Get use case from container  
        coverage_usecase = ctx.obj.container['coverage_usecase']
        
        # Configure parameters
        config_overrides = {
            'output_formats': list(output_format)
        }
        
        with ui.create_status_spinner("Measuring code coverage..."):
            # Run coverage measurement asynchronously
            results = asyncio.run(coverage_usecase.measure_and_report(
                project_path=project_path,
                source_files=list(source_files) if source_files else None,
                test_files=list(test_files) if test_files else None,
                **config_overrides
            ))
        
        # Display results using Rich components
        if results.get('success'):
            _display_coverage_results(results)
        else:
            ui.display_error(
                results.get('error_message', 'Unknown error occurred'),
                "Coverage Failed"
            )
            sys.exit(1)
    
    except Exception as e:
        ui.display_error(f"Coverage measurement failed: {e}", "Coverage Error")
        if ctx.obj.verbose:
            import traceback
            ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


@app.command()
@click.argument('project_path', type=click.Path(exists=True, path_type=Path), 
                required=False, default=None)
@click.option('--history', is_flag=True, help='Include generation history')
@click.option('--statistics', is_flag=True, help='Include summary statistics')
@click.option('--limit', '-l', type=int, default=10, help='Limit history entries')
@click.pass_context
def status(
    ctx: click.Context,
    project_path: Optional[Path],
    history: bool,
    statistics: bool,
    limit: int
) -> None:
    """Show generation status, history, and statistics."""
    try:
        # Get use case from container
        status_usecase = ctx.obj.container['status_usecase']
        
        with ui.create_status_spinner("Retrieving status information..."):
            # Run status retrieval asynchronously
            results = asyncio.run(status_usecase.get_generation_status(
                project_path=project_path,
                include_history=history,
                include_statistics=statistics
            ))
        
        # Display results using Rich components
        _display_status_results(results, limit)
    
    except Exception as e:
        ui.display_error(f"Status retrieval failed: {e}", "Status Error")
        if ctx.obj.verbose:
            import traceback
            ui.display_info(traceback.format_exc(), "Debug Information")
        sys.exit(1)


# ============================================================================
# RESULT DISPLAY FUNCTIONS
# ============================================================================

def _display_generation_results(results: Dict[str, Any]) -> None:
    """Display test generation results using Rich components."""
    summary_data = {
        'total_files': results.get('files_discovered', 0),
        'files_with_tests': results.get('files_written', 0), 
        'overall_coverage': results.get('final_coverage', {}).get('overall_line_coverage', 0),
        'tests_generated': results.get('tests_generated', 0),
        'generation_success_rate': (
            results.get('files_written', 0) / max(results.get('files_processed', 1), 1)
        )
    }
    
    panel = ui.create_project_summary_panel(summary_data)
    ui.print_panel(panel)
    
    # Show coverage improvement
    coverage_delta = results.get('coverage_delta', {})
    if coverage_delta.get('line_coverage_delta', 0) > 0:
        ui.display_success(
            f"Coverage improved by {coverage_delta['line_coverage_delta']:.1%}",
            "Coverage Improvement"
        )


def _display_analysis_results(results) -> None:
    """Display analysis results using Rich components."""
    if hasattr(results, 'files_to_process'):
        # Results is an AnalysisReport object
        files_to_process = results.files_to_process
        reasons = results.reasons
        test_presence = results.existing_test_presence
    else:
        # Results is a dictionary
        files_to_process = results.get('files_to_process', [])
        reasons = results.get('reasons', {})
        test_presence = results.get('existing_test_presence', {})
    
    analysis_data = {
        'files_to_process': files_to_process,
        'reasons': reasons,
        'existing_test_presence': test_presence
    }
    
    tree = ui.create_analysis_tree(analysis_data)
    ui.print_tree(tree)


def _display_coverage_results(results: Dict[str, Any]) -> None:
    """Display coverage results using Rich components."""
    coverage_data = results.get('coverage_data', {})
    if coverage_data:
        table = ui.create_coverage_table(coverage_data)
        ui.print_table(table)
    
    # Show summary information
    summary = results.get('coverage_summary', {})
    if summary:
        ui.display_info(
            f"Overall coverage: {summary.get('overall_line_coverage', 0):.1%} line, "
            f"{summary.get('overall_branch_coverage', 0):.1%} branch",
            "Coverage Summary"
        )


def _display_status_results(results: Dict[str, Any], limit: int) -> None:
    """Display status results using Rich components."""
    current_state = results.get('current_state', {})
    history = results.get('generation_history', [])
    statistics = results.get('summary_statistics', {})
    
    # Display current state
    if current_state:
        state_info = []
        if current_state.get('last_generation_timestamp'):
            from datetime import datetime
            timestamp = datetime.fromtimestamp(current_state['last_generation_timestamp'])
            state_info.append(f"Last generation: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if current_state.get('generation_summary'):
            gen_summary = current_state['generation_summary']
            state_info.append(
                f"Files processed: {gen_summary.get('total_files_processed', 0)}"
            )
        
        if state_info:
            ui.display_info('\n'.join(state_info), "Current State")
    
    # Display history if available
    if history:
        console.print("\n[bold]Recent Activity:[/]")
        for entry in history[:limit]:
            timestamp = entry.get('timestamp', 0)
            if timestamp:
                from datetime import datetime
                dt = datetime.fromtimestamp(timestamp)
                console.print(f"  • {dt.strftime('%Y-%m-%d %H:%M')} - "
                            f"{entry.get('entry_type', 'unknown')}: "
                            f"{entry.get('status', 'unknown')}")
    
    # Display statistics if available
    if statistics:
        stats_info = []
        if statistics.get('total_runs'):
            stats_info.append(f"Total runs: {statistics['total_runs']}")
        if statistics.get('success_rate'):
            stats_info.append(f"Success rate: {statistics['success_rate']:.1%}")
        if statistics.get('average_coverage_percentage'):
            stats_info.append(f"Average coverage: {statistics['average_coverage_percentage']:.1%}")
        
        if stats_info:
            ui.display_info('\n'.join(stats_info), "Summary Statistics")


# Add evaluation commands
from .evaluation_commands import add_evaluation_commands
add_evaluation_commands(app)

# Add utility commands
from .utility_commands import add_utility_commands
add_utility_commands(app)


if __name__ == "__main__":
    app()