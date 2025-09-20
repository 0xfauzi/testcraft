"""
Enhanced TestCraft UI Demo - Comprehensive Showcase

This demo showcases the massively improved Rich UI system with:
- Advanced progress tracking and real-time dashboards
- Beautiful file processing tables with rich formatting
- Enhanced logging with structured messages and context
- Sophisticated error handling with suggestions
- Performance metrics displays
- Interactive progress indicators
- Beautiful panels and layouts

Run this demo to see the dramatic improvements in TestCraft's UI!
"""

import asyncio
import random
import sys
import time
from pathlib import Path
from typing import Any

# Add the parent directory to Python path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from testcraft.adapters.io.enhanced_logging import (
    get_operation_logger,
    setup_enhanced_logging,
)
from testcraft.adapters.io.enhanced_ui import EnhancedUIAdapter


def simulate_file_processing(files: list[str]) -> list[dict[str, Any]]:
    """Simulate processing files with realistic data."""
    results = []

    for _i, file_path in enumerate(files):
        # Simulate realistic processing times and outcomes
        processing_time = random.uniform(0.5, 3.0)
        success_rate = 0.85  # 85% success rate
        success = random.random() < success_rate

        time.sleep(processing_time * 0.1)  # Simulate some work (shortened for demo)

        file_data = {
            "file_path": file_path,
            "status": "completed" if success else random.choice(["failed", "partial"]),
            "progress": 1.0 if success else random.uniform(0.3, 0.8),
            "tests_generated": random.randint(3, 12) if success else 0,
            "coverage": random.uniform(0.7, 0.95)
            if success
            else random.uniform(0.2, 0.6),
            "duration": processing_time,
        }
        results.append(file_data)

    return results


def demo_enhanced_progress_tracking():
    """Demo advanced progress tracking with multiple stages."""
    ui = EnhancedUIAdapter()
    get_operation_logger("demo")

    ui.console.print("\n🚀 [title]Enhanced Progress Tracking Demo[/title]")
    ui.console.rule("[cyan]Multi-Stage Operations[/]")

    # Demo 1: Multi-stage operation with detailed progress
    with ui.create_operation_tracker(
        "Complex Test Generation", total_steps=5
    ) as tracker:
        tracker.advance_step("🔍 Discovering Python files", 1)
        time.sleep(1)

        tracker.advance_step("📊 Analyzing code coverage", 1)
        time.sleep(0.8)

        tracker.advance_step("🧠 Generating tests with AI", 1)
        time.sleep(1.5)

        tracker.advance_step("✅ Running and refining tests", 1)
        time.sleep(1.2)

        tracker.advance_step("📝 Writing results to disk", 1)
        time.sleep(0.5)

        # Log some progress during the operation
        tracker.log_progress("Generated 45 test cases across 12 files", "info")
        tracker.log_progress("Average test quality score: 8.7/10", "info")


def demo_real_time_dashboard():
    """Demo real-time updating dashboard."""
    ui = EnhancedUIAdapter()

    ui.console.print("\n📊 [title]Real-Time Dashboard Demo[/title]")
    ui.console.rule("[cyan]Live Status Updates[/]")

    # Create sample files for processing
    sample_files = [
        "src/core/engine.py",
        "src/adapters/llm_adapter.py",
        "src/utils/file_handler.py",
        "src/models/test_case.py",
        "src/cli/commands.py",
        "src/parsers/python_parser.py",
    ]

    with ui.create_real_time_dashboard("TestCraft Processing Dashboard") as dashboard:
        dashboard.update_footer("🚀 Processing started...")

        files_data = []
        total_files = len(sample_files)

        for i, file_path in enumerate(sample_files):
            # Update current status
            file_name = Path(file_path).name
            status_text = f"Processing {file_name} ({i + 1}/{total_files})"
            dashboard.update_footer(status_text)

            # Simulate processing
            processing_time = random.uniform(0.3, 1.0)
            time.sleep(processing_time)

            # Add result
            success = random.random() > 0.2  # 80% success rate
            file_data = {
                "file_path": file_path,
                "status": "completed" if success else "failed",
                "progress": 1.0 if success else random.uniform(0.4, 0.8),
                "tests_generated": random.randint(2, 8) if success else 0,
                "coverage": random.uniform(0.6, 0.9) if success else 0,
                "duration": processing_time,
            }
            files_data.append(file_data)

            # Update main display with current results
            if files_data:
                from rich.panel import Panel
                from rich.table import Table

                # Create mini summary table
                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("File", width=20)
                table.add_column("Status", width=10, justify="center")
                table.add_column("Tests", width=8, justify="center")

                for fd in files_data[-3:]:  # Show last 3 files
                    file_name = Path(fd["file_path"]).name
                    status = "✅" if fd["status"] == "completed" else "❌"
                    tests = str(fd["tests_generated"])
                    table.add_row(file_name, status, tests)

                dashboard.update_main_content(Panel(table, title="Recent Results"))

                # Update sidebar with stats
                completed = sum(1 for f in files_data if f["status"] == "completed")
                total_tests = sum(f["tests_generated"] for f in files_data)

                stats_content = f"""
[green]✅ Completed:[/] {completed}
[yellow]🔄 Processing:[/] {total_files - len(files_data)}
[cyan]🧪 Tests Generated:[/] {total_tests}
[blue]📈 Success Rate:[/] {completed / len(files_data) * 100:.1f}%
"""
                dashboard.update_sidebar(Panel(stats_content, title="Live Stats"))

        dashboard.update_footer("✅ All files processed!")
        time.sleep(2)  # Let user see final results


def demo_enhanced_file_table():
    """Demo the enhanced file processing table."""
    ui = EnhancedUIAdapter()

    ui.console.print("\n📁 [title]Enhanced File Processing Table Demo[/title]")
    ui.console.rule("[cyan]Rich File Status Display[/]")

    # Generate realistic test data
    sample_files = [
        "src/main.py",
        "src/config.py",
        "src/utils.py",
        "src/models.py",
        "src/database.py",
        "src/api.py",
        "src/auth.py",
        "src/cache.py",
        "src/logging.py",
        "src/monitoring.py",
    ]

    files_data = simulate_file_processing(sample_files)

    # Display with enhanced table
    ui.display_file_progress_table(files_data, "Test Generation Results")

    # Show performance metrics
    total_duration = sum(f["duration"] for f in files_data)
    success_count = sum(1 for f in files_data if f["status"] == "completed")
    total_tests = sum(f["tests_generated"] for f in files_data)

    metrics = {
        "file_processing": {
            "duration": total_duration,
            "items_processed": len(files_data),
            "success_rate": success_count / len(files_data),
            "tests_generated": total_tests,
        }
    }

    ui.display_metrics_panel(metrics, "Processing Performance")


def demo_enhanced_logging():
    """Demo the enhanced structured logging system."""
    ui = EnhancedUIAdapter()
    logger = get_operation_logger("demo_logging")

    ui.console.print("\n📝 [title]Enhanced Logging Demo[/title]")
    ui.console.rule("[cyan]Structured Messages & Rich Formatting[/]")

    # Demo various logging scenarios
    with logger.operation_context("file_processing", batch_size=5, mode="immediate"):
        logger.file_operation_start("src/example.py", "test_generation")
        time.sleep(0.5)

        logger.batch_progress(
            "test_generation", completed=3, total=10, current_item="src/models.py"
        )
        time.sleep(0.3)

        logger.file_operation_complete(
            "src/example.py",
            "test_generation",
            duration=2.5,
            success=True,
            tests_generated=8,
            coverage=0.85,
            lines_processed=156,
        )

        # Demo error logging with context
        try:
            # Simulate an error
            raise ValueError("Invalid configuration: missing API key")
        except ValueError as e:
            suggestions = [
                "Check your .env file for API_KEY setting",
                "Verify the API key format (should be 32+ characters)",
                "Try regenerating your API key from the provider dashboard",
            ]
            logger.error_with_context(
                "Configuration validation failed",
                e,
                suggestions=suggestions,
                config_file="testcraft.toml",
                line_number=42,
            )

        # Demo performance summary
        performance_metrics = {
            "duration": 15.7,
            "items_processed": 10,
            "success_rate": 0.9,
            "memory_usage": 128 * 1024 * 1024,  # 128MB
            "cache_hits": 85,
            "api_calls": 23,
        }

        logger.performance_summary("test_generation", performance_metrics)


def demo_error_handling_with_suggestions():
    """Demo enhanced error handling with helpful suggestions."""
    ui = EnhancedUIAdapter()

    ui.console.print("\n❌ [title]Enhanced Error Handling Demo[/title]")
    ui.console.rule("[cyan]Helpful Error Messages & Suggestions[/]")

    # Demo different types of errors with suggestions

    # Configuration error
    config_suggestions = [
        "Verify your TOML file syntax using an online validator",
        "Check for missing required fields: llm_provider, api_key",
        "Ensure file permissions allow reading the config file",
        "Try creating a new config with 'testcraft init-config'",
    ]
    ui.display_error_with_suggestions(
        "Failed to parse configuration file: Invalid TOML syntax at line 23",
        config_suggestions,
        "Configuration Error",
    )

    time.sleep(2)

    # API error
    api_suggestions = [
        "Check your internet connection",
        "Verify your API key is valid and not expired",
        "Try reducing batch size to avoid rate limits",
        "Check the service status at status.openai.com",
    ]
    ui.display_error_with_suggestions(
        "LLM API request failed: 429 Rate limit exceeded", api_suggestions, "API Error"
    )

    time.sleep(2)

    # File system error
    fs_suggestions = [
        "Ensure the target directory exists and is writable",
        "Check available disk space (need at least 100MB)",
        "Verify file permissions for the project directory",
        "Try running with elevated privileges if necessary",
    ]
    ui.display_error_with_suggestions(
        "Failed to write test file: Permission denied",
        fs_suggestions,
        "File System Error",
    )


def demo_success_summary():
    """Demo the comprehensive success summary display."""
    ui = EnhancedUIAdapter()

    ui.console.print("\n🎉 [title]Success Summary Demo[/title]")
    ui.console.rule("[cyan]Comprehensive Results Display[/]")

    # Create realistic success data
    files_data = simulate_file_processing(
        ["src/main.py", "src/auth.py", "src/database.py", "src/api.py", "src/utils.py"]
    )

    # Calculate summary metrics
    total_duration = sum(f["duration"] for f in files_data)
    success_count = sum(1 for f in files_data if f["status"] == "completed")
    total_tests = sum(f["tests_generated"] for f in files_data)

    summary_data = {
        "message": f"Successfully generated tests for {success_count} out of {len(files_data)} files! 🎊",
        "metrics": {
            "test_generation": {
                "duration": total_duration,
                "items_processed": len(files_data),
                "success_rate": success_count / len(files_data),
                "memory_usage": 85 * 1024 * 1024,  # 85MB
            },
            "quality_analysis": {
                "duration": 3.2,
                "items_processed": total_tests,
                "avg_coverage": sum(
                    f["coverage"] for f in files_data if f["coverage"] > 0
                )
                / success_count,
                "success_rate": 0.95,
            },
        },
        "files_processed": files_data,
    }

    ui.display_success_summary(summary_data)


async def main():
    """Run the complete enhanced UI demonstration."""
    ui = EnhancedUIAdapter()

    # Setup enhanced logging
    setup_enhanced_logging(ui.console)

    # Welcome banner
    ui.console.clear()
    welcome_text = """
    🚀 [title]TestCraft Enhanced UI System Demo[/title] 🚀

    [info]Welcome to the dramatically improved TestCraft experience![/]
    [info]This demo showcases the comprehensive Rich UI overhaul with:[/]

    ✨ [highlight]Advanced progress tracking with multi-stage operations[/]
    📊 [highlight]Real-time dashboards with live status updates[/]
    📁 [highlight]Beautiful file processing tables with rich formatting[/]
    📝 [highlight]Enhanced structured logging with context and suggestions[/]
    ❌ [highlight]Intelligent error handling with helpful recommendations[/]
    🎉 [highlight]Comprehensive success summaries with performance metrics[/]

    [primary]Prepare for a visual feast! 🍰[/]
    """

    ui.console.print(welcome_text)
    ui.console.print("\n[dim]Press Enter to start the demo...[/]", end="")
    input()

    try:
        # Run all demonstrations
        demo_enhanced_progress_tracking()

        ui.console.print("\n[dim]Press Enter to continue...[/]", end="")
        input()

        demo_real_time_dashboard()

        ui.console.print("\n[dim]Press Enter to continue...[/]", end="")
        input()

        demo_enhanced_file_table()

        ui.console.print("\n[dim]Press Enter to continue...[/]", end="")
        input()

        demo_enhanced_logging()

        ui.console.print("\n[dim]Press Enter to continue...[/]", end="")
        input()

        demo_error_handling_with_suggestions()

        ui.console.print("\n[dim]Press Enter to continue...[/]", end="")
        input()

        demo_success_summary()

        # Final celebration
        ui.console.rule("🎊 [title]Demo Complete![/title] 🎊", style="green")
        finale_text = """
[success]🎉 Enhanced UI Demo completed successfully![/]

[info]The TestCraft UI system now features:[/]
  🎨 [highlight]Sophisticated progress tracking[/] with multi-stage operations
  📊 [highlight]Real-time dashboards[/] with live status updates
  📁 [highlight]Rich file processing tables[/] with beautiful formatting
  📝 [highlight]Enhanced structured logging[/] with context and rich formatting
  ❌ [highlight]Intelligent error handling[/] with helpful suggestions
  🎉 [highlight]Comprehensive success displays[/] with performance metrics
  ⚡ [highlight]Animated progress indicators[/] and status updates
  🖼️  [highlight]Beautiful panels and layouts[/] for organized information display

[primary]Your CLI will never look boring again! 🚀[/]

[dim]This represents a massive improvement over the previous basic UI.[/]
[dim]Users now get rich, informative, and beautiful feedback for all operations.[/]
"""

        ui.console.print(finale_text)

    except KeyboardInterrupt:
        ui.console.print("\n[warning]⚠️ Demo interrupted by user[/]")
    except Exception as e:
        ui.display_error(f"Demo failed: {str(e)}", "Demo Error")
    finally:
        ui.finalize()


if __name__ == "__main__":
    asyncio.run(main())
