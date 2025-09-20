"""
Live File Tracking Demo for TestCraft

This demo showcases the new live file status tracking system that provides
real-time granular updates during test generation and refinement operations.

Features demonstrated:
- Real-time file processing status with detailed operations
- Live progress tracking through generation, writing, testing, and refinement phases
- Beautiful file processing table with status indicators and metrics
- Performance statistics and completion rates
- Detailed error tracking and reporting
"""

import asyncio
import random
import sys
from pathlib import Path

# Add the parent directory to Python path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from testcraft.adapters.io.enhanced_ui import EnhancedUIAdapter
from testcraft.adapters.io.file_status_tracker import (
    FileStatus,
    FileStatusTracker,
    LiveFileTracking,
)


async def simulate_test_generation_pipeline(
    tracker: FileStatusTracker, file_paths: list[str]
):
    """Simulate the complete test generation pipeline with live updates."""

    for i, file_path in enumerate(file_paths):
        file_name = Path(file_path).name

        try:
            # Phase 1: Analysis
            tracker.update_file_status(
                file_path,
                FileStatus.ANALYZING,
                operation="Code Analysis",
                step="Parsing source code and extracting functions",
                progress=10.0,
            )
            await asyncio.sleep(random.uniform(0.3, 0.8))

            # Phase 2: Generation
            tracker.update_file_status(
                file_path,
                FileStatus.GENERATING,
                operation="LLM Generation",
                step="Generating test cases with AI",
                progress=30.0,
            )
            await asyncio.sleep(random.uniform(1.0, 2.5))

            # Update generation progress
            tracker.update_file_status(
                file_path,
                FileStatus.GENERATING,
                operation="LLM Generation",
                step="Processing response and formatting tests",
                progress=60.0,
            )
            await asyncio.sleep(random.uniform(0.2, 0.6))

            # Phase 3: Writing
            tracker.update_file_status(
                file_path,
                FileStatus.WRITING,
                operation="File Writing",
                step="Saving generated tests to disk",
                progress=75.0,
            )
            await asyncio.sleep(random.uniform(0.1, 0.3))

            # Simulate success/failure
            success_rate = 0.8  # 80% success rate
            generation_success = random.random() < success_rate

            if not generation_success:
                # Generation failed
                tracker.update_file_status(
                    file_path,
                    FileStatus.FAILED,
                    operation="Generation Failed",
                    step="LLM returned invalid test code",
                    progress=0.0,
                )
                tracker.update_generation_result(
                    file_path, False, 0, "Invalid test syntax"
                )
                continue

            # Update generation metrics
            tests_generated = random.randint(3, 12)
            tracker.update_generation_result(
                file_path,
                True,
                tests_generated,
                test_file_path=f"tests/test_{file_name}",
            )

            # Phase 4: Initial Testing
            test_file_path = f"tests/test_{file_name}"
            tracker.update_file_status(
                file_path,
                FileStatus.TESTING,
                operation="Initial Testing",
                step="Running pytest on generated tests",
                progress=80.0,
            )
            await asyncio.sleep(random.uniform(0.5, 1.2))

            # Simulate test results
            initial_test_success = random.random() < 0.6  # 60% pass initially

            if initial_test_success:
                # Tests pass immediately
                tracker.update_file_status(
                    file_path,
                    FileStatus.COMPLETED,
                    operation="Tests Passing",
                    step="All tests pass on first run",
                    progress=100.0,
                )
                tracker.update_refinement_result(test_file_path, 1, True)
            else:
                # Need refinement
                max_refinement_iterations = random.randint(1, 4)

                for iteration in range(max_refinement_iterations):
                    tracker.update_file_status(
                        file_path,
                        FileStatus.REFINING,
                        operation=f"Refinement {iteration + 1}",
                        step=f"Fixing test failures (iteration {iteration + 1})",
                        progress=80.0 + (iteration * 5),
                    )
                    await asyncio.sleep(random.uniform(0.8, 1.5))

                    # Simulate refinement progress
                    tracker.update_file_status(
                        file_path,
                        FileStatus.REFINING,
                        operation=f"Refinement {iteration + 1}",
                        step="Using LLM to fix test failures",
                        progress=85.0 + (iteration * 5),
                    )
                    await asyncio.sleep(random.uniform(0.4, 0.8))

                    # Check if refinement succeeds
                    refinement_success_chance = 0.7 + (
                        iteration * 0.1
                    )  # Increasing chance
                    if random.random() < refinement_success_chance:
                        # Refinement successful
                        tracker.update_file_status(
                            file_path,
                            FileStatus.COMPLETED,
                            operation="Tests Passing",
                            step=f"All tests pass after {iteration + 1} refinement(s)",
                            progress=100.0,
                        )
                        tracker.update_refinement_result(
                            test_file_path, iteration + 1, True
                        )
                        break
                else:
                    # Max iterations reached, failed
                    error_msgs = [
                        "Syntax errors in generated tests",
                        "Import resolution failures",
                        "Mock configuration issues",
                        "Assertion logic errors",
                    ]
                    tracker.update_file_status(
                        file_path,
                        FileStatus.FAILED,
                        operation="Refinement Failed",
                        step=f"Max iterations ({max_refinement_iterations}) reached",
                        progress=0.0,
                    )
                    tracker.update_refinement_result(
                        test_file_path,
                        max_refinement_iterations,
                        False,
                        [random.choice(error_msgs)],
                    )

        except Exception as e:
            # Handle unexpected errors
            tracker.update_file_status(
                file_path,
                FileStatus.FAILED,
                operation="System Error",
                step=f"Unexpected error: {str(e)}",
                progress=0.0,
            )


async def demo_live_file_tracking():
    """Demonstrate live file tracking during generation and refinement."""
    ui = EnhancedUIAdapter()

    # Welcome message
    ui.console.print("\n🔄 [title]Live File Tracking Demo[/title]")
    ui.console.print(
        "Demonstrating real-time granular file status updates during generation and refinement\n"
    )

    # Sample files to process
    sample_files = [
        "src/core/authentication.py",
        "src/models/user_profile.py",
        "src/services/payment_processor.py",
        "src/utils/data_validator.py",
        "src/api/endpoints.py",
        "src/database/queries.py",
        "src/cache/redis_manager.py",
        "src/notifications/email_sender.py",
    ]

    ui.console.print(
        f"[cyan]Processing {len(sample_files)} files with live status tracking...[/]\n"
    )

    # Initialize and start live tracking
    with LiveFileTracking(ui, "TestCraft Live Generation & Refinement") as live_tracker:
        file_tracker = live_tracker.initialize_and_start(sample_files)

        # Run the simulated pipeline
        await simulate_test_generation_pipeline(file_tracker, sample_files)

        # Let user see final results
        await asyncio.sleep(3)

        # Show final summary
        stats = file_tracker.get_summary_stats()

    # Display final summary outside of live tracking
    ui.console.print("\n")
    ui.console.rule("📊 [title]Final Summary[/title]", style="green")

    final_summary = f"""
[success]🎉 Live file tracking demonstration completed![/]

[info]📈 Processing Statistics:[/]
  • [green]Total Files:[/] {stats["total_files"]}
  • [green]Completed:[/] {stats["completed"]} ({stats["success_rate"]:.1%} success rate)
  • [red]Failed:[/] {stats["failed"]}
  • [yellow]Total Duration:[/] {stats["total_duration"]:.1f}s
  • [blue]Processing Rate:[/] {stats["files_per_minute"]:.1f} files/min
  
[info]🧪 Test Generation:[/]
  • [cyan]Tests Generated:[/] {stats["total_tests_generated"]}
  • [magenta]Pytest Runs:[/] {stats["total_pytest_runs"]}
  • [yellow]Avg Processing Time:[/] {stats["avg_duration"]:.1f}s per file

[highlight]✨ Key Features Demonstrated:[/]
  • [bold]Real-time status updates[/] for each file through all phases
  • [bold]Detailed operation tracking[/] (Analysis → Generation → Writing → Testing → Refinement)
  • [bold]Live progress indicators[/] with percentage completion
  • [bold]Granular step descriptions[/] showing exactly what's happening
  • [bold]Performance metrics[/] with timing and success rates
  • [bold]Beautiful live table display[/] with status icons and colors
  • [bold]Statistics sidebar[/] with real-time updates
  • [bold]Overall progress tracking[/] with time estimates

[primary]This system provides users with complete visibility into the generation process! 🚀[/]
"""

    ui.console.print(final_summary)


async def demo_error_scenarios():
    """Demonstrate error handling and edge cases in live tracking."""
    ui = EnhancedUIAdapter()

    ui.console.print("\n⚠️ [title]Error Scenarios Demo[/title]")
    ui.console.print("Demonstrating error handling and edge cases\n")

    error_files = [
        "src/problematic_file.py",
        "src/complex_module.py",
        "src/legacy_code.py",
    ]

    with LiveFileTracking(ui, "Error Handling Demonstration") as live_tracker:
        file_tracker = live_tracker.initialize_and_start(error_files)

        for file_path in error_files:
            # Simulate different types of errors
            error_type = random.choice(
                ["syntax_error", "api_timeout", "permission_denied", "out_of_memory"]
            )

            # Start processing
            file_tracker.update_file_status(
                file_path,
                FileStatus.ANALYZING,
                operation="Code Analysis",
                step="Extracting functions and classes",
                progress=10.0,
            )
            await asyncio.sleep(0.5)

            if error_type == "syntax_error":
                file_tracker.update_file_status(
                    file_path,
                    FileStatus.FAILED,
                    operation="Parsing Failed",
                    step="File contains syntax errors that prevent analysis",
                    progress=0.0,
                )
            elif error_type == "api_timeout":
                file_tracker.update_file_status(
                    file_path,
                    FileStatus.GENERATING,
                    operation="LLM Generation",
                    step="Waiting for language model response...",
                    progress=40.0,
                )
                await asyncio.sleep(1.0)
                file_tracker.update_file_status(
                    file_path,
                    FileStatus.FAILED,
                    operation="API Timeout",
                    step="Language model request timed out after 30s",
                    progress=0.0,
                )
            elif error_type == "permission_denied":
                file_tracker.update_file_status(
                    file_path,
                    FileStatus.WRITING,
                    operation="Writing Tests",
                    step="Saving generated test file...",
                    progress=80.0,
                )
                await asyncio.sleep(0.3)
                file_tracker.update_file_status(
                    file_path,
                    FileStatus.FAILED,
                    operation="Write Failed",
                    step="Permission denied writing to test directory",
                    progress=0.0,
                )
            else:  # out_of_memory
                file_tracker.update_file_status(
                    file_path,
                    FileStatus.REFINING,
                    operation="Refinement 2",
                    step="Processing large test file...",
                    progress=90.0,
                )
                await asyncio.sleep(0.8)
                file_tracker.update_file_status(
                    file_path,
                    FileStatus.FAILED,
                    operation="Memory Error",
                    step="Insufficient memory to process large file",
                    progress=0.0,
                )

            # Add error details
            file_tracker.update_generation_result(
                file_path,
                success=False,
                error=f"Processing failed: {error_type.replace('_', ' ')}",
            )

        await asyncio.sleep(2)


async def main():
    """Run the complete live file tracking demonstration."""
    ui = EnhancedUIAdapter()

    # Setup enhanced logging for better demo experience
    from testcraft.adapters.io.enhanced_logging import setup_enhanced_logging

    setup_enhanced_logging(ui.console)

    # Welcome banner
    ui.console.clear()
    welcome_text = """
    🔄 [title]TestCraft Live File Tracking Demo[/title] 🔄
    
    [info]Welcome to the revolutionary live file status tracking system![/]
    [info]This demo shows granular real-time updates during generation and refinement.[/]
    
    🌟 [highlight]Features Showcased:[/]
    
    ⚡ [primary]Real-time Status Updates[/] - See exactly what's happening to each file
    📊 [primary]Live Progress Tables[/] - Beautiful visual feedback with status indicators  
    🔍 [primary]Granular Operation Details[/] - Detailed step-by-step progress
    📈 [primary]Performance Metrics[/] - Live statistics and completion rates
    🎯 [primary]Phase Tracking[/] - Analysis → Generation → Writing → Testing → Refinement
    ⚠️ [primary]Error Handling[/] - Detailed error tracking and reporting
    
    [secondary]No more wondering what's happening during long operations![/]
    [secondary]Users get complete visibility into every step of the process.[/]
    
    [primary]🚀 Prepare for the future of CLI feedback! 🚀[/]
    """

    ui.console.print(welcome_text)
    ui.console.print("\n[dim]Press Enter to start the live tracking demo...[/]", end="")
    input()

    try:
        # Demo 1: Normal live file tracking
        await demo_live_file_tracking()

        ui.console.print(
            "\n[dim]Press Enter to see error handling scenarios...[/]", end=""
        )
        input()

        # Demo 2: Error scenarios
        await demo_error_scenarios()

        # Final message
        ui.console.rule("🎊 [title]Demo Complete![/title] 🎊", style="success")
        finale_text = """
[success]🎉 Live File Tracking Demo completed successfully![/]

[info]The enhanced tracking system now provides:[/]
  🔄 [highlight]Real-time granular updates[/] for every file operation
  📊 [highlight]Live visual feedback[/] with beautiful status tables
  ⚡ [highlight]Phase-by-phase progress[/] through the complete pipeline
  🎯 [highlight]Detailed operation descriptions[/] showing current step
  📈 [highlight]Performance metrics[/] with timing and success rates
  ⚠️ [highlight]Comprehensive error tracking[/] with detailed failure information
  🖼️ [highlight]Beautiful live layouts[/] with organized information display

[primary]Users now have complete visibility into what TestCraft is doing! 🔍[/]

[dim]This represents a massive improvement in user experience.[/]
[dim]No more staring at spinners wondering what's happening.[/]
[dim]Every operation is tracked and displayed in real-time with beautiful formatting.[/]
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
