"""
Demonstration of TestCraft's Beautiful Rich UI Components

This script showcases the enhanced Rich-based UI system with beautiful
colors, icons, progress indicators, and interactive elements.
"""

import time

from testcraft.adapters.io.ui_rich import RichUIAdapter


def demo_beautiful_theme() -> None:
    """Demonstrate the beautiful theme colors and styling."""
    ui = RichUIAdapter()

    ui.console.print("\n🎨 [title]TestCraft Enhanced Theme Showcase[/title]\n")

    # Show different status messages with beautiful colors
    ui.display_success("All tests passed successfully! 🎉", "Success")
    ui.display_info("Processing test generation workflow", "Information")
    ui.display_warning("Some coverage gaps detected", "Warning")
    ui.display_error("Configuration file not found", "Error")

    ui.print_divider("Color Palette")

    # Demonstrate color palette
    colors = [
        ("success", "🟢 Excellent results"),
        ("coverage_excellent", "🏆 95%+ coverage"),
        ("coverage_high", "🟢 85%+ coverage"),
        ("coverage_good", "🟡 70%+ coverage"),
        ("coverage_medium", "🟠 50%+ coverage"),
        ("coverage_low", "🔴 <50% coverage"),
        ("status_pass", "✅ Tests passing"),
        ("status_fail", "❌ Tests failing"),
        ("highlight", "🔥 Important highlights"),
        ("primary", "💎 Primary elements"),
        ("accent", "⚡ Accent elements"),
    ]

    for color, text in colors:
        ui.console.print(f"[{color}]{text}[/]")


def demo_beautiful_tables() -> None:
    """Demonstrate beautiful tables with icons and enhanced formatting."""
    ui = RichUIAdapter()

    ui.print_divider("📊 Beautiful Tables")

    # Coverage table demo
    coverage_data = {
        "files": {
            "src/core/engine.py": {
                "line_coverage": 0.98,
                "branch_coverage": 0.95,
                "missing_lines": [42],
            },
            "src/adapters/llm.py": {
                "line_coverage": 0.87,
                "branch_coverage": 0.82,
                "missing_lines": [15, 28, 45, 67],
            },
            "src/utils/helpers.py": {
                "line_coverage": 0.65,
                "branch_coverage": 0.58,
                "missing_lines": [5, 12, 18, 25, 33, 41, 48],
            },
            "src/cli/commands.py": {
                "line_coverage": 0.42,
                "branch_coverage": 0.38,
                "missing_lines": [10, 15, 22, 28, 35, 42, 48, 55, 62],
            },
        },
        "overall_line_coverage": 0.73,
        "overall_branch_coverage": 0.68,
    }

    ui.rich_cli.print_table(ui.rich_cli.create_coverage_table(coverage_data))

    # Test results table demo
    test_results = [
        {
            "source_file": "src/core/engine.py",
            "test_file": "tests/test_engine.py",
            "status": "success",
            "tests_generated": 12,
            "pass_rate": 1.0,
        },
        {
            "source_file": "src/adapters/llm.py",
            "test_file": "tests/test_llm.py",
            "status": "success",
            "tests_generated": 8,
            "pass_rate": 0.875,
        },
        {
            "source_file": "src/utils/helpers.py",
            "test_file": "tests/test_helpers.py",
            "status": "partial",
            "tests_generated": 5,
            "pass_rate": 0.60,
        },
        {
            "source_file": "src/cli/commands.py",
            "test_file": "tests/test_commands.py",
            "status": "failed",
            "tests_generated": 0,
            "pass_rate": 0.0,
        },
    ]

    ui.rich_cli.print_table(ui.rich_cli.create_test_results_table(test_results))


def demo_beautiful_panels() -> None:
    """Demonstrate beautiful panels and summaries."""
    ui = RichUIAdapter()

    ui.print_divider("📈 Beautiful Panels & Summaries")

    # Project summary panel
    project_data = {
        "total_files": 25,
        "files_with_tests": 20,
        "overall_coverage": 0.78,
        "tests_generated": 45,
        "generation_success_rate": 0.88,
    }

    ui.rich_cli.print_panel(ui.rich_cli.create_project_summary_panel(project_data))

    # Recommendations panel
    recommendations = [
        "Add integration tests for the core engine module",
        "Improve edge case testing in utility functions",
        "Consider adding property-based tests for complex logic",
        "Set up automated coverage reporting in CI/CD",
    ]

    ui.rich_cli.print_panel(ui.rich_cli.create_recommendations_panel(recommendations))


def demo_progress_indicators() -> None:
    """Demonstrate beautiful progress indicators."""
    ui = RichUIAdapter()

    ui.print_divider("⚡ Beautiful Progress Indicators")

    # Spinner demo
    ui.display_progress(
        {"message": "🔍 Analyzing codebase structure..."}, progress_type="spinner"
    )
    time.sleep(2)

    # Progress bar demo
    total_files = 10
    for i in range(total_files + 1):
        ui.display_progress(
            {
                "current": i,
                "total": total_files,
                "message": f"🧪 Generating tests ({i}/{total_files})",
                "percentage": i / total_files,
            },
            progress_type="bar",
            task_id=getattr(demo_progress_indicators, "_task_id", None),
        )

        if (
            not hasattr(demo_progress_indicators, "_task_id")
            and hasattr(ui, "_active_progress")
            and ui._active_progress
        ):
            demo_progress_indicators._task_id = list(ui._active_progress.tasks.keys())[
                0
            ]

        time.sleep(0.3)

    ui._stop_progress_indicators()


def demo_code_highlighting() -> None:
    """Demonstrate beautiful code syntax highlighting."""
    ui = RichUIAdapter()

    ui.print_divider("💻 Beautiful Code Highlighting")

    sample_code = '''
def generate_tests(source_file: Path, coverage_data: CoverageResult) -> TestGenerationResult:
    """
    Generate comprehensive tests for a Python source file.

    Args:
        source_file: Path to the source file to test
        coverage_data: Current coverage information

    Returns:
        Test generation results with success status
    """
    try:
        # Parse the source file to extract testable elements
        parser = PythonParser()
        elements = parser.extract_elements(source_file)

        # Generate test cases using LLM
        llm_adapter = get_llm_adapter()
        test_cases = []

        for element in elements:
            if element.should_test(coverage_data):
                test_case = llm_adapter.generate_test(element)
                test_cases.append(test_case)

        return TestGenerationResult(
            success=True,
            tests_generated=len(test_cases),
            test_cases=test_cases
        )

    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        return TestGenerationResult(success=False, error=str(e))
'''

    ui.display_code_snippet(sample_code, "python", "Test Generation Function")


def demo_interactive_wizard() -> None:
    """Demonstrate the beautiful interactive configuration wizard."""
    ui = RichUIAdapter()

    ui.print_divider("🧙‍♂️ Interactive Configuration Wizard")


    ui.console.print(
        "\n[info]🎯 This is a demo - no actual configuration will be saved[/]\n"
    )

    # In a real scenario, this would run interactively
    # For demo purposes, we'll just show what it looks like
    ui.console.print("[highlight]Demo: Interactive wizard would appear here with:[/]")
    ui.console.print("  • 🧙‍♂️ Step-by-step configuration flow")
    ui.console.print("  • 🎨 Beautiful progress indicators")
    ui.console.print("  • ✨ Rich formatting and validation")
    ui.console.print("  • 📋 Final configuration summary")


def demo_comprehensive_layout() -> None:
    """Demonstrate the comprehensive layout system."""
    ui = RichUIAdapter()

    ui.print_divider("🎨 Comprehensive Layout System")

    data = {
        "summary_data": {
            "total_files": 25,
            "files_with_tests": 22,
            "overall_coverage": 0.84,
            "tests_generated": 67,
            "generation_success_rate": 0.91,
        },
        "coverage_data": {
            "files": {
                "src/main.py": {"line_coverage": 0.95, "branch_coverage": 0.88},
                "src/utils.py": {"line_coverage": 0.78, "branch_coverage": 0.72},
            }
        },
        "test_results": [
            {
                "source_file": "src/main.py",
                "status": "success",
                "tests_generated": 5,
                "pass_rate": 1.0,
            }
        ],
        "recommendations": [
            "Excellent coverage! Consider adding edge case tests",
            "Add integration tests for end-to-end workflows",
        ],
    }

    ui.create_beautiful_summary(data, layout_style="comprehensive")


def main() -> None:
    """Run the complete Rich UI demonstration."""
    ui = RichUIAdapter()

    # Welcome banner
    welcome = """
    🎉 [title]Welcome to TestCraft's Beautiful Rich UI Demo![/title] 🎉

    [info]This demonstration showcases the enhanced Rich-based interface[/]
    [info]with beautiful colors, icons, progress bars, and interactive elements.[/]

    [highlight]✨ Get ready for a visual treat! ✨[/]
    """

    ui.console.print(welcome)
    ui.print_divider()

    try:
        # Run all demonstrations
        demo_beautiful_theme()
        time.sleep(1)

        demo_beautiful_tables()
        time.sleep(1)

        demo_beautiful_panels()
        time.sleep(1)

        demo_progress_indicators()
        time.sleep(1)

        demo_code_highlighting()
        time.sleep(1)

        demo_interactive_wizard()
        time.sleep(1)

        demo_comprehensive_layout()

        # Finale
        ui.print_divider("🎊 Demo Complete!")
        ui.rich_cli.console.print(
            """
[success]🎊 Demo completed successfully![/]

[info]The TestCraft Rich UI system now features:[/]
  🎨 [highlight]Enhanced theme[/] with beautiful colors and gradients
  📊 [highlight]Interactive tables[/] with icons and visual hierarchy
  📈 [highlight]Sophisticated panels[/] with rich formatting
  ⚡ [highlight]Progress indicators[/] with spinners and progress bars
  💻 [highlight]Syntax highlighting[/] for code snippets
  🧙‍♂️ [highlight]Interactive wizards[/] for configuration flows
  🖼️  [highlight]Comprehensive layouts[/] for complex data display

[primary]Ready to make your CLI beautiful? 🚀[/]
        """
        )

    except KeyboardInterrupt:
        ui.console.print("\n[warning]⚠️  Demo interrupted by user[/]")
    except Exception as e:
        ui.display_error(f"Demo failed: {str(e)}", "Demo Error")
    finally:
        ui.finalize()


if __name__ == "__main__":
    main()
