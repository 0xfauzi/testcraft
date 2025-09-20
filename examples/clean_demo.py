"""
Clean TestCraft Demo - Minimalist and Stylish

Simple demonstration of the new minimal UI design:
- Clean typography with reduced visual noise
- Essential information only
- Subtle colors and minimal formatting
- Professional but unobtrusive appearance
"""

import sys
from pathlib import Path

# Add the parent directory to Python path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from testcraft.adapters.io.enhanced_ui import EnhancedUIAdapter


def demo_clean_tables():
    """Demo clean, minimal table design."""
    ui = EnhancedUIAdapter()

    ui.console.print("[header]testcraft[/]\n")

    # Clean sample data
    files_data = [
        {
            "file_path": "src/main.py",
            "status": "completed",
            "progress": 100.0,
            "tests_generated": 5,
            "duration": 2.3,
        },
        {
            "file_path": "src/auth.py",
            "status": "active",
            "progress": 60.0,
            "tests_generated": 0,
            "duration": 1.1,
        },
        {
            "file_path": "src/models.py",
            "status": "failed",
            "progress": 30.0,
            "tests_generated": 0,
            "duration": 0.8,
        },
        {
            "file_path": "src/utils.py",
            "status": "waiting",
            "progress": 0.0,
            "tests_generated": 0,
            "duration": 0.0,
        },
    ]

    ui.display_file_progress_table(files_data, "processing")


def demo_clean_success():
    """Demo minimal success display."""
    ui = EnhancedUIAdapter()

    ui.console.print("\n[header]results[/]\n")

    summary_data = {
        "message": "generated tests for 3 of 4 files",
        "metrics": {
            "generation": {"duration": 8.2, "items_processed": 4, "success_rate": 0.75}
        },
    }

    ui.display_success_summary(summary_data)


def demo_clean_error():
    """Demo minimal error handling."""
    ui = EnhancedUIAdapter()

    ui.console.print("\n[header]errors[/]\n")

    ui.display_error_with_suggestions(
        "api key not found in environment",
        [
            "add API_KEY to .env file",
            "check key format is correct",
            "verify account has credits",
        ],
        "config",
    )


def main():
    """Run clean demo."""
    ui = EnhancedUIAdapter()

    ui.console.clear()
    ui.console.print("[header]testcraft minimal ui demo[/]")
    ui.console.print("[muted]clean • simple • focused[/]\n")

    demo_clean_tables()
    demo_clean_success()
    demo_clean_error()

    ui.console.print("\n[header]design principles[/]")
    ui.console.print("[success]✓[/] minimal color palette")
    ui.console.print("[success]✓[/] essential information only")
    ui.console.print("[success]✓[/] clean typography")
    ui.console.print("[success]✓[/] reduced visual noise")
    ui.console.print("[success]✓[/] professional appearance\n")


if __name__ == "__main__":
    main()
