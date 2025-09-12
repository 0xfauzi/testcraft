#!/usr/bin/env python3
"""
TestCraft Textual TUI Demo

This demo shows how to use TestCraft's new Textual-based Terminal User Interface.
It demonstrates the main features including file processing, statistics display,
and interactive configuration.

Usage:
    python examples/textual_demo.py

Requirements:
    - TestCraft with textual dependencies installed
    - Terminal with decent size (80x24 minimum recommended)
    - Python 3.11+
"""

import asyncio
import sys
from pathlib import Path

# Add testcraft to path for demo
sys.path.insert(0, str(Path(__file__).parent.parent))

from testcraft.adapters.textual.app import TestCraftTextualApp


def main():
    """Run the TestCraft Textual TUI demo."""
    print("🚀 Starting TestCraft Textual TUI Demo...")
    print("📝 Navigate with keyboard shortcuts:")
    print("   G - Generate tests")
    print("   A - Analyze code") 
    print("   C - Coverage analysis")
    print("   S - Status & history")
    print("   W - Configuration wizard")
    print("   Q - Quit")
    print()
    print("Press any key to launch the TUI...")
    
    try:
        input()  # Wait for user input
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        return
    
    # Create and run the Textual app
    try:
        app = TestCraftTextualApp()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Demo session ended by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def run_headless_demo():
    """Run a headless demo for testing purposes."""
    print("🤖 Running headless Textual TUI demo...")
    
    # Create app instance
    app = TestCraftTextualApp()
    
    # Use Textual's testing capabilities
    async def demo_interactions():
        """Simulate some demo interactions."""
        print("  • App created successfully")
        
        # Test app initialization
        print("  • Testing app initialization...")
        
        # Simulate some file updates
        print("  • Simulating file processing...")
        app.update_file_status("example.py", "running", 25.0, 2, 1.5)
        app.update_file_status("utils.py", "done", 100.0, 5, 2.8)
        app.update_file_status("models.py", "failed", 50.0, 0, 1.2, "Syntax error")
        
        # Update stats
        app.update_stats({
            "files_total": 10,
            "files_done": 7,
            "files_failed": 1,
            "files_running": 1,
            "files_pending": 1,
            "tests_generated": 42,
            "operation": "Demo Mode",
        })
        
        print("  ✅ Headless demo completed successfully")
    
    # Run the demo
    try:
        asyncio.run(demo_interactions())
    except Exception as e:
        print(f"  ❌ Headless demo failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TestCraft Textual TUI Demo")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless demo for testing (no interactive TUI)"
    )
    
    args = parser.parse_args()
    
    if args.headless:
        success = run_headless_demo()
        sys.exit(0 if success else 1)
    else:
        main()
