#!/usr/bin/env python3
"""
Setup script for pre-commit hooks that match GitHub CI checks.

This script ensures local development matches CI exactly.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """Set up pre-commit hooks matching GitHub CI."""
    print("🚀 Setting up pre-commit hooks for TestCraft...")
    print("📋 This will configure local checks that match GitHub CI exactly\n")

    # Check if we're in a git repository
    if not Path(".git").exists():
        print("❌ Error: Not in a git repository")
        sys.exit(1)

    # Install pre-commit if not available
    print("📦 Installing pre-commit...")
    if not run_command(["uv", "pip", "install", "pre-commit"], "Install pre-commit"):
        sys.exit(1)

    # Install the hooks
    if not run_command(
        ["uv", "run", "pre-commit", "install"], "Install pre-commit hooks"
    ):
        sys.exit(1)

    # Test the setup
    print("\n🧪 Testing pre-commit setup...")
    if run_command(
        ["uv", "run", "pre-commit", "run", "--all-files"], "Run all pre-commit checks"
    ):
        print("\n🎉 Pre-commit setup successful!")
        print("\n📋 Your local environment now matches GitHub CI:")
        print("  ✅ ruff check (linting)")
        print("  ✅ ruff format (code formatting)")
        print("  ✅ mypy (type checking)")
        print("  ✅ safety (security scanning)")
        print("  ✅ documentation check")
        print("  ✅ standard file checks")
        print("\n💡 The hooks will run automatically on every commit!")
        print("   To run manually: uv run pre-commit run --all-files")
    else:
        print("\n⚠️  Pre-commit setup completed but some checks failed.")
        print("   This is normal for the first run - fix any issues and commit again.")
        print("   Manual run: uv run pre-commit run --all-files")


if __name__ == "__main__":
    main()
