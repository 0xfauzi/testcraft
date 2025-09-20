#!/usr/bin/env python3
"""Test script to debug the full generation pipeline."""

import os
import sys
from pathlib import Path

# Add testcraft to path
sys.path.insert(0, "/Users/wumpinihussein/Documents/code/ai/testcraft/testcraft")

from testcraft.adapters.parsing.codebase_parser import CodebaseParser
from testcraft.domain.models import TestGenerationPlan


def test_full_pipeline():
    """Test the full pipeline exactly as GenerateUseCase does."""

    # Simulate the exact same setup as GenerateUseCase
    parser = CodebaseParser()

    # Test on the same file the real command is failing on
    test_file = Path(
        "/Users/wumpinihussein/Documents/code/ai/weather-collector/src/weather_collector/config.py"
    )

    print(f"🔍 Testing full pipeline on: {test_file}")
    print(f"   File exists: {test_file.exists()}")
    print(f"   Absolute path: {test_file.absolute()}")
    print(f"   Current working directory: {os.getcwd()}")

    try:
        # Step 1: Parse the file exactly as GenerateUseCase does
        print("\n=== Step 1: Parse file ===")
        parse_result = parser.parse_file(test_file)
        elements = parse_result.get("elements", [])

        print(f"Parse result keys: {list(parse_result.keys())}")
        print(f"Elements type: {type(elements)}")
        print(f"Elements length: {len(elements)}")

        # Step 2: Check elements exactly as GenerateUseCase does
        print("\n=== Step 2: Check elements ===")
        if not elements:
            print("❌ Skipping: no testable elements found")
            return
        else:
            print(f"✅ Found {len(elements)} testable elements")
            for i, elem in enumerate(elements):
                print(f"   {i + 1}. {elem.name} ({elem.type})")

        # Step 3: Find existing test files (simplified)
        print("\n=== Step 3: Find existing tests ===")
        existing_tests = []  # Simplified
        print(f"Existing tests: {existing_tests}")

        # Step 4: Create TestGenerationPlan exactly as GenerateUseCase does
        print("\n=== Step 4: Create TestGenerationPlan ===")
        plan = TestGenerationPlan(
            elements_to_test=elements,
            existing_tests=existing_tests,
            coverage_before=None,
        )
        print("✅ TestGenerationPlan created successfully!")
        print(f"   Elements in plan: {len(plan.elements_to_test)}")

        # Test what happens when we convert to dict/JSON (for debugging)
        print("\n=== Step 5: Plan serialization test ===")
        plan_dict = plan.dict()
        print(f"Plan dict elements count: {len(plan_dict['elements_to_test'])}")

        return plan

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_with_relative_paths():
    """Test with relative paths like TestCraft might use."""
    print(f"\n{'=' * 60}")
    print("🔍 Testing with relative paths...")

    # Change to weather collector directory
    original_cwd = os.getcwd()
    try:
        os.chdir("/Users/wumpinihussein/Documents/code/ai/weather-collector")
        print(f"Changed CWD to: {os.getcwd()}")

        parser = CodebaseParser()

        # Test with relative path
        relative_path = Path("src/weather_collector/config.py")
        print(f"Testing relative path: {relative_path}")
        print(f"Relative path exists: {relative_path.exists()}")

        parse_result = parser.parse_file(relative_path)
        elements = parse_result.get("elements", [])

        print("Relative path parsing:")
        print(f"   Elements found: {len(elements)}")

        if elements:
            for i, elem in enumerate(elements):
                print(f"   {i + 1}. {elem.name} ({elem.type})")
        else:
            print("   ❌ No elements found with relative path!")

    finally:
        os.chdir(original_cwd)
        print(f"Restored CWD to: {os.getcwd()}")


if __name__ == "__main__":
    plan = test_full_pipeline()
    test_with_relative_paths()

    if plan:
        print("\n✅ Full pipeline test PASSED")
    else:
        print("\n❌ Full pipeline test FAILED")
