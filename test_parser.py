#!/usr/bin/env python3
"""Test script to debug the AST parsing issue."""

import sys
from pathlib import Path

# Add testcraft to path
sys.path.insert(0, "/Users/wumpinihussein/Documents/code/ai/testcraft/testcraft")

from testcraft.adapters.parsing.codebase_parser import CodebaseParser
from testcraft.domain.models import (TestElement, TestElementType,
                                     TestGenerationPlan)


def test_parser():
    """Test the parser on a real file."""
    parser = CodebaseParser()

    # Test on the weather collector config file
    test_file = Path(
        "/Users/wumpinihussein/Documents/code/ai/weather-collector/src/weather_collector/config.py"
    )

    print(f"Testing parser on: {test_file}")
    print(f"File exists: {test_file.exists()}")

    try:
        result = parser.parse_file(test_file)

        print(f"\nParsing result:")
        print(f"- Language: {result.get('language')}")
        print(f"- Parse errors: {result.get('parse_errors', [])}")
        print(f"- Number of elements: {len(result.get('elements', []))}")
        print(f"- Number of imports: {len(result.get('imports', []))}")

        elements = result.get("elements", [])
        if elements:
            print(f"\nFound elements:")
            for i, elem in enumerate(elements):
                print(f"  {i+1}. {elem.name} ({elem.type}) - lines {elem.line_range}")
                if elem.docstring:
                    print(f"     Docstring: {elem.docstring[:100]}...")

                # Test TestElement validation
                print(f"     Element type: {type(elem)}")
                print(f"     Is TestElement instance: {isinstance(elem, TestElement)}")

        else:
            print("\n❌ No elements found!")

        # Test TestGenerationPlan creation with the elements
        print(f"\n=== Testing TestGenerationPlan creation ===")
        try:
            if elements:
                plan = TestGenerationPlan(
                    elements_to_test=elements,
                    existing_tests=[],
                    coverage_before=None,
                )
                print(f"✅ TestGenerationPlan created successfully!")
                print(f"   - Elements in plan: {len(plan.elements_to_test)}")
            else:
                print("❌ Cannot create plan - no elements!")
        except Exception as plan_error:
            print(f"❌ TestGenerationPlan creation failed: {plan_error}")
            import traceback

            traceback.print_exc()

        imports = result.get("imports", [])
        if imports:
            print(f"\nFound imports:")
            for imp in imports[:5]:  # Show first 5
                print(f"  - {imp}")

    except Exception as e:
        print(f"❌ Parser failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_parser()
