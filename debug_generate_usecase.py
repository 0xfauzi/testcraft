#!/usr/bin/env python3
"""Test script to debug the GenerateUseCase._create_generation_plan_for_file method."""

import asyncio
import os
import sys
from pathlib import Path

# Add testcraft to path
sys.path.insert(0, "/Users/wumpinihussein/Documents/code/ai/testcraft/testcraft")


async def test_generate_usecase_method():
    """Test the exact _create_generation_plan_for_file method that's failing."""

    from testcraft.cli.dependency_injection import create_dependency_container
    from testcraft.config.loader import load_config

    # Load config exactly as the CLI does
    project_path = Path("/Users/wumpinihussein/Documents/code/ai/weather-collector")
    os.chdir(project_path)

    print(f"🔍 Testing GenerateUseCase method in: {project_path}")

    try:
        # Load config exactly as CLI does
        config = load_config()
        print("✅ Config loaded successfully")

        # Create DI container exactly as CLI does
        container = create_dependency_container(config)
        print("✅ DI container created successfully")

        # Get the generate usecase from DI
        generate_usecase = container["generate_usecase"]
        print(f"✅ GenerateUseCase retrieved from DI: {type(generate_usecase)}")

        # Test on the same file that's failing
        test_file = project_path / "src/weather_collector/config.py"
        print(f"🔍 Testing on file: {test_file}")

        # Call the actual method that's failing
        try:
            result = await generate_usecase._create_generation_plan_for_file(test_file)
            print("✅ _create_generation_plan_for_file completed")

            if result is None:
                print("❌ Result is None - no testable elements found!")
            else:
                print(f"✅ Result type: {type(result)}")
                print(f"   Elements to test: {len(result.elements_to_test)}")
                print(f"   Existing tests: {len(result.existing_tests)}")
                print(f"   Coverage before: {result.coverage_before}")

                for i, element in enumerate(result.elements_to_test):
                    print(f"   Element {i + 1}: {element.name} ({element.type})")

        except Exception as method_exception:
            print(
                f"❌ Exception in _create_generation_plan_for_file: {method_exception}"
            )
            import traceback

            print(f"❌ Method traceback: {traceback.format_exc()}")

    except Exception as e:
        print(f"❌ Setup exception occurred: {e}")
        import traceback

        print(f"❌ Setup traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(test_generate_usecase_method())
