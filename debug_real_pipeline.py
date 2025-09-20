#!/usr/bin/env python3
"""Test script to debug the real generation pipeline using exact same DI setup."""

import os
import sys
from pathlib import Path

# Add testcraft to path
sys.path.insert(0, "/Users/wumpinihussein/Documents/code/ai/testcraft/testcraft")


async def test_real_pipeline():
    """Test using the exact same DI setup as the real command."""

    from testcraft.cli.dependency_injection import create_dependency_container
    from testcraft.config.loader import load_config

    # Load config exactly as the CLI does
    project_path = Path("/Users/wumpinihussein/Documents/code/ai/weather-collector")
    os.chdir(project_path)

    print(f"🔍 Testing real pipeline in: {project_path}")
    print(f"   Current working directory: {os.getcwd()}")

    try:
        # Load config exactly as CLI does
        config = load_config()
        print("✅ Config loaded successfully")

        # Create DI container exactly as CLI does
        container = create_dependency_container(config)
        print("✅ DI container created successfully")

        # Get the parser from DI
        parser = container["parser_adapter"]
        print(f"✅ Parser retrieved from DI: {type(parser)}")

        # Test on the same file that's failing
        test_file = project_path / "src/weather_collector/config.py"
        print(f"🔍 Testing on file: {test_file}")
        print(f"   File exists: {test_file.exists()}")
        print(f"   File is readable: {test_file.is_file()}")

        # Parse using the exact same parser instance
        result = parser.parse_file(test_file)
        print("✅ Parsing completed")
        print(f"   Result type: {type(result)}")
        print(
            f"   Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}"
        )

        elements = result.get("elements", [])
        print(f"   Elements found: {len(elements)}")

        if elements:
            for i, element in enumerate(elements):
                print(f"   Element {i + 1}: {element.name} ({element.type})")
        else:
            print("   ❌ NO ELEMENTS FOUND!")
            print(f"   Raw result: {result}")

    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        import traceback

        print(f"❌ Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_real_pipeline())
