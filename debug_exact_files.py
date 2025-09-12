#!/usr/bin/env python3
"""Test script to debug the exact same files the real generate command processes."""

import asyncio
import os
import sys
from pathlib import Path

# Add testcraft to path
sys.path.insert(0, "/Users/wumpinihussein/Documents/code/ai/testcraft/testcraft")


async def test_exact_files():
    """Test the exact same files that the real generate command is processing."""

    from testcraft.cli.dependency_injection import create_dependency_container
    from testcraft.config.loader import load_config

    # Load config exactly as the CLI does
    project_path = Path("/Users/wumpinihussein/Documents/code/ai/weather-collector")
    os.chdir(project_path)

    print(f"🔍 Testing exact files that real generate command processes")

    try:
        # Load config exactly as CLI does
        config = load_config()

        # Create DI container exactly as CLI does
        container = create_dependency_container(config)

        # Get the file discovery service
        file_discovery = container["file_discovery"]

        # Discover files exactly as the real command does
        discovered_files = file_discovery.discover_source_files(".")
        print(f"✅ Discovered {len(discovered_files)} files (same as real command)")

        # Get the parser
        parser = container["parser_adapter"]

        # Test each file that the real command is trying to process
        for i, file_path_str in enumerate(discovered_files):
            file_path = Path(file_path_str)  # Convert to Path object
            print(f"\n🔍 File {i+1}: {file_path}")
            print(f"   File exists: {file_path.exists()}")

            try:
                # Parse using the exact same method as _create_generation_plan_for_file
                parse_result = parser.parse_file(file_path)
                elements = parse_result.get("elements", [])

                print(f"   Parse result type: {type(parse_result)}")
                print(f"   Elements found: {len(elements)}")

                if not elements:
                    print(
                        f"   ❌ NO ELEMENTS - would be skipped! (This matches the real command)"
                    )
                    print(f"   Parse result keys: {list(parse_result.keys())}")
                    print(f"   Parse errors: {parse_result.get('parse_errors', [])}")
                else:
                    print(f"   ✅ Elements found:")
                    for j, element in enumerate(elements):
                        print(f"      Element {j+1}: {element.name} ({element.type})")

            except Exception as parse_exception:
                print(f"   ❌ Parse exception: {parse_exception}")

            # Only test first few files to avoid too much output
            if i >= 9:  # Test the same 10 files as the real command
                break

    except Exception as e:
        print(f"❌ Setup exception occurred: {e}")
        import traceback

        print(f"❌ Setup traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(test_exact_files())
