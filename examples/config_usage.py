#!/usr/bin/env python3
"""Example usage of the TestCraft configuration system."""

import os
from pathlib import Path

from testcraft.config import ConfigLoader, TestCraftConfig


def main():
    """Demonstrate configuration system usage."""

    print("=== TestCraft Configuration System Demo ===\n")

    # 1. Basic usage - load default configuration
    print("1. Loading default configuration:")
    config = TestCraftConfig()
    print(f"   Framework: {config.generation.test_framework}")
    print(f"   Batch size: {config.generation.batch_size}")
    print(f"   Cost limit: ${config.cost_management.cost_thresholds.daily_limit}")

    # 2. Using ConfigLoader for more control
    print("\n2. Using ConfigLoader:")
    loader = ConfigLoader()
    config = loader.load_config()
    summary = loader.get_config_summary()
    print(f"   Config file: {summary['config_file']}")
    print(f"   Environment manager: {summary['environment_manager']}")

    # 3. Environment variable overrides
    print("\n3. Environment variable overrides:")
    os.environ["TESTCRAFT_GENERATION__BATCH_SIZE"] = "8"
    os.environ["TESTCRAFT_GENERATION__TEST_FRAMEWORK"] = "unittest"

    # Reload to pick up env vars
    config = loader.load_config(reload=True)
    print(f"   Framework (from env): {config.generation.test_framework}")
    print(f"   Batch size (from env): {config.generation.batch_size}")

    # 4. CLI overrides (highest priority)
    print("\n4. CLI overrides:")
    cli_overrides = {
        "cost_management": {"max_file_size_kb": 100},
        "generation": {"enable_refinement": False},
    }

    config = loader.load_config(cli_overrides=cli_overrides, reload=True)
    print(f"   Max file size (CLI): {config.cost_management.max_file_size_kb}KB")
    print(f"   Enable refinement (CLI): {config.generation.enable_refinement}")

    # 5. Accessing nested values
    print("\n5. Accessing nested configuration values:")
    print(
        f"   generation.test_framework: {config.get_nested_value('generation.test_framework')}"
    )
    print(
        f"   environment.auto_detect: {config.get_nested_value('environment.auto_detect')}"
    )
    print(
        f"   generation.prompt_budgets.total_chars: {config.get_nested_value('generation.prompt_budgets.total_chars')}"
    )
    print(
        f"   invalid.key (default): {config.get_nested_value('invalid.key', 'NOT_FOUND')}"
    )

    # 6. Configuration updates
    print("\n6. Configuration updates:")
    updates = {"test_patterns": {"exclude": ["custom_exclude.py"]}}

    updated_config = config.update_from_dict(updates)
    print(f"   Original excludes: {config.test_patterns.exclude[:2]}...")
    print(f"   Updated excludes: {updated_config.test_patterns.exclude}")

    # 7. Create sample config file
    print("\n7. Creating sample configuration file:")
    sample_path = Path("sample_testcraft_config.yml")
    if sample_path.exists():
        sample_path.unlink()  # Remove if exists

    created_path = loader.create_sample_config(sample_path)
    print(f"   Sample config created at: {created_path}")
    print(f"   File size: {created_path.stat().st_size} bytes")

    # 8. Unified LLM configuration
    print("\n8. Unified LLM configuration:")
    print(f"   Default provider: {config.llm.default_provider}")
    print(f"   Temperature: {config.llm.temperature}")
    print(f"   Max retries: {config.llm.max_retries}")
    print(f"   OpenAI model: {config.llm.openai_model}")
    print(f"   Anthropic model: {config.llm.anthropic_model}")
    
    # Show provider switching
    print("\n9. Provider switching demonstration:")
    provider_examples = {
        "openai": "For general use and o-series reasoning models",
        "anthropic": "For large context and extended thinking",
        "azure-openai": "For enterprise deployments",
        "bedrock": "For AWS infrastructure integration"
    }
    
    for provider, description in provider_examples.items():
        print(f"   {provider}: {description}")
    
    print(f"\n   Current: Using {config.llm.default_provider}")
    print("   Switch: Set TESTCRAFT_LLM__DEFAULT_PROVIDER=anthropic")

    # Clean up
    sample_path.unlink()
    for key in [
        "TESTCRAFT_GENERATION__BATCH_SIZE",
        "TESTCRAFT_GENERATION__TEST_FRAMEWORK",
    ]:
        os.environ.pop(key, None)

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
