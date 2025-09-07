#!/usr/bin/env python3
"""Example usage of the TestCraft configuration system."""

import os
from pathlib import Path
from testcraft.config import TestCraftConfig, ConfigLoader


def main():
    """Demonstrate configuration system usage."""
    
    print("=== TestCraft Configuration System Demo ===\n")
    
    # 1. Basic usage - load default configuration
    print("1. Loading default configuration:")
    config = TestCraftConfig()
    print(f"   Framework: {config.style.framework}")
    print(f"   Min coverage: {config.coverage.minimum_line_coverage}%")
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
    os.environ['TESTCRAFT_COVERAGE__MINIMUM_LINE_COVERAGE'] = '90'
    os.environ['TESTCRAFT_STYLE__FRAMEWORK'] = 'unittest'
    
    # Reload to pick up env vars
    config = loader.load_config(reload=True)
    print(f"   Framework (from env): {config.style.framework}")
    print(f"   Min coverage (from env): {config.coverage.minimum_line_coverage}%")
    
    # 4. CLI overrides (highest priority)
    print("\n4. CLI overrides:")
    cli_overrides = {
        'cost_management': {
            'max_file_size_kb': 100
        },
        'quality': {
            'enable_mutation_testing': False
        }
    }
    
    config = loader.load_config(cli_overrides=cli_overrides, reload=True)
    print(f"   Max file size (CLI): {config.cost_management.max_file_size_kb}KB")
    print(f"   Mutation testing (CLI): {config.quality.enable_mutation_testing}")
    
    # 5. Accessing nested values
    print("\n5. Accessing nested configuration values:")
    print(f"   coverage.minimum_line_coverage: {config.get_nested_value('coverage.minimum_line_coverage')}")
    print(f"   environment.auto_detect: {config.get_nested_value('environment.auto_detect')}")
    print(f"   invalid.key (default): {config.get_nested_value('invalid.key', 'NOT_FOUND')}")
    
    # 6. Configuration updates
    print("\n6. Configuration updates:")
    updates = {
        'test_patterns': {
            'exclude': ['custom_exclude.py']
        }
    }
    
    updated_config = config.update_from_dict(updates)
    print(f"   Original excludes: {config.test_patterns.exclude[:2]}...")
    print(f"   Updated excludes: {updated_config.test_patterns.exclude}")
    
    # 7. Create sample config file
    print("\n7. Creating sample configuration file:")
    sample_path = Path('sample_testcraft_config.yml')
    if sample_path.exists():
        sample_path.unlink()  # Remove if exists
        
    created_path = loader.create_sample_config(sample_path)
    print(f"   Sample config created at: {created_path}")
    print(f"   File size: {created_path.stat().st_size} bytes")
    
    # Clean up
    sample_path.unlink()
    for key in ['TESTCRAFT_COVERAGE__MINIMUM_LINE_COVERAGE', 'TESTCRAFT_STYLE__FRAMEWORK']:
        os.environ.pop(key, None)
    
    print("\n=== Demo Complete ===")


if __name__ == '__main__':
    main()
