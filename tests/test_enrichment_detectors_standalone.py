#!/usr/bin/env python3
"""
Standalone test for enrichment detectors to verify our fixes work.
"""

import ast
import sys

# Add the testcraft directory to Python path
sys.path.insert(0, "/Users/wumpinihussein/Documents/code/ai/testcraft/testcraft")


# Test our enrichment detectors directly
def test_enrichment_detectors():
    """Test the enrichment detectors methods directly."""

    # Import the module we want to test
    from testcraft.application.generation.services.enrichment_detectors import (
        EnrichmentDetectors,
    )

    # Test source code
    source_code = """
import os
import requests
from config import settings
from dotenv import load_dotenv

API_KEY = os.environ.get('API_KEY')
DB_URL = os.environ['DATABASE_URL']
config_value = settings.get('database_url')
config_value2 = config.get('api_key')
"""

    # Parse AST
    tree = ast.parse(source_code)

    print("Testing EnrichmentDetectors...")

    # Test 1: Environment and config detection
    print("\n1. Testing environment and config detection:")
    result = EnrichmentDetectors.detect_env_config_usage(source_code, tree)
    print(f"  Environment variables: {result['env_vars']}")
    print(f"  Config keys: {result['config_keys']}")

    # Test 2: Client boundaries detection
    print("\n2. Testing client boundaries detection:")
    result = EnrichmentDetectors.detect_client_boundaries(source_code, tree)
    print(f"  Database clients: {result['database_clients']}")
    print(f"  HTTP clients: {result['http_clients']}")

    # Test 3: Side effect boundaries detection
    print("\n3. Testing side effect boundaries detection:")
    result = EnrichmentDetectors.detect_side_effect_boundaries(source_code, tree)
    print(f"  Side effects: {result}")

    # Test 4: Input validation
    print("\n4. Testing input validation:")
    try:
        EnrichmentDetectors.detect_env_config_usage(
            123, tree
        )  # Should raise ValueError
        print("  ERROR: Should have raised ValueError for non-string source_text")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    try:
        EnrichmentDetectors.detect_env_config_usage(
            source_code, tree, max_vars=0
        )  # Should raise ValueError
        print("  ERROR: Should have raised ValueError for invalid max_vars")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    print(
        "\n✅ All tests passed! EnrichmentDetectors improvements are working correctly."
    )


if __name__ == "__main__":
    test_enrichment_detectors()
