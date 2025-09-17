#!/usr/bin/env python3
"""
TestCraft Unified LLM System Demo

This example demonstrates the unified LLM adapter system that allows
seamless switching between providers while preserving provider-specific
capabilities and using consistent PromptRegistry-based prompts.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from testcraft.config import TestCraftConfig
from testcraft.adapters.llm.router import LLMRouter
from testcraft.ports.cost_port import CostPort


def demo_provider_interchangeability():
    """Demonstrate how providers are fully interchangeable."""
    print("=== TestCraft Unified LLM System Demo ===\n")
    
    # Load config (could use any provider)
    config = TestCraftConfig()
    llm_config = config.llm.model_dump()
    
    print("1. Provider Interchangeability:")
    print(f"   Current provider: {llm_config.get('default_provider', 'openai')}")
    print("   ✓ Switch providers by changing config.llm.default_provider")
    print("   ✓ All operations return identical response schemas")
    print("   ✓ Provider-specific features preserved automatically\n")
    
    # Initialize router (works with any configured provider)
    cost_port = CostPort()
    router = LLMRouter(config=llm_config, cost_port=cost_port)
    
    return router


def demo_capabilities_detection(router):
    """Demonstrate capability detection across providers."""
    print("2. Provider Capabilities:")
    
    capabilities = router.get_capabilities()
    if capabilities:
        print(f"   Thinking mode support: {capabilities.get('supports_thinking_mode', False)}")
        print(f"   Reasoning model: {capabilities.get('is_reasoning_model', False)}")
        print(f"   Max context tokens: {capabilities.get('max_context_tokens', 'Unknown')}")
        print(f"   Max output tokens: {capabilities.get('max_output_tokens', 'Unknown')}")
        
        if capabilities.get('max_thinking_tokens'):
            print(f"   Max thinking tokens: {capabilities['max_thinking_tokens']}")
    else:
        print("   Capability detection requires valid API credentials")
    print()


def demo_unified_operations(router):
    """Demonstrate unified operations with consistent schemas."""
    print("3. Unified Operations (requires API keys):")
    
    # Sample code for demonstration
    sample_code = '''
def calculate_area(radius: float) -> float:
    """Calculate the area of a circle."""
    import math
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return math.pi * radius ** 2
'''
    
    print("   All operations use PromptRegistry for consistent prompts")
    print("   All operations return standardized response schemas")
    print()
    
    # Check if we have API credentials before making calls
    providers = ['openai', 'anthropic', 'azure-openai', 'bedrock']
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'), 
        'azure-openai': os.getenv('AZURE_OPENAI_API_KEY'),
        'bedrock': os.getenv('AWS_ACCESS_KEY_ID')
    }
    
    current_provider = router.default_provider
    if not api_keys.get(current_provider):
        print(f"   ⚠️  No API key found for {current_provider}")
        print("   Set appropriate environment variable to test operations:")
        if current_provider == 'openai':
            print("     export OPENAI_API_KEY='your-key'")
        elif current_provider == 'anthropic':
            print("     export ANTHROPIC_API_KEY='your-key'")
        elif current_provider == 'azure-openai':
            print("     export AZURE_OPENAI_API_KEY='your-key'")
        elif current_provider == 'bedrock':
            print("     export AWS_ACCESS_KEY_ID='your-key'")
            print("     export AWS_SECRET_ACCESS_KEY='your-secret'")
        print()
        return
    
    try:
        print(f"   Testing with {current_provider} provider...")
        
        # Test generate_tests operation
        print("   • generate_tests() - Uses llm_test_generation prompts")
        result = router.generate_tests(
            code_content=sample_code,
            test_framework="pytest"
        )
        print(f"     Response schema: {list(result.keys())}")
        
        # Test analyze_code operation  
        print("   • analyze_code() - Uses llm_code_analysis prompts")
        result = router.analyze_code(sample_code)
        print(f"     Response schema: {list(result.keys())}")
        
        # Test refine_content operation
        print("   • refine_content() - Uses llm_content_refinement prompts")
        result = router.refine_content(
            "print('hello')", 
            "Add proper error handling"
        )
        print(f"     Response schema: {list(result.keys())}")
        
        # Test generate_test_plan operation
        print("   • generate_test_plan() - Uses llm_test_planning_v1 prompts")
        result = router.generate_test_plan(sample_code)
        print(f"     Response schema: {list(result.keys())}")
        
    except Exception as e:
        print(f"   ⚠️  Operation failed: {e}")
        print("   This is expected if API credentials are invalid")
    print()


def demo_provider_switching():
    """Demonstrate switching providers at runtime."""
    print("4. Runtime Provider Switching:")
    print("   # Switch to Claude for large context")
    print("   config.llm.default_provider = 'anthropic'")
    print("   router = LLMRouter(config=config.llm.model_dump())")
    print()
    print("   # Switch to OpenAI o-series for reasoning")
    print("   config.llm.default_provider = 'openai'")
    print("   config.llm.openai_model = 'o1-preview'")
    print("   router = LLMRouter(config=config.llm.model_dump())")
    print()
    print("   # Switch to Azure for enterprise")
    print("   config.llm.default_provider = 'azure-openai'")
    print("   router = LLMRouter(config=config.llm.model_dump())")
    print()


def demo_best_practices():
    """Show best practices for using the unified system."""
    print("5. Best Practices:")
    print("   ✓ Use LLMRouter instead of direct adapter instantiation")
    print("   ✓ All prompts managed centrally via PromptRegistry")
    print("   ✓ Provider capabilities detected automatically")
    print("   ✓ Token budgeting handled per-request via TokenCalculator")
    print("   ✓ Metadata normalized across all providers")
    print("   ✓ Cost tracking unified via CostPort")
    print("   ✓ Error handling standardized via LLMError boundary")
    print()


def main():
    """Run the unified LLM system demonstration."""
    try:
        router = demo_provider_interchangeability()
        demo_capabilities_detection(router)
        demo_unified_operations(router)
        demo_provider_switching()
        demo_best_practices()
        
        print("=== Demo Complete ===")
        print("\nKey Takeaways:")
        print("• All four providers (OpenAI, Anthropic, Azure, Bedrock) are fully interchangeable")
        print("• Provider-specific features (thinking modes, reasoning) work seamlessly")
        print("• Consistent response schemas and metadata across all providers")
        print("• PromptRegistry ensures consistent prompts; no more inline prompts")
        print("• Switch providers by changing one configuration value")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure TestCraft is properly installed and configured")


if __name__ == "__main__":
    main()
