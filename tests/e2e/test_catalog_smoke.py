"""End-to-end smoke tests for catalog integration with TestCraft workflows.

This module contains minimal smoke tests that verify the catalog system
works properly with typical TestCraft use cases and workflows.

Extracted from the original test_catalog_validation.py as part of the refactoring
to split concerns and keep smoke tests focused and lightweight.
"""

import pytest
from unittest.mock import Mock

from testcraft.config.model_catalog_loader import resolve_model
from tests.conftest import provider_model_cases


class TestGenerationWorkflowSmoke:
    """Smoke tests for generation workflow with catalog limits."""

    def test_generate_usecase_respects_catalog_limits(self):
        """Test that generate use case respects catalog limits end-to-end."""
        from testcraft.application.generate_usecase import GenerateUseCase
        from testcraft.adapters.llm.openai import OpenAIAdapter
        
        # Test with a model that should be in catalog
        model = "gpt-4o"
        entry = resolve_model("openai", model)
        
        if not entry:
            pytest.skip(f"Model openai/{model} not found in catalog")
            
        # Create real adapters with mocked dependencies
        mock_writer = Mock()
        mock_coverage = Mock()
        mock_refine = Mock() 
        mock_context = Mock()
        mock_parser = Mock()
        mock_state = Mock()
        mock_telemetry = Mock()
        mock_file_discovery = Mock()
        
        # Real LLM adapter that should respect catalog limits
        llm_adapter = OpenAIAdapter(model=model)
        
        use_case = GenerateUseCase(
            llm_port=llm_adapter,
            writer_port=mock_writer,
            coverage_port=mock_coverage,
            refine_port=mock_refine,
            context_port=mock_context,
            parser_port=mock_parser,
            state_port=mock_state,
            telemetry_port=mock_telemetry,
            file_discovery_service=mock_file_discovery,
        )
        
        # Verify the use case uses catalog-aware components
        assert hasattr(llm_adapter, 'token_calculator')
        calc = llm_adapter.token_calculator
        
        # The token calculator should respect catalog limits
        assert calc.limits.max_context == entry.limits.max_context
        assert calc.limits.max_output <= entry.limits.default_max_output
        
        # Max tokens should be within catalog bounds
        assert llm_adapter.max_tokens <= entry.limits.default_max_output

    def test_analyze_usecase_respects_catalog_limits(self):
        """Test that analyze use case respects catalog limits end-to-end."""
        from testcraft.application.analyze_usecase import AnalyzeUseCase
        from testcraft.adapters.llm.claude import ClaudeAdapter
        
        model = "claude-3-5-sonnet"
        entry = resolve_model("anthropic", model)
        
        if not entry:
            pytest.skip(f"Model anthropic/{model} not found in catalog")
            
        # Create mocked dependencies for AnalyzeUseCase
        mock_coverage = Mock()
        mock_state = Mock()
        mock_telemetry = Mock()
        mock_file_discovery = Mock()
        
        # Real LLM adapter that should respect catalog limits
        llm_adapter = ClaudeAdapter(model=model)
        
        # AnalyzeUseCase doesn't take llm_port, but we can test the adapter independently
        use_case = AnalyzeUseCase(
            coverage_port=mock_coverage,
            state_port=mock_state,
            telemetry_port=mock_telemetry,
            file_discovery_service=mock_file_discovery,
        )
        
        # Verify catalog compliance on the adapter we would use
        calc = llm_adapter.token_calculator
        assert calc.limits.max_context == entry.limits.max_context
        assert calc.limits.max_output <= entry.limits.default_max_output
        assert llm_adapter.max_tokens <= entry.limits.default_max_output


class TestRefinementWorkflowSmoke:
    """Smoke tests for refinement workflow with catalog integration."""

    def test_refine_workflow_respects_catalog_limits(self):
        """Test that refine workflow respects catalog limits end-to-end."""
        from testcraft.adapters.llm.openai import OpenAIAdapter
        from testcraft.adapters.llm.pricing import calculate_cost
        
        model = "gpt-4o-mini"  # Use mini for cost-effectiveness
        entry = resolve_model("openai", model)
        
        if not entry:
            pytest.skip(f"Model openai/{model} not found in catalog")
            
        # Create adapter for refinement
        llm_adapter = OpenAIAdapter(model=model)
        
        # Verify catalog compliance
        calc = llm_adapter.token_calculator
        assert calc.limits.max_context == entry.limits.max_context
        assert calc.limits.max_output <= entry.limits.default_max_output
        
        # Test that refinement use case gets appropriate token allocation
        refine_tokens = calc.calculate_max_tokens("refinement")
        assert refine_tokens <= entry.limits.default_max_output
        assert refine_tokens > 0
        
        # Test cost calculation integration
        if entry.pricing and entry.pricing.per_million:
            mock_usage = {"prompt_tokens": 1000, "completion_tokens": refine_tokens}
            cost = calculate_cost(mock_usage, model, "openai")
            assert cost > 0  # Should calculate a positive cost


class TestWorkflowTokenBudgets:
    """Smoke tests for typical workflow token budgets."""

    @pytest.mark.parametrize("provider,model", [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-3-5-sonnet"),
    ])
    def test_typical_workflow_token_budgets_stay_within_catalog_bounds(self, provider, model):
        """Test that typical workflow token budgets stay within catalog bounds."""
        from testcraft.adapters.llm.token_calculator import TokenCalculator
        
        entry = resolve_model(provider, model)
        if not entry:
            pytest.skip(f"Model {provider}/{model} not found in catalog")
            
        calc = TokenCalculator(provider=provider, model=model)
        
        # Test different workflow scenarios
        workflows = [
            ("test_generation", "Large test file generation"),
            ("code_analysis", "Complex code analysis"),
            ("refinement", "Test refinement and improvement"),
        ]
        
        for use_case, description in workflows:
            # Test with different complexity levels
            for complexity in ["simple", "moderate", "complex"]:
                max_tokens = calc.calculate_max_tokens(use_case)
                
                # Should always respect catalog limits
                assert max_tokens <= entry.limits.default_max_output, f"Failed for {provider}/{model} {use_case}"
                assert max_tokens > 0, f"No tokens allocated for {provider}/{model} {use_case}"
                
                # Test thinking tokens if supported
                if calc.supports_thinking_mode():
                    thinking_tokens = calc.calculate_thinking_tokens(use_case, complexity)
                    if thinking_tokens is not None:
                        assert thinking_tokens <= entry.limits.max_thinking, f"Thinking tokens exceeded for {provider}/{model}"


class TestCostTrackingSmoke:
    """Smoke tests for cost tracking integration."""

    def test_cost_tracking_integration_with_catalog_pricing(self):
        """Test that cost tracking properly integrates with catalog pricing."""
        from testcraft.adapters.llm.pricing import get_pricing, calculate_cost
        from testcraft.adapters.llm.token_calculator import TokenCalculator
        from testcraft.config.model_catalog_loader import load_catalog
        
        # Find models with pricing in catalog
        catalog = load_catalog()
        models_with_pricing = []
        
        for model in catalog.models:
            if model.pricing and model.pricing.per_million and model.pricing.per_million.input:
                models_with_pricing.append((model.provider, model.model_id))
        
        if not models_with_pricing:
            pytest.skip("No models with pricing found in catalog")
            
        # Test first available model to keep test fast
        provider, model_id = models_with_pricing[0]
        entry = resolve_model(provider, model_id)
        
        if not entry:
            pytest.skip(f"Model {provider}/{model_id} not found in catalog")
            
        # Get pricing
        pricing = get_pricing(model_id, provider)
        assert pricing["input"] > 0
        assert pricing["output"] > 0
        
        # Calculate typical usage tokens
        calc = TokenCalculator(provider=provider, model=model_id)
        typical_output = calc.calculate_max_tokens("test_generation")
        
        # Simulate typical usage
        typical_usage = {
            "prompt_tokens": 2000,  # Typical input
            "completion_tokens": typical_output
        }
        
        # Calculate cost
        cost = calculate_cost(typical_usage, model_id, provider)
        assert cost > 0
        
        # Cost should be reasonable (not negative, not extremely high)
        assert cost < 100.0  # Should be less than $100 for typical usage


class TestErrorHandlingSmoke:
    """Smoke tests for error handling."""

    def test_error_handling_with_catalog_validation(self):
        """Test error handling when catalog validation fails."""
        from testcraft.adapters.llm.token_calculator import TokenCalculator
        from testcraft.adapters.llm.pricing import get_pricing, PricingError
        
        # Test with nonexistent model
        unknown_calc = TokenCalculator(provider="unknown", model="unknown")
        
        # Should fall back to safe defaults without crashing
        assert unknown_calc.limits.max_context > 0
        assert unknown_calc.limits.max_output > 0
        
        # Should still be able to calculate tokens
        tokens = unknown_calc.calculate_max_tokens("test_generation")
        assert tokens > 0
        
        # Test pricing error handling
        with pytest.raises(PricingError):
            get_pricing("nonexistent-model", "nonexistent-provider")
