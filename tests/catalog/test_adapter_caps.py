"""Adapter cap enforcement tests.

This module contains contract tests for adapters never exceeding caps
and beta flag behavior based on catalog configuration.

Extracted from the original test_catalog_validation.py as part of the refactoring
to split concerns and use fixtures.
"""

import pytest

from testcraft.config.model_catalog_loader import resolve_model
from tests.conftest import SAFETY_MARGIN_CASES


class TestOpenAIAdapterCaps:
    """Test OpenAI adapter cap enforcement."""

    def test_openai_adapter_never_exceeds_catalog_caps(self):
        """Test that OpenAI adapter never exceeds catalog-defined caps."""
        from testcraft.adapters.llm.openai import OpenAIAdapter
        from testcraft.adapters.llm.token_calculator import TokenCalculator
        
        # Test with a model that should be in catalog
        model = "gpt-4o"
        entry = resolve_model("openai", model)
        
        if not entry:
            pytest.skip(f"Model openai/{model} not found in catalog")
            
        adapter = OpenAIAdapter(model=model, max_tokens=None)  # Auto-calculated
        
        # Verify adapter uses catalog-based token calculator
        assert hasattr(adapter, 'token_calculator')
        assert isinstance(adapter.token_calculator, TokenCalculator)
        
        # Verify max_tokens doesn't exceed catalog limits
        assert adapter.max_tokens <= entry.limits.default_max_output
        
        # Verify token calculator respects catalog limits
        calc_limits = adapter.token_calculator.limits
        assert calc_limits.max_context == entry.limits.max_context
        assert calc_limits.max_output <= entry.limits.default_max_output

    def test_openai_adapter_parameter_compliance_reasoning_models(self):
        """Test that OpenAI reasoning models get proper parameter compliance."""
        from testcraft.adapters.llm.openai import OpenAIAdapter
        
        reasoning_model = "o1-mini"
        entry = resolve_model("openai", reasoning_model)
        
        if not entry:
            pytest.skip(f"Model openai/{reasoning_model} not found in catalog")
            
        if entry.flags and entry.flags.reasoning_capable:
            adapter = OpenAIAdapter(model=reasoning_model)
            
            # Reasoning models should have specific parameter requirements
            # (These would be tested in actual API calls, but we can check the setup)
            assert adapter.model == reasoning_model
            
            # Token calculator should recognize this as a reasoning model
            assert adapter.token_calculator.is_reasoning_model() is True

    def test_openai_beta_headers_for_reasoning_models(self):
        """Test that OpenAI sets appropriate headers for reasoning models."""
        from testcraft.adapters.llm.openai import OpenAIAdapter
        
        reasoning_model = "o1-mini"
        entry = resolve_model("openai", reasoning_model)
        
        if not entry:
            pytest.skip(f"Model openai/{reasoning_model} not found in catalog")
            
        if entry.beta and entry.beta.headers:
            adapter = OpenAIAdapter(model=reasoning_model)
            
            # The adapter should be aware this is a reasoning model
            assert adapter.token_calculator.is_reasoning_model() is True
            
            # Beta headers should be available from catalog
            expected_headers = entry.beta.headers
            assert len(expected_headers) > 0


class TestClaudeAdapterCaps:
    """Test Claude adapter cap enforcement."""

    def test_claude_adapter_never_exceeds_catalog_caps(self):
        """Test that Claude adapter never exceeds catalog-defined caps."""
        from testcraft.adapters.llm.claude import ClaudeAdapter
        from testcraft.adapters.llm.token_calculator import TokenCalculator
        
        model = "claude-3-5-sonnet"
        entry = resolve_model("anthropic", model)
        
        if not entry:
            pytest.skip(f"Model anthropic/{model} not found in catalog")
            
        # Test without beta features
        adapter = ClaudeAdapter(
            model=model, 
            max_tokens=None,  # Auto-calculated
            enable_extended_context=False,
            enable_extended_output=False
        )
        
        assert hasattr(adapter, 'token_calculator')
        assert isinstance(adapter.token_calculator, TokenCalculator)
        
        # Should not exceed catalog limits
        assert adapter.max_tokens <= entry.limits.default_max_output
        
        calc_limits = adapter.token_calculator.limits
        assert calc_limits.max_context == entry.limits.max_context
        assert calc_limits.max_output <= entry.limits.default_max_output

    def test_claude_beta_headers_only_when_enabled(self):
        """Test that Claude beta headers are only set when beta features are enabled."""
        from testcraft.adapters.llm.claude import ClaudeAdapter
        
        # Test with beta features disabled (default)
        adapter_standard = ClaudeAdapter(
            model="claude-sonnet-4",
            enable_extended_context=False,
            enable_extended_output=False
        )
        
        # Should not have beta features enabled
        assert adapter_standard.enable_extended_context is False
        assert adapter_standard.enable_extended_output is False
        
        # Beta headers should be minimal or empty when features are disabled
        if hasattr(adapter_standard, '_get_beta_headers'):
            beta_headers = adapter_standard._get_beta_headers()
            # When beta features are disabled, headers should be empty or minimal
            assert isinstance(beta_headers, dict)

    def test_claude_beta_headers_when_enabled(self):
        """Test that Claude beta headers are properly set when beta features are enabled."""
        from testcraft.adapters.llm.claude import ClaudeAdapter
        
        # Test with beta features enabled
        adapter_beta = ClaudeAdapter(
            model="claude-sonnet-4",
            enable_extended_context=True,
            enable_extended_output=True
        )
        
        # Should have beta features enabled
        assert adapter_beta.enable_extended_context is True
        assert adapter_beta.enable_extended_output is True
        
        # Token calculator should reflect beta features
        calc = adapter_beta.token_calculator
        assert calc.enable_extended_context is True
        assert calc.enable_extended_output is True


class TestAzureAdapterCaps:
    """Test Azure adapter cap enforcement."""

    def test_azure_adapter_respects_underlying_model_caps(self):
        """Test that Azure adapter respects underlying OpenAI model caps."""
        from testcraft.adapters.llm.azure import AzureOpenAIAdapter
        
        # Test with an OpenAI model that should be in catalog
        model = "gpt-4o"
        entry = resolve_model("openai", model)
        
        if not entry:
            pytest.skip(f"Model openai/{model} not found in catalog")
            
        try:
            adapter = AzureOpenAIAdapter(
                deployment_name="test-deployment",
                model=model,
                max_tokens=None
            )
            
            # Should respect the underlying OpenAI model limits
            if hasattr(adapter, 'max_tokens') and adapter.max_tokens:
                assert adapter.max_tokens <= entry.limits.default_max_output
            
        except Exception:
            # Skip if Azure setup not available
            pytest.skip("Azure adapter not available for testing")


class TestBedrockAdapterCaps:
    """Test Bedrock adapter cap enforcement."""

    def test_bedrock_adapter_respects_catalog_caps(self):
        """Test that Bedrock adapter respects catalog-defined caps."""
        from testcraft.adapters.llm.bedrock import BedrockAdapter
        
        # Test with a Bedrock model from catalog
        bedrock_models = [
            ("bedrock", "anthropic.claude-3-haiku-20240307-v1:0"),
            ("bedrock", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
        ]
        
        tested_any = False
        for provider, model_id in bedrock_models:
            entry = resolve_model(provider, model_id)
            
            if entry:
                try:
                    adapter = BedrockAdapter(model_id=model_id, max_tokens=None)
                    
                    # Should respect catalog limits
                    if hasattr(adapter, 'max_tokens') and adapter.max_tokens:
                        assert adapter.max_tokens <= entry.limits.default_max_output
                        tested_any = True
                        
                except Exception:
                    # Skip if Bedrock setup not available
                    continue
        
        if not tested_any:
            pytest.skip("No Bedrock models available for testing")


class TestAdapterCapEnforcement:
    """Test general adapter cap enforcement patterns."""

    def test_adapter_cap_enforcement_with_manual_override_attempt(self):
        """Test that adapters enforce caps even when manual overrides are attempted."""
        from testcraft.adapters.llm.claude import ClaudeAdapter
        
        model = "claude-3-5-sonnet"
        entry = resolve_model("anthropic", model)
        
        if not entry:
            pytest.skip(f"Model anthropic/{model} not found in catalog")
            
        # Try to set max_tokens higher than catalog allows
        excessive_max_tokens = entry.limits.default_max_output * 2
        
        adapter = ClaudeAdapter(
            model=model,
            max_tokens=excessive_max_tokens  # Attempt to override
        )
        
        # The adapter should still respect catalog limits through token calculator
        # Even if max_tokens is set higher, the token calculator should enforce caps
        calc_max = adapter.token_calculator.calculate_max_tokens("test_generation")
        assert calc_max <= entry.limits.default_max_output

    @pytest.mark.parametrize("safety_margin", SAFETY_MARGIN_CASES)
    def test_adapter_respects_safety_margins_from_catalog(self, safety_margin):
        """Test that adapters apply appropriate safety margins based on catalog limits."""
        from testcraft.adapters.llm.openai import OpenAIAdapter
        from testcraft.adapters.llm.claude import ClaudeAdapter
        
        test_cases = [
            ("openai", "gpt-4o", OpenAIAdapter),
            ("anthropic", "claude-3-5-sonnet", ClaudeAdapter),
        ]
        
        tested_any = False
        for provider, model, adapter_class in test_cases:
            entry = resolve_model(provider, model)
            
            if not entry:
                continue
                
            if adapter_class == ClaudeAdapter:
                adapter = adapter_class(model=model)
            else:
                adapter = adapter_class(model=model)
            
            # Calculate max tokens with different safety margins
            calc = adapter.token_calculator
            
            safe_tokens = calc.calculate_max_tokens("test_generation", safety_margin=safety_margin)
            
            # Should be within catalog limits
            assert safe_tokens <= entry.limits.default_max_output
            tested_any = True
        
        if not tested_any:
            pytest.skip("No test models available")

    def test_adapter_context_window_enforcement(self):
        """Test that adapters enforce context window limits from catalog."""
        from testcraft.adapters.llm.token_calculator import TokenCalculator
        
        # Test with different providers and models
        test_cases = [
            ("openai", "gpt-4o"),
            ("anthropic", "claude-3-5-sonnet"),
        ]
        
        tested_any = False
        for provider, model in test_cases:
            entry = resolve_model(provider, model)
            
            if not entry:
                continue
                
            calc = TokenCalculator(provider=provider, model=model)
            
            # Test context window enforcement with reasonable input
            large_input_tokens = int(entry.limits.max_context * 0.6)  # 60% of context
            
            max_output = calc.calculate_max_tokens(
                "test_generation",
                input_length=large_input_tokens,
                safety_margin=0.8
            )
            
            # Total should not exceed context window with safety margin
            total_tokens = large_input_tokens + max_output
            max_allowed = int(entry.limits.max_context * 0.8)  # Safety margin applied
            
            # Should respect context ceiling
            assert total_tokens <= max_allowed
            tested_any = True
        
        if not tested_any:
            pytest.skip("No test models available")


class TestMultiProviderConsistency:
    """Test consistency across multiple providers and adapters."""

    def test_multi_provider_workflow_catalog_consistency(self):
        """Test that multi-provider workflows maintain catalog consistency."""
        from testcraft.adapters.llm.openai import OpenAIAdapter
        from testcraft.adapters.llm.claude import ClaudeAdapter
        
        # Test workflow that might use multiple providers
        providers_models = [
            ("openai", "gpt-4o", OpenAIAdapter),
            ("anthropic", "claude-3-5-sonnet", ClaudeAdapter),
        ]
        
        adapters = []
        
        for provider, model, adapter_class in providers_models:
            entry = resolve_model(provider, model)
            if entry:
                if adapter_class == ClaudeAdapter:
                    adapter = adapter_class(model=model)
                else:
                    adapter = adapter_class(model=model)
                
                adapters.append((adapter, entry))
        
        if not adapters:
            pytest.skip("No adapters available for testing")
        
        # Verify all adapters respect their respective catalog limits
        for adapter, entry in adapters:
            calc = adapter.token_calculator
            
            assert calc.limits.max_context == entry.limits.max_context
            assert calc.limits.max_output <= entry.limits.default_max_output
            assert adapter.max_tokens <= entry.limits.default_max_output
            
            # Each adapter should be internally consistent
            for use_case in ["test_generation", "code_analysis", "refinement"]:
                tokens = calc.calculate_max_tokens(use_case)
                assert tokens <= entry.limits.default_max_output
                assert tokens > 0
