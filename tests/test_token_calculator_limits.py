from __future__ import annotations

import pytest

from testcraft.adapters.llm.token_calculator import TokenCalculator
from testcraft.config.model_catalog_loader import load_catalog


def test_catalog_loads_once_and_contains_models():
    catalog = load_catalog()
    assert catalog.models, "Catalog should contain model entries"
    # Ensure basic providers exist from the provided TOML fixture
    providers = {m.provider for m in catalog.models}
    assert {"openai", "anthropic"}.issubset(providers)


@pytest.mark.parametrize(
    "provider,model,use_case",
    [
        ("openai", "gpt-4.1", "test_generation"),
        ("openai", "o4-mini", "code_analysis"),
        ("anthropic", "claude-3-7-sonnet", "refinement"),
    ],
)
def test_max_tokens_do_not_exceed_limits(provider: str, model: str, use_case: str):
    calc = TokenCalculator(provider=provider, model=model)
    max_tokens = calc.calculate_max_tokens(use_case=use_case)
    assert max_tokens > 0
    # Safety margin is applied internally; ensure not above model max_output
    assert max_tokens <= calc.limits.max_output


def test_context_ceiling_enforced():
    # Pick a model with known context from the catalog
    calc = TokenCalculator(provider="openai", model="gpt-4.1")
    # Force input length near safe context to trigger ceiling logic
    safe_context = int(calc.limits.max_context * 0.8)
    input_len = safe_context - 100
    out = calc.calculate_max_tokens(use_case="test_generation", input_length=input_len)
    assert out <= 100  # leaves <=100 to stay under safe context


def test_unknown_model_safe_defaults():
    # Unknown pair should not raise and should return conservative limits
    calc = TokenCalculator(provider="openai", model="unknown-model-xyz")
    assert calc.limits.max_context == 200000
    assert calc.limits.max_output == 32000
    assert calc.calculate_max_tokens(use_case="code_analysis") > 0


def test_thinking_tokens_for_anthropic_present():
    calc = TokenCalculator(provider="anthropic", model="claude-3-7-sonnet")
    thinking = calc.calculate_thinking_tokens(use_case="code_analysis", complexity_level="complex")
    assert thinking is None or thinking > 0


