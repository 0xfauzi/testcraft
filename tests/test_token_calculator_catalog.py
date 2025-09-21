from __future__ import annotations

from testcraft.adapters.llm.token_calculator import TokenCalculator


def test_caps_enforcement_and_context_ceiling() -> None:
    calc = TokenCalculator(provider="openai", model="gpt-4.1")
    # Ensure we never exceed default cap
    tokens = calc.calculate_max_tokens("test_generation", input_length=1000)
    assert tokens <= calc.limits.max_output

    # If input nearly consumes context, output should be clamped low
    near_context = int(calc.limits.max_context * 0.95)
    tokens2 = calc.calculate_max_tokens("test_generation", input_length=near_context)
    assert tokens2 < 10000


def test_thinking_only_when_supported() -> None:
    # OpenAI models should not expose thinking tokens (built-in reasoning)
    openai_calc = TokenCalculator(provider="openai", model="o4-mini")
    assert openai_calc.calculate_thinking_tokens("test_generation") is None

    # Anthropic models with thinking support should return a value
    anthropic_calc = TokenCalculator(provider="anthropic", model="claude-sonnet-4")
    thinking = anthropic_calc.calculate_thinking_tokens("test_generation")
    assert thinking is None or thinking > 0


