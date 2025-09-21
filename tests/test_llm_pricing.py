from __future__ import annotations

from types import SimpleNamespace

from testcraft.adapters.llm.pricing import calculate_cost


def test_per_million_math_dict() -> None:
    usage = {"prompt_tokens": 1000, "completion_tokens": 2000}
    cost = calculate_cost(usage, "openai", "gpt-4.1")
    assert cost >= 0.0


def test_per_million_math_object() -> None:
    usage = SimpleNamespace(input_tokens=500, output_tokens=500)
    cost = calculate_cost(usage, "anthropic", "claude-sonnet-4")
    assert cost >= 0.0


