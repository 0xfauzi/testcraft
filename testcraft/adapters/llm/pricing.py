"""Centralized pricing utilities based on the Model Catalog.

- get_pricing(provider, model) → per-million pricing
- calculate_cost(usage, provider, model) → USD cost
"""

from __future__ import annotations

from typing import Any

from ...config.model_catalog import (
    get_pricing as catalog_get_pricing,
)
from ...config.model_catalog import (
    normalize_model_id as catalog_normalize,
)


def get_pricing(provider: str, model: str) -> dict[str, float]:
    canonical_provider, canonical_model = catalog_normalize(provider, model)
    per_m = catalog_get_pricing(canonical_provider, canonical_model)
    return {
        "input_per_million": float(per_m.input),
        "output_per_million": float(per_m.output),
    }


def _extract_token_counts(usage: Any) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) across SDK variants/dicts."""
    prompt_tokens = None
    completion_tokens = None

    # Object-style
    try:
        if hasattr(usage, "prompt_tokens"):
            prompt_tokens = usage.prompt_tokens
        elif hasattr(usage, "input_tokens"):
            prompt_tokens = usage.input_tokens

        if hasattr(usage, "completion_tokens"):
            completion_tokens = usage.completion_tokens
        elif hasattr(usage, "output_tokens"):
            completion_tokens = usage.output_tokens
    except Exception:
        prompt_tokens = None
        completion_tokens = None

    # Dict-style fallbacks
    if prompt_tokens is None and isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    if completion_tokens is None and isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")

    return int(prompt_tokens or 0), int(completion_tokens or 0)


def calculate_cost(usage: Any, provider: str, model: str) -> float:
    """Compute cost using per-million pricing and token counts.

    total = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
    """
    prices = get_pricing(provider, model)
    prompt_tokens, completion_tokens = _extract_token_counts(usage)
    prompt_cost = (prompt_tokens * prices["input_per_million"]) / 1_000_000.0
    completion_cost = (completion_tokens * prices["output_per_million"]) / 1_000_000.0
    return float(prompt_cost + completion_cost)
