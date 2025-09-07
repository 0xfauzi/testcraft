from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RoutingDecision:
    provider: Literal["openai", "claude", "azure", "bedrock"]
    model: str


def score_complexity(code: str) -> int:
    """Very simple complexity proxy: length + function count."""
    functions = code.count("def ")
    return len(code) // 400 + functions


def route_model(code: str) -> RoutingDecision:
    score = score_complexity(code)
    if score < 2:
        return RoutingDecision(provider="openai", model="gpt-4o-mini")
    if score < 5:
        return RoutingDecision(provider="claude", model="claude-3-haiku")
    if score < 8:
        return RoutingDecision(provider="azure", model="gpt-4o-mini")
    return RoutingDecision(provider="bedrock", model="anthropic.claude-3-sonnet")


