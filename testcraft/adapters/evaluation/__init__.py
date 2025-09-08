"""
Evaluation adapters for test quality assessment and A/B testing.

This package provides adapters for evaluating generated tests using
automated acceptance checks, LLM-as-judge evaluation, and statistical
analysis for prompt optimization.
"""

from .main_adapter import TestcraftEvaluationAdapter

__all__ = ["TestcraftEvaluationAdapter"]
