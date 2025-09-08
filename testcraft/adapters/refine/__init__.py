"""
Refine adapters for test refinement operations.

This module contains adapters for refining existing tests based on various inputs,
including pytest failures, code coverage gaps, and quality analysis.
"""

from .main_adapter import RefineAdapter

__all__ = ["RefineAdapter"]
