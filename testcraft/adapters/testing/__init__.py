"""
Testing adapters for test discovery and classification.

This module contains adapters for various test discovery mechanisms including
pytest collection, AST-based classification, and coverage-based probing.
"""

from .pytest_collector import PytestCollectionAdapter

__all__ = ["PytestCollectionAdapter"]
