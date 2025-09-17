"""Pydantic models for TestCraft configuration.

DEPRECATED: This module has been split into a package for better organization.
All models are re-exported from testcraft.config.models package for backward compatibility.
"""

# Re-export all models from the new package structure
from .models import *  # noqa: F403, F401

# Preserve the original module docstring note for reference
"""
Note: security, prompt_engineering, and context settings were removed
because no runtime logic consumes them yet. Reintroduce only when wired.
"""