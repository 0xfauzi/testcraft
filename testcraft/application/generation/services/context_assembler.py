"""Backward-compatibility shim for context assembly.

This module preserves the legacy import path while delegating to the new
modular implementation under `services/context`.
"""

from __future__ import annotations

# Re-export the facade and logger so all existing imports continue to work
from .context import ContextAssembler  # noqa: F401
from .context.assembler import logger  # noqa: F401

__all__ = ["ContextAssembler", "logger"]



