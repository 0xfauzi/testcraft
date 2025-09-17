"""
Context services package.

Exposes the facade `ContextAssembler` while organizing implementation details
into cohesive submodules to keep files small and responsibilities clear.
"""

from .assembler import ContextAssembler  # re-export facade

__all__ = ["ContextAssembler"]



