"""
Adapters for the testcraft system.

This module contains all adapter implementations that provide concrete
implementations of the port interfaces defined in the ports module.
"""

from . import context, io, llm, parsing, refine, telemetry

# Note: coverage module temporarily excluded from __init__.py imports
# Individual coverage adapters can still be imported directly when needed

__all__ = [
    "context",
    "io",
    "llm",
    "parsing",
    "refine",
    "telemetry",
]
