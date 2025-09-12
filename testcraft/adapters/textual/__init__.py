"""
Textual UI adapter for TestCraft.

Provides a modern terminal user interface using the Textual framework.
Includes both standalone TUI applications and UI adapters that implement
the UIPort interface.
"""

from .app import TestCraftTextualApp
from .ui_textual import TextualUIAdapter

__all__ = ["TestCraftTextualApp", "TextualUIAdapter"]
