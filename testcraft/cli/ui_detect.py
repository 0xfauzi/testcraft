"""UI style detection utilities."""

import os
import sys

from ..adapters.io.ui_rich import UIStyle


def detect_ui_style(ui_flag: str | None) -> UIStyle:
    """Detect appropriate UI style based on flag, environment, and TTY status."""
    # Priority 1: Explicit --ui flag
    if ui_flag:
        if ui_flag.lower() == "minimal":
            return UIStyle.MINIMAL
        elif ui_flag.lower() == "classic":
            return UIStyle.CLASSIC
    
    # Priority 2: Environment variable
    env_ui = os.getenv("TESTCRAFT_UI")
    if env_ui:
        if env_ui.lower() == "minimal":
            return UIStyle.MINIMAL
        elif env_ui.lower() == "classic":
            return UIStyle.CLASSIC
    
    # Priority 3: Auto-detect based on environment
    if os.getenv("CI") == "true" or not sys.stdout.isatty():
        return UIStyle.MINIMAL
    
    # Default to classic for interactive terminals
    return UIStyle.CLASSIC
