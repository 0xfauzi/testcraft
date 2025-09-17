"""Shared CLI context for Click commands."""

from typing import Any

from ..adapters.io.enhanced_ui import EnhancedUIAdapter
from ..adapters.io.rich_cli import RichCliComponents
from ..adapters.io.ui_rich import UIStyle
from ..config.models import TestCraftConfig


class ClickContext:
    """Context object for Click commands."""

    def __init__(self):
        self.config: TestCraftConfig | None = None
        self.container: dict[str, Any] | None = None
        self.ui: EnhancedUIAdapter | None = None  # Will be initialized in app()
        self.rich_cli: RichCliComponents | None = None  # Will be initialized in app()
        self.ui_style: UIStyle = UIStyle.CLASSIC  # Will be set in app()
        self.verbose: bool = False
        self.quiet: bool = False
        self.ui_flag_explicit: bool = False
        self.dry_run: bool = False
