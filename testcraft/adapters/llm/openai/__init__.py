"""OpenAI adapter package."""

import logging
from pathlib import Path

from .adapter import OpenAIAdapter
from ....config.model_catalog_loader import resolve_model

# Export logger for test mocking
logger = logging.getLogger(__name__)

# Export Path for test mocking
Path = Path

__all__ = ["OpenAIAdapter", "logger", "Path", "resolve_model"]
