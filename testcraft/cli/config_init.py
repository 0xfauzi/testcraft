"""Configuration initialization with support for multiple formats.

This module provides backward compatibility by re-exporting from the refactored modules.
The original large config_init.py has been split into a modular structure:

- config_init.initializer: Main orchestration logic
- config_init.preferences: User preference collection  
- config_init.io: File I/O operations
- config_init.generators: Format-specific generators (TOML, YAML, JSON)

For legacy compatibility, the main classes are re-exported here.
"""

# Re-export from the new modular structure
from .config_init import ConfigInitializationError, ConfigInitializer

__all__ = [
    "ConfigInitializer", 
    "ConfigInitializationError",
]
