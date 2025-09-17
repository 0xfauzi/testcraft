"""
Pytest refiner service for test execution and refinement.

DEPRECATED: This module has been refactored into the refinement/ package.
This file now serves as a compatibility shim that re-exports PytestRefiner
from the new modular structure.

For new code, import from: testcraft.application.generation.services.refinement
"""

# Re-export PytestRefiner from the new modular structure
from .refinement import PytestRefiner

# Re-export the static method for backward compatibility
extract_import_path_from_failure = PytestRefiner.extract_import_path_from_failure

__all__ = ["PytestRefiner", "extract_import_path_from_failure"]
