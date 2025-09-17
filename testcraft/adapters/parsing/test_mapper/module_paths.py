"""
Module path derivation utilities for test mapping.

This module provides wrapper utilities around ModulePathDeriver, maintaining
the lazy import and caching pattern to avoid circular dependencies.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModulePathHelper:
    """Helper class for deriving module paths with lazy initialization."""
    
    def __init__(self):
        """Initialize with lazy loading of ModulePathDeriver to avoid circular imports."""
        self._module_path_deriver = None
    
    def derive_source_module_paths(self, source_path: Path, project_root: Optional[Path] = None) -> list[str]:
        """Derive possible module import paths for a source file."""
        try:
            if project_root is None:
                project_root = source_path.parent
            
            # Initialize ModulePathDeriver lazily to avoid circular imports
            if self._module_path_deriver is None:
                from ....application.generation.services.structure import ModulePathDeriver
                self._module_path_deriver = ModulePathDeriver()
            
            # Use ModulePathDeriver for authoritative module path resolution
            # Note: derive_module_path returns a dictionary, not a string
            result = self._module_path_deriver.derive_module_path(source_path, project_root)
            
            # Extract the actual module path from the result dictionary
            module_path = result.get("module_path", "") if isinstance(result, dict) else str(result)
            fallback_paths = result.get("fallback_paths", []) if isinstance(result, dict) else []
            
            paths = []
            
            # Add the primary module path if valid
            if module_path and module_path.strip():
                paths.append(module_path)
                
                # Add parent module paths (e.g., from "a.b.c" -> ["a.b", "a"])
                parts = module_path.split('.')
                for i in range(len(parts) - 1, 0, -1):
                    parent_path = '.'.join(parts[:i])
                    if parent_path not in paths:
                        paths.append(parent_path)
            
            # Add fallback paths if available
            for fallback in fallback_paths:
                if fallback and fallback not in paths:
                    paths.append(fallback)
            
            # Add simple filename without extension as final fallback
            simple_name = source_path.stem
            if simple_name not in paths:
                paths.append(simple_name)
                
            return paths
            
        except Exception as e:
            logger.debug(f"Failed to derive module paths for {source_path}: {e}")
            # Fallback to simple name-based paths
            return [source_path.stem]
    
    def derive_source_module_paths_legacy(self, source_path: Path, project_root: Optional[Path]) -> list[str]:
        """Legacy version of module path derivation (kept for backward compatibility)."""
        try:
            if project_root:
                if self._module_path_deriver is None:
                    from ....application.generation.services.structure import ModulePathDeriver
                    self._module_path_deriver = ModulePathDeriver()
                
                self._module_path_deriver.set_project_root(project_root)
            
            # Get primary and fallback module paths
            result = self._module_path_deriver.derive_module_path(source_path)
            
            module_paths = []
            if result.get('primary_path'):
                module_paths.append(result['primary_path'])
            
            if result.get('fallback_paths'):
                module_paths.extend(result['fallback_paths'])
            
            return module_paths
            
        except Exception as e:
            # Fallback to simple path-based derivation
            try:
                if project_root and source_path.is_relative_to(project_root):
                    rel_path = source_path.relative_to(project_root)
                    module_path = str(rel_path.with_suffix('')).replace('/', '.')
                    return [module_path]
            except Exception:
                pass
            
            return []
