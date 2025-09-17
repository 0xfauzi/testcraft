"""
Pytest Collection Adapter - Tier 1 ground truth test discovery.

Uses pytest's collection mechanism to discover tests, providing the most accurate
and configuration-respecting test discovery. This is the preferred method when
pytest is available and the environment is properly configured.
"""

import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from ...config.models import TestDiscoveryConfig

logger = logging.getLogger(__name__)


class PytestCollectionError(Exception):
    """Exception raised when pytest collection fails."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None, timeout: bool = False):
        super().__init__(message)
        self.cause = cause
        self.timeout = timeout


class PytestCollectionResult:
    """Result of pytest collection operation."""
    
    def __init__(
        self,
        files: set[Path],
        nodes: list[str],
        by_file: Optional[dict[Path, list[str]]] = None,
        duration_ms: float = 0.0,
        success: bool = True,
        failure_reason: Optional[str] = None
    ):
        self.files = files
        self.nodes = nodes
        self.by_file = by_file or {}
        self.duration_ms = duration_ms
        self.success = success
        self.failure_reason = failure_reason
        
    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "files": [str(f) for f in self.files],
            "nodes": self.nodes,
            "by_file": {str(k): v for k, v in self.by_file.items()},
            "duration_ms": self.duration_ms,
            "success": self.success,
            "failure_reason": self.failure_reason,
        }


class PytestCollectionAdapter:
    """
    Adapter for discovering tests using pytest's collection mechanism.
    
    This provides the most accurate test discovery as it respects pytest configuration
    files (pytest.ini, pyproject.toml, etc.) and handles complex test discovery patterns.
    """
    
    def __init__(self, config: Optional[TestDiscoveryConfig] = None):
        """
        Initialize the pytest collection adapter.
        
        Args:
            config: Test discovery configuration
        """
        self.config = config or TestDiscoveryConfig()
        self._cache: dict[str, tuple[PytestCollectionResult, float]] = {}
        
    def collect(
        self,
        project_root: Path,
        extra_args: Optional[list[str]] = None,
        timeout_sec: Optional[int] = None
    ) -> PytestCollectionResult:
        """
        Collect tests using pytest's collection mechanism.
        
        Args:
            project_root: Root directory of the project
            extra_args: Additional arguments to pass to pytest
            timeout_sec: Timeout in seconds (overrides config default)
            
        Returns:
            PytestCollectionResult with discovered tests
            
        Raises:
            PytestCollectionError: If collection fails or times out
        """
        start_time = time.time()
        timeout = timeout_sec or self.config.collector_timeout_sec
        
        # Check cache first if enabled
        if self.config.cache_ttl_sec > 0:
            cache_key = self._generate_cache_key(project_root, extra_args)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.debug(f"Using cached pytest collection result for {project_root}")
                return cached_result
        
        try:
            # Build pytest command
            cmd = self._build_pytest_command(project_root, extra_args)
            
            logger.debug(f"Running pytest collection: {' '.join(cmd)}")
            
            # Execute pytest collection with timeout
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._get_clean_env()
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Parse collection output
            collection_result = self._parse_collection_output(
                result.stdout,
                result.stderr,
                result.returncode,
                duration_ms,
                project_root
            )
            
            # Cache successful results
            if collection_result.success and self.config.cache_ttl_sec > 0:
                cache_key = self._generate_cache_key(project_root, extra_args)
                self._cache_result(cache_key, collection_result)
            
            return collection_result
            
        except subprocess.TimeoutExpired as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(f"Pytest collection timed out after {timeout}s for {project_root}")
            return PytestCollectionResult(
                files=set(),
                nodes=[],
                duration_ms=duration_ms,
                success=False,
                failure_reason=f"Timeout after {timeout}s"
            )
            
        except subprocess.CalledProcessError as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(f"Pytest collection failed for {project_root}: {e}")
            return PytestCollectionResult(
                files=set(),
                nodes=[],
                duration_ms=duration_ms,
                success=False,
                failure_reason=f"Process failed with code {e.returncode}"
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(f"Unexpected error during pytest collection for {project_root}: {e}")
            return PytestCollectionResult(
                files=set(),
                nodes=[],
                duration_ms=duration_ms,
                success=False,
                failure_reason=f"Unexpected error: {str(e)}"
            )
    
    def _build_pytest_command(self, project_root: Path, extra_args: Optional[list[str]]) -> list[str]:
        """Build the pytest command with appropriate arguments."""
        cmd = [sys.executable, "-m", "pytest"]
        
        # Core collection arguments
        cmd.extend(["--collect-only", "-q", "--tb=no"])
        
        # Add extra arguments if provided
        if extra_args:
            cmd.extend(extra_args)
        
        return cmd
    
    def _get_clean_env(self) -> dict[str, str]:
        """Get a clean environment for pytest execution."""
        import os
        
        # Start with minimal environment
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "USER": os.environ.get("USER", ""),
        }
        
        # Add virtual environment variables if present
        for venv_var in ["VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "PIPENV_ACTIVE"]:
            if venv_var in os.environ:
                env[venv_var] = os.environ[venv_var]
        
        return env
    
    def _parse_collection_output(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        duration_ms: float,
        project_root: Path | None = None
    ) -> PytestCollectionResult:
        """Parse pytest collection output to extract test information."""
        files = set()
        nodes = []
        by_file = {}
        
        if returncode != 0:
            # Collection failed
            failure_reason = f"pytest exited with code {returncode}"
            if stderr:
                failure_reason += f": {stderr[:500]}"  # Truncate long error messages
            
            return PytestCollectionResult(
                files=files,
                nodes=nodes,
                by_file=by_file,
                duration_ms=duration_ms,
                success=False,
                failure_reason=failure_reason
            )
        
        # Parse stdout to extract test nodes
        for line in stdout.splitlines():
            line = line.strip()
            
            # Skip empty lines and pytest output headers
            if not line or line.startswith("=") or line.startswith("collecting..."):
                continue
            
            # Skip pytest warnings and info messages
            if any(marker in line.lower() for marker in ["warning", "deprecated", "error"]):
                continue
            
            # Look for test nodeids (format: path/file.py::TestClass::test_method)
            if "::" in line:
                # This is a test nodeid
                nodes.append(line)
                
                # Extract file path from nodeid
                file_part = line.split("::")[0]
                try:
                    file_path = Path(file_part)
                    if file_path.suffix == ".py":
                        # Convert relative paths to absolute paths using project_root
                        if not file_path.is_absolute() and project_root:
                            file_path = (project_root / file_path).resolve()
                        files.add(file_path)
                        
                        # Group nodes by file
                        if file_path not in by_file:
                            by_file[file_path] = []
                        by_file[file_path].append(line)
                        
                except Exception as e:
                    logger.debug(f"Failed to parse file path from nodeid '{line}': {e}")
                    continue
            
            # Also check for simple file listings (some pytest versions output differently)
            elif line.endswith(".py") and not line.startswith("-"):
                try:
                    file_path = Path(line)
                    if file_path.suffix == ".py":
                        # Convert relative paths to absolute paths using project_root
                        if not file_path.is_absolute() and project_root:
                            file_path = (project_root / file_path).resolve()
                        files.add(file_path)
                except Exception:
                    continue
        
        logger.debug(f"Parsed {len(files)} test files and {len(nodes)} test nodes")
        
        return PytestCollectionResult(
            files=files,
            nodes=nodes,
            by_file=by_file,
            duration_ms=duration_ms,
            success=True
        )
    
    def _generate_cache_key(self, project_root: Path, extra_args: Optional[list[str]]) -> str:
        """Generate cache key based on project state."""
        key_data = {
            "project_root": str(project_root),
            "extra_args": extra_args or [],
        }
        
        # Include relevant config file hashes
        config_files = [
            project_root / "pytest.ini",
            project_root / "pyproject.toml",
            project_root / "setup.cfg",
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    content = config_file.read_text()
                    key_data[f"config_{config_file.name}"] = hashlib.md5(content.encode()).hexdigest()
                except Exception:
                    continue
        
        # Include requirements hash if available
        requirements_files = [
            project_root / "requirements.txt",
            project_root / "pyproject.toml",  # For dependency info
        ]
        
        for req_file in requirements_files:
            if req_file.exists():
                try:
                    content = req_file.read_text()
                    key_data[f"deps_{req_file.name}"] = hashlib.md5(content.encode()).hexdigest()
                    break  # Only need one requirements reference
                except Exception:
                    continue
        
        # Create cache key hash
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _get_cached_result(self, cache_key: str) -> Optional[PytestCollectionResult]:
        """Get cached result if still valid."""
        if cache_key not in self._cache:
            return None
        
        result, timestamp = self._cache[cache_key]
        
        # Check if cache entry is still valid
        if time.time() - timestamp > self.config.cache_ttl_sec:
            del self._cache[cache_key]
            return None
        
        return result
    
    def _cache_result(self, cache_key: str, result: PytestCollectionResult) -> None:
        """Cache a collection result."""
        self._cache[cache_key] = (result, time.time())
        
        # Simple cache size management
        if len(self._cache) > 100:  # Keep cache reasonably sized
            # Remove oldest entries
            sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_items[:20]:  # Remove 20 oldest
                del self._cache[key]
    
    def clear_cache(self) -> None:
        """Clear the collection result cache."""
        self._cache.clear()
        logger.debug("Pytest collection cache cleared")
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_enabled": self.config.cache_ttl_sec > 0,
            "cache_ttl_sec": self.config.cache_ttl_sec,
        }
