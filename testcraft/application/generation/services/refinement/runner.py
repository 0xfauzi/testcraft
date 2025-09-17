"""
Pytest runner and result classification.

Handles pytest execution, result classification, and XFAIL detection.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .....config.models import RefineConfig

logger = logging.getLogger(__name__)


class PytestRunner:
    """Handles pytest execution and result classification."""

    def __init__(self, executor: ThreadPoolExecutor, config: RefineConfig):
        """
        Initialize the pytest runner.

        Args:
            executor: Thread pool executor for async operations
            config: Refinement configuration for pytest args and settings
        """
        self._executor = executor
        self._config = config
        
        # Get configurable pytest args with defaults; honor first-failure behavior
        self._pytest_args = list(self._config.pytest_args_for_refinement)
        # If configured to stop on first failure within a file, add -x if not already present
        try:
            if getattr(self._config, "refine_on_first_failure_only", True):
                if "-x" not in self._pytest_args and "--maxfail=1" not in self._pytest_args:
                    self._pytest_args.append("-x")
        except Exception:
            # Safe default already includes -x in defaults; ignore if not available
            pass

    async def run_pytest(self, test_path: str) -> dict[str, Any]:
        """
        Run pytest on a specific test file and return results.

        Uses the async_runner abstraction which wraps subprocess patterns
        for async workflows.

        Args:
            test_path: Path to the test file to run

        Returns:
            Dictionary with pytest execution results including stdout, stderr, returncode
        """
        from .....adapters.io.async_runner import run_python_module_async_with_executor

        try:
            # Use configurable pytest arguments for refinement
            pytest_args = [str(test_path)] + list(self._pytest_args)

            # Use the reusable async subprocess abstraction with executor
            stdout, stderr, returncode = await run_python_module_async_with_executor(
                executor=self._executor,
                module_name="pytest",
                args=pytest_args,
                timeout=60,
                raise_on_error=False,  # Handle failures ourselves like refine adapter
            )

            result = {
                "stdout": stdout or "",
                "stderr": stderr or "",
                "returncode": returncode,
                "command": f"python -m pytest {' '.join(pytest_args)}",
            }

            # Add classification flags to help gating refinement
            combined = (stdout or "") + "\n" + (stderr or "")
            result.update(self.classify_pytest_result(combined, returncode))
            return result

        except Exception as e:
            logger.warning("Failed to run pytest for %s: %s", test_path, e)
            result = {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "command": f"python -m pytest {test_path}",
                "error": str(e),
            }
            result.update(self.classify_pytest_result(result["stderr"], -1))
            return result

    def classify_pytest_result(self, output: str, returncode: int) -> dict[str, Any]:
        """
        Classify pytest outcome to decide if LLM refinement should proceed.

        We skip refinement and do NOT send failures to the LLM when pytest
        didn't actually run tests (collection/usage/internal errors), because
        the LLM cannot fix environment or import problems reliably.
        """
        text = (output or "").lower()
        unrefinable = False
        category = "unknown"

        indicators = {
            "collection_error": ["collected 0 items", "collection error", "errors during collection"],
            "import_error": ["importerror", "module not found", "no module named"],
            "usage_error": ["usage:", "error: not found: ", "unrecognized arguments"],
            "internal_error": ["internal error", "! interrupted: ", "traceback (most recent call last):"],
        }

        for cat, keys in indicators.items():
            if any(k in text for k in keys):
                category = cat
                unrefinable = True
                break

        # If returncode is 0, it's passing regardless
        if returncode == 0:
            category = "passed"
            unrefinable = False

        return {"unrefinable": unrefinable, "failure_category": category}

    def detect_xfail_in_output(self, pytest_output: str) -> bool:
        """
        Detect if pytest output contains XFAIL markers.
        
        Args:
            pytest_output: Combined stdout/stderr from pytest
            
        Returns:
            True if XFAIL markers are detected
        """
        xfail_indicators = [
            "XFAIL",
            "xfailed",
            "@pytest.mark.xfail",
            "pytest.xfail",
            "expected failure",
            "XPASS",  # Also catch unexpected passes of xfail tests
        ]
        
        output_lower = pytest_output.lower()
        for indicator in xfail_indicators:
            if indicator.lower() in output_lower:
                logger.debug(f"Detected XFAIL indicator: {indicator}")
                return True
        
        return False
