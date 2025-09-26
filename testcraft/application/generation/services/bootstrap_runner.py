"""
Bootstrap runner service for managing sys.path configuration during test execution.

This service provides mechanisms to ensure that generated tests can import the modules
they need by either writing conftest.py files or setting PYTHONPATH environment variables.
"""

import os
from enum import Enum
from pathlib import Path

from testcraft.domain.models import ImportMap


class BootstrapStrategy(Enum):
    """Strategy for bootstrapping the Python path."""

    CONFTEST_FILE = "conftest_file"
    PYTHONPATH_ENV = "pythonpath_env"
    NO_BOOTSTRAP = "no_bootstrap"


class BootstrapRunner:
    """Service for managing test bootstrap strategies."""

    def __init__(self, prefer_conftest: bool = True):
        """Initialize the bootstrap runner.

        Args:
            prefer_conftest: Whether to prefer writing conftest.py files over
                           setting PYTHONPATH environment variables
        """
        self._prefer_conftest = prefer_conftest

    def ensure_bootstrap(
        self, import_map: ImportMap, tests_dir: Path
    ) -> BootstrapStrategy:
        """Determine and apply the appropriate bootstrap strategy.

        Args:
            import_map: Import map containing bootstrap information
            tests_dir: Directory where tests are located

        Returns:
            The bootstrap strategy that was chosen
        """
        if not import_map.needs_bootstrap or not import_map.sys_path_roots:
            return BootstrapStrategy.NO_BOOTSTRAP

        conftest_path = tests_dir / "conftest.py"

        # If conftest already exists and we prefer conftest, use it
        if conftest_path.exists() and self._prefer_conftest:
            return BootstrapStrategy.CONFTEST_FILE
        # If we prefer conftest and no conftest exists, create one
        if self._prefer_conftest and import_map.bootstrap_conftest:
            self.write_bootstrap_conftest(tests_dir, import_map.bootstrap_conftest)
            return BootstrapStrategy.CONFTEST_FILE

        # Otherwise, use PYTHONPATH
        return BootstrapStrategy.PYTHONPATH_ENV

    def write_bootstrap_conftest(self, tests_dir: Path, conftest_content: str) -> None:
        """Write a conftest.py file with bootstrap content.

        Args:
            tests_dir: Directory to write conftest.py to
            conftest_content: Content to write to conftest.py
        """
        conftest_path = tests_dir / "conftest.py"
        conftest_path.parent.mkdir(parents=True, exist_ok=True)
        conftest_path.write_text(conftest_content)

    def set_pythonpath_env(self, sys_path_roots: list[str]) -> dict[str, str]:
        """Set PYTHONPATH environment variable for bootstrap.

        Args:
            sys_path_roots: List of paths to add to sys.path

        Returns:
            Dictionary of environment variables to set
        """
        # Filter out empty strings and resolve to absolute paths
        valid_roots = [
            str(Path(root).resolve())
            for root in sys_path_roots
            if root and root.strip()
        ]

        if not valid_roots:
            return {}

        # Remove duplicates while preserving order
        unique_roots: list[str] = []
        seen: set[str] = set()
        for root in valid_roots:
            if root not in seen:
                seen.add(root)
                unique_roots.append(root)
        # Get existing PYTHONPATH
        existing_pythonpath = os.environ.get("PYTHONPATH", "")

        # Combine existing paths with new paths (prepend new paths)
        all_paths = unique_roots + existing_pythonpath.split(os.pathsep)
        all_paths = [path for path in all_paths if path]  # Remove empty strings

        # Create final PYTHONPATH
        final_pythonpath = os.pathsep.join(all_paths)

        return {"PYTHONPATH": final_pythonpath}

    def cleanup_bootstrap_conftest(self, tests_dir: Path) -> None:
        """Clean up bootstrap conftest.py files.

        Args:
            tests_dir: Directory to clean up conftest.py from
        """
        conftest_path = tests_dir / "conftest.py"

        if conftest_path.exists():
            content = conftest_path.read_text()

            # Check if this looks like a bootstrap-generated conftest
            if "sys.path.insert" in content or "sys.path.append" in content:
                conftest_path.unlink()
