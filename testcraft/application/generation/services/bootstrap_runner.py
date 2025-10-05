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

        # If conftest already exists and we prefer conftest, check if it needs bootstrap content
        if conftest_path.exists() and self._prefer_conftest:
            if self._update_existing_conftest_if_needed(
                conftest_path, import_map.bootstrap_conftest
            ):
                return BootstrapStrategy.CONFTEST_FILE
            else:
                return BootstrapStrategy.PYTHONPATH_ENV

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

        Raises:
            ValueError: If conftest_content is empty or tests_dir is not writable
        """
        if not conftest_content or not conftest_content.strip():
            raise ValueError("conftest_content cannot be empty")

        if not os.access(tests_dir, os.W_OK):
            raise ValueError(f"Directory {tests_dir} is not writable")

        conftest_path = tests_dir / "conftest.py"
        conftest_path.parent.mkdir(parents=True, exist_ok=True)

        # Add a marker comment to identify bootstrap-generated content
        bootstrap_marker = "# TestCraft bootstrap - DO NOT EDIT MANUALLY\n"
        marked_content = bootstrap_marker + conftest_content

        conftest_path.write_text(marked_content)

    def _update_existing_conftest_if_needed(
        self, conftest_path: Path, bootstrap_content: str
    ) -> bool:
        """Check if existing conftest needs bootstrap content and update if necessary.

        Args:
            conftest_path: Path to the existing conftest.py file
            bootstrap_content: Bootstrap content to potentially add

        Returns:
            True if conftest was updated with bootstrap content, False otherwise
        """
        if not bootstrap_content or not bootstrap_content.strip():
            return False

        try:
            existing_content = conftest_path.read_text()
        except (OSError, UnicodeDecodeError):
            # If we can't read the file, assume it needs bootstrap content
            return False

        # Check if bootstrap content is already present
        bootstrap_marker = "# TestCraft bootstrap - DO NOT EDIT MANUALLY"
        if bootstrap_marker in existing_content:
            return True  # Already has bootstrap content

        # Check if it has sys.path modifications (broader detection)
        if any(
            pattern in existing_content
            for pattern in ["sys.path.insert", "sys.path.append", "sys.path.prepend"]
        ):
            return True  # Already has sys.path modifications

        # Add bootstrap content to existing conftest
        try:
            bootstrap_marker = "# TestCraft bootstrap - DO NOT EDIT MANUALLY\n"
            updated_content = (
                existing_content.rstrip()
                + "\n\n"
                + bootstrap_marker
                + bootstrap_content
            )
            conftest_path.write_text(updated_content)
            return True
        except OSError:
            # If we can't write, fall back to not using bootstrap
            return False

        header = (
            "# AUTOGENERATED BY TestCraft BootstrapRunner\n"
            "# This file is created to ensure imports during tests.\n"
            "# SAFE TO DELETE if not needed.\n"
        )
        conftest_path.write_text(header + conftest_content, encoding="utf-8")
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
            try:
                content = conftest_path.read_text()
            except (OSError, UnicodeDecodeError):
                # If we can't read the file, don't try to clean it up
                return

            # Check for bootstrap-generated content markers
            bootstrap_markers = [
                "# TestCraft bootstrap - DO NOT EDIT MANUALLY",
                "sys.path.insert",
                "sys.path.append",
                "sys.path.prepend",
            ]

            # Only clean up if we find bootstrap markers
            has_bootstrap_content = any(
                marker in content for marker in bootstrap_markers
            )

            if has_bootstrap_content:
                try:
                    conftest_path.unlink()
                except OSError:
                    # If we can't delete, silently ignore
                    pass
