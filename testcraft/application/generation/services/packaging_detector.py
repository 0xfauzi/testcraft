"""
Packaging detection service for resolving Python project structure and import paths.

This service analyzes project structure to determine:
- Whether directories like 'src' are packages or source roots
- Canonical import paths for modules
- Project layout patterns (flat, src/, etc.)
- Import validation and safety rules
"""

from __future__ import annotations

import ast
import logging
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PackagingInfo:
    """Information about project packaging structure."""

    def __init__(
        self,
        project_root: Path,
        source_roots: list[Path],
        src_is_package: bool,
        package_directories: set[Path],
        module_import_map: dict[str, str],
        disallowed_import_prefixes: list[str],
    ):
        self.project_root = project_root
        self.source_roots = source_roots
        self.src_is_package = src_is_package
        self.package_directories = package_directories
        self.module_import_map = module_import_map
        self.disallowed_import_prefixes = disallowed_import_prefixes

    def get_canonical_import(self, file_path: Path) -> str | None:
        """Get canonical import path for a file."""
        abs_path = str(file_path.resolve())
        return self.module_import_map.get(abs_path)

    def is_import_allowed(self, import_path: str) -> bool:
        """Check if an import path is allowed."""
        for prefix in self.disallowed_import_prefixes:
            if import_path.startswith(prefix):
                return False
        return True


class PackagingDetector:
    """
    Service for detecting Python project packaging structure.

    Analyzes project layout to determine correct import paths and packaging rules.
    """

    @staticmethod
    def detect_packaging(project_root: Path) -> PackagingInfo:
        """
        Detect packaging structure for a project.

        Args:
            project_root: Root directory of the project

        Returns:
            PackagingInfo with detected structure information
        """
        try:
            logger.debug("Detecting packaging structure for %s", project_root)

            # Step 1: Parse pyproject.toml for explicit configuration
            pyproject_info = PackagingDetector._parse_pyproject_toml(project_root)

            # Step 2: Detect source roots
            source_roots = PackagingDetector._detect_source_roots(
                project_root, pyproject_info
            )

            # Step 3: Determine if 'src' is a package or source root
            src_is_package = PackagingDetector._is_src_package(project_root)

            # Step 4: Find all package directories
            package_directories = PackagingDetector._find_package_directories(
                project_root, source_roots
            )

            # Step 5: Build module import map
            module_import_map = PackagingDetector._build_module_import_map(
                project_root, source_roots, package_directories
            )

            # Step 6: Determine disallowed import prefixes
            disallowed_prefixes = PackagingDetector._get_disallowed_prefixes(
                src_is_package, source_roots
            )

            packaging_info = PackagingInfo(
                project_root=project_root,
                source_roots=source_roots,
                src_is_package=src_is_package,
                package_directories=package_directories,
                module_import_map=module_import_map,
                disallowed_import_prefixes=disallowed_prefixes,
            )

            logger.debug(
                "Detected packaging: src_is_package=%s, source_roots=%s, disallowed_prefixes=%s",
                src_is_package,
                [str(sr) for sr in source_roots],
                disallowed_prefixes,
            )

            return packaging_info

        except Exception as e:
            logger.warning("Failed to detect packaging structure: %s", e)
            # Return minimal fallback info
            return PackagingInfo(
                project_root=project_root,
                source_roots=[project_root],
                src_is_package=False,
                package_directories=set(),
                module_import_map={},
                disallowed_import_prefixes=[],
            )

    @staticmethod
    def _parse_pyproject_toml(project_root: Path) -> dict[str, Any]:
        """Parse pyproject.toml for packaging configuration."""
        pyproject_path = project_root / "pyproject.toml"
        if not pyproject_path.exists():
            return {}

        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)

            # Extract relevant packaging information
            info = {}

            # Build system configuration
            build_system = data.get("build-system", {})
            info["build_backend"] = build_system.get("build-backend", "")

            # Project configuration
            project = data.get("project", {})
            info["name"] = project.get("name", "")

            # Tool-specific configurations
            tools = data.get("tool", {})

            # setuptools configuration
            setuptools = tools.get("setuptools", {})
            if "package-dir" in setuptools:
                info["package_dir"] = setuptools["package-dir"]
            if "packages" in setuptools:
                info["packages"] = setuptools["packages"]

            # pytest configuration (for test paths)
            pytest_config = tools.get("pytest", {})
            if "ini_options" in pytest_config:
                ini_options = pytest_config["ini_options"]
                if "testpaths" in ini_options:
                    info["testpaths"] = ini_options["testpaths"]
                if "pythonpath" in ini_options:
                    info["pythonpath"] = ini_options["pythonpath"]

            return info

        except Exception as e:
            logger.debug("Failed to parse pyproject.toml: %s", e)
            return {}

    @staticmethod
    def _detect_source_roots(
        project_root: Path, pyproject_info: dict[str, Any]
    ) -> list[Path]:
        """Detect source root directories."""
        source_roots = []

        # Strategy 1: Explicit configuration from pyproject.toml
        if "package_dir" in pyproject_info:
            package_dir = pyproject_info["package_dir"]
            if isinstance(package_dir, dict):
                # {"": "src"} means src is the source root
                if "" in package_dir:
                    src_dir = project_root / package_dir[""]
                    if src_dir.exists():
                        source_roots.append(src_dir)

        if "pythonpath" in pyproject_info:
            pythonpath = pyproject_info["pythonpath"]
            if isinstance(pythonpath, list):
                for path_str in pythonpath:
                    path = project_root / path_str
                    if path.exists():
                        source_roots.append(path)
            elif isinstance(pythonpath, str):
                path = project_root / pythonpath
                if path.exists():
                    source_roots.append(path)

        # Strategy 2: Common patterns
        common_source_dirs = ["src", "lib", project_root.name]
        for dir_name in common_source_dirs:
            src_dir = project_root / dir_name
            if src_dir.exists() and src_dir.is_dir():
                # Check if it contains Python packages
                if any(
                    (src_dir / item).is_dir()
                    and (src_dir / item / "__init__.py").exists()
                    for item in src_dir.iterdir()
                    if not item.name.startswith(".")
                ):
                    source_roots.append(src_dir)

        # Strategy 3: Project root as fallback
        if not source_roots:
            source_roots.append(project_root)

        # Remove duplicates while preserving order
        seen = set()
        unique_roots = []
        for root in source_roots:
            root_resolved = root.resolve()
            if root_resolved not in seen:
                seen.add(root_resolved)
                unique_roots.append(root)

        return unique_roots

    @staticmethod
    def _is_src_package(project_root: Path) -> bool:
        """Determine if 'src' directory is a Python package."""
        src_dir = project_root / "src"
        if not src_dir.exists():
            return False

        # If src has __init__.py, it's a package
        if (src_dir / "__init__.py").exists():
            return True

        # If src contains only one subdirectory with __init__.py, src is likely a source root
        subdirs = [
            item
            for item in src_dir.iterdir()
            if item.is_dir() and not item.name.startswith(".")
        ]

        if len(subdirs) == 1 and (subdirs[0] / "__init__.py").exists():
            return False  # src is a source root, not a package

        # If src contains multiple packages, it's likely a source root
        package_count = sum(
            1 for subdir in subdirs if (subdir / "__init__.py").exists()
        )

        if package_count > 1:
            return False  # src is a source root

        # Default: assume src is not a package
        return False

    @staticmethod
    def _find_package_directories(
        project_root: Path, source_roots: list[Path]
    ) -> set[Path]:
        """Find all Python package directories."""
        packages = set()

        for source_root in source_roots:
            try:
                for item in source_root.rglob("__init__.py"):
                    package_dir = item.parent
                    packages.add(package_dir)
            except Exception as e:
                logger.debug("Error scanning for packages in %s: %s", source_root, e)

        return packages

    @staticmethod
    def _build_module_import_map(
        project_root: Path,
        source_roots: list[Path],
        package_directories: set[Path],
    ) -> dict[str, str]:
        """Build mapping from file paths to import paths."""
        import_map = {}

        for source_root in source_roots:
            try:
                # Find all Python files
                for py_file in source_root.rglob("*.py"):
                    if py_file.name.startswith("."):
                        continue

                    # Calculate relative path from source root
                    try:
                        rel_path = py_file.relative_to(source_root)
                    except ValueError:
                        continue

                    # Convert to module path
                    module_parts = list(rel_path.parts)

                    # Remove .py extension
                    if module_parts[-1].endswith(".py"):
                        module_parts[-1] = module_parts[-1][:-3]

                    # Handle __init__.py files
                    if module_parts[-1] == "__init__":
                        module_parts = module_parts[:-1]

                    # Skip empty module paths
                    if not module_parts:
                        continue

                    # Build dotted import path
                    import_path = ".".join(module_parts)

                    # Store mapping
                    abs_file_path = str(py_file.resolve())
                    import_map[abs_file_path] = import_path

            except Exception as e:
                logger.debug("Error building import map for %s: %s", source_root, e)

        return import_map

    @staticmethod
    def _get_disallowed_prefixes(
        src_is_package: bool, source_roots: list[Path]
    ) -> list[str]:
        """Get list of disallowed import prefixes."""
        disallowed = []

        # If src is not a package, disallow "src." imports
        if not src_is_package:
            disallowed.append("src.")

        # Add other common non-package directories
        non_package_dirs = {"tests", "test", "docs", "examples", "scripts", "tools"}
        for source_root in source_roots:
            if source_root.name in non_package_dirs:
                disallowed.append(f"{source_root.name}.")

        return disallowed


class EntityInterfaceDetector:
    """
    Detector for entity interfaces and ORM models.

    Identifies classes that should not be instantiated in tests (like SQLAlchemy models)
    and provides interface information for stubbing.
    """

    @staticmethod
    def detect_entities(file_path: Path) -> dict[str, Any]:
        """
        Detect entity classes and their interfaces in a Python file.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            Dictionary with entity information
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            entities = {}

            # Look for class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    entity_info = EntityInterfaceDetector._analyze_class(node, content)
                    if entity_info:
                        entities[node.name] = entity_info

            return {
                "entities": entities,
                "imports": EntityInterfaceDetector._extract_imports(tree),
                "has_orm_models": any(
                    info.get("kind") == "sqlalchemy.Model" for info in entities.values()
                ),
            }

        except Exception as e:
            logger.debug("Failed to detect entities in %s: %s", file_path, e)
            return {"entities": {}, "imports": [], "has_orm_models": False}

    @staticmethod
    def _analyze_class(node: ast.ClassDef, content: str) -> dict[str, Any] | None:
        """Analyze a class definition to determine if it's an entity."""
        info = {
            "name": node.name,
            "kind": "regular_class",
            "instantiate_real": True,
            "attributes_read_by_uut": [],
            "constructor_signature": None,
        }

        # Check base classes for ORM patterns
        for base in node.bases:
            base_name = EntityInterfaceDetector._get_name_from_node(base)
            if base_name:
                if "Model" in base_name or "Base" in base_name:
                    info["kind"] = "sqlalchemy.Model"
                    info["instantiate_real"] = False
                elif "Document" in base_name:
                    info["kind"] = "mongoengine.Document"
                    info["instantiate_real"] = False
                elif "BaseModel" in base_name:
                    info["kind"] = "pydantic.BaseModel"
                    info["instantiate_real"] = (
                        True  # Pydantic models are safe to instantiate
                    )

        # Extract attributes from class body
        attributes = []
        for class_node in node.body:
            if isinstance(class_node, ast.AnnAssign) and isinstance(
                class_node.target, ast.Name
            ):
                attributes.append(class_node.target.id)
            elif isinstance(class_node, ast.Assign):
                for target in class_node.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)

        info["attributes_read_by_uut"] = attributes

        # Analyze constructor
        for class_node in node.body:
            if (
                isinstance(class_node, ast.FunctionDef)
                and class_node.name == "__init__"
            ):
                info["constructor_signature"] = (
                    EntityInterfaceDetector._get_function_signature(class_node, content)
                )
                break

        return info

    @staticmethod
    def _get_name_from_node(node: ast.AST) -> str | None:
        """Extract name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    @staticmethod
    def _get_function_signature(node: ast.FunctionDef, content: str) -> str | None:
        """Extract function signature from AST node."""
        try:
            if hasattr(node, "lineno"):
                lines = content.split("\n")
                if node.lineno - 1 < len(lines):
                    return lines[node.lineno - 1].strip()
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_imports(tree: ast.AST) -> list[dict[str, Any]]:
        """Extract import information from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "type": "import",
                            "module": alias.name,
                            "alias": alias.asname,
                        }
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    imports.append(
                        {
                            "type": "from_import",
                            "module": node.module,
                            "name": alias.name,
                            "alias": alias.asname,
                        }
                    )

        return imports
