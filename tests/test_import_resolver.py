"""
Tests for ImportResolver service.

Tests canonical import resolution for various project layouts:
- Flat layout (packages in project root)
- src/ layout (packages under src/)
- Monorepo layout (multiple package roots)
- Edge cases (no packages, custom layouts)
"""

import tempfile
from pathlib import Path

import pytest

from testcraft.application.generation.services.import_resolver import (
    ImportResolver,
)


class TestImportResolver:
    """Test ImportResolver service."""

    @pytest.fixture
    def resolver(self):
        """Create ImportResolver instance."""
        return ImportResolver()

    @pytest.fixture
    def temp_project(self):
        """Create temporary project structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            yield project_root

    def test_flat_layout_simple_module(self, resolver, temp_project):
        """Test import resolution for flat layout with simple module."""
        # Setup: flat layout with package at root
        # project_root/
        #   pyproject.toml
        #   mypackage/
        #     __init__.py
        #     utils.py

        (temp_project / "pyproject.toml").write_text('[project]\nname = "myproject"\n')
        pkg_dir = temp_project / "mypackage"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        utils_file = pkg_dir / "utils.py"
        utils_file.write_text("def helper(): pass")

        # Test
        result = resolver.resolve(utils_file)

        # Verify
        assert isinstance(result, dict)
        assert result["target_import"] == "import mypackage.utils as _under_test"
        assert len(result["sys_path_roots"]) == 1
        assert str(temp_project.resolve()) in result["sys_path_roots"]
        assert result["needs_bootstrap"] is False  # Project root, pytest should handle
        assert result["bootstrap_conftest"] == ""

    def test_src_layout_module(self, resolver, temp_project):
        """Test import resolution for src/ layout."""
        # Setup: src/ layout
        # project_root/
        #   pyproject.toml
        #   src/
        #     mypackage/
        #       __init__.py
        #       core.py

        (temp_project / "pyproject.toml").write_text('[project]\nname = "myproject"\n')
        src_dir = temp_project / "src"
        src_dir.mkdir()
        pkg_dir = src_dir / "mypackage"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        core_file = pkg_dir / "core.py"
        core_file.write_text("class MyClass: pass")

        # Test
        result = resolver.resolve(core_file)

        # Verify
        assert result["target_import"] == "import mypackage.core as _under_test"
        assert len(result["sys_path_roots"]) == 1
        assert str((temp_project / "src").resolve()) in result["sys_path_roots"]
        assert result["needs_bootstrap"] is True  # src/ layout needs bootstrap

        # Verify bootstrap conftest content
        conftest = result["bootstrap_conftest"]
        assert "import sys" in conftest
        assert "import pathlib" in conftest
        assert str((temp_project / "src").resolve()) in conftest
        assert "sys.path.insert(0, str(p))" in conftest

    def test_nested_subpackage(self, resolver, temp_project):
        """Test import resolution for nested subpackages."""
        # Setup: nested package structure
        # project_root/
        #   pyproject.toml
        #   mypackage/
        #     __init__.py
        #     sub/
        #       __init__.py
        #       module.py

        (temp_project / "pyproject.toml").write_text('[project]\nname = "myproject"\n')
        pkg_dir = temp_project / "mypackage"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        sub_dir = pkg_dir / "sub"
        sub_dir.mkdir()
        (sub_dir / "__init__.py").write_text("")
        module_file = sub_dir / "module.py"
        module_file.write_text("def nested_func(): pass")

        # Test
        result = resolver.resolve(module_file)

        # Verify
        assert result["target_import"] == "import mypackage.sub.module as _under_test"
        assert result["needs_bootstrap"] is False

    def test_monorepo_multiple_packages(self, resolver, temp_project):
        """Test import resolution with multiple package roots."""
        # Setup: monorepo layout with multiple source directories
        # project_root/
        #   pyproject.toml
        #   libs/
        #     pkg_a/
        #       __init__.py
        #       module_a.py
        #     pkg_b/
        #       __init__.py
        #       module_b.py

        (temp_project / "pyproject.toml").write_text('[project]\nname = "monorepo"\n')
        libs_dir = temp_project / "libs"
        libs_dir.mkdir()

        # Package A
        pkg_a_dir = libs_dir / "pkg_a"
        pkg_a_dir.mkdir()
        (pkg_a_dir / "__init__.py").write_text("")
        module_a_file = pkg_a_dir / "module_a.py"
        module_a_file.write_text("def func_a(): pass")

        # Package B
        pkg_b_dir = libs_dir / "pkg_b"
        pkg_b_dir.mkdir()
        (pkg_b_dir / "__init__.py").write_text("")
        module_b_file = pkg_b_dir / "module_b.py"
        module_b_file.write_text("def func_b(): pass")

        # Test package A
        result_a = resolver.resolve(module_a_file)
        assert result_a["target_import"] == "import pkg_a.module_a as _under_test"

        # Test package B
        result_b = resolver.resolve(module_b_file)
        assert result_b["target_import"] == "import pkg_b.module_b as _under_test"

    def test_pyproject_toml_package_dir_configuration(self, resolver, temp_project):
        """Test with explicit package-dir configuration in pyproject.toml."""
        # Setup: pyproject.toml specifies package directory
        (temp_project / "pyproject.toml").write_text("""
[project]
name = "myproject"

[tool.setuptools]
package-dir = {"" = "src"}
""")

        src_dir = temp_project / "src"
        src_dir.mkdir()
        pkg_dir = src_dir / "mypackage"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        module_file = pkg_dir / "module.py"
        module_file.write_text("def configured_func(): pass")

        # Test
        result = resolver.resolve(module_file)

        # Verify
        assert result["target_import"] == "import mypackage.module as _under_test"
        assert str((temp_project / "src").resolve()) in result["sys_path_roots"]
        assert result["needs_bootstrap"] is True

    def test_no_pyproject_toml_heuristic_detection(self, resolver, temp_project):
        """Test heuristic detection when no pyproject.toml exists."""
        # Setup: no pyproject.toml, just package structure
        pkg_dir = temp_project / "mypackage"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        module_file = pkg_dir / "utils.py"
        module_file.write_text("def heuristic_func(): pass")

        # Test
        result = resolver.resolve(module_file)

        # Verify
        assert result["target_import"] == "import mypackage.utils as _under_test"
        assert str(temp_project.resolve()) in result["sys_path_roots"]

    def test_bootstrap_conftest_format(self, resolver, temp_project):
        """Test the exact format of generated bootstrap conftest."""
        # Setup src/ layout to trigger bootstrap
        (temp_project / "pyproject.toml").write_text('[project]\nname = "test"\n')
        src_dir = temp_project / "src"
        src_dir.mkdir()
        pkg_dir = src_dir / "pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        module_file = pkg_dir / "mod.py"
        module_file.write_text("pass")

        # Test
        result = resolver.resolve(module_file)
        conftest = result["bootstrap_conftest"]

        # Verify exact format per specification
        expected_lines = [
            "import sys",
            "import pathlib",
            "# Auto-generated bootstrap for repository-aware test execution",
            f"for p in ['{src_dir.resolve()}']:",
            "    p = pathlib.Path(p).resolve()",
            "    if str(p) not in sys.path:",
            "        sys.path.insert(0, str(p))",
        ]

        for line in expected_lines:
            assert line in conftest

    def test_edge_case_no_packages(self, resolver, temp_project):
        """Test edge case with no Python packages."""
        # Setup: just a standalone Python file
        standalone_file = temp_project / "standalone.py"
        standalone_file.write_text("def standalone_func(): pass")

        # Test - should handle gracefully or raise clear error
        with pytest.raises((ValueError, FileNotFoundError)):
            resolver.resolve(standalone_file)

    def test_cache_behavior(self, resolver, temp_project):
        """Test that packaging info is cached properly."""
        # Setup simple package
        (temp_project / "pyproject.toml").write_text('[project]\nname = "cached"\n')
        pkg_dir = temp_project / "pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")

        file1 = pkg_dir / "mod1.py"
        file1.write_text("def func1(): pass")
        file2 = pkg_dir / "mod2.py"
        file2.write_text("def func2(): pass")

        # Test - resolve for two files in same project
        result1 = resolver.resolve(file1)
        result2 = resolver.resolve(file2)

        # Verify caching worked (same sys_path_roots)
        assert result1["sys_path_roots"] == result2["sys_path_roots"]
        assert result1["needs_bootstrap"] == result2["needs_bootstrap"]

        # But different target imports
        assert result1["target_import"] == "import pkg.mod1 as _under_test"
        assert result2["target_import"] == "import pkg.mod2 as _under_test"

    def test_import_map_type_consistency(self, resolver, temp_project):
        """Test that returned ImportMap matches expected TypedDict structure."""
        # Setup
        (temp_project / "pyproject.toml").write_text('[project]\nname = "typed"\n')
        pkg_dir = temp_project / "pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        module_file = pkg_dir / "mod.py"
        module_file.write_text("def typed_func(): pass")

        # Test
        result = resolver.resolve(module_file)

        # Verify all required keys exist with correct types
        assert isinstance(result, dict)
        assert "target_import" in result
        assert "sys_path_roots" in result
        assert "needs_bootstrap" in result
        assert "bootstrap_conftest" in result

        assert isinstance(result["target_import"], str)
        assert isinstance(result["sys_path_roots"], list)
        assert isinstance(result["needs_bootstrap"], bool)
        assert isinstance(result["bootstrap_conftest"], str)

        # All sys_path_roots should be strings
        assert all(isinstance(root, str) for root in result["sys_path_roots"])
