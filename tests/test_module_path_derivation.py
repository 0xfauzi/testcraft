"""
Tests for module path derivation utility.

Comprehensive tests for ModulePathDeriver across various project layouts:
- src/ layouts
- flat package structures  
- nested packages
- namespace packages
- edge cases and error conditions
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from testcraft.application.generation.services.structure import ModulePathDeriver


class TestModulePathDeriver:
    """Test module path derivation across different project structures."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = None

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_project(self, structure: dict, add_markers: bool = True) -> Path:
        """
        Create a temporary project with given structure.
        
        Args:
            structure: Dict representing directory/file structure
            add_markers: Whether to add project markers (pyproject.toml, etc.)
        
        Returns:
            Path to the created project root
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        
        def create_path(current_path: Path, struct: dict):
            for name, content in struct.items():
                path = current_path / name
                if isinstance(content, dict):
                    # Directory
                    path.mkdir(exist_ok=True)
                    create_path(path, content)
                else:
                    # File
                    if name.endswith('.py'):
                        path.write_text(content or "# Test file\n", encoding='utf-8')
                    else:
                        path.write_text(content or "", encoding='utf-8')
        
        create_path(self.temp_dir, structure)
        
        # Add project markers
        if add_markers:
            (self.temp_dir / "pyproject.toml").write_text(
                '[project]\nname = "test-project"\n', encoding='utf-8'
            )
        
        return self.temp_dir

    def test_src_layout_simple(self):
        """Test module path derivation for simple src/ layout."""
        project_root = self.create_test_project({
            "src": {
                "mypackage": {
                    "__init__.py": "",
                    "module.py": "def test_function(): pass\n",
                }
            }
        })
        
        file_path = project_root / "src" / "mypackage" / "module.py"
        result = ModulePathDeriver.derive_module_path(file_path, project_root)
        
        assert result["module_path"] in ["mypackage.module", "src.mypackage.module"]
        assert result["validation_status"] in ["validated", "unvalidated"]
        assert "import_suggestion" in result

    def test_src_layout_nested(self):
        """Test module path derivation for nested src/ layout."""
        project_root = self.create_test_project({
            "src": {
                "mypackage": {
                    "__init__.py": "",
                    "subpackage": {
                        "__init__.py": "",
                        "deep_module.py": "class TestClass: pass\n",
                    }
                }
            }
        })
        
        file_path = project_root / "src" / "mypackage" / "subpackage" / "deep_module.py"
        result = ModulePathDeriver.derive_module_path(file_path, project_root)
        
        assert "mypackage.subpackage.deep_module" in result["module_path"] or "src.mypackage.subpackage.deep_module" in result["module_path"]
        assert "import_suggestion" in result

    def test_flat_package_layout(self):
        """Test module path derivation for flat package layout."""
        project_root = self.create_test_project({
            "mypackage": {
                "__init__.py": "",
                "core.py": "def main(): pass\n",
                "utils.py": "def helper(): pass\n",
            }
        })
        
        file_path = project_root / "mypackage" / "core.py"
        result = ModulePathDeriver.derive_module_path(file_path, project_root)
        
        assert result["module_path"] == "mypackage.core"
        assert "import_suggestion" in result

    def test_init_file_handling(self):
        """Test module path derivation for __init__.py files."""
        project_root = self.create_test_project({
            "src": {
                "mypackage": {
                    "__init__.py": "from .core import main\n",
                    "core.py": "def main(): pass\n",
                }
            }
        })
        
        init_file = project_root / "src" / "mypackage" / "__init__.py"
        result = ModulePathDeriver.derive_module_path(init_file, project_root)
        
        assert result["module_path"] in ["mypackage", "src.mypackage"]
        assert "import_suggestion" in result

    def test_namespace_package(self):
        """Test module path derivation for PEP 420 namespace packages."""
        project_root = self.create_test_project({
            "namespace": {
                # No __init__.py (namespace package)
                "plugin.py": "class Plugin: pass\n",
            }
        })
        
        file_path = project_root / "namespace" / "plugin.py"
        result = ModulePathDeriver.derive_module_path(file_path, project_root)
        
        assert result["module_path"] == "namespace.plugin"
        assert "import_suggestion" in result

    def test_project_root_detection(self):
        """Test automatic project root detection."""
        project_root = self.create_test_project({
            "src": {
                "mypackage": {
                    "module.py": "def test(): pass\n",
                }
            }
        })
        
        file_path = project_root / "src" / "mypackage" / "module.py"
        
        # Test without providing project_root (auto-detection)
        result = ModulePathDeriver.derive_module_path(file_path, None)
        
        assert result["module_path"]  # Should find some module path
        assert "validation_status" in result

    def test_multiple_project_markers(self):
        """Test project root detection with multiple markers."""
        project_root = self.create_test_project({
            "src": {
                "mypackage": {
                    "module.py": "def test(): pass\n",
                }
            },
            "setup.py": "# setup file\n",
            "requirements.txt": "requests\n",
        }, add_markers=False)
        
        # Add multiple markers
        (project_root / "pyproject.toml").write_text('[project]\nname = "test"\n')
        (project_root / ".git").mkdir()
        
        file_path = project_root / "src" / "mypackage" / "module.py"
        result = ModulePathDeriver.derive_module_path(file_path, None)
        
        assert result["module_path"]
        assert "validation_status" in result

    def test_candidate_generation_strategies(self):
        """Test different module path candidate generation strategies."""
        project_root = self.create_test_project({
            "src": {
                "mypackage": {
                    "submodule": {
                        "deep.py": "class Deep: pass\n",
                    }
                }
            }
        })
        
        file_path = project_root / "src" / "mypackage" / "submodule" / "deep.py"
        result = ModulePathDeriver.derive_module_path(file_path, project_root)
        
        # Should have multiple fallback paths
        fallback_paths = result.get("fallback_paths", [])
        assert len(fallback_paths) >= 1  # Should have alternatives
        
        # Check that we get reasonable candidates
        module_path = result["module_path"]
        assert "deep" in module_path
        assert "mypackage" in module_path or "src" in module_path

    def test_validation_with_sys_path_modification(self):
        """Test import validation with sys.path modification."""
        project_root = self.create_test_project({
            "mypackage": {
                "__init__.py": "",
                "testmodule.py": "def test_func(): return 'success'\n",
            }
        })
        
        file_path = project_root / "mypackage" / "testmodule.py"
        
        # Mock successful import validation
        with patch('importlib.util.find_spec') as mock_find_spec:
            mock_spec = type('MockSpec', (), {})()
            mock_spec.origin = str(file_path)
            mock_find_spec.return_value = mock_spec
            
            result = ModulePathDeriver.derive_module_path(file_path, project_root)
            
            assert result["validation_status"] == "validated"
            assert result["module_path"] == "mypackage.testmodule"

    def test_validation_failure_handling(self):
        """Test handling of import validation failures."""
        project_root = self.create_test_project({
            "broken": {
                "module.py": "import nonexistent_module\n",
            }
        })
        
        file_path = project_root / "broken" / "module.py"
        
        # Mock failed import validation
        with patch('importlib.util.find_spec') as mock_find_spec:
            mock_find_spec.return_value = None  # No spec found
            
            result = ModulePathDeriver.derive_module_path(file_path, project_root)
            
            assert result["validation_status"] == "unvalidated"
            assert result["module_path"]  # Should still provide best guess
            assert len(result.get("failed_validations", [])) > 0

    def test_import_suggestion_generation(self):
        """Test generation of appropriate import suggestions."""
        project_root = self.create_test_project({
            "src": {
                "myapi": {
                    "__init__.py": "",
                    "handlers.py": "class APIHandler: pass\n",
                }
            }
        })
        
        file_path = project_root / "src" / "myapi" / "handlers.py"
        result = ModulePathDeriver.derive_module_path(file_path, project_root)
        
        import_suggestion = result.get("import_suggestion", "")
        
        assert "from" in import_suggestion
        assert "import" in import_suggestion
        assert "myapi.handlers" in import_suggestion or "src.myapi.handlers" in import_suggestion

    def test_error_handling_invalid_path(self):
        """Test error handling for invalid file paths."""
        non_existent_path = Path("/non/existent/file.py")
        result = ModulePathDeriver.derive_module_path(non_existent_path, None)
        
        assert result["validation_status"] in ["failed", "error"]
        assert "error" in result

    def test_error_handling_non_python_file(self):
        """Test handling of non-Python files."""
        project_root = self.create_test_project({
            "data.txt": "not python code\n",
        })
        
        file_path = project_root / "data.txt"
        result = ModulePathDeriver.derive_module_path(file_path, project_root)
        
        # Should still attempt to derive path but may not validate
        assert "module_path" in result

    def test_circular_symlink_handling(self):
        """Test handling of circular symlinks (if supported on platform)."""
        project_root = self.create_test_project({
            "mypackage": {
                "__init__.py": "",
                "module.py": "def test(): pass\n",
            }
        })
        
        try:
            # Create a circular symlink
            symlink_path = project_root / "circular"
            symlink_path.symlink_to(project_root)
            
            # Try to derive path from within the symlink
            file_path = symlink_path / "mypackage" / "module.py"
            result = ModulePathDeriver.derive_module_path(file_path, project_root)
            
            # Should handle gracefully without infinite recursion
            assert "module_path" in result
            
        except (OSError, NotImplementedError):
            # Skip if symlinks not supported
            pytest.skip("Platform doesn't support symlinks")

    def test_deeply_nested_structure(self):
        """Test with deeply nested package structure."""
        project_root = self.create_test_project({
            "src": {
                "level1": {
                    "__init__.py": "",
                    "level2": {
                        "__init__.py": "",
                        "level3": {
                            "__init__.py": "",
                            "level4": {
                                "__init__.py": "",
                                "deep.py": "class VeryDeep: pass\n",
                            }
                        }
                    }
                }
            }
        })
        
        file_path = project_root / "src" / "level1" / "level2" / "level3" / "level4" / "deep.py"
        result = ModulePathDeriver.derive_module_path(file_path, project_root)
        
        module_path = result["module_path"]
        assert "level1.level2.level3.level4.deep" in module_path
        assert len(module_path.split(".")) >= 4  # Should have deep nesting

    def test_project_root_edge_cases(self):
        """Test edge cases in project root detection."""
        project_root = self.create_test_project({
            "mypackage": {
                "module.py": "def test(): pass\n",
            }
        }, add_markers=False)  # No markers
        
        file_path = project_root / "mypackage" / "module.py"
        result = ModulePathDeriver.derive_module_path(file_path, None)
        
        # Should fallback to file's parent directory
        assert "module_path" in result
        assert result["module_path"]  # Non-empty

    def test_performance_with_large_directory(self):
        """Test performance doesn't degrade with large directory structures."""
        # Create a directory with many files
        large_structure = {
            "src": {
                "mypackage": {
                    "__init__.py": "",
                }
            }
        }
        
        # Add many sibling files
        for i in range(50):
            large_structure["src"]["mypackage"][f"module_{i}.py"] = f"# Module {i}\n"
        
        project_root = self.create_test_project(large_structure)
        
        file_path = project_root / "src" / "mypackage" / "module_0.py"
        
        import time
        start = time.time()
        result = ModulePathDeriver.derive_module_path(file_path, project_root)
        end = time.time()
        
        # Should complete in reasonable time (< 1 second)
        assert end - start < 1.0
        assert result["module_path"] == "mypackage.module_0" or "src.mypackage.module_0" in result["module_path"]

    def test_special_characters_in_paths(self):
        """Test handling of special characters in file/directory names."""
        project_root = self.create_test_project({
            "my-package": {  # Hyphen in name
                "__init__.py": "",
                "my_module.py": "def test(): pass\n",  # Underscore in name
            }
        })
        
        file_path = project_root / "my-package" / "my_module.py"
        result = ModulePathDeriver.derive_module_path(file_path, project_root)
        
        # Should handle special characters appropriately
        module_path = result["module_path"]
        # Python module names replace hyphens with underscores typically
        assert "my_module" in module_path
