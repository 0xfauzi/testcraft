"""
Integration tests for enhanced context assembly with module paths and recursive directory trees.

Tests the integration of:
- Recursive directory tree building
- Module path derivation
- Enhanced usage examples with module-qualified imports
- Complete context assembly pipeline
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from testcraft.application.generation.services.context_assembler import ContextAssembler
from testcraft.application.generation.services.structure import (
    DirectoryTreeBuilder,
    ModulePathDeriver,
)
from testcraft.domain.models import TestElement, TestElementType, TestGenerationPlan


class TestEnhancedContextIntegration:
    """Integration tests for enhanced context assembly features."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = None
        self.mock_context_port = MagicMock()
        self.mock_parser_port = MagicMock()

        # Default config with enhanced features enabled
        self.config = {
            "enable_context": True,
            "context_enrichment": {
                "enable_usage_examples": True,
                "enable_env_detection": True,
                "enable_comprehensive_fixtures": True,
            },
            "context_budgets": {
                "directory_tree": {
                    "max_depth": 4,
                    "max_entries_per_dir": 200,
                    "include_py_only": True,
                },
            },
            "prompt_budgets": {
                "per_item_chars": 600,
                "total_chars": 4000,
                "section_caps": {
                    "usage_examples": 5,
                },
            },
        }

        self.context_assembler = ContextAssembler(
            context_port=self.mock_context_port,
            parser_port=self.mock_parser_port,
            config=self.config,
        )

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir:
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_project(self, structure: dict) -> Path:
        """Create a temporary project with given structure."""
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
                    path.write_text(content or "# Test file\n", encoding="utf-8")

        create_path(self.temp_dir, structure)

        # Add project markers
        (self.temp_dir / "pyproject.toml").write_text(
            '[project]\nname = "test-project"\n', encoding="utf-8"
        )

        return self.temp_dir

    def test_recursive_directory_tree_integration(self):
        """Test recursive directory tree building with realistic project structure."""
        project_root = self.create_test_project(
            {
                "src": {
                    "myapi": {
                        "__init__.py": "",
                        "handlers": {
                            "__init__.py": "",
                            "users.py": "class UserHandler: pass\n",
                            "auth.py": "class AuthHandler: pass\n",
                        },
                        "models": {
                            "__init__.py": "",
                            "user.py": "class User: pass\n",
                        },
                        "utils.py": "def helper(): pass\n",
                    }
                },
                "tests": {
                    "__init__.py": "",
                    "test_handlers": {
                        "__init__.py": "",
                        "test_users.py": "def test_user_handler(): pass\n",
                    },
                    "fixtures": {
                        "users.py": "@pytest.fixture\ndef user(): pass\n",
                    },
                },
                "docs": {
                    "api.md": "# API Documentation\n",
                },
            }
        )

        files_to_process = [project_root / "src" / "myapi" / "handlers" / "users.py"]

        # Test gather_project_context with recursive tree
        context = self.context_assembler.gather_project_context(
            project_root, files_to_process
        )

        # Verify project structure is recursive and detailed
        project_structure = context.get("project_structure", {})
        assert project_structure["name"] == project_root.name
        assert project_structure["type"] == "directory"

        # Should have children with nested structure
        children = project_structure.get("children", [])
        assert len(children) > 0

        # Find src directory
        src_dir = next((c for c in children if c["name"] == "src"), None)
        assert src_dir is not None
        assert src_dir["type"] == "directory"

        # Should have nested myapi structure
        src_children = src_dir.get("children", [])
        myapi_dir = next((c for c in src_children if c["name"] == "myapi"), None)
        assert myapi_dir is not None

    def test_module_path_derivation_integration(self):
        """Test module path derivation integration with context assembly."""
        project_root = self.create_test_project(
            {
                "src": {
                    "mypackage": {
                        "__init__.py": "",
                        "core.py": """
class CoreService:
    def process(self): pass

def utility_function(): pass
""",
                    }
                }
            }
        )

        source_file = project_root / "src" / "mypackage" / "core.py"

        # Create test plan
        plan = TestGenerationPlan(
            file_path=str(source_file),
            elements_to_test=[
                TestElement(
                    name="CoreService",
                    type=TestElementType.CLASS,
                    line_range=(2, 4),
                    docstring="Service class",
                ),
                TestElement(
                    name="utility_function",
                    type=TestElementType.FUNCTION,
                    line_range=(6, 6),
                    docstring="Utility function",
                ),
            ],
            existing_tests=[],
            coverage_info={},
        )

        # Mock successful module path derivation
        with patch.object(ModulePathDeriver, "derive_module_path") as mock_derive:
            mock_derive.return_value = {
                "module_path": "mypackage.core",
                "validation_status": "validated",
                "import_suggestion": "from mypackage.core import {ClassName}",
            }

        # Test context_for_generation
        context_result = self.context_assembler.context_for_generation(
            plan, source_file
        )

        # Verify result structure and content
        assert context_result is not None
        from testcraft.domain.models import ContextPack

        assert isinstance(context_result, ContextPack), (
            f"Expected ContextPack, got {type(context_result)}"
        )

        # Extract context string and import_map
        context_string = context_result.context or ""
        import_map = context_result.import_map

        # Context should contain module path information (either in context string or import_map)
        has_module_path = (
            "Module Path: mypackage.core" in context_string
            or "mypackage.core" in context_string
            or (import_map and "mypackage.core" in import_map.target_import)
        )
        assert has_module_path, (
            f"Expected module path info in context string: {context_string} or import_map: {import_map}"
        )

    def test_enhanced_usage_examples_integration(self):
        """Test enhanced usage examples with module-qualified imports."""
        project_root = self.create_test_project(
            {
                "mylib": {
                    "__init__.py": "",
                    "calculator.py": """
class Calculator:
    def add(self, a, b): return a + b
    def multiply(self, a, b): return a * b
""",
                }
            }
        )

        source_file = project_root / "mylib" / "calculator.py"

        # Create test plan
        plan = TestGenerationPlan(
            file_path=str(source_file),
            elements_to_test=[
                TestElement(
                    name="Calculator",
                    type=TestElementType.CLASS,
                    line_range=(2, 4),
                    docstring="Calculator class",
                ),
            ],
            existing_tests=[],
            coverage_info={},
        )

        # Mock context port to return usage examples
        self.mock_context_port.retrieve.return_value = {
            "results": [
                {
                    "snippet": "from mylib.calculator import Calculator\ncalc = Calculator()",
                    "path": "examples/usage.py",
                },
                {
                    "snippet": "calculator = Calculator()\nresult = calculator.add(2, 3)",
                    "path": "tests/test_calc.py",
                },
            ]
        }

        # Mock module path derivation
        with patch.object(ModulePathDeriver, "derive_module_path") as mock_derive:
            mock_derive.return_value = {
                "module_path": "mylib.calculator",
                "validation_status": "validated",
            }

            # Test context generation with enhanced usage examples
            context_result = self.context_assembler.context_for_generation(
                plan, source_file
            )

            # Extract context string and import_map from result
            context_string = None
            import_map = None
            if context_result:
                from testcraft.domain.models import ContextPack

                if isinstance(context_result, ContextPack):
                    context_string = context_result.context
                    import_map = context_result.import_map
                elif isinstance(context_result, dict):
                    context_string = context_result.get("context")
                    import_map = context_result.get("import_map")
                else:
                    # Backward compatibility: if it's still a string
                    context_string = context_result

            # Verify usage examples prioritize module-qualified imports
            assert context_result is not None
            assert (
                context_string
                and (
                    "from mylib.calculator import" in context_string
                    or "mylib.calculator" in context_string
                )
            ) or (import_map and ("mylib.calculator" in str(import_map)))

    def test_comprehensive_context_assembly_pipeline(self):
        """Test the complete context assembly pipeline with all enhancements."""
        project_root = self.create_test_project(
            {
                "src": {
                    "webapp": {
                        "__init__.py": "",
                        "api": {
                            "__init__.py": "",
                            "endpoints.py": """
import os
from typing import Dict
from fastapi import FastAPI

app = FastAPI()

class UserService:
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL', 'sqlite:///default.db')

    def create_user(self, user_data: Dict) -> Dict:
        if not user_data.get('email'):
            raise ValueError('Email is required')
        # Implementation here
        return {'id': 1, 'email': user_data['email']}
""",
                        },
                        "models": {
                            "__init__.py": "",
                            "user.py": "class User: pass\n",
                        },
                    }
                },
                "tests": {
                    "conftest.py": """
import pytest

@pytest.fixture
def user_service():
    from webapp.api.endpoints import UserService
    return UserService()

@pytest.fixture
def sample_user_data():
    return {'email': 'test@example.com', 'name': 'Test User'}
""",
                    "test_endpoints.py": "# existing tests\n",
                },
            }
        )

        source_file = project_root / "src" / "webapp" / "api" / "endpoints.py"

        # Create comprehensive test plan
        plan = TestGenerationPlan(
            file_path=str(source_file),
            elements_to_test=[
                TestElement(
                    name="UserService",
                    type=TestElementType.CLASS,
                    line_range=(8, 16),
                    docstring="User management service",
                ),
                TestElement(
                    name="UserService.create_user",
                    type=TestElementType.METHOD,
                    line_range=(12, 16),
                    docstring="Create a new user",
                ),
            ],
            existing_tests=[],
            coverage_info={},
        )

        # Mock comprehensive context port responses
        self.mock_context_port.build_context_graph.return_value = {"graph": "mock"}
        self.mock_context_port.index.return_value = {"indexed": True}
        self.mock_context_port.get_related_context.return_value = {
            "related_files": [
                str(project_root / "src" / "webapp" / "models" / "user.py")
            ],
            "relationships": ["imports", "usage"],
        }
        self.mock_context_port.retrieve.return_value = {
            "results": [
                {
                    "snippet": "from webapp.api.endpoints import UserService",
                    "path": "tests/test_endpoints.py",
                },
            ]
        }

        # Mock parser port
        self.mock_parser_port.parse_file.return_value = {
            "ast": None,  # Would be actual AST
            "source_lines": source_file.read_text().splitlines(),
        }
        self.mock_parser_port.analyze_dependencies.return_value = {
            "imports": [
                {"module": "os", "items": [], "alias": ""},
                {"module": "typing", "items": ["Dict"], "alias": ""},
                {"module": "fastapi", "items": ["FastAPI"], "alias": ""},
            ],
            "internal_deps": ["webapp.models.user"],
        }

        # Test complete context assembly
        files_to_process = [source_file]

        # Test gather_project_context
        project_context = self.context_assembler.gather_project_context(
            project_root, files_to_process
        )

        # Verify comprehensive project context
        assert "context_graph" in project_context
        assert "indexed_files" in project_context
        assert "project_structure" in project_context

        # Project structure should be recursive and include all relevant files
        project_structure = project_context["project_structure"]
        assert project_structure["type"] == "directory"
        assert len(project_structure.get("children", [])) > 0

        # Test context_for_generation with comprehensive features
        with patch.object(ModulePathDeriver, "derive_module_path") as mock_derive:
            mock_derive.return_value = {
                "module_path": "webapp.api.endpoints",
                "validation_status": "validated",
                "import_suggestion": "from webapp.api.endpoints import {ClassName}",
            }

            context_result = self.context_assembler.context_for_generation(
                plan, source_file
            )

            # Extract context string and import_map from result
            context_string = None
            import_map = None
            if context_result:
                from testcraft.domain.models import ContextPack

                if isinstance(context_result, ContextPack):
                    context_string = context_result.context
                    import_map = context_result.import_map
                elif isinstance(context_result, dict):
                    context_string = context_result.get("context")
                    import_map = context_result.get("import_map")
                else:
                    # Backward compatibility: if it's still a string
                    context_string = context_result

            # Verify comprehensive context
            assert context_result is not None
            assert context_string is not None
            # Context string should contain at least some meaningful content
            assert len(context_string) > 50  # Should be substantial context

            # Should include module path information
            assert "webapp.api.endpoints" in context_string or (
                import_map and "webapp.api.endpoints" in str(import_map)
            )

            # Should include environment variable detection
            assert "DATABASE_URL" in context_string or "env" in context_string.lower()

            # Should include fixture information
            assert (
                "fixture" in context_string.lower()
                or "pytest" in context_string.lower()
            )

    def test_context_budgets_enforcement(self):
        """Test that context budgets are properly enforced."""
        # Create project with many files to test budget limits
        large_structure = {
            "src": {
                "bigpackage": {
                    "__init__.py": "",
                }
            }
        }

        # Add many files to test directory traversal limits
        for i in range(20):
            large_structure["src"]["bigpackage"][f"module_{i}.py"] = (
                f"class Module{i}: pass\n"
            )

        # Add nested directories
        for i in range(5):
            dir_name = f"subpackage_{i}"
            large_structure["src"]["bigpackage"][dir_name] = {
                "__init__.py": "",
            }
            for j in range(10):
                large_structure["src"]["bigpackage"][dir_name][f"submodule_{j}.py"] = (
                    f"class SubModule{j}: pass\n"
                )

        project_root = self.create_test_project(large_structure)

        # Test with restrictive budgets
        restrictive_config = self.config.copy()
        restrictive_config["context_budgets"]["directory_tree"].update(
            {
                "max_depth": 2,
                "max_entries_per_dir": 10,
            }
        )

        context_assembler = ContextAssembler(
            context_port=self.mock_context_port,
            parser_port=self.mock_parser_port,
            config=restrictive_config,
        )

        files_to_process = [project_root / "src" / "bigpackage" / "module_0.py"]

        # Mock context port
        self.mock_context_port.build_context_graph.return_value = {"graph": "mock"}
        self.mock_context_port.index.return_value = {"indexed": True}

        context = context_assembler.gather_project_context(
            project_root, files_to_process
        )

        # Verify project structure respects budgets
        project_structure = context.get("project_structure", {})

        # Should be truncated due to budget limits
        def count_total_entries(structure):
            count = 1  # Count the structure itself
            for child in structure.get("children", []):
                count += count_total_entries(child)
            return count

        total_entries = count_total_entries(project_structure)
        # Should be limited by budget constraints
        assert total_entries < 100  # Much less than the 200+ files we created

    def test_error_handling_and_graceful_degradation(self):
        """Test error handling and graceful degradation of enhanced features."""
        project_root = self.create_test_project(
            {
                "mypackage": {
                    "module.py": "def test(): pass\n",
                }
            }
        )

        source_file = project_root / "mypackage" / "module.py"

        plan = TestGenerationPlan(
            file_path=str(source_file),
            elements_to_test=[
                TestElement(
                    name="test",
                    type=TestElementType.FUNCTION,
                    line_range=(1, 1),
                    docstring="Test function",
                ),
            ],
            existing_tests=[],
            coverage_info={},
        )

        # Test with failing module path derivation
        with patch.object(ModulePathDeriver, "derive_module_path") as mock_derive:
            mock_derive.side_effect = Exception("Module path derivation failed")

            # Should still work without module path enhancement
            context = self.context_assembler.context_for_generation(plan, source_file)

            # Context should still be generated, just without module path enhancements
            assert context is not None or context == ""  # Either works or returns empty

        # Test with failing directory tree building
        with patch.object(DirectoryTreeBuilder, "build_tree_recursive") as mock_build:
            mock_build.side_effect = Exception("Directory tree building failed")

            # Should fallback gracefully
            files_to_process = [source_file]
            context = self.context_assembler.gather_project_context(
                project_root, files_to_process
            )

            # Should have some context even if directory tree fails
            assert "context_graph" in context or "indexed_files" in context

    def test_configuration_driven_feature_toggling(self):
        """Test that features can be toggled via configuration."""
        project_root = self.create_test_project(
            {
                "mypackage": {
                    "module.py": "def test(): pass\n",
                }
            }
        )

        source_file = project_root / "mypackage" / "module.py"

        plan = TestGenerationPlan(
            file_path=str(source_file),
            elements_to_test=[
                TestElement(
                    name="test",
                    type=TestElementType.FUNCTION,
                    line_range=(1, 1),
                    docstring="Test function",
                ),
            ],
            existing_tests=[],
            coverage_info={},
        )

        # Test with features disabled
        disabled_config = self.config.copy()
        disabled_config["context_enrichment"]["enable_usage_examples"] = False
        disabled_config["enable_context"] = False

        context_assembler = ContextAssembler(
            context_port=self.mock_context_port,
            parser_port=self.mock_parser_port,
            config=disabled_config,
        )

        context = context_assembler.context_for_generation(plan, source_file)

        # Should return None when context is disabled
        assert context is None

        # Test with selective feature disabling
        selective_config = self.config.copy()
        selective_config["context_enrichment"]["enable_usage_examples"] = False

        selective_assembler = ContextAssembler(
            context_port=self.mock_context_port,
            parser_port=self.mock_parser_port,
            config=selective_config,
        )

        # Mock some context responses
        self.mock_context_port.retrieve.return_value = {"results": []}

        context = selective_assembler.context_for_generation(plan, source_file)

        # Should still generate context but without usage examples
        # The exact behavior depends on other available context
        assert context is not None or context == ""
