"""
Tests for missing_symbols resolution loop.

This module tests the SymbolResolver and LLMOrchestrator integration,
simulating missing symbol discovery and successful re-plans as specified
in the context assembly specification.
"""

import json
from pathlib import Path
from unittest.mock import Mock

from testcraft.application.generate_usecase import GenerateUseCase
from testcraft.application.generation.services.llm_orchestrator import LLMOrchestrator
from testcraft.application.generation.services.symbol_resolver import SymbolResolver
from testcraft.domain.models import (
    Budget,
    ContextPack,
    Conventions,
    Focal,
    GwtSnippets,
    ImportMap,
    PropertyContext,
    Target,
    TestElementType,
)
from testcraft.ports.parser_port import ParserPort


class MockParserPort(ParserPort):
    """Mock parser port for testing."""

    def __init__(self, mock_elements=None, mock_source_content=None):
        self.mock_elements = mock_elements or []
        self.mock_source_content = mock_source_content or {}

    def parse_file(self, file_path: Path, language=None, **kwargs):
        return {
            "ast": Mock(),
            "elements": self.mock_elements,
            "imports": [],
            "language": "python",
            "parse_errors": [],
            "source_content": self.mock_source_content,
            "file_path": str(file_path),
            "source_lines": [""] * 100,  # Mock source lines
        }

    def extract_functions(self, file_path, include_private=False, **kwargs):
        return [
            elem
            for elem in self.mock_elements
            if getattr(elem, "type", None) == TestElementType.FUNCTION
        ]

    def extract_classes(self, file_path, include_abstract=True, **kwargs):
        return [
            elem
            for elem in self.mock_elements
            if getattr(elem, "type", None) == TestElementType.CLASS
        ]

    def extract_methods(self, file_path, class_name=None, **kwargs):
        # Filter by class_name if provided
        if class_name:
            return [
                elem
                for elem in self.mock_elements
                if getattr(elem, "type", None) == TestElementType.METHOD
                and getattr(elem, "name", "").startswith(f"{class_name}.")
            ]
        else:
            return [
                elem
                for elem in self.mock_elements
                if getattr(elem, "type", None) == TestElementType.METHOD
            ]

    def map_tests(self, source_elements, existing_tests=None, **kwargs):
        return {
            "test_mapping": {},
            "missing_tests": [],
            "coverage_gaps": [],
            "test_suggestions": [],
        }

    def analyze_dependencies(self, file_path, **kwargs):
        return {
            "imports": [],
            "dependencies": [],
            "internal_deps": [],
            "circular_deps": [],
        }


class MockLLMPort:
    """Mock LLM port for testing."""

    def __init__(self, responses=None):
        self.responses = responses or []

    def generate_tests(
        self, code_content, context=None, test_framework="pytest", **kwargs
    ):
        """Mock implementation of generate_tests method matching LLMPort interface."""
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, dict):
                # Return proper LLM response structure
                return Mock(choices=[Mock(message=Mock(content=json.dumps(response)))])
            else:
                return Mock(choices=[Mock(message=Mock(content=response))])
        # Default response for missing symbols scenario
        return Mock(
            choices=[
                Mock(
                    message=Mock(
                        content='{"plan": ["test_basic_case"], "missing_symbols": [], "import_line": "from test_module import TestClass"}'
                    )
                )
            ]
        )

    def analyze_code(self, code_content, analysis_type="comprehensive", **kwargs):
        """Mock analyze_code method."""
        return Mock(
            choices=[
                Mock(
                    message=Mock(
                        content='{"testability_score": 0.8, "complexity_metrics": {}, "recommendations": [], "potential_issues": []}'
                    )
                )
            ]
        )

    def refine_content(
        self, original_content, refinement_instructions, *, system_prompt=None, **kwargs
    ):
        """Mock refine_content method."""
        return Mock(
            choices=[
                Mock(
                    message=Mock(
                        content='{"refined_content": "refined code", "changes_made": "minor fixes", "confidence": 0.9}'
                    )
                )
            ]
        )

    def generate_test_plan(self, code_content, context=None, **kwargs):
        """Mock generate_test_plan method."""
        return Mock(
            choices=[
                Mock(
                    message=Mock(
                        content='{"test_plan": {"unit_tests": []}, "test_coverage_areas": [], "test_priorities": [], "estimated_complexity": "low", "confidence": 0.8, "metadata": {}}'
                    )
                )
            ]
        )


class TestSymbolResolver:
    """Test the SymbolResolver service."""

    def test_resolve_single_symbol_function(self, tmp_path):
        """Test resolving a single function symbol."""
        # Create test module file
        module_file = tmp_path / "test_module.py"
        module_file.write_text('''
def test_function(param1: str, param2: int) -> bool:
    """Test function docstring."""
    return len(param1) > param2

class TestClass:
    def test_method(self, x: int) -> str:
        """Test method docstring."""
        return str(x)
''')

        # Mock parser elements - use real TestElement objects
        from testcraft.domain.models import TestElement

        mock_elements = [
            TestElement(
                name="test_function",
                type=TestElementType.FUNCTION,
                line_range=(1, 5),
                docstring="Test function docstring.",
            ),
            TestElement(
                name="TestClass",
                type=TestElementType.CLASS,
                line_range=(7, 10),
                docstring=None,
            ),
            TestElement(
                name="TestClass.test_method",
                type=TestElementType.METHOD,
                line_range=(11, 15),
                docstring="Test method docstring.",
            ),
        ]

        mock_source_content = {
            "test_function": "def test_function(param1: str, param2: int) -> bool:\n    return len(param1) > param2",
            "TestClass": "class TestClass:\n    pass",
            "TestClass.test_method": "def test_method(self, x: int) -> str:\n        return str(x)",
        }

        parser_port = MockParserPort(mock_elements, mock_source_content)

        resolver = SymbolResolver(parser_port)
        result = resolver.resolve_single_symbol("test_module.test_function", tmp_path)

        assert result is not None
        assert result.name == "test_module.test_function"
        assert result.kind == "func"
        assert (
            "def test_function(...):" in result.signature
        )  # Simplified signature format
        assert result.doc == "Test function docstring."
        assert "omitted" in result.body  # Body should be omitted

    def test_resolve_single_symbol_method(self, tmp_path):
        """Test resolving a single method symbol."""
        # Create test module file
        module_file = tmp_path / "test_module.py"
        module_file.write_text('''
class TestClass:
    def test_method(self, x: int) -> str:
        """Test method docstring."""
        return str(x)
''')

        # Mock parser elements using real TestElement objects
        from testcraft.domain.models import TestElement

        mock_elements = [
            TestElement(
                name="TestClass",
                type=TestElementType.CLASS,
                line_range=(1, 5),
                docstring=None,
            ),
            TestElement(
                name="TestClass.test_method",
                type=TestElementType.METHOD,
                line_range=(2, 5),
                docstring="Test method docstring.",
            ),
        ]

        mock_source_content = {
            "TestClass.test_method": "def test_method(self, x: int) -> str:\n        return str(x)",
        }

        parser_port = MockParserPort(mock_elements, mock_source_content)

        resolver = SymbolResolver(parser_port)
        result = resolver.resolve_single_symbol(
            "test_module.TestClass.test_method", tmp_path
        )

        assert result is not None
        assert result.name == "test_module.TestClass.test_method"
        assert result.kind == "func"
        assert (
            "def test_method(self, ...):" in result.signature
        )  # Simplified signature format
        assert result.doc == "Test method docstring."

    def test_resolve_symbol_not_found(self, tmp_path):
        """Test resolving a symbol that doesn't exist."""
        parser_port = MockParserPort()
        resolver = SymbolResolver(parser_port)

        result = resolver.resolve_single_symbol("nonexistent.module", tmp_path)
        assert result is None

    def test_resolve_symbols_multiple(self, tmp_path):
        """Test resolving multiple symbols."""
        # Create test module file
        module_file = tmp_path / "test_module.py"
        module_file.write_text("""
def func1():
    pass

def func2():
    pass
""")

        # Mock parser elements using real TestElement objects
        from testcraft.domain.models import TestElement

        mock_elements = [
            TestElement(
                name="func1",
                type=TestElementType.FUNCTION,
                line_range=(1, 3),
                docstring=None,
            ),
            TestElement(
                name="func2",
                type=TestElementType.FUNCTION,
                line_range=(4, 6),
                docstring=None,
            ),
        ]

        mock_source_content = {
            "func1": "def func1():\n    pass",
            "func2": "def func2():\n    pass",
        }

        parser_port = MockParserPort(mock_elements, mock_source_content)

        resolver = SymbolResolver(parser_port)
        results = resolver.resolve_symbols(
            ["test_module.func1", "test_module.func2"], tmp_path
        )

        assert len(results) == 2
        assert results[0].name == "test_module.func1"
        assert results[1].name == "test_module.func2"

    def test_symbol_caching(self, tmp_path):
        """Test that symbols are cached after first resolution."""
        parser_port = MockParserPort()
        resolver = SymbolResolver(parser_port)

        # First call should cache None result
        result1 = resolver.resolve_single_symbol("nonexistent.symbol", tmp_path)
        assert result1 is None

        # Second call should use cache
        result2 = resolver.resolve_single_symbol("nonexistent.symbol", tmp_path)
        assert result2 is None

        # Cache should contain the symbol
        assert "nonexistent.symbol" in resolver._cache

    def test_parse_multipart_symbol_name(self, tmp_path):
        """Test parsing of symbol names with 4+ parts (bug fix verification)."""
        parser_port = MockParserPort()
        resolver = SymbolResolver(parser_port)

        # Test 4-part symbol name: "pkg.subpkg.Class.method"
        module, class_name, symbol = resolver._parse_symbol_name(
            "pkg.subpkg.Class.method"
        )
        assert module == "pkg.subpkg"
        assert class_name == "Class"
        assert symbol == "method"

        # Test 5-part symbol name: "a.b.c.Class.method"
        module, class_name, symbol = resolver._parse_symbol_name("a.b.c.Class.method")
        assert module == "a.b.c"
        assert class_name == "Class"
        assert symbol == "method"

    def test_resolve_dotted_module_path(self, tmp_path):
        """Test resolving dotted module paths to file paths (bug fix verification)."""
        # Create nested module structure: testcraft/domain/models.py
        nested_dir = tmp_path / "testcraft" / "domain"
        nested_dir.mkdir(parents=True)
        module_file = nested_dir / "models.py"
        module_file.write_text("def test_func():\n    pass")

        parser_port = MockParserPort()
        resolver = SymbolResolver(parser_port)

        # Should correctly convert "testcraft.domain.models" to "testcraft/domain/models.py"
        result_path = resolver._find_module_file("testcraft.domain.models", tmp_path)
        assert result_path is not None
        assert result_path == module_file

    def test_element_matching_fallback(self, tmp_path):
        """Test defensive element matching with fallback (bug fix verification)."""
        from testcraft.domain.models import TestElement

        # Test case 1: Element without class prefix in name (fallback scenario)
        element_without_prefix = TestElement(
            name="test_method",  # Just method name, no class prefix
            type=TestElementType.METHOD,
            line_range=(1, 5),
            docstring="Test method",
        )

        parser_port = MockParserPort()
        resolver = SymbolResolver(parser_port)

        # Should match using fallback logic
        assert resolver._matches_symbol(
            element_without_prefix, "module", "TestClass", "test_method"
        )

        # Test case 2: Element with class prefix in name (exact match)
        element_with_prefix = TestElement(
            name="TestClass.test_method",
            type=TestElementType.METHOD,
            line_range=(1, 5),
            docstring="Test method",
        )

        # Should match using exact match logic
        assert resolver._matches_symbol(
            element_with_prefix, "module", "TestClass", "test_method"
        )

        # Test case 3: Element with missing attributes (defensive check)
        class IncompleteElement:
            """Mock element missing required attributes."""

            pass

        incomplete_element = IncompleteElement()
        # Should return False and not crash
        assert not resolver._matches_symbol(
            incomplete_element, "module", "TestClass", "test_method"
        )


class TestLLMOrchestrator:
    """Test the LLMOrchestrator with symbol resolution."""

    def test_plan_stage_with_missing_symbols(self, tmp_path):
        """Test PLAN stage that encounters missing symbols and resolves them."""
        # Create test module file
        module_file = tmp_path / "test_module.py"
        module_file.write_text('''
def helper_function():
    """Helper function for testing."""
    return "help"

class TestClass:
    def test_method(self):
        pass
''')

        # Mock parser elements for helper function
        from testcraft.domain.models import TestElement

        helper_elements = [
            TestElement(
                name="helper_function",
                type=TestElementType.FUNCTION,
                line_range=(2, 5),
                docstring="Helper function for testing.",
            ),
        ]

        helper_source_content = {
            "helper_function": 'def helper_function():\n    return "help"',
        }

        parser_port = MockParserPort(helper_elements, helper_source_content)

        # Mock LLM responses: first with missing symbols, then successful plan
        llm_responses = [
            {
                "plan": ["test_helper_function", "test_with_helper"],
                "missing_symbols": ["test_module.helper_function"],
                "import_line": "from test_module import TestClass",
            },
            {
                "plan": ["test_helper_function", "test_with_helper"],
                "missing_symbols": [],  # No missing symbols on retry
                "import_line": "from test_module import TestClass",
            },
        ]

        llm_port = MockLLMPort(llm_responses)

        # Mock context assembler
        context_assembler = Mock()
        context_assembler.gather_project_context.return_value = {}

        # Create orchestrator
        orchestrator = LLMOrchestrator(
            llm_port=llm_port,
            parser_port=parser_port,
            context_assembler=context_assembler,
        )

        # Create a simple context pack
        context_pack = ContextPack(
            target=Target(module_file=str(module_file), object="TestClass.test_method"),
            import_map=ImportMap(
                target_import="from test_module import TestClass",
                sys_path_roots=[str(tmp_path)],
                needs_bootstrap=False,
                bootstrap_conftest="",
            ),
            focal=Focal(
                source="def test_method(self):\n    pass",
                signature="def test_method(self):",
                docstring=None,
            ),
            resolved_defs=[],
            property_context=PropertyContext(
                ranked_methods=[],
                gwt_snippets=GwtSnippets(given=[], when=[], then=[]),
                test_bundles=[],
            ),
            conventions=Conventions(),
            budget=Budget(),
        )

        # Execute plan stage
        plan = orchestrator.plan_stage(context_pack, tmp_path)

        assert plan is not None
        assert plan["plan"] == ["test_helper_function", "test_with_helper"]
        assert plan["missing_symbols"] == []

    def test_plan_stage_max_retries_exceeded(self, tmp_path):
        """Test PLAN stage when maximum retries are exceeded."""
        # Create test module file
        module_file = tmp_path / "test_module.py"
        module_file.write_text("def test_func():\n    pass")

        # Create a MockParserPort with no elements so missing symbols can't be resolved
        parser_port = MockParserPort(mock_elements=[], mock_source_content={})

        # Mock LLM responses: always return missing symbols to trigger retries
        llm_responses = [
            {
                "plan": ["test_case"],
                "missing_symbols": ["missing.symbol"],
                "import_line": "from test_module import test_func",
            },
            {
                "plan": ["test_case"],
                "missing_symbols": ["missing.symbol"],
                "import_line": "from test_module import test_func",
            },
            {
                "plan": ["test_case"],
                "missing_symbols": ["missing.symbol"],
                "import_line": "from test_module import test_func",
            },  # This will exceed max retries
        ]

        llm_port = MockLLMPort(llm_responses)

        context_assembler = Mock()
        context_assembler.gather_project_context.return_value = {}

        from testcraft.config.models import OrchestratorConfig

        orchestrator = LLMOrchestrator(
            llm_port=llm_port,
            parser_port=parser_port,
            context_assembler=context_assembler,
            config=OrchestratorConfig(max_plan_retries=2),  # Only 2 retries allowed
        )

        context_pack = ContextPack(
            target=Target(module_file=str(module_file), object="test_func"),
            import_map=ImportMap(
                target_import="from test_module import test_func",
                sys_path_roots=[str(tmp_path)],
                needs_bootstrap=False,
                bootstrap_conftest="",
            ),
            focal=Focal(
                source="def test_func():\n    pass",
                signature="def test_func():",
                docstring=None,
            ),
            resolved_defs=[],
            property_context=PropertyContext(
                ranked_methods=[],
                gwt_snippets=GwtSnippets(given=[], when=[], then=[]),
                test_bundles=[],
            ),
            conventions=Conventions(),
            budget=Budget(),
        )

        # Execute plan stage - should return None due to exceeded retries
        plan = orchestrator.plan_stage(context_pack, tmp_path)
        assert plan is None

    def test_refine_stage_with_missing_symbols(self, tmp_path):
        """Test REFINE stage that encounters missing symbols and resolves them."""
        # Create test module file
        module_file = tmp_path / "test_module.py"
        module_file.write_text('''
def helper_function():
    """Helper function for testing."""
    return "help"
''')

        # Mock parser elements for helper function
        helper_elements = [
            Mock(
                name="helper_function",
                type="function",
                docstring="Helper function for testing.",
            ),
        ]

        helper_source_content = {
            "helper_function": 'def helper_function():\n    return "help"',
        }

        parser_port = MockParserPort(helper_elements, helper_source_content)

        # Mock LLM responses: first with missing symbols, then successful refinement
        existing_code = """
def test_something():
    result = helper_function()
    assert result == "help"
"""

        feedback = {
            "result": "fail",
            "trace_excerpt": "NameError: name 'helper_function' is not defined",
            "coverage": {},
            "mutants_survived": [],
        }

        llm_responses = [
            {
                "missing_symbols": ["test_module.helper_function"],
                "refined_code": existing_code,  # This will be ignored due to missing symbols
            },
            'def test_something():\n    result = helper_function()\n    assert result == "help"',  # Successful refinement
        ]

        llm_port = MockLLMPort(llm_responses)

        context_assembler = Mock()
        context_assembler.gather_project_context.return_value = {}

        orchestrator = LLMOrchestrator(
            llm_port=llm_port,
            parser_port=parser_port,
            context_assembler=context_assembler,
        )

        context_pack = ContextPack(
            target=Target(module_file=str(module_file), object="test_something"),
            import_map=ImportMap(
                target_import="from test_module import helper_function",
                sys_path_roots=[str(tmp_path)],
                needs_bootstrap=False,
                bootstrap_conftest="",
            ),
            focal=Focal(
                source="def test_something():\n    pass",
                signature="def test_something():",
                docstring=None,
            ),
            resolved_defs=[],
            property_context=PropertyContext(
                ranked_methods=[],
                gwt_snippets=GwtSnippets(given=[], when=[], then=[]),
                test_bundles=[],
            ),
            conventions=Conventions(),
            budget=Budget(),
        )

        # Execute refine stage
        refined_code = orchestrator.refine_stage(
            context_pack, existing_code, feedback, tmp_path
        )

        assert refined_code is not None
        assert "def test_something():" in refined_code
        assert "result = helper_function()" in refined_code

    def test_refine_stage_max_retries_exceeded(self, tmp_path):
        """Test REFINE stage when maximum retries are exceeded."""
        # Create test module file
        module_file = tmp_path / "test_module.py"
        module_file.write_text("def test_func():\n    pass")

        parser_port = MockParserPort()

        existing_code = "def test_something():\n    assert True"

        feedback = {
            "result": "fail",
            "trace_excerpt": "error",
            "coverage": {},
            "mutants_survived": [],
        }

        # Mock LLM responses: always return missing symbols to trigger retries
        llm_responses = [
            {"missing_symbols": ["missing.symbol"]},
            {"missing_symbols": ["missing.symbol"]},
            {"missing_symbols": ["missing.symbol"]},
            {"missing_symbols": ["missing.symbol"]},  # This will exceed max retries (3)
        ]

        llm_port = MockLLMPort(llm_responses)

        context_assembler = Mock()
        context_assembler.gather_project_context.return_value = {}

        from testcraft.config.models import OrchestratorConfig

        orchestrator = LLMOrchestrator(
            llm_port=llm_port,
            parser_port=parser_port,
            context_assembler=context_assembler,
            config=OrchestratorConfig(max_refine_retries=3),  # Only 3 retries allowed
        )

        context_pack = ContextPack(
            target=Target(module_file=str(module_file), object="test_something"),
            import_map=ImportMap(
                target_import="from test_module import test_func",
                sys_path_roots=[str(tmp_path)],
                needs_bootstrap=False,
                bootstrap_conftest="",
            ),
            focal=Focal(
                source="def test_something():\n    pass",
                signature="def test_something():",
                docstring=None,
            ),
            resolved_defs=[],
            property_context=PropertyContext(
                ranked_methods=[],
                gwt_snippets=GwtSnippets(given=[], when=[], then=[]),
                test_bundles=[],
            ),
            conventions=Conventions(),
            budget=Budget(),
        )

        # Execute refine stage - should return original code due to exceeded retries
        refined_code = orchestrator.refine_stage(
            context_pack, existing_code, feedback, tmp_path
        )
        assert refined_code == existing_code

    def test_full_plan_and_generate_workflow(self, tmp_path):
        """Test the complete PLAN/GENERATE workflow."""
        # Create test module file
        module_file = tmp_path / "test_module.py"
        module_file.write_text('''
def test_function():
    """Test function docstring."""
    return "test"
''')

        parser_port = MockParserPort()

        # Mock LLM responses
        llm_responses = [
            {
                "plan": ["test_basic_case", "test_edge_cases"],
                "missing_symbols": [],  # No missing symbols
                "import_line": "from test_module import test_function",
            },
            "def test_basic_case():\n    result = test_function()\n    assert result == 'test'\n\ndef test_edge_cases():\n    assert True",
        ]

        llm_port = MockLLMPort(llm_responses)

        context_assembler = Mock()
        context_assembler.gather_project_context.return_value = {}

        orchestrator = LLMOrchestrator(
            llm_port=llm_port,
            parser_port=parser_port,
            context_assembler=context_assembler,
        )

        context_pack = ContextPack(
            target=Target(module_file=str(module_file), object="test_function"),
            import_map=ImportMap(
                target_import="from test_module import test_function",
                sys_path_roots=[str(tmp_path)],
                needs_bootstrap=False,
                bootstrap_conftest="",
            ),
            focal=Focal(
                source="def test_function():\n    return 'test'",
                signature="def test_function():",
                docstring="Test function docstring.",
            ),
            resolved_defs=[],
            property_context=PropertyContext(
                ranked_methods=[],
                gwt_snippets=GwtSnippets(given=[], when=[], then=[]),
                test_bundles=[],
            ),
            conventions=Conventions(),
            budget=Budget(),
        )

        # Execute full workflow
        result = orchestrator.plan_and_generate(context_pack, tmp_path)

        assert result["plan"]["plan"] == ["test_basic_case", "test_edge_cases"]
        assert "def test_basic_case():" in result["generated_code"]
        assert "def test_edge_cases():" in result["generated_code"]

    def test_generate_usecase_integration_with_symbol_resolution(self, tmp_path):
        """Test that GenerateUseCase properly initializes symbol resolution components."""
        from unittest.mock import Mock

        # Mock all the dependencies
        mock_llm = Mock()
        mock_writer = Mock()
        mock_coverage = Mock()
        mock_refine = Mock()
        mock_context = Mock()
        mock_parser = Mock()
        mock_state = Mock()
        mock_telemetry = Mock()

        # Create GenerateUseCase with symbol resolution enabled
        config = {
            "enable_symbol_resolution": True,
            "max_plan_retries": 2,
            "max_refine_retries": 3,
            "batch_size": 5,
            "test_framework": "pytest",
        }

        usecase = GenerateUseCase(
            llm_port=mock_llm,
            writer_port=mock_writer,
            coverage_port=mock_coverage,
            refine_port=mock_refine,
            context_port=mock_context,
            parser_port=mock_parser,
            state_port=mock_state,
            telemetry_port=mock_telemetry,
            config=config,
        )

        # Verify the integration components are properly initialized
        assert hasattr(usecase, "_symbol_resolver")
        assert hasattr(usecase, "_context_pack_builder")
        assert hasattr(usecase, "_llm_orchestrator")

        # Verify configuration is passed correctly
        assert usecase._config["enable_symbol_resolution"] is True
        assert usecase._config["max_plan_retries"] == 2
        assert usecase._config["max_refine_retries"] == 3

        # Verify the orchestrator is configured with the right parameters
        assert usecase._llm_orchestrator._max_plan_retries == 2
        assert usecase._llm_orchestrator._max_refine_retries == 3
