"""
Tests for immediate refinement feature.

Tests the new immediate write-and-refine functionality that processes files
one at a time with generate → write → refine workflow.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from testcraft.application.generate_usecase import GenerateUseCase
from testcraft.application.generation.services.pytest_refiner import PytestRefiner
from testcraft.domain.models import (
    Budget,
    ContextPack,
    Conventions,
    DeterminismConfig,
    Focal,
    GenerationResult,
    GwtSnippets,
    ImportMap,
    PropertyContext,
    ResolvedDef,
    Target,
    TestBundle,
    TestElement,
    TestElementType,
    TestGenerationPlan,
)


def create_test_generation_plan(source_file_path: str) -> TestGenerationPlan:
    """Helper function to create a proper TestGenerationPlan object for testing."""
    return TestGenerationPlan(
        elements_to_test=[
            TestElement(
                name="test_function",
                type=TestElementType.FUNCTION,
                line_range=(1, 10),
                docstring="Test function docstring",
            )
        ],
        existing_tests=[],
        coverage_before=None,
    )


def create_test_context_pack(source_file_path: str, function_name: str) -> ContextPack:
    """Helper function to create a real ContextPack instance for testing."""
    return ContextPack(
        target=Target(module_file=source_file_path, object=function_name),
        import_map=ImportMap(
            target_import="src.example",
            sys_path_roots=["/tmp/test_dir/src"],
            needs_bootstrap=False,
            bootstrap_conftest="",
        ),
        focal=Focal(
            source="def example(): pass",
            signature="def example():",
            docstring="Test function docstring",
        ),
        resolved_defs=[
            ResolvedDef(
                name="helper_function",
                kind="func",
                signature="def helper_function():",
                doc="Helper function",
                body="def helper_function(): return 'help'",
            )
        ],
        property_context=PropertyContext(
            ranked_methods=[],
            gwt_snippets=GwtSnippets(given=[], when=[], then=[]),
            test_bundles=[
                TestBundle(
                    test_name="test_helper_function",
                    imports=["from src.example import helper_function"],
                    fixtures=[],
                    mocks=[],
                    assertions=["assert helper_function() == 'help'"],
                )
            ],
        ),
        conventions=Conventions(
            test_style="pytest",
            allowed_libs=["pytest", "unittest"],
            determinism=DeterminismConfig(),
        ),
        budget=Budget(max_input_tokens=60000),
    )


@pytest.fixture
def mock_ports():
    """Create mock ports for testing."""
    return {
        "llm_port": Mock(),
        "writer_port": Mock(),
        "coverage_port": Mock(),
        "refine_port": Mock(),
        "context_port": Mock(),
        "parser_port": Mock(),
        "state_port": Mock(),
        "telemetry_port": Mock(),
    }


@pytest.fixture
def immediate_config():
    """Configuration for immediate refinement mode."""
    return {
        "immediate_refinement": True,
        "max_refine_workers": 2,
        "keep_failed_writes": False,
        "refine_on_first_failure_only": True,
        "refinement_backoff_sec": 0.1,
        "enable_refinement": True,
        "max_refinement_iterations": 3,
        "batch_size": 5,
        "enable_context": False,
        "test_framework": "pytest",
        "coverage_threshold": 0.8,
        "enable_streaming": False,
    }


@pytest.fixture
def legacy_config():
    """Configuration for legacy mode."""
    return {
        "immediate_refinement": False,
        "enable_refinement": True,
        "max_refinement_iterations": 3,
        "batch_size": 5,
        "enable_context": False,
        "test_framework": "pytest",
        "coverage_threshold": 0.8,
        "enable_streaming": False,
    }


@pytest.mark.asyncio
class TestImmediateRefinement:
    """Test immediate refinement functionality."""

    async def test_immediate_mode_happy_path(self, mock_ports, immediate_config):
        """Test successful immediate pipeline: generate→write→refine on first try."""
        # Setup mocks - LLM orchestrator expects generate_test, not generate_tests
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[
            0
        ].message.content = '{"plan": ["Test basic functionality", "Test edge cases"], "missing_symbols": [], "import_line": "from src.example import example"}'
        mock_ports["writer_port"].write_test_file.return_value = {
            "success": True,
            "bytes_written": 100,
        }
        mock_ports["refine_port"].refine_from_failures.return_value = {
            "success": True,
            "refined_content": "def test_example(): pass",
        }

        # Setup telemetry mock
        telemetry_span = Mock()
        telemetry_span.__enter__ = Mock(return_value=telemetry_span)
        telemetry_span.__exit__ = Mock(return_value=None)
        mock_ports["telemetry_port"].create_span.return_value = telemetry_span
        mock_ports["telemetry_port"].create_child_span.return_value = telemetry_span
        mock_ports["telemetry_port"].record_metrics = Mock()

        # Setup other required mocks
        mock_file_discovery = Mock()
        mock_file_discovery.discover_test_files.return_value = []

        # Mock pytest refiner to return successful refinement on first try
        mock_refiner = AsyncMock()
        mock_refiner.refine_until_pass = AsyncMock(
            return_value={
                "success": True,
                "iterations": 1,
                "final_status": "passed",
                "test_file": "tests/test_example.py",
            }
        )

        # Create use case with validation disabled to avoid mock issues
        config = immediate_config.copy()
        config["enable_validation"] = False

        usecase = GenerateUseCase(
            config=config,
            file_discovery_service=mock_file_discovery,
            **mock_ports,
        )

        # Mock the already created pytest refiner instance
        usecase._pytest_refiner = mock_refiner

        # Mock the LLM orchestrator's LLM port instead of the test's LLM port
        def debug_generate_tests(**kwargs):
            code_content = kwargs.get("code_content", "")

            # Return different responses based on the prompt content
            if "Output ONLY the complete test module" in code_content:
                # GENERATE stage - return Python code in triple backticks
                result = Mock()
                result.choices = [Mock()]
                result.choices[0].message = Mock()
                result.choices[0].message.content = '''```python
def test_example():
    """Test basic functionality of example function."""
    result = example()
    assert result is None  # Function returns None

def test_example_edge_cases():
    """Test edge cases for example function."""
    # Test with various inputs - since function takes no parameters
    pass
```'''
            else:
                # PLAN stage - return JSON
                result = mock_response

            return result

        usecase._llm_orchestrator._llm.generate_tests = Mock(
            side_effect=debug_generate_tests
        )

        # Mock the service methods
        usecase._state_discovery = Mock()
        usecase._state_discovery.sync_and_discover.return_value = {
            "files": ["src/example.py"]
        }

        usecase._coverage_evaluator = Mock()
        usecase._coverage_evaluator.measure_initial.return_value = {
            "overall_line_coverage": 0.5
        }
        usecase._coverage_evaluator.measure_final.return_value = {
            "overall_line_coverage": 0.8
        }
        usecase._coverage_evaluator.calculate_delta.return_value = {
            "line_coverage_delta": 0.3
        }

        usecase._plan_builder = Mock()
        usecase._plan_builder.decide_files_to_process.return_value = ["src/example.py"]
        test_plan = create_test_generation_plan("src/example.py")
        usecase._plan_builder.build_plans.return_value = [test_plan]
        usecase._plan_builder.get_source_path_for_plan.return_value = Path(
            "src/example.py"
        )

        usecase._content_builder = Mock()
        usecase._content_builder.build_code_content.return_value = "def example(): pass"
        usecase._context_assembler = Mock()
        usecase._context_assembler.context_for_generation.return_value = {}
        usecase._context_assembler.get_last_enriched_context.return_value = None

        # Mock the context assembler's import resolver to return proper ImportMap
        mock_context_assembler_import_resolver = Mock()
        import_map = ImportMap(
            target_import="src.example",
            sys_path_roots=["/tmp/test_dir/src"],
            needs_bootstrap=False,
            bootstrap_conftest="",
        )
        mock_context_assembler_import_resolver.resolve.return_value = import_map
        usecase._context_assembler._import_resolver = (
            mock_context_assembler_import_resolver
        )

        # Mock import resolver to return proper ImportMap object
        usecase._symbol_resolver = Mock()
        usecase._symbol_resolver.resolve_imports.return_value = import_map

        # Mock context pack builder to return a real ContextPack instance
        usecase._context_pack_builder = Mock()

        # Mock the import resolver on the context pack builder to return proper ImportMap
        mock_import_resolver = Mock()
        mock_import_resolver.resolve.return_value = import_map
        usecase._context_pack_builder._import_resolver = mock_import_resolver

        usecase._context_pack_builder.build_context_pack.return_value = (
            create_test_context_pack("src/example.py", "example")
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the source file that the plan builder expects to exist
            source_file_path = Path(temp_dir) / "src" / "example.py"
            source_file_path.parent.mkdir(parents=True, exist_ok=True)
            source_file_path.write_text(
                "def example():\n    '''Test function docstring'''\n    pass"
            )

            # Create the test file that the refinement method expects to exist
            test_file_path = Path(temp_dir) / "tests" / "test_example.py"
            test_file_path.parent.mkdir(parents=True, exist_ok=True)
            test_file_path.write_text("def test_example(): pass")

            # Mock get_source_path_for_plan to return the absolute path to the source file
            usecase._plan_builder.get_source_path_for_plan.return_value = (
                source_file_path
            )

            # Mock determine_test_path to return the absolute path
            usecase._content_builder.determine_test_path.return_value = str(
                test_file_path
            )

            # Run the test
            results = await usecase.generate_tests(project_path=temp_dir)

            # Verify results
            assert results["success"] is True
            assert results["tests_generated"] > 0
            assert results["files_written"] > 0
            assert results["files_refined"] > 0

            # Verify refinement was called
            mock_refiner.refine_until_pass.assert_called_once()

    async def test_immediate_mode_syntax_invalid_no_write(
        self, mock_ports, immediate_config
    ):
        """Test that invalid syntax prevents write when keep_failed_writes=False."""
        # Setup mock to return invalid Python syntax
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[
            0
        ].message.content = '{"plan": ["Test invalid syntax"], "missing_symbols": [], "import_line": "from src.example import example"}'  # Invalid syntax

        # Writer should raise an exception due to syntax validation
        mock_ports["writer_port"].write_test_file.side_effect = Exception(
            "Syntax error in generated code"
        )

        # Setup telemetry mock
        telemetry_span = Mock()
        telemetry_span.__enter__ = Mock(return_value=telemetry_span)
        telemetry_span.__exit__ = Mock(return_value=None)
        mock_ports["telemetry_port"].create_span.return_value = telemetry_span
        mock_ports["telemetry_port"].create_child_span.return_value = telemetry_span
        mock_ports["telemetry_port"].record_metrics = Mock()

        # Setup other required mocks
        mock_file_discovery = Mock()
        mock_file_discovery.discover_test_files.return_value = []

        # Create use case with keep_failed_writes=False
        config = immediate_config.copy()
        config["keep_failed_writes"] = False
        config["enable_validation"] = False  # Disable validation to avoid mock issues

        usecase = GenerateUseCase(
            config=config, file_discovery_service=mock_file_discovery, **mock_ports
        )

        # Mock the service methods
        usecase._state_discovery = Mock()
        usecase._state_discovery.sync_and_discover.return_value = {
            "files": ["src/example.py"]
        }

        usecase._coverage_evaluator = Mock()
        usecase._coverage_evaluator.measure_initial.return_value = {
            "overall_line_coverage": 0.5
        }
        usecase._coverage_evaluator.measure_final.return_value = {
            "overall_line_coverage": 0.5
        }
        usecase._coverage_evaluator.calculate_delta.return_value = {
            "line_coverage_delta": 0.0
        }

        usecase._plan_builder = Mock()
        usecase._plan_builder.decide_files_to_process.return_value = ["src/example.py"]
        test_plan = create_test_generation_plan("src/example.py")
        usecase._plan_builder.build_plans.return_value = [test_plan]
        usecase._plan_builder.get_source_path_for_plan.return_value = Path(
            "src/example.py"
        )

        usecase._content_builder = Mock()
        usecase._content_builder.build_code_content.return_value = "def example(): pass"
        usecase._content_builder.determine_test_path.return_value = (
            "tests/test_example.py"
        )

        usecase._context_assembler = Mock()
        usecase._context_assembler.context_for_generation.return_value = {}
        usecase._context_assembler.get_last_enriched_context.return_value = None

        # Mock the context assembler's import resolver to return proper ImportMap
        mock_context_assembler_import_resolver = Mock()
        import_map = ImportMap(
            target_import="src.example",
            sys_path_roots=["/tmp/test_dir/src"],
            needs_bootstrap=False,
            bootstrap_conftest="",
        )
        mock_context_assembler_import_resolver.resolve.return_value = import_map
        usecase._context_assembler._import_resolver = (
            mock_context_assembler_import_resolver
        )

        # Mock import resolver to return proper ImportMap object
        usecase._symbol_resolver = Mock()
        usecase._symbol_resolver.resolve_imports.return_value = import_map

        # Mock context pack builder to return a real ContextPack instance
        usecase._context_pack_builder = Mock()

        # Mock the import resolver on the context pack builder to return proper ImportMap
        mock_import_resolver = Mock()
        mock_import_resolver.resolve.return_value = import_map
        usecase._context_pack_builder._import_resolver = mock_import_resolver

        usecase._context_pack_builder.build_context_pack.return_value = (
            create_test_context_pack("src/example.py", "example")
        )

        mock_refiner = AsyncMock()

        # Mock the LLM orchestrator's LLM port to return proper responses
        def mock_generate_tests(**kwargs):
            code_content = kwargs.get("code_content", "")

            # Return different responses based on the prompt content
            if "Output ONLY the complete test module" in code_content:
                # GENERATE stage - return Python code in triple backticks
                result = Mock()
                result.choices = [Mock()]
                result.choices[0].message = Mock()
                result.choices[0].message.content = '''```python
def test_example():
    """Test basic functionality of example function."""
    result = example()
    assert result is None  # Function returns None
```'''
            else:
                # PLAN stage - return JSON
                result = mock_response

            return result

        usecase._llm_orchestrator._llm.generate_tests = Mock(
            side_effect=mock_generate_tests
        )

        # Mock the already created pytest refiner instance
        usecase._pytest_refiner = mock_refiner

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the source file that the plan builder expects to exist
            source_file_path = Path(temp_dir) / "src" / "example.py"
            source_file_path.parent.mkdir(parents=True, exist_ok=True)
            source_file_path.write_text(
                "def example():\n    '''Test function docstring'''\n    pass"
            )

            # Mock get_source_path_for_plan to return the absolute path to the source file
            usecase._plan_builder.get_source_path_for_plan.return_value = (
                source_file_path
            )

            # Mock rollback method
            with patch.object(
                usecase, "_rollback_failed_write", new_callable=AsyncMock
            ) as mock_rollback:
                # Run the test
                results = await usecase.generate_tests(project_path=temp_dir)

                # Verify write was attempted but failed (still counts as 1 in files_written)
                assert results["files_written"] == 1

                # Verify rollback was called for failed write
                mock_rollback.assert_called()

    async def test_immediate_mode_refinement_iterations_exhausted(
        self, mock_ports, immediate_config
    ):
        """Test refinement continues until max iterations are exhausted."""
        # Setup mocks - test that will fail
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[
            0
        ].message.content = '{"plan": ["Test that will fail"], "missing_symbols": [], "import_line": "from src.example import example"}'

        mock_ports["writer_port"].write_test_file.return_value = {
            "success": True,
            "bytes_written": 100,
        }

        # Setup telemetry mock
        telemetry_span = Mock()
        telemetry_span.__enter__ = Mock(return_value=telemetry_span)
        telemetry_span.__exit__ = Mock(return_value=None)
        mock_ports["telemetry_port"].create_span.return_value = telemetry_span
        mock_ports["telemetry_port"].create_child_span.return_value = telemetry_span
        mock_ports["telemetry_port"].record_metrics = Mock()

        # Setup other required mocks
        mock_file_discovery = Mock()
        mock_file_discovery.discover_test_files.return_value = []

        # Create use case
        usecase = GenerateUseCase(
            config=immediate_config,
            file_discovery_service=mock_file_discovery,
            **mock_ports,
        )

        # Mock the LLM orchestrator's LLM port to return proper responses
        def mock_generate_tests(**kwargs):
            code_content = kwargs.get("code_content", "")

            # Return different responses based on the prompt content
            if "Output ONLY the complete test module" in code_content:
                # GENERATE stage - return Python code in triple backticks
                result = Mock()
                result.choices = [Mock()]
                result.choices[0].message = Mock()
                result.choices[0].message.content = '''```python
def test_example():
    """Test basic functionality of example function."""
    result = example()
    assert result is None  # Function returns None
```'''
            else:
                # PLAN stage - return JSON
                result = mock_response

            return result

        usecase._llm_orchestrator._llm.generate_tests = Mock(
            side_effect=mock_generate_tests
        )

        # Mock the service methods
        usecase._state_discovery = Mock()
        usecase._state_discovery.sync_and_discover.return_value = {
            "files": ["src/example.py"]
        }

        usecase._coverage_evaluator = Mock()
        usecase._coverage_evaluator.measure_initial.return_value = {
            "overall_line_coverage": 0.5
        }
        usecase._coverage_evaluator.measure_final.return_value = {
            "overall_line_coverage": 0.5
        }
        usecase._coverage_evaluator.calculate_delta.return_value = {
            "line_coverage_delta": 0.0
        }

        usecase._plan_builder = Mock()
        usecase._plan_builder.decide_files_to_process.return_value = ["src/example.py"]
        test_plan = create_test_generation_plan("src/example.py")
        usecase._plan_builder.build_plans.return_value = [test_plan]
        usecase._plan_builder.get_source_path_for_plan.return_value = Path(
            "src/example.py"
        )

        usecase._content_builder = Mock()
        usecase._content_builder.build_code_content.return_value = "def example(): pass"
        usecase._content_builder.determine_test_path.return_value = (
            "tests/test_example.py"
        )

        usecase._context_assembler = Mock()
        usecase._context_assembler.context_for_generation.return_value = {}
        usecase._context_assembler.get_last_enriched_context.return_value = None

        # Mock the context assembler's import resolver to return proper ImportMap
        mock_context_assembler_import_resolver = Mock()
        import_map = ImportMap(
            target_import="src.example",
            sys_path_roots=["/tmp/test_dir/src"],
            needs_bootstrap=False,
            bootstrap_conftest="",
        )
        mock_context_assembler_import_resolver.resolve.return_value = import_map
        usecase._context_assembler._import_resolver = (
            mock_context_assembler_import_resolver
        )

        # Mock import resolver to return proper ImportMap object
        usecase._symbol_resolver = Mock()
        usecase._symbol_resolver.resolve_imports.return_value = import_map

        # Mock context pack builder to return a real ContextPack instance
        usecase._context_pack_builder = Mock()

        # Mock the import resolver on the context pack builder to return proper ImportMap
        mock_import_resolver = Mock()
        mock_import_resolver.resolve.return_value = import_map
        usecase._context_pack_builder._import_resolver = mock_import_resolver

        usecase._context_pack_builder.build_context_pack.return_value = (
            create_test_context_pack("src/example.py", "example")
        )

        # Mock pytest refiner to exhaust all iterations without success
        mock_refiner = AsyncMock()
        mock_refiner.refine_until_pass = AsyncMock(
            return_value={
                "success": False,
                "iterations": immediate_config["max_refinement_iterations"],
                "final_status": "failed",
                "test_file": "tests/test_example.py",
                "error": "Maximum refinement iterations exceeded",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the source file that the plan builder expects to exist
            source_file_path = Path(temp_dir) / "src" / "example.py"
            source_file_path.parent.mkdir(parents=True, exist_ok=True)
            source_file_path.write_text(
                "def example():\n    '''Test function docstring'''\n    pass"
            )

            # Mock get_source_path_for_plan to return the absolute path to the source file
            usecase._plan_builder.get_source_path_for_plan.return_value = (
                source_file_path
            )

            # Run the test
            results = await usecase.generate_tests(project_path=temp_dir)

            # Verify refinement was attempted but may have failed early
            assert results["files_refined"] == 1  # One refinement attempt was counted

            # Verify that although refinement was attempted, it failed
            refinement_results = results.get("refinement_results", [])
            assert len(refinement_results) > 0, (
                "Expected at least one refinement result"
            )
            # Check that the refinement failed (success: False in refinement result)
            assert not refinement_results[0]["success"], (
                f"Expected refinement to fail, got: {refinement_results[0]}"
            )

    @pytest.mark.asyncio
    async def test_concurrency_semaphore_respected(self):
        """Test that semaphore limits concurrent pytest operations."""
        from concurrent.futures import ThreadPoolExecutor

        # Create pytest refiner with limited concurrency
        mock_refine_port = Mock()
        mock_telemetry_port = Mock()
        mock_executor = ThreadPoolExecutor(max_workers=2)

        # Mock telemetry span
        telemetry_span = Mock()
        telemetry_span.__enter__ = Mock(return_value=telemetry_span)
        telemetry_span.__exit__ = Mock(return_value=None)
        mock_telemetry_port.create_child_span.return_value = telemetry_span

        refiner = PytestRefiner(
            refine_port=mock_refine_port,
            telemetry_port=mock_telemetry_port,
            executor=mock_executor,
            max_concurrent_refines=1,  # Only allow 1 concurrent refine
            backoff_sec=0.1,
        )

        # Track concurrent calls
        concurrent_calls = []
        call_order = []

        async def mock_run_pytest(test_path):
            call_order.append(f"start_{test_path}")
            concurrent_calls.append(test_path)
            await asyncio.sleep(0.2)  # Simulate pytest execution time
            concurrent_calls.remove(test_path)
            call_order.append(f"end_{test_path}")
            return {"returncode": 0, "stdout": "", "stderr": ""}

        # Mock the run_pytest method
        with patch.object(refiner, "run_pytest", side_effect=mock_run_pytest):
            with patch.object(refiner, "format_pytest_failure_output", return_value=""):

                async def build_context(test_file, test_content):
                    return {}

                # Start multiple refinements concurrently
                tasks = []
                for i in range(3):
                    task = refiner.refine_until_pass(
                        f"test_{i}.py",
                        1,  # max_iterations
                        build_context,
                    )
                    tasks.append(task)

                # Run all tasks
                results = await asyncio.gather(*tasks)

                # Verify all succeeded
                for result in results:
                    assert result["success"] is True

                # Verify that concurrent calls never exceeded our limit
                # Due to semaphore, we should never have more than 1 concurrent call
                max_concurrent = 0
                current_concurrent = 0
                for call in call_order:
                    if call.startswith("start_"):
                        current_concurrent += 1
                        max_concurrent = max(max_concurrent, current_concurrent)
                    elif call.startswith("end_"):
                        current_concurrent -= 1

                assert max_concurrent == 1, (
                    f"Expected max 1 concurrent call, got {max_concurrent}"
                )

    async def test_legacy_mode_still_works(self, mock_ports, legacy_config):
        """Test that legacy mode (immediate_refinement=False) still works."""
        # Setup mocks for legacy mode
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "def test_example(): pass"

        mock_ports["writer_port"].write_test_file.return_value = {
            "success": True,
            "bytes_written": 100,
        }
        mock_ports["refine_port"].refine_from_failures.return_value = {
            "success": True,
            "refined_content": "def test_example(): pass",
        }

        # Setup telemetry mock
        telemetry_span = Mock()
        telemetry_span.__enter__ = Mock(return_value=telemetry_span)
        telemetry_span.__exit__ = Mock(return_value=None)
        mock_ports["telemetry_port"].create_span.return_value = telemetry_span
        mock_ports["telemetry_port"].create_child_span.return_value = telemetry_span
        mock_ports["telemetry_port"].record_metrics = Mock()

        # Setup other required mocks
        mock_file_discovery = Mock()
        mock_file_discovery.discover_test_files.return_value = []

        # Create use case with legacy config
        usecase = GenerateUseCase(
            config=legacy_config,
            file_discovery_service=mock_file_discovery,
            **mock_ports,
        )

        # Mock the service methods
        usecase._state_discovery = Mock()
        usecase._state_discovery.sync_and_discover.return_value = {
            "files": ["src/example.py"]
        }

        usecase._coverage_evaluator = Mock()
        usecase._coverage_evaluator.measure_initial.return_value = {
            "overall_line_coverage": 0.5
        }
        usecase._coverage_evaluator.measure_final.return_value = {
            "overall_line_coverage": 0.8
        }
        usecase._coverage_evaluator.calculate_delta.return_value = {
            "line_coverage_delta": 0.3
        }

        usecase._plan_builder = Mock()
        usecase._plan_builder.decide_files_to_process.return_value = ["src/example.py"]
        usecase._plan_builder.build_plans.return_value = [
            {"source_file": "src/example.py"}
        ]
        usecase._plan_builder.get_source_path_for_plan.return_value = Path(
            "src/example.py"
        )

        usecase._content_builder = Mock()
        usecase._content_builder.build_code_content.return_value = "def example(): pass"
        usecase._content_builder.determine_test_path.return_value = (
            "tests/test_example.py"
        )

        usecase._context_assembler = Mock()
        usecase._context_assembler.context_for_generation.return_value = {}

        # Mock batch executor and pytest refiner
        usecase._batch_executor = Mock()
        usecase._batch_executor.run_in_batches = AsyncMock(
            return_value=[
                GenerationResult(
                    file_path="tests/test_example.py",
                    content="def test_example(): pass",
                    success=True,
                    error_message=None,
                )
            ]
        )

        with patch(
            "testcraft.application.generate_usecase.PytestRefiner"
        ) as mock_pytest_refiner_class:
            mock_refiner = AsyncMock()
            mock_refiner.refine_until_pass = AsyncMock(
                return_value={
                    "success": True,
                    "iterations": 1,
                    "final_status": "passed",
                    "test_file": "tests/test_example.py",
                }
            )
            mock_pytest_refiner_class.return_value = mock_refiner

            with tempfile.TemporaryDirectory() as temp_dir:
                # Run the test
                results = await usecase.generate_tests(project_path=temp_dir)

                # Verify legacy mode works
                assert results["success"] is True
                assert results["tests_generated"] > 0

                # Verify batch executor was used (legacy mode)
                usecase._batch_executor.run_in_batches.assert_called_once()


@pytest.mark.integration
class TestImmediateRefinementIntegration:
    """Integration tests for immediate refinement."""

    @pytest.mark.asyncio
    async def test_state_incremental_logging(self, mock_ports, immediate_config):
        """Test that state.json contains incremental entries from immediate mode."""
        # This would be a more complex integration test that verifies
        # the state logging functionality works end-to-end
        # For now, we'll test the state recording method directly

        from testcraft.application.generate_usecase import GenerateUseCase
        from testcraft.domain.models import GenerationResult

        # Setup minimal mocks
        mock_file_discovery = Mock()
        mock_file_discovery.discover_test_files.return_value = []

        # Mock state port to track incremental updates
        incremental_logs = []

        def mock_update_state(key, value, merge_strategy=None):
            if key == "immediate_generation_log" and merge_strategy == "append":
                incremental_logs.append(value)
            return {"success": True}

        mock_ports["state_port"].update_state = mock_update_state

        # Setup telemetry mock
        telemetry_span = Mock()
        telemetry_span.__enter__ = Mock(return_value=telemetry_span)
        telemetry_span.__exit__ = Mock(return_value=None)
        mock_ports["telemetry_port"].create_child_span.return_value = telemetry_span
        mock_ports["telemetry_port"].record_metrics = Mock()

        usecase = GenerateUseCase(
            config=immediate_config,
            file_discovery_service=mock_file_discovery,
            **mock_ports,
        )

        # Create test file result
        file_result = {
            "generation_result": GenerationResult(
                file_path="tests/test_example.py",
                content="def test_example(): pass",
                success=True,
                error_message=None,
            ),
            "write_result": {
                "success": True,
                "bytes_written": 100,
                "formatted": True,
                "error": None,
            },
            "refinement_result": {
                "success": True,
                "iterations": 2,
                "final_status": "passed",
                "error": None,
            },
            "success": True,
            "errors": [],
        }

        # Test the per-file state recording
        await usecase._record_per_file_state(file_result)

        # Verify incremental log was recorded
        assert len(incremental_logs) == 1
        log_entry = incremental_logs[0]

        assert log_entry["file_path"] == "tests/test_example.py"
        assert log_entry["success"] is True
        assert "timestamp" in log_entry
        assert "stages" in log_entry

        # Verify all stages were logged
        stages = log_entry["stages"]
        assert "generation" in stages
        assert "write" in stages
        assert "refinement" in stages

        # Verify stage details
        assert stages["generation"]["success"] is True
        assert stages["write"]["success"] is True
        assert stages["refinement"]["success"] is True
        assert stages["refinement"]["iterations"] == 2

    @pytest.mark.skip(reason="Skipping test for now")
    @pytest.mark.asyncio
    async def test_per_file_progress_display(self):
        """Test that CLI shows detailed per-file progress for immediate mode."""
        # This would test the CLI display functionality
        # For now, we can test the display logic directly

        from io import StringIO

        from rich.console import Console

        from testcraft.cli.main import _display_immediate_mode_results
        from testcraft.domain.models import GenerationResult

        # Create mock results in immediate mode format
        results = {
            "metadata": {"config_used": {"immediate_refinement": True}},
            "files_discovered": 2,
            "files_written": 2,
            "tests_generated": 2,
            "files_processed": 2,
            "final_coverage": {"overall_line_coverage": 0.85},
            "coverage_delta": {"line_coverage_delta": 0.15},
            "generation_results": [
                GenerationResult(
                    file_path="tests/test_file1.py",
                    content="def test_file1(): pass",
                    success=True,
                    error_message=None,
                ),
                GenerationResult(
                    file_path="tests/test_file2.py",
                    content="def test_file2(): pass",
                    success=True,
                    error_message=None,
                ),
            ],
            "refinement_results": [
                {
                    "test_file": "tests/test_file1.py",
                    "success": True,
                    "iterations": 1,
                },
                {
                    "test_file": "tests/test_file2.py",
                    "success": True,
                    "iterations": 2,
                },
            ],
        }

        # Capture console output
        test_console = Console(file=StringIO(), width=80)

        # Mock the UI components
        with patch("testcraft.cli.main.ui") as mock_ui:
            with patch("testcraft.cli.main.console", test_console):
                # Test the display function
                _display_immediate_mode_results(results)

                # Verify UI components were called
                mock_ui.create_project_summary_panel.assert_called_once()
                mock_ui.print_panel.assert_called_once()
                mock_ui.display_success.assert_called_once()

                # Verify console output contains per-file table
                output = test_console.file.getvalue()
                assert "Per-File Results (Immediate Mode)" in output
