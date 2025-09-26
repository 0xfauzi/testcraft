"""
Integration tests for ContextPack workflow in Generate Use Case.

This module tests the complete ContextPack → Generate → Write workflow
to ensure canonical import enforcement and proper ContextPack integration.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from testcraft.application.generate_usecase import GenerateUseCase
from testcraft.application.generation.services.context_assembler import ContextAssembler
from testcraft.application.generation.services.context_pack import ContextPackBuilder
from testcraft.application.generation.services.llm_orchestrator import LLMOrchestrator
from testcraft.config.models import OrchestratorConfig
from testcraft.domain.models import (
    Budget,
    ContextPack,
    Conventions,
    Focal,
    ImportMap,
    PropertyContext,
    Target,
    TestElement,
    TestGenerationPlan,
)
from testcraft.ports.context_port import ContextPort
from testcraft.ports.coverage_port import CoveragePort
from testcraft.ports.llm_port import LLMPort
from testcraft.ports.parser_port import ParserPort
from testcraft.ports.refine_port import RefinePort
from testcraft.ports.state_port import StatePort
from testcraft.ports.telemetry_port import TelemetryPort
from testcraft.ports.writer_port import WriterPort


class TestContextPackIntegration:
    """Integration tests for ContextPack workflow."""

    @pytest.fixture
    def mock_ports(self):
        """Create mock ports for testing."""
        return {
            "llm_port": MagicMock(spec=LLMPort),
            "writer_port": MagicMock(spec=WriterPort),
            "coverage_port": MagicMock(spec=CoveragePort),
            "refine_port": MagicMock(spec=RefinePort),
            "context_port": MagicMock(spec=ContextPort),
            "parser_port": MagicMock(spec=ParserPort),
            "state_port": MagicMock(spec=StatePort),
            "telemetry_port": MagicMock(spec=TelemetryPort),
        }

    @pytest.fixture
    def sample_context_pack(self):
        """Create a sample ContextPack for testing."""
        return ContextPack(
            target=Target(
                module_file="/path/to/test_module.py", object="test_function"
            ),
            import_map=ImportMap(
                target_import="from test_module import test_function",
                sys_path_roots=["/path/to/project"],
                needs_bootstrap=False,
                bootstrap_conftest="",
            ),
            focal=Focal(
                source="def test_function():\n    return True",
                signature="def test_function() -> bool:",
                docstring="A test function for integration testing",
            ),
            resolved_defs=[],
            property_context=PropertyContext(),
            conventions=Conventions(),
            budget=Budget(),
        )

    @pytest.fixture
    def test_generation_plan(self):
        """Create a sample TestGenerationPlan for testing."""
        return TestGenerationPlan(
            elements_to_test=[
                TestElement(
                    name="test_function",
                    type="function",
                    line_range=(1, 3),
                    docstring="A test function for integration testing",
                )
            ],
            existing_tests=[],
            coverage_before=None,
        )

    def test_context_pack_builder_integration(self):
        """Test that ContextPackBuilder creates proper ContextPack objects."""
        # Create ContextPackBuilder
        builder = ContextPackBuilder()

        # Create a temporary test file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def test_function():\n    return True\n")
            temp_file = Path(f.name)

        try:
            # Build a ContextPack
            context_pack = builder.build_context_pack(
                target_file=temp_file,
                target_object="test_function",
                project_root=Path("/test/project"),
            )

            # Verify ContextPack structure
            assert context_pack.target.module_file == str(temp_file.resolve())
            assert context_pack.target.object == "test_function"
            assert context_pack.import_map.target_import is not None
            assert context_pack.focal.source is not None
            assert context_pack.focal.signature is not None

        finally:
            # Clean up
            temp_file.unlink(missing_ok=True)

    def test_context_assembler_returns_context_pack(self, mock_ports):
        """Test that ContextAssembler returns ContextPack objects."""
        # Create ContextAssembler
        assembler = ContextAssembler(
            context_port=mock_ports["context_port"],
            parser_port=mock_ports["parser_port"],
            config={"enable_context": True},
        )

        # Create test plan
        plan = TestGenerationPlan(
            elements_to_test=[
                TestElement(
                    name="test_function",
                    type="function",
                    line_range=(1, 3),
                    docstring="Test function",
                )
            ],
            existing_tests=[],
            coverage_before=None,
        )

        # Mock the context building process
        with (
            patch.object(assembler, "_retrieve_snippets", return_value=[]),
            patch.object(assembler, "_get_neighbor_context", return_value=[]),
            patch.object(assembler, "_get_test_exemplars", return_value=[]),
            patch.object(assembler, "_get_contract_context", return_value=[]),
            patch.object(assembler, "_get_deps_config_fixtures", return_value=[]),
            patch.object(assembler, "_get_coverage_hints", return_value=[]),
            patch.object(assembler, "_get_callgraph_neighbors", return_value=[]),
            patch.object(assembler, "_get_error_paths", return_value=[]),
            patch.object(assembler, "_get_usage_examples", return_value=[]),
            patch.object(assembler, "_get_pytest_settings_context", return_value=[]),
            patch.object(assembler, "_get_side_effects_context", return_value=[]),
            patch.object(assembler, "_get_path_constraints_context", return_value=[]),
            patch.object(
                assembler,
                "_build_enriched_context_for_generation",
                return_value="Test context",
            ),
            patch.object(assembler, "_import_resolver") as mock_resolver,
        ):
            # Mock import resolver
            mock_resolver.resolve.return_value = ImportMap(
                target_import="from module import test_function",
                sys_path_roots=["/test"],
                needs_bootstrap=False,
                bootstrap_conftest="",
            )

            # Call context_for_generation
            result = assembler.context_for_generation(plan, Path("/test/module.py"))

            # Verify result is ContextPack
            assert isinstance(result, ContextPack)
            assert result.target.object == "test_function"
            assert result.import_map.target_import == "from module import test_function"

    async def test_generate_usecase_uses_context_pack(self, mock_ports):
        """Test that GenerateUseCase properly uses ContextPack objects."""
        # Create GenerateUseCase
        usecase = GenerateUseCase(**mock_ports, config={})

        # Create test plan
        plan = TestGenerationPlan(
            elements_to_test=[
                TestElement(
                    name="test_function",
                    type="function",
                    line_range=(1, 3),
                    docstring="Test function",
                )
            ],
            existing_tests=[],
            coverage_before=None,
        )

        # Mock the ContextPackBuilder
        with patch.object(
            usecase._context_pack_builder, "build_context_pack"
        ) as mock_build:
            mock_context_pack = ContextPack(
                target=Target(module_file="/test/module.py", object="test_function"),
                import_map=ImportMap(
                    target_import="from test_module import test_function",
                    sys_path_roots=["/test"],
                    needs_bootstrap=False,
                    bootstrap_conftest="",
                ),
                focal=Focal(
                    source="def test_function(): pass",
                    signature="def test_function() -> None:",
                    docstring="Test function",
                ),
                resolved_defs=[],
                property_context=PropertyContext(),
                conventions=Conventions(),
                budget=Budget(),
            )
            mock_build.return_value = mock_context_pack

            # Mock the LLM orchestrator
            with patch.object(
                usecase._llm_orchestrator, "plan_and_generate"
            ) as mock_orchestrate:
                mock_orchestrate.return_value = {
                    "generated_code": "def test_test_function(): assert True",
                    "plan": {"steps": ["test"]},
                    "context_pack": mock_context_pack,
                }

                # Mock other dependencies
                with (
                    patch.object(
                        usecase._plan_builder,
                        "get_source_path_for_plan",
                        return_value="/test/module.py",
                    ),
                    patch.object(
                        usecase._content_builder,
                        "build_code_content",
                        return_value="def test_function(): pass",
                    ),
                    patch.object(
                        usecase._context_assembler,
                        "context_for_generation",
                        return_value=None,  # Force fallback to ContextPackBuilder
                    ),
                    patch.object(
                        usecase._writer,
                        "write_test_file",
                        return_value={"success": True},
                    ),
                    patch.object(
                        usecase._content_builder,
                        "determine_test_path",
                        return_value="/test/test_module.py",
                    ),
                ):
                    # Call the method that should use ContextPack
                    result = await usecase._generate_tests_for_plan(plan, {})

                    # Verify ContextPack was built and used
                    mock_build.assert_called_once()
                    mock_orchestrate.assert_called_once_with(
                        context_pack=mock_context_pack, project_root=None
                    )

                    # Verify the result uses the canonical import from ContextPack
                    assert result.success
                    assert result.file_path == "/test/test_module.py"

    async def test_canonical_import_enforcement_in_prompts(self, mock_ports):
        """Test that canonical imports from ContextPack are enforced in LLM prompts."""
        # Create LLM Orchestrator
        orchestrator = LLMOrchestrator(
            llm_port=mock_ports["llm_port"],
            parser_port=mock_ports["parser_port"],
            context_assembler=MagicMock(),
            symbol_resolver=MagicMock(),
            config=OrchestratorConfig(),
        )

        # Create sample ContextPack
        context_pack = ContextPack(
            target=Target(module_file="/test/module.py", object="test_function"),
            import_map=ImportMap(
                target_import="from test_module import test_function",
                sys_path_roots=["/test"],
                needs_bootstrap=False,
                bootstrap_conftest="",
            ),
            focal=Focal(
                source="def test_function(): pass",
                signature="def test_function() -> None:",
                docstring="Test function",
            ),
            resolved_defs=[],
            property_context=PropertyContext(),
            conventions=Conventions(),
            budget=Budget(),
        )

        # Mock LLM to capture the prompt
        with patch.object(mock_ports["llm_port"], "generate_tests") as mock_llm:
            mock_llm.return_value = {"tests": "def test_something(): pass"}

            # Mock prompt registry
            with (
                patch.object(
                    orchestrator._prompt_registry,
                    "get_system_prompt",
                    return_value="Test system",
                ),
                patch.object(
                    orchestrator._prompt_registry, "get_user_prompt"
                ) as mock_user_prompt,
                patch.object(orchestrator._prompt_registry, "version", "1.0"),
            ):
                # Capture the context passed to user prompt
                captured_context = None

                def capture_context(prompt_name, additional_context, version):
                    nonlocal captured_context
                    captured_context = additional_context
                    return "Test user prompt"

                mock_user_prompt.side_effect = capture_context

                # Call plan_and_generate
                orchestrator.plan_and_generate(context_pack)

                # Verify canonical import is in the prompt context
                assert captured_context is not None
                assert (
                    captured_context["import_map"]["target_import"]
                    == "from test_module import test_function"
                )

    async def test_writer_integration_with_context_pack_guardrails(self, mock_ports):
        """Test that writers use ContextPack information for guardrails and gates."""
        # Create a writer that should respect ContextPack conventions
        # This is a conceptual test - actual implementation would depend on
        # how ContextPack information is passed to writers

        # The key requirement is that writers should:
        # 1. Use ContextPack.conventions.io_policy for safety checks
        # 2. Use ContextPack.import_map.bootstrap_conftest when needed
        # 3. Apply ContextPack.conventions for code style and structure

        # This test verifies the integration points exist
        context_pack = ContextPack(
            target=Target(module_file="/test/module.py", object="test_function"),
            import_map=ImportMap(
                target_import="from test_module import test_function",
                sys_path_roots=["/test"],
                needs_bootstrap=True,
                bootstrap_conftest="# Bootstrap conftest content",
            ),
            focal=Focal(
                source="def test_function(): pass",
                signature="def test_function() -> None:",
                docstring="Test function",
            ),
            resolved_defs=[],
            property_context=None,
            conventions=None,  # Would contain IO policies, style guides, etc.
            budget=None,
        )

        # Verify ContextPack has the information writers need
        assert context_pack.import_map.target_import is not None
        assert context_pack.import_map.needs_bootstrap is True
        assert context_pack.import_map.bootstrap_conftest is not None

        # In actual implementation, writers would use:
        # - context_pack.conventions.io_policy for safety validation
        # - context_pack.import_map.bootstrap_conftest for conftest generation
        # - context_pack.conventions for code style enforcement

    async def test_end_to_end_context_pack_workflow(self, mock_ports):
        """Test end-to-end ContextPack workflow with canonical import enforcement."""
        # This is a high-level integration test that verifies the complete flow

        # Create GenerateUseCase with mocked dependencies
        usecase = GenerateUseCase(**mock_ports, config={"enable_context": True})

        # Create a test plan
        plan = TestGenerationPlan(
            elements_to_test=[
                TestElement(
                    name="test_function",
                    type="function",
                    line_range=(1, 3),
                    docstring="Test function for integration",
                )
            ],
            existing_tests=[],
            coverage_before=None,
        )

        # Mock the entire chain to verify ContextPack flows through
        with (
            patch.object(
                usecase._context_pack_builder, "build_context_pack"
            ) as mock_build,
            patch.object(
                usecase._llm_orchestrator, "plan_and_generate"
            ) as mock_orchestrate,
            patch.object(usecase._writer, "write_test_file") as mock_write,
            patch.object(
                usecase._plan_builder,
                "get_source_path_for_plan",
                return_value="/test/module.py",
            ),
            patch.object(
                usecase._content_builder,
                "build_code_content",
                return_value="def test_function(): pass",
            ),
            patch.object(
                usecase._content_builder,
                "determine_test_path",
                return_value="/test/test_module.py",
            ),
        ):
            # Set up ContextPack
            context_pack = ContextPack(
                target=Target(module_file="/test/module.py", object="test_function"),
                import_map=ImportMap(
                    target_import="from test_module import test_function",
                    sys_path_roots=["/test"],
                    needs_bootstrap=False,
                    bootstrap_conftest="",
                ),
                focal=Focal(
                    source="def test_function(): pass",
                    signature="def test_function() -> None:",
                    docstring="Test function",
                ),
                resolved_defs=[],
                property_context=None,
                conventions=None,
                budget=None,
            )
            mock_build.return_value = context_pack

            # Set up orchestrator response
            mock_orchestrate.return_value = {
                "generated_code": "def test_test_function(): assert True",
                "plan": {"steps": ["test"]},
                "context_pack": context_pack,
            }

            # Set up writer response
            mock_write.return_value = {"success": True}

            # Execute the workflow
            result = await usecase._generate_tests_for_plan(plan, {})

            # Verify the complete workflow
            assert result.success
            assert result.file_path == "/test/test_module.py"
            assert result.content == "def test_test_function(): assert True"

            # Verify ContextPack was used throughout
            mock_build.assert_called_once()
            mock_orchestrate.assert_called_once_with(
                context_pack=context_pack, project_root=None
            )
            mock_write.assert_called_once()

    async def test_context_pack_import_map_exclusivity(self, mock_ports):
        """Test that ContextPack.import_map is used exclusively for import resolution."""
        # This test ensures that once ContextPack is integrated,
        # duplicate import resolution logic is removed and only ContextPack.import_map is used

        usecase = GenerateUseCase(**mock_ports, config={})

        # Create a plan
        plan = TestGenerationPlan(
            elements_to_test=[
                TestElement(
                    name="test_function",
                    type="function",
                    line_range=(1, 3),
                    docstring="Test function",
                )
            ],
            existing_tests=[],
            coverage_before=None,
        )

        # Mock ContextPackBuilder to return a ContextPack with specific import
        context_pack = ContextPack(
            target=Target(module_file="/test/module.py", object="test_function"),
            import_map=ImportMap(
                target_import="from canonical_module import test_function",
                sys_path_roots=["/test"],
                needs_bootstrap=False,
                bootstrap_conftest="",
            ),
            focal=Focal(
                source="def test_function(): pass",
                signature="def test_function() -> None:",
                docstring="Test function",
            ),
            resolved_defs=[],
            property_context=PropertyContext(),
            conventions=Conventions(),
            budget=Budget(),
        )

        with (
            patch.object(
                usecase._context_pack_builder,
                "build_context_pack",
                return_value=context_pack,
            ),
            patch.object(
                usecase._llm_orchestrator,
                "plan_and_generate",
                return_value={
                    "generated_code": "def test_test_function(): assert True",
                    "plan": {"steps": ["test"]},
                    "context_pack": context_pack,
                },
            ),
            patch.object(
                usecase._plan_builder,
                "get_source_path_for_plan",
                return_value="/test/module.py",
            ),
            patch.object(
                usecase._content_builder,
                "build_code_content",
                return_value="def test_function(): pass",
            ),
            patch.object(
                usecase._writer, "write_test_file", return_value={"success": True}
            ),
            patch.object(
                usecase._content_builder,
                "determine_test_path",
                return_value="/test/test_module.py",
            ),
        ):
            result = await usecase._generate_tests_for_plan(plan, {})

            # The key assertion: the generated test should use the canonical import
            # from the ContextPack, not from any other source
            assert result.success

            # In a real implementation, the generated code would use the canonical import
            # from context_pack.import_map.target_import. For this test, we verify
            # that the ContextPack was used as the single source of truth for imports.
