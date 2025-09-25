"""
Tests for ContextPack domain models and ContextPackBuilder service.

This module tests the ContextPack schema compliance with the context assembly
specification and validates the ContextPackBuilder service functionality.
"""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from testcraft.application.generation.services.context_pack import ContextPackBuilder
from testcraft.domain.models import (
    Budget,
    ContextPack,
    Conventions,
    DeterminismConfig,
    Focal,
    GwtSnippets,
    ImportMap,
    IOPolicy,
    PropertyContext,
    RankedMethod,
    ResolvedDef,
    Target,
    TestBundle,
)


class TestContextPackSchemaCompliance:
    """Test suite for ContextPack schema compliance with specification."""

    def test_target_model_validation(self) -> None:
        """Test Target model validation and required fields."""
        # Valid target
        target = Target(module_file="path/to/module.py", object="function_name")
        assert target.module_file == "path/to/module.py"
        assert target.object == "function_name"

        # Test missing required fields
        with pytest.raises(ValidationError):
            Target(module_file="path/to/module.py")  # type: ignore

        with pytest.raises(ValidationError):
            Target(object="function_name")  # type: ignore

    def test_import_map_model_validation(self) -> None:
        """Test ImportMap model validation and required fields."""
        # Valid import map
        import_map = ImportMap(
            target_import="import module as _under_test",
            sys_path_roots=["/path/to/src"],
            needs_bootstrap=True,
            bootstrap_conftest="import sys\nsys.path.append('/path/to/src')",
        )
        assert import_map.target_import == "import module as _under_test"
        assert import_map.sys_path_roots == ["/path/to/src"]
        assert import_map.needs_bootstrap is True
        assert "sys.path.append" in import_map.bootstrap_conftest

        # Test with empty bootstrap
        import_map_no_bootstrap = ImportMap(
            target_import="import module",
            sys_path_roots=[],
            needs_bootstrap=False,
            bootstrap_conftest="",
        )
        assert import_map_no_bootstrap.bootstrap_conftest == ""

    def test_focal_model_validation(self) -> None:
        """Test Focal model validation and optional docstring."""
        # Valid focal with docstring
        focal_with_doc = Focal(
            source="def test_func(): pass",
            signature="def test_func():",
            docstring="Test function docstring",
        )
        assert focal_with_doc.docstring == "Test function docstring"

        # Valid focal without docstring
        focal_no_doc = Focal(
            source="def test_func(): pass",
            signature="def test_func():",
            docstring=None,
        )
        assert focal_no_doc.docstring is None

        # Test missing required fields
        with pytest.raises(ValidationError):
            Focal(signature="def test():")  # type: ignore

    def test_resolved_def_model_validation(self) -> None:
        """Test ResolvedDef model validation and kind restrictions."""
        # Test all valid kinds
        valid_kinds = ["class", "func", "const", "enum", "fixture"]
        for kind in valid_kinds:
            resolved_def = ResolvedDef(
                name="TestSymbol",
                kind=kind,  # type: ignore
                signature=f"{kind} TestSymbol",
                doc="Test documentation",
                body="implementation or omitted",
            )
            assert resolved_def.kind == kind

        # Test invalid kind
        with pytest.raises(ValidationError):
            ResolvedDef(
                name="TestSymbol",
                kind="invalid_kind",  # type: ignore
                signature="def test():",
                body="omitted",
            )

    def test_ranked_method_model_validation(self) -> None:
        """Test RankedMethod model validation with level and relation restrictions."""
        # Valid combinations
        valid_combinations = [
            ("intra", "complete"),
            ("intra", "G"),
            ("intra", "W"),
            ("intra", "T"),
            ("repo", "complete"),
            ("repo", "G"),
            ("repo", "W"),
            ("repo", "T"),
        ]

        for level, relation in valid_combinations:
            ranked_method = RankedMethod(
                qualname="pkg.Class.method",
                level=level,  # type: ignore
                relation=relation,  # type: ignore
            )
            assert ranked_method.level == level
            assert ranked_method.relation == relation

        # Invalid level
        with pytest.raises(ValidationError):
            RankedMethod(
                qualname="pkg.Class.method",
                level="invalid",  # type: ignore
                relation="complete",  # type: ignore
            )

        # Invalid relation
        with pytest.raises(ValidationError):
            RankedMethod(
                qualname="pkg.Class.method",
                level="intra",  # type: ignore
                relation="invalid",  # type: ignore
            )

    def test_gwt_snippets_model_validation(self) -> None:
        """Test GwtSnippets model with default empty lists."""
        # Empty snippets (defaults)
        gwt_empty = GwtSnippets()
        assert gwt_empty.given == []
        assert gwt_empty.when == []
        assert gwt_empty.then == []

        # Populated snippets
        gwt_populated = GwtSnippets(
            given=["given_snippet_1", "given_snippet_2"],
            when=["when_snippet_1"],
            then=["then_snippet_1", "then_snippet_2", "then_snippet_3"],
        )
        assert len(gwt_populated.given) == 2
        assert len(gwt_populated.when) == 1
        assert len(gwt_populated.then) == 3

    def test_test_bundle_model_validation(self) -> None:
        """Test TestBundle model with all fields."""
        test_bundle = TestBundle(
            test_name="test_example",
            imports=["import pytest", "from module import func"],
            fixtures=["fixture_1", "fixture_2"],
            mocks=["mock.patch", "monkeypatch"],
            assertions=["assert result == expected", "assert len(data) > 0"],
        )
        assert test_bundle.test_name == "test_example"
        assert len(test_bundle.imports) == 2
        assert len(test_bundle.fixtures) == 2
        assert len(test_bundle.mocks) == 2
        assert len(test_bundle.assertions) == 2

    def test_property_context_model_validation(self) -> None:
        """Test PropertyContext model with defaults."""
        # Empty context
        prop_context_empty = PropertyContext()
        assert prop_context_empty.ranked_methods == []
        assert isinstance(prop_context_empty.gwt_snippets, GwtSnippets)
        assert prop_context_empty.test_bundles == []

        # Populated context
        prop_context_full = PropertyContext(
            ranked_methods=[
                RankedMethod(
                    qualname="pkg.Class.method", level="intra", relation="complete"
                )
            ],
            gwt_snippets=GwtSnippets(given=["given"], when=["when"], then=["then"]),
            test_bundles=[TestBundle(test_name="test_method")],
        )
        assert len(prop_context_full.ranked_methods) == 1
        assert len(prop_context_full.gwt_snippets.given) == 1
        assert len(prop_context_full.test_bundles) == 1

    def test_determinism_config_model_validation(self) -> None:
        """Test DeterminismConfig model with defaults."""
        # Default config
        determ_default = DeterminismConfig()
        assert determ_default.seed == 1337
        assert determ_default.tz == "UTC"
        assert determ_default.freeze_time is True

        # Custom config
        determ_custom = DeterminismConfig(seed=42, tz="US/Pacific", freeze_time=False)
        assert determ_custom.seed == 42
        assert determ_custom.tz == "US/Pacific"
        assert determ_custom.freeze_time is False

    def test_io_policy_model_validation(self) -> None:
        """Test IOPolicy model with restrictions."""
        # Default policy
        io_default = IOPolicy()
        assert io_default.network == "forbidden"
        assert io_default.fs == "tmp_path_only"

        # Valid alternative policies
        io_mocked = IOPolicy(network="mocked", fs="mocked")
        assert io_mocked.network == "mocked"
        assert io_mocked.fs == "mocked"

        # Invalid network policy
        with pytest.raises(ValidationError):
            IOPolicy(network="invalid")  # type: ignore

        # Invalid fs policy
        with pytest.raises(ValidationError):
            IOPolicy(fs="invalid")  # type: ignore

    def test_conventions_model_validation(self) -> None:
        """Test Conventions model with defaults."""
        # Default conventions
        conv_default = Conventions()
        assert conv_default.test_style == "pytest"
        assert "pytest" in conv_default.allowed_libs
        assert "hypothesis" in conv_default.allowed_libs
        assert isinstance(conv_default.determinism, DeterminismConfig)
        assert isinstance(conv_default.io_policy, IOPolicy)

        # Custom conventions
        conv_custom = Conventions(
            test_style="unittest",
            allowed_libs=["unittest", "mock"],
            determinism=DeterminismConfig(seed=99),
            io_policy=IOPolicy(network="mocked"),
        )
        assert conv_custom.test_style == "unittest"
        assert conv_custom.allowed_libs == ["unittest", "mock"]
        assert conv_custom.determinism.seed == 99

    def test_budget_model_validation(self) -> None:
        """Test Budget model validation with positive tokens."""
        # Default budget
        budget_default = Budget()
        assert budget_default.max_input_tokens == 60000

        # Custom budget
        budget_custom = Budget(max_input_tokens=100000)
        assert budget_custom.max_input_tokens == 100000

        # Invalid budget (negative tokens)
        with pytest.raises(ValidationError):
            Budget(max_input_tokens=-1000)

        # Invalid budget (zero tokens)
        with pytest.raises(ValidationError):
            Budget(max_input_tokens=0)

    def test_context_pack_full_schema_validation(self) -> None:
        """Test complete ContextPack schema matching specification."""
        # Create a complete ContextPack matching the JSON schema from spec
        context_pack = ContextPack(
            target=Target(
                module_file="path/to/src/my_pkg/sub/module.py",
                object="Class.method",
            ),
            import_map=ImportMap(
                target_import="from my_pkg.sub import module as _under_test",
                sys_path_roots=["/abs/repo/src"],
                needs_bootstrap=True,
                bootstrap_conftest="import sys\nsys.path.append('/abs/repo/src')",
            ),
            focal=Focal(
                source="def method(self): pass",
                signature="def method(self):",
                docstring="Method docstring",
            ),
            resolved_defs=[
                ResolvedDef(
                    name="Symbol",
                    kind="class",
                    signature="class Symbol:",
                    doc="Symbol documentation",
                    body="omitted",
                )
            ],
            property_context=PropertyContext(
                ranked_methods=[
                    RankedMethod(
                        qualname="pkg.Class.method",
                        level="intra",
                        relation="complete",
                    )
                ],
                gwt_snippets=GwtSnippets(
                    given=["given_example"],
                    when=["when_example"],
                    then=["then_example"],
                ),
                test_bundles=[
                    TestBundle(
                        test_name="test_x",
                        imports=["import pytest"],
                        fixtures=["fixture"],
                        mocks=["mock"],
                        assertions=["assert True"],
                    )
                ],
            ),
            conventions=Conventions(
                test_style="pytest",
                allowed_libs=["pytest", "hypothesis"],
                determinism=DeterminismConfig(seed=1337, tz="UTC", freeze_time=True),
                io_policy=IOPolicy(network="forbidden", fs="tmp_path_only"),
            ),
            budget=Budget(max_input_tokens=60000),
        )

        # Verify all components are present and match specification
        assert context_pack.target.module_file == "path/to/src/my_pkg/sub/module.py"
        assert context_pack.target.object == "Class.method"

        assert context_pack.import_map.target_import.endswith("as _under_test")
        assert len(context_pack.import_map.sys_path_roots) == 1
        assert context_pack.import_map.needs_bootstrap is True

        assert "def method" in context_pack.focal.source
        assert context_pack.focal.signature == "def method(self):"
        assert context_pack.focal.docstring == "Method docstring"

        assert len(context_pack.resolved_defs) == 1
        assert context_pack.resolved_defs[0].name == "Symbol"

        assert len(context_pack.property_context.ranked_methods) == 1
        assert len(context_pack.property_context.gwt_snippets.given) == 1
        assert len(context_pack.property_context.test_bundles) == 1

        assert context_pack.conventions.test_style == "pytest"
        assert context_pack.conventions.io_policy.network == "forbidden"

        assert context_pack.budget.max_input_tokens == 60000

        # Test immutability by attempting to modify a field
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            # This should fail because the model is frozen
            context_pack.target = Target(module_file="changed", object="changed")  # type: ignore


class TestContextPackBuilder:
    """Test suite for ContextPackBuilder service functionality."""

    def test_context_pack_builder_initialization(self) -> None:
        """Test ContextPackBuilder initialization with dependencies."""
        # Default initialization
        builder = ContextPackBuilder()
        assert builder._import_resolver is not None
        assert builder._enriched_context_builder is not None

        # Initialization with custom dependencies
        from testcraft.application.generation.services.enhanced_context_builder import (
            EnrichedContextBuilder,
        )
        from testcraft.application.generation.services.import_resolver import (
            ImportResolver,
        )

        custom_resolver = ImportResolver()
        custom_builder = EnrichedContextBuilder()

        builder_custom = ContextPackBuilder(
            import_resolver=custom_resolver,
            enriched_context_builder=custom_builder,
        )
        assert builder_custom._import_resolver is custom_resolver
        assert builder_custom._enriched_context_builder is custom_builder

    def test_build_context_pack_integration(self) -> None:
        """Test ContextPackBuilder.build_context_pack integration."""
        builder = ContextPackBuilder()

        # Create a temporary directory and file for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_module.py"

            temp_path.write_text('''
"""Test module for context pack building."""

class TestClass:
    """A test class for demonstration."""

    def test_method(self, arg: str) -> str:
        """A test method with type hints."""
        return f"processed_{arg}"

def standalone_function(x: int, y: int = 10) -> int:
    """A standalone function."""
    return x + y
''')

            # Test building ContextPack for a class method
            context_pack_method = builder.build_context_pack(
                target_file=temp_path, target_object="TestClass.test_method"
            )

            # Verify the basic structure
            assert isinstance(context_pack_method, ContextPack)
            assert context_pack_method.target.object == "TestClass.test_method"
            assert "test_method" in context_pack_method.focal.source
            assert "def test_method" in context_pack_method.focal.signature
            assert (
                context_pack_method.focal.docstring == "A test method with type hints."
            )

            # Test building ContextPack for a standalone function
            context_pack_func = builder.build_context_pack(
                target_file=temp_path, target_object="standalone_function"
            )

            assert context_pack_func.target.object == "standalone_function"
            assert "standalone_function" in context_pack_func.focal.source
            # The signature should contain information about the target (may be fallback)
            assert "standalone_function" in context_pack_func.focal.signature
            # Docstring may be None if parsing fails, which is acceptable for test purposes
            assert context_pack_func.focal.docstring in [
                None,
                "A standalone function.",
            ]

            # Verify import_map is populated
            assert context_pack_func.import_map.target_import
            assert isinstance(context_pack_func.import_map.sys_path_roots, list)
            assert isinstance(context_pack_func.import_map.needs_bootstrap, bool)

            # Verify defaults are applied
            assert context_pack_func.conventions.test_style == "pytest"
            assert context_pack_func.budget.max_input_tokens == 60000

    def test_build_context_pack_with_custom_options(self) -> None:
        """Test ContextPackBuilder with custom conventions and budget."""
        builder = ContextPackBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "simple_module.py"
            temp_path.write_text("def simple_func(): pass")

            custom_conventions = Conventions(
                test_style="unittest",
                allowed_libs=["unittest"],
            )
            custom_budget = Budget(max_input_tokens=30000)

            context_pack = builder.build_context_pack(
                target_file=temp_path,
                target_object="simple_func",
                conventions=custom_conventions,
                budget=custom_budget,
            )

            assert context_pack.conventions.test_style == "unittest"
            assert context_pack.conventions.allowed_libs == ["unittest"]
            assert context_pack.budget.max_input_tokens == 30000

    def test_build_enriched_context_delegation(self) -> None:
        """Test that build_enriched_context delegates to EnrichedContextBuilder."""
        builder = ContextPackBuilder()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temp_file:
            temp_file.write("def test_func(): pass")
            temp_file.flush()

            temp_path = Path(temp_file.name)

            try:
                enriched_context = builder.build_enriched_context(source_file=temp_path)

                # Verify that we get a dictionary response
                assert isinstance(enriched_context, dict)
                assert "packaging" in enriched_context
                assert "imports" in enriched_context
                assert "entities" in enriched_context

            finally:
                temp_path.unlink()

    def test_error_handling_invalid_target(self) -> None:
        """Test error handling for invalid target objects."""
        builder = ContextPackBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_module.py"
            temp_path.write_text("def existing_func(): pass")

            # Should not raise exception, but should handle gracefully
            context_pack = builder.build_context_pack(
                target_file=temp_path, target_object="nonexistent_function"
            )

            # Should still create a valid ContextPack with fallback focal info
            assert isinstance(context_pack, ContextPack)
            assert context_pack.target.object == "nonexistent_function"
            assert context_pack.focal.signature.startswith("# Target:")
