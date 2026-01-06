"""Integration test for manual-fix workflow (simplified to avoid real file system dependencies)."""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture
def mock_ports_for_manual_fix():
    """Create all mock ports required for GenerateUseCase with manual-fix."""
    from testcraft.application.generation.services.context_assembler import (
        ContextAssembler,
    )
    from testcraft.application.generation.services.context_pack import (
        ContextPackBuilder,
    )
    from testcraft.domain.models import (
        Budget,
        ContextPack,
        Conventions,
        Focal,
        ImportMap,
        PropertyContext,
        Target,
    )

    # Mock context pack for manual-fix
    context_pack = ContextPack(
        target=Target(module_file="src/module.py", object="broken_func"),
        import_map=ImportMap(
            target_import="from my_pkg import module as _under_test",
            sys_path_roots=["/repo/src"],
            needs_bootstrap=True,
            bootstrap_conftest="import sys; sys.path.insert(0, '/repo/src')",
        ),
        focal=Focal(
            source="def broken_func(): return 42",
            signature="broken_func() -> int",
            docstring=None,
        ),
        resolved_defs=[],
        property_context=PropertyContext(),
        conventions=Conventions(),
        budget=Budget(),
    )

    # Ports
    llm_port = Mock()
    llm_port.generate_tests = Mock(
        return_value={
            "manual_fix_response": """
```python
def test_bug_broken_func():
    result = broken_func()
    assert result == 10  # Expected 10, got 42 - production bug
```

```markdown
# BUG: broken_func returns wrong constant

**Summary**: Function returns hardcoded 42 instead of calculated value.

**Expected**: 10
**Actual**: 42

**Root Cause**: Line 1 of module.py - hardcoded return value.
```
            """
        }
    )

    writer_port = Mock()
    writer_port.write_test_file = Mock(
        return_value={"success": True, "bytes_written": 100}
    )

    coverage_port = Mock()
    coverage_port.measure_coverage = Mock(return_value={"overall_line_coverage": 0.8})

    refine_port = Mock()

    context_port = Mock()

    parser_port = Mock()
    parser_port.parse_file = Mock(return_value=[])

    state_port = Mock()
    state_port.update_state = Mock()
    state_port.persist_state = Mock()

    telemetry_port = Mock()
    span_mock = MagicMock()
    span_mock.__enter__ = Mock(return_value=span_mock)
    span_mock.__exit__ = Mock(return_value=False)
    telemetry_port.create_span = Mock(return_value=span_mock)
    telemetry_port.create_child_span = Mock(return_value=span_mock)
    telemetry_port.record_metrics = Mock()
    telemetry_port.flush = Mock()

    context_assembler = Mock(spec=ContextAssembler)
    context_assembler.gather_project_context = Mock(return_value={})
    context_assembler.context_for_generation = Mock(return_value=context_pack)
    context_assembler.context_for_refinement = Mock(return_value={})
    context_assembler.get_last_enriched_context = Mock(return_value=None)

    context_pack_builder = Mock(spec=ContextPackBuilder)
    context_pack_builder.build_context_pack = Mock(return_value=context_pack)
    context_pack_builder._find_project_root = Mock(return_value=Path("/repo"))

    return {
        "llm_port": llm_port,
        "writer_port": writer_port,
        "coverage_port": coverage_port,
        "refine_port": refine_port,
        "context_port": context_port,
        "parser_port": parser_port,
        "state_port": state_port,
        "telemetry_port": telemetry_port,
        "context_assembler": context_assembler,
        "context_pack_builder": context_pack_builder,
    }


@pytest.mark.asyncio
async def test_manual_fix_end_to_end_workflow(mock_ports_for_manual_fix, tmp_path):
    """
    Test the manual-fix workflow end-to-end with valid mocks.

    Verifies:
    - Context pack builds correctly with mocked builder
    - LLM orchestrator returns manual-fix response
    - Response is parsed into test code + bug note
    - Artifact is persisted with correct structure
    - Telemetry is emitted
    """
    from testcraft.application.manual_fix_usecase import ManualFixGuidanceUseCase
    from testcraft.domain.manual_fix import ManualFixRequest

    ports = mock_ports_for_manual_fix

    # Configure manual-fix use case
    config = {
        "manual_fix": {
            "enable": True,
            "on_fail": False,  # Direct invocation, not auto-trigger
            "auto_accept": True,
            "output_dir": str(tmp_path / "manual_fixes"),
        }
    }

    # Auto-accept presenter
    presenter = Mock()
    presenter.present = Mock(return_value=True)

    # Create orchestrator
    from testcraft.application.generation.services.llm_orchestrator import (
        LLMOrchestrator,
    )
    from testcraft.config.models import OrchestratorConfig

    orch_config = OrchestratorConfig(enable_manual_fix=True)
    orchestrator = LLMOrchestrator(
        llm_port=ports["llm_port"],
        parser_port=ports["parser_port"],
        context_assembler=ports["context_assembler"],
        context_pack_builder=ports["context_pack_builder"],
        symbol_resolver=Mock(),
        config=orch_config,
    )

    # Create use case
    usecase = ManualFixGuidanceUseCase(
        llm_orchestrator=orchestrator,
        parser_port=ports["parser_port"],
        context_assembler=ports["context_assembler"],
        context_pack_builder=ports["context_pack_builder"],
        telemetry_port=ports["telemetry_port"],
        presenter=presenter,
        config=config,
    )

    # Build request
    request = ManualFixRequest(
        project_root=Path("/repo"),
        target_file=Path("/repo/src/module.py"),
        target_object="broken_func",
        trace_excerpt="AssertionError: expected 10, got 42",
        notes="Repro: call broken_func() directly",
    )

    # Execute workflow
    result = await usecase.run(request)

    # Verify result
    assert result.accepted is True
    assert result.artifact_path is not None
    assert result.artifact_path.exists()
    assert result.recommendation is not None

    # Verify artifact content
    content = result.artifact_path.read_text()
    assert "Manual Fix Recommendation" in content
    assert "test_bug_broken_func" in content
    assert "BUG: broken_func returns wrong constant" in content
    assert "Canonical Import: from my_pkg import module as _under_test" in content

    # Verify telemetry was called
    ports["telemetry_port"].create_span.assert_called()
    ports["telemetry_port"].create_child_span.assert_called()
    ports["telemetry_port"].record_metrics.assert_called()
