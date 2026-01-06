"""Unit tests for ManualFixGuidanceUseCase."""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from testcraft.application.manual_fix_usecase import ManualFixGuidanceUseCase
from testcraft.domain.manual_fix import ManualFixRequest


@pytest.fixture
def mock_orchestrator():
    """Mock LLMOrchestrator."""
    orch = Mock()
    orch.manual_fix_stage = Mock(
        return_value={
            "manual_fix_response": """
```python
def test_bug_func_assertion():
    assert 1 == 2
```

```markdown
# BUG: func returns wrong value
Expected: 1
Actual: 2
```
            """,
            "context_pack": {},
            "stage": "manual_fix",
        }
    )
    return orch


@pytest.fixture
def mock_context_pack_builder():
    """Mock ContextPackBuilder."""
    builder = Mock()
    from testcraft.domain.models import (
        Budget,
        ContextPack,
        Conventions,
        Focal,
        ImportMap,
        PropertyContext,
        Target,
    )

    context_pack = ContextPack(
        target=Target(module_file="src/module.py", object="func"),
        import_map=ImportMap(
            target_import="from my_pkg import module as _under_test",
            sys_path_roots=["/repo/src"],
            needs_bootstrap=True,
            bootstrap_conftest="import sys; sys.path.insert(0, '/repo/src')",
        ),
        focal=Focal(
            source="def func(): return 2", signature="func() -> int", docstring=None
        ),
        resolved_defs=[],
        property_context=PropertyContext(),
        conventions=Conventions(),
        budget=Budget(),
    )
    builder.build_context_pack = Mock(return_value=context_pack)
    return builder


@pytest.fixture
def mock_telemetry():
    """Mock TelemetryPort."""
    telem = Mock()
    span_mock = MagicMock()
    span_mock.__enter__ = Mock(return_value=span_mock)
    span_mock.__exit__ = Mock(return_value=False)
    telem.create_span = Mock(return_value=span_mock)
    telem.create_child_span = Mock(return_value=span_mock)
    telem.record_metrics = Mock()
    return telem


@pytest.fixture
def mock_presenter_accept():
    """Mock presenter that auto-accepts."""
    presenter = Mock()
    presenter.present = Mock(return_value=True)
    return presenter


@pytest.fixture
def mock_presenter_reject():
    """Mock presenter that rejects."""
    presenter = Mock()
    presenter.present = Mock(return_value=False)
    return presenter


@pytest.mark.asyncio
async def test_usecase_parses_and_persists_on_acceptance(
    mock_orchestrator,
    mock_context_pack_builder,
    mock_telemetry,
    mock_presenter_accept,
    tmp_path,
):
    """Test that use case parses response, presents, and writes artifact on acceptance."""
    config = {
        "manual_fix": {
            "output_dir": str(tmp_path / "manual_fixes"),
            "auto_accept": False,
        }
    }

    usecase = ManualFixGuidanceUseCase(
        llm_orchestrator=mock_orchestrator,
        parser_port=Mock(),
        context_assembler=Mock(),
        context_pack_builder=mock_context_pack_builder,
        telemetry_port=mock_telemetry,
        presenter=mock_presenter_accept,
        config=config,
    )

    request = ManualFixRequest(
        project_root=Path("/repo"),
        target_file=Path("/repo/src/module.py"),
        target_object="func",
        trace_excerpt="AssertionError",
        notes="Test notes",
    )

    result = await usecase.run(request)

    assert result.accepted is True
    assert result.artifact_path is not None
    assert result.artifact_path.exists()
    assert result.recommendation is not None
    assert "def test_bug_func_assertion" in result.recommendation.test_code
    assert "BUG: func returns wrong value" in result.recommendation.bug_note_markdown


@pytest.mark.asyncio
async def test_usecase_does_not_persist_on_rejection(
    mock_orchestrator,
    mock_context_pack_builder,
    mock_telemetry,
    mock_presenter_reject,
    tmp_path,
):
    """Test that use case does not write artifact when user rejects."""
    config = {
        "manual_fix": {
            "output_dir": str(tmp_path / "manual_fixes"),
            "auto_accept": False,
        }
    }

    usecase = ManualFixGuidanceUseCase(
        llm_orchestrator=mock_orchestrator,
        parser_port=Mock(),
        context_assembler=Mock(),
        context_pack_builder=mock_context_pack_builder,
        telemetry_port=mock_telemetry,
        presenter=mock_presenter_reject,
        config=config,
    )

    request = ManualFixRequest(
        project_root=Path("/repo"),
        target_file=Path("/repo/src/module.py"),
        target_object="func",
    )

    result = await usecase.run(request)

    assert result.accepted is False
    assert result.artifact_path is None
    # Recommendation should still be created for inspection
    assert result.recommendation is not None


@pytest.mark.asyncio
async def test_usecase_auto_accept_bypasses_presenter(
    mock_orchestrator,
    mock_context_pack_builder,
    mock_telemetry,
    tmp_path,
):
    """Test that auto_accept=True bypasses presenter and writes artifact."""
    config = {
        "manual_fix": {
            "output_dir": str(tmp_path / "manual_fixes"),
            "auto_accept": True,
        }
    }

    presenter = Mock()
    presenter.present = Mock()  # Should not be called

    usecase = ManualFixGuidanceUseCase(
        llm_orchestrator=mock_orchestrator,
        parser_port=Mock(),
        context_assembler=Mock(),
        context_pack_builder=mock_context_pack_builder,
        telemetry_port=mock_telemetry,
        presenter=presenter,
        config=config,
    )

    request = ManualFixRequest(
        project_root=Path("/repo"),
        target_file=Path("/repo/src/module.py"),
        target_object="func",
    )

    result = await usecase.run(request)

    assert result.accepted is True
    assert result.artifact_path is not None
    # Presenter should NOT have been called
    presenter.present.assert_not_called()


@pytest.mark.asyncio
async def test_usecase_deduplicates_by_hash(
    mock_orchestrator,
    mock_context_pack_builder,
    mock_telemetry,
    mock_presenter_accept,
    tmp_path,
):
    """Test that duplicate recommendations (same hash) reuse existing artifact."""
    config = {
        "manual_fix": {
            "output_dir": str(tmp_path / "manual_fixes"),
            "auto_accept": False,
        }
    }

    usecase = ManualFixGuidanceUseCase(
        llm_orchestrator=mock_orchestrator,
        parser_port=Mock(),
        context_assembler=Mock(),
        context_pack_builder=mock_context_pack_builder,
        telemetry_port=mock_telemetry,
        presenter=mock_presenter_accept,
        config=config,
    )

    request = ManualFixRequest(
        project_root=Path("/repo"),
        target_file=Path("/repo/src/module.py"),
        target_object="func",
    )

    # First run
    result1 = await usecase.run(request)
    artifact1 = result1.artifact_path

    # Second run with same content (same hash)
    result2 = await usecase.run(request)
    artifact2 = result2.artifact_path

    assert artifact1 == artifact2
    assert (
        result1.recommendation.recommendation_hash
        == result2.recommendation.recommendation_hash
    )


def test_extract_fenced_blocks_with_python_and_markdown():
    """Test fenced block extraction with expected format."""
    usecase = ManualFixGuidanceUseCase(
        llm_orchestrator=Mock(),
        parser_port=Mock(),
        context_assembler=Mock(),
        context_pack_builder=Mock(),
        telemetry_port=Mock(),
        presenter=Mock(),
        config={},
    )

    text = """
Some preamble text.

```python
def test_bug():
    assert False
```

And now the bug note:

```markdown
# BUG TITLE
Summary: The function fails.
```
    """

    test_code, bug_md = usecase._extract_fenced_blocks(text)

    assert test_code is not None
    assert "def test_bug" in test_code
    assert bug_md is not None
    assert "BUG TITLE" in bug_md


def test_extract_fenced_blocks_fallback_to_second_fence():
    """Test fallback when no explicit markdown fence."""
    usecase = ManualFixGuidanceUseCase(
        llm_orchestrator=Mock(),
        parser_port=Mock(),
        context_assembler=Mock(),
        context_pack_builder=Mock(),
        telemetry_port=Mock(),
        presenter=Mock(),
        config={},
    )

    text = """
```python
def test_x():
    pass
```

```
BUG NOTE: The code is broken.
Steps to reproduce.
```
    """

    test_code, bug_md = usecase._extract_fenced_blocks(text)

    assert test_code is not None
    assert "def test_x" in test_code
    assert bug_md is not None
    assert "BUG NOTE" in bug_md


def test_extract_fenced_blocks_missing_code():
    """Test extraction when python block is missing."""
    usecase = ManualFixGuidanceUseCase(
        llm_orchestrator=Mock(),
        parser_port=Mock(),
        context_assembler=Mock(),
        context_pack_builder=Mock(),
        telemetry_port=Mock(),
        presenter=Mock(),
        config={},
    )

    text = """
```markdown
Only bug note, no test code.
```
    """

    test_code, bug_md = usecase._extract_fenced_blocks(text)

    # Should get None for test_code, bug_md should be present
    assert test_code is None
    assert bug_md is not None
    assert "Only bug note" in bug_md
