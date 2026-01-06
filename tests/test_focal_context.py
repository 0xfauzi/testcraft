"""Regression tests for focal context extraction."""

from pathlib import Path
from types import SimpleNamespace

from testcraft.adapters.parsing.codebase_parser import CodebaseParser
from testcraft.application.generation.services.context_assembler import ContextAssembler
from testcraft.domain.models import (
    Target,
    TestElement,
    TestElementType,
    TestGenerationPlan,
)


def _make_assembler() -> tuple[ContextAssembler, CodebaseParser]:
    context_port = SimpleNamespace(
        build_context_graph=lambda *_, **__: {},
        index=lambda *_, **__: {},
        get_related_context=lambda *_, **__: {},
    )
    parser = CodebaseParser()
    assembler = ContextAssembler(
        context_port=context_port, parser_port=parser, config={}
    )
    return assembler, parser


def _make_plan(
    element: TestElement, source_path: Path, project_root: Path
) -> TestGenerationPlan:
    return TestGenerationPlan(
        elements_to_test=[element],
        existing_tests=[],
        coverage_before=None,
        file_path=source_path,
        project_root=project_root,
        module_path="sample",
    )


def test_build_focal_object_includes_real_source(tmp_path):
    source_path = tmp_path / "sample.py"
    source_path.write_text(
        "\n".join(
            [
                "class Sample:",
                "    '''Sample docstring.'''",
                "",
                "    def run(self):",
                "        return 42",
            ]
        )
    )

    assembler, parser = _make_assembler()
    parse_result = parser.parse_file(source_path)
    element = next(e for e in parse_result["elements"] if e.name == "Sample")
    plan = _make_plan(element, source_path, tmp_path)
    target = Target(module_file=str(source_path), object="Sample")

    focal = assembler._build_focal_object(source_path, plan, target)

    assert focal is not None
    assert "class Sample" in focal.source
    assert "def run" in focal.source
    assert not focal.is_placeholder
    assert focal.placeholder_reason is None


def test_focal_placeholder_when_source_missing(tmp_path):
    source_path = tmp_path / "missing.py"
    assembler, parser = _make_assembler()
    element = TestElement(
        name="Missing",
        type=TestElementType.CLASS,
        line_range=(1, 1),
        docstring=None,
    )
    plan = _make_plan(element, source_path, tmp_path)
    target = Target(module_file=str(source_path), object="Missing")

    focal = assembler._build_focal_object(source_path, plan, target)

    assert focal is not None
    assert focal.is_placeholder
    assert focal.placeholder_reason is not None
