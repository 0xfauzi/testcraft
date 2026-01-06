from pathlib import Path
from typing import Any

import pytest

from testcraft.adapters.context.main_adapter import TestcraftContextAdapter
from testcraft.adapters.io.file_discovery import FileDiscoveryService
from testcraft.adapters.io.state_json import StateJsonAdapter
from testcraft.adapters.io.writer_ast_merge import WriterASTMergeAdapter
from testcraft.adapters.parsing.codebase_parser import CodebaseParser
from testcraft.adapters.telemetry.noop_adapter import NoOpTelemetryAdapter
from testcraft.application.generate_usecase import GenerateUseCase
from testcraft.domain.models import RefineOutcome
from testcraft.ports.coverage_port import CoveragePort
from testcraft.ports.llm_port import LLMPort
from testcraft.ports.parser_port import ParserPort
from testcraft.ports.refine_port import RefinePort
from testcraft.ports.state_port import StatePort
from testcraft.ports.telemetry_port import TelemetryPort
from testcraft.ports.writer_port import WriterPort


class FakeLLMPort(LLMPort):
    """Deterministic LLM adapter used for integration testing."""

    def __init__(self) -> None:
        self._call_count = 0

    def generate_tests(
        self,
        code_content: str,
        context: str | None = None,
        test_framework: str = "pytest",
        **kwargs,
    ) -> dict[str, Any]:
        self._call_count += 1
        if self._call_count == 1:
            # PLAN stage response with no missing symbols
            return {
                "tests": '{"plan": [{"objective": "cover add function"}], "missing_symbols": []}',
                "coverage_focus": [],
                "confidence": 0.9,
                "metadata": {"stage": "plan"},
            }

        return {
            "tests": """```python
import pytest
from my_pkg import calculator as _under_test


def test_addition():
    assert _under_test.add(1, 2) == 3
```""",
            "coverage_focus": ["addition"],
            "confidence": 0.9,
            "metadata": {"stage": "generate"},
        }

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs
    ) -> dict[str, float | dict[str, int] | list[str]]:
        return {
            "testability_score": 0.5,
            "complexity_metrics": {},
            "recommendations": [],
            "potential_issues": [],
        }

    def refine_content(
        self,
        original_content: str,
        refinement_instructions: str,
        *,
        system_prompt: str | None = None,
        **kwargs,
    ) -> dict[str, str | list[str] | float]:
        return {
            "refined_content": original_content,
            "changes_made": [],
            "confidence": 0.0,
        }

    def generate_test_plan(
        self, code_content: str, context: str | None = None, **kwargs
    ) -> dict[str, list]:
        return {
            "test_plan": [],
            "test_coverage_areas": [],
            "test_priorities": [],
            "estimated_complexity": [],
        }


class StubCoveragePort(CoveragePort):
    def measure_coverage(
        self,
        source_files: list[str],
        test_files: list[str] | None = None,
        **kwargs,
    ):
        return {}

    def report_coverage(
        self, coverage_data: dict, output_format: str = "detailed", **kwargs
    ):
        return {"report_content": "", "summary_stats": {}, "format": output_format}

    def get_coverage_summary(self, coverage_data: dict) -> dict[str, float]:
        return {
            "overall_line_coverage": 0.0,
            "overall_branch_coverage": 0.0,
            "files_covered": 0,
            "total_lines": 0,
            "missing_coverage": {},
        }

    def identify_gaps(self, coverage_data: dict, threshold: float = 0.8):
        return {}

    def get_measurement_method(self) -> str:
        return "noop"


class StubRefinePort(RefinePort):
    def refine(
        self,
        test_files,
        source_files=None,
        refinement_goals=None,
        **kwargs,
    ) -> RefineOutcome:
        raise NotImplementedError("Refinement is disabled in this test.")

    def analyze_test_quality(self, test_file, source_file=None, **kwargs):
        return {}

    def suggest_improvements(
        self, test_file, improvement_type="comprehensive", **kwargs
    ):
        return {}

    def optimize_test_structure(self, test_file, optimization_goals=None, **kwargs):
        return {}

    def refine_from_failures(
        self,
        test_file,
        failure_output,
        source_context=None,
        max_iterations=3,
        **kwargs,
    ):
        return {"success": False}

    def enhance_test_coverage(
        self,
        test_file,
        source_file,
        coverage_gaps=None,
        **kwargs,
    ):
        return {}


@pytest.mark.asyncio
async def test_generate_usecase_integration(tmp_path: Path):
    project_root = tmp_path
    (project_root / "my_pkg").mkdir()
    (project_root / "tests").mkdir()

    module_path = project_root / "my_pkg" / "calculator.py"
    module_path.write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )

    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "my-pkg"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    llm_port: LLMPort = FakeLLMPort()
    writer_port: WriterPort = WriterASTMergeAdapter()
    coverage_port: CoveragePort = StubCoveragePort()
    refine_port: RefinePort = StubRefinePort()
    context_port = TestcraftContextAdapter()
    parser_port: ParserPort = CodebaseParser()
    state_port: StatePort = StateJsonAdapter()
    telemetry_port: TelemetryPort = NoOpTelemetryAdapter()
    file_discovery = FileDiscoveryService()

    generation_overrides = {
        "immediate_refinement": False,
        "enable_refinement": False,
        "batch_size": 1,
        "disable_ruff_format": True,
        "keep_failed_writes": True,
    }

    use_case = GenerateUseCase(
        llm_port=llm_port,
        writer_port=writer_port,
        coverage_port=coverage_port,
        refine_port=refine_port,
        context_port=context_port,
        parser_port=parser_port,
        state_port=state_port,
        telemetry_port=telemetry_port,
        file_discovery_service=file_discovery,
        config=generation_overrides,
    )

    result = await use_case.generate_tests(
        project_path=project_root, target_files=[module_path]
    )

    assert result["success"], result
    generated = result["generation_results"]
    assert generated and generated[0].success

    expected_test_path = project_root / "tests" / "my_pkg" / "test_calculator.py"
    assert expected_test_path.exists()
    test_content = expected_test_path.read_text(encoding="utf-8")
    assert "def test_addition" in test_content
    assert "from my_pkg import calculator as _under_test" in test_content
