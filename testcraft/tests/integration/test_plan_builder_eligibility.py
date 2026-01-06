from pathlib import Path

from testcraft.adapters.io.file_discovery import FileDiscoveryService
from testcraft.adapters.parsing.codebase_parser import CodebaseParser
from testcraft.adapters.telemetry.noop_adapter import NoOpTelemetryAdapter
from testcraft.application.generation.services.plan_builder import PlanBuilder


def test_plan_builder_creates_plan_for_simple_function(tmp_path: Path):
    project = tmp_path
    src = project / "src" / "pkg"
    src.mkdir(parents=True)
    p = src / "alpha.py"
    p.write_text(
        """
def add(a, b):
    return a + b
""",
        encoding="utf-8",
    )

    discovery = FileDiscoveryService()
    parser = CodebaseParser()
    telemetry = NoOpTelemetryAdapter()
    builder = PlanBuilder(
        parser_port=parser,
        file_discovery_service=discovery,
        telemetry_port=telemetry,
        coverage_threshold=0.8,
    )

    builder.set_project_context(project, test_files=[])
    plans = builder.build_plans([p])

    assert plans, "Expected at least one generation plan for a module with a function"
