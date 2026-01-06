from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import CommandSpec, ScenarioInput, run_cli_chain
from .test_cli_end_to_end import _clean_project_artifacts


@pytest.mark.integration
@pytest.mark.live_service
def test_cli_coverage_live(
    scenario_matrix,
    dummy_project_factory,
    cli_env_factory,
    run_cli,
):
    batch_scenario = next(
        spec for spec in scenario_matrix if spec.name == "batch_refine"
    )

    project_root: Path = dummy_project_factory(multi_source=True)
    logs_dir = project_root / ".testcraft" / "cli_logs" / "coverage"

    live_env = cli_env_factory(project_root=project_root)
    sanitized_env = {
        key: value
        for key, value in live_env.items()
        if not key.startswith("TESTCRAFT_E2E_")
    }
    sanitized_env.setdefault("TESTCRAFT_UI", "minimal")

    provider = live_env.get("TESTCRAFT_E2E_PROVIDER", "openai")
    extra_sources = [
        Path(path) if not isinstance(path, Path) else path
        for path in batch_scenario.metadata["extra_sources"]  # type: ignore[index]
    ]

    scenario_input = ScenarioInput(
        project_root=project_root,
        source_file=Path("src/demo_app/calculator.py"),
        extra_source_files=extra_sources,
        tests_root=project_root / "tests",
        logs_dir=logs_dir / "batch",
        provider=provider,
        env=live_env,
    )

    try:
        run_cli_chain(
            batch_scenario.build_commands(scenario_input),
            run_cli_impl=run_cli,
            cwd=project_root,
            env=sanitized_env,
            logs_dir=scenario_input.logs_dir,
        )

        tests_dir = project_root / "tests" / "src" / "demo_app"
        test_files = sorted(path for path in tests_dir.glob("test_*.py"))
        assert test_files, "Expected generated test files for coverage"

        coverage_output_dir = project_root / ".testcraft" / "artifacts" / "coverage"
        coverage_args = [
            "coverage",
            ".",
            "--output-dir",
            str(coverage_output_dir.relative_to(project_root)),
            "--format",
            "detailed",
            "--format",
            "json",
            "--include",
            "src/demo_app",
            "--omit",
            "tests/unit",
        ]

        # Add explicit source/test file lists
        coverage_sources = [
            scenario_input.source_file,
            *scenario_input.extra_source_files,
        ]
        for source in coverage_sources:
            coverage_args.extend(["--source-files", str(source)])
        for test_file in test_files:
            relative_test = test_file.relative_to(project_root)
            coverage_args.extend(["--test-files", str(relative_test)])

        coverage_command = CommandSpec(
            args=coverage_args,
            logfile="coverage_command.log",
        )

        exit_code, output, _log_path, _duration = run_cli(
            coverage_command.args,
            cwd=project_root,
            env=sanitized_env,
            logfile_name=str(
                (logs_dir / coverage_command.logfile).relative_to(project_root)
            ),
        )
        assert exit_code == 0, f"Coverage command failed:\n{output}"

        assert coverage_output_dir.exists(), "Coverage output directory missing"
        artifacts = list(coverage_output_dir.rglob("*"))
        assert artifacts, "Coverage output directory empty"
        assert any(path.suffix in {".json", ".xml"} for path in artifacts)

    finally:
        _clean_project_artifacts(project_root, extra_paths=[logs_dir])
