from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import ScenarioContext, ScenarioInput, ScenarioSpec, run_cli_chain
from .test_cli_end_to_end import _clean_project_artifacts


def _command_key(args: list[str]) -> str:
    tokens: list[str] = [arg for arg in args if not arg.startswith("-")]
    if not tokens:
        return args[0]
    if tokens[0] == "models" and len(tokens) >= 2:
        return f"{tokens[0]} {tokens[1]}"
    return tokens[0]


def _load_state(project_root: Path) -> dict:
    state_path = project_root / ".testcraft_state.json"
    return json.loads(state_path.read_text(encoding="utf-8"))


@pytest.mark.integration
@pytest.mark.live_service
@pytest.mark.parametrize(
    "scenario_name",
    [
        "plan_and_generate",
        "manual_fix_followup",
        "coverage_pass",
        "batch_refine",
    ],
)
def test_cli_journeys_live(
    scenario_name: str,
    scenario_matrix: list[ScenarioSpec],
    dummy_project_factory,
    cli_env_factory,
    run_cli,
):
    scenario_lookup = {spec.name: spec for spec in scenario_matrix}
    if scenario_name not in scenario_lookup:
        pytest.skip(f"Scenario '{scenario_name}' not defined in matrix")
    scenario = scenario_lookup[scenario_name]

    project_root: Path = dummy_project_factory(scenario.multi_source_project)
    logs_dir = project_root / ".testcraft" / "cli_logs" / scenario.name

    live_env = cli_env_factory(project_root=project_root)
    sanitized_env = {
        key: value
        for key, value in live_env.items()
        if not key.startswith("TESTCRAFT_E2E_")
    }
    sanitized_env.setdefault("TESTCRAFT_UI", "minimal")
    sanitized_env.setdefault("TESTCRAFT_LOG_LEVEL", "INFO")
    if scenario.env_overrides:
        sanitized_env.update(scenario.env_overrides)

    provider = live_env.get("TESTCRAFT_E2E_PROVIDER", "openai")
    source_file = Path("src/demo_app/calculator.py")
    extra_sources = [
        Path(path) if not isinstance(path, Path) else path
        for path in scenario.metadata.get("extra_sources", [])
    ]

    scenario_input = ScenarioInput(
        project_root=project_root,
        source_file=source_file,
        extra_source_files=extra_sources,
        tests_root=project_root / "tests",
        logs_dir=logs_dir,
        provider=provider,
        env=live_env,
    )

    commands = scenario.build_commands(scenario_input)

    try:
        results = run_cli_chain(
            commands,
            run_cli_impl=run_cli,
            cwd=project_root,
            env=sanitized_env,
            logs_dir=logs_dir,
        )
    except Exception:
        _clean_project_artifacts(project_root, extra_paths=[logs_dir])
        raise

    context = ScenarioContext(
        project_root=project_root,
        env=sanitized_env,
        logs_dir=logs_dir,
        results=results,
    )

    try:
        state_data = _load_state(project_root)
        result_map = {_command_key(result.spec.args): result for result in results}

        if scenario_name == "plan_and_generate":
            plan_result = result_map.get("plan")
            assert plan_result is not None, "Plan command missing from scenario"
            plan_path_rel: Path = scenario.metadata["plan_output_rel"]  # type: ignore[index]
            plan_path = project_root / plan_path_rel
            assert plan_path.exists(), "Plan output file not created"
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            quality_gates = plan_payload.get("quality_gates") or plan_payload.get(
                "qualityGates"
            )
            assert quality_gates, "Plan output missing quality gates"
            generation_config = (
                (state_data.get("last_generation_run") or {})
                .get("config_used", {})
                .get("generation", {})
            )
            assert generation_config.get("plan_first") is True

        elif scenario_name == "manual_fix_followup":
            artifacts_dir = project_root / ".testcraft" / "artifacts"
            markdowns = list(artifacts_dir.glob("**/*.md"))
            assert markdowns, "Manual-fix scenario should emit markdown artifacts"
            generation_config = (
                (state_data.get("last_generation_run") or {})
                .get("config_used", {})
                .get("generation", {})
            )
            assert generation_config.get("disable_ruff_format") is True
            assert generation_config.get("enable_streaming") is True

        elif scenario_name == "coverage_pass":
            coverage_result = result_map.get("coverage")
            assert coverage_result is not None, "Coverage command did not execute"
            assert coverage_result.exit_code == 0
            coverage_dir_rel: Path = scenario.metadata["coverage_dir_rel"]  # type: ignore[index]
            coverage_dir = project_root / coverage_dir_rel
            assert coverage_dir.exists(), "Coverage directory missing"
            artifacts = list(coverage_dir.glob("**/*"))
            assert any(path.suffix == ".json" for path in artifacts), (
                "Coverage JSON missing"
            )
            assert any(path.suffix in {".xml", ".html"} for path in artifacts), (
                "Coverage report missing XML/HTML output"
            )

        elif scenario_name == "batch_refine":
            generation_summary = (state_data.get("last_generation_run") or {}).get(
                "generation_summary", {}
            )
            assert generation_summary.get("total_files_processed", 0) >= 2
            generation_config = (
                (state_data.get("last_generation_run") or {})
                .get("config_used", {})
                .get("generation", {})
            )
            assert generation_config.get("batch_size") == 2
            assert generation_config.get("enable_symbol_resolution") is True
            assert generation_config.get("immediate_refinement") is False

        for assertion in scenario.assertions:
            assertion(context)
    finally:
        _clean_project_artifacts(project_root, extra_paths=[logs_dir])
