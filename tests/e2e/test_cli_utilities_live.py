from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import CommandSpec, ScenarioInput, ScenarioSpec, run_cli_chain
from .test_cli_end_to_end import _clean_project_artifacts


def _find_scenario(name: str, scenarios: list[ScenarioSpec]) -> ScenarioSpec:
    for scenario in scenarios:
        if scenario.name == name:
            return scenario
    raise KeyError(name)


@pytest.mark.integration
@pytest.mark.live_service
@pytest.mark.utility
def test_cli_utilities_live(
    scenario_matrix: list[ScenarioSpec],
    dummy_project_factory,
    cli_env_factory,
    run_cli,
):
    project_root: Path = dummy_project_factory()
    logs_dir = project_root / ".testcraft" / "cli_logs" / "utilities"

    live_env = cli_env_factory(project_root=project_root, fake_mode="auto-confirm")
    sanitized_env = {
        key: value
        for key, value in live_env.items()
        if not key.startswith("TESTCRAFT_E2E_")
    }
    sanitized_env.setdefault("TESTCRAFT_UI", "minimal")
    sanitized_env.setdefault("TESTCRAFT_LOG_LEVEL", "INFO")

    provider = live_env.get("TESTCRAFT_E2E_PROVIDER", "openai")
    baseline = _find_scenario("live_baseline", scenario_matrix)
    baseline_input = ScenarioInput(
        project_root=project_root,
        source_file=Path("src/demo_app/calculator.py"),
        extra_source_files=[],
        tests_root=project_root / "tests",
        logs_dir=logs_dir / "baseline",
        provider=provider,
        env=live_env,
    )

    try:
        # Prime project with baseline journey to ensure state exists
        run_cli_chain(
            baseline.build_commands(baseline_input),
            run_cli_impl=run_cli,
            cwd=project_root,
            env=sanitized_env,
            logs_dir=baseline_input.logs_dir,
        )

        utility_commands = [
            CommandSpec(
                args=[
                    "env",
                    "--system",
                    "--python",
                    "--dependencies",
                ],
                logfile="utilities/01_env.log",
            ),
            CommandSpec(
                args=[
                    "cost",
                    "--period",
                    "weekly",
                    "--projections",
                    "--breakdown",
                ],
                logfile="utilities/02_cost.log",
            ),
            CommandSpec(
                args=[
                    "debug_state",
                    "--telemetry",
                    "--config",
                    "--format",
                    "json",
                ],
                logfile="utilities/03_debug_state.log",
            ),
            CommandSpec(
                args=[
                    "sync_state",
                    "--reload",
                ],
                logfile="utilities/04_sync_state.log",
            ),
            CommandSpec(
                args=[
                    "reset_state",
                    "--categories",
                    "generation_state",
                    "--no-backup",
                    "--confirm",
                ],
                logfile="utilities/05_reset_state.log",
            ),
            CommandSpec(
                args=[
                    "status",
                    ".",
                ],
                logfile="utilities/06_status_post_reset.log",
            ),
            CommandSpec(
                args=["sync_state"],
                logfile="utilities/07_sync_state_restore.log",
            ),
            CommandSpec(
                args=["version"],
                logfile="utilities/08_version.log",
            ),
        ]

        results = run_cli_chain(
            utility_commands,
            run_cli_impl=run_cli,
            cwd=project_root,
            env=sanitized_env,
            logs_dir=logs_dir,
        )

        state_path = project_root / ".testcraft_state.json"
        assert state_path.exists(), "State file should exist before reset"

        for spec, result in zip(utility_commands, results, strict=False):
            cmd = spec.args[0]
            if cmd == "env":
                assert "Python" in result.output or "System" in result.output
            elif cmd == "cost":
                assert result.output.strip(), "Cost command produced no output"
            elif cmd == "debug_state":
                debug_json = json.loads(result.output)
                assert "config" in debug_json or "debug_state" in debug_json
            elif cmd == "sync_state" and len(spec.args) > 1:
                assert "Sync" in result.output or "state" in result.output
            elif cmd == "reset_state":
                assert "Reset" in result.output or "Categories" in result.output
                if state_path.exists():
                    assert state_path.stat().st_size == 0
                else:
                    # File removed entirely
                    pass
            elif cmd == "status":
                assert result.output.strip(), "Status should recreate state"
                assert state_path.exists(), (
                    "Status should recreate .testcraft_state.json"
                )
            elif cmd == "sync_state":
                # Restore call
                assert result.output.strip()
            elif cmd == "version":
                assert "TestCraft version" in result.output

    finally:
        _clean_project_artifacts(project_root, extra_paths=[logs_dir])
