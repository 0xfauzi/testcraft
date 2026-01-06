from __future__ import annotations

import json
import shutil
import tomllib
from collections.abc import Iterable
from pathlib import Path

import pytest
import tomli_w

from .conftest import CommandResult, CommandSpec, run_cli_chain, strip_ansi


def _clean_project_artifacts(
    project_root: Path, extra_paths: Iterable[Path] | None = None
) -> None:
    purge_targets = {
        project_root / ".testcraft",
        project_root / ".testcraft_state.json",
        project_root / ".testcraft.toml",
        project_root / "tests" / "src",
        project_root / "tests" / "__pycache__",
        project_root / ".ruff_cache",
        project_root / ".pytest_cache",
    }
    if extra_paths:
        purge_targets.update(extra_paths)

    for path in purge_targets:
        if not path.exists():
            continue
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.live_service
@pytest.mark.slow
def test_generate_cli_with_plan(
    dummy_project: Path,
    cli_env: dict[str, str],
    run_cli,
):
    project_root = dummy_project
    logs_dir = project_root / ".testcraft" / "cli_logs" / "baseline"

    live_env = cli_env.copy()
    sanitized_env = {
        key: value
        for key, value in live_env.items()
        if not key.startswith("TESTCRAFT_E2E_")
    }
    sanitized_env.setdefault("TESTCRAFT_UI", "classic")
    sanitized_env.setdefault("TESTCRAFT_LOG_LEVEL", "DEBUG")

    source_file = Path("src/demo_app/calculator.py")
    llm_provider = live_env.get("TESTCRAFT_E2E_PROVIDER", "openai")
    e2e_model = live_env.get("TESTCRAFT_E2E_MODEL")

    def _configure_project(_: CommandResult, cwd: Path, _env: dict[str, str]) -> None:
        config_path = cwd / ".testcraft.toml"
        assert config_path.exists(), "init-config should create .testcraft.toml"
        config_data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        config_data.setdefault("llm", {})
        config_data["llm"]["default_provider"] = llm_provider
        generation_section = config_data.setdefault("generation", {})
        generation_section["enable_refinement"] = True
        generation_section["immediate_refinement"] = True
        generation_section.setdefault("plan_first", True)
        refine_section = generation_section.setdefault("refine", {})
        refine_section["enable"] = True
        if e2e_model:
            if llm_provider == "openai":
                config_data["llm"]["openai_model"] = e2e_model
            elif llm_provider == "anthropic":
                config_data["llm"]["anthropic_model"] = e2e_model
            elif llm_provider == "azure-openai":
                config_data["llm"]["azure_openai_deployment"] = e2e_model
            elif llm_provider == "bedrock":
                config_data["llm"]["bedrock_model_id"] = e2e_model
        else:
            if llm_provider == "openai":
                config_data["llm"].setdefault("openai_model", "o4-mini")
            elif llm_provider == "anthropic":
                config_data["llm"].setdefault(
                    "anthropic_model", "claude-3-haiku-20240307"
                )
        config_path.write_text(tomli_w.dumps(config_data), encoding="utf-8")

    commands = [
        CommandSpec(
            args=["init-config", "--minimal"],
            logfile="baseline_01_init.log",
            post_run=_configure_project,
        ),
        CommandSpec(
            args=[
                "--verbose",
                "analyze",
                ".",
                "--target-files",
                str(source_file),
            ],
            logfile="baseline_02_analyze.log",
        ),
        CommandSpec(
            args=[
                "generate",
                ".",
                "--target-files",
                str(source_file),
                "--auto-accept-fixes",
                "--plan-first",
                "--auto-accept-plan",
                "--keep-failed-writes",
            ],
            logfile="baseline_03_generate.log",
            timeout=900,
        ),
        CommandSpec(
            args=[
                "status",
                ".",
                "--history",
                "--statistics",
            ],
            logfile="baseline_04_status.log",
        ),
        CommandSpec(
            args=[
                "--ui",
                "minimal",
                "models",
                "show",
                "--provider",
                llm_provider,
                "--format",
                "json",
            ],
            logfile="baseline_05_models.log",
        ),
    ]

    try:
        results = run_cli_chain(
            commands,
            run_cli_impl=run_cli,
            cwd=project_root,
            env=sanitized_env,
            logs_dir=logs_dir,
        )
    except Exception:
        raise

    try:
        init_result, analyze_result, generate_result, status_result, models_result = (
            results
        )

        assert init_result.exit_code == 0, init_result.output
        assert generate_result.exit_code == 0, generate_result.output

        analyze_output = strip_ansi(analyze_result.output)
        assert str(source_file) in analyze_output, (
            "Analyze output should mention the target file"
        )

        generate_log = strip_ansi(generate_result.output)
        assert "Planning workflow starting" in generate_log, "Plan stage was not logged"
        assert "AUTO-ACCEPT PLAN" in generate_log or "Auto-accept plan" in generate_log

        state_path = project_root / ".testcraft_state.json"
        assert state_path.exists(), "State file missing"
        state_data = json.loads(state_path.read_text(encoding="utf-8"))
        last_run = state_data.get("last_generation_run") or {}
        summary = last_run.get("generation_summary") or {}
        assert summary.get("total_files_processed", 0) >= 1, "No files processed"
        assert summary.get("successful_generations", 0) >= 1, (
            "Generation reported no successes"
        )
        assert summary.get("failed_generations", 0) == 0, "Generation recorded failures"
        config_used = last_run.get("config_used") or {}
        recorded_provider = (config_used.get("llm") or {}).get("default_provider")
        if recorded_provider:
            assert recorded_provider.lower() == llm_provider.lower()

        artifacts_root = project_root / ".testcraft" / "artifacts"
        assert artifacts_root.exists(), "Artifacts directory missing"
        expected_dirs = ["generation_plan", "generated_test", "llm_response"]
        for dirname in expected_dirs:
            artifact_dir = artifacts_root / dirname
            assert artifact_dir.exists(), f"Missing artifact directory: {dirname}"
            assert any(path.is_file() for path in artifact_dir.rglob("*")), (
                f"Artifact directory {dirname} was empty"
            )

        tests_root = project_root / "tests"
        generated_tests = [
            path
            for path in tests_root.rglob("*.py")
            if path.name not in {"__init__.py"}
        ]
        assert generated_tests, "Expected generated tests to be created"

        status_log = strip_ansi(status_result.output)
        assert "Generation history" in status_log or "Statistics" in status_log
        refreshed_state = json.loads(state_path.read_text(encoding="utf-8"))
        assert refreshed_state.get("last_generation_run"), (
            "Status should not clear history"
        )

        models_payload = strip_ansi(models_result.output).strip()
        data = json.loads(models_payload)
        assert isinstance(data, list) and data, (
            "Models show should return model entries"
        )
    finally:
        pass
