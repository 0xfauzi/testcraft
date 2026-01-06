from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import CommandSpec, run_cli_chain, strip_ansi
from .test_cli_end_to_end import _clean_project_artifacts


def _sanitize_env(env: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in env.items() if not k.startswith("TESTCRAFT_E2E_")}


@pytest.mark.integration
@pytest.mark.fake_service
def test_cli_offline_warning(dummy_project_factory, cli_env_factory, run_cli):
    project_root: Path = dummy_project_factory()
    logs_dir = project_root / ".testcraft" / "cli_logs" / "offline"
    source_file = Path("src/demo_app/calculator.py")

    live_env = cli_env_factory(
        live=False,
        project_root=project_root,
        fake_mode="fake-llm,auto-confirm",
    )
    env = _sanitize_env(live_env)
    env.setdefault("TESTCRAFT_UI", "minimal")

    plan_output = Path(".testcraft/artifacts/offline_plan.json")

    commands = [
        CommandSpec(
            [
                "init-config",
                "--minimal",
            ],
            logfile="offline/01_init.log",
        ),
        CommandSpec(
            [
                "analyze",
                ".",
                "--target-files",
                str(source_file),
            ],
            logfile="offline/02_analyze.log",
        ),
        CommandSpec(
            [
                "generate",
                ".",
                "--target-files",
                str(source_file),
                "--auto-accept-fixes",
            ],
            logfile="offline/03_generate.log",
        ),
        CommandSpec(
            [
                "plan",
                "--disable-gates",
                "--target-file",
                str(source_file),
                "--target-object",
                "add",
                "--output",
                str(plan_output),
            ],
            logfile="offline/04_plan.log",
        ),
    ]

    try:
        results = run_cli_chain(
            commands,
            run_cli_impl=run_cli,
            cwd=project_root,
            env=env,
            logs_dir=logs_dir,
        )

        generate_output = strip_ansi(results[2].output)
        assert "LLM provider 'fake' unavailable" in generate_output
        assert "offline" in generate_output.lower()

        artifacts_root = project_root / ".testcraft" / "artifacts"
        generated_dir = artifacts_root / "generated_test"
        assert not generated_dir.exists() or not any(generated_dir.iterdir())

        plan_path = project_root / plan_output
        assert plan_path.exists(), "Plan output missing in offline mode"
        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
        assert plan_payload.get("plan_id")

        state_path = project_root / ".testcraft_state.json"
        assert not state_path.exists(), "Offline run should not persist state"

    finally:
        _clean_project_artifacts(project_root, extra_paths=[logs_dir])


@pytest.mark.integration
@pytest.mark.fake_service
def test_cli_failure_paths(dummy_project_factory, cli_env_factory, run_cli):
    project_root: Path = dummy_project_factory()
    logs_dir = project_root / ".testcraft" / "cli_logs" / "failures"

    live_env = cli_env_factory(
        live=False,
        project_root=project_root,
        fake_mode="fake-cost,fake-utility,auto-confirm",
    )
    env = _sanitize_env(live_env)
    env.setdefault("TESTCRAFT_UI", "minimal")

    commands = [
        CommandSpec(
            [
                "init-config",
                "--minimal",
            ],
            logfile="failures/01_init.log",
        ),
        CommandSpec(
            [
                "sync_state",
                "--reload",
                "--no-persist",
            ],
            logfile="failures/02_sync.log",
            expect_exit=1,
        ),
        CommandSpec(
            [
                "reset_state",
                "--categories",
                "generation_state",
                "--no-backup",
                "--confirm",
            ],
            logfile="failures/03_reset.log",
            expect_exit=1,
        ),
    ]

    try:
        results = run_cli_chain(
            commands,
            run_cli_impl=run_cli,
            cwd=project_root,
            env=env,
            logs_dir=logs_dir,
        )

        sync_output = strip_ansi(results[1].output)
        assert "Sync state disabled" in sync_output

        reset_output = strip_ansi(results[2].output)
        assert (
            "Reset state failed" in reset_output or "disabled" in reset_output.lower()
        )

        state_path = project_root / ".testcraft_state.json"
        assert not state_path.exists(), "Failure paths should not create state file"

    finally:
        _clean_project_artifacts(project_root, extra_paths=[logs_dir])


@pytest.mark.integration
@pytest.mark.fake_service
def test_cli_analyze_then_generate_failure(
    dummy_project_factory, cli_env_factory, run_cli
):
    project_root: Path = dummy_project_factory()
    logs_dir = project_root / ".testcraft" / "cli_logs" / "generate_failure"
    source_file = Path("src/demo_app/calculator.py")

    live_env = cli_env_factory(
        live=False,
        project_root=project_root,
        fake_mode="fake-llm,fake-generate-fail,auto-confirm",
    )
    env = _sanitize_env(live_env)
    env.setdefault("TESTCRAFT_UI", "minimal")

    commands = [
        CommandSpec(
            [
                "init-config",
                "--minimal",
            ],
            logfile="generate_failure/01_init.log",
        ),
        CommandSpec(
            [
                "analyze",
                ".",
                "--target-files",
                str(source_file),
            ],
            logfile="generate_failure/02_analyze.log",
        ),
        CommandSpec(
            [
                "generate",
                ".",
                "--target-files",
                str(source_file),
                "--auto-accept-fixes",
            ],
            logfile="generate_failure/03_generate.log",
            expect_exit=1,
        ),
        CommandSpec(
            [
                "status",
                ".",
                "--limit",
                "1",
            ],
            logfile="generate_failure/04_status.log",
        ),
    ]

    try:
        results = run_cli_chain(
            commands,
            run_cli_impl=run_cli,
            cwd=project_root,
            env=env,
            logs_dir=logs_dir,
        )

        generate_output = strip_ansi(results[2].output)
        assert "Simulated generation failure" in generate_output

        state_path = project_root / ".testcraft_state.json"
        assert state_path.exists(), "Status should create state file after failure"
        state_data = json.loads(state_path.read_text(encoding="utf-8"))
        assert not state_data.get("last_generation_run"), (
            "Failure should not record history"
        )

    finally:
        _clean_project_artifacts(project_root, extra_paths=[logs_dir])


@pytest.mark.integration
@pytest.mark.fake_service
def test_cli_manual_fix_without_accept(dummy_project_factory, cli_env_factory, run_cli):
    project_root: Path = dummy_project_factory()
    logs_dir = project_root / ".testcraft" / "cli_logs" / "manual_fix_decline"
    source_file = Path("src/demo_app/calculator.py")

    live_env = cli_env_factory(
        live=False,
        project_root=project_root,
        fake_mode="fake-llm,manual-fix-decline",
    )
    env = _sanitize_env(live_env)
    env.setdefault("TESTCRAFT_UI", "minimal")

    commands = [
        CommandSpec(
            [
                "init-config",
                "--minimal",
            ],
            logfile="manual_fix_decline/01_init.log",
        ),
        CommandSpec(
            [
                "manual-fix",
                str(project_root),
                "--target-file",
                str(source_file),
                "--target-object",
                "add",
                "--notes",
                "decline-test",
                "--trace-excerpt",
                "trace",
            ],
            logfile="manual_fix_decline/02_manual_fix.log",
            expect_exit=0,
        ),
    ]

    try:
        results = run_cli_chain(
            commands,
            run_cli_impl=run_cli,
            cwd=project_root,
            env=env,
            logs_dir=logs_dir,
        )

        manual_fix_output = strip_ansi(results[1].output)
        assert "Manual-fix not accepted" in manual_fix_output

        artifacts_dir = project_root / ".testcraft" / "artifacts"
        markdowns = (
            list(artifacts_dir.glob("**/*.md")) if artifacts_dir.exists() else []
        )
        assert not markdowns, "Manual fix decline should not persist artifacts"

    finally:
        _clean_project_artifacts(project_root, extra_paths=[logs_dir])
