from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import CommandSpec, run_cli_chain, strip_ansi
from .test_cli_end_to_end import _clean_project_artifacts


@pytest.mark.integration
@pytest.mark.integration
@pytest.mark.utility
def test_cli_models_matrix(
    dummy_project_factory,
    cli_env_factory,
    run_cli,
    tmp_catalog_copy: Path,
):
    project_root: Path = dummy_project_factory()
    logs_dir = project_root / ".testcraft" / "cli_logs" / "models"

    env_live = cli_env_factory(project_root=project_root)
    sanitized_env = {
        key: value
        for key, value in env_live.items()
        if not key.startswith("TESTCRAFT_E2E_")
    }
    sanitized_env.setdefault("TESTCRAFT_UI", "minimal")

    provider = env_live.get("TESTCRAFT_E2E_PROVIDER", "openai")

    commands = [
        CommandSpec(
            args=["models", "show"],
            logfile="models/01_show_table.log",
        ),
        CommandSpec(
            args=["models", "verify"],
            logfile="models/02_verify.log",
        ),
        CommandSpec(
            args=[
                "models",
                "diff",
                "--file",
                str(tmp_catalog_copy),
            ],
            logfile="models/03_diff.log",
        ),
        CommandSpec(
            args=[
                "models",
                "show",
                "--provider",
                provider,
                "--format",
                "json",
            ],
            logfile="models/04_show_json.log",
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

        show_table = strip_ansi(results[0].output)
        assert "Provider/Model" in show_table.splitlines()[0]

        verify_output = strip_ansi(results[1].output)
        assert "OK" in verify_output

        diff_output = strip_ansi(results[2].output)
        assert "Added:" in diff_output and "Changed:" in diff_output
        assert "  +" not in diff_output and "  *" not in diff_output

        show_json_output = strip_ansi(results[3].output)
        payload = json.loads(show_json_output)
        assert isinstance(payload, list)
        assert all(entry.get("provider") == provider for entry in payload)

    finally:
        _clean_project_artifacts(project_root, extra_paths=[logs_dir])
