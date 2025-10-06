import os
import pytest
from click.testing import CliRunner
from testcraft.cli.main import app


@pytest.mark.integration
def test_generate_dry_run_succeeds_without_llm_credentials(monkeypatch):
    # Clear LLM keys to simulate typical local env without creds
    for k in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
    ]:
        monkeypatch.delenv(k, raising=False)

    runner = CliRunner()
    # Global options must precede the subcommand in Click
    result = runner.invoke(
        app,
        [
            "--dry-run",
            "--ui",
            "classic",  # classic shows info panels in output
            "generate",
            ".",
            "-f",
            "testcraft/domain/models.py",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dry Run Mode" in result.output
