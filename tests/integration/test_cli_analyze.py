import pytest
from click.testing import CliRunner
from testcraft.cli.main import app


@pytest.mark.integration
def test_analyze_smoke_for_single_file():
    runner = CliRunner()
    # Global options must come before subcommand
    result = runner.invoke(app, ["--ui", "classic", "analyze", ".", "-f", "testcraft/domain/models.py"])  # classic shows headings
    assert result.exit_code == 0, result.output
    assert "Analysis Complete" in result.output
