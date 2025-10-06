import pytest
from click.testing import CliRunner
from testcraft.cli.main import app


@pytest.mark.integration
def test_coverage_summary_single_file():
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--ui",
            "minimal",
            "coverage",
            ".",
            "-s",
            "testcraft/domain/models.py",
            "-o",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Overall coverage:" in result.output
