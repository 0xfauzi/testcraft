import pytest
from click.testing import CliRunner
from testcraft.cli.main import app


@pytest.mark.integration
def test_status_runs_without_prior_state():
    runner = CliRunner()
    result = runner.invoke(app, ["--ui", "minimal", "status"])
    assert result.exit_code == 0, result.output
