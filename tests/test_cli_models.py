from __future__ import annotations

import json
from click.testing import CliRunner

from testcraft.cli.main import app


def test_models_show_json() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["models", "show", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list) and len(data) > 0


def test_models_verify_ok() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["models", "verify"])
    assert result.exit_code == 0, result.output


def test_models_diff_self() -> None:
    from pathlib import Path

    catalog = Path(__file__).resolve().parents[1] / "testcraft" / "config" / "model_catalog.toml"
    runner = CliRunner()
    result = runner.invoke(app, ["models", "diff", "--file", str(catalog)])
    assert result.exit_code == 0, result.output
    assert "Changed:" in result.output


