"""Smoke test for coverage CLI producing Cobertura XML.

This is a minimal integration test that verifies the `testcraft coverage`
command can generate a valid XML report for a tiny throwaway project.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.integration
def test_coverage_cli_generates_xml_report() -> None:
    """Ensure Cobertura XML is produced successfully for a tiny project."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create minimal source file
        src_dir = project_root / "src"
        src_dir.mkdir()
        (src_dir / "calculator.py").write_text(
            """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
        )

        # Create minimal tests (covers only one function to keep it simple)
        tests_dir = project_root / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_calculator.py").write_text(
            """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calculator import add

def test_add():
    assert add(2, 3) == 5
"""
        )

        # Where XML should be written
        output_dir = project_root / ".artifacts" / "coverage"

        # Run coverage CLI
        result = subprocess.run(
            [
                "testcraft",
                "coverage",
                str(project_root),
                "--source-files",
                str(src_dir / "calculator.py"),
                "--test-files",
                str(tests_dir / "test_calculator.py"),
                "--format",
                "xml",
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Verify XML file exists and is non-empty, with a Cobertura-like root
        xml_path = output_dir / "coverage.xml"
        assert xml_path.exists(), f"Expected XML report at {xml_path}"
        xml_text = xml_path.read_text()
        assert "<coverage" in xml_text, "Missing <coverage root element in XML output>"
