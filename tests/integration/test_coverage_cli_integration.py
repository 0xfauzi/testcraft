"""Integration tests for coverage CLI command."""

import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_project():
    """Create a temporary test project with source and test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create source file
        src_dir = project_root / "src"
        src_dir.mkdir()
        src_file = src_dir / "calculator.py"
        src_file.write_text("""
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
""")

        # Create test file that covers only some functions
        tests_dir = project_root / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_calculator.py"
        test_file.write_text("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calculator import add, subtract

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(0, 1) == -1
""")

        yield project_root


@pytest.mark.integration
def test_coverage_cli_basic_execution(temp_project):
    """Test basic coverage CLI execution."""
    result = subprocess.run(
        [
            "testcraft",
            "coverage",
            str(temp_project),
            "--source-files",
            str(temp_project / "src" / "calculator.py"),
            "--test-files",
            str(temp_project / "tests" / "test_calculator.py"),
            "--format",
            "summary",
        ],
        capture_output=True,
        text=True,
        cwd=temp_project,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert "Overall coverage" in result.stdout or "Coverage Summary" in result.stdout


@pytest.mark.integration
def test_coverage_cli_xml_format(temp_project):
    """Test coverage CLI with XML output format."""
    output_dir = temp_project / ".artifacts" / "coverage"

    result = subprocess.run(
        [
            "testcraft",
            "coverage",
            str(temp_project),
            "--source-files",
            str(temp_project / "src" / "calculator.py"),
            "--test-files",
            str(temp_project / "tests" / "test_calculator.py"),
            "--format",
            "xml",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=temp_project,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Verify XML file was created
    xml_files = list(output_dir.glob("coverage.xml"))
    assert len(xml_files) > 0, f"No XML file found in {output_dir}"

    # Verify XML file has content
    xml_content = xml_files[0].read_text()
    assert '<?xml version="1.0" ?>' in xml_content
    assert "<coverage" in xml_content


@pytest.mark.integration
def test_coverage_cli_json_format(temp_project):
    """Test coverage CLI with JSON output format."""
    result = subprocess.run(
        [
            "testcraft",
            "coverage",
            str(temp_project),
            "--source-files",
            str(temp_project / "src" / "calculator.py"),
            "--test-files",
            str(temp_project / "tests" / "test_calculator.py"),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=temp_project,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    # Output should be valid (exact format depends on UI adapter)


@pytest.mark.integration
def test_coverage_cli_with_include_omit(temp_project):
    """Test coverage CLI with include and omit patterns."""
    result = subprocess.run(
        [
            "testcraft",
            "coverage",
            str(temp_project),
            "--source-files",
            str(temp_project / "src" / "calculator.py"),
            "--test-files",
            str(temp_project / "tests" / "test_calculator.py"),
            "--format",
            "summary",
            "--include",
            str(temp_project / "src"),
            "--omit",
            "*/test_*.py",
        ],
        capture_output=True,
        text=True,
        cwd=temp_project,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"


@pytest.mark.integration
def test_coverage_cli_aggregation(temp_project):
    """Test coverage aggregation across multiple runs."""
    output_dir = temp_project / ".artifacts" / "coverage"
    output_dir.mkdir(parents=True, exist_ok=True)

    # First run: only test add and subtract
    test_file_1 = temp_project / "tests" / "test_partial_1.py"
    test_file_1.write_text("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calculator import add

def test_add():
    assert add(2, 3) == 5
""")

    result1 = subprocess.run(
        [
            "testcraft",
            "coverage",
            str(temp_project),
            "--source-files",
            str(temp_project / "src" / "calculator.py"),
            "--test-files",
            str(test_file_1),
            "--format",
            "xml",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=temp_project,
    )

    assert result1.returncode == 0, f"First run failed: {result1.stderr}"

    # Second run: test multiply (different function)
    test_file_2 = temp_project / "tests" / "test_partial_2.py"
    test_file_2.write_text("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calculator import multiply

def test_multiply():
    assert multiply(3, 4) == 12
""")

    result2 = subprocess.run(
        [
            "testcraft",
            "coverage",
            str(temp_project),
            "--source-files",
            str(temp_project / "src" / "calculator.py"),
            "--test-files",
            str(test_file_2),
            "--format",
            "xml",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=temp_project,
    )

    assert result2.returncode == 0, f"Second run failed: {result2.stderr}"

    # Verify multiple coverage data files exist
    coverage_files = list(output_dir.glob(".coverage.*"))
    assert len(coverage_files) >= 2, (
        f"Expected at least 2 coverage files, found {len(coverage_files)}"
    )


@pytest.mark.integration
def test_coverage_cli_multiple_formats(temp_project):
    """Test coverage CLI with multiple output formats."""
    output_dir = temp_project / ".artifacts" / "coverage"

    result = subprocess.run(
        [
            "testcraft",
            "coverage",
            str(temp_project),
            "--source-files",
            str(temp_project / "src" / "calculator.py"),
            "--test-files",
            str(temp_project / "tests" / "test_calculator.py"),
            "--format",
            "xml",
            "--format",
            "json",
            "--format",
            "summary",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=temp_project,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Verify XML file was created
    xml_files = list(output_dir.glob("coverage.xml"))
    assert len(xml_files) > 0, "XML report not generated"
