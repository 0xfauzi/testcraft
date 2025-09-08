#!/usr/bin/env python3
"""
Local CI Pipeline Testing Script for testcraft.

This script simulates the CI pipeline locally to help debug and validate
CI configuration before pushing to GitHub. It runs the same checks that
would run in GitHub Actions.

Run from project root: python scripts/test_ci_pipeline.py
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class LocalCITester:
    """Run CI pipeline checks locally for testing and validation."""

    def __init__(self, project_root: Path = None, verbose: bool = False):
        """Initialize the local CI tester."""
        self.project_root = project_root or Path.cwd()
        self.verbose = verbose
        self.results: dict[str, dict] = {}

    def run_all_checks(self) -> bool:
        """
        Run all CI checks locally.

        Returns:
            True if all checks pass, False otherwise
        """
        print("🧪 Starting local CI pipeline testing...")
        print(f"📁 Project root: {self.project_root}")

        checks = [
            ("Lint Check (ruff)", self._run_ruff_check),
            ("Format Check (ruff)", self._run_format_check),
            ("Format Check (black)", self._run_black_check),
            ("Type Check (mypy)", self._run_mypy_check),
            ("Test Suite (pytest)", self._run_pytest),
            ("Documentation Check", self._run_doc_check),
            ("Prompt Regression Test", self._run_prompt_regression),
            ("Security Scan", self._run_security_scan),
        ]

        all_passed = True

        for check_name, check_func in checks:
            print(f"\n🔍 Running: {check_name}")
            success, details = check_func()

            self.results[check_name] = {
                "success": success,
                "details": details,
                "timestamp": datetime.now().isoformat(),
            }

            if success:
                print(f"✅ {check_name}: PASSED")
                if self.verbose and details.get("output"):
                    print(f"   Output: {details['output'][:200]}...")
            else:
                print(f"❌ {check_name}: FAILED")
                if details.get("error"):
                    print(f"   Error: {details['error']}")
                all_passed = False

        # Generate summary report
        self._generate_summary_report()

        if all_passed:
            print("\n🎉 All local CI checks PASSED!")
            print("🚀 Your changes should pass CI when pushed to GitHub")
        else:
            print("\n❌ Some local CI checks FAILED!")
            print("🔧 Fix the issues before pushing to GitHub")

        return all_passed

    def _run_command(
        self, cmd: list[str], cwd: Path | None = None
    ) -> tuple[bool, dict]:
        """Run a command and return success status and details."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            success = result.returncode == 0
            details = {
                "returncode": result.returncode,
                "output": result.stdout,
                "error": result.stderr,
                "command": " ".join(cmd),
            }

            return success, details

        except subprocess.TimeoutExpired:
            return False, {
                "error": "Command timed out after 5 minutes",
                "command": " ".join(cmd),
            }
        except Exception as e:
            return False, {"error": str(e), "command": " ".join(cmd)}

    def _run_ruff_check(self) -> tuple[bool, dict]:
        """Run ruff linting check."""
        return self._run_command(["uv", "run", "ruff", "check", "."])

    def _run_format_check(self) -> tuple[bool, dict]:
        """Run ruff format check."""
        return self._run_command(["uv", "run", "ruff", "format", "--check", "."])

    def _run_black_check(self) -> tuple[bool, dict]:
        """Run black format check."""
        return self._run_command(["uv", "run", "black", "--check", "--diff", "."])

    def _run_mypy_check(self) -> tuple[bool, dict]:
        """Run mypy type checking."""
        return self._run_command(
            ["uv", "run", "mypy", "testcraft/", "--show-error-codes", "--pretty"]
        )

    def _run_pytest(self) -> tuple[bool, dict]:
        """Run pytest test suite."""
        cmd = [
            "uv",
            "run",
            "pytest",
            "--cov=testcraft",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-report=html",
            "-v",
        ]
        return self._run_command(cmd)

    def _run_doc_check(self) -> tuple[bool, dict]:
        """Run documentation checks."""
        # First ensure markdown is installed
        install_cmd = ["uv", "pip", "install", "markdown"]
        install_success, _ = self._run_command(install_cmd)

        if not install_success:
            return False, {"error": "Failed to install markdown package"}

        # Run documentation check script
        return self._run_command(["python", "scripts/doc_check.py", "--project-root=."])

    def _run_prompt_regression(self) -> tuple[bool, dict]:
        """Run prompt regression testing."""
        return self._run_command(
            [
                "python",
                "scripts/prompt_regression_test.py",
                "--output-dir=local-test-artifacts",
            ]
        )

    def _run_security_scan(self) -> tuple[bool, dict]:
        """Run security scanning."""
        # Install safety if not available
        install_cmd = ["uv", "pip", "install", "safety"]
        install_success, _ = self._run_command(install_cmd)

        if not install_success:
            return False, {"error": "Failed to install safety package"}

        # Run safety check
        return self._run_command(
            [
                "uv",
                "run",
                "safety",
                "check",
                "--json",
                "--output",
                "local-safety-report.json",
            ]
        )

    def _generate_summary_report(self) -> None:
        """Generate summary report of local CI testing."""
        report_path = self.project_root / "local-ci-test-report.json"

        summary = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "project_root": str(self.project_root),
                "total_checks": len(self.results),
                "passed_checks": sum(1 for r in self.results.values() if r["success"]),
                "failed_checks": sum(
                    1 for r in self.results.values() if not r["success"]
                ),
            },
            "results": self.results,
            "recommendations": self._generate_recommendations(),
        }

        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n📊 Detailed report saved to: {report_path}")

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        failed_checks = [
            name for name, result in self.results.items() if not result["success"]
        ]

        if not failed_checks:
            recommendations.append("🎉 All checks passed! Your code is ready for CI.")
            return recommendations

        if "Lint Check (ruff)" in failed_checks:
            recommendations.append(
                "🔧 Fix linting issues by running: uv run ruff check . --fix"
            )

        if (
            "Format Check (ruff)" in failed_checks
            or "Format Check (black)" in failed_checks
        ):
            recommendations.append(
                "🎨 Fix formatting by running: uv run black . && uv run ruff format ."
            )

        if "Type Check (mypy)" in failed_checks:
            recommendations.append("🔍 Fix type issues shown in mypy output above")

        if "Test Suite (pytest)" in failed_checks:
            recommendations.append(
                "🧪 Fix failing tests - check the pytest output for details"
            )

        if "Documentation Check" in failed_checks:
            recommendations.append(
                "📚 Fix documentation issues - check README.md and docstrings"
            )

        if "Prompt Regression Test" in failed_checks:
            recommendations.append(
                "📝 Fix prompt-related issues - check prompt registry and evaluation system"
            )

        if "Security Scan" in failed_checks:
            recommendations.append(
                "🔒 Review security scan results and update vulnerable dependencies"
            )

        recommendations.append(
            "🔄 Re-run this script after fixing issues to verify they're resolved"
        )

        return recommendations

    def run_specific_check(self, check_name: str) -> bool:
        """Run a specific check only."""
        check_map = {
            "lint": self._run_ruff_check,
            "format": self._run_format_check,
            "black": self._run_black_check,
            "type": self._run_mypy_check,
            "test": self._run_pytest,
            "docs": self._run_doc_check,
            "prompt": self._run_prompt_regression,
            "security": self._run_security_scan,
        }

        if check_name not in check_map:
            print(f"❌ Unknown check: {check_name}")
            print(f"Available checks: {list(check_map.keys())}")
            return False

        print(f"🔍 Running specific check: {check_name}")
        success, details = check_map[check_name]()

        if success:
            print(f"✅ {check_name}: PASSED")
        else:
            print(f"❌ {check_name}: FAILED")
            if details.get("error"):
                print(f"Error: {details['error']}")
            if details.get("output") and self.verbose:
                print(f"Output: {details['output']}")

        return success


def main():
    """Main entry point for the local CI tester."""
    import argparse

    parser = argparse.ArgumentParser(description="Test CI pipeline locally")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--check",
        choices=[
            "lint",
            "format",
            "black",
            "type",
            "test",
            "docs",
            "prompt",
            "security",
        ],
        help="Run only a specific check",
    )
    parser.add_argument(
        "--list-checks", action="store_true", help="List available checks and exit"
    )

    args = parser.parse_args()

    if args.list_checks:
        print("Available CI checks:")
        checks = [
            "lint",
            "format",
            "black",
            "type",
            "test",
            "docs",
            "prompt",
            "security",
        ]
        for check in checks:
            print(f"  • {check}")
        return 0

    tester = LocalCITester(project_root=args.project_root, verbose=args.verbose)

    if args.check:
        success = tester.run_specific_check(args.check)
    else:
        success = tester.run_all_checks()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
