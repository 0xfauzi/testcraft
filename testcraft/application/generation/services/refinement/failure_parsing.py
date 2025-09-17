"""
Failure parsing and output formatting.

Handles pytest failure output parsing, import path extraction, and formatting.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class FailureParser:
    """Handles pytest failure output parsing and formatting."""

    @staticmethod
    def extract_import_path_from_failure(failure_output: str) -> str:
        """
        Extract the active import path from pytest failure traceback.
        
        This method looks for import paths in traceback lines to determine the 
        actual module path used at runtime, which is more reliable than source 
        tree aliases when mocking/patching.
        
        Args:
            failure_output: Pytest failure output containing traceback
            
        Returns:
            Detected import path or empty string if not found
        """
        if not failure_output:
            return ""
        
        # Look for common traceback patterns that reveal import paths
        patterns = [
            # Standard traceback lines like: "  File ".../weather_collector/scheduler.py", line 45, in run"
            r'File\s+"[^"]*[/\\]([^/\\]+(?:[/\\][^/\\]+)*\.py)"',
            # Module import errors like: "ModuleNotFoundError: No module named 'weather_collector.scheduler'"
            r"No module named ['\"]([^'\"]+)['\"]",
            # Import traceback patterns like: "from weather_collector.scheduler import JobScheduler"
            r"from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import",
            # AttributeError patterns: "'weather_collector.scheduler' object has no attribute"
            r"['\"]([a-zA-Z_][a-zA-Z0-9_.]*)['\"][^'\"]*object has no attribute",
        ]
        
        detected_paths = []
        
        # Look for traceback import paths
        for pattern in patterns:
            matches = re.findall(pattern, failure_output, re.IGNORECASE)
            for match in matches:
                # Convert file paths to module paths
                if "/" in match or "\\" in match:
                    # Convert file path to module path (remove .py, replace slashes with dots)
                    module_path = match.replace("\\", "/").replace(".py", "").replace("/", ".")
                    # Remove common prefixes that aren't part of the actual module path
                    for prefix in ["src.", "lib.", "app.", "tests."]:
                        if module_path.startswith(prefix):
                            module_path = module_path[len(prefix):]
                else:
                    # Direct module path
                    module_path = match
                
                if module_path and module_path not in detected_paths:
                    detected_paths.append(module_path)
        
        # Prioritize paths from trace lines over error messages
        # Longer, more specific paths are usually better
        if detected_paths:
            # Sort by length (longer = more specific) and take the first one
            detected_paths.sort(key=len, reverse=True)
            active_path = detected_paths[0]
            logger.debug("Detected import path from failure output: %s", active_path)
            return active_path
        
        # Fallback: try to extract from the general structure
        test_runners = ["pytest", "unittest"]
        for runner in test_runners:
            if runner in failure_output.lower():
                # Look for module-style paths in the output
                module_patterns = [
                    r"\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*){1,4})\b"
                ]
                for pattern in module_patterns:
                    matches = re.findall(pattern, failure_output)
                    for match in matches:
                        if (len(match.split(".")) >= 2 and 
                            not match.startswith("test") and
                            match not in ["sys.path", "os.path", "json.loads"]):
                            logger.debug("Fallback import path detection: %s", match)
                            return match
        
        logger.debug("No import path detected from failure output")
        return ""

    def format_pytest_failure_output(self, pytest_result: dict[str, Any]) -> str:
        """
        Format pytest execution results into a clean failure output string.

        Args:
            pytest_result: Results from run_pytest

        Returns:
            Formatted failure output suitable for LLM refinement
        """
        parts = []

        # Add command that was run
        if pytest_result.get("command"):
            parts.append(f"Command: {pytest_result['command']}")
            parts.append("")

        # Add return code
        returncode = pytest_result.get("returncode", -1)
        parts.append(f"Exit Code: {returncode}")
        parts.append("")

        # Add stdout (test results)
        stdout = pytest_result.get("stdout", "").strip()
        if stdout:
            parts.append("Test Output:")
            parts.append(stdout)
            parts.append("")

        # Add stderr (errors)
        stderr = pytest_result.get("stderr", "").strip()
        if stderr:
            parts.append("Error Output:")
            parts.append(stderr)
            parts.append("")

        # Add any additional error information
        if pytest_result.get("error"):
            parts.append(f"Execution Error: {pytest_result['error']}")

        return "\n".join(parts)

    def extract_failing_tests_from_output(self, failure_output: str) -> list[str]:
        """Extract specific failing test names from pytest output."""
        failing_tests = []
        
        if not failure_output:
            return failing_tests
        
        lines = failure_output.split('\n')
        
        # Look for pytest FAILED patterns like "FAILED tests/test_file.py::test_function_name"
        failed_pattern = re.compile(r'FAILED\s+([^:\s]+::[^:\s]+(?:::[^:\s]+)?)')
        for line in lines:
            match = failed_pattern.search(line)
            if match:
                test_ref = match.group(1)
                if test_ref not in failing_tests:
                    failing_tests.append(test_ref)
        
        # Look for short test summary info format
        # Like "tests/test_weatherscheduler.py::test_collect_weather_data_happy_path FAILED"
        summary_pattern = re.compile(r'([^:\s]+::\w+)\s+FAILED')
        for line in lines:
            match = summary_pattern.search(line)
            if match:
                test_ref = match.group(1) 
                if test_ref not in failing_tests:
                    failing_tests.append(test_ref)
                
        # Look for test function names in failure section headers
        failure_header_pattern = re.compile(r'^_{10,}\s+(test_\w+)\s+_{10,}')
        for line in lines:
            match = failure_header_pattern.search(line)
            if match:
                test_func = match.group(1)
                # Try to find corresponding file
                for prev_line in lines[max(0, lines.index(line)-10):lines.index(line)]:
                    if '.py' in prev_line and '::' in prev_line:
                        file_match = re.search(r'([^:\s]+\.py)', prev_line)
                        if file_match:
                            full_ref = f"{file_match.group(1)}::{test_func}"
                            if full_ref not in failing_tests:
                                failing_tests.append(full_ref)
                            break
        
        return list(set(failing_tests))  # Remove duplicates

    def extract_failure_context(self, failure_output: str) -> dict[str, Any]:
        """Extract targeted failure context from pytest output."""
        context = {
            "error_types": [],
            "key_messages": [],
            "assertion_details": "",
            "import_errors": [],
            "traceback_highlights": [],
            "syntax_errors": []
        }
        
        if not failure_output:
            return context
        
        lines = failure_output.split('\n')
        
        # Extract error types and messages
        error_patterns = [
            r'(\w*Error): (.+)',
            r'(\w*Exception): (.+)', 
            r'(AssertionError): (.+)',
            r'(ImportError|ModuleNotFoundError): (.+)',
            r'(SyntaxError): (.+)'
        ]
        
        for line in lines:
            for pattern in error_patterns:
                match = re.search(pattern, line)
                if match:
                    error_type = match.group(1)
                    error_msg = match.group(2).strip()
                    full_error = f"{error_type}: {error_msg}"
                    
                    if full_error not in context["error_types"]:
                        context["error_types"].append(full_error)
                        
                    if error_type in ["ImportError", "ModuleNotFoundError"]:
                        context["import_errors"].append(error_msg)
                    elif error_type == "SyntaxError":
                        context["syntax_errors"].append(error_msg)
        
        # Extract assertion failures with context
        for i, line in enumerate(lines):
            if ('assert' in line.lower() and 
                ('failed' in line.lower() or '==' in line or 'is not' in line or 'is ' in line)):
                context["assertion_details"] = line.strip()
                # Get surrounding context (up to 3 lines before and after)
                start = max(0, i-3)
                end = min(len(lines), i+4)
                context["traceback_highlights"] = [l.strip() for l in lines[start:end] if l.strip()]
                break
        
        # Extract key failure messages
        key_indicators = [
            'FAILED', 'ERROR', 'failed with exit code', 'ModuleNotFoundError', 
            'ImportError', 'No module named', 'SyntaxError', 'AttributeError'
        ]
        for line in lines:
            for indicator in key_indicators:
                if indicator in line:
                    clean_line = line.strip()
                    if (len(clean_line) > 15 and 
                        clean_line not in context["key_messages"] and
                        len(context["key_messages"]) < 5):  # Limit to top 5 messages
                        context["key_messages"].append(clean_line[:150])  # Limit length
                    break
        
        return context

    def detect_suspected_prod_bug(self, failure_output: str) -> dict[str, Any] | None:
        """
        DEPRECATED: Basic pattern matching for production bugs.
        This is only used as a fallback when LLM analysis is not available.
        The LLM should be the primary source of bug detection and description.
        
        Args:
            failure_output: Formatted pytest failure output
            
        Returns:
            Dictionary with bug detection info or None
        """
        # This method is now primarily a fallback for when LLM analysis isn't available
        # The real bug detection and description should come from the LLM's analysis
        
        # Just do very basic checks to see if this MIGHT be a production bug
        # The LLM will provide the actual detailed analysis
        
        # Look for common error patterns that suggest production issues
        prod_bug_indicators = [
            "AssertionError",  # Value mismatches
            "AttributeError",  # Missing attributes
            "TypeError",       # Type issues
            "KeyError",        # Missing keys
            "IndexError",      # Index issues
            "ZeroDivisionError",  # Division errors
            "ValueError",      # Value errors
        ]
        
        for indicator in prod_bug_indicators:
            if indicator in failure_output:
                # Don't try to extract details - let the LLM do that
                return {
                    "suspected": True,
                    "type": "potential_prod_bug",
                    "pattern": indicator,
                    "confidence": 0.5,  # Low confidence - this is just a hint
                    "description": "Potential production bug detected - awaiting LLM analysis for details",
                }
        
        return None
