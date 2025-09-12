"""
Pytest refiner service for test execution and refinement.

Handles pytest execution, failure output formatting, and iterative
test refinement until tests pass or max iterations are reached.
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from ....config.models import RefineConfig
from ....ports.refine_port import RefinePort
from ....ports.telemetry_port import TelemetryPort
from ....ports.writer_port import WriterPort
from ....adapters.io.file_status_tracker import FileStatusTracker, FileStatus

logger = logging.getLogger(__name__)


class PytestRefiner:
    """
    Service for pytest execution and test refinement.

    Provides pytest execution, failure formatting, and iterative refinement
    of test files until they pass or maximum iterations are reached.
    """

    def __init__(
        self,
        refine_port: RefinePort,
        telemetry_port: TelemetryPort,
        executor: ThreadPoolExecutor,
        config: RefineConfig | None = None,
        max_concurrent_refines: int = 2,
        backoff_sec: float = 0.2,
        status_tracker: Optional[FileStatusTracker] = None,
        writer_port: WriterPort | None = None,
    ):
        """
        Initialize the pytest refiner.

        Args:
            refine_port: Port for test refinement operations
            telemetry_port: Port for telemetry operations
            executor: Thread pool executor for async operations
            config: Refinement configuration for pytest args and settings
            max_concurrent_refines: Maximum concurrent pytest/refine operations
            backoff_sec: Backoff time between refinement iterations
            status_tracker: Optional file status tracker for live updates
            writer_port: Optional writer port for safe file operations
        """
        self._refine = refine_port
        self._telemetry = telemetry_port
        self._executor = executor
        self._config = config or RefineConfig()
        self._backoff_sec = backoff_sec
        self._status_tracker = status_tracker
        self._writer = writer_port
        
        # Get configurable pytest args with defaults
        self._pytest_args = self._config.pytest_args_for_refinement
        
        # Create semaphore to limit concurrent pytest operations
        self._refine_semaphore = asyncio.Semaphore(max_concurrent_refines)

    async def run_pytest(self, test_path: str) -> dict[str, Any]:
        """
        Run pytest on a specific test file and return results.

        Uses the async_runner abstraction which wraps subprocess patterns
        for async workflows.

        Args:
            test_path: Path to the test file to run

        Returns:
            Dictionary with pytest execution results including stdout, stderr, returncode
        """
        from ....adapters.io.async_runner import run_python_module_async_with_executor

        try:
            # Use configurable pytest arguments for refinement
            pytest_args = [str(test_path)] + list(self._pytest_args)

            # Use the reusable async subprocess abstraction with executor
            stdout, stderr, returncode = await run_python_module_async_with_executor(
                executor=self._executor,
                module_name="pytest",
                args=pytest_args,
                timeout=60,
                raise_on_error=False,  # Handle failures ourselves like refine adapter
            )

            result = {
                "stdout": stdout or "",
                "stderr": stderr or "",
                "returncode": returncode,
                "command": f"python -m pytest {' '.join(pytest_args)}",
            }

            # Add classification flags to help gating refinement
            combined = (stdout or "") + "\n" + (stderr or "")
            result.update(self._classify_pytest_result(combined, returncode))
            return result

        except Exception as e:
            logger.warning("Failed to run pytest for %s: %s", test_path, e)
            result = {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "command": f"python -m pytest {test_path}",
                "error": str(e),
            }
            result.update(self._classify_pytest_result(result["stderr"], -1))
            return result

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
        import re
        
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
    
    def detect_xfail_in_output(self, pytest_output: str) -> bool:
        """
        Detect if pytest output contains XFAIL markers.
        
        Args:
            pytest_output: Combined stdout/stderr from pytest
            
        Returns:
            True if XFAIL markers are detected
        """
        xfail_indicators = [
            "XFAIL",
            "xfailed",
            "@pytest.mark.xfail",
            "pytest.xfail",
            "expected failure",
            "XPASS",  # Also catch unexpected passes of xfail tests
        ]
        
        output_lower = pytest_output.lower()
        for indicator in xfail_indicators:
            if indicator.lower() in output_lower:
                logger.debug(f"Detected XFAIL indicator: {indicator}")
                return True
        
        return False

    def _classify_pytest_result(self, output: str, returncode: int) -> dict[str, Any]:
        """
        Classify pytest outcome to decide if LLM refinement should proceed.

        We skip refinement and do NOT send failures to the LLM when pytest
        didn't actually run tests (collection/usage/internal errors), because
        the LLM cannot fix environment or import problems reliably.
        """
        text = (output or "").lower()
        unrefinable = False
        category = "unknown"

        indicators = {
            "collection_error": ["collected 0 items", "collection error", "errors during collection"],
            "import_error": ["importerror", "module not found", "no module named"],
            "usage_error": ["usage:", "error: not found: ", "unrecognized arguments"],
            "internal_error": ["internal error", "! interrupted: ", "traceback (most recent call last):"],
        }

        for cat, keys in indicators.items():
            if any(k in text for k in keys):
                category = cat
                unrefinable = True
                break

        # If returncode is 0, it's passing regardless
        if returncode == 0:
            category = "passed"
            unrefinable = False

        return {"unrefinable": unrefinable, "failure_category": category}
    
    async def _mark_test_with_bug_info(
        self, test_file: Path, bug_detection: dict[str, Any], failure_output: str
    ) -> None:
        """
        Mark the test file with production bug information.
        
        Args:
            test_file: Path to the test file
            bug_detection: Bug detection information (should contain 'description' from LLM)
            failure_output: The pytest failure output
        """
        try:
            if not test_file.exists():
                logger.warning("Test file %s does not exist, cannot mark with bug info", test_file)
                return
            
            current_content = test_file.read_text(encoding='utf-8')
            
            # Get the bug description from LLM analysis
            bug_description = bug_detection.get('description', 'Production bug detected')
            confidence = bug_detection.get('confidence', 0.9) * 100
            
            # Create a prominent header comment
            bug_marker = f'''"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! PRODUCTION BUG DETECTED - TEST DELIBERATELY FAILING
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! 
!!! LLM Analysis Confidence: {confidence:.0f}%
!!! 
!!! WHAT'S WRONG:
!!! {bug_description}
!!!
!!! This test is failing because the production code has a bug,
!!! not because the test is incorrect. The test expectations are correct
!!! and should NOT be modified to match the buggy behavior.
!!!
!!! FAILURE OUTPUT (truncated):
{self._indent_text(failure_output[:400], "!!! ")}
!!!
!!! ACTION REQUIRED:
!!! 1. Fix the bug in the production code based on the analysis above
!!! 2. Remove this comment block after fixing the bug
!!! 3. The test should pass once the production code is corrected
!!!
!!! See BUG_REPORT_{test_file.stem}.md for complete details
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

import pytest

# Mark test as expected to fail due to production bug
pytestmark = pytest.mark.xfail(
    reason="{bug_description[:150]}",
    strict=True  # Fail if test unexpectedly passes
)

'''
            
            # Prepend the bug marker to the file
            marked_content = bug_marker + current_content
            
            # Write the marked content back
            test_file.write_text(marked_content, encoding='utf-8')
            
            logger.info("Marked test file %s with production bug information", test_file)
            
        except Exception as e:
            logger.error("Failed to mark test file with bug info: %s", e)
    
    async def _annotate_failed_test(
        self,
        test_file: Path,
        failure_output: str,
        reason_status: str,
        iterations: int,
        fix_instructions: str | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Build and insert a high-visibility annotation into the test file when refinement fails.

        Respects RefineConfig:
        - annotate_failed_tests
        - annotation_placement ("top" | "bottom")
        - annotation_include_failure_excerpt
        - annotation_max_failure_chars
        - annotation_style ("docstring" | "hash")
        - include_llm_fix_instructions
        """
        if not self._config.annotate_failed_tests:
            return
            
        if extra is None:
            extra = {}
            
        try:
            if not test_file.exists():
                logger.warning("Test file %s does not exist, cannot annotate", test_file)
                return
            
            current_content = test_file.read_text(encoding='utf-8')
            
            # Check for existing annotation to avoid duplicates
            if "TESTCRAFT_FAILED_REFINEMENT_GUIDE" in current_content:
                logger.debug("Test file %s already has failed refinement annotation", test_file)
                return
            
            # Extract enhanced failure analysis
            failing_tests = self._extract_failing_tests_from_output(failure_output or "")
            failure_context = self._extract_failure_context(failure_output or "")
            
            # Generate enhanced fix instructions
            enhanced_instructions = self._generate_enhanced_fix_instructions(
                failing_tests, failure_context, fix_instructions or ""
            )
            
            # Build annotation content
            banner_lines = [
                "==============================================================================",
                "TEST REFINEMENT FAILED — MANUAL FIX REQUIRED",
                f"File: {test_file}",
                f"Status: {reason_status} | Iterations: {iterations}",
                "# TESTCRAFT_FAILED_REFINEMENT_GUIDE",
                "==============================================================================",
                "",
            ]
            
            # Optional failure excerpt (more targeted now)
            if self._config.annotation_include_failure_excerpt and failure_output:
                # Prioritize showing key failure messages if available
                if failure_context.get("key_messages"):
                    banner_lines.extend([
                        "KEY FAILURE MESSAGES:",
                        '\n'.join(f"  • {msg}" for msg in failure_context["key_messages"][:3]),
                        "",
                    ])
                else:
                    # Fallback to trimmed full output
                    trimmed_failure = failure_output[:self._config.annotation_max_failure_chars]
                    if len(failure_output) > self._config.annotation_max_failure_chars:
                        trimmed_failure += "\n... (truncated)"
                    banner_lines.extend([
                        "FAILURE OUTPUT (trimmed):",
                        trimmed_failure,
                        "",
                    ])
            
            # Enhanced fix guide
            if self._config.include_llm_fix_instructions:
                banner_lines.extend([
                    "ENHANCED FIX GUIDE:",
                    enhanced_instructions,
                    "",
                ])
            elif extra.get("preflight_suggestions"):
                banner_lines.extend([
                    "PREFLIGHT SUGGESTIONS:",
                    self._format_fix_instructions_as_todos(extra["preflight_suggestions"]),
                    "",
                ])
            
            # Optional hints
            if extra.get("active_import_path"):
                banner_lines.extend([
                    f"Active import path: {extra['active_import_path']}",
                    "",
                ])
            
            # Specific test command if we identified failing tests
            if failing_tests:
                test_cmd = f"pytest {failing_tests[0]} -vv --tb=short"
            else:
                test_cmd = f"pytest {test_file} -vv --tb=short"
            
            # Always include instructions
            banner_lines.extend([
                f"Run: {test_cmd}",
                "Remove this block after tests pass.",
            ])
            
            # Format according to style
            if self._config.annotation_style == "docstring":
                banner_content = '"""\n' + '\n'.join(banner_lines) + '\n"""\n\n'
            else:  # hash style
                banner_content = '\n'.join(f"# {line}" for line in banner_lines) + '\n\n'
            
            # Place according to placement setting
            if self._config.annotation_placement == "top":
                new_content = banner_content + current_content
            else:  # bottom
                new_content = current_content + '\n\n' + banner_content
            
            # Write using writer port if available, otherwise fallback
            if self._writer:
                try:
                    await self._write_with_writer_port(test_file, new_content)
                except Exception as e:
                    logger.warning("WriterPort failed, using fallback: %s", e)
                    await self._write_with_fallback(test_file, new_content)
            else:
                await self._write_with_fallback(test_file, new_content)
            
            logger.info("Added failed refinement annotation to %s", test_file)
            
            # Add telemetry
            if self._telemetry:
                with self._telemetry.create_child_span("failed_annotation_added") as span:
                    span.set_attribute("failed_annotation_added", True)
                    span.set_attribute("annotation_style", self._config.annotation_style)
                    span.set_attribute("annotation_placement", self._config.annotation_placement)
                    span.set_attribute("included_llm_instructions", bool(fix_instructions))
                    span.set_attribute("annotation_size", len(banner_content))
                    
        except Exception as e:
            logger.error("Failed to annotate test file with failure info: %s", e)
    
    def _format_fix_instructions_as_todos(self, instructions: str) -> str:
        """Convert fix instructions to TODO checklist format."""
        if not instructions:
            return "No specific instructions available."
        
        lines = instructions.split('\n')
        todo_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Convert bullet points or numbered items to checklist
            if stripped.startswith('- ') or stripped.startswith('* '):
                todo_lines.append(f"- [ ] {stripped[2:].strip()}")
            elif stripped[0].isdigit() and ('. ' in stripped[:4] or ') ' in stripped[:4]):
                # Handle "1. " or "1) " style
                todo_lines.append(f"- [ ] {stripped.split('. ', 1)[-1].split(') ', 1)[-1].strip()}")
            else:
                todo_lines.append(f"- [ ] {stripped}")
        
        return '\n'.join(todo_lines) if todo_lines else instructions
    
    def _extract_failing_tests_from_output(self, failure_output: str) -> list[str]:
        """Extract specific failing test names from pytest output."""
        import re
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
    
    def _extract_failure_context(self, failure_output: str) -> dict[str, Any]:
        """Extract targeted failure context from pytest output."""
        import re
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
    
    def _generate_enhanced_fix_instructions(self, 
                                           failing_tests: list[str], 
                                           failure_context: dict[str, Any], 
                                           original_instructions: str) -> str:
        """Generate enhanced, specific fix instructions."""
        instructions = []
        
        # Header with specific failing tests
        if failing_tests:
            instructions.append("🔍 SPECIFIC FAILING TESTS:")
            for i, test in enumerate(failing_tests[:5]):  # Limit to first 5
                instructions.append(f"   {i+1}. {test}")
            if len(failing_tests) > 5:
                instructions.append(f"   ... and {len(failing_tests) - 5} more tests failing")
            instructions.append("")
        
        # Error analysis section
        if failure_context.get("error_types"):
            instructions.append("❌ ERROR ANALYSIS:")
            for error in failure_context["error_types"][:3]:  # Top 3 errors
                instructions.append(f"   • {error}")
            instructions.append("")
        
        # Import-specific fixes
        if failure_context.get("import_errors"):
            instructions.append("📦 IMPORT FIXES NEEDED:")
            for import_error in failure_context["import_errors"][:2]:
                if "No module named" in import_error:
                    # Extract module name from error
                    module_match = re.search(r"'([^']+)'", import_error)
                    if module_match:
                        module = module_match.group(1)
                        instructions.append(f"   • Missing module '{module}' - add proper mock:")
                        instructions.append(f"     with patch('{module}') as mock_{module.replace('.', '_')}:")
                    else:
                        instructions.append(f"   • Fix import issue: {import_error[:80]}")
                else:
                    instructions.append(f"   • Resolve: {import_error[:80]}")
            instructions.append("")
        
        # Syntax error guidance  
        if failure_context.get("syntax_errors"):
            instructions.append("🔧 SYNTAX FIXES:")
            for syntax_error in failure_context["syntax_errors"][:2]:
                instructions.append(f"   • {syntax_error[:100]}")
            instructions.append("   • Check for missing imports, incorrect indentation, or typos")
            instructions.append("")
        
        # Assertion-specific guidance
        if failure_context.get("assertion_details"):
            instructions.append("⚖️  ASSERTION ANALYSIS:")
            instructions.append(f"   • Failed: {failure_context['assertion_details'][:100]}")
            if failure_context.get("traceback_highlights"):
                instructions.append("   • Context:")
                for highlight in failure_context["traceback_highlights"][:3]:
                    if highlight and len(highlight.strip()) > 5:
                        instructions.append(f"     {highlight[:80]}")
            instructions.append("   • Review expected vs actual values in test logic")
            instructions.append("")
        
        # Prioritized action plan
        instructions.append("📋 STEP-BY-STEP FIX PLAN:")
        
        step = 1
        if failing_tests:
            test_names = [t.split("::")[-1] for t in failing_tests[:2]]
            instructions.append(f"{step}. Focus on these specific tests: {', '.join(test_names)}")
            step += 1
        
        if failure_context.get("import_errors"):
            instructions.append(f"{step}. Fix import issues FIRST (they often block other fixes)")
            step += 1
        
        if failure_context.get("syntax_errors"):
            instructions.append(f"{step}. Resolve syntax errors before running tests")
            step += 1
        
        if failure_context.get("error_types"):
            primary_error = failure_context["error_types"][0].split(":")[0]
            if "Import" in primary_error or "Module" in primary_error:
                instructions.append(f"{step}. Add proper mocks for missing modules using @patch decorator")
            elif "Assertion" in primary_error:
                instructions.append(f"{step}. Debug assertion: check expected values vs actual mock returns")
            elif "AttributeError" in primary_error:
                instructions.append(f"{step}. Fix attribute names and verify mock object configuration")
            elif "SyntaxError" in primary_error:
                instructions.append(f"{step}. Check Python syntax, imports, and indentation")
            else:
                instructions.append(f"{step}. Address {primary_error} by reviewing test setup")
            step += 1
        
        instructions.append(f"{step}. Test fix with: pytest {failing_tests[0] if failing_tests else '[test_file]'} -v")
        instructions.append(f"{step+1}. Verify mocks are properly configured and don't interfere with each other")
        
        # Add meaningful LLM instructions if available
        if (original_instructions and 
            "No obvious canonicalization issues" not in original_instructions and
            len(original_instructions.strip()) > 20):
            instructions.append("")
            instructions.append("🤖 ADDITIONAL AI ANALYSIS:")
            # Extract meaningful parts from original instructions
            for line in original_instructions.split('\n'):
                line = line.strip()
                if line and len(line) > 10 and not line.startswith('- ['):
                    instructions.append(f"   • {line}")
        
        return '\n'.join(instructions)
    
    async def _write_with_writer_port(self, test_file: Path, content: str) -> None:
        """Write using the writer port."""
        self._writer.write_file(file_path=test_file, content=content, overwrite=True)
    
    async def _write_with_fallback(self, test_file: Path, content: str) -> None:
        """Fallback writing method."""
        test_file.write_text(content, encoding='utf-8')
        # Optional: format the content
        try:
            from ....adapters.io.python_formatters import format_python_content
            formatted = format_python_content(content)
            if formatted != content:
                test_file.write_text(formatted, encoding='utf-8')
        except Exception as e:
            logger.debug("Could not format Python content: %s", e)
    
    async def _create_bug_report(
        self, test_path: str, bug_detection: dict[str, Any], 
        failure_output: str, iteration: int
    ) -> None:
        """
        Create a detailed bug report file.
        
        Args:
            test_path: Path to the test file
            bug_detection: Bug detection information (should contain 'description' from LLM)
            failure_output: The pytest failure output
            iteration: Current refinement iteration
        """
        try:
            from datetime import datetime
            
            test_file = Path(test_path)
            report_dir = test_file.parent
            report_name = f"BUG_REPORT_{test_file.stem}.md"
            report_path = report_dir / report_name
            
            timestamp = datetime.now().isoformat()
            
            # Get the bug description from LLM analysis
            bug_description = bug_detection.get('description', 'Production bug detected')
            confidence = bug_detection.get('confidence', 0.9) * 100
            
            report_content = f"""# 🐛 PRODUCTION BUG REPORT

## ⚠️ CRITICAL: Production Bug Detected

**Generated**: {timestamp}
**Test File**: `{test_path}`
**LLM Analysis Confidence**: {confidence:.0f}%
**Refinement Iteration**: {iteration}

---

## 📋 Summary

TestCraft's AI has detected a **production code bug** while attempting to refine this test.

### What's Wrong:
**{bug_description}**

The test expectations appear correct, but the production code is not behaving as expected.

**This is NOT a test issue - this is a production code issue that needs to be fixed.**

## 🔍 Detection Details

### AI Analysis
- **Confidence Level**: {confidence:.0f}%
- **Detection Method**: LLM analysis of test failure patterns
- **Detailed Explanation**: {bug_description}

### Test Failure Output
```
{failure_output}
```

## 🎯 Recommended Actions

1. **DO NOT MODIFY THE TEST** - The test expectations are correct
2. **INVESTIGATE THE PRODUCTION CODE** - Look for the bug in the implementation
3. **FIX THE BUG** in the production code
4. **RE-RUN THE TEST** - It should pass once the bug is fixed
5. **REMOVE THE BUG MARKERS** from the test file after fixing

## 📝 Notes

- The test file has been marked with `pytest.mark.xfail` to indicate this is a known issue
- The test will deliberately fail until the production bug is fixed
- This prevents the bug from being masked by incorrect test modifications

## 🔧 How to Fix

1. Review the failure output above
2. Locate the production code that's being tested
3. Debug why it's returning incorrect values/behavior
4. Fix the implementation
5. Re-run: `pytest {test_path}`
6. Remove the bug report markers from the test file

---

*This report was automatically generated by TestCraft's strict refinement policy.*
*Do not modify tests to hide production bugs!*
"""
            
            # Write the report
            report_path.write_text(report_content, encoding='utf-8')
            
            logger.warning(
                "\n" + "="*80 + "\n"
                "🚨 PRODUCTION BUG DETECTED BY AI! 🚨\n"
                "="*80 + "\n"
                f"Bug report created: {report_path}\n"
                f"Test file marked: {test_path}\n"
                f"AI Analysis: {bug_description}\n"
                f"Confidence: {confidence:.0f}%\n"
                "="*80
            )
            
        except Exception as e:
            logger.error("Failed to create bug report: %s", e)
    
    def _indent_text(self, text: str, prefix: str) -> str:
        """Indent text with a prefix on each line."""
        lines = text.split('\n')
        return '\n'.join(prefix + line for line in lines)
    
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
        import re
        
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

    async def refine_until_pass(
        self,
        test_path: str,
        max_iterations: int,
        build_source_context_fn: Callable[[Path, str], Awaitable[dict[str, Any] | None]],
    ) -> dict[str, Any]:
        """
        Refine a test file through pytest execution and LLM refinement.

        This method implements the complete refinement workflow:
        1. Run pytest to get failure output
        2. If tests pass, return success
        3. If tests fail, use refine port to fix failures
        4. Repeat until max iterations or tests pass
        5. Detect no-change scenarios to avoid infinite loops

        Args:
            test_path: Path to the test file to refine
            max_iterations: Maximum number of refinement iterations
            build_source_context_fn: Function to build source context for refinement

        Returns:
            Dictionary with refinement results including success status,
            iterations used, final pytest status, and any errors
        """
        test_file = Path(test_path)

        # Track content between iterations to detect no-change scenarios
        previous_content = None
        if test_file.exists():
            try:
                previous_content = test_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to read test file %s: %s", test_path, e)

        # Use semaphore to limit concurrent refinement operations
        async with self._refine_semaphore:
            with self._telemetry.create_child_span("refine_test_file") as span:
                span.set_attribute("test_file", test_path)
                span.set_attribute("max_iterations", max_iterations)
                span.set_attribute("backoff_sec", self._backoff_sec)
                
                # Update status to testing
                if self._status_tracker:
                    self._status_tracker.update_file_status(
                        test_path,
                        FileStatus.TESTING,
                        operation="Initial Testing",
                        step="Running pytest to check current status",
                        progress=80.0
                    )

                # Track last fix metadata for annotation purposes
                last_fix_instructions = None
                last_active_import_path = ""
                last_preflight = ""

                for iteration in range(max_iterations):
                    span.set_attribute(f"iteration_{iteration}_started", True)
                    
                    # Update status for each iteration
                    if self._status_tracker:
                        if iteration == 0:
                            self._status_tracker.update_file_status(
                                test_path,
                                FileStatus.TESTING,
                                operation="Testing",
                                step=f"Running initial pytest check",
                                progress=80.0 + (iteration * 5)
                            )
                        else:
                            self._status_tracker.update_file_status(
                                test_path,
                                FileStatus.REFINING,
                                operation=f"Refinement {iteration}",
                                step=f"Running pytest (iteration {iteration + 1})",
                                progress=80.0 + (iteration * 5)
                            )

                    try:
                        # Step 1: Run pytest to get current test status
                        pytest_result = await self.run_pytest(test_path)
                        span.set_attribute(
                            f"iteration_{iteration}_pytest_returncode",
                            pytest_result["returncode"],
                        )

                        # Step 2: Check if tests are now passing
                        if pytest_result["returncode"] == 0:
                            # Check for XFAIL markers if configured to fail on them
                            combined_output = pytest_result.get("stdout", "") + pytest_result.get("stderr", "")
                            
                            if self._config.fail_on_xfail_markers and self.detect_xfail_in_output(combined_output):
                                logger.warning(
                                    "Tests marked with XFAIL detected for %s, treating as failure (fail_on_xfail_markers=True)",
                                    test_path
                                )
                                span.set_attribute("xfail_detected", True)
                                
                                # Create a bug detection dict for XFAIL
                                xfail_bug_detection = {
                                    "suspected": True,
                                    "type": "xfail_marker",
                                    "pattern": "XFAIL/xfailed markers",
                                    "confidence": 0.9,
                                    "description": "Test contains XFAIL markers which may mask production bugs"
                                }
                                
                                # Mark the test and create report
                                await self._mark_test_with_bug_info(
                                    test_file,
                                    xfail_bug_detection,
                                    combined_output
                                )
                                
                                await self._create_bug_report(
                                    test_path,
                                    xfail_bug_detection,
                                    combined_output,
                                    iteration + 1
                                )
                                
                                # Update status to failed due to XFAIL
                                if self._status_tracker:
                                    self._status_tracker.update_file_status(
                                        test_path,
                                        FileStatus.FAILED,
                                        operation="XFAIL Detected",
                                        step="Tests contain xfail markers (potential production bug)",
                                        progress=0.0
                                    )
                                
                                return {
                                    "test_file": test_path,
                                    "success": False,
                                    "iterations": iteration + 1,
                                    "final_status": "xfail_detected",
                                    "error": "Tests contain xfail markers which may mask production bugs",
                                    "last_failure": combined_output,
                                    "suspected_prod_bug": "XFAIL markers detected - tests may be masking real bugs",
                                    "bug_report_created": True,
                                }
                            
                            # Tests pass! Refinement successful
                            span.set_attribute("refinement_successful", True)
                            span.set_attribute("final_iteration", iteration)
                            
                            # Update status to completed
                            if self._status_tracker:
                                self._status_tracker.update_file_status(
                                    test_path,
                                    FileStatus.COMPLETED,
                                    operation="Tests Passing",
                                    step=f"All tests pass after {iteration + 1} iteration(s)",
                                    progress=100.0
                                )
                                self._status_tracker.update_refinement_result(
                                    test_path, iteration + 1, True
                                )

                            return {
                                "test_file": test_path,
                                "success": True,
                                "iterations": iteration + 1,
                                "final_status": "passed",
                                "final_output": pytest_result.get("stdout", ""),
                                "refinement_details": f"Tests passing after {iteration + 1} iteration(s)",
                            }

                        # Step 3: Tests are failing. If failure is unrefinable, stop early.
                        if pytest_result.get("unrefinable", False):
                            category = pytest_result.get("failure_category", "unknown")
                            logger.warning(
                                "Skipping LLM refinement for %s due to unrefinable pytest failure: %s",
                                test_path,
                                category,
                            )
                            span.set_attribute("unrefinable_failure", category)
                            if self._status_tracker:
                                self._status_tracker.update_file_status(
                                    test_path,
                                    FileStatus.FAILED,
                                    operation="Pytest Failure (Unrefinable)",
                                    step=f"{category} — fix environment/imports before refinement",
                                    progress=0.0,
                                )
                            # Annotate failed test with unrefinable failure info
                            if self._config.annotate_failed_tests:
                                await self._annotate_failed_test(
                                    test_file=Path(test_path),
                                    failure_output=self.format_pytest_failure_output(pytest_result),
                                    reason_status=f"unrefinable_{category}",
                                    iterations=iteration + 1,
                                    fix_instructions=None,  # No LLM instructions for unrefinable
                                    extra={}
                                )
                            
                            return {
                                "test_file": test_path,
                                "success": False,
                                "iterations": iteration + 1,
                                "final_status": category,
                                "error": "Unrefinable pytest failure; fix environment/imports or test discovery",
                                "last_failure": self.format_pytest_failure_output(pytest_result),
                            }

                        # Otherwise attempt refinement
                        failure_output = self.format_pytest_failure_output(pytest_result)
                        
                        # Note: Basic bug detection is now just a hint
                        # The real bug analysis comes from the LLM via refine_from_failures
                        
                        # Update status for refinement
                        if self._status_tracker:
                            self._status_tracker.update_file_status(
                                test_path,
                                FileStatus.REFINING,
                                operation=f"Refinement {iteration + 1}",
                                step="Using LLM to fix test failures",
                                progress=85.0 + (iteration * 3)
                            )

                        # Build source context for better refinement
                        # Read current test file content for context
                        try:
                            test_content = test_file.read_text(encoding='utf-8') if test_file.exists() else ""
                        except Exception as e:
                            logger.warning("Failed to read test file %s: %s", test_file, e)
                            test_content = ""
                        
                        source_context = await build_source_context_fn(test_file, test_content)

                        # Use refine port to fix the failures
                        refine_result = self._refine.refine_from_failures(
                            test_file=test_file,
                            failure_output=failure_output,
                            source_context=source_context,
                            max_iterations=self._config.max_retries + 1,  # Use config max_retries
                            timeout_seconds=int(self._config.max_total_minutes * 60),  # Convert minutes to seconds
                        )

                        # Update tracking metadata from refine result
                        last_fix_instructions = refine_result.get("fix_instructions")
                        last_active_import_path = refine_result.get("active_import_path", "")
                        last_preflight = refine_result.get("preflight_suggestions", "")

                        span.set_attribute(
                            f"iteration_{iteration}_refine_success",
                            refine_result.get("success", False),
                        )
                        span.set_attribute(
                            f"iteration_{iteration}_refine_final_status",
                            refine_result.get("final_status", "unknown"),
                        )

                        # Check for early stopping conditions
                        final_status = refine_result.get("final_status")
                        
                        # Handle production bug detection from LLM
                        if final_status == "prod_bug_suspected":
                            suspected_bug_desc = refine_result.get("suspected_prod_bug", "Production bug detected by LLM")
                            logger.warning(
                                "LLM detected production bug on iteration %d for %s: %s",
                                iteration + 1,
                                test_path,
                                suspected_bug_desc
                            )
                            span.set_attribute("prod_bug_suspected", True)
                            span.set_attribute("bug_description", suspected_bug_desc)
                            
                            # Create bug detection dict with LLM's analysis
                            bug_detection = {
                                "suspected": True,
                                "type": "llm_detected",
                                "pattern": "LLM analysis",
                                "confidence": 0.9,  # High confidence since LLM analyzed it
                                "description": suspected_bug_desc
                            }
                            
                            # Mark the test with production bug information
                            await self._mark_test_with_bug_info(
                                test_file,
                                bug_detection,
                                failure_output
                            )
                            
                            # Create a bug report file
                            await self._create_bug_report(
                                test_path,
                                bug_detection,
                                failure_output,
                                iteration + 1
                            )
                            
                            # Update status
                            if self._status_tracker:
                                self._status_tracker.update_file_status(
                                    test_path,
                                    FileStatus.FAILED,
                                    operation="Production Bug Detected by LLM",
                                    step=suspected_bug_desc[:100],  # Truncate for display
                                    progress=0.0
                                )
                            
                            return {
                                "test_file": test_path,
                                "success": False,
                                "iterations": iteration + 1,
                                "final_status": "prod_bug_suspected",
                                "error": f"Production bug detected: {suspected_bug_desc}",
                                "last_failure": failure_output,
                                "suspected_prod_bug": suspected_bug_desc,
                                "bug_report_created": True,
                            }
                        
                        # Handle schema-invalid output - allow single retry per iteration
                        if final_status == "llm_invalid_output":
                            logger.warning(
                                "LLM returned invalid output on iteration %d for %s: %s",
                                iteration + 1,
                                test_path,
                                refine_result.get("error", "Unknown error")
                            )
                            span.set_attribute("invalid_output_iteration", iteration + 1)
                            
                            # Check if schema repair was attempted (from the refine adapter)
                            schema_repaired = refine_result.get("repaired", False)
                            if schema_repaired:
                                logger.info(
                                    "Schema repair was attempted for %s on iteration %d - continuing to next iteration",
                                    test_path, iteration + 1
                                )
                                # Continue to next iteration after repair attempt
                                continue
                            else:
                                # Schema repair wasn't available or failed - this shouldn't happen with new code
                                logger.error(
                                    "Schema validation failed without repair attempt for %s on iteration %d",
                                    test_path, iteration + 1
                                )
                                continue
                        
                        # Handle layered validation statuses that indicate "no real change"
                        no_change_statuses = {
                            "content_identical", 
                            "content_cosmetic_noop", 
                            "content_semantically_identical",
                            "llm_no_change"
                        }
                        
                        if final_status in no_change_statuses:
                            # Check if we should stop on no change (default is True)
                            stop_on_no_change = getattr(self._config, 'stop_on_no_change', True) if hasattr(self, '_config') else True
                            
                            # Also check cosmetic handling config
                            treat_cosmetic_as_no_change = getattr(self._config, 'treat_cosmetic_as_no_change', True) if hasattr(self, '_config') else True
                            
                            # For cosmetic changes, respect the cosmetic config setting
                            if final_status == "content_cosmetic_noop" and not treat_cosmetic_as_no_change:
                                logger.info(
                                    "LLM made cosmetic changes on iteration %d for %s, continuing (treat_cosmetic_as_no_change=False)",
                                    iteration + 1,
                                    test_path,
                                )
                                # Continue to next iteration since cosmetic changes are allowed
                                continue
                            
                            if stop_on_no_change:
                                # Provide a more descriptive message based on the specific status
                                status_messages = {
                                    "content_identical": "LLM returned identical content",
                                    "content_cosmetic_noop": "LLM made only cosmetic changes (whitespace/formatting)",
                                    "content_semantically_identical": "LLM made changes that are semantically equivalent",
                                    "llm_no_change": "LLM explicitly returned no changes"
                                }
                                
                                reason = status_messages.get(final_status, "LLM refinement made no meaningful changes")
                                
                                logger.info(
                                    "%s on iteration %d for %s, stopping early (stop_on_no_change=True)",
                                    reason, iteration + 1, test_path,
                                )
                                span.set_attribute("stopped_reason", final_status)
                                span.set_attribute("no_change_detected", True)
                                span.set_attribute("specific_no_change_reason", reason)
                                
                                # Annotate failed test with no-change failure info
                                if self._config.annotate_failed_tests:
                                    await self._annotate_failed_test(
                                        test_file=Path(test_path),
                                        failure_output=failure_output,
                                        reason_status="no_change_detected",
                                        iterations=iteration + 1,
                                        fix_instructions=last_fix_instructions,
                                        extra={
                                            "active_import_path": last_active_import_path,
                                            "preflight_suggestions": last_preflight,
                                        }
                                    )
                                
                                return {
                                    "test_file": test_path,
                                    "success": False,
                                    "iterations": iteration + 1,
                                    "final_status": "no_change_detected",
                                    "error": reason,
                                    "last_failure": failure_output,
                                    "no_change_details": {
                                        "validation_status": final_status,
                                        "reason": reason,
                                        "diff_snippet": refine_result.get("diff_snippet", ""),
                                    }
                                }
                            else:
                                logger.info(
                                    "LLM returned no meaningful changes on iteration %d for %s, continuing (stop_on_no_change=False)",
                                    iteration + 1,
                                    test_path,
                                )
                                # Continue to next iteration
                                continue
                        
                        # Handle syntax errors - should stop
                        if final_status == "syntax_error":
                            logger.error(
                                "LLM returned syntactically invalid Python on iteration %d for %s",
                                iteration + 1,
                                test_path,
                            )
                            span.set_attribute("stopped_reason", "syntax_error")
                            
                            # Annotate failed test with syntax error failure info
                            if self._config.annotate_failed_tests:
                                await self._annotate_failed_test(
                                    test_file=Path(test_path),
                                    failure_output=failure_output,
                                    reason_status="syntax_error",
                                    iterations=iteration + 1,
                                    fix_instructions=last_fix_instructions,
                                    extra={
                                        "active_import_path": last_active_import_path,
                                        "preflight_suggestions": last_preflight,
                                    }
                                )
                            
                            return {
                                "test_file": test_path,
                                "success": False,
                                "iterations": iteration + 1,
                                "final_status": "syntax_error",
                                "error": refine_result.get("error", "Syntax error in refined content"),
                                "last_failure": failure_output,
                            }

                        if not refine_result.get("success"):
                            logger.warning(
                                "Refinement failed on iteration %d for %s: %s",
                                iteration + 1,
                                test_path,
                                refine_result.get("error", "Unknown error"),
                            )
                            continue

                        # Step 4: Check if content actually changed (no-change detection)
                        refined_content = refine_result.get("refined_content")
                        if (
                            refined_content
                            and refined_content.strip() == (previous_content or "").strip()
                        ):
                            # Content didn't change, avoid infinite loop
                            logger.warning(
                                "No content changes detected in iteration %d for %s, stopping refinement",
                                iteration + 1,
                                test_path,
                            )
                            span.set_attribute("stopped_reason", "no_content_change")

                            # Annotate failed test with no content change failure info
                            if self._config.annotate_failed_tests:
                                await self._annotate_failed_test(
                                    test_file=Path(test_path),
                                    failure_output=failure_output,
                                    reason_status="no_content_change_detected",
                                    iterations=iteration + 1,
                                    fix_instructions=last_fix_instructions,
                                    extra={
                                        "active_import_path": last_active_import_path,
                                        "preflight_suggestions": last_preflight,
                                    }
                                )

                            return {
                                "test_file": test_path,
                                "success": False,
                                "iterations": iteration + 1,
                                "final_status": "no_change_detected",
                                "error": "Refinement made no changes to test content",
                                "last_failure": failure_output,
                            }

                        # Step 5: Content changed, update for next iteration
                        previous_content = refined_content

                        # Step 6: Add exponential backoff only after successful write attempts
                        if iteration < max_iterations - 1 and self._backoff_sec > 0:
                            # Exponential backoff: base * (2 ^ iteration) but cap at reasonable limit
                            backoff_time = min(self._backoff_sec * (2 ** iteration), 5.0)
                            logger.debug(
                                "Applying backoff of %.2fs after successful refinement iteration %d",
                                backoff_time, iteration + 1
                            )
                            await asyncio.sleep(backoff_time)

                    except Exception as e:
                        logger.warning(
                            "Refinement iteration %d failed for %s: %s",
                            iteration + 1,
                            test_path,
                            e,
                        )
                        span.set_attribute(f"iteration_{iteration}_error", str(e))
                        continue

                # Step 6: All refinement attempts exhausted
                span.set_attribute("refinement_successful", False)
                span.set_attribute("stopped_reason", "max_iterations_exceeded")

                # Run final pytest to get latest status
                try:
                    final_pytest = await self.run_pytest(test_path)
                    final_status = "passed" if final_pytest["returncode"] == 0 else "failed"
                    final_output = self.format_pytest_failure_output(final_pytest)
                except Exception:
                    final_status = "unknown"
                    final_output = "Could not determine final test status"

                # Annotate failed test with max iterations exhausted failure info
                if self._config.annotate_failed_tests:
                    await self._annotate_failed_test(
                        test_file=Path(test_path),
                        failure_output=final_output,
                        reason_status="max_iterations_exceeded",
                        iterations=max_iterations,
                        fix_instructions=last_fix_instructions,
                        extra={
                            "active_import_path": last_active_import_path,
                            "preflight_suggestions": last_preflight,
                        }
                    )

                return {
                    "test_file": test_path,
                    "success": False,
                    "iterations": max_iterations,
                    "final_status": final_status,
                    "error": f"Maximum refinement iterations ({max_iterations}) exceeded",
                    "last_failure": final_output,
                }
