"""
Test annotation and bug marking logic.

Handles test file annotation with failure information and production bug marking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .....config.models import RefineConfig
from .....ports.telemetry_port import TelemetryPort
from .....ports.writer_port import WriterPort
from .failure_parsing import FailureParser

logger = logging.getLogger(__name__)


class TestAnnotator:
    """Handles test file annotation and bug marking."""

    def __init__(
        self, 
        config: RefineConfig, 
        telemetry_port: TelemetryPort | None = None,
        writer_port: WriterPort | None = None
    ):
        """
        Initialize the test annotator.

        Args:
            config: Refinement configuration
            telemetry_port: Optional telemetry port for spans
            writer_port: Optional writer port for safe file operations
        """
        self._config = config
        self._telemetry = telemetry_port
        self._writer = writer_port
        self._failure_parser = FailureParser()

    async def mark_test_with_bug_info(
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

    async def annotate_failed_test(
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
            failing_tests = self._failure_parser.extract_failing_tests_from_output(failure_output or "")
            failure_context = self._failure_parser.extract_failure_context(failure_output or "")
            
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

    async def create_bug_report(
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
                    import re
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
        if self._writer:
            self._writer.write_file(file_path=test_file, content=content, overwrite=True)

    async def _write_with_fallback(self, test_file: Path, content: str) -> None:
        """Fallback writing method."""
        test_file.write_text(content, encoding='utf-8')
        # Optional: format the content
        try:
            from .....adapters.io.python_formatters import format_python_content
            formatted = format_python_content(content)
            if formatted != content:
                test_file.write_text(formatted, encoding='utf-8')
        except Exception as e:
            logger.debug("Could not format Python content: %s", e)

    def _indent_text(self, text: str, prefix: str) -> str:
        """Indent text with a prefix on each line."""
        lines = text.split('\n')
        return '\n'.join(prefix + line for line in lines)
