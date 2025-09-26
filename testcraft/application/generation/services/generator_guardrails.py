"""
Generator guardrails for test generation validation.

Provides pre-emit and post-generation validation to catch common issues:
- Wrong import paths
- ORM model instantiation in decorators
- Missing safety patterns
- Loop termination issues
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class ValidationIssue:
    """Represents a validation issue found in generated test code."""

    def __init__(
        self,
        severity: str,
        category: str,
        message: str,
        line_number: int | None = None,
        suggested_fix: str | None = None,
    ):
        self.severity = severity  # "error", "warning", "info"
        self.category = category  # "import", "instantiation", "safety", etc.
        self.message = message
        self.line_number = line_number
        self.suggested_fix = suggested_fix

    def __str__(self) -> str:
        location = f" (line {self.line_number})" if self.line_number else ""
        return f"{self.severity.upper()}: {self.message}{location}"


class GeneratorGuardrails:
    """
    Validation system for generated test code.

    Performs static analysis to catch common issues before tests are written.
    """

    def __init__(self, enriched_context: dict[str, Any]) -> None:
        """
        Initialize guardrails with enriched context.

        Args:
            enriched_context: Context from EnrichedContextBuilder
        """
        self.enriched_context = enriched_context
        self.packaging = enriched_context.get("packaging", {})
        self.entities = enriched_context.get("entities", {})
        self.safety_rules = enriched_context.get("test_safety_rules", [])

    def validate_generated_test(
        self, test_content: str
    ) -> tuple[bool, list[ValidationIssue]]:
        """
        Validate generated test content.

        Args:
            test_content: Generated test code as string

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        try:
            # Parse the test content
            try:
                tree = ast.parse(test_content)
            except SyntaxError as e:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="syntax",
                        message=f"Syntax error in generated code: {e}",
                        line_number=getattr(e, "lineno", None),
                    )
                )
                return False, issues

            # Run validation checks
            issues.extend(self._validate_imports(test_content, tree))
            issues.extend(self._validate_canonical_import_first(test_content, tree))
            issues.extend(self._validate_instantiation_safety(test_content, tree))
            issues.extend(self._validate_loop_safety(test_content, tree))
            issues.extend(self._validate_mocking_patterns(test_content, tree))
            issues.extend(self._validate_parametrization_safety(test_content, tree))

            # Determine if validation passes
            has_errors = any(issue.severity == "error" for issue in issues)
            is_valid = not has_errors

            return is_valid, issues

        except Exception as e:
            logger.warning("Validation failed with exception: %s", e)
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="validation",
                    message=f"Validation system error: {e}",
                )
            )
            return False, issues

    def auto_fix_issues(
        self, test_content: str, issues: list[ValidationIssue]
    ) -> tuple[str, list[ValidationIssue]]:
        """
        Attempt to automatically fix validation issues.

        Args:
            test_content: Original test content
            issues: List of validation issues

        Returns:
            Tuple of (fixed_content, remaining_issues)
        """
        fixed_content = test_content
        remaining_issues = []

        for issue in issues:
            if issue.suggested_fix and issue.category == "import":
                # Fix import issues
                fixed_content = self._apply_import_fix(fixed_content, issue)
            elif issue.category == "instantiation" and issue.suggested_fix:
                # Fix instantiation issues
                fixed_content = self._apply_instantiation_fix(fixed_content, issue)
            else:
                # Cannot auto-fix this issue
                remaining_issues.append(issue)

        return fixed_content, remaining_issues

    def _validate_imports(self, content: str, tree: ast.AST) -> list[ValidationIssue]:
        """Validate import statements."""
        issues = []

        disallowed_prefixes = self.packaging.get("disallowed_import_prefixes", [])

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    for prefix in disallowed_prefixes:
                        if module_name.startswith(prefix):
                            canonical_import = self.enriched_context.get(
                                "imports", {}
                            ).get("canonical_import")
                            suggested_fix = (
                                canonical_import
                                if canonical_import
                                else f"Remove '{prefix}' prefix"
                            )

                            issues.append(
                                ValidationIssue(
                                    severity="error",
                                    category="import",
                                    message=f"Disallowed import prefix '{prefix}' in '{module_name}'",
                                    line_number=getattr(node, "lineno", None),
                                    suggested_fix=suggested_fix,
                                )
                            )

            elif isinstance(node, ast.ImportFrom) and node.module:
                module_name = node.module
                for prefix in disallowed_prefixes:
                    if module_name.startswith(prefix):
                        canonical_import = self.enriched_context.get("imports", {}).get(
                            "canonical_import"
                        )
                        suggested_fix = (
                            canonical_import
                            if canonical_import
                            else f"Remove '{prefix}' prefix"
                        )

                        issues.append(
                            ValidationIssue(
                                severity="error",
                                category="import",
                                message=f"Disallowed import prefix '{prefix}' in 'from {module_name}'",
                                line_number=getattr(node, "lineno", None),
                                suggested_fix=suggested_fix,
                            )
                        )

        return issues

    def _validate_canonical_import_first(
        self, content: str, tree: ast.AST
    ) -> list[ValidationIssue]:
        """Validate that canonical import is first non-comment import."""
        issues = []

        canonical_import = self.enriched_context.get("imports", {}).get("target_import")
        if not canonical_import:
            # No canonical import defined, skip this validation
            return issues

        # Find first non-comment import
        first_import = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom) and getattr(
                node, "module", None
            ):
                # Check if it's a comment by examining the source line
                lines = content.split("\n")
                line_num = getattr(node, "lineno", 1) - 1  # 0-based index
                if 0 <= line_num < len(lines):
                    line = lines[line_num].strip()
                    if not line.startswith("#") and line:  # Not a comment and not empty
                        first_import = node
                        break

        if first_import is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="import",
                    message="No import statements found in generated test",
                    suggested_fix=f"Add 'import {canonical_import}' as the first import",
                )
            )
            return issues

        # Check if first import matches canonical import
        if isinstance(first_import, ast.Import):
            actual_import = first_import.names[0].name
        else:  # ast.ImportFrom
            actual_import = first_import.module or ""

        if actual_import != canonical_import:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="import",
                    message=f"First import '{actual_import}' != canonical import '{canonical_import}'",
                    line_number=getattr(first_import, "lineno", None),
                    suggested_fix=f"Change first import to '{canonical_import}'",
                )
            )

        return issues

    def _validate_instantiation_safety(
        self, content: str, tree: ast.AST
    ) -> list[ValidationIssue]:
        """Validate that ORM models are not instantiated unsafely."""
        issues = []

        # Find ORM models that shouldn't be instantiated
        unsafe_entities = {
            name: info
            for name, info in self.entities.items()
            if not info.get("instantiate_real", True)
        }

        if not unsafe_entities:
            return issues

        # Check for instantiation in parametrize decorators
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if self._is_parametrize_decorator(decorator):
                        # Check if any unsafe entities are instantiated in the decorator
                        for entity_name in unsafe_entities:
                            if self._has_entity_instantiation(decorator, entity_name):
                                issues.append(
                                    ValidationIssue(
                                        severity="error",
                                        category="instantiation",
                                        message=f"ORM model '{entity_name}' instantiated in @pytest.mark.parametrize",
                                        line_number=getattr(decorator, "lineno", None),
                                        suggested_fix=f"Use flags in parametrize, create {entity_name} stub in test body",
                                    )
                                )

        return issues

    def _validate_loop_safety(
        self, content: str, tree: ast.AST
    ) -> list[ValidationIssue]:
        """Validate that loops have termination conditions."""
        issues = []

        # Look for while loops that might run indefinitely
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                # Check if it's a common infinite loop pattern
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    # while True: loop
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            category="loop_safety",
                            message="Infinite 'while True:' loop detected - ensure proper mocking for termination",
                            line_number=getattr(node, "lineno", None),
                            suggested_fix="Mock time.sleep or use KeyboardInterrupt to break loop",
                        )
                    )
                elif isinstance(node.test, ast.Name) and node.test.id == "running":
                    # while running: loop
                    issues.append(
                        ValidationIssue(
                            severity="info",
                            category="loop_safety",
                            message="State-based loop detected - ensure 'running' flag is properly controlled",
                            line_number=getattr(node, "lineno", None),
                            suggested_fix="Mock or control the 'running' variable in tests",
                        )
                    )

        return issues

    def _validate_mocking_patterns(
        self, content: str, tree: ast.AST
    ) -> list[ValidationIssue]:
        """Validate that external dependencies are properly mocked."""
        issues = []

        boundaries = self.enriched_context.get("boundaries_to_mock", {})

        # Check if boundary functions are used without mocking
        for category, boundary_items in boundaries.items():
            for item in boundary_items:
                if item in content:
                    # Check if there's a corresponding mock
                    mock_patterns = [
                        f"mock_{item}",
                        "monkeypatch.setattr",
                        "@patch",
                        "Mock()",
                        "MagicMock()",
                    ]

                    has_mock = any(pattern in content for pattern in mock_patterns)

                    if not has_mock:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                category="mocking",
                                message=f"'{item}' used without apparent mocking ({category} boundary)",
                                suggested_fix=f"Add mock for {item} to prevent side effects",
                            )
                        )

        return issues

    def _validate_parametrization_safety(
        self, content: str, tree: ast.AST
    ) -> list[ValidationIssue]:
        """Validate parametrization patterns for safety."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if self._is_parametrize_decorator(decorator):
                        # Check for complex object construction in parametrize
                        if self._has_complex_construction(decorator):
                            issues.append(
                                ValidationIssue(
                                    severity="warning",
                                    category="parametrization",
                                    message="Complex object construction in @pytest.mark.parametrize",
                                    line_number=getattr(decorator, "lineno", None),
                                    suggested_fix="Use simple values in parametrize, construct objects in test body",
                                )
                            )

        return issues

    def _is_parametrize_decorator(self, decorator: ast.AST) -> bool:
        """Check if a decorator is pytest.mark.parametrize."""
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute):
                # pytest.mark.parametrize
                if (
                    isinstance(decorator.func.value, ast.Attribute)
                    and isinstance(decorator.func.value.value, ast.Name)
                    and decorator.func.value.value.id == "pytest"
                    and decorator.func.value.attr == "mark"
                    and decorator.func.attr == "parametrize"
                ):
                    return True
                # mark.parametrize (if mark imported directly)
                elif (
                    isinstance(decorator.func.value, ast.Name)
                    and decorator.func.value.id == "mark"
                    and decorator.func.attr == "parametrize"
                ):
                    return True
        elif isinstance(decorator, ast.Attribute):
            # @parametrize (if imported directly)
            if decorator.attr == "parametrize":
                return True
        elif isinstance(decorator, ast.Name):
            # @parametrize (if imported as parametrize)
            if decorator.id == "parametrize":
                return True

        return False

    def _has_entity_instantiation(self, decorator: ast.AST, entity_name: str) -> bool:
        """Check if an entity is instantiated in a decorator."""
        for node in ast.walk(decorator):
            if isinstance(node, ast.Call):
                # Direct call: Entity(...)
                if isinstance(node.func, ast.Name) and node.func.id == entity_name:
                    return True
                # Attribute call: module.Entity(...)
                elif (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == entity_name
                ):
                    return True
        return False

    def _has_complex_construction(self, decorator: ast.AST) -> bool:
        """Check if decorator has complex object construction."""
        for node in ast.walk(decorator):
            if isinstance(node, ast.Call):
                # Function calls in parametrize are potentially complex
                if isinstance(node.func, ast.Name):
                    # Skip simple built-ins
                    if node.func.id not in {
                        "list",
                        "dict",
                        "tuple",
                        "set",
                        "str",
                        "int",
                        "float",
                        "bool",
                    }:
                        return True
        return False

    def _apply_import_fix(self, content: str, issue: ValidationIssue) -> str:
        """Apply automatic fix for import issues."""
        if not issue.suggested_fix:
            return content

        # Simple regex-based fix for import statements
        disallowed_prefixes = self.packaging.get("disallowed_import_prefixes", [])

        for prefix in disallowed_prefixes:
            # Fix "import src.module" -> "import module"
            pattern = rf"import {re.escape(prefix)}(\w+)"
            replacement = r"import \1"
            content = re.sub(pattern, replacement, content)

            # Fix "from src.module import" -> "from module import"
            pattern = rf"from {re.escape(prefix)}(\w+)"
            replacement = r"from \1"
            content = re.sub(pattern, replacement, content)

        return content

    def _apply_instantiation_fix(self, content: str, issue: ValidationIssue) -> str:
        """Apply automatic fix for instantiation issues."""
        # This would be more complex - for now, just return original content
        # In practice, this might involve rewriting parametrize decorators
        return content


class TestContentValidator:
    """
    High-level validator for test content with auto-fixing capabilities.
    """

    @staticmethod
    def validate_and_fix(
        test_content: str, enriched_context: dict[str, Any]
    ) -> tuple[str, bool, list[ValidationIssue]]:
        """
        Validate test content and attempt to auto-fix issues.

        Args:
            test_content: Generated test code
            enriched_context: Context from EnrichedContextBuilder

        Returns:
            Tuple of (fixed_content, is_valid, all_issues_found)

        Note: all_issues_found includes both fixed and unfixed issues for telemetry
        """
        guardrails = GeneratorGuardrails(enriched_context)

        # Initial validation
        is_valid, issues = guardrails.validate_generated_test(test_content)

        if not is_valid:
            # Attempt auto-fixes
            fixed_content, remaining_issues = guardrails.auto_fix_issues(
                test_content, issues
            )

            # Re-validate after fixes
            if remaining_issues != issues:  # Some fixes were applied
                final_valid, final_issues = guardrails.validate_generated_test(
                    fixed_content
                )

                # Mark fixed issues for telemetry
                fixed_issues = []
                for original_issue in issues:
                    if original_issue not in remaining_issues:
                        # This issue was fixed
                        fixed_issue = ValidationIssue(
                            severity="info",
                            category=f"{original_issue.category}_fixed",
                            message=f"Auto-fixed: {original_issue.message}",
                            line_number=original_issue.line_number,
                            suggested_fix=original_issue.suggested_fix,
                        )
                        fixed_issues.append(fixed_issue)

                # Return all issues (fixed + remaining) for telemetry
                all_issues = fixed_issues + final_issues
                return fixed_content, final_valid, all_issues

            return fixed_content, is_valid, remaining_issues

        return test_content, is_valid, issues
