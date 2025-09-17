"""
AST-based Test File Classifier - Tier 2A test discovery.

Classifies Python files as test files, support files, or regular source files
using Abstract Syntax Tree analysis. This provides more accurate classification
than simple filename patterns.
"""

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClassificationSignals:
    """Signals detected during AST analysis."""
    
    # Test indicators
    has_test_functions: bool = False
    has_test_classes: bool = False  
    has_unittest_subclass: bool = False
    has_pytest_marks: bool = False
    has_pytest_fixtures: bool = False
    has_pytest_imports: bool = False
    has_hypothesis_decorators: bool = False
    has_assert_statements: bool = False
    
    # Support indicators  
    has_fixtures_only: bool = False
    is_conftest: bool = False
    has_fixture_definitions: bool = False
    
    # Counts for decision making
    test_function_count: int = 0
    test_class_count: int = 0
    fixture_count: int = 0
    assert_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_indicators": {
                "has_test_functions": self.has_test_functions,
                "has_test_classes": self.has_test_classes,
                "has_unittest_subclass": self.has_unittest_subclass,
                "has_pytest_marks": self.has_pytest_marks,
                "has_pytest_fixtures": self.has_pytest_fixtures,
                "has_pytest_imports": self.has_pytest_imports,
                "has_hypothesis_decorators": self.has_hypothesis_decorators,
                "has_assert_statements": self.has_assert_statements,
            },
            "support_indicators": {
                "has_fixtures_only": self.has_fixtures_only,
                "is_conftest": self.is_conftest,
                "has_fixture_definitions": self.has_fixture_definitions,
            },
            "counts": {
                "test_function_count": self.test_function_count,
                "test_class_count": self.test_class_count,
                "fixture_count": self.fixture_count,
                "assert_count": self.assert_count,
            }
        }


@dataclass
class Classification:
    """Result of file classification."""
    
    is_test: bool
    is_support: bool
    confidence: float  # 0.0 to 1.0
    signals: ClassificationSignals
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_test": self.is_test,
            "is_support": self.is_support,
            "confidence": self.confidence,
            "signals": self.signals.to_dict(),
        }


class TestFileClassifier:
    """
    AST-based classifier for identifying test files and support files.
    
    Uses Abstract Syntax Tree analysis to detect test patterns, fixtures,
    and other testing-related constructs to accurately classify Python files.
    """
    
    def __init__(self):
        """Initialize the test file classifier."""
        self._cache: dict[str, Classification] = {}
        
        # Test function/class naming patterns
        self.test_function_patterns = ["test_", "_test"]
        self.test_class_patterns = ["Test"]
        
        # Known testing framework imports
        self.pytest_imports = {
            "pytest", "pytest.fixture", "pytest.mark", "_pytest",
        }
        
        self.unittest_imports = {
            "unittest", "unittest.TestCase", "unittest.mock",
        }
        
        self.hypothesis_imports = {
            "hypothesis", "hypothesis.given", "hypothesis.strategies",
        }
        
        # Fixture-related patterns
        self.fixture_decorators = {
            "pytest.fixture", "fixture", "pytest_fixture",
        }
        
        # Support file patterns
        self.support_file_names = {
            "conftest.py", "fixtures.py", "test_utils.py", "testing_utils.py",
        }
    
    def classify(self, file_path: Path) -> Classification:
        """
        Classify a Python file as test, support, or neither.
        
        Args:
            file_path: Path to the Python file to classify
            
        Returns:
            Classification result with is_test, is_support flags and confidence
        """
        # Check cache first (keyed by file path and mtime)
        cache_key = self._get_cache_key(file_path)
        if cache_key in self._cache:
            logger.debug(f"Using cached classification for {file_path}")
            return self._cache[cache_key]
        
        try:
            # Parse the file with AST
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))
            
            # Analyze AST to detect signals
            signals = self._analyze_ast(tree, file_path)
            
            # Make classification decision
            classification = self._classify_from_signals(signals, file_path)
            
            # Cache the result
            self._cache[cache_key] = classification
            
            return classification
            
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            # Return conservative classification for unparseable files
            return Classification(
                is_test=False,
                is_support=False,
                confidence=0.0,
                signals=ClassificationSignals()
            )
            
        except Exception as e:
            logger.warning(f"Unexpected error classifying {file_path}: {e}")
            return Classification(
                is_test=False,
                is_support=False,
                confidence=0.0,
                signals=ClassificationSignals()
            )
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path) -> ClassificationSignals:
        """Analyze AST to detect classification signals."""
        signals = ClassificationSignals()
        
        # Check if this is conftest.py
        signals.is_conftest = file_path.name == "conftest.py"
        
        # Walk the AST to collect signals
        for node in ast.walk(tree):
            self._analyze_node(node, signals)
        
        # Post-process signals
        self._post_process_signals(signals)
        
        return signals
    
    def _analyze_node(self, node: ast.AST, signals: ClassificationSignals) -> None:
        """Analyze a single AST node for classification signals."""
        
        if isinstance(node, ast.FunctionDef):
            self._analyze_function(node, signals)
            
        elif isinstance(node, ast.ClassDef):
            self._analyze_class(node, signals)
            
        elif isinstance(node, ast.Import):
            self._analyze_import(node, signals)
            
        elif isinstance(node, ast.ImportFrom):
            self._analyze_import_from(node, signals)
            
        elif isinstance(node, ast.Assert):
            signals.assert_count += 1
            signals.has_assert_statements = True
    
    def _analyze_function(self, node: ast.FunctionDef, signals: ClassificationSignals) -> None:
        """Analyze a function definition for test/fixture patterns."""
        func_name = node.name
        
        # Check for test function naming patterns
        if any(pattern in func_name for pattern in self.test_function_patterns):
            signals.has_test_functions = True
            signals.test_function_count += 1
        
        # Check for pytest fixtures
        for decorator in node.decorator_list:
            if self._is_fixture_decorator(decorator):
                signals.has_pytest_fixtures = True
                signals.has_fixture_definitions = True
                signals.fixture_count += 1
            elif self._is_pytest_mark_decorator(decorator):
                signals.has_pytest_marks = True
            elif self._is_hypothesis_decorator(decorator):
                signals.has_hypothesis_decorators = True
    
    def _analyze_class(self, node: ast.ClassDef, signals: ClassificationSignals) -> None:
        """Analyze a class definition for test patterns."""
        class_name = node.name
        
        # Check for test class naming patterns
        if any(class_name.startswith(pattern) for pattern in self.test_class_patterns):
            signals.has_test_classes = True
            signals.test_class_count += 1
        
        # Check for unittest.TestCase inheritance
        for base in node.bases:
            if self._is_unittest_testcase(base):
                signals.has_unittest_subclass = True
                signals.has_test_classes = True
                signals.test_class_count += 1
    
    def _analyze_import(self, node: ast.Import, signals: ClassificationSignals) -> None:
        """Analyze import statement for testing framework imports."""
        for alias in node.names:
            module_name = alias.name
            
            if any(pytest_module in module_name for pytest_module in self.pytest_imports):
                signals.has_pytest_imports = True
            elif any(unittest_module in module_name for unittest_module in self.unittest_imports):
                # unittest imports don't necessarily indicate test files
                pass
            elif any(hyp_module in module_name for hyp_module in self.hypothesis_imports):
                signals.has_hypothesis_decorators = True
    
    def _analyze_import_from(self, node: ast.ImportFrom, signals: ClassificationSignals) -> None:
        """Analyze from-import statement for testing framework imports."""
        if not node.module:
            return
        
        module_name = node.module
        
        if any(pytest_module in module_name for pytest_module in self.pytest_imports):
            signals.has_pytest_imports = True
            
            # Check for specific pytest imports
            for alias in node.names:
                if alias.name in ["fixture", "mark"]:
                    signals.has_pytest_fixtures = True
        
        elif any(hyp_module in module_name for hyp_module in self.hypothesis_imports):
            signals.has_hypothesis_decorators = True
    
    def _is_fixture_decorator(self, decorator: ast.expr) -> bool:
        """Check if decorator is a pytest fixture."""
        if isinstance(decorator, ast.Name):
            return decorator.id in self.fixture_decorators
        elif isinstance(decorator, ast.Attribute):
            return self._get_attribute_name(decorator) in self.fixture_decorators
        elif isinstance(decorator, ast.Call):
            # Handle @pytest.fixture() or @fixture()
            func = decorator.func
            if isinstance(func, ast.Name):
                return func.id in self.fixture_decorators
            elif isinstance(func, ast.Attribute):
                return self._get_attribute_name(func) in self.fixture_decorators
        return False
    
    def _is_pytest_mark_decorator(self, decorator: ast.expr) -> bool:
        """Check if decorator is a pytest mark."""
        if isinstance(decorator, ast.Attribute):
            attr_name = self._get_attribute_name(decorator)
            return attr_name.startswith("pytest.mark.")
        elif isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Attribute):
                attr_name = self._get_attribute_name(func)
                return attr_name.startswith("pytest.mark.")
        return False
    
    def _is_hypothesis_decorator(self, decorator: ast.expr) -> bool:
        """Check if decorator is from hypothesis library."""
        if isinstance(decorator, ast.Name):
            return decorator.id == "given"
        elif isinstance(decorator, ast.Attribute):
            attr_name = self._get_attribute_name(decorator)
            return "hypothesis" in attr_name or attr_name.startswith("given")
        elif isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name):
                return func.id == "given"
            elif isinstance(func, ast.Attribute):
                attr_name = self._get_attribute_name(func)
                return "hypothesis" in attr_name or attr_name.startswith("given")
        return False
    
    def _is_unittest_testcase(self, base: ast.expr) -> bool:
        """Check if base class is unittest.TestCase."""
        if isinstance(base, ast.Name):
            return base.id == "TestCase"
        elif isinstance(base, ast.Attribute):
            attr_name = self._get_attribute_name(base)
            return attr_name in ["unittest.TestCase", "TestCase"]
        return False
    
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name from AST node."""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
        
        return ".".join(reversed(parts))
    
    def _post_process_signals(self, signals: ClassificationSignals) -> None:
        """Post-process signals to determine additional flags."""
        # Check if file has fixtures but no tests (support file pattern)
        if signals.fixture_count > 0 and signals.test_function_count == 0 and signals.test_class_count == 0:
            signals.has_fixtures_only = True
    
    def _classify_from_signals(self, signals: ClassificationSignals, file_path: Path) -> Classification:
        """Make classification decision based on detected signals."""
        
        # Strong test indicators
        test_score = 0.0
        
        if signals.has_test_functions:
            test_score += 3.0
        if signals.has_test_classes:
            test_score += 3.0
        if signals.has_unittest_subclass:
            test_score += 3.0
        if signals.has_pytest_marks:
            test_score += 2.0
        if signals.has_hypothesis_decorators:
            test_score += 2.0
        if signals.has_assert_statements and (signals.has_test_functions or signals.has_test_classes):
            test_score += 1.0
        if signals.has_pytest_imports and (signals.has_test_functions or signals.has_test_classes):
            test_score += 1.0
        
        # Support file indicators
        support_score = 0.0
        
        if signals.is_conftest:
            support_score += 5.0  # conftest.py is always support
        if signals.has_fixtures_only:
            support_score += 3.0
        if file_path.name in self.support_file_names:
            support_score += 2.0
        if signals.has_fixture_definitions and not (signals.has_test_functions or signals.has_test_classes):
            support_score += 2.0
        
        # Make classification decision
        is_test = test_score >= 3.0
        is_support = support_score >= 2.0 or signals.is_conftest
        
        # Calculate confidence
        max_score = max(test_score, support_score)
        if max_score == 0:
            confidence = 0.0
        else:
            confidence = min(1.0, max_score / 5.0)  # Normalize to 0-1
        
        # Ensure mutual exclusivity: support files are not test files
        if is_support and is_test:
            # Prefer support classification if it's conftest or has fixtures only
            if signals.is_conftest or signals.has_fixtures_only:
                is_test = False
            else:
                # File could be both - lean towards test if it has actual tests
                is_support = False
        
        return Classification(
            is_test=is_test,
            is_support=is_support,
            confidence=confidence,
            signals=signals
        )
    
    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key based on file path and modification time."""
        try:
            mtime = file_path.stat().st_mtime
            return f"{file_path}:{mtime}"
        except Exception:
            return str(file_path)
    
    def clear_cache(self) -> None:
        """Clear the classification cache."""
        self._cache.clear()
        logger.debug("Test file classification cache cleared")
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cached_files": list(self._cache.keys()),
        }
