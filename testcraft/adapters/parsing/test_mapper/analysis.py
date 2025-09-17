"""
Test file analysis dataclasses and AST walker implementation.

This module contains the core dataclasses and AST analysis logic for mapping
test files to source code. It focuses on AST-first analysis of imports, function
calls, attribute access, and mock patches.
"""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ImportInfo:
    """Information about an import statement in a test file."""
    
    module_path: str
    alias: Optional[str] = None
    imported_names: list[str] = field(default_factory=list)
    import_type: str = "module"  # "module", "from", "attribute"
    
    def matches_module_path(self, target_module_path: str) -> bool:
        """Check if this import matches a target module path."""
        return (
            self.module_path == target_module_path or
            self.module_path.startswith(target_module_path + ".") or
            target_module_path.startswith(self.module_path + ".")
        )


@dataclass 
class UsageInfo:
    """Information about how an imported module is used in a test file."""
    
    function_calls: list[str] = field(default_factory=list)
    attribute_accesses: list[str] = field(default_factory=list)
    mock_patches: list[str] = field(default_factory=list)
    
    def get_usage_score(self) -> int:
        """Calculate usage score based on detected usage patterns."""
        score = 0
        score += len(self.function_calls)
        score += len(self.attribute_accesses)
        score += len(self.mock_patches) * 2  # Mock patches are strong indicators
        return score


@dataclass
class TestFileAnalysis:
    """Complete analysis of a test file's imports and usage."""
    
    file_path: Path
    imports: list[ImportInfo] = field(default_factory=list)
    usage: UsageInfo = field(default_factory=UsageInfo)
    module_paths_referenced: set[str] = field(default_factory=set)
    
    def get_mapping_score_for_module(self, module_path: str) -> int:
        """Get mapping score for a specific module path."""
        score = 0
        
        # Check for direct import matches
        for import_info in self.imports:
            if import_info.matches_module_path(module_path):
                score += 1
                
        # Check for usage patterns
        if module_path in self.module_paths_referenced:
            score += self.usage.get_usage_score()
            
        return score


class TestFileAnalyzer:
    """AST-based analyzer for test files."""
    
    def __init__(self):
        """Initialize the analyzer with AST analysis cache."""
        # Cache for AST analyses with modification time for invalidation
        self._ast_analysis_cache: dict[Path, tuple[TestFileAnalysis, float]] = {}
    
    def analyze_test_file(self, test_file: Path) -> Optional[TestFileAnalysis]:
        """Analyze a test file for imports and usage patterns using AST."""
        # Check cache first
        if test_file in self._ast_analysis_cache:
            cached_analysis, mtime = self._ast_analysis_cache[test_file]
            try:
                current_mtime = test_file.stat().st_mtime
                if current_mtime <= mtime:
                    return cached_analysis
            except Exception:
                pass
        
        try:
            content = test_file.read_text(encoding='utf-8')
            tree = ast.parse(content, filename=str(test_file))
            
            analysis = TestFileAnalysis(file_path=test_file)
            
            # Walk the AST to collect imports and usage
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self._process_import(node, analysis)
                elif isinstance(node, ast.ImportFrom):
                    self._process_import_from(node, analysis)
                elif isinstance(node, ast.Call):
                    self._process_call(node, analysis)
                elif isinstance(node, ast.Attribute):
                    self._process_attribute(node, analysis)
            
            # Cache the analysis
            try:
                mtime = test_file.stat().st_mtime
                self._ast_analysis_cache[test_file] = (analysis, mtime)
            except Exception:
                pass
            
            return analysis
            
        except Exception as e:
            # Return None if analysis fails - this is not critical
            return None

    def _process_import(self, node: ast.Import, analysis: TestFileAnalysis) -> None:
        """Process an import statement."""
        for alias in node.names:
            import_info = ImportInfo(
                module_path=alias.name,
                alias=alias.asname,
                import_type="module"
            )
            analysis.imports.append(import_info)
            analysis.module_paths_referenced.add(alias.name)

    def _process_import_from(self, node: ast.ImportFrom, analysis: TestFileAnalysis) -> None:
        """Process a from-import statement."""
        if not node.module:
            return
            
        imported_names = [alias.name for alias in node.names]
        import_info = ImportInfo(
            module_path=node.module,
            imported_names=imported_names,
            import_type="from"
        )
        analysis.imports.append(import_info)
        analysis.module_paths_referenced.add(node.module)

    def _process_call(self, node: ast.Call, analysis: TestFileAnalysis) -> None:
        """Process function calls to detect usage and mock patches."""
        # Get the full call name
        call_name = self._get_call_name(node)
        if call_name:
            analysis.usage.function_calls.append(call_name)
            
            # Check for mock patch patterns
            if "patch" in call_name.lower():
                # Look for string arguments that might be module paths
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if "." in arg.value:  # Looks like a module path
                            analysis.usage.mock_patches.append(arg.value)
                            analysis.module_paths_referenced.add(arg.value.split('.')[0])

    def _process_attribute(self, node: ast.Attribute, analysis: TestFileAnalysis) -> None:
        """Process attribute access."""
        attr_name = self._get_attribute_name(node)
        if attr_name and len(attr_name.split('.')) > 1:  # Multi-part attribute
            analysis.usage.attribute_accesses.append(attr_name)

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Get the full name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return self._get_attribute_name(node.func)
        return None

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
