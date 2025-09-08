#!/usr/bin/env python3
"""
Documentation verification script for testcraft CI pipeline.

This script verifies:
1. Markdown syntax validity
2. Internal link integrity  
3. Code example syntax
4. Project structure consistency
5. API documentation coverage

Run from project root: python scripts/doc_check.py
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("⚠️  Warning: markdown package not available, skipping advanced syntax checks")


class DocumentationChecker:
    """Comprehensive documentation verification for CI/CD."""
    
    def __init__(self, project_root: Path = None):
        """Initialize the documentation checker."""
        self.project_root = project_root or Path.cwd()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def run_all_checks(self) -> bool:
        """
        Run all documentation checks.
        
        Returns:
            True if all checks pass, False otherwise
        """
        print("📚 Starting documentation verification...")
        
        checks = [
            ("README Syntax Check", self._check_readme_syntax),
            ("Internal Links Check", self._check_internal_links),
            ("Code Examples Check", self._check_code_examples),
            ("Project Structure Check", self._check_project_structure),
            ("CLI Documentation Check", self._check_cli_documentation),
            ("API Coverage Check", self._check_api_coverage),
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"\n🔍 Running: {check_name}")
            try:
                if check_func():
                    print(f"✅ {check_name}: PASSED")
                else:
                    print(f"❌ {check_name}: FAILED")
                    all_passed = False
            except Exception as e:
                print(f"💥 {check_name}: ERROR - {str(e)}")
                self.errors.append(f"{check_name}: {str(e)}")
                all_passed = False
        
        # Print summary
        print("\n" + "="*50)
        if self.errors:
            print("❌ ERRORS FOUND:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if all_passed and not self.errors:
            print("🎉 All documentation checks PASSED!")
        else:
            print("❌ Documentation checks FAILED!")
            
        return all_passed and not self.errors
    
    def _check_readme_syntax(self) -> bool:
        """Check README.md for syntax errors and basic structure."""
        readme_path = self.project_root / "README.md"
        
        if not readme_path.exists():
            self.errors.append("README.md not found")
            return False
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Cannot read README.md: {e}")
            return False
        
        # Check for empty README
        if len(content.strip()) < 100:
            self.errors.append("README.md is too short (less than 100 characters)")
            return False
        
        # Check markdown syntax if available
        if MARKDOWN_AVAILABLE:
            try:
                markdown.markdown(content)
                print("  ✓ Markdown syntax is valid")
            except Exception as e:
                self.errors.append(f"README.md has invalid markdown syntax: {e}")
                return False
        
        # Check for required sections
        required_sections = [
            "# TestCraft", "## Features", "## Installation", 
            "## Usage", "## Development"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            self.warnings.append(f"README.md missing recommended sections: {missing_sections}")
        
        # Check for code blocks
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL)
        if not code_blocks:
            self.warnings.append("README.md has no code examples")
        else:
            print(f"  ✓ Found {len(code_blocks)} code examples")
        
        return True
    
    def _check_internal_links(self) -> bool:
        """Check internal links in markdown files."""
        markdown_files = list(self.project_root.rglob("*.md"))
        broken_links = []
        
        for md_file in markdown_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception:
                continue
            
            # Find markdown links
            links = re.findall(r'\[.*?\]\(([^)]+)\)', content)
            
            for link in links:
                # Skip external links
                if link.startswith(('http://', 'https://', 'mailto:', '#')):
                    continue
                
                # Handle relative paths
                if link.startswith('./'):
                    link = link[2:]
                
                link_path = md_file.parent / link if not link.startswith('/') else self.project_root / link[1:]
                
                if not link_path.exists():
                    broken_links.append(f"{md_file}: {link}")
        
        if broken_links:
            self.errors.append(f"Broken internal links found: {broken_links}")
            return False
        
        print(f"  ✓ Checked {len(markdown_files)} markdown files for broken links")
        return True
    
    def _check_code_examples(self) -> bool:
        """Validate Python code examples in documentation."""
        readme_path = self.project_root / "README.md"
        
        if not readme_path.exists():
            return True  # Already handled in syntax check
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return True  # Already handled in syntax check
        
        # Extract Python code blocks
        python_blocks = re.findall(r'```(?:python|bash)\n(.*?)\n```', content, re.DOTALL)
        
        syntax_errors = []
        for i, code_block in enumerate(python_blocks):
            # Skip bash commands and simple CLI examples
            if any(cmd in code_block for cmd in ['git clone', 'cd ', 'uv ', 'testcraft ']):
                continue
            
            # Try to parse Python code
            try:
                if 'def ' in code_block or 'import ' in code_block or 'from ' in code_block:
                    ast.parse(code_block)
            except SyntaxError as e:
                syntax_errors.append(f"Code block {i+1}: {e}")
        
        if syntax_errors:
            self.warnings.append(f"Python syntax issues in README code examples: {syntax_errors}")
            # Note: This is a warning, not an error, since some examples might be pseudo-code
        
        print(f"  ✓ Validated {len(python_blocks)} code examples")
        return True
    
    def _check_project_structure(self) -> bool:
        """Verify documented project structure matches reality."""
        
        # Expected directories based on README
        expected_dirs = [
            "testcraft/domain",
            "testcraft/application", 
            "testcraft/adapters",
            "testcraft/ports",
            "testcraft/cli"
        ]
        
        missing_dirs = []
        for dir_path in expected_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists() or not full_path.is_dir():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            self.errors.append(f"Documented directories missing: {missing_dirs}")
            return False
        
        # Check for key files
        expected_files = [
            "pyproject.toml",
            "testcraft/__init__.py",
            "testcraft/cli/main.py"
        ]
        
        missing_files = []
        for file_path in expected_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.errors.append(f"Expected files missing: {missing_files}")
            return False
        
        print(f"  ✓ All documented directories and key files exist")
        return True
    
    def _check_cli_documentation(self) -> bool:
        """Check CLI documentation consistency."""
        readme_path = self.project_root / "README.md"
        
        if not readme_path.exists():
            return True  # Already handled
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
        except Exception:
            return True
        
        # Check for CLI examples in README
        cli_examples = re.findall(r'testcraft\s+(\w+)', readme_content)
        
        if not cli_examples:
            self.warnings.append("No CLI usage examples found in README")
            return True
        
        print(f"  ✓ Found {len(set(cli_examples))} CLI commands documented")
        
        # Check if main CLI file exists and has basic structure
        cli_main_path = self.project_root / "testcraft" / "cli" / "main.py"
        if cli_main_path.exists():
            try:
                with open(cli_main_path, 'r') as f:
                    cli_content = f.read()
                
                # Check for Click decorators (assuming Click-based CLI)
                if '@click.' not in cli_content and '@app.' not in cli_content:
                    self.warnings.append("CLI main.py doesn't appear to use Click framework")
                
            except Exception:
                pass
        
        return True
    
    def _check_api_coverage(self) -> bool:
        """Check API documentation coverage for public interfaces."""
        
        # Find all Python files in main package
        testcraft_path = self.project_root / "testcraft"
        if not testcraft_path.exists():
            self.warnings.append("testcraft package directory not found")
            return True
        
        python_files = list(testcraft_path.rglob("*.py"))
        undocumented_modules = []
        
        for py_file in python_files:
            # Skip __pycache__ and test files
            if "__pycache__" in str(py_file) or py_file.name.startswith("test_"):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for module docstring
                if '"""' not in content and "'''" not in content:
                    # Skip __init__.py files as they might not need docstrings
                    if py_file.name != "__init__.py":
                        undocumented_modules.append(str(py_file.relative_to(self.project_root)))
                
            except Exception:
                continue
        
        if undocumented_modules:
            # This is a warning since not all modules need extensive documentation
            self.warnings.append(f"Modules without docstrings: {undocumented_modules[:5]}{'...' if len(undocumented_modules) > 5 else ''}")
        
        print(f"  ✓ Checked {len(python_files)} Python files for documentation")
        return True


def main():
    """Main entry point for the documentation checker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check testcraft documentation")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    checker = DocumentationChecker(project_root=args.project_root)
    success = checker.run_all_checks()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
