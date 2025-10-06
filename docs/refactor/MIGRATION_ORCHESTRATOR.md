# Migration Guide: Legacy to Orchestrator-Only Architecture

**Effective Date**: October 2025
**Breaking Change**: YES
**Rollback Available**: 30 days

---

## Overview

As of October 2025, TestCraft uses **LLMOrchestrator exclusively** for all test generation and refinement. This guide helps you migrate code that used legacy prompt systems.

---

## What Changed

### Removed ❌

1. **Legacy Prompt Types**:
   - `test_generation`
   - `refinement`

2. **LLM Adapter Prompts**:
   - `llm_test_generation`
   - `llm_code_analysis`
   - `llm_content_refinement`

3. **High-Level Usage Patterns**:
   - Direct calls to `llm.generate_tests()` with simple prompts
   - Direct calls to `llm.refine_content()` with simple prompts
   - Fallback path in `GenerateUseCase`

### Kept ✅

1. **LLMPort Protocol Methods** (used internally by orchestrator):
   ```python
   def generate_tests(self, code_content: str, **kwargs) -> dict[str, Any]:
       """LOW-LEVEL: Orchestrator calls this"""

   def refine_content(self, original: str, instructions: str, **kwargs) -> dict[str, Any]:
       """LOW-LEVEL: Orchestrator calls this"""
   ```

2. **Orchestrator Prompts**:
   - `orchestrator_plan`
   - `orchestrator_generate`
   - `orchestrator_refine`
   - `orchestrator_manual_fix`

3. **Evaluation Prompts**:
   - All LLM-as-judge prompts

---

## Migration Examples

### Example 1: Basic Test Generation

#### ❌ OLD CODE (No Longer Works)

```python
from testcraft.adapters.llm.claude import ClaudeAdapter

llm = ClaudeAdapter(api_key="...")
result = llm.generate_tests(
    code_content=source_code,
    context="simple context string",
)
tests = result["tests"]
```

#### ✅ NEW CODE (Option A - Use GenerateUseCase)

```python
from testcraft.cli.dependency_injection import create_generate_usecase

use_case = create_generate_usecase(config={
    "llm": {"provider": "anthropic", "model": "claude-sonnet-4"},
    "test_framework": "pytest",
})

result = await use_case.generate_tests(
    target_files=["path/to/source.py"],
    project_path=".",
)

# Access generated tests
for file_result in result["files"]:
    if file_result["success"]:
        tests = file_result["content"]
```

#### ✅ NEW CODE (Option B - Use Orchestrator Directly)

```python
from testcraft.application.generation.services.llm_orchestrator import LLMOrchestrator
from testcraft.application.generation.services.context_pack import ContextPackBuilder

# Build dependencies
llm_port = ClaudeAdapter(api_key="...")
parser_port = CodebaseParser()
# ... other dependencies ...

# Create orchestrator
orchestrator = LLMOrchestrator(
    llm_port=llm_port,
    parser_port=parser_port,
    context_assembler=context_assembler,
    context_pack_builder=ContextPackBuilder(),
    symbol_resolver=symbol_resolver,
)

# Build ContextPack
context_pack = ContextPackBuilder().build_context_pack(
    target_file=Path("src/module.py"),
    target_object="MyClass",
    project_root=Path("."),
)

# Generate tests
result = orchestrator.plan_and_generate(
    context_pack=context_pack,
    project_root=Path("."),
)

tests = result["generated_code"]
```

---

### Example 2: Test Refinement

#### ❌ OLD CODE

```python
from testcraft.adapters.refine.main_adapter import RefineAdapter

adapter = RefineAdapter(llm=llm_port)
result = adapter.refine_from_failures(
    test_file="tests/test_module.py",
    failure_output=pytest_output,
)
```

#### ✅ NEW CODE

```python
from testcraft.adapters.refine.main_adapter import RefineAdapter

# ⚠️ BREAKING CHANGE: Now requires ParserPort
adapter = RefineAdapter(
    llm=llm_port,
    parser_port=parser_port,  # NEW - REQUIRED
)

result = adapter.refine_from_failures(
    test_file="tests/test_module.py",
    failure_output=pytest_output,
    source_context={
        "source_file": "src/module.py",  # Helps build better context
        "target_object": "MyClass",
    },
)

# Result structure unchanged
if result["success"]:
    refined_tests = result["refined_content"]
```

---

### Example 3: Custom Prompt Usage

#### ❌ OLD CODE

```python
from testcraft.prompts.registry import PromptRegistry

registry = PromptRegistry()
system_prompt = registry.get_system_prompt("test_generation")
user_prompt = registry.get_user_prompt("test_generation", code_content=code)

result = llm.generate_tests(
    code_content=user_prompt,
    system_prompt=system_prompt,
)
```

#### ✅ NEW CODE

```python
from testcraft.prompts.registry import PromptRegistry
from testcraft.application.generation.services.llm_orchestrator import LLMOrchestrator

registry = PromptRegistry()

# Use orchestrator prompts
plan_system = registry.get_system_prompt("orchestrator_plan")
plan_user = registry.get_user_prompt("orchestrator_plan", context_pack=context_pack)

# Let orchestrator handle the full pipeline
orchestrator = LLMOrchestrator(...)
result = orchestrator.plan_and_generate(context_pack=context_pack)
```

---

### Example 4: CLI Usage (Unchanged)

#### ✅ CLI Usage Still Works

```bash
# No changes needed to CLI usage
testcraft generate src/mymodule.py

# All CLI commands work the same
testcraft generate src/ --recursive
testcraft generate src/mymodule.py --dry-run
```

**Note**: CLI internally uses orchestrator now, but interface unchanged.

---

## Breaking Changes

### 1. RefineAdapter Constructor

**Before**:
```python
adapter = RefineAdapter(llm=llm_port)
```

**After**:
```python
adapter = RefineAdapter(
    llm=llm_port,
    parser_port=parser_port,  # NEW - REQUIRED
)
```

**Why**: RefineAdapter now uses orchestrator internally, which needs ParserPort for context building.

**Fix**: Add ParserPort dependency:
```python
from testcraft.adapters.parsing.codebase_parser import CodebaseParser

parser_port = CodebaseParser()
adapter = RefineAdapter(llm=llm_port, parser_port=parser_port)
```

---

### 2. Removed Prompt Types

**Before**:
```python
prompt = registry.get_system_prompt("test_generation")
```

**After**:
```python
# Raises KeyError: 'test_generation'
```

**Fix**: Use orchestrator prompts:
```python
prompt = registry.get_system_prompt("orchestrator_generate")
```

**Available Prompts**:
- `orchestrator_plan`
- `orchestrator_generate`
- `orchestrator_refine`
- `orchestrator_manual_fix`

---

### 3. GenerateUseCase Fails Fast

**Before**:
```python
# Silently fell back to simple generation if context unavailable
result = use_case.execute(...)
# Result might be low-quality but "successful"
```

**After**:
```python
# Fails fast with clear error message
result = use_case.execute(...)
# Returns error if context unavailable
```

**Why**: Forces fixing root cause instead of masking with poor fallback.

**Fix**: Ensure source files are valid Python:
```bash
# Validate file before generating
python -m py_compile src/mymodule.py

# Check parsing
python -c "import ast; ast.parse(open('src/mymodule.py').read())"
```

---

## Benefits of Migration

### Better Test Quality

**Before** (simple prompts):
```python
# Generated test might have:
- Undefined imports
- Missing symbols
- Incorrect API usage
```

**After** (orchestrator):
```python
# Generated test has:
✅ All imports resolved
✅ All symbols defined
✅ Correct API usage (from context)
✅ Better coverage
```

### Better Error Messages

**Before**:
```
Error: Test generation failed
```

**After**:
```
❌ Error: Cannot generate tests without ContextPack.
Context building failed for: /path/to/file.py

Diagnosis: Syntax error in file: unexpected EOF while parsing
Solution: Fix Python syntax errors in source file
```

### Single Source of Truth

**Before**: Three parallel codepaths
**After**: One orchestrator pipeline

**Benefits**:
- Easier maintenance
- Consistent behavior
- Predictable quality

---

## Rollback Instructions

If you need to temporarily rollback:

```bash
# Option 1: Use backup branch
git checkout refactor/pre-orchestrator-consolidation-backup

# Option 2: Revert specific commits
git revert <merge-commit-sha>

# Option 3: Use older version
pip install testcraft==<previous-version>
```

**Note**: Backup branch maintained for 30 days after merge.

---

## FAQ

### Q: Why remove the fallback?

**A**: The fallback silently produced low-quality tests without warning users. Fail-fast forces fixing the root cause (parsing issues) instead of masking them.

### Q: What if orchestrator is slower?

**A**: Initial tests show similar performance. The quality improvement justifies modest performance cost. If performance is critical, report an issue and we'll optimize.

### Q: Can I still use simple prompts?

**A**: No. Orchestrator-only architecture provides better quality and maintainability. If you have specific needs, open an issue to discuss.

### Q: How do I debug context building failures?

**A**: Enable verbose logging:
```bash
testcraft generate myfile.py --verbose
cat .testcraft/logs/testcraft.log
```

### Q: Will this break my CI/CD?

**A**: CLI usage unchanged. If you use TestCraft via CLI, no changes needed. If you use it programmatically, follow migration examples above.

---

## Support

**Questions?** Open an issue on GitHub
**Problems?** Check troubleshooting guide in `docs/troubleshooting.md`
**Bugs?** Report with logs from `.testcraft/logs/`

---

**Last Updated**: 2025-10-05
**Status**: Active
**Rollback Available Until**: 2025-11-05 (30 days)
