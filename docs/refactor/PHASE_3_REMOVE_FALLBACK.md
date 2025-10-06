# Phase 3: Remove Fallback & Enforce Orchestrator

**Priority**: HIGH
**Impact**: BREAKING CHANGE - Fail-fast instead of fallback

---

## Overview

This phase removes the legacy fallback path in `GenerateUseCase` that silently degrades to simple test generation when ContextPack is unavailable. The new behavior will fail fast with a clear error message.

---

## Task 3.1: Remove Legacy Fallback in GenerateUseCase

### Current State Analysis

**File**: `testcraft/application/generate_usecase.py`
**Lines**: 529-540

**Current Code**:
```python
else:
    # Fall back to legacy LLM call when context is not available
    llm_result = await self._llm.generate_tests(
        code_content=code_content,
        context=enhanced_context,
        test_framework=self._config["test_framework"],
    )
    test_content = llm_result.get("tests", "")
```

**Problem**: Silently degrades quality without warning

---

### Implementation

#### Step 1: Replace Fallback with Fail-Fast

**Location**: `testcraft/application/generate_usecase.py:529-540`

**New Code**:
```python
else:
    # Context unavailable - FAIL FAST with clear error
    error_msg = (
        f"Cannot generate tests without ContextPack. "
        f"Context building failed for {source_path}. "
        f"This indicates a parsing or context assembly issue that must be resolved."
    )
    logger.error(error_msg)

    # Log telemetry if available
    if self._telemetry:
        self._telemetry.record_event(
            event_type="context_building_failed",
            metadata={
                "source_path": str(source_path) if source_path else None,
                "error_type": "context_unavailable",
                "file_exists": source_path.exists() if source_path else False,
                "is_python": str(source_path).endswith(".py") if source_path else False,
            },
        )

    # Return detailed error result
    return GenerationResult(
        file_path=str(source_path) if source_path else "unknown",
        content=None,
        success=False,
        error_message=error_msg,
        metadata={
            "error_type": "context_unavailable",
            "source_path": str(source_path) if source_path else None,
            "remediation": [
                "Check that source file exists and is readable",
                "Ensure file contains valid Python code",
                "Check parsing logs for syntax errors",
                "Verify file is not empty",
                "Try parsing file manually with ast.parse()",
            ],
        },
    )
```

#### Step 2: Add Helper Function for Diagnostics

**Location**: Add as private method in `GenerateUseCase`

```python
def _diagnose_context_failure(self, source_path: Path | None) -> dict[str, Any]:
    """
    Diagnose why context building failed.

    Args:
        source_path: Path to source file (or None)

    Returns:
        Dictionary with diagnostic information
    """
    if not source_path:
        return {
            "reason": "No source path provided",
            "fix": "Ensure target file is specified",
        }

    if not source_path.exists():
        return {
            "reason": f"File does not exist: {source_path}",
            "fix": "Check file path is correct",
        }

    if not source_path.is_file():
        return {
            "reason": f"Path is not a file: {source_path}",
            "fix": "Provide path to a Python file, not directory",
        }

    if source_path.stat().st_size == 0:
        return {
            "reason": "File is empty",
            "fix": "Add content to file before generating tests",
        }

    # Try parsing
    try:
        import ast
        content = source_path.read_text()
        ast.parse(content)
        return {
            "reason": "File parses correctly but context building failed",
            "fix": "Check logs for context assembler errors",
        }
    except SyntaxError as e:
        return {
            "reason": f"Syntax error in file: {e}",
            "fix": "Fix Python syntax errors in source file",
        }
    except Exception as e:
        return {
            "reason": f"Error reading/parsing file: {e}",
            "fix": "Check file encoding and permissions",
        }
```

#### Step 3: Enhanced Error Message with Diagnostics

Update the error message to include diagnostics:

```python
else:
    # Diagnose the failure
    diagnostics = self._diagnose_context_failure(source_path)

    error_msg = (
        f"Cannot generate tests without ContextPack.\n"
        f"Context building failed for: {source_path}\n"
        f"\n"
        f"Diagnosis: {diagnostics['reason']}\n"
        f"Solution: {diagnostics['fix']}\n"
        f"\n"
        f"This indicates a parsing or context assembly issue that must be resolved."
    )
    logger.error(error_msg)

    # ... rest of telemetry and return ...
```

---

### Testing

#### Test 1: Valid File (Should Use Orchestrator)

```bash
# Should succeed with orchestrator
testcraft generate testcraft/domain/models.py --verbose

# Check logs for "Using orchestrator" message
grep -i "orchestrator" .testcraft/logs/*.log
```

**Expected**: Test generated successfully using orchestrator

#### Test 2: Invalid File (Should Fail Fast)

```bash
# Create invalid Python file
echo "def incomplete(" > /tmp/test_invalid.py

# Try to generate (should fail)
testcraft generate /tmp/test_invalid.py

# Check error message
# Expected: Clear error about syntax error
```

**Expected Output**:
```
❌ Error: Cannot generate tests without ContextPack.
Context building failed for: /tmp/test_invalid.py

Diagnosis: Syntax error in file: unexpected EOF while parsing
Solution: Fix Python syntax errors in source file

This indicates a parsing or context assembly issue that must be resolved.
```

#### Test 3: Nonexistent File (Should Fail Fast)

```bash
testcraft generate /tmp/nonexistent.py
```

**Expected Output**:
```
❌ Error: Cannot generate tests without ContextPack.
Context building failed for: /tmp/nonexistent.py

Diagnosis: File does not exist: /tmp/nonexistent.py
Solution: Check file path is correct
```

#### Test 4: Empty File (Should Fail Fast)

```bash
touch /tmp/empty.py
testcraft generate /tmp/empty.py
```

**Expected Output**:
```
❌ Error: Cannot generate tests without ContextPack.
Context building failed for: /tmp/empty.py

Diagnosis: File is empty
Solution: Add content to file before generating tests
```

---

### Telemetry Integration

#### Events to Track

```python
# When context building fails
self._telemetry.record_event(
    event_type="context_building_failed",
    metadata={
        "source_path": str(source_path),
        "reason": diagnostics["reason"],
        "file_exists": source_path.exists(),
        "file_size": source_path.stat().st_size if source_path.exists() else 0,
        "is_python": str(source_path).endswith(".py"),
    },
)

# When fail-fast is triggered
self._telemetry.record_event(
    event_type="generation_failed_fast",
    metadata={
        "reason": "context_unavailable",
        "diagnostics": diagnostics,
    },
)
```

#### Metrics to Monitor

After deployment:
- **context_building_failed** event count
- **generation_failed_fast** event count
- Common failure reasons (from diagnostics)
- User response to error messages

---

## Task 3.2: Update Error Handling in CLI

### Objective

Ensure CLI properly displays fail-fast errors to users.

### Implementation

**File**: `testcraft/cli/main.py`

**Current** (around line 350):
```python
# CLI may swallow detailed errors
result = use_case.execute(...)
```

**Enhanced**:
```python
try:
    result = use_case.execute(...)

    # Check for context failures
    if not result["success"]:
        for file_result in result.get("files", []):
            if file_result.get("error_type") == "context_unavailable":
                # Display detailed error with remediation
                ctx.obj.ui.display_error_with_suggestions(
                    file_result["error_message"],
                    file_result["metadata"].get("remediation", []),
                    "Context Building Failed"
                )
except GenerateUseCaseError as e:
    ctx.obj.ui.display_error(str(e), "Generation Failed")
    sys.exit(1)
```

---

## Task 3.3: Add Integration Tests

### Test: Fail-Fast Behavior

**File**: `tests/test_generate_usecase_fail_fast.py`

```python
import pytest
from pathlib import Path
from testcraft.application.generate_usecase import GenerateUseCase


def test_fail_fast_on_invalid_python(tmp_path, generate_use_case):
    """Test that invalid Python triggers fail-fast."""
    # Create invalid Python file
    invalid_file = tmp_path / "invalid.py"
    invalid_file.write_text("def incomplete(")

    # Attempt generation
    result = generate_use_case.execute(
        target_files=[str(invalid_file)],
        project_path=str(tmp_path),
    )

    # Should fail fast
    assert not result["success"]
    assert "context_unavailable" in str(result)
    assert "syntax error" in result["error_message"].lower()


def test_fail_fast_on_nonexistent_file(tmp_path, generate_use_case):
    """Test that nonexistent file triggers fail-fast."""
    nonexistent = tmp_path / "nonexistent.py"

    result = generate_use_case.execute(
        target_files=[str(nonexistent)],
        project_path=str(tmp_path),
    )

    assert not result["success"]
    assert "does not exist" in result["error_message"].lower()


def test_fail_fast_on_empty_file(tmp_path, generate_use_case):
    """Test that empty file triggers fail-fast."""
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")

    result = generate_use_case.execute(
        target_files=[str(empty_file)],
        project_path=str(tmp_path),
    )

    assert not result["success"]
    assert "empty" in result["error_message"].lower()


def test_success_with_valid_file(tmp_path, generate_use_case):
    """Test that valid file succeeds with orchestrator."""
    valid_file = tmp_path / "valid.py"
    valid_file.write_text("def example(): pass")

    result = generate_use_case.execute(
        target_files=[str(valid_file)],
        project_path=str(tmp_path),
    )

    # Should succeed (or fail for different reason, not context)
    if not result["success"]:
        assert "context_unavailable" not in str(result)
```

---

## Task 3.4: Update Documentation

### User-Facing Docs

**File**: `docs/troubleshooting.md` (create if doesn't exist)

```markdown
# Troubleshooting Test Generation

## Error: "Cannot generate tests without ContextPack"

### Cause
TestCraft's context builder failed to analyze your source file.

### Common Reasons

1. **Syntax Errors**
   - **Symptom**: "Syntax error in file: ..."
   - **Fix**: Run `python -m py_compile your_file.py` to find syntax errors

2. **File Not Found**
   - **Symptom**: "File does not exist: ..."
   - **Fix**: Check file path is correct

3. **Empty File**
   - **Symptom**: "File is empty"
   - **Fix**: Add code to file before generating tests

4. **Encoding Issues**
   - **Symptom**: "Error reading/parsing file: ..."
   - **Fix**: Ensure file is UTF-8 encoded

### Advanced Debugging

```bash
# Check if file parses
python -c "import ast; ast.parse(open('your_file.py').read())"

# Enable verbose logging
testcraft generate your_file.py --verbose

# Check logs
cat .testcraft/logs/testcraft.log
```

### Still Stuck?

If you've verified the file is valid Python and the error persists:
1. Check `.testcraft/logs/testcraft.log` for detailed errors
2. Try generating tests for a simpler file first
3. Report issue with logs if problem persists
```

---

## Validation

### Checklist

- [ ] Fallback code removed (lines 529-540)
- [ ] Fail-fast implementation added
- [ ] Diagnostics helper implemented
- [ ] Enhanced error messages with remediation
- [ ] Telemetry events added
- [ ] CLI error handling updated
- [ ] Integration tests added
- [ ] Documentation updated
- [ ] Manual testing complete

### Manual Test Script

```bash
#!/bin/bash
echo "Testing fail-fast behavior..."

# Test 1: Valid file
echo "def example(): pass" > /tmp/test_valid.py
testcraft generate /tmp/test_valid.py
echo "✅ Valid file test complete"

# Test 2: Invalid syntax
echo "def incomplete(" > /tmp/test_invalid.py
testcraft generate /tmp/test_invalid.py 2>&1 | grep -i "syntax"
echo "✅ Invalid syntax test complete"

# Test 3: Nonexistent file
testcraft generate /tmp/nonexistent.py 2>&1 | grep -i "not exist"
echo "✅ Nonexistent file test complete"

# Test 4: Empty file
touch /tmp/test_empty.py
testcraft generate /tmp/test_empty.py 2>&1 | grep -i "empty"
echo "✅ Empty file test complete"

echo "All fail-fast tests complete"
```

---

## Rollback

If this phase breaks something:

```bash
# Revert GenerateUseCase changes
git checkout testcraft/application/generate_usecase.py

# Re-run tests
pytest tests/ -v
```

---

## Phase 3 Completion

- [ ] Fallback removed and fail-fast implemented
- [ ] All tests pass
- [ ] Manual validation complete
- [ ] Error messages clear and actionable
- [ ] Telemetry in place
- [ ] Documentation updated

**Only proceed to Phase 4 when ALL items checked**

---

**Next**: [PHASE_4_PROMPT_CLEANUP.md](./PHASE_4_PROMPT_CLEANUP.md)
