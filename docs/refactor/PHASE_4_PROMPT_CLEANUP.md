# Phase 4: Prompt Registry Cleanup

**Priority**: HIGH
**Impact**: BREAKING CHANGE - Legacy prompts removed

---

## Overview

This phase removes all legacy and LLM adapter prompts from the registry, keeping only orchestrator and evaluation prompts. This is a significant cleanup that removes ~330 lines of code.

---

## Task 4.1: Document Prompts Being Removed

### Objective

Create comprehensive documentation of all removed prompts for future reference and migration support.

### Execution

**File**: `docs/refactor/REMOVED_PROMPTS.md`

Already created in previous step. Ensure it includes:
- List of all removed prompts with line numbers
- Purpose of each prompt
- Replacement prompt(s)
- Migration examples
- Recovery instructions

---

## Task 4.2: Remove Legacy Prompts from Registry

### Step 1: Backup Current Registry

```bash
# Create backup before changes
cp testcraft/prompts/registry.py testcraft/prompts/registry.py.backup

# Commit backup to git
git add testcraft/prompts/registry.py.backup
git commit -m "chore: Backup prompt registry before cleanup"
```

### Step 2: Remove from Template Dictionaries

**File**: `testcraft/prompts/registry.py`
**Lines**: 103-145

#### System Templates (Lines 103-123)

**REMOVE these entries**:
```python
"test_generation": self._system_prompt_generation_v1(),
"refinement": self._system_prompt_refinement_v1(),
"llm_test_generation": self._system_prompt_llm_test_generation_v1(),
"llm_code_analysis": self._system_prompt_llm_code_analysis_v1(),
"llm_content_refinement": self._system_prompt_llm_content_refinement_v1(),
```

**KEEP these entries**:
```python
"orchestrator_plan": self._system_prompt_orchestrator_plan_v1(),
"orchestrator_generate": self._system_prompt_orchestrator_generate_v1(),
"orchestrator_refine": self._system_prompt_orchestrator_refine_v1(),
"orchestrator_manual_fix": self._system_prompt_orchestrator_manual_fix_v1(),
"llm_judge_v1": self._system_prompt_llm_judge_v1(),
"pairwise_comparison_v1": self._system_prompt_pairwise_comparison_v1(),
"rubric_evaluation_v1": self._system_prompt_rubric_evaluation_v1(),
"statistical_analysis_v1": self._system_prompt_statistical_analysis_v1(),
"bias_mitigation_v1": self._system_prompt_bias_mitigation_v1(),
```

#### User Templates (Lines 125-145)

**REMOVE these entries** (same as above):
```python
"test_generation": self._user_prompt_generation_v1(),
"refinement": self._user_prompt_refinement_v1(),
"llm_test_generation": self._user_prompt_llm_test_generation_v1(),
"llm_code_analysis": self._user_prompt_llm_code_analysis_v1(),
"llm_content_refinement": self._user_prompt_llm_content_refinement_v1(),
```

**KEEP** all orchestrator and evaluation prompts.

### Step 3: Delete Prompt Methods

**Lines to DELETE**: 271-601 (approximately)

Delete these entire methods:
1. `_system_prompt_generation_v1()` (lines ~271-312)
2. `_system_prompt_refinement_v1()` (lines ~314-353)
3. `_user_prompt_generation_v1()` (lines ~355-378)
4. `_user_prompt_refinement_v1()` (lines ~380-396)
5. `_system_prompt_llm_test_generation_v1()` (lines ~401-430)
6. `_system_prompt_llm_code_analysis_v1()` (lines ~432-457)
7. `_system_prompt_llm_content_refinement_v1()` (lines ~459-534)
8. `_user_prompt_llm_test_generation_v1()` (lines ~536-564)
9. `_user_prompt_llm_code_analysis_v1()` (lines ~566-576)
10. `_user_prompt_llm_content_refinement_v1()` (lines ~578-601)

**Deletion Command**:
```bash
# Use sed to delete lines 271-601 (adjust numbers as needed)
sed -i.bak '271,601d' testcraft/prompts/registry.py

# Or use your editor to delete these methods manually
```

### Step 4: Remove Unused Schemas

**File**: `testcraft/prompts/registry.py`
**Method**: `_schema_for()` (lines ~1177-1575)

**DELETE these schema cases**:
```python
# In the if-elif chain, remove:
elif schema_type == "generation_output":
    # DELETE entire block
elif schema_type == "generation_output_enhanced":
    # DELETE entire block
elif schema_type == "refinement_output":
    # DELETE entire block
elif schema_type == "refinement_output_enhanced":
    # DELETE entire block
elif schema_type == "llm_test_generation_output":
    # DELETE entire block
elif schema_type == "llm_code_analysis_output":
    # DELETE entire block
elif schema_type == "llm_content_refinement_output":
    # DELETE entire block
```

**KEEP** all orchestrator and evaluation schemas.

### Step 5: Update Module Docstring

**File**: `testcraft/prompts/registry.py`
**Lines**: 1-50 (approximately)

**Replace docstring** with:
```python
"""
Prompt templates and registry with versioning.

This module provides a versioned prompt registry for TestCraft's AI-powered
test generation and evaluation systems.

## Active Prompt Systems

### 1. Orchestrator Prompts (4-Stage Pipeline)

TestCraft uses a sophisticated 4-stage pipeline for test generation:

- **orchestrator_plan**: Analyze code and create comprehensive test plans
  with symbol resolution for missing imports/dependencies
- **orchestrator_generate**: Generate tests following the approved plan
  with all necessary context and resolved symbols
- **orchestrator_refine**: Repair failing tests with minimal, targeted changes
  using enhanced context and symbol resolution
- **orchestrator_manual_fix**: Create failing tests + bug reports when product
  bugs are detected (not test issues)

### 2. Evaluation Prompts (LLM-as-Judge)

For test quality evaluation and A/B testing:

- **llm_judge_v1**: Evaluate single test quality against criteria
- **pairwise_comparison_v1**: Compare two test implementations
- **rubric_evaluation_v1**: Score tests against defined rubric
- **statistical_analysis_v1**: Analyze test execution patterns
- **bias_mitigation_v1**: Detect and mitigate evaluation bias

## Orchestrator Features

- **ContextPack**: Structured context with Target, Focal, ImportMap,
  ResolvedDefs, PropertyContext, Conventions, and Budget
- **Symbol Resolution**: Iterative missing_symbols detection and resolution
- **GWT Patterns**: Given/When/Then snippets for property-based testing
- **Canonical Imports**: Strict import_map.target_import enforcement
- **4-Stage Quality Gates**: Plan → Generate → Refine → Manual Fix

## Removed Prompts (Oct 2025)

The following prompt systems were removed during orchestrator consolidation:
- test_generation/refinement (legacy simple prompts)
- llm_test_generation/llm_code_analysis/llm_content_refinement (LLM adapter prompts)

See docs/refactor/REMOVED_PROMPTS.md for migration guide.

## Design Principles

- Light anti-injection guidance (not heavy prompt engineering)
- Safe template rendering using {{}} for variables
- Framework-agnostic design (aligns with PromptPort protocol)
- Versioned prompts for reproducibility (currently v1)

## Usage

```python
from testcraft.prompts.registry import PromptRegistry

registry = PromptRegistry(version="v1")

# Get orchestrator prompts
plan_system = registry.get_system_prompt("orchestrator_plan")
plan_user = registry.get_user_prompt("orchestrator_plan", **context)

# Get evaluation prompts
judge_system = registry.get_system_prompt("llm_judge_v1")
judge_user = registry.get_user_prompt("llm_judge_v1", **test_data)
```
"""
```

---

## Task 4.3: Update Tests

### Update Prompt Registry Tests

**File**: `scripts/prompt_regression_test.py`

#### Update Expected Prompts

```python
EXPECTED_SYSTEM_PROMPTS = [
    # Orchestrator prompts (KEEP)
    "orchestrator_plan",
    "orchestrator_generate",
    "orchestrator_refine",
    "orchestrator_manual_fix",
    # Evaluation prompts (KEEP)
    "llm_judge_v1",
    "pairwise_comparison_v1",
    "rubric_evaluation_v1",
    "statistical_analysis_v1",
    "bias_mitigation_v1",
]

EXPECTED_USER_PROMPTS = EXPECTED_SYSTEM_PROMPTS  # Same set

REMOVED_PROMPTS = [
    # Legacy prompts (REMOVED)
    "test_generation",
    "refinement",
    # LLM adapter prompts (REMOVED)
    "llm_test_generation",
    "llm_code_analysis",
    "llm_content_refinement",
]
```

#### Add Test for Removed Prompts

```python
def test_removed_prompts_raise_key_error():
    """Verify that removed prompts raise KeyError with helpful message."""
    registry = PromptRegistry()

    for prompt_type in REMOVED_PROMPTS:
        # Should raise KeyError
        with pytest.raises(KeyError) as exc_info:
            registry.get_system_prompt(prompt_type)

        # Error message should be helpful
        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "removed" in error_msg or prompt_type in error_msg

        # Same for user prompts
        with pytest.raises(KeyError):
            registry.get_user_prompt(prompt_type)


def test_all_orchestrator_prompts_exist():
    """Verify all orchestrator prompts are present."""
    registry = PromptRegistry()

    orchestrator_prompts = [
        "orchestrator_plan",
        "orchestrator_generate",
        "orchestrator_refine",
        "orchestrator_manual_fix",
    ]

    for prompt_type in orchestrator_prompts:
        # Should not raise
        system = registry.get_system_prompt(prompt_type)
        user = registry.get_user_prompt(prompt_type)

        # Should return non-empty strings
        assert system and isinstance(system, str)
        assert user and isinstance(user, str)


def test_all_evaluation_prompts_exist():
    """Verify all evaluation prompts are present."""
    registry = PromptRegistry()

    eval_prompts = [
        "llm_judge_v1",
        "pairwise_comparison_v1",
        "rubric_evaluation_v1",
        "statistical_analysis_v1",
        "bias_mitigation_v1",
    ]

    for prompt_type in eval_prompts:
        system = registry.get_system_prompt(prompt_type)
        user = registry.get_user_prompt(prompt_type)

        assert system and isinstance(system, str)
        assert user and isinstance(user, str)
```

---

## Task 4.4: Validation

### Step 1: Verify Registry Still Works

```bash
# Test orchestrator prompts
python -c "
from testcraft.prompts.registry import PromptRegistry

registry = PromptRegistry()

# Verify orchestrator prompts exist
for prompt in ['orchestrator_plan', 'orchestrator_generate', 'orchestrator_refine']:
    system = registry.get_system_prompt(prompt)
    user = registry.get_user_prompt(prompt)
    print(f'✅ {prompt}: {len(system)} + {len(user)} chars')
"
```

**Expected**: All prompts load successfully

### Step 2: Verify Removed Prompts Error

```bash
# Test removed prompts raise errors
python -c "
from testcraft.prompts.registry import PromptRegistry

registry = PromptRegistry()

try:
    registry.get_system_prompt('test_generation')
    print('❌ Should have raised KeyError')
except KeyError as e:
    print(f'✅ Correctly raises KeyError: {e}')
"
```

**Expected**: KeyError with clear message

### Step 3: Run Prompt Regression Tests

```bash
python scripts/prompt_regression_test.py
```

**Expected**: All tests pass

### Step 4: Check Code Still Runs

```bash
# Verify test generation still works
testcraft generate testcraft/domain/models.py --dry-run
```

**Expected**: Works normally (uses orchestrator prompts)

---

## Task 4.5: Measure Impact

### Lines of Code Removed

```bash
# Count lines removed
git diff HEAD testcraft/prompts/registry.py | grep '^-' | wc -l
```

**Expected**: ~330 lines removed

### File Size Reduction

```bash
# Before cleanup
wc -l testcraft/prompts/registry.py.backup

# After cleanup
wc -l testcraft/prompts/registry.py

# Reduction
echo "Reduction: $(($(wc -l < registry.py.backup) - $(wc -l < registry.py))) lines"
```

**Expected**: Significant reduction

---

## Rollback

If something breaks:

```bash
# Restore backup
cp testcraft/prompts/registry.py.backup testcraft/prompts/registry.py

# Or use git
git checkout testcraft/prompts/registry.py

# Re-run tests
pytest tests/ -v
```

---

## Phase 4 Completion Checklist

- [ ] Backup created
- [ ] Legacy prompt entries removed from dictionaries
- [ ] Legacy prompt methods deleted (~330 lines)
- [ ] Unused schemas removed
- [ ] Module docstring updated
- [ ] Prompt regression tests updated
- [ ] All tests pass
- [ ] Manual validation complete
- [ ] Impact measured and documented
- [ ] REMOVED_PROMPTS.md complete

**Only proceed to Phase 5 when ALL items checked**

---

**Next**: [PHASE_5_TEST_UPDATES.md](./PHASE_5_TEST_UPDATES.md)
