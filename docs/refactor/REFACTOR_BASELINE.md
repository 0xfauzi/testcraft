# Refactoring Baseline Metrics

**Purpose**: Document current state before orchestrator consolidation refactor
**Date**: 2025-10-06
**Branch**: `refactor/orchestrator-consolidation` (working branch)
**Backup Branch**: `refactor/pre-orchestrator-consolidation-backup`
**Commit**: e36e22f76eeeeb82e3752208341abc080dca841c

---

## Test Metrics

### Test Suite Status
- **Total tests**: 0 (tests were deleted in commit e36e22f)
- **Passing**: 0
- **Failing**: 0
- **Skipped**: 0
- **Test pass rate**: N/A
- **Note**: This refactoring starts with no tests; tests will be recreated post-refactor

### Coverage
- **Overall line coverage**: N/A (no tests)
- **Branch coverage**: N/A
- **testcraft/ coverage**: N/A
- **Critical modules coverage**: N/A
- **Post-refactor target**: ≥80% coverage

---

## Code Metrics

### Lines of Code
- **Total project LOC**: 196,414
- **testcraft/ LOC**: 62,438
- **tests/ LOC**: 0
- **Key files**:
  - `generate_usecase.py`: 1,736 lines (with dry-run changes)
  - `prompts/registry.py`: ~1,200 lines (includes ~330 lines to remove)
  - `cli/main.py`: 1,000 lines (with dry-run changes)

### Files with Legacy Code
- **Legacy prompt references**: 459 grep matches
- **`generate_tests` method calls**: 9 callsites
- **`refine_content` method calls**: 2 callsites
- **Files to modify**: 14 (per ENHANCED_REFACTOR_PLAN.md)

---

## Callsites to Remove/Modify

### High-Level Method Calls
- **`generate_tests()` high-level calls**: 9 occurrences
- **`refine_content()` high-level calls**: 2 occurrences
- **Legacy prompt references**: 459 total references

### Prompt Registry Usage
- **Prompts to remove**: 10 prompt methods (~330 lines)
  - `test_generation`
  - `refinement`
  - `llm_test_generation`
  - `llm_code_analysis`
  - `llm_content_refinement`
  - Plus their system/user prompt variants

### Expected Removals
- **Lines to delete from prompts**: ~330
- **Lines to modify in use cases**: ~50
- **Lines to add (refactoring)**: ~200
- **Net LOC reduction**: ~180 lines (plus ~330 from prompts = **~510 total**)

---

## Performance Baseline

### Test Generation Time (Dry-Run)
- **Dry-run implementation**: ✅ Complete at use case layer
- **CLI dry-run**: ✅ Complete with proper messaging
- **Actual testing**: ⚠️ BLOCKED by LLMRouter being commented out (pre-existing issue)

### Memory Usage
- **Baseline not captured**: Application cannot run due to LLMRouter issue
- **Will measure post-fix**: After LLMRouter is uncommented/fixed

### Token Usage
- **Baseline not available**: Requires working application
- **Will measure post-refactor**: Compare orchestrator vs legacy

---

## Dependency Analysis

### Critical Dependencies
- **LLMOrchestrator** depends on:
  - LLMPort
  - ParserPort
  - ContextAssembler
  - ContextPackBuilder
  - SymbolResolver

- **RefineAdapter** currently depends on:
  - LLMPort
  - (Missing: ParserPort - will be added)

- **GenerateUseCase** uses:
  - LLMOrchestrator (primary path)
  - LLM adapter fallback (to be removed)

---

## Known Issues (Pre-Refactor)

### Issue 1: --dry-run Mode
- **Status**: ✅ FIXED in Phase 0
- **Fixed in**: `generate_usecase.py`, `cli/main.py`, `dependency_injection.py`
- **Commit**: Pending (will be committed with Phase 0)

### Issue 2: LLMRouter Completely Commented Out
- **Status**: ✅ FIXED
- **Fixed in**: `testcraft/adapters/llm/router.py` - Complete reimplementation
- **Commit**: b4d3e91
- **Solution**: Implemented complete LLMRouter with provider factory pattern
- **Testing**: ✅ Application fully functional, all commands work

### Issue 3: Silent Quality Degradation
- **Status**: Present (fallback in generate_usecase.py)
- **Will be fixed in**: Phase 3

### Issue 4: RefineAdapter Missing ParserPort
- **Status**: Present (cannot use orchestrator)
- **Will be fixed in**: Phase 2

---

## Validation Criteria

After refactoring, we will validate:

- [ ] Test count increases from 0
- [ ] Test pass rate approaches 100%
- [ ] Coverage ≥80%
- [ ] Performance within ±20%
- [ ] Memory usage within ±20%
- [ ] No new linter errors
- [ ] No new type checking errors

---

## Collection Commands

Run these commands to refine the baseline later:

```bash
# Lines of code
find . -name "*.py" | xargs wc -l | tail -1  # Total
find testcraft -name "*.py" | xargs wc -l | tail -1  # testcraft/
find tests -name "*.py" | xargs wc -l | tail -1  # tests/

# Callsites
grep -r "test_generation\|refinement\|llm_test_generation" testcraft --include="*.py" | wc -l
```

---

## Sign-off

- **Prepared by**: AI Assistant
- **Reviewed by**: ___
- **Date captured**: 2025-10-06
- **Git commit**: e36e22f76eeeeb82e3752208341abc080dca841c

---

**Note**: This baseline enables us to validate that the refactoring:
1. Maintains all functionality
2. Preserves test coverage
3. Doesn't degrade performance
4. Achieves expected code reduction

Fill in remaining TBDs during later phases.
