# 🔍 ENHANCED BULLETPROOF REFACTORING PLAN
## Orchestrator-Only Architecture Consolidation

> **Version**: 2.0 Enhanced
> **Date**: October 5, 2025
> **Status**: VALIDATED & ENHANCED

---

## 🎯 EXECUTIVE SUMMARY

This plan consolidates all test generation to use **LLMOrchestrator exclusively**, removing legacy prompt paths while maintaining 100% functionality. The enhanced plan addresses critical gaps found in the original proposal.

### ⚠️ CRITICAL CORRECTIONS TO ORIGINAL PLAN

#### **Major Architectural Misunderstanding Identified**

❌ **WRONG ASSUMPTION**: "Remove `generate_tests()`, `refine_content()` from LLM adapters"

✅ **ACTUAL ARCHITECTURE**:
- `generate_tests()` and `refine_content()` are **LOW-LEVEL methods** in the LLMPort protocol
- The orchestrator **USES** these methods internally
- What we're removing is **HIGH-LEVEL USAGE** with legacy prompts, not the methods themselves

**Corrected Removal Strategy**:
```python
# ❌ DON'T remove this from adapters:
def generate_tests(self, code_content: str, **kwargs) -> dict[str, Any]:
    """Low-level LLM call used by orchestrator"""
    # This stays - orchestrator needs it

# ✅ DO remove this usage pattern:
llm_result = await self._llm.generate_tests(
    code_content=code_content,
    context=enhanced_context,  # Using legacy prompts
    test_framework=self._config["test_framework"],
)
```

---

## 📊 KEY FINDINGS FROM VALIDATION

### ✅ What's Already Working Well
1. **Orchestrator is already initialized** in `GenerateUseCase.__init__()` (lines 168-175)
2. **Orchestrator is already used** for main generation path (line 526)
3. **RefineAdapter exists** and is properly integrated
4. **No circular dependencies** - architecture is sound

### ⚠️ Critical Issues Found

#### **Issue 1: --dry-run Mode Doesn't Work**
**Location**: `testcraft/cli/main.py:303-309`
**Problem**: Only displays message, doesn't prevent execution
**Impact**: HIGH - Cannot safely test refactoring

#### **Issue 2: RefineAdapter Missing ParserPort**
**Location**: `testcraft/adapters/refine/main_adapter.py:34-40`
**Problem**: Cannot initialize orchestrator without ParserPort
**Impact**: HIGH - Breaks orchestrator integration

#### **Issue 3: Legacy Fallback Still Active**
**Location**: `testcraft/application/generate_usecase.py:534-540`
**Problem**: Falls back to simple generation instead of failing fast
**Impact**: MEDIUM - Silently degrades quality

---

## 📋 ENHANCED PHASE-BY-PHASE PLAN

### **PHASE 0: PRE-FLIGHT FIXES** ⚠️ (NEW)

**MANDATORY**: Must complete before any refactoring

#### Task 0.1: Fix --dry-run Mode
See: `docs/refactor/PHASE_0_PREFLIGHT.md`

#### Task 0.2: Create Rollback Branch
```bash
git checkout -b refactor/pre-orchestrator-consolidation-backup
git push origin refactor/pre-orchestrator-consolidation-backup
git checkout -b refactor/orchestrator-consolidation
```

#### Task 0.3: Baseline Metrics Collection
```bash
pytest tests/ -v --cov=testcraft --cov-report=json --junitxml=baseline_results.xml
find testcraft -name "*.py" | xargs wc -l > baseline_loc.txt
```

---

### **PHASE 1: COMPREHENSIVE IMPACT ANALYSIS** 📊

#### Task 1.1: Complete Callsite Audit
See: `docs/refactor/CALLSITE_AUDIT.md`

#### Task 1.2: Architecture Decision Document
See: `docs/refactor/ADR-001-orchestrator-consolidation.md`

---

### **PHASE 2: REFACTOR RefineAdapter** 🔧

**Key Change**: Add orchestrator integration with lazy initialization

#### Task 2.1: Add ParserPort Dependency
**File**: `testcraft/adapters/refine/main_adapter.py`

```python
def __init__(
    self,
    llm: LLMPort,
    parser_port: ParserPort,  # NEW - REQUIRED
    config: RefineConfig | None = None,
    writer_port: WriterPort | None = None,
    telemetry_port: TelemetryPort | None = None,
    llm_orchestrator: LLMOrchestrator | None = None,  # NEW - OPTIONAL
):
```

#### Task 2.2: Implement Orchestrator Integration
See: `docs/refactor/PHASE_2_REFINE_ADAPTER.md`

---

### **PHASE 3: REMOVE FALLBACK & ENFORCE ORCHESTRATOR** ⚔️

#### Task 3.1: Remove Legacy Fallback
**File**: `testcraft/application/generate_usecase.py:534-540`

**REPLACE**:
```python
else:
    # Fall back to legacy LLM call when context is not available
    llm_result = await self._llm.generate_tests(...)
```

**WITH**:
```python
else:
    # Context unavailable - FAIL FAST
    error_msg = (
        f"Cannot generate tests without ContextPack. "
        f"Context building failed for {source_path}."
    )
    return GenerationResult(success=False, error_message=error_msg)
```

---

### **PHASE 4: PROMPT REGISTRY CLEANUP** 🧹

#### Task 4.1: Document Removed Prompts
See: `docs/refactor/REMOVED_PROMPTS.md`

#### Task 4.2: Remove Legacy Prompts
**File**: `testcraft/prompts/registry.py`

**Remove from dictionaries** (lines 105-110, 127-132):
- `test_generation`
- `refinement`
- `llm_test_generation`
- `llm_code_analysis`
- `llm_content_refinement`

**Delete methods** (lines 271-601):
- `_system_prompt_generation_v1()`
- `_system_prompt_refinement_v1()`
- `_user_prompt_generation_v1()`
- `_user_prompt_refinement_v1()`
- `_system_prompt_llm_test_generation_v1()`
- `_system_prompt_llm_code_analysis_v1()`
- `_system_prompt_llm_content_refinement_v1()`
- `_user_prompt_llm_test_generation_v1()`
- `_user_prompt_llm_code_analysis_v1()`
- `_user_prompt_llm_content_refinement_v1()`

---

### **PHASE 5: TEST UPDATES** 🧪

#### Task 5.1: Update LLM Adapter Tests
**File**: `tests/test_llm_adapters.py`

**Strategy**:
- KEEP: Tests for core LLM methods (`_create_message`, streaming, token counting)
- UPDATE: Change `generate_tests()` tests to verify LOW-LEVEL functionality
- REMOVE: Tests for deprecated high-level usage patterns

#### Task 5.2: Update Integration Tests
**Files**:
- `tests/test_pytest_refiner_integration.py`
- `tests/test_immediate_refinement.py`
- `tests/test_strict_refinement.py`

**Strategy**: Mock orchestrator instead of llm methods

---

### **PHASE 6: DOCUMENTATION UPDATES** 📚

#### Task 6.1: Update Architecture Docs
**File**: `docs/architecture.md`

Add orchestrator-centric architecture diagram and explanation.

#### Task 6.2: Create Migration Guide
**File**: `docs/MIGRATION_ORCHESTRATOR.md`

Complete migration examples for all usage patterns.

---

### **PHASE 7: VALIDATION & ROLLOUT** ✅

#### Task 7.1: Comprehensive Test Suite
```bash
scripts/validate_refactor.sh  # See docs/refactor/VALIDATION_SCRIPTS.md
```

#### Task 7.2: Performance Benchmarking
```bash
python scripts/benchmark_refactor.py  # See docs/refactor/VALIDATION_SCRIPTS.md
```

---

### **PHASE 8: GIT WORKFLOW & DEPLOYMENT** 🚀

#### Task 8.1: Commit Strategy
7 atomic commits with clear breaking change markers.

#### Task 8.2: Create Pull Request
With comprehensive description and rollback plan.

---

### **PHASE 9: POST-MERGE VALIDATION** 🎯

#### Task 9.1: Smoke Tests
#### Task 9.2: Monitor for Issues (7 days)

---

## ✅ SUCCESS CRITERIA

Refactoring is **COMPLETE AND SUCCESSFUL** when:

1. ✅ All generation uses `LLMOrchestrator.plan_and_generate()`
2. ✅ All refinement uses `LLMOrchestrator.refine_stage()`
3. ✅ No fallback paths to simple prompts exist
4. ✅ Legacy prompts removed from registry
5. ✅ All tests pass with ≥80% coverage
6. ✅ Documentation updated
7. ✅ Performance maintained or improved
8. ✅ No critical issues after 7 days in production

---

## 📊 EXPECTED OUTCOMES

### Code Quality
- **Lines Removed**: ~1,500 (prompts + deprecated methods)
- **Lines Modified**: ~500 (RefineAdapter + tests)
- **Net Change**: -1,000 lines (10% reduction)

### Test Quality
- **Improved**: Symbol resolution prevents undefined name errors
- **Improved**: Enhanced context leads to better coverage
- **Improved**: 4-stage pipeline ensures quality gates

### Maintainability
- **Single Source of Truth**: All generation through orchestrator
- **Clear Architecture**: No parallel paths
- **Better Observability**: 4-stage pipeline for debugging

---

## 🚨 ROLLBACK PLAN

If something goes wrong:

```bash
# Option 1: Revert the merge
git revert <merge-commit-sha> -m 1

# Option 2: Restore backup branch
git checkout refactor/pre-orchestrator-consolidation-backup

# Option 3: Cherry-pick specific fixes
git cherry-pick <commit-sha>
```

**Rollback Criteria**:
- Critical functionality broken
- >20% performance degradation
- Test pass rate drops >10%
- Unrecoverable errors in production

---

## 📈 DETAILED PHASE DOCUMENTATION

- **Phase 0**: [PHASE_0_PREFLIGHT.md](./PHASE_0_PREFLIGHT.md)
- **Phase 2**: [PHASE_2_REFINE_ADAPTER.md](./PHASE_2_REFINE_ADAPTER.md)
- **Callsite Audit**: [CALLSITE_AUDIT.md](./CALLSITE_AUDIT.md)
- **ADR**: [ADR-001-orchestrator-consolidation.md](./ADR-001-orchestrator-consolidation.md)
- **Removed Prompts**: [REMOVED_PROMPTS.md](./REMOVED_PROMPTS.md)
- **Validation Scripts**: [VALIDATION_SCRIPTS.md](./VALIDATION_SCRIPTS.md)

---

**END OF MAIN PLAN**
**Next**: Review detailed phase documentation
