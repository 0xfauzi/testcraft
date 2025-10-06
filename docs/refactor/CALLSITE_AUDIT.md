# Callsite Audit: Legacy Prompts and High-Level LLM Calls

**Date**: 2025-10-06
**Purpose**: Complete audit of all legacy prompt usage and high-level LLM method calls
**Baseline Commit**: b07b7d3

---

## Executive Summary

| Category | Count | Priority | Action |
|----------|-------|----------|--------|
| Legacy prompt references | 41 | HIGH | Remove in Phase 4 |
| `generate_tests()` high-level calls | 1 | HIGH | Remove in Phase 3 |
| `generate_tests()` low-level calls | 8 | N/A | Keep (used by orchestrator) |
| `refine_content()` calls | 2 | N/A | Keep (delegation/low-level) |

---

## 1. Legacy Prompt References (41 callsites)

### 1.1 Production Code Using Legacy Prompts

#### HIGH PRIORITY: RefineAdapter

**File**: `testcraft/adapters/refine/main_adapter.py:681`

```python
system_prompt = prompt_registry.get_system_prompt("llm_content_refinement")
```

**Impact**: HIGH - Active production code
**Action**: Update to use orchestrator in Phase 2
**Dependencies**: None

---

### 1.2 Prompt Registry Definitions (PHASE 4)

All following references are in `testcraft/prompts/registry.py`:

#### Dictionary Definitions (Lines 105-132)

**Lines 105-110**: System prompt mapping dictionary
```python
"test_generation": self._system_prompt_generation_v1(),
"refinement": self._system_prompt_refinement_v1(),
"llm_test_generation": self._system_prompt_llm_test_generation_v1(),
"llm_content_refinement": self._system_prompt_llm_content_refinement_v1(),
```

**Lines 127-132**: User prompt mapping dictionary
```python
"test_generation": self._user_prompt_generation_v1(),
"refinement": self._user_prompt_refinement_v1(),
"llm_test_generation": self._user_prompt_llm_test_generation_v1(),
"llm_content_refinement": self._user_prompt_llm_content_refinement_v1(),
```

**Action**: Remove all 10 entries in Phase 4

---

#### Legacy Prompt Method Implementations

**DELETE in Phase 4** (~330 lines total):

1. **Line 271**: `_system_prompt_generation_v1()`
2. **Line 314**: `_system_prompt_refinement_v1()`
3. **Line 380**: `_user_prompt_refinement_v1()`
4. **Line 401**: `_system_prompt_llm_test_generation_v1()`
5. **Line 459**: `_system_prompt_llm_content_refinement_v1()`
6. **Line 536**: `_user_prompt_llm_test_generation_v1()`
7. **Line 578**: `_user_prompt_llm_content_refinement_v1()`
8. Plus corresponding `_user_prompt_generation_v1()` (not shown in grep)

---

#### Schema Definitions

**Lines to update** (references in schema code):

- Line 1282: `if schema_type == "refinement_output"`
- Line 1317: `if schema_type == "refinement_output_enhanced"`
- Line 1390: `if schema_type == "llm_test_generation_output"`
- Line 1525: `if schema_type == "llm_content_refinement_output"`

**Action**: Evaluate if schemas are still needed for orchestrator; likely keep but may rename

---

#### Documentation and Comments

**Lines 5-7**: Module docstring references
```python
- System prompts for test generation and refinement
- JSON Schemas for structured LLM outputs (generation and refinement)
```

**Action**: Update docstring in Phase 4

---

**Lines 152, 162, 254**: Function parameters and conditionals
- Default parameter values
- Conditional logic based on prompt_type

**Action**: Update or remove in Phase 4

---

## 2. High-Level `generate_tests()` Calls

### Analysis

Total: **9 callsites**
- **1 LEGACY** (needs removal)
- **8 CORRECT** (keep)

---

### 2.1 LEGACY FALLBACK - REMOVE IN PHASE 3 ⚠️

**File**: `testcraft/application/generate_usecase.py:542`

```python
    # Fall back to legacy LLM call when context is not available
    llm_result = await self._llm.generate_tests(
        code_content=code_content,
        context=enhanced_context,
        test_framework=self._config["test_framework"],
    )
```

**Priority**: HIGH
**Phase**: 3
**Action**: Replace with fail-fast error
**Reason**: This is the silent quality degradation we're eliminating

**Replacement**:
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

### 2.2 CLI Orchestration - KEEP ✅

**File**: `testcraft/cli/main.py:426, 448`

```python
generate_usecase.generate_tests(...)
```

**Priority**: N/A
**Action**: KEEP - This is the high-level entry point
**Reason**: CLI calls the use case, which internally uses orchestrator

---

### 2.3 Router Delegation - KEEP ✅

**File**: `testcraft/adapters/llm/router.py:193`

```python
return adapter.generate_tests(code_content, context, test_framework, **kwargs)
```

**Priority**: N/A
**Action**: KEEP - Part of LLMPort protocol implementation
**Reason**: Router delegates to selected adapter

---

### 2.4 Azure Adapter Example - EVALUATE

**File**: `testcraft/adapters/llm/azure.py:104`

```python
result = adapter.generate_tests(code_content)
```

**Priority**: LOW
**Action**: Evaluate if this is example/test code
**Context**: Appears to be in adapter implementation
**Decision**: Keep if it's a valid use case; likely example usage

---

### 2.5 Orchestrator LOW-LEVEL Calls - KEEP ✅

**File**: `testcraft/application/generation/services/llm_orchestrator.py`

Four calls at lines: **360, 460, 554, 615**

```python
# Line 360 - Single-shot generation
response = self._llm.generate_tests(code_content=full_prompt)

# Line 460 - PLAN stage
response = self._llm.generate_tests(code_content=plan_prompt)

# Line 554 - GENERATE stage
response = self._llm.generate_tests(code_content=generate_prompt)

# Line 615 - REFINE stage
response = self._llm.generate_tests(code_content=refine_prompt)
```

**Priority**: N/A
**Action**: KEEP - These are CORRECT
**Reason**: Orchestrator uses `generate_tests()` as a LOW-LEVEL method with orchestrator-built prompts

**This is the INTENDED architecture** ✅

---

## 3. `refine_content()` Calls

### Analysis

Total: **2 callsites** - Both CORRECT ✅

---

### 3.1 Router Delegation - KEEP ✅

**File**: `testcraft/adapters/llm/router.py:233`

```python
return adapter.refine_content(
    original_content,
    refinement_instructions,
    system_prompt=system_prompt,
    **kwargs,
)
```

**Action**: KEEP - Protocol implementation
**Reason**: Router delegates to selected adapter

---

### 3.2 RefineAdapter LOW-LEVEL Call - KEEP ✅

**File**: `testcraft/adapters/refine/main_adapter.py:687`

```python
response = self.llm.refine_content(...)
```

**Action**: KEEP - LOW-LEVEL usage
**Reason**: RefineAdapter will use orchestrator in Phase 2, but the low-level call remains valid

---

## 4. Action Plan Summary

### Phase 2: RefineAdapter Refactor
- [ ] Update `refine/main_adapter.py:681` to use orchestrator
- [ ] Add ParserPort dependency to RefineAdapter
- [ ] Remove direct legacy prompt usage

### Phase 3: Remove Fallback
- [ ] Delete `generate_usecase.py:542` fallback code
- [ ] Replace with fail-fast error
- [ ] Update error messaging

### Phase 4: Prompt Cleanup
- [ ] Remove 10 prompt dictionary entries (lines 105-132)
- [ ] Delete ~330 lines of legacy prompt methods
- [ ] Update module docstring (lines 5-7)
- [ ] Evaluate schema definitions (keep if used by orchestrator)
- [ ] Update conditional logic (lines 152, 162, 254)

### Low Priority / Evaluate Later
- [ ] Check `azure.py:104` - determine if example or production code

---

## 5. Files Requiring Modification

### HIGH PRIORITY (Phases 2-4)

| File | Lines Changed | Phase | Action |
|------|---------------|-------|--------|
| `adapters/refine/main_adapter.py` | ~50 | 2 | Add orchestrator integration |
| `application/generate_usecase.py` | ~10 | 3 | Remove fallback, add fail-fast |
| `prompts/registry.py` | ~360 | 4 | Delete legacy prompts |

### MEDIUM PRIORITY (Phase 5)

| File | Lines Changed | Phase | Action |
|------|---------------|-------|--------|
| Tests for refine adapter | TBD | 5 | Update mocking |
| Tests for generate usecase | TBD | 5 | Update expectations |

---

## 6. Risk Assessment

### HIGH RISK
- **Prompt registry changes**: May break external code using these prompts
- **Mitigation**: Migration guide, clear breaking change docs

### MEDIUM RISK
- **RefineAdapter changes**: Breaking change to constructor signature
- **Mitigation**: Update all callers, deprecation notice

### LOW RISK
- **Fallback removal**: Internal change, no external API impact
- **Generate_tests LOW-LEVEL calls**: No changes needed (these are correct)

---

## 7. Validation Checklist

After completing all phases:

- [ ] No references to `test_generation` prompt
- [ ] No references to `refinement` prompt
- [ ] No references to `llm_test_generation` prompt
- [ ] No references to `llm_content_refinement` prompt
- [ ] Only orchestrator prompts remain in registry
- [ ] All `generate_tests()` calls are either:
  - CLI/use case entry points (high-level)
  - Orchestrator internal calls (low-level)
  - Router delegation (protocol)
- [ ] No legacy fallback path in GenerateUseCase
- [ ] RefineAdapter uses orchestrator
- [ ] All tests updated and passing

---

**Audit Complete**: 2025-10-06
**Next Step**: Create ADR-001 (Architecture Decision Record)
