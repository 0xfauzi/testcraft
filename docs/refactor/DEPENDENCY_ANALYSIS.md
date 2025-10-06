# Dependency Analysis: Orchestrator Consolidation

**Date**: 2025-10-06
**Purpose**: Map dependencies to determine safe refactoring order

---

## Component Dependency Map

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Layer                             │
│  (testcraft/cli/main.py, dependency_injection.py)       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ calls
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Application Layer                           │
│           GenerateUseCase (orchestrator)                 │
│  Uses: LLMOrchestrator, RefinePort, ContextPort          │
└──────┬──────────────────┬─────────────────┬────────────┘
       │                  │                 │
       │ uses             │ uses            │ delegates to
       ▼                  ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│LLMOrchestrator│  │ RefineAdapter │  │  Other Adapters  │
│  (Phase 1-2)  │  │  (Phase 2)    │  │   (unchanged)    │
└──────┬────────┘  └──────┬────────┘  └──────────────────┘
       │                  │
       │ calls            │ calls (will use orchestrator)
       ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                    LLMPort Protocol                      │
│  (generate_tests, refine_content - LOW-LEVEL methods)   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ routed by
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    LLMRouter                             │
│          (Selects: Claude, OpenAI, Azure, Bedrock)       │
└─────────────────────────────────────────────────────────┘
```

---

## Refactoring Order (Based on Dependencies)

### Phase 2: RefineAdapter
**Dependencies**: None (leaf component)
**Dependents**: GenerateUseCase
**Safe to refactor**: YES ✅

**Changes**:
- Add `ParserPort` parameter (breaking change)
- Add orchestrator integration
- Update `dependency_injection.py` to pass parser

**Impact**: MEDIUM - Breaking change to constructor

---

### Phase 3: GenerateUseCase Fallback
**Dependencies**: LLMOrchestrator (already exists)
**Dependents**: CLI (unchanged interface)
**Safe to refactor**: YES ✅

**Changes**:
- Remove fallback at line 542
- Add fail-fast error

**Impact**: LOW - Internal change, no API breakage

---

### Phase 4: Prompt Registry
**Dependencies**: None
**Dependents**: RefineAdapter (will be updated in Phase 2)
**Safe to refactor**: YES ✅ (after Phase 2)

**Changes**:
- Remove 10 prompt entries
- Delete ~330 lines of prompt methods
- Update schemas if needed

**Impact**: HIGH - Breaking change for external users

---

## Critical Dependency Notes

### 1. LLMOrchestrator
- **Status**: ✅ Already implemented and working
- **Used by**: GenerateUseCase (primary path)
- **Dependencies**: LLMPort, ParserPort, ContextAssembler, SymbolResolver
- **Action**: No changes needed

### 2. ContextPackBuilder
- **Status**: ✅ Fixed in Phase 0 (parser parameter required)
- **Used by**: GenerateUseCase
- **Dependencies**: ParserPort (now properly injected)
- **Action**: No further changes needed

### 3. LLMPort Protocol
- **Status**: ✅ Stable interface
- **Implemented by**: ClaudeAdapter, OpenAIAdapter, AzureOpenAIAdapter, BedrockAdapter
- **Used by**: Orchestrator, RefineAdapter, Router
- **Action**: **NO CHANGES** - This is the stable abstraction

### 4. LLMRouter
- **Status**: ✅ Implemented in Phase 0
- **Used by**: DI container
- **Dependencies**: Individual adapters
- **Action**: No changes needed

---

## Dependency Chain for Each Phase

### Phase 2: RefineAdapter Dependencies

**Before Refactor**:
```
RefineAdapter
  ├─ LLMPort ✅
  └─ PromptRegistry (llm_content_refinement) ❌ LEGACY
```

**After Refactor**:
```
RefineAdapter
  ├─ LLMPort ✅
  ├─ ParserPort ✅ NEW
  └─ LLMOrchestrator ✅ NEW
      ├─ LLMPort ✅
      ├─ ParserPort ✅
      ├─ ContextAssembler ✅
      └─ SymbolResolver ✅
```

**Risk**: LOW - All dependencies already available

---

### Phase 3: Remove Fallback Dependencies

**Current Fallback**:
```
GenerateUseCase (line 542)
  └─ LLM.generate_tests() (simple call)
```

**After Removal**:
```
GenerateUseCase
  └─ Fail-fast error (no dependencies)
```

**Risk**: LOW - Pure removal, no new dependencies

---

### Phase 4: Prompt Registry Dependencies

**Before Removal**:
```
PromptRegistry
  ├─ Legacy prompt methods (10)
  └─ Legacy schemas (4)
      └─ Used by: RefineAdapter (1 callsite)
```

**After Removal**:
```
PromptRegistry
  ├─ Orchestrator prompt methods ✅
  ├─ Evaluation prompt methods ✅
  └─ Orchestrator schemas ✅
```

**Risk**: MEDIUM - Must complete Phase 2 first

---

## Circular Dependency Check

✅ **NO CIRCULAR DEPENDENCIES FOUND**

All dependencies flow in one direction:
```
CLI → UseCase → Orchestrator → LLMPort → Adapters
                ↓
            RefinePort → RefineAdapter → (will use Orchestrator)
```

---

## External Dependencies

### Python Packages (Unchanged)
- `anthropic>=0.66.0` ✅
- `openai>=1.106.1` ✅
- `langchain-aws>=0.1.0` ✅
- `azure-identity>=1.15.0` ✅

### Internal Modules (Unchanged)
- `testcraft.domain.models` ✅
- `testcraft.ports.*` ✅
- `testcraft.config.models` ✅

**No new external dependencies required** ✅

---

## Test Dependencies

### Current Test Structure
```
tests/
  ├─ test_llm_adapters.py (LOW-LEVEL tests)
  ├─ test_pytest_refiner_integration.py (INTEGRATION)
  ├─ test_immediate_refinement.py (INTEGRATION)
  └─ test_strict_refinement.py (INTEGRATION)
```

**Phase 5 Update Strategy**:
1. Keep LOW-LEVEL adapter tests (they're correct)
2. Update INTEGRATION tests to mock orchestrator instead of adapters
3. Add orchestrator-specific tests if missing

**Risk**: LOW - Test structure is well-defined

---

## Dependency Analysis Summary

| Phase | New Dependencies | Breaking Changes | Risk |
|-------|------------------|------------------|------|
| 2 | ParserPort (to RefineAdapter) | Constructor signature | MEDIUM |
| 3 | None | None (internal change) | LOW |
| 4 | None | Prompt removal (external) | HIGH |

---

## Recommended Refactoring Order

1. ✅ **Phase 0**: Pre-flight (COMPLETE)
2. ✅ **Phase 1**: Impact analysis (COMPLETE)
3. **Phase 2**: RefineAdapter (safe, no circular deps)
4. **Phase 3**: Remove fallback (safe, depends on Phase 2 being stable)
5. **Phase 4**: Clean up prompts (safe after Phases 2-3)
6. **Phases 5-9**: Tests, docs, validation, deployment

**No blockers identified** ✅

---

**Analysis Complete**: 2025-10-06
**Confidence**: HIGH
**Ready to proceed**: YES
