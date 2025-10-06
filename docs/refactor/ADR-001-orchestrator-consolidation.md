# ADR-001: Orchestrator-Only Test Generation Architecture

**Status**: Accepted ✅
**Date**: 2025-10-05 (Proposed) / 2025-10-06 (Accepted)
**Deciders**: Development Team
**Technical Story**: Consolidate test generation to use LLMOrchestrator exclusively
**Phase 0**: Complete (dry-run, baseline, LLMRouter fixed)
**Phase 1**: In Progress (impact analysis)

---

## Context

### Current State

We currently have **three parallel test generation paths**:

1. **Legacy Prompts** (`test_generation`/`refinement`)
   - Simple prompts with minimal context
   - Original implementation from early development
   - Limited symbol resolution
   - No structured context engineering

2. **LLM Adapter Prompts** (`llm_test_generation`/`llm_content_refinement`)
   - Slightly enriched context
   - Per-adapter customization
   - Still relatively simple
   - No symbol resolution loop

3. **Orchestrator Prompts** (`orchestrator_plan`/`generate`/`refine`/`manual_fix`)
   - ✅ Enhanced context engineering (ContextPack)
   - ✅ Iterative symbol resolution
   - ✅ 4-stage pipeline with explicit planning
   - ✅ GWT (Given/When/Then) patterns
   - ✅ Better test quality and fewer errors

### The Problem

**Maintenance Burden**:
- Three parallel codepaths to maintain
- Prompt drift between implementations
- Inconsistent test quality depending on path taken
- Difficult to add features (must update 3 places)

**Quality Issues**:
- Legacy/adapter paths produce lower quality tests
- Silent fallback degrades quality without warning
- No symbol resolution in simple paths = more undefined name errors

**Confusion**:
- Developers unsure which path is used when
- Documentation fragmented across approaches
- Testing requires covering multiple paths

### Evidence

From codebase analysis:

```python
# GenerateUseCase.py:534-540 - Fallback degradation
else:
    # Falls back to simple generation when ContextPack unavailable
    llm_result = await self._llm.generate_tests(
        code_content=code_content,
        context=enhanced_context,  # Simple string, not ContextPack
    )
```

**Issue**: Silent quality degradation with no warning to user.

---

## Decision

**We will consolidate ALL test generation to use the LLMOrchestrator exclusively.**

### What We're Removing

1. **Legacy Prompt USAGE** (not the methods themselves):
   - Fallback path in `GenerateUseCase` (lines 534-540)
   - Direct high-level calls to `generate_tests()` with simple prompts
   - Direct calls to `refine_content()` with simple prompts

2. **Legacy Prompt DEFINITIONS**:
   - `test_generation` prompt templates
   - `refinement` prompt templates
   - `llm_test_generation` prompt templates
   - `llm_code_analysis` prompt templates
   - `llm_content_refinement` prompt templates

### What We're Keeping

1. **LLMPort Protocol Methods** (orchestrator uses these):
   ```python
   # These are LOW-LEVEL methods used by orchestrator
   def generate_tests(self, code_content: str, **kwargs) -> dict[str, Any]:
       """LOW-LEVEL: Send text to LLM, get response"""
       # Orchestrator calls this with its enhanced prompts

   def refine_content(self, original: str, instructions: str, **kwargs) -> dict[str, Any]:
       """LOW-LEVEL: Send refinement request to LLM"""
       # Orchestrator calls this with ContextPack-derived instructions
   ```

2. **Orchestrator Prompts**:
   - `orchestrator_plan` - Create test plans with symbol resolution
   - `orchestrator_generate` - Generate tests following plan
   - `orchestrator_refine` - Repair failing tests minimally
   - `orchestrator_manual_fix` - Create failing tests for product bugs

3. **Evaluation Prompts**:
   - `llm_judge_v1` - Test quality evaluation
   - `pairwise_comparison_v1` - A/B testing
   - `rubric_evaluation_v1` - Scoring against criteria
   - `statistical_analysis_v1` - Pattern analysis
   - `bias_mitigation_v1` - Bias detection

---

## Rationale

### Why Orchestrator-Only?

#### 1. Better Test Quality

**Symbol Resolution**:
```python
# Orchestrator iteratively resolves missing symbols
missing_symbols = ["UserService", "DatabaseAdapter"]
# → Fetches precise definitions
# → Includes in ContextPack
# → No more undefined name errors
```

**Enhanced Context**:
```python
# ContextPack structure
ContextPack(
    target=Target(module_file="src/auth.py", object="UserService"),
    import_map=ImportMap(target_import="from src.auth import UserService"),
    focal=Focal(source="class UserService:...", signature="..."),
    resolved_defs=[ResolvedDef(name="DatabaseAdapter", ...)],
    property_context=PropertyContext(gwt_snippets=[...]),
    conventions=Conventions(test_framework="pytest"),
    budget=Budget(max_tokens=8000),
)
```

**Result**: Tests have all context needed, generate correct imports, use actual APIs.

#### 2. Single Source of Truth

**Before** (3 paths):
```
generate_tests() → test_generation prompt → Basic test
                 → llm_test_generation prompt → Better test
                 → orchestrator pipeline → Best test
```

**After** (1 path):
```
generate_tests() → orchestrator pipeline → Consistent quality
```

**Benefit**:
- One codebase to maintain
- One set of prompts to tune
- Predictable quality

#### 3. Better Observability

**4-Stage Pipeline** makes debugging clear:
```
1. PLAN stage:
   - What tests to create?
   - What symbols are needed?
   → Missing symbols detected → Resolution loop

2. GENERATE stage:
   - Create tests following plan
   - Use resolved context
   → Generated code produced

3. REFINE stage (if tests fail):
   - Analyze failures
   - Apply minimal fixes
   → Repaired tests

4. MANUAL_FIX stage (if product bug):
   - Create failing test
   - Document bug
   → Test + bug report
```

Each stage is separately observable in logs/telemetry.

#### 4. Forced Quality

**Before**:
```python
else:
    # Silently degrade to simple generation
    result = llm.generate_tests(code, context="...")
```

**After**:
```python
else:
    # FAIL FAST with clear error
    raise GenerationError(
        "Cannot generate without ContextPack. "
        "This indicates a parsing/context assembly issue."
    )
```

**Benefit**: Forces fixing root cause instead of masking with poor-quality fallback.

---

## Consequences

### Positive

✅ **Better Test Quality**:
- Symbol resolution prevents undefined name errors
- Enhanced context leads to better test coverage
- 4-stage pipeline ensures quality gates

✅ **Simpler Maintenance**:
- Single codebase for test generation
- One set of prompts to maintain
- Clear architecture

✅ **Better Observability**:
- 4-stage pipeline easy to debug
- Clear failure points
- Better telemetry

✅ **Predictable Behavior**:
- No silent quality degradation
- Consistent results
- Clear error messages

### Negative

❌ **Breaking Changes**:
- External code calling legacy paths will break
- `RefineAdapter` now requires `ParserPort` parameter
- Legacy prompt types no longer available

❌ **No Gradual Migration**:
- All-or-nothing change
- Cannot keep legacy as fallback
- Requires updating all callers simultaneously

❌ **Potential Performance Impact**:
- Orchestrator is more complex (more LLM calls for symbol resolution)
- May increase token usage
- May increase latency for complex files

### Mitigation Strategies

**For Breaking Changes**:
1. Provide comprehensive migration guide
2. Update all internal callers before removal
3. Keep rollback branch for 30 days
4. Document breaking changes clearly

**For Performance**:
1. Benchmark before/after
2. Add configurable symbol resolution depth
3. Cache parsed files to reduce repeated parsing
4. Monitor token usage and adjust budgets

**For Migration Risk**:
1. Comprehensive test coverage before change
2. Validation scripts to ensure no regression
3. Gradual rollout (feature flag if possible)
4. Quick rollback plan

---

## Alternatives Considered

### Alternative 1: Keep Legacy as Fallback

**Pros**:
- Graceful degradation
- No hard errors
- Easier migration

**Cons**:
- Maintains complexity
- Masks real issues
- Inconsistent quality
- **REJECTED**: Violates "fail fast" principle

### Alternative 2: Gradual Migration with Feature Flag

**Pros**:
- Can test orchestrator gradually
- Easy rollback
- Less risky

**Cons**:
- Maintains two codepaths during migration
- Feature flag complexity
- Longer maintenance burden
- **REJECTED**: Not worth the complexity for internal refactor

### Alternative 3: Keep LLM Adapter Prompts

**Pros**:
- Less radical change
- Adapters can customize
- Smaller refactor

**Cons**:
- Still have 2 parallel paths
- Still have prompt drift
- Doesn't solve core issues
- **REJECTED**: Doesn't address root problem

---

## Implementation Plan

See: [ENHANCED_REFACTOR_PLAN.md](./ENHANCED_REFACTOR_PLAN.md)

**Summary**:
1. Phase 0: Pre-flight fixes (dry-run, baseline)
2. Phase 1: Impact analysis (callsite audit)
3. Phase 2: Refactor RefineAdapter
4. Phase 3: Remove fallback
5. Phase 4: Clean up prompts
6. Phase 5: Update tests
7. Phase 6: Update documentation
8. Phase 7: Validation
9. Phase 8: Deployment
10. Phase 9: Post-merge monitoring

---

## Validation

### Success Criteria

1. ✅ All generation uses `orchestrator.plan_and_generate()`
2. ✅ All refinement uses `orchestrator.refine_stage()`
3. ✅ No fallback paths exist
4. ✅ Legacy prompts removed
5. ✅ All tests pass (≥80% coverage)
6. ✅ Performance maintained or improved
7. ✅ No critical issues after 7 days

### Rollback Criteria

**Rollback if**:
- Critical functionality broken
- >20% performance degradation
- Test pass rate drops >10%
- Unrecoverable production errors

---

## References

- [ENHANCED_REFACTOR_PLAN.md](./ENHANCED_REFACTOR_PLAN.md) - Full implementation plan
- [CALLSITE_AUDIT.md](./CALLSITE_AUDIT.md) - Complete audit of affected code
- [PHASE_0_PREFLIGHT.md](./PHASE_0_PREFLIGHT.md) - Pre-flight fixes
- [PHASE_2_REFINE_ADAPTER.md](./PHASE_2_REFINE_ADAPTER.md) - RefineAdapter refactor
- [REMOVED_PROMPTS.md](./REMOVED_PROMPTS.md) - Documentation of removed prompts

---

**Approved**: [Pending]
**Effective Date**: [TBD after approval]
