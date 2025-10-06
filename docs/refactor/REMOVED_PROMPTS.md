# Removed Prompts Documentation

**Removal Date**: 2025-10-05
**Reason**: Consolidation to orchestrator-only architecture
**ADR**: See [ADR-001-orchestrator-consolidation.md](./ADR-001-orchestrator-consolidation.md)

---

## Overview

This document catalogs all prompts removed during the orchestrator consolidation refactor. These prompts have been superseded by the orchestrator's 4-stage pipeline prompts.

---

## Legacy Prompts (REMOVED)

### 1. test_generation

**System Prompt Location**: `testcraft/prompts/registry.py:271-312`
**User Prompt Location**: `testcraft/prompts/registry.py:355-378`
**Purpose**: Basic test generation with minimal context
**Replacement**: `orchestrator_plan` + `orchestrator_generate`

**Why Removed**:
- No symbol resolution
- Simple string context instead of structured ContextPack
- No iterative refinement
- Lower test quality

**Migration Path**:
```python
# ❌ OLD CODE:
system = registry.get_system_prompt("test_generation")
user = registry.get_user_prompt("test_generation", code_content=code)
result = llm.generate_tests(code_content=code, context=user)

# ✅ NEW CODE:
orchestrator = LLMOrchestrator(...)
context_pack = build_context_pack(target_file, target_object)
result = orchestrator.plan_and_generate(context_pack=context_pack)
```

---

### 2. refinement

**System Prompt Location**: `testcraft/prompts/registry.py:314-353`
**User Prompt Location**: `testcraft/prompts/registry.py:380-396`
**Purpose**: Basic test refinement based on failures
**Replacement**: `orchestrator_refine`

**Why Removed**:
- No context re-packing
- No symbol resolution for missing imports
- Simple failure analysis
- No structured feedback

**Migration Path**:
```python
# ❌ OLD CODE:
system = registry.get_system_prompt("refinement")
user = registry.get_user_prompt("refinement",
    test_content=test_code,
    failure_output=pytest_output
)
result = llm.refine_content(original=test_code, instructions=user)

# ✅ NEW CODE:
orchestrator = LLMOrchestrator(...)
feedback = {
    "result": "failed",
    "trace_excerpt": pytest_output,
    "coverage_gaps": {},
}
refined = orchestrator.refine_stage(
    context_pack=context_pack,
    existing_code=test_code,
    feedback=feedback
)
```

---

## LLM Adapter Prompts (REMOVED)

### 3. llm_test_generation

**System Prompt Location**: `testcraft/prompts/registry.py:401-430`
**User Prompt Location**: `testcraft/prompts/registry.py:536-564`
**Purpose**: Test generation with enriched context
**Replacement**: `orchestrator_plan` + `orchestrator_generate`

**Why Removed**:
- Still lacks symbol resolution
- No structured ContextPack
- Parallel implementation creates prompt drift
- Duplicates orchestrator functionality

**Key Difference from Legacy**:
- Included more context (imports, dependencies)
- Better structured prompts
- But still no iterative symbol resolution

---

### 4. llm_code_analysis

**System Prompt Location**: `testcraft/prompts/registry.py:432-457`
**User Prompt Location**: `testcraft/prompts/registry.py:566-576`
**Purpose**: Code analysis for testability
**Replacement**: None (feature may be re-implemented separately)

**Why Removed**:
- Not currently used in production
- Not part of core test generation flow
- Can be re-added if needed as separate feature

**Note**: If code analysis is needed in the future, it should be implemented as a separate use case using the orchestrator for consistency.

---

### 5. llm_content_refinement

**System Prompt Location**: `testcraft/prompts/registry.py:459-534`
**User Prompt Location**: `testcraft/prompts/registry.py:578-601`
**Purpose**: Content refinement with instructions
**Replacement**: `orchestrator_refine`

**Why Removed**:
- Simpler than orchestrator refinement
- No context re-packing
- No missing symbol resolution
- Parallel implementation

**Migration Example**:
```python
# ❌ OLD CODE (in RefineAdapter):
system = registry.get_system_prompt("llm_content_refinement")
user = registry.get_user_prompt("llm_content_refinement",
    current_content=test_code,
    failure_output=pytest_output,
    active_import_path=import_path,
    preflight_suggestions=suggestions,
)
response = self.llm.refine_content(
    original_content=test_code,
    refinement_instructions=user,
    system_prompt=system,
)

# ✅ NEW CODE (in RefineAdapter):
orchestrator = self._ensure_orchestrator()
context_pack = self._build_context_pack_from_source(test_file, source_context)
feedback = {
    "result": "failed",
    "trace_excerpt": pytest_output,
}
refined = orchestrator.refine_stage(
    context_pack=context_pack,
    existing_code=test_code,
    feedback=feedback,
)
```

---

## Kept Prompts

### Orchestrator Prompts (KEPT)

These prompts are the **new standard** for all test generation:

1. **orchestrator_plan** - Create comprehensive test plans with symbol resolution
2. **orchestrator_generate** - Generate tests following approved plans
3. **orchestrator_refine** - Repair failing tests with minimal changes
4. **orchestrator_manual_fix** - Create failing tests + bug reports for product defects

**Key Features**:
- ✅ ContextPack structured context
- ✅ Iterative missing_symbols resolution
- ✅ GWT (Given/When/Then) patterns
- ✅ Canonical import enforcement
- ✅ 4-stage quality pipeline

### Evaluation Prompts (KEPT)

These prompts are for LLM-as-judge and A/B testing:

1. **llm_judge_v1** - Single test quality evaluation
2. **pairwise_comparison_v1** - Compare two test implementations
3. **rubric_evaluation_v1** - Score tests against defined criteria
4. **statistical_analysis_v1** - Analyze test execution patterns
5. **bias_mitigation_v1** - Detect and mitigate evaluation bias

**Reason for Keeping**:
- Separate concern from test generation
- Used by evaluation harness, not generation pipeline
- No orchestrator equivalent needed

---

## Schema Changes

### Removed Schemas

The following output schemas were also removed from `registry.py:_schema_for()`:

1. `generation_output` - Legacy test generation output
2. `generation_output_enhanced` - Enhanced legacy output
3. `refinement_output` - Legacy refinement output
4. `refinement_output_enhanced` - Enhanced refinement output
5. `llm_test_generation_output` - LLM adapter generation output
6. `llm_code_analysis_output` - Code analysis output
7. `llm_content_refinement_output` - LLM adapter refinement output

### Kept Schemas

- All orchestrator output schemas
- All evaluation output schemas

---

## Impact Analysis

### Files Affected by Removal

**Direct Impact**:
- `testcraft/prompts/registry.py` - Prompts and schemas deleted
- `testcraft/application/generate_usecase.py` - Fallback removed
- `testcraft/adapters/refine/main_adapter.py` - Switched to orchestrator

**Test Impact**:
- `tests/test_llm_adapters.py` - Updated test focus
- `tests/test_refine_adapters.py` - Updated mocking strategy
- `scripts/prompt_regression_test.py` - Updated expected prompts

**Documentation Impact**:
- `README.md` - Architecture updated
- `docs/architecture.md` - Flow diagrams updated
- This file - Complete documentation of removed prompts

---

## Recovery

If you need to recover a removed prompt (e.g., for comparison or migration):

```bash
# View removed prompt in git history
git show refactor/pre-orchestrator-consolidation-backup:testcraft/prompts/registry.py

# Extract specific prompt method
git show refactor/pre-orchestrator-consolidation-backup:testcraft/prompts/registry.py \
  | sed -n '/def _system_prompt_generation_v1/,/^    def /p'

# Restore entire file temporarily
git show refactor/pre-orchestrator-consolidation-backup:testcraft/prompts/registry.py \
  > /tmp/old_registry.py
```

---

## Future Considerations

### When to Add New Prompts

**DO add new prompts when**:
- Implementing new orchestrator stages
- Adding new evaluation criteria
- Creating specialized analysis tools

**DON'T add new prompts when**:
- Creating "simpler" alternatives to orchestrator
- Implementing per-adapter customizations
- Building parallel generation paths

### Maintaining Consistency

All future test generation-related prompts should:
1. Use ContextPack structured context
2. Support symbol resolution
3. Integrate with orchestrator pipeline
4. Follow 4-stage quality gates

---

## External Usage Warning

⚠️ **If you have external code** that called these prompts directly:

```python
# This will now fail:
prompt = registry.get_system_prompt("test_generation")
# KeyError: 'test_generation'

# You must migrate to:
orchestrator = LLMOrchestrator(...)
result = orchestrator.plan_and_generate(context_pack)
```

See [MIGRATION_ORCHESTRATOR.md](./MIGRATION_ORCHESTRATOR.md) for complete migration guide.

---

## Questions?

**Q: Can I add legacy prompts back?**
A: No. They were removed to eliminate prompt drift and maintenance burden. Use orchestrator prompts exclusively.

**Q: What if orchestrator is too slow?**
A: Benchmark first. If performance is an issue, optimize the orchestrator (e.g., adjust symbol resolution depth, cache context packs).

**Q: Can I customize prompts per LLM adapter?**
A: No. All adapters use the same orchestrator prompts for consistency. Adapter-specific behavior should be in the adapter's `generate_tests()` implementation, not in prompts.

**Q: Where is llm_code_analysis now?**
A: Removed. It wasn't used. If needed, implement as separate use case using orchestrator.

---

**Last Updated**: 2025-10-05
**Maintained By**: Development Team
**Related**: [ADR-001](./ADR-001-orchestrator-consolidation.md), [ENHANCED_REFACTOR_PLAN.md](./ENHANCED_REFACTOR_PLAN.md)
